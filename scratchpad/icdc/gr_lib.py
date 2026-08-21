"""Golden repair floor probe -- shared library.

Loads the 100 LiteTensorDataTest golden layouts (bbox-ized exactly like the
evaluator baseline extractor), builds the golden relative-order constraint
graphs, and provides a topology-fixed LP that minimises HPWL subject to
  * preserved relative order (=> overlap-free by construction)
  * frozen shapes (=> area / fixed-shape hard constraints preserved)
  * frozen preplaced positions
  * requested boundary-abutment equalities
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "repo_iccad2026_evaluate", REPO / "scripts" / "iccad2026_evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["repo_iccad2026_evaluate"] = ev
    spec.loader.exec_module(ev)
    return ev


def poly_to_bbox(block):
    valid = block[block[:, 0] != -1]
    if len(valid) > 0:
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        return (float(x_min), float(y_min),
                float(x_max - x_min), float(y_max - y_min))
    return (0.0, 0.0, 1.0, 1.0)


@dataclass
class Case:
    test_id: int
    n: int
    rects: list                      # golden bbox rects
    area_target: torch.Tensor
    b2b: torch.Tensor
    p2b: torch.Tensor
    pins: torch.Tensor
    constraints: torch.Tensor
    baseline: dict
    # decoded constraint columns
    fixed: np.ndarray = field(default=None)
    preplaced: np.ndarray = field(default=None)
    mib: np.ndarray = field(default=None)
    clust: np.ndarray = field(default=None)
    bound: np.ndarray = field(default=None)

    @property
    def frame(self):
        x0 = min(r[0] for r in self.rects)
        y0 = min(r[1] for r in self.rects)
        x1 = max(r[0] + r[2] for r in self.rects)
        y1 = max(r[1] + r[3] for r in self.rects)
        return x0, y0, x1, y1


def load_cases(ev, data_path="../"):
    ds = ev.FloorplanDatasetLiteTest(str(data_path))
    cases = []
    for idx in range(len(ds)):
        s = ds[idx]
        area_target, b2b, p2b, pins, cons = s["input"]
        polygons, metrics = s["label"]
        n = int((area_target != -1).sum().item())
        rects = [poly_to_bbox(polygons[i]) for i in range(n)]

        hb = ev.calculate_hpwl_b2b(rects, b2b)
        hp = ev.calculate_hpwl_p2b(rects, p2b, pins)
        area = ev.calculate_bbox_area(rects)
        if metrics is not None and len(metrics) >= 8:
            if metrics[0] > 0:
                area = float(metrics[0])
            if metrics[-2] > 0:
                hb = float(metrics[-2])
            if metrics[-1] >= 0:
                hp = float(metrics[-1])
        baseline = {"hpwl_baseline": hb + hp, "area_baseline": area}

        cb = cons[:n]
        nc = cb.shape[1]

        def col(k):
            return cb[:, k].numpy().astype(np.int64) if nc > k else np.zeros(n, np.int64)

        cases.append(Case(idx, n, rects, area_target, b2b, p2b, pins, cons,
                          baseline, col(0), col(1), col(2), col(3), col(4)))
    return cases


def evaluate(ev, case, rects):
    """Exact evaluator call, runtime neutralised."""
    return ev.evaluate_solution(
        {"positions": list(rects), "runtime": 1.0},
        case.baseline, case.constraints, case.b2b, case.p2b, case.pins,
        case.area_target, case.rects, median_runtime=1.0)


# ---------------------------------------------------------------- topology

def build_topology(rects, tol=1e-9):
    """Return (hor_edges, ver_edges) with edge (i,j) meaning c_i+s_i <= c_j."""
    n = len(rects)
    x = np.array([r[0] for r in rects])
    y = np.array([r[1] for r in rects])
    w = np.array([r[2] for r in rects])
    h = np.array([r[3] for r in rects])
    hor, ver = [], []
    overlaps = 0
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
            oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            if ox > 1e-6 and oy > 1e-6:
                overlaps += 1
            if oy > tol:
                hor.append((i, j) if x[i] + w[i] <= x[j] + tol else (j, i))
            elif ox > tol:
                ver.append((i, j) if y[i] + h[i] <= y[j] + tol else (j, i))
            else:
                gx = max(x[j] - (x[i] + w[i]), x[i] - (x[j] + w[j]))
                gy = max(y[j] - (y[i] + h[i]), y[i] - (y[j] + h[j]))
                if gx >= gy:
                    hor.append((i, j) if x[i] + w[i] <= x[j] + tol else (j, i))
                else:
                    ver.append((i, j) if y[i] + h[i] <= y[j] + tol else (j, i))
    return hor, ver, overlaps


def transitive_reduce(n, edges):
    if not edges:
        return []
    A = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        A[i, j] = True
    # transitive closure (Floyd-Warshall over booleans, n<=120)
    C = A.copy()
    for k in range(n):
        C |= np.outer(C[:, k], C[k, :])
    redundant = A & (A.astype(np.int8) @ C.astype(np.int8) > 0)
    keep = A & ~redundant
    return [(i, j) for i, j in zip(*np.nonzero(keep))]


def longest_path_bounds(n, edges, size, lo_c, hi_c):
    """lo_c/hi_c are per-node bounds on the coordinate c_i itself.

    Returns (mn, mx): the tightest feasible range for each c_i under the
    precedence constraints c_i + size_i <= c_j plus the box bounds.  The
    system is a difference-constraint system over a DAG, so it is feasible
    iff mn[i] <= mx[i] for every i -- an exact test, no LP needed."""
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    indeg = np.zeros(n, np.int64)
    for i, j in edges:
        succ[i].append(j)
        pred[j].append(i)
        indeg[j] += 1
    # topological order
    order = []
    stack = [i for i in range(n) if indeg[i] == 0]
    deg = indeg.copy()
    while stack:
        u = stack.pop()
        order.append(u)
        for v in succ[u]:
            deg[v] -= 1
            if deg[v] == 0:
                stack.append(v)
    if len(order) != n:
        raise RuntimeError("constraint graph has a cycle")
    mn = np.array(lo_c, dtype=float)
    for u in order:
        for v in succ[u]:
            if mn[u] + size[u] > mn[v]:
                mn[v] = mn[u] + size[u]
    mx = np.array(hi_c, dtype=float)
    for u in reversed(order):
        for v in succ[u]:
            if mx[v] - size[u] < mx[u]:
                mx[u] = mx[v] - size[u]
    return mn, mx


# ---------------------------------------------------------------- axis LP

def solve_axis(n, size, lo, hi, edges, nets, pin_terms, eq, verbose=False,
               prox_weight=0.0, prox_target=None, extra_le=()):
    """Minimise sum_e w_e * |centre_i - centre_j| (+ pin terms) subject to
    c_i + size_i <= c_j for edges, bounds [lo_i, hi_i - size_i], and equalities.

    Returns (coords ndarray, objective) or (None, None) if infeasible.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    nets = [(i, j, wt) for (i, j, wt) in nets if i != j]
    E = len(nets) + len(pin_terms)
    npx = n + E
    if prox_weight > 0.0:
        npx += n

    rows, cols, vals, b = [], [], [], []

    def add_row(entries, rhs):
        r = len(b)
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        b.append(rhs)

    for (i, j) in edges:
        add_row([(i, 1.0), (j, -1.0)], -size[i])

    for entries, rhs in extra_le:
        add_row(list(entries), rhs)

    for e, (i, j, wt) in enumerate(nets):
        t = n + e
        add_row([(i, 1.0), (j, -1.0), (t, -1.0)], (size[j] - size[i]) / 2.0)
        add_row([(j, 1.0), (i, -1.0), (t, -1.0)], (size[i] - size[j]) / 2.0)

    for k, (i, p, wt) in enumerate(pin_terms):
        t = n + len(nets) + k
        add_row([(i, 1.0), (t, -1.0)], p - size[i] / 2.0)
        add_row([(i, -1.0), (t, -1.0)], -p + size[i] / 2.0)

    if prox_weight > 0.0:
        base = n + E
        for i in range(n):
            add_row([(i, 1.0), (base + i, -1.0)], prox_target[i])
            add_row([(i, -1.0), (base + i, -1.0)], -prox_target[i])

    A = coo_matrix((vals, (rows, cols)), shape=(len(b), npx)).tocsr()

    c = np.zeros(npx)
    for e, (i, j, wt) in enumerate(nets):
        c[n + e] = wt
    for k, (i, p, wt) in enumerate(pin_terms):
        c[n + len(nets) + k] = wt
    if prox_weight > 0.0:
        c[n + E:] = prox_weight

    bounds = []
    for i in range(n):
        if i in eq:
            v = float(eq[i])
            bounds.append((v, v))
        else:
            bounds.append((float(lo[i]), float(hi[i] - size[i])))
    bounds.extend([(0.0, None)] * (npx - n))

    res = linprog(c, A_ub=A, b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        return None, None
    return np.asarray(res.x[:n], dtype=float), float(res.fun)


# ---------------------------------------------------------------- helpers

def build_assign_robust(rects, axis_of=None):
    """Pair->axis assignment oriented by CENTRE order (not edge order).

    Centre order is a total order per axis, so each axis graph is guaranteed
    acyclic even when the input layout has micro-overlaps (our solver's output
    is only overlap-free to the evaluator's 1e-6 tolerance, unlike golden).
    Ties broken by block index.
    """
    n = len(rects)
    assign = {}
    for i in range(n):
        xi, yi, wi, hi_ = rects[i]
        cxi, cyi = xi + wi / 2.0, yi + hi_ / 2.0
        for j in range(i + 1, n):
            xj, yj, wj, hj = rects[j]
            cxj, cyj = xj + wj / 2.0, yj + hj / 2.0
            gx = max(xj - (xi + wi), xi - (xj + wj))
            gy = max(yj - (yi + hi_), yi - (yj + hj))
            if gx >= gy:
                lo_, hi__ = (i, j) if (cxi, i) <= (cxj, j) else (j, i)
                assign[(i, j)] = ("h", lo_, hi__)
            else:
                lo_, hi__ = (i, j) if (cyi, i) <= (cyj, j) else (j, i)
                assign[(i, j)] = ("v", lo_, hi__)
    return assign


def solve_axis_ext(n, size, lo, hi, edges, nets, pin_terms, eq,
                   min_blocks=(), max_blocks=(), extra_le=(), objective=True,
                   hpwl_weight=1.0, frame_weight=0.0):
    """Same as solve_axis, but boundary abutment is expressed the way the
    evaluator actually defines it: block b touches the low wall iff
    c_b == min_i c_i, and the high wall iff c_b + s_b == max_i (c_i + s_i).

    Two auxiliary variables (cmin, cmax) turn that into 2n + k linear rows,
    so the bounding frame is free to shrink instead of being nailed to the
    golden frame.  Shrinking is score-free (area_gap is clipped at 0).
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    nets = [(i, j, wt) for (i, j, wt) in nets if i != j]
    if not objective:
        nets, pin_terms = [], []
        frame_weight = 0.0
    E = len(nets) + len(pin_terms)
    use_ext = bool(min_blocks) or bool(max_blocks) or frame_weight > 0.0
    npx = n + E + (2 if use_ext else 0)
    vmin, vmax = n + E, n + E + 1

    rows, cols, vals, b = [], [], [], []

    def add_row(entries, rhs):
        r = len(b)
        for c_, v in entries:
            rows.append(r)
            cols.append(c_)
            vals.append(v)
        b.append(rhs)

    for (i, j) in edges:
        add_row([(i, 1.0), (j, -1.0)], -size[i])
    for entries, rhs in extra_le:
        add_row(list(entries), rhs)

    if use_ext:
        want_min = bool(min_blocks) or frame_weight > 0.0
        want_max = bool(max_blocks) or frame_weight > 0.0
        for i in range(n):
            if want_min:
                add_row([(vmin, 1.0), (i, -1.0)], 0.0)         # cmin <= c_i
            if want_max:
                add_row([(i, 1.0), (vmax, -1.0)], -size[i])    # c_i+s_i <= cmax
        for bi in min_blocks:
            add_row([(bi, 1.0), (vmin, -1.0)], 0.0)            # c_b <= cmin
        for bi in max_blocks:
            add_row([(vmax, 1.0), (bi, -1.0)], size[bi])       # cmax <= c_b+s_b

    for e, (i, j, wt) in enumerate(nets):
        t = n + e
        add_row([(i, 1.0), (j, -1.0), (t, -1.0)], (size[j] - size[i]) / 2.0)
        add_row([(j, 1.0), (i, -1.0), (t, -1.0)], (size[i] - size[j]) / 2.0)
    for k, (i, p, wt) in enumerate(pin_terms):
        t = n + len(nets) + k
        add_row([(i, 1.0), (t, -1.0)], p - size[i] / 2.0)
        add_row([(i, -1.0), (t, -1.0)], -p + size[i] / 2.0)

    A = coo_matrix((vals, (rows, cols)), shape=(len(b), npx)).tocsr()
    c = np.zeros(npx)
    for e, (i, j, wt) in enumerate(nets):
        c[n + e] = wt * hpwl_weight
    for k, (i, p, wt) in enumerate(pin_terms):
        c[n + len(nets) + k] = wt * hpwl_weight
    if use_ext and frame_weight > 0.0:
        # minimise the axis extent (cmax - cmin); at the optimum the two aux
        # vars are pinned to the true max/min, so this is exactly the bbox side
        c[vmax] += frame_weight
        c[vmin] -= frame_weight

    bounds = []
    for i in range(n):
        if i in eq:
            v = float(eq[i])
            bounds.append((v, v))
        else:
            bounds.append((float(lo[i]), float(hi[i] - size[i])))
    bounds.extend([(0.0, None)] * E)
    if use_ext:
        bounds.append((float(np.min(lo)), None))
        bounds.append((None, float(np.max(hi))))

    res = linprog(c, A_ub=A, b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        return None
    return np.asarray(res.x[:n], dtype=float)


def boundary_bits(code):
    return {"L": bool(code & 1), "R": bool(code & 2),
            "T": bool(code & 4), "B": bool(code & 8)}


def boundary_status(rects, bound):
    """Per-block satisfaction of its boundary code against the layout bbox."""
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    eps = 1e-6
    out = {}
    for i, code in enumerate(bound):
        code = int(code)
        if code == 0:
            continue
        bx, by, bw, bh = rects[i]
        t = {1: abs(bx - x0) < eps, 2: abs(bx + bw - x1) < eps,
             4: abs(by + bh - y1) < eps, 8: abs(by - y0) < eps}
        miss = [bit for bit in (1, 2, 4, 8) if code & bit and not t[bit]]
        out[i] = (code, miss)
    return out


def cluster_contacts(rects, clust, tol=1e-6):
    """Golden contact spanning forest inside every cluster.

    Returns (keep, comps) where keep is a list of
    (i, j, axis, delta) meaning: block i abuts block j on `axis`
    (i is the lower one), sharing an edge of length >= delta.
    comps maps cluster id -> list of connected components (block lists).
    """
    n = len(rects)
    keep, comps = [], {}
    for g in sorted(set(int(c) for c in clust)):
        if g <= 0:
            continue
        idx = [i for i in range(n) if int(clust[i]) == g]
        parent = {i: i for i in idx}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        pairs = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                xi, yi, wi, hi = rects[i]
                xj, yj, wj, hj = rects[j]
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                if oy > tol and abs(xi + wi - xj) < tol:
                    pairs.append((i, j, "x", oy))
                elif oy > tol and abs(xj + wj - xi) < tol:
                    pairs.append((j, i, "x", oy))
                elif ox > tol and abs(yi + hi - yj) < tol:
                    pairs.append((i, j, "y", ox))
                elif ox > tol and abs(yj + hj - yi) < tol:
                    pairs.append((j, i, "y", ox))
        pairs.sort(key=lambda p: -p[3])
        for i, j, ax, ov in pairs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                keep.append((i, j, ax, min(ov, 1.0)))
        groups = {}
        for i in idx:
            groups.setdefault(find(i), []).append(i)
        comps[g] = list(groups.values())
    return keep, comps


def contact_rows(contacts, rects, axis):
    """Linear rows (<=) that preserve the golden contacts on one axis."""
    rows = []
    for (i, j, ax, delta) in contacts:
        wi = rects[i][2] if axis == "x" else rects[i][3]
        wj = rects[j][2] if axis == "x" else rects[j][3]
        if ax == axis:
            # abutment equality: c_j - c_i == size_i
            rows.append(([(j, 1.0), (i, -1.0)], wi))
            rows.append(([(i, 1.0), (j, -1.0)], -wi))
        else:
            # keep a shared edge of length >= delta on the other axis
            rows.append(([(j, 1.0), (i, -1.0)], wi - delta))
            rows.append(([(i, 1.0), (j, -1.0)], wj - delta))
    return rows


def collect_nets(case, axis):
    """Return (nets, pin_terms) for the given axis ('x' or 'y')."""
    n = case.n
    nets = []
    for edge in case.b2b:
        if int(edge[0]) == -1:
            continue
        i, j, wt = int(edge[0]), int(edge[1]), float(edge[2])
        if i < n and j < n:
            nets.append((i, j, wt))
    pin_terms = []
    if case.p2b is not None and len(case.p2b):
        k = 0 if axis == "x" else 1
        for edge in case.p2b:
            if int(edge[0]) == -1:
                continue
            pi, bi, wt = int(edge[0]), int(edge[1]), float(edge[2])
            if bi < n and pi < len(case.pins):
                pin_terms.append((bi, float(case.pins[pi][k]), wt))
    return nets, pin_terms
