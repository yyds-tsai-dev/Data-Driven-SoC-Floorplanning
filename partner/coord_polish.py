"""PARTNER_COORD_POLISH -- order-preserving simultaneous-axis coordinate polish.

A post-pass over the pipeline's FINAL legal layout.  Everything upstream
(column SA, refiner, direct/flow candidates, vkill) searches over *topology*;
once a topology is fixed the block coordinates are still only locally optimal,
because every coordinate stage in the pipeline is a coordinate descent over
groups.  This pass instead solves, for the frozen topology, the two
*simultaneous* per-axis coordinate problems

    min  w_hpwl * ( SUM_e w_e |cen_i - cen_j| + SUM_k w_k |cen_b - pin_k| )
       + w_frame * ( max_i(c_i + s_i) - min_i c_i )
    s.t. c_i + s_i <= c_j                       for every separation edge
         c_i in [axis_lo, axis_hi - s_i]        (the bbox may shrink, never grow)
         c_i == c_i^0                           for preplaced blocks

which is a linear program.  Because the separation DAG is taken FROM the input
layout (one constraint per block pair, oriented by centroid order), the pass is

  * overlap-free by construction  -- every pair keeps a separating edge,
  * exact-area / fixed-shape safe -- shapes are never touched,
  * preplaced-safe                -- those coordinates are pinned and then
                                     restored bit-exactly,
  * MIB-safe                      -- MIB is a *shape* constraint and shapes
                                     are frozen (asserted anyway).

Only V_boundary / V_grouping can move, and both are re-counted exactly by the
acceptance gate, which is `violation_killer`'s evaluator-faithful proxy
(the official evaluator is never imported by the solver).

Backends
--------
`scipy` (HiGHS, interior point) solves each axis exactly and is the only
backend with measured value.  scipy is NOT in the contest requirements
(FloorSet/iccad2026contest/requirements.txt lists only
torch/numpy/shapely/matplotlib/tqdm/requests), so `auto` degrades to "pass
disabled" when it is missing rather than to the `numpy` backend: the
dependency-free projected-conjugate-subgradient path (built on
`csa_coordinate_solver`) cannot express the wall-abutment rows, and those rows
carry 100% of the gain -- a free HPWL polish is rejected by the acceptance
gate on 26 of 26 active validation cases because it drags wall-flush blocks
off their walls.  `numpy` therefore stays an explicit, experimental opt-in.
`PARTNER_COORD_POLISH_BACKEND` = auto | scipy | numpy.

Containment
-----------
Every failure mode -- import error, infeasible LP, budget overrun, guard
failure, or simply "the proxy did not improve" -- returns the input layout
object unchanged (`partner/../src/floorset_arch/refine/api.py` pattern).
With `PARTNER_COORD_POLISH` unset the module is never imported by
`contest_optimizer`, so the default path is bit-identical.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]

# Own overlap guard, 10x stricter than the evaluator's 1e-6 (same value the
# violation-killer post-pass uses).
OV_GUARD = 1e-7


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def coord_polish_on() -> bool:
    """True iff the pass is enabled.  Bare default = off = bit-exact."""
    return bool(os.environ.get("PARTNER_COORD_POLISH"))


# ---------------------------------------------------------------------------
# evaluator-faithful scoring surface
# ---------------------------------------------------------------------------

class _LiteScorer:
    """Duck-typed stand-in for `_ColumnOptimizer`.

    `violation_killer._Ctx` / `_violations_exact` consume exactly six things
    from their `opt`: `_hpwl`, `_bnd_idx`, `_bnd_codes`, `cluster_groups`,
    `_mib_arrays`, `n_soft_den`.  Building a real `_ColumnOptimizer` here would
    re-run the whole column setup (unit building, frame choice, kernel
    compilation) for a pass that only needs to *score*; this class supplies the
    same six fields from the raw contest tensors in O(n + |E|).

    The arrays are built with the same filters as
    `column_sa_legalizer._build_hpwl_arrays` / `_build_soft_norm`, so the LP
    objective and the acceptance gate see one identical net set.
    """

    __slots__ = ("n", "eI", "eJ", "eW", "pB", "pW", "pX", "pY",
                 "_bnd_idx", "_bnd_codes", "cluster_groups", "_mib_arrays",
                 "n_soft_den", "fixed", "preplaced", "mib", "cluster",
                 "boundary")

    def __init__(self, n: int, constraints, b2b, p2b, pins) -> None:
        self.n = int(n)
        self._build_hpwl(b2b, p2b, pins)
        self._build_soft(constraints)

    # -- HPWL (identical arithmetic to the evaluator's centroid HPWL) -----
    def _build_hpwl(self, b2b, p2b, pins) -> None:
        n = self.n
        if (b2b is not None and len(b2b) > 0 and b2b.dim() == 2
                and b2b.shape[1] >= 3):
            arr = b2b.detach().cpu().numpy()
            m = ((arr[:, 0] != -1) & (arr[:, 0] < n) & (arr[:, 1] < n)
                 & (arr[:, 0] >= 0) & (arr[:, 1] >= 0))
            self.eI = arr[m, 0].astype(np.int64)
            self.eJ = arr[m, 1].astype(np.int64)
            self.eW = arr[m, 2].astype(np.float64)
        else:
            self.eI = np.zeros(0, dtype=np.int64)
            self.eJ = np.zeros(0, dtype=np.int64)
            self.eW = np.zeros(0, dtype=np.float64)
        if (p2b is not None and pins is not None and len(p2b) > 0
                and p2b.dim() == 2 and p2b.shape[1] >= 3 and len(pins) > 0):
            arr = p2b.detach().cpu().numpy()
            pn = pins.detach().cpu().numpy()
            m = ((arr[:, 0] != -1) & (arr[:, 0] >= 0) & (arr[:, 0] < len(pn))
                 & (arr[:, 1] >= 0) & (arr[:, 1] < n))
            pidx = arr[m, 0].astype(np.int64)
            self.pB = arr[m, 1].astype(np.int64)
            self.pW = arr[m, 2].astype(np.float64)
            self.pX = pn[pidx, 0].astype(np.float64)
            self.pY = pn[pidx, 1].astype(np.float64)
        else:
            self.pB = np.zeros(0, dtype=np.int64)
            self.pW = np.zeros(0, dtype=np.float64)
            self.pX = np.zeros(0, dtype=np.float64)
            self.pY = np.zeros(0, dtype=np.float64)

    def _build_soft(self, constraints) -> None:
        n = self.n

        def col(k: int) -> np.ndarray:
            if constraints is None:
                return np.zeros(n, dtype=np.float64)
            arr = constraints.detach().cpu().numpy()
            if arr.ndim != 2 or arr.shape[1] <= k or len(arr) < n:
                return np.zeros(n, dtype=np.float64)
            return arr[:n, k].astype(np.float64)

        self.fixed = col(0) != 0
        self.preplaced = col(1) != 0
        self.mib = np.rint(col(2)).astype(np.int64)
        self.cluster = np.rint(col(3)).astype(np.int64)
        self.boundary = np.rint(col(4)).astype(np.int64)

        bnd = np.nonzero(self.boundary > 0)[0].astype(np.int64)
        self._bnd_idx = bnd
        self._bnd_codes = self.boundary[bnd]

        clus: Dict[int, List[int]] = {}
        for i in range(n):
            g = int(self.cluster[i])
            if g > 0:
                clus.setdefault(g, []).append(i)
        self.cluster_groups = clus

        mibs: Dict[int, List[int]] = {}
        for i in range(n):
            g = int(self.mib[i])
            if g > 0:
                mibs.setdefault(g, []).append(i)
        self._mib_arrays = [np.array(v, dtype=np.int64)
                            for v in mibs.values() if len(v) >= 2]

        n_soft = int(len(bnd))
        for v in mibs.values():
            n_soft += max(0, len(v) - 1)
        for v in clus.values():
            n_soft += max(0, len(v) - 1)
        self.n_soft_den = max(n_soft, 1)

    def _hpwl(self, pos: np.ndarray) -> float:
        cx = pos[:, 0] + pos[:, 2] * 0.5
        cy = pos[:, 1] + pos[:, 3] * 0.5
        total = 0.0
        if len(self.eI):
            total += float(np.sum(self.eW * (np.abs(cx[self.eI] - cx[self.eJ])
                                             + np.abs(cy[self.eI] - cy[self.eJ]))))
        if len(self.pB):
            total += float(np.sum(self.pW * (np.abs(self.pX - cx[self.pB])
                                             + np.abs(self.pY - cy[self.pB]))))
        return total


# ---------------------------------------------------------------------------
# separation topology
# ---------------------------------------------------------------------------

def build_topology(P: np.ndarray):
    """Centroid-ordered pair -> axis assignment (vectorised `gr_lib`).

    For every pair the axis with the LARGER separation gap wins; the direction
    is taken from the CENTROID order on that axis, which is a total order and
    therefore leaves each axis graph acyclic even when the input layout has
    micro-overlaps (the pipeline is only overlap-free to the evaluator's 1e-6).
    Ties break by block index.

    Returns `(hor, ver)`, each an (m, 2) int array where row (a, b) encodes
    `c_a + s_a <= c_b` on that axis.
    """
    n = len(P)
    x0 = P[:, 0]
    y0 = P[:, 1]
    w = P[:, 2]
    h = P[:, 3]
    x1 = x0 + w
    y1 = y0 + h
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h
    # gap[i, j] = max(x0_j - x1_i, x0_i - x1_j): >0 separated, <0 overlapping
    gx = np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])
    gy = np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])
    iu, ju = np.triu_indices(n, 1)
    horiz = gx[iu, ju] >= gy[iu, ju]
    # centroid order: with i < j, (c_i, i) <= (c_j, j)  <=>  c_i <= c_j
    lead_x = cx[iu] <= cx[ju]
    lead_y = cy[iu] <= cy[ju]
    lead = np.where(horiz, lead_x, lead_y)
    a = np.where(lead, iu, ju)
    b = np.where(lead, ju, iu)
    hor = np.stack([a[horiz], b[horiz]], axis=1)
    ver = np.stack([a[~horiz], b[~horiz]], axis=1)
    return hor, ver


def transitive_reduce(n: int, edges: np.ndarray) -> np.ndarray:
    """Drop edges implied by a longer path.

    `c_i + s_i <= c_j` and `c_j + s_j <= c_k` imply `c_i + s_i <= c_k` for
    non-negative sizes, so the reduction preserves the feasible set exactly
    while shrinking the LP by an order of magnitude.
    """
    if len(edges) == 0:
        return edges
    A = np.zeros((n, n), dtype=bool)
    A[edges[:, 0], edges[:, 1]] = True
    C = A.copy()
    for k in range(n):                       # boolean Floyd-Warshall, n <= 120
        C |= np.outer(C[:, k], C[k, :])
    redundant = A & ((A.astype(np.int16) @ C.astype(np.int16)) > 0)
    keep = A & ~redundant
    ii, jj = np.nonzero(keep)
    return np.stack([ii, jj], axis=1)


def cluster_contacts(P: np.ndarray, cluster: np.ndarray, tol: float = 1e-6):
    """Spanning forest of the abutments that currently hold each cluster
    together, as (lo, hi, axis, overlap) tuples.

    Preserving these contacts is what stops the polish from silently splitting
    a satisfied grouping constraint; the acceptance gate would catch it, but
    catching it means throwing the whole polish away.
    """
    n = len(P)
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    keep: List[Tuple[int, int, str, float]] = []
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        g = int(cluster[i])
        if g > 0:
            groups.setdefault(g, []).append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        parent = {i: i for i in idx}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        pairs = []
        for u in range(len(idx)):
            for v in range(u + 1, len(idx)):
                i, j = idx[u], idx[v]
                ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
                oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
                if oy > tol and abs(x1[i] - x0[j]) < tol:
                    pairs.append((i, j, "x", oy))
                elif oy > tol and abs(x1[j] - x0[i]) < tol:
                    pairs.append((j, i, "x", oy))
                elif ox > tol and abs(y1[i] - y0[j]) < tol:
                    pairs.append((i, j, "y", ox))
                elif ox > tol and abs(y1[j] - y0[i]) < tol:
                    pairs.append((j, i, "y", ox))
        pairs.sort(key=lambda p: -p[3])
        for i, j, ax, ov in pairs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                keep.append((i, j, ax, min(ov, 1.0)))
    return keep


def contact_rows(contacts, P: np.ndarray, axis: str):
    """`<=` rows that hold the cluster contacts on one axis."""
    rows = []
    k = 2 if axis == "x" else 3
    for (i, j, ax, delta) in contacts:
        si = float(P[i, k])
        sj = float(P[j, k])
        if ax == axis:                        # abutment equality c_j - c_i == s_i
            rows.append((((j, 1.0), (i, -1.0)), si))
            rows.append((((i, 1.0), (j, -1.0)), -si))
        else:                                 # keep >= delta of shared edge
            rows.append((((j, 1.0), (i, -1.0)), si - delta))
            rows.append((((i, 1.0), (j, -1.0)), sj - delta))
    return rows


# ---------------------------------------------------------------------------
# axis solvers
# ---------------------------------------------------------------------------

def solve_axis_scipy(n, size, lo, hi, edges, nets, pin_terms, eq,
                     min_blocks=(), max_blocks=(), extra_le=(),
                     hpwl_weight=1.0, frame_weight=0.0,
                     time_limit: Optional[float] = None):
    """Exact per-axis LP (HiGHS).  Port of the offline prototype's
    `gr_lib.solve_axis_ext`.

    Boundary abutment is expressed the way the evaluator defines it -- block b
    touches the low wall iff `c_b == min_i c_i` -- via two auxiliary variables,
    so the frame is free to SHRINK rather than being nailed to its input value
    (shrinking is score-free: `area_gap` is clipped at 0 from below).
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    nets = [(i, j, wt) for (i, j, wt) in nets if i != j]
    E = len(nets) + len(pin_terms)
    use_ext = bool(min_blocks) or bool(max_blocks) or frame_weight > 0.0
    npx = n + E + (2 if use_ext else 0)
    vmin, vmax = n + E, n + E + 1

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    b: List[float] = []

    def add_row(entries, rhs):
        r = len(b)
        for c_, v in entries:
            rows.append(r)
            cols.append(c_)
            vals.append(v)
        b.append(rhs)

    for (i, j) in edges:
        add_row(((i, 1.0), (j, -1.0)), -size[i])
    for entries, rhs in extra_le:
        add_row(entries, rhs)

    if use_ext:
        want_min = bool(min_blocks) or frame_weight > 0.0
        want_max = bool(max_blocks) or frame_weight > 0.0
        for i in range(n):
            if want_min:
                add_row(((vmin, 1.0), (i, -1.0)), 0.0)          # cmin <= c_i
            if want_max:
                add_row(((i, 1.0), (vmax, -1.0)), -size[i])     # c_i+s_i <= cmax
        for bi in min_blocks:
            add_row(((bi, 1.0), (vmin, -1.0)), 0.0)             # c_b <= cmin
        for bi in max_blocks:
            add_row(((vmax, 1.0), (bi, -1.0)), size[bi])        # cmax <= c_b+s_b

    for e, (i, j, wt) in enumerate(nets):
        t = n + e
        add_row(((i, 1.0), (j, -1.0), (t, -1.0)), (size[j] - size[i]) / 2.0)
        add_row(((j, 1.0), (i, -1.0), (t, -1.0)), (size[i] - size[j]) / 2.0)
    for k, (i, p, wt) in enumerate(pin_terms):
        t = n + len(nets) + k
        add_row(((i, 1.0), (t, -1.0)), p - size[i] / 2.0)
        add_row(((i, -1.0), (t, -1.0)), -p + size[i] / 2.0)

    A = coo_matrix((vals, (rows, cols)), shape=(len(b), npx)).tocsr()
    c = np.zeros(npx)
    for e, (i, j, wt) in enumerate(nets):
        c[n + e] = wt * hpwl_weight
    for k, (i, p, wt) in enumerate(pin_terms):
        c[n + len(nets) + k] = wt * hpwl_weight
    if use_ext and frame_weight > 0.0:
        c[vmax] += frame_weight
        c[vmin] -= frame_weight

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
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

    # HiGHS honours a wall-clock limit, which is what makes the pass genuinely
    # time-boxed: an LP call is otherwise atomic and a 2.6 s tail would walk
    # straight into the per-case runtime.  A timed-out solve reports
    # success=False and is treated exactly like an infeasible one -- revert.
    options = None
    if time_limit is not None:
        options = {"time_limit": max(0.01, float(time_limit))}
    res = linprog(c, A_ub=A, b_ub=np.asarray(b), bounds=bounds,
                  # IPM over the default simplex: measured identical optima
                  # (mean gain -0.01161 either way on three saved production
                  # runs) at half the wall clock -- t_sum 5.9 s vs 10.9 s,
                  # worst case 1.01 s vs 2.26 s.
                  method=os.environ.get("PARTNER_COORD_POLISH_LP", "highs-ipm"),
                  options=options)
    if not res.success or res.x is None:
        return None
    return np.asarray(res.x[:n], dtype=float)


class _FrameObjective:
    """`AxisHpwlObjective` plus the bbox-extent term.

    `value`/`subgrad` are the only surface `csa_shifts` touches, so the frame
    term is added by delegation instead of by editing the vendored solver.
    """

    __slots__ = ("base", "G", "lo", "hi", "w")

    def __init__(self, base, lo: np.ndarray, hi: np.ndarray, w: float) -> None:
        self.base = base
        self.G = base.G
        self.lo = lo
        self.hi = hi
        self.w = float(w)

    def value(self, d: np.ndarray) -> float:
        f = self.base.value(d)
        if self.w > 0.0:
            f += self.w * (float((self.hi + d).max()) - float((self.lo + d).min()))
        return f

    def subgrad(self, d: np.ndarray) -> np.ndarray:
        g = self.base.subgrad(d)
        if self.w > 0.0:
            g[int(np.argmax(self.hi + d))] += self.w
            g[int(np.argmin(self.lo + d))] -= self.w
        return g


def solve_axis_numpy(n, size, lo, hi, edges, nets, pin_terms, eq,
                     cur, hpwl_weight=1.0, frame_weight=0.0,
                     iters: int = 60, deadline: Optional[float] = None):
    """Dependency-free per-axis solve: projected conjugate subgradient over the
    SAME separation polytope, warm-started at the current layout.

    Reuses `csa_coordinate_solver` at per-block granularity (one "group" per
    block).  Because `d = 0` is the current layout and `csa_shifts` returns the
    best point it visited, the axis objective can only go down -- the backend
    trades optimality for having no dependency, never legality.

    Cluster-contact equalities and boundary abutment rows are NOT expressible
    in the one-sided shift polytope; this backend therefore runs polish-only
    and leans on the acceptance gate to reject any layout whose exact
    V_boundary / V_grouping got worse.
    """
    from csa_coordinate_solver import (AxisHpwlObjective, ShiftPolytope,
                                       csa_shifts)

    cur = np.asarray(cur, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    cen = cur + 0.5 * size

    eu = np.asarray([i for (i, j, w) in nets], dtype=np.int64)
    ev = np.asarray([j for (i, j, w) in nets], dtype=np.int64)
    ew = np.asarray([w * hpwl_weight for (i, j, w) in nets], dtype=np.float64)
    eb = (cen[eu] - cen[ev]) if len(eu) else np.zeros(0)
    pu = np.asarray([i for (i, p, w) in pin_terms], dtype=np.int64)
    pw = np.asarray([w * hpwl_weight for (i, p, w) in pin_terms], dtype=np.float64)
    pb = (cen[pu] - np.asarray([p for (i, p, w) in pin_terms])) if len(pu) \
        else np.zeros(0)
    obj = AxisHpwlObjective(n, eu, ev, eb, ew, pu, pb, pw)

    d_lo = np.asarray(lo, dtype=np.float64) - cur
    d_hi = (np.asarray(hi, dtype=np.float64) - size) - cur
    pinned = np.zeros(n, dtype=bool)
    for i in eq:
        pinned[i] = True
        d_lo[i] = 0.0
        d_hi[i] = 0.0
    d_lo = np.minimum(d_lo, 0.0)
    d_hi = np.maximum(d_hi, 0.0)

    edges_in: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    edges_out: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)
    for (i, j) in edges:
        i, j = int(i), int(j)
        c = float(cur[i] + size[i] - cur[j])       # d_j - d_i >= c
        edges_in[j].append((i, c))
        edges_out[i].append((j, c))
        indeg[j] += 1
    order: List[int] = []
    stack = [i for i in range(n) if indeg[i] == 0]
    deg = indeg.copy()
    while stack:
        u = stack.pop()
        order.append(u)
        for v, _c in edges_out[u]:
            deg[v] -= 1
            if deg[v] == 0:
                stack.append(v)
    if len(order) != n:
        return None                                # cyclic: refuse, revert

    poly = ShiftPolytope(n, d_lo, d_hi, edges_in, edges_out, order, pinned,
                         sweeps=_envi("PARTNER_COORD_POLISH_SWEEPS", 12),
                         omega=_envf("PARTNER_COORD_POLISH_OMEGA", 1.6))
    full = _FrameObjective(obj, cur.copy(), cur + size, frame_weight)
    span = float(max(hi) - min(lo))
    step = _envf("PARTNER_COORD_POLISH_STEP", 0.05) * max(span, 1e-9)
    best_d, best_f, f0, _k = csa_shifts(
        full, poly, step=step, iters=int(iters),
        decay=_envf("PARTNER_COORD_POLISH_DECAY", 0.93), deadline=deadline)
    if not (best_f < f0 - 1e-12):
        return None
    return cur + best_d


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def _overlap_ok(P: np.ndarray) -> bool:
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > OV_GUARD) & (oy > OV_GUARD)
    np.fill_diagonal(bad, False)
    return not bool(bad.any())


def _hard_ok(P0: np.ndarray, P: np.ndarray, scorer: _LiteScorer) -> bool:
    """Hard legality, re-checked rather than assumed.

    Shapes must be bit-identical (that is what keeps exact-area, fixed-shape
    and MIB intact), preplaced origins must be bit-identical, and no pair may
    overlap by more than `OV_GUARD` on both axes.
    """
    if P.shape != P0.shape or not np.isfinite(P).all():
        return False
    if not np.array_equal(P[:, 2:], P0[:, 2:]):
        return False
    pre = scorer.preplaced
    if pre.any() and not np.array_equal(P[pre, :2], P0[pre, :2]):
        return False
    # MIB is a shape constraint; frozen shapes make it invariant, but an
    # invariant worth asserting is an invariant worth testing.
    for g in scorer._mib_arrays:
        before = {(round(float(P0[i, 2]), 4), round(float(P0[i, 3]), 4)) for i in g}
        after = {(round(float(P[i, 2]), 4), round(float(P[i, 3]), 4)) for i in g}
        if after != before:
            return False
    return _overlap_ok(P)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def polish_layout(out: Sequence[Rect], area_targets, constraints,
                  target_positions, b2b, p2b, pins,
                  deadline: Optional[float] = None,
                  verbose: bool = False) -> List[Rect]:
    """Entry point used by `contest_optimizer.MyOptimizer.solve`.

    Returns `out` unchanged on ANY failure, guard rejection, budget overrun or
    non-improvement.  Never raises.
    """
    try:
        res = _polish(list(out), area_targets, constraints, target_positions,
                      b2b, p2b, pins, deadline, verbose)
        return list(out) if res is None else res
    except Exception:
        if verbose or os.environ.get("PARTNER_COORD_POLISH_DEBUG"):
            traceback.print_exc()
        return list(out)


def _merge_pairs(eI: np.ndarray, eJ: np.ndarray, eW: np.ndarray):
    """Sum the weights of parallel b2b terms on the same unordered pair."""
    if not len(eI):
        return []
    keep = eI != eJ
    a = np.minimum(eI[keep], eJ[keep])
    b = np.maximum(eI[keep], eJ[keep])
    w = eW[keep]
    if not len(a):
        return []
    key = np.stack([a, b], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    tot = np.bincount(inv, weights=w, minlength=len(uniq))
    return [(int(uniq[k, 0]), int(uniq[k, 1]), float(tot[k]))
            for k in range(len(uniq))]


def _merge_pins(pB: np.ndarray, pC: np.ndarray, pW: np.ndarray):
    """Sum the weights of pin terms that share a block AND a coordinate.

    Two pins at different coordinates are two different kinks in the piecewise
    objective and must stay separate; two at the same coordinate are one.
    """
    if not len(pB):
        return []
    coords, cinv = np.unique(pC, return_inverse=True)
    key = np.stack([pB, cinv], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    tot = np.bincount(inv, weights=pW, minlength=len(uniq))
    return [(int(uniq[k, 0]), float(coords[uniq[k, 1]]), float(tot[k]))
            for k in range(len(uniq))]


def _lp_rows(n: int, n_edges: int, n_nets: int, n_pins: int,
             n_extra: int, ext: bool) -> int:
    return n_edges + n_extra + 2 * (n_nets + n_pins) + (2 * n if ext else 0)


def _polish(out: List[Rect], area_targets, constraints, target_positions,
            b2b, p2b, pins, deadline, verbose):
    t0 = time.time()
    debug = bool(verbose or os.environ.get("PARTNER_COORD_POLISH_DEBUG"))
    n = len(out)
    if n < _envi("PARTNER_COORD_POLISH_MIN_N", 95):
        return None

    # 600 ms rather than the 300 ms the spec suggested.  The box is a
    # time/score frontier; measured over three saved production runs at the
    # final configuration (highs-ipm, merged terms, boundary mode 2):
    #     200 ms -0.0086 @ +0.040 s/case | 300 ms -0.0109 @ +0.047
    #     400 ms -0.0111 @ +0.050        | 600 ms -0.0146 @ +0.055
    # At the campaign's -0.012 noRT per +0.1 s avg exchange rate that is a net
    # -0.0038 / -0.0053 / -0.0051 / -0.0080 -- 600 ms wins outright.  Above
    # 600 ms nothing changes: the worst observed per-case polish is 0.55 s, so
    # a larger box buys no extra work and only widens the runtime tail risk.
    budget = _envf("PARTNER_COORD_POLISH_BUDGET_MS", 600.0) / 1000.0
    t_end = t0 + budget
    if deadline is not None:
        t_end = min(t_end, float(deadline))
    if t_end - t0 <= 0.01:
        return None

    P0 = np.asarray([[float(r[0]), float(r[1]), float(r[2]), float(r[3])]
                     for r in out], dtype=np.float64)
    scorer = _LiteScorer(n, constraints, b2b, p2b, pins)

    from violation_killer import _Ctx
    ctx = _Ctx(scorer, P0)
    base_score, base_V = ctx.score(P0)

    size_x = P0[:, 2].copy()
    size_y = P0[:, 3].copy()
    x_lo = float(P0[:, 0].min())
    x_hi = float((P0[:, 0] + size_x).max())
    y_lo = float(P0[:, 1].min())
    y_hi = float((P0[:, 1] + size_y).max())

    hor, ver = build_topology(P0)
    hor = transitive_reduce(n, hor)
    ver = transitive_reduce(n, ver)

    # Exact term aggregation.  |cen_i - cen_j| is symmetric, so parallel b2b
    # rows on the same unordered pair collapse into one term with the summed
    # weight; likewise two pins sitting at the same coordinate on one axis.
    # Both are identities, not approximations, and on the validation tail they
    # remove ~35% of the net terms and most of the pin terms -- every one of
    # which costs the LP two rows and a variable.
    nets = _merge_pairs(scorer.eI, scorer.eJ, scorer.eW)
    pin_x = _merge_pins(scorer.pB, scorer.pX, scorer.pW)
    pin_y = _merge_pins(scorer.pB, scorer.pY, scorer.pW)

    pre_idx = np.nonzero(scorer.preplaced)[0]
    eq_x = {int(i): float(P0[i, 0]) for i in pre_idx}
    eq_y = {int(i): float(P0[i, 1]) for i in pre_idx}

    # Backend selection.  `auto` means "scipy or nothing", NOT "scipy or the
    # numpy fallback": the fallback cannot express the wall-abutment rows, and
    # those rows are where 100% of the measured gain lives (a free polish is
    # rejected by the gate in 26/26 active cases because it pulls wall-flush
    # blocks off their walls).  Falling back would therefore spend runtime for
    # a guaranteed zero.  Without scipy the pass simply does not run.
    backend = os.environ.get("PARTNER_COORD_POLISH_BACKEND", "auto").lower()
    if backend in ("auto", "scipy"):
        try:
            import scipy.optimize  # noqa: F401
            backend = "scipy"
        except Exception:
            if debug:
                print("[polish] scipy unavailable; pass disabled", flush=True)
            return None

    use_contacts = (backend == "scipy"
                    and _envi("PARTNER_COORD_POLISH_CONTACTS", 1) != 0)
    contacts = cluster_contacts(P0, scorer.cluster) if use_contacts else []
    rows_x = contact_rows(contacts, P0, "x") if contacts else []
    rows_y = contact_rows(contacts, P0, "y") if contacts else []

    # Second safety net behind the HiGHS wall-clock limit: refuse an absurdly
    # large instance up front instead of paying the setup cost and then timing
    # out inside the solver.  Deliberately loose -- the time box, not this
    # number, is the primary control.
    max_rows = _envi("PARTNER_COORD_POLISH_MAX_ROWS", 40000)
    est = max(_lp_rows(n, len(hor), len(nets), len(pin_x), len(rows_x), True),
              _lp_rows(n, len(ver), len(nets), len(pin_y), len(rows_y), True))
    if backend == "scipy" and est > max_rows:
        if debug:
            print(f"[polish] n={n} skip: est rows {est} > {max_rows}",
                  flush=True)
        return None

    # Objective weights.  The official cost weights HPWL and bbox area by
    # 1/hpwl_baseline and 1/area_baseline; those baselines come from the golden
    # layout and are NOT available at solve time, so the CURRENT layout's own
    # HPWL / area take their place.  That preserves the alpha=0.5 trade-off
    # ratio up to a factor (1 + hpwl_gap) / (1 + area_gap), and the acceptance
    # gate -- not this objective -- is what actually decides.
    hp0 = max(ctx.hp0, 1e-9)
    A0 = max(ctx.A0, 1e-9)
    w_hpwl = 0.5 / hp0
    span_y = max(y_hi - y_lo, 1e-9)

    # Which wall abutments the LP is told to hold.
    #   1 = every boundary-coded bit (the offline prototype's behaviour)
    #   2 = only the bits the input layout ALREADY satisfies
    # Mode 1 asks the LP to REPAIR walls the pipeline could not reach, which on
    # the validation tail makes the x/y system infeasible for 12 of 26 active
    # cases -- and an infeasible attempt is a full LP of runtime for nothing.
    # Mode 2 keeps the same preservation effect without the repair demand.
    # Mode 2 is the default on measured evidence: three saved production runs,
    # mean dNoRT -0.01455 vs -0.01161 for mode 1, with 26/26 active cases
    # improving instead of 14/26 and zero regressions either way.
    bnd_mode = _envi("PARTNER_COORD_POLISH_BOUNDARY", 2)
    want_bnd: Dict[int, int] = {}
    if backend == "scipy" and bnd_mode != 0 and len(scorer._bnd_idx):
        if bnd_mode >= 2:
            eps = 1e-6
            for i, code in zip(scorer._bnd_idx, scorer._bnd_codes):
                i, code = int(i), int(code)
                held = 0
                if (code & 1) and abs(P0[i, 0] - x_lo) < eps:
                    held |= 1
                if (code & 2) and abs(P0[i, 0] + P0[i, 2] - x_hi) < eps:
                    held |= 2
                if (code & 4) and abs(P0[i, 1] + P0[i, 3] - y_hi) < eps:
                    held |= 4
                if (code & 8) and abs(P0[i, 1] - y_lo) < eps:
                    held |= 8
                if held:
                    want_bnd[i] = held
        else:
            want_bnd = {int(i): int(code)
                        for i, code in zip(scorer._bnd_idx, scorer._bnd_codes)}

    def attempt(sel) -> Optional[np.ndarray]:
        if time.time() >= t_end:
            return None
        ta = time.time()
        if backend == "scipy":
            sx = solve_axis_scipy(
                n, size_x, np.full(n, x_lo), np.full(n, x_hi), hor, nets,
                pin_x, eq_x,
                [i for i in sel if want_bnd.get(i, 0) & 1],
                [i for i in sel if want_bnd.get(i, 0) & 2],
                rows_x, w_hpwl, 0.5 * span_y / A0,
                time_limit=t_end - time.time())
            if debug:
                print(f"[polish] n={n} est={est} x-LP {time.time() - ta:.3f}s",
                      flush=True)
            if sx is None or time.time() >= t_end:
                return None
            cur_w = float((sx + size_x).max() - sx.min())
            sy = solve_axis_scipy(
                n, size_y, np.full(n, y_lo), np.full(n, y_hi), ver, nets,
                pin_y, eq_y,
                [i for i in sel if want_bnd.get(i, 0) & 8],
                [i for i in sel if want_bnd.get(i, 0) & 4],
                rows_y, w_hpwl, 0.5 * cur_w / A0,
                time_limit=t_end - time.time())
            if sy is None:
                return None
        else:
            sx = solve_axis_numpy(
                n, size_x, np.full(n, x_lo), np.full(n, x_hi), hor, nets,
                pin_x, eq_x, P0[:, 0], w_hpwl, 0.5 * span_y / A0,
                iters=_envi("PARTNER_COORD_POLISH_ITERS", 60),
                deadline=t_end)
            if sx is None:
                sx = P0[:, 0].copy()
            if time.time() >= t_end:
                return None
            cur_w = float((sx + size_x).max() - sx.min())
            sy = solve_axis_numpy(
                n, size_y, np.full(n, y_lo), np.full(n, y_hi), ver, nets,
                pin_y, eq_y, P0[:, 1], w_hpwl, 0.5 * cur_w / A0,
                iters=_envi("PARTNER_COORD_POLISH_ITERS", 60),
                deadline=t_end)
            if sy is None:
                sy = P0[:, 1].copy()

        Q = P0.copy()
        Q[:, 0] = sx
        Q[:, 1] = sy
        # Re-impose the cluster abutments EXACTLY: the LP holds them to its own
        # tolerance, but shapely's union needs bit-equal edges to call two
        # blocks connected.
        for (i, j, ax, _d) in contacts:
            if ax == "x":
                Q[j, 0] = Q[i, 0] + Q[i, 2]
            else:
                Q[j, 1] = Q[i, 1] + Q[i, 3]
        # Preplaced blocks are restored bit-exactly rather than trusted to the
        # solver's equality bounds.
        if len(pre_idx):
            Q[pre_idx, :2] = P0[pre_idx, :2]
        return Q

    # The boundary-abutment rows are not a repair channel -- they are a
    # PRESERVATION channel.  Measured on three saved production runs: the
    # free (sel = []) polish lowers HPWL but pulls wall-flush blocks off their
    # walls, so V_boundary explodes (1 -> 11, 2 -> 32) and the gate rejects it
    # in 26/26 active cases.  Every point of the -0.0116 comes from the
    # constrained attempt, so the free fallback is pure wasted runtime and is
    # opt-in only (PARTNER_COORD_POLISH_FREE_FALLBACK=1).
    sels: List[Sequence[int]] = []
    if want_bnd:
        sels.append(sorted(want_bnd))
    if not want_bnd or _envi("PARTNER_COORD_POLISH_FREE_FALLBACK", 0) != 0:
        sels.append([])

    best_score = base_score
    best_P: Optional[np.ndarray] = None
    for sel in sels:
        Q = attempt(sel)
        if Q is None:
            continue
        if not _hard_ok(P0, Q, scorer):
            if debug:
                print(f"[polish] n={n} sel={len(sel)} rejected by hard guard",
                      flush=True)
            continue
        s, V = ctx.score(Q)
        if debug:
            print(f"[polish] n={n} sel={len(sel)} score {base_score:.6f} -> "
                  f"{s:.6f} V {base_V} -> {V}", flush=True)
        if s < best_score - 1e-12:
            best_score = s
            best_P = Q
            break
        if time.time() >= t_end:
            break

    if best_P is None:
        return None
    if time.time() - t0 > budget * _envf("PARTNER_COORD_POLISH_OVERRUN", 3.0):
        # A polish that blew far past its box is evidence the size predictor
        # is wrong for this instance shape; keep the layout but say so.
        if debug:
            print(f"[polish] n={n} overran budget "
                  f"({time.time() - t0:.3f}s > {budget:.3f}s)", flush=True)
    return [(float(best_P[i, 0]), float(best_P[i, 1]),
             float(best_P[i, 2]), float(best_P[i, 3])) for i in range(n)]
