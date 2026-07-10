"""Sequence-pair SA feasibility probe (Track: non-column global solver class).

Question under test: can a NON-column representation (sequence-pair + SA) beat
the production column-slicing backbone on the heaviest tail case (test idx=99,
n=120)? Decision gate (written into the report):

  best cost < 1.30  -> class SURVIVES (crude SP already beats the column ceiling)
  1.30 - 1.38       -> MARGINAL  (report throughput bottleneck + extrapolation)
  > 1.38            -> crude SP does not beat production (report cause)

Design (honest, evaluator-faithful, no golden info used as a solver input):
  * Genotype = sequence pair (Gamma+, Gamma-) over the n blocks + a per-block
    discrete aspect index (exact area w*h=a). MIB-group members share one aspect
    (=> mib_violations = 0 by construction). Fixed / preplaced blocks keep their
    exact input (golden) dimensions.
  * Phenotype = coordinates from SP longest-path packing (overlap-free BY
    CONSTRUCTION for the free blocks). Preplaced blocks are honored EXACTLY by
    forcing their pinned coordinate inside the ASAP longest path; an SP whose
    order conflicts with a pin yields an overlap and is scored infeasible
    (cost 10) -- the SA is thus free to explore only pin-consistent orders.
    This is the honest handling: preplaced position deviation is a HARD
    constraint in the evaluator (infeasible), so it is never faked.
  * Objective = the OFFICIAL cost, recomputed in numpy for throughput:
      cost = (1 + 0.5*(max0(hpwl_gap)+max0(area_gap))) * exp(2*v_rel)
      v_rel = (boundary_v + grouping_v + mib_v) / n_soft
    calibrated against the shapely-backed official score_case at startup and on
    every new best (max abs diff reported).

Seeds: (a) random SP; (b) SP extracted from the PRODUCTION column-backbone
layout (tests whether SP moves can improve on production's own topology).

No src/ or FloorSet/ edits. Reuses gen_decoder_probe.{score_case, _baseline,
_golden_rects, _opt_target_positions, _edge_lists, _n_of} and
column_slicing._parse_constraints.

Usage (from FloorSet/iccad2026contest with PYTHONPATH set, see gen_decoder_probe):
  uv run python <repo>/scripts/probes/sp_sa_probe.py \
      --prod-cache <scratch>/prod99_cache.json --idx 99 \
      --restarts 48 --time-budget 120 --out <scratch>/sp99.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
_REPO = None
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        _REPO = _p
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
if _REPO is None:
    _REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
    sys.path.insert(0, str(_REPO / "src"))

from gen_decoder_probe import (  # noqa: E402
    _n_of, _baseline, _golden_rects, _opt_target_positions, _edge_lists,
    score_case,
)
from floorset_arch.legalizer.column_slicing import _parse_constraints  # noqa: E402

Rect = Tuple[float, float, float, float]
EPS = 1e-6

# Discrete aspect candidates (log-spaced ratio w/h). w=sqrt(a*r), h=sqrt(a/r).
ASPECTS = [0.25, 0.333, 0.5, 0.667, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
INFEAS = 10.0


# =============================================================================
# Static per-instance context (picklable, shared to workers)
# =============================================================================
class Ctx:
    def __init__(self, sample, idx: int):
        self.idx = idx
        n = _n_of(sample)
        self.n = n
        at, b2b, p2b, pins, cons = sample["input"]
        self.area = np.array([float(at[i]) for i in range(n)], dtype=np.float64)
        fixed, preplaced, mib, cluster, boundary = _parse_constraints(cons[:n], n)
        self.fixed = np.array(fixed, dtype=bool)
        self.preplaced = np.array(preplaced, dtype=bool)
        self.mib = np.array(mib, dtype=np.int64)
        self.cluster = np.array(cluster, dtype=np.int64)
        self.boundary = np.array(boundary, dtype=np.int64)

        golden = _golden_rects(sample, n)
        self.golden = golden
        tpos = _opt_target_positions(sample, n, golden)
        # exact (golden) dims for fixed/preplaced; pins for preplaced
        self.gold_w = np.array([golden[i][2] for i in range(n)])
        self.gold_h = np.array([golden[i][3] for i in range(n)])
        self.pin_x = np.array([golden[i][0] for i in range(n)])
        self.pin_y = np.array([golden[i][1] for i in range(n)])

        # nets as numpy arrays (valid edges only)
        b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
        bi, bj, bw = [], [], []
        for (i, j, w) in b2b_e:
            if i != -1 and 0 <= i < n and 0 <= j < n and i != j:
                bi.append(i); bj.append(j); bw.append(w)
        self.bi = np.array(bi, dtype=np.int64)
        self.bj = np.array(bj, dtype=np.int64)
        self.bw = np.array(bw, dtype=np.float64)
        self.pins_xy = np.array(pin_l, dtype=np.float64) if len(pin_l) else np.zeros((0, 2))
        pp, pb, pw = [], [], []
        npin = len(pin_l)
        for (p, b, w) in p2b_e:
            if p != -1 and 0 <= b < n and 0 <= p < npin:
                pp.append(p); pb.append(b); pw.append(w)
        self.pp = np.array(pp, dtype=np.int64)
        self.pb = np.array(pb, dtype=np.int64)
        self.pw = np.array(pw, dtype=np.float64)

        # baseline (golden) hpwl / area
        base = _baseline(sample, n, b2b, p2b, pins)
        self.hpwl_base = float(base["hpwl_baseline"])
        self.area_base = float(base["area_baseline"])

        # n_soft
        n_boundary = int((self.boundary != 0).sum())
        n_soft = n_boundary
        for g in range(1, int(self.mib.max()) + 1 if self.mib.size else 1):
            gs = int((self.mib == g).sum())
            n_soft += max(0, gs - 1)
        for g in range(1, int(self.cluster.max()) + 1 if self.cluster.size else 1):
            gs = int((self.cluster == g).sum())
            n_soft += max(0, gs - 1)
        self.n_soft = max(1, n_soft)

        # boundary bit masks
        self.b_left = (self.boundary & 1) != 0
        self.b_right = (self.boundary & 2) != 0
        self.b_top = (self.boundary & 4) != 0
        self.b_bottom = (self.boundary & 8) != 0
        self.b_any = self.boundary != 0

        # cluster member index lists (for grouping union-find)
        self.clusters: List[List[int]] = []
        for g in range(1, int(self.cluster.max()) + 1 if self.cluster.size else 1):
            mem = [i for i in range(n) if self.cluster[i] == g]
            if len(mem) > 1:
                self.clusters.append(mem)

        # MIB group -> members; free-soft = not fixed/preplaced
        self.free_soft = np.array(
            [not (self.fixed[i] or self.preplaced[i]) for i in range(n)], bool)
        self.mib_groups: Dict[int, List[int]] = {}
        for i in range(n):
            if self.mib[i] > 0 and self.free_soft[i]:
                self.mib_groups.setdefault(int(self.mib[i]), []).append(i)
        # aspect "variables": each free-soft block that is NOT in a MIB group is
        # its own var; each MIB group is one shared var.
        self.mib_of = {}  # block -> group id (only free-soft mib blocks)
        for g, mem in self.mib_groups.items():
            for i in mem:
                self.mib_of[i] = g


# =============================================================================
# Shapes from aspect indices
# =============================================================================
def make_shapes(ctx: Ctx, aspect_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = ctx.n
    w = ctx.gold_w.copy()
    h = ctx.gold_h.copy()
    for i in range(n):
        if not ctx.free_soft[i]:
            continue
        r = ASPECTS[aspect_idx[i]]
        a = ctx.area[i]
        w[i] = math.sqrt(a * r)
        h[i] = math.sqrt(a / r)
    return w, h


# =============================================================================
# SP longest-path packing with forced preplaced pins  (overlap-free for free
# blocks; pin conflicts => overlap, caught by _overlaps)
# =============================================================================
def _longest(order: np.ndarray, pos_minus: np.ndarray, dim: np.ndarray,
             pin: np.ndarray, pinned: np.ndarray) -> np.ndarray:
    n = order.shape[0]
    coord = np.zeros(n)
    proc_pm = np.empty(n)
    proc_val = np.empty(n)
    k = 0
    for b in order:
        if pinned[b]:
            c = pin[b]
        elif k > 0:
            m = proc_pm[:k] < pos_minus[b]
            c = proc_val[:k][m].max() if m.any() else 0.0
            if c < 0.0:
                c = 0.0
        else:
            c = 0.0
        coord[b] = c
        proc_pm[k] = pos_minus[b]
        proc_val[k] = c + dim[b]
        k += 1
    return coord


def pack(ctx: Ctx, pos_plus: np.ndarray, pos_minus: np.ndarray,
         w: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # x-graph topo order = ascending pos_plus; predecessors = a with pos_minus<
    order_x = np.argsort(pos_plus, kind="stable")
    x = _longest(order_x, pos_minus, w, ctx.pin_x, ctx.preplaced)
    # y-graph topo order = descending pos_plus; predecessors = a with pos_minus<
    order_y = order_x[::-1].copy()
    y = _longest(order_y, pos_minus, h, ctx.pin_y, ctx.preplaced)
    return x, y


def _overlaps(x, y, w, h) -> int:
    xr = x + w
    yt = y + h
    ox = np.minimum(xr[:, None], xr[None, :]) - np.maximum(x[:, None], x[None, :])
    oy = np.minimum(yt[:, None], yt[None, :]) - np.maximum(y[:, None], y[None, :])
    ov = (ox > 1e-6) & (oy > 1e-6)
    np.fill_diagonal(ov, False)
    return int(ov.sum() // 2)


# =============================================================================
# Faithful fast cost (numpy)
# =============================================================================
def _grouping_v(ctx: Ctx, x, y, w, h) -> int:
    total = 0
    xr = x + w
    yt = y + h
    for mem in ctx.clusters:
        m = len(mem)
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for ai in range(m):
            for bi in range(ai + 1, m):
                i = mem[ai]
                j = mem[bi]
                # vertical shared edge (x abut) with y-overlap>0
                x_abut = (abs(xr[i] - x[j]) < EPS or abs(xr[j] - x[i]) < EPS)
                y_ov = min(yt[i], yt[j]) - max(y[i], y[j])
                h_abut = (abs(yt[i] - y[j]) < EPS or abs(yt[j] - y[i]) < EPS)
                x_ov = min(xr[i], xr[j]) - max(x[i], x[j])
                if (x_abut and y_ov > EPS) or (h_abut and x_ov > EPS):
                    ra, rb = find(ai), find(bi)
                    if ra != rb:
                        parent[ra] = rb
        comps = len({find(a) for a in range(m)})
        total += comps - 1
    return total


def _boundary_v(ctx: Ctx, x, y, w, h) -> int:
    if not ctx.b_any.any():
        return 0
    xr = x + w
    yt = y + h
    xmin = x.min(); ymin = y.min(); xmax = xr.max(); ymax = yt.max()
    ok = np.ones(ctx.n, dtype=bool)
    ok &= ~(ctx.b_left & (np.abs(x - xmin) >= EPS))
    ok &= ~(ctx.b_right & (np.abs(xr - xmax) >= EPS))
    ok &= ~(ctx.b_top & (np.abs(yt - ymax) >= EPS))
    ok &= ~(ctx.b_bottom & (np.abs(y - ymin) >= EPS))
    return int((ctx.b_any & ~ok).sum())


# Overlap penalty for the SA guidance surrogate (each residual overlap pair adds
# this to the uncapped cost, so the search always sees a gradient toward
# feasibility even when the true evaluator would flatly return M=10).
OVERLAP_PEN = 2.0


def eval_layout(ctx: Ctx, x, y, w, h, want_break=False):
    """Return (surrogate, overlaps, true_cost, break_dict).

    surrogate = UNCAPPED qf*vf + OVERLAP_PEN*overlaps  (SA guidance; smooth
        gradient everywhere, including the infeasible/garbage region).
    true_cost = the official capped cost (M=10 if any overlap) -- reporting only.
    """
    overlaps = _overlaps(x, y, w, h)
    cx = x + w / 2.0
    cy = y + h / 2.0
    hb = float(np.sum(ctx.bw * (np.abs(cx[ctx.bi] - cx[ctx.bj])
                                + np.abs(cy[ctx.bi] - cy[ctx.bj])))) if ctx.bi.size else 0.0
    if ctx.pp.size:
        px = ctx.pins_xy[ctx.pp, 0]
        py = ctx.pins_xy[ctx.pp, 1]
        hp = float(np.sum(ctx.pw * (np.abs(px - cx[ctx.pb]) + np.abs(py - cy[ctx.pb]))))
    else:
        hp = 0.0
    hpwl = hb + hp
    area = float((x + w).max() - x.min()) * float((y + h).max() - y.min())
    hpwl_gap = (hpwl - ctx.hpwl_base) / max(ctx.hpwl_base, 1e-6)
    area_gap = (area - ctx.area_base) / max(ctx.area_base, 1e-6)
    bnd = _boundary_v(ctx, x, y, w, h)
    grp = _grouping_v(ctx, x, y, w, h)
    v_rel = (bnd + grp) / ctx.n_soft
    qf = 1.0 + 0.5 * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    vf = math.exp(2.0 * v_rel)
    raw = qf * vf
    surrogate = raw + OVERLAP_PEN * overlaps
    true_cost = INFEAS if overlaps > 0 else min(raw, INFEAS - 1e-6)
    if want_break:
        return surrogate, overlaps, true_cost, {
            "overlaps": overlaps, "hpwl_gap": hpwl_gap, "area_gap": area_gap,
            "v_rel": v_rel, "boundary_v": bnd, "grouping_v": grp,
            "hpwl": hpwl, "area": area}
    return surrogate, overlaps, true_cost, None


def fast_cost(ctx: Ctx, x, y, w, h, want_break=False):
    """Backward-compatible wrapper: returns the true capped cost (+break)."""
    surrogate, overlaps, true_cost, brk = eval_layout(ctx, x, y, w, h, want_break)
    if want_break:
        b = dict(brk)
        b["infeasible"] = overlaps > 0
        return true_cost, b
    return true_cost


def to_rects(x, y, w, h) -> List[Rect]:
    return [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(len(x))]


# =============================================================================
# SP extraction from a placement (for the production seed)
# =============================================================================
def extract_sp(ctx: Ctx, rects: List[Rect]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = ctx.n
    x = np.array([r[0] for r in rects]); y = np.array([r[1] for r in rects])
    w = np.array([r[2] for r in rects]); h = np.array([r[3] for r in rects])
    xr = x + w; yt = y + h
    cx = x + w / 2; cy = y + h / 2
    # before_plus[i][j]=True if i before j in Gamma+ ; likewise minus
    plus_adj = [[] for _ in range(n)]
    minus_adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            left_ij = xr[i] <= x[j] + EPS
            left_ji = xr[j] <= x[i] + EPS
            below_ij = yt[i] <= y[j] + EPS
            below_ji = yt[j] <= y[i] + EPS
            gx = max(x[i] - xr[j], x[j] - xr[i])  # x separation gap (neg if overlap)
            gy = max(y[i] - yt[j], y[j] - yt[i])
            use_x = None
            if (left_ij or left_ji) and (below_ij or below_ji):
                use_x = gx >= gy
            elif left_ij or left_ji:
                use_x = True
            elif below_ij or below_ji:
                use_x = False
            else:
                use_x = abs(cx[i] - cx[j]) >= abs(cy[i] - cy[j])
            if use_x:
                if (left_ij and not left_ji) or (left_ij and left_ji and cx[i] <= cx[j]):
                    a, b = i, j  # i left of j
                elif left_ji:
                    a, b = j, i
                else:
                    a, b = (i, j) if cx[i] <= cx[j] else (j, i)
                # a left of b: a before b in BOTH
                plus_adj[a].append(b); minus_adj[a].append(b)
            else:
                if (below_ij and not below_ji) or (below_ij and below_ji and cy[i] <= cy[j]):
                    lo, hi = i, j  # lo below hi
                elif below_ji:
                    lo, hi = j, i
                else:
                    lo, hi = (i, j) if cy[i] <= cy[j] else (j, i)
                # lo below hi: convention ~A&B -> lo below hi means
                # pos_plus[lo]>pos_plus[hi], pos_minus[lo]<pos_minus[hi]
                # => hi before lo in Gamma+ ; lo before hi in Gamma-
                plus_adj[hi].append(lo)
                minus_adj[lo].append(hi)

    def topo(adj):
        indeg = [0] * n
        for u in range(n):
            for v in adj[u]:
                indeg[v] += 1
        from collections import deque
        # tie-break by geometry for determinism
        q = deque(sorted([i for i in range(n) if indeg[i] == 0]))
        out = []
        while q:
            u = q.popleft()
            out.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return out if len(out) == n else None

    gp = topo(plus_adj)
    gm = topo(minus_adj)
    if gp is None or gm is None:
        # fallback: key sort
        gp = list(np.argsort(cx + cy, kind="stable"))
        gm = list(np.argsort(cx - cy, kind="stable"))
    gamma_plus = np.array(gp, dtype=np.int64)
    gamma_minus = np.array(gm, dtype=np.int64)
    # aspect idx from production shapes (nearest), MIB shared
    aspect = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ctx.free_soft[i]:
            r = w[i] / max(h[i], 1e-9)
            aspect[i] = int(np.argmin([abs(math.log(a) - math.log(r)) for a in ASPECTS]))
    aspect = _tie_mib(ctx, aspect)
    return gamma_plus, gamma_minus, aspect


def _tie_mib(ctx: Ctx, aspect: np.ndarray) -> np.ndarray:
    for g, mem in ctx.mib_groups.items():
        aspect[mem] = aspect[mem[0]]
    return aspect


def pos_from_seq(seq: np.ndarray) -> np.ndarray:
    pos = np.empty_like(seq)
    pos[seq] = np.arange(seq.shape[0])
    return pos


# =============================================================================
# SA worker
# =============================================================================
def sa_run(args_tuple):
    (ctx, seed_kind, seed_sp, seed, time_budget, t0_frac, t1_frac) = args_tuple
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    n = ctx.n

    if seed_kind == "prod" and seed_sp is not None:
        gp = seed_sp[0].copy(); gm = seed_sp[1].copy(); aspect = seed_sp[2].copy()
        # light perturbation so restarts differ
        for _ in range(rng.randint(0, 5)):
            a, b = rng.randrange(n), rng.randrange(n)
            gp[a], gp[b] = gp[b], gp[a]
    else:
        gp = np.arange(n); nprng.shuffle(gp)
        gm = np.arange(n); nprng.shuffle(gm)
        aspect = np.array([rng.randrange(len(ASPECTS)) if ctx.free_soft[i] else 0
                           for i in range(n)], dtype=np.int64)
        aspect = _tie_mib(ctx, aspect)

    pos_p = pos_from_seq(gp)
    pos_m = pos_from_seq(gm)
    w, h = make_shapes(ctx, aspect)
    x, y = pack(ctx, pos_p, pos_m, w, h)
    cur, cur_ov, cur_true, _ = eval_layout(ctx, x, y, w, h)  # cur = surrogate

    seed_true = cur_true
    seed_feasible = cur_ov == 0
    best_feas = cur_true if cur_ov == 0 else float("inf")
    best_state = (gp.copy(), gm.copy(), aspect.copy()) if cur_ov == 0 else None
    best_surr = cur  # best surrogate seen (for progress even if never feasible)
    best_surr_state = (gp.copy(), gm.copy(), aspect.copy())
    evals = 1
    accepts = 0
    feas_evals = 1 if cur_ov == 0 else 0
    t_start = time.time()
    T0 = t0_frac
    T1 = t1_frac
    while True:
        el = time.time() - t_start
        if el >= time_budget:
            break
        frac = el / time_budget
        T = T0 * (T1 / T0) ** frac
        # batch moves between time checks
        for _ in range(200):
            mv = rng.random()
            undo = None
            if mv < 0.32:  # swap in Gamma+
                a, b = rng.randrange(n), rng.randrange(n)
                pos_p[gp[a]], pos_p[gp[b]] = pos_p[gp[b]], pos_p[gp[a]]
                gp[a], gp[b] = gp[b], gp[a]
                undo = ("gp", a, b)
            elif mv < 0.64:  # swap in Gamma-
                a, b = rng.randrange(n), rng.randrange(n)
                pos_m[gm[a]], pos_m[gm[b]] = pos_m[gm[b]], pos_m[gm[a]]
                gm[a], gm[b] = gm[b], gm[a]
                undo = ("gm", a, b)
            elif mv < 0.80:  # swap same block-pair in BOTH sequences
                bi, bj = rng.randrange(n), rng.randrange(n)
                ap, bp = pos_p[bi], pos_p[bj]
                am, bm = pos_m[bi], pos_m[bj]
                gp[ap], gp[bp] = gp[bp], gp[ap]
                pos_p[bi], pos_p[bj] = pos_p[bj], pos_p[bi]
                gm[am], gm[bm] = gm[bm], gm[am]
                pos_m[bi], pos_m[bj] = pos_m[bj], pos_m[bi]
                undo = ("both", ap, bp, am, bm)
            else:  # aspect change on a random free-soft var
                cand = int(nprng.integers(0, n))
                if not ctx.free_soft[cand]:
                    continue
                old = aspect[cand].copy() if hasattr(aspect[cand], "copy") else int(aspect[cand])
                grp = ctx.mib_of.get(cand)
                members = ctx.mib_groups[grp] if grp is not None else [cand]
                newa = rng.randrange(len(ASPECTS))
                oldvals = {m: int(aspect[m]) for m in members}
                for m in members:
                    aspect[m] = newa
                w, h = make_shapes(ctx, aspect)
                undo = ("aspect", oldvals)

            if undo is not None and undo[0] != "aspect":
                pass  # shapes unchanged

            xx, yy = pack(ctx, pos_p, pos_m, w, h)
            cand_surr, cand_ov, cand_true, _ = eval_layout(ctx, xx, yy, w, h)
            evals += 1
            if cand_ov == 0:
                feas_evals += 1
            d = cand_surr - cur
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
                cur = cand_surr
                accepts += 1
                if cand_surr < best_surr:
                    best_surr = cand_surr
                    best_surr_state = (gp.copy(), gm.copy(), aspect.copy())
                if cand_ov == 0 and cand_true < best_feas:
                    best_feas = cand_true
                    best_state = (gp.copy(), gm.copy(), aspect.copy())
            else:
                # undo
                u = undo[0]
                if u == "gp":
                    _, a, b = undo
                    pos_p[gp[a]], pos_p[gp[b]] = pos_p[gp[b]], pos_p[gp[a]]
                    gp[a], gp[b] = gp[b], gp[a]
                elif u == "gm":
                    _, a, b = undo
                    pos_m[gm[a]], pos_m[gm[b]] = pos_m[gm[b]], pos_m[gm[a]]
                    gm[a], gm[b] = gm[b], gm[a]
                elif u == "both":
                    _, ap, bp, am, bm = undo
                    gp[ap], gp[bp] = gp[bp], gp[ap]
                    pos_p[gp[ap]], pos_p[gp[bp]] = ap, bp
                    gm[am], gm[bm] = gm[bm], gm[am]
                    pos_m[gm[am]], pos_m[gm[bm]] = am, bm
                elif u == "aspect":
                    _, oldvals = undo
                    for m, v in oldvals.items():
                        aspect[m] = v
                    w, h = make_shapes(ctx, aspect)

    elapsed = time.time() - t_start
    found_feasible = best_state is not None
    state = best_state if found_feasible else best_surr_state
    return {
        "seed_kind": seed_kind, "seed": seed,
        "seed_true": seed_true, "seed_feasible": seed_feasible,
        "found_feasible": found_feasible,
        "best_feas": best_feas if found_feasible else None,
        "best_surr": best_surr,
        "evals": evals, "accepts": accepts, "feas_evals": feas_evals,
        "elapsed": elapsed, "eps": evals / max(elapsed, 1e-6),
        "accept_rate": accepts / max(evals, 1),
        "feas_frac": feas_evals / max(evals, 1),
        "best_state": (state[0].tolist(), state[1].tolist(), state[2].tolist()),
    }


# =============================================================================
# Driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=99)
    ap.add_argument("--prod-cache", required=True)
    ap.add_argument("--restarts", type=int, default=48)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--time-budget", type=float, default=120.0)
    ap.add_argument("--t0", type=float, default=0.06)
    ap.add_argument("--t1", type=float, default=0.0015)
    ap.add_argument("--prod-frac", type=float, default=0.5,
                    help="fraction of restarts seeded from production SP")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    dp = args.data_path or str(_REPO / "FloorSet")
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(dp)
    sample = ds[args.idx]
    ctx = Ctx(sample, args.idx)
    n = ctx.n

    # production layout from cache
    with open(args.prod_cache) as f:
        cache = {int(k): v for k, v in json.load(f).items()}
    prod_rects = [tuple(r) for r in cache[args.idx]]
    prod_sc = score_case(sample, prod_rects, n)

    # golden score
    gold_sc = score_case(sample, ctx.golden, n)

    # ---- calibration: fast_cost vs score_case ----
    seed_sp = extract_sp(ctx, prod_rects)
    gp0, gm0, asp0 = seed_sp
    w0, h0 = make_shapes(ctx, asp0)
    x0, y0 = pack(ctx, pos_from_seq(gp0), pos_from_seq(gm0), w0, h0)
    fc_seed, brk = fast_cost(ctx, x0, y0, w0, h0, want_break=True)
    sc_seed = score_case(sample, to_rects(x0, y0, w0, h0), n)
    calib = []
    for lbl, rects in [("golden", ctx.golden), ("prod", prod_rects),
                       ("decoded_seed", to_rects(x0, y0, w0, h0))]:
        sc = score_case(sample, rects, n)
        xx = np.array([r[0] for r in rects]); yy = np.array([r[1] for r in rects])
        ww = np.array([r[2] for r in rects]); hh = np.array([r[3] for r in rects])
        fc = fast_cost(ctx, xx, yy, ww, hh)
        calib.append((lbl, sc["cost"], fc, abs(sc["cost"] - fc), sc["feasible"]))

    print("=" * 70)
    print(f"idx={args.idx} n={n} n_soft={ctx.n_soft} "
          f"clusters={[len(c) for c in ctx.clusters]} "
          f"mib_groups={[len(m) for m in ctx.mib_groups.values()]} "
          f"preplaced={int(ctx.preplaced.sum())} fixed={int(ctx.fixed.sum())} "
          f"boundary={int(ctx.b_any.sum())}")
    print(f"GOLDEN   cost={gold_sc['cost']:.4f}  hpwl_gap={gold_sc['hpwl_gap']:.4f} "
          f"area_gap={gold_sc['area_gap']:.4f} v_rel={gold_sc['v_rel']:.4f}")
    print(f"PRODUCT  cost={prod_sc['cost']:.4f}  hpwl_gap={prod_sc['hpwl_gap']:.4f} "
          f"area_gap={prod_sc['area_gap']:.4f} v_rel={prod_sc['v_rel']:.4f} "
          f"(bnd={prod_sc['boundary_v']} grp={prod_sc['grouping_v']})")
    print("--- calibration fast_cost vs score_case (max diff must be small) ---")
    for lbl, c_sc, c_fc, d, feas in calib:
        print(f"  {lbl:14s} score_case={c_sc:.4f} fast={c_fc:.4f} |d|={d:.4f} feas={feas}")
    print(f"decoded-seed feasible={sc_seed['feasible']} score_case={sc_seed['cost']:.4f} "
          f"overlaps(sc)={sc_seed['overlap']}")
    print("=" * 70)

    # ---- launch restarts ----
    tasks = []
    n_prod = int(round(args.restarts * args.prod_frac))
    for r in range(args.restarts):
        kind = "prod" if r < n_prod else "random"
        tasks.append((ctx, kind, seed_sp if kind == "prod" else None,
                      1000 + r, args.time_budget, args.t0, args.t1))

    results = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(sa_run, t) for t in tasks]
        for fu in as_completed(futs):
            results.append(fu.result())
    wall = time.time() - t_all

    # verify each restart best with score_case, keep global best
    best_overall = None
    for res in results:
        gp = np.array(res["best_state"][0]); gm = np.array(res["best_state"][1])
        asp = np.array(res["best_state"][2])
        w, h = make_shapes(ctx, asp)
        x, y = pack(ctx, pos_from_seq(gp), pos_from_seq(gm), w, h)
        rects = to_rects(x, y, w, h)
        sc = score_case(sample, rects, n)
        _s, _ov, _tc, brk = eval_layout(ctx, x, y, w, h, want_break=True)
        res["verified_cost"] = sc["cost"]
        res["verified_feasible"] = sc["feasible"]
        res["fast_cost"] = _tc
        res["break"] = brk if isinstance(brk, dict) else {}
        res["calib_diff"] = abs(sc["cost"] - _tc)
        if sc["feasible"] and (best_overall is None or sc["cost"] < best_overall["verified_cost"]):
            best_overall = res
        res.pop("best_state", None)

    prod_bests = [r for r in results if r["seed_kind"] == "prod" and r["verified_feasible"]]
    rand_bests = [r for r in results if r["seed_kind"] == "random" and r["verified_feasible"]]
    tot_eps = sum(r["eps"] for r in results)
    mean_acc = np.mean([r["accept_rate"] for r in results])

    def best_of(lst):
        return min((r["verified_cost"] for r in lst), default=float("nan"))

    mean_feasfrac = np.mean([r["feas_frac"] for r in results])
    print("\n" + "=" * 70)
    print(f"RESULTS  wall={wall:.1f}s restarts={len(results)} "
          f"total_eps={tot_eps:.0f} mean_accept={mean_acc:.3f} "
          f"mean_feas_frac={mean_feasfrac:.3f}")
    print(f"  best-from-PROD-seed   : {best_of(prod_bests):.4f}  "
          f"(feasible {len(prod_bests)}/{sum(1 for r in results if r['seed_kind']=='prod')})")
    print(f"  best-from-RANDOM-seed : {best_of(rand_bests):.4f}  "
          f"(feasible {len(rand_bests)}/{sum(1 for r in results if r['seed_kind']=='random')})")
    if best_overall:
        b = best_overall
        br = b["break"]
        print(f"  GLOBAL BEST cost={b['verified_cost']:.4f} (fast={b['fast_cost']:.4f} "
              f"calib_diff={b['calib_diff']:.4f}) seed={b['seed_kind']}")
        print(f"    hpwl_gap={br.get('hpwl_gap'):.4f} area_gap={br.get('area_gap'):.4f} "
              f"v_rel={br.get('v_rel'):.4f} bnd={br.get('boundary_v')} grp={br.get('grouping_v')}")
        print(f"  PRODUCTION cost={prod_sc['cost']:.4f}  -> delta={b['verified_cost']-prod_sc['cost']:+.4f}")
        gate = ("SURVIVES(<1.30)" if b['verified_cost'] < 1.30
                else "MARGINAL(1.30-1.38)" if b['verified_cost'] <= 1.38
                else "NO-GO(>1.38)")
        print(f"  GATE: {gate}")
    per_restart_eps = np.median([r["eps"] for r in results])
    per_restart_evals = np.median([r["evals"] for r in results])
    print(f"  throughput: median {per_restart_eps:.0f} evals/s/restart, "
          f"median {per_restart_evals:.0f} evals/restart")
    print("=" * 70)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "idx": args.idx, "n": n, "wall": wall,
                "golden": gold_sc, "production": prod_sc,
                "calib": calib,
                "best_prod": best_of(prod_bests),
                "best_random": best_of(rand_bests),
                "best_overall": {k: v for k, v in (best_overall or {}).items()
                                 if k != "break"} if best_overall else None,
                "best_break": best_overall["break"] if best_overall else None,
                "results": results,
            }, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
