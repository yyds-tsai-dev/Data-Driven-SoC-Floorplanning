"""Standalone analytical global-placement (ePlace/CSF-style) candidate arm.

Pure-numpy, gradient-based (Adam) global placer that treats the floorplan as
a continuous optimization: block centers (and, for soft blocks, log-width)
are pushed by an HPWL attraction term, a smoothed pairwise-overlap repulsion
term, an outline containment projection, a boundary-wall attraction term for
L/R/T/B-tagged blocks, and a cluster-centroid attraction term.

This module is standalone and does NOT plug into ``solve()`` -- it is a
probe-only candidate generator.  Overlaps in its output are EXPECTED (and
deliberately targeted); a downstream legalizer is assumed to consume the raw
(x, y, w, h) candidates.

Tensor layout (mirrors ``partner/icdc/data.py`` / ``partner/contest_optimizer.py``
``solve()``):
    area_targets[n]                        block area (torch or numpy)
    constraints[n, 5]                      columns:
        [:, 0] fixed_flag        (shape frozen, position free)
        [:, 1] preplaced_flag    (shape AND position frozen)
        [:, 2] mib_group_id      (0 = none; >0 shares one shape variable)
        [:, 3] cluster_group_id  (0 = none; >0 -> star pseudo-net to centroid)
        [:, 4] boundary_code     (bitmask: 1=L, 2=R, 4=T, 8=B)
    target_positions[n, 4] = (x, y, w, h)  valid columns depend on flags:
        fixed:     w, h valid (x, y = -1 sentinel)
        preplaced: x, y, w, h all valid
    b2b_connectivity[k, 3] = (i, j, weight); rows with i == -1 are padding.
    p2b_connectivity[k, 3] = (pin_idx, block_idx, weight); pin_idx == -1 pads.
    pins_pos[m, 2] = (x, y) absolute pin coordinates.


===========================================================================
2026-08-21 NUMERICS REVIEW -- why v1 lost 72% of wirelength, and the fix
===========================================================================

Probe symptom (test-ids 90-99, n=111-120): median hpwl_ratio
(ePlace_raw / column_legal) = 1.72, overlap_frac converged to ~0.001,
3/10 configs flagged "diverged".  The arm solved the *overlap* problem
essentially exactly and never optimized wirelength at all.

MEASURED TERM SCALES (this dataset; ``/tmp/diag_scales.py`` instrumentation,
cases 79/90/95/99):
    total area A ~ 3.3e4, characteristic length L = sqrt(A) ~ 183,
    mean block width ~ 16.
    b2b edge weights are ALREADY NORMALIZED: mean ~ 0.0029, max 0.034.
    Per-block wirelength gradient magnitude (sum of incident weights)
        = 0.04 .. 0.60   (mean ~ 0.09 - 0.36)
    Per-active-pair overlap gradient at LAMBDA_O_END = 10 is
        lambda_o * oy ~ 10 * 8 = 80, and a block has ~5-20 active pairs.
    => overlap force / wirelength force per block = 1.3e3 .. 1.1e4.
    Likewise LAMBDA_W = 10 and LAMBDA_B_END = 5 are each 15-100x the
    ENTIRE wirelength gradient of a block.

CULPRIT 1 (dominant) -- no inter-term normalization.  v1 lines 40-44
(LAMBDA_O_START/END, LAMBDA_W, LAMBDA_B_START/END) are absolute constants,
but the terms have different physical units: overlap-gradient has units of
*length* (~L/sqrt(n)) while the HPWL gradient has units of *normalized net
weight* (~3e-3).  Because Adam whitens per coordinate, the update direction
is ~100% the overlap/wall/outline gradient; HPWL contributes <0.1% of the
direction.  The optimizer never saw the wirelength objective.
FIX: adaptive lambda from gradient-norm balancing, exactly as ePlace does
(lambda_0 = ||grad_WL||_1 / ||grad_D||_1), plus a closed-loop controller on
the *measured* overlap fraction so the run terminates INSIDE a target
overlap band instead of driving overlap to zero.

CULPRIT 2 -- missing dh/dw term in the soft-shape gradient (v1:354-357).
    d(ox*oy)/dw_i = 0.5*oy*1[ox>0] + ox*1[oy>0]*(dh_i/dw_i)*0.5,
and dh_i/dw_i = -area_i/w_i^2 was omitted.  The retained half is >= 0 for
every active pair, so grad_s was strictly positive every iteration and every
soft block's width monotonically collapsed onto the ASPECT_MIN clip: all soft
blocks ended as 1:3 slivers.  That is bad for HPWL and bad for the
legalizer.  FIX: correct gradient -> grad_s_i = lambda_o*0.5*(oy*w_i - ox*h_i)
(equilibrium at the aspect that balances x- and y-overlap), plus a weak
pull-to-square regularizer.

CULPRIT 3 -- random uniform init (v1:209-210).  With 300 sign-descent steps
and an overlap-dominated direction there is no chance to recover a good
wirelength topology.  PeF spends 9.4% of runtime on a QP wirelength seed.
FIX: closed-form quadratic (star/clique Laplacian + p2b pin anchors +
preplaced elimination + weak center anchor) seed, one dense n x n solve,
n <= 120 => ~100 us.  Per-config diversity now comes from outline (ws,
gamma) plus a small jitter rather than from full randomness.

CULPRIT 4 -- non-smooth |.| subgradient via np.sign (v1:319-331).  sign()
carries no magnitude, so combined with Adam (which also whitens) the
position update is pure sign-descent: it chatters around the weighted-median
optimum and cannot converge.  FIX: soft-abs |d| ~ sqrt(d^2 + eps^2) with
eps = 3% of the outline diagonal (cheap stand-in for ePlace's WA/LSE
smoothing); gradient d/sqrt(d^2+eps^2) is smooth and magnitude-bearing.

CULPRIT 5 -- the "divergence" self-check is apples-to-oranges (v1:304/443/459).
obj0 was evaluated at (LAMBDA_O_START=0.1, LAMBDA_B_START=0.5) and obj_final
at (10.0, 5.0), i.e. 100x / 10x heavier penalties.  A run that improves can
still show obj_final > obj0.  The 3/10 "diverged" cases (n=114-117) are at
least partly mis-flagged.  FIX: evaluate the acceptance objective at
IDENTICAL reference weights, plus real divergence guards: per-axis gradient
percentile clipping, a trust region on per-iteration displacement, a
non-finite rollback, and -- most importantly -- anytime best-iterate
snapshotting (return the lowest-HPWL iterate whose overlap is inside the
band, not the last iterate).

CULPRIT 6 (runtime, not quality) -- ``np.add.at`` (v1:321-357) is the slow
ufunc.at path, and the per-iteration Python loops over cluster/MIB group ids
(v1:382-404, 426-432) re-derive index masks every step.  FIX: np.bincount
scatter-add and precomputed group index arrays.

REVISED RECIPE (implemented below)
    coordinates      : rescaled by L = sqrt(total_area) => outline ~ 1 x 1
    net weights      : normalized to sum 1 => wirelength objective O(1)
    init             : QP wirelength seed + jitter, clipped to outline
    schedule         : phase A (first 15% iters) alpha ~ ALPHA_START, i.e.
                       wirelength-dominated; phase B (15%->75%) geometric
                       overlap-target descent seed_overlap -> OV_TARGET;
                       phase C (75%->100%) hold OV_TARGET.
                       alpha (the balance knob) is moved multiplicatively by
                       a feedback controller on measured overlap_frac.
    lambda_o         : alpha * ||grad_wl||_1 / ||grad_ov_raw||_1  (ePlace)
    lambda_b/lambda_g: beta/gamma * mean per-block wirelength force
    lr               : cosine-decayed Adam, LR_CENTER = 0.025 (outline units)
                       per-block mobility (w_mean/w_i)^0.5 in [0.5, 2.0]
                       (papers precondition by block area; Adam whitens that
                       away, so the mobility factor is re-applied post-Adam)
    trust region     : per-iteration displacement <= 3% of outline diagonal
    iteration budget : 90 (controller gains auto-rescale by ITER_REF/iters, so
                       other budgets stay in-band; below ~75 the spreading
                       dynamics run out of steps and overlap blows past 0.20)
    outline grid     : six gammas, one ws (ws measured dead for best-of-N HPWL)
    acceptance       : anytime best-HPWL iterate with overlap_frac <= OV_SNAP,
                       falling back to the least-overlapping iterate; configs
                       above OV_REJECT are not offered (salvage keeps the list
                       non-empty)

MEASURED RESULT (scripts/probes/eplace_probe.py --test-ids 79-99 --iters 90
--configs 6, column baseline = partner/contest_optimizer.py under _SOLVER_ENV
+ DIRECT_OFF=1 PARTNER_FLOW_SLOTS=0):
    median hpwl_ratio  1.72  ->  ~0.73     (bar: <= 0.95)      PASS
    overlap_frac       0.001 ->  0.09-0.19 (bar: 0.03-0.20)    PASS
    diverged configs   3/10  ->  0/21                          PASS
    runtime (6 cfg)    ~0.44 s/case       (bar: <= 0.30 s)     MISS
                       4 configs (identical median ratio) fits at ~0.23 s.
    bbox_area vs column layout: 0.98x -- the HPWL win is NOT bought by
    inflating the outline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module constants (objective weights / optimizer hyperparameters)
# ---------------------------------------------------------------------------
# Legacy names kept for API/back-compat; LAMBDA_O_* are now the *alpha*
# (dimensionless gradient-balance) endpoints, not absolute penalty weights.
LAMBDA_O_START = 0.05        # alpha at the start of the spreading phase
LAMBDA_O_END = 4.0           # alpha ceiling (controller clamp)
LAMBDA_W = 0.0               # outline penalty is handled by projection now
LAMBDA_B_START = 0.3         # wall pull, in units of mean per-block WL force
LAMBDA_B_END = 3.0
ASPECT_MIN = 1.0 / 3.0
ASPECT_MAX = 3.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
S_LR = 0.008                 # Adam lr on log-width
# Outline grid.  Measured: whitespace (ws) is a DEAD knob for best-of-N
# HPWL (median ratio identical for n_configs 3..6 on the old 2x3 grid),
# while outline aspect (gamma) is the live one -- so spend all six slots
# on gamma.  Ordered so that any prefix of length k is still diverse.
WS_GRID = (0.05,)
GAMMA_GRID = (1.0, 0.7, 1.25, 0.85, 1.6, 0.55)

# --- revised-recipe constants ---------------------------------------------
LR_CENTER = 0.025            # Adam lr on centers, in outline units
TRUST_REGION = 0.03          # max per-iteration displacement / outline diag
GRAD_CLIP_PCT = 99.0         # percentile used by the outlier gradient guard
GRAD_CLIP_TRIGGER = 10.0     # only clip when max|g| > TRIGGER * p99
SOFT_ABS_EPS = 0.03          # HPWL smoothing, fraction of outline diagonal
OV_TARGET = 0.06             # residual overlap fraction we hand the legalizer
OV_SNAP_MAX = 0.13           # best-iterate snapshot admissibility ceiling
OV_REJECT = 0.18             # a candidate above this overlap fraction is not
                             # offered to the legalizer (unless nothing else exists)
WL_PHASE = 0.15              # fraction of iters that stay wirelength-dominated
SPREAD_END = 0.75            # by this fraction the overlap target reaches OV_TARGET
ALPHA_UP = 1.06              # controller gains
ALPHA_DOWN = 0.97
ALPHA_MIN = 1e-3
ALPHA_MAX = 2000.0          # controller clamp (must not saturate: at 30 the
                            # controller had no authority and every case stalled)
ITER_REF = 180               # iteration budget the controller gains are tuned for
WL_FLOOR = 1e-3              # lower bound on ||grad_wl||_1 (zero-connectivity guard)
CLUSTER_BETA = 0.5           # cluster pull, in units of mean per-block WL force
ASPECT_REG = 0.25            # pull-to-square on log-width, rel. to overlap grad
QP_ANCHOR = 0.02             # weak center anchor weight in the QP seed
SEED_JITTER = 0.02           # per-config init jitter, in outline units
NBR_REFRESH = 5              # iterations between neighbor-list rebuilds
NBR_MARGIN = 0.045            # neighbor-list slack, fraction of outline diagonal
OUTLINE_MIN_UTIL = 1.02      # minimum outline area / total block area
ROLLBACK_EVERY = 10          # iterations between non-finite checks / rollback snapshots


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    """Convert a torch tensor or array-like to a float64 numpy array."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _configs(n_configs: int) -> List[Tuple[float, float]]:
    """Cartesian-ish sample of (ws, gamma) up to n_configs entries."""
    grid = [(ws, gamma) for ws in WS_GRID for gamma in GAMMA_GRID]
    return grid[:n_configs] if n_configs <= len(grid) else grid + grid[: n_configs - len(grid)]


def _adam_step(param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
               t: int, lr: float) -> np.ndarray:
    m[...] = ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad
    v[...] = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (grad * grad)
    m_hat = m / (1 - ADAM_BETA1 ** t)
    v_hat = v / (1 - ADAM_BETA2 ** t)
    return param - lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


def _ramp(iter_idx: int, iters: int, start: float, end: float) -> float:
    if iters <= 1:
        return end
    frac = min(1.0, iter_idx / float(iters - 1))
    return start + frac * (end - start)


def _cosine_lr(base_lr: float, iter_idx: int, iters: int) -> float:
    if iters <= 1:
        return base_lr
    frac = iter_idx / float(iters - 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * frac))


def _scatter(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """Fast scatter-add (np.bincount) replacement for np.add.at."""
    if idx.size == 0:
        return np.zeros(n)
    return np.bincount(idx, weights=vals, minlength=n)


def _guard_grad(g: np.ndarray) -> np.ndarray:
    """Outlier gradient guard: clip only when a coordinate is pathological.

    Uses mean(|g|) rather than a percentile -- np.percentile costs a partial
    sort and measured 35% of the whole inner loop.
    """
    mag = np.abs(g)
    if mag.size == 0:
        return g
    mx = float(mag.max())
    mean = float(mag.mean())
    if mean > 0.0 and mx > GRAD_CLIP_TRIGGER * mean:
        thr = GRAD_CLIP_TRIGGER * mean
        return np.clip(g, -thr, thr)
    return g


def _spd_solve(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve an SPD system without LAPACK ``gesv``.

    ``np.linalg.solve`` measured 576 ms on a 116x116 system in this
    environment (threaded-OpenBLAS pathology) while ``np.linalg.cholesky``
    measured 0.57 ms, so factor with Cholesky and run the two triangular
    solves in Python (n <= 120 -> ~0.5 ms).
    """
    n = mat.shape[0]
    lc = np.linalg.cholesky(mat)
    diag = np.diag(lc)
    y = np.zeros_like(rhs)
    for k in range(n):
        y[k] = (rhs[k] - lc[k, :k] @ y[:k]) / diag[k]
    x = np.zeros_like(rhs)
    for k in range(n - 1, -1, -1):
        x[k] = (y[k] - lc[k + 1:, k] @ x[k + 1:]) / diag[k]
    return x


def _qp_seed(n: int, b2b_i: np.ndarray, b2b_j: np.ndarray, b2b_w: np.ndarray,
             p2b_pin: np.ndarray, p2b_blk: np.ndarray, p2b_w: np.ndarray,
             pins: np.ndarray, preplaced: np.ndarray, pre_cx: np.ndarray,
             pre_cy: np.ndarray, w_out: float, h_out: float
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form quadratic wirelength seed (PeF-style QP initial solution).

    Builds the connectivity Laplacian from b2b edges, adds p2b pin anchors,
    a weak anchor to the outline center (keeps the system non-singular for
    unconnected blocks), eliminates preplaced blocks, and solves once per
    axis.  Dense n x n solve; n <= 120 makes this ~0.1 ms.
    """
    tot_w = float(b2b_w.sum() + p2b_w.sum())
    anchor = QP_ANCHOR * max(tot_w, 1e-12) / max(n, 1)

    # Laplacian assembly via a single flat bincount.
    flat_idx: List[np.ndarray] = []
    flat_val: List[np.ndarray] = []
    if b2b_i.size:
        flat_idx.append(b2b_i * n + b2b_i)
        flat_val.append(b2b_w)
        flat_idx.append(b2b_j * n + b2b_j)
        flat_val.append(b2b_w)
        flat_idx.append(b2b_i * n + b2b_j)
        flat_val.append(-b2b_w)
        flat_idx.append(b2b_j * n + b2b_i)
        flat_val.append(-b2b_w)
    if p2b_blk.size:
        flat_idx.append(p2b_blk * n + p2b_blk)
        flat_val.append(p2b_w)
    if flat_idx:
        lap = np.bincount(np.concatenate(flat_idx).astype(np.int64),
                          weights=np.concatenate(flat_val),
                          minlength=n * n).reshape(n, n)
    else:
        lap = np.zeros((n, n))

    rx = np.zeros(n)
    ry = np.zeros(n)
    if p2b_blk.size:
        rx += _scatter(p2b_blk, p2b_w * pins[p2b_pin, 0], n)
        ry += _scatter(p2b_blk, p2b_w * pins[p2b_pin, 1], n)

    di = np.diag_indices(n)
    lap[di] += anchor
    rx += anchor * (w_out / 2.0)
    ry += anchor * (h_out / 2.0)

    if preplaced.any():
        idx = np.where(preplaced)[0]
        rx -= lap[:, idx] @ pre_cx[idx]
        ry -= lap[:, idx] @ pre_cy[idx]
        lap[idx, :] = 0.0
        lap[:, idx] = 0.0
        lap[idx, idx] = 1.0
        rx[idx] = pre_cx[idx]
        ry[idx] = pre_cy[idx]

    try:
        sol = _spd_solve(lap, np.stack([rx, ry], axis=1))
        cx = sol[:, 0]
        cy = sol[:, 1]
    except np.linalg.LinAlgError:
        cx = np.linalg.lstsq(lap, rx, rcond=None)[0]
        cy = np.linalg.lstsq(lap, ry, rcond=None)[0]
    if not (np.isfinite(cx).all() and np.isfinite(cy).all()):
        cx = np.full(n, w_out / 2.0)
        cy = np.full(n, h_out / 2.0)
    return cx, cy


def _run_config(area: np.ndarray, cons: np.ndarray, tp: np.ndarray,
                 b2b: np.ndarray, p2b: np.ndarray, pins: np.ndarray,
                 ws: float, gamma: float, iters: int, seed: int,
                 mean_b2b_weight: float) -> Dict[str, Any]:
    n = area.shape[0]
    rng = np.random.default_rng(seed)

    fixed = cons[:, 0] != 0
    preplaced = cons[:, 1] != 0
    mib_gid = cons[:, 2].astype(np.int64)
    cluster_gid = cons[:, 3].astype(np.int64)
    bcode = cons[:, 4].astype(np.int64)

    soft = ~fixed & ~preplaced  # free shape AND free position

    a_tot_phys = float(area.sum())
    # --- FIX (culprit 1): rescale to unit total area so every term is O(1).
    scale = math.sqrt(max(a_tot_phys, 1e-12))
    area_s = area / max(a_tot_phys, 1e-12)          # sums to 1
    tp_s = tp / scale
    pins_s = pins / scale if pins.size else pins
    a_tot = 1.0

    w_out = math.sqrt((1.0 + ws) * gamma)
    h_out = math.sqrt((1.0 + ws) / gamma)

    # Grow the outline so preplaced blocks are contained.
    has_pre = bool(preplaced.any())
    pre_x2 = pre_y2 = None
    if has_pre:
        pre_x2 = tp_s[preplaced, 0] + tp_s[preplaced, 2]
        pre_y2 = tp_s[preplaced, 1] + tp_s[preplaced, 3]
        w_out = max(w_out, float(pre_x2.max()))
        h_out = max(h_out, float(pre_y2.max()))

    # Wall-pinned outline override: detect an L/R preplaced pair implying a
    # fixed outline width (and similarly T/B for height).
    w_star = None
    h_star = None
    if has_pre:
        l_mask = preplaced & (bcode & 1 != 0)
        r_mask = preplaced & (bcode & 2 != 0)
        t_mask = preplaced & (bcode & 4 != 0)
        b_mask = preplaced & (bcode & 8 != 0)
        if l_mask.any() and r_mask.any():
            l_x = float(tp_s[l_mask, 0].min())
            r_x2 = float((tp_s[r_mask, 0] + tp_s[r_mask, 2]).max())
            w_star = r_x2 - l_x
        if t_mask.any() and b_mask.any():
            b_y = float(tp_s[b_mask, 1].min())
            t_y2 = float((tp_s[t_mask, 1] + tp_s[t_mask, 3]).max())
            h_star = t_y2 - b_y
    if w_star is not None and (seed % 2 == 0):
        w_out = max(w_star, float(pre_x2.max()) if has_pre else 0.0)
    if h_star is not None and (seed % 2 == 0):
        h_out = max(h_star, float(pre_y2.max()) if has_pre else 0.0)

    # FIX: the wall-pinned override can SHRINK an axis without compensating
    # the other one, yielding an outline smaller than the total block area
    # (measured on test_id 93: 0.895 * A_tot -- the target overlap band is
    # then geometrically unreachable and the run stalls at ~0.22 overlap).
    # Restore feasibility by growing the un-pinned axis first.
    need = OUTLINE_MIN_UTIL * a_tot
    if w_out * h_out < need:
        w_pinned = w_star is not None and (seed % 2 == 0)
        h_pinned = h_star is not None and (seed % 2 == 0)
        if w_pinned and not h_pinned:
            h_out = need / max(w_out, 1e-12)
        elif h_pinned and not w_pinned:
            w_out = need / max(h_out, 1e-12)
        else:
            grow = math.sqrt(need / max(w_out * h_out, 1e-12))
            w_out *= grow
            h_out *= grow
    diag_len = math.hypot(w_out, h_out)

    # ---- shape state: log-width s for the representative of each MIB
    # group and every non-grouped soft block; frozen w/h for fixed/preplaced.
    w = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    frozen = fixed | preplaced
    w[frozen] = tp_s[frozen, 2]
    h[frozen] = tp_s[frozen, 3]

    group_ids = sorted(set(int(g) for g in mib_gid if g > 0))
    s = np.log(np.sqrt(np.maximum(area_s, 1e-18)))
    s0 = s.copy()                                   # square-aspect reference
    group_s_owner: Dict[int, int] = {}
    group_frozen: Dict[int, bool] = {}
    group_mean_area: Dict[int, float] = {}
    for gid in group_ids:
        members = np.where(mib_gid == gid)[0]
        frozen_members = members[frozen[members]]
        soft_members = members[soft[members]]
        group_mean_area[gid] = float(area_s[members].mean())
        if len(frozen_members) > 0:
            group_frozen[gid] = True
            ref = frozen_members[0]
            w[soft_members] = w[ref]
            h[soft_members] = h[ref]
        else:
            group_frozen[gid] = False
            owner = int(members[0])
            group_s_owner[gid] = owner
            init_s = 0.5 * math.log(max(group_mean_area[gid], 1e-18))
            s[members] = init_s
            s0[members] = init_s

    ungrouped_soft = soft & (mib_gid == 0)
    s[ungrouped_soft] = s0[ungrouped_soft] + rng.normal(
        0.0, 0.10, size=int(ungrouped_soft.sum()))

    def dims_from_s(s_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ww = np.exp(s_vec)
        hh = area_s / np.maximum(ww, 1e-18)
        return ww, hh

    s_var_list = sorted(set(int(v) for v in np.where(ungrouped_soft)[0])
                        | set(int(v) for v in group_s_owner.values()))
    s_var_idx = np.array(s_var_list, dtype=np.int64) if s_var_list else np.zeros(0, np.int64)
    s_var_mask = np.zeros(n, dtype=bool)
    s_var_mask[s_var_idx] = True

    # FIX (culprit 6): precompute unfrozen MIB group index arrays once.
    mib_soft_groups: List[Tuple[int, np.ndarray]] = []
    for gid in group_ids:
        if group_frozen[gid]:
            continue
        members = np.where(mib_gid == gid)[0]
        mib_soft_groups.append((group_s_owner[gid], members[soft[members]]))

    def _mirror_groups() -> None:
        for owner, soft_members in mib_soft_groups:
            w[soft_members] = w[owner]
            h[soft_members] = h[owner]

    ww0, hh0 = dims_from_s(s)
    w[~frozen] = ww0[~frozen]
    h[~frozen] = hh0[~frozen]
    _mirror_groups()

    # ---- connectivity (dense arrays, padding rows dropped).
    if b2b is not None and b2b.size:
        b2b_valid = b2b[b2b[:, 0] != -1]
        b2b_i = b2b_valid[:, 0].astype(np.int64)
        b2b_j = b2b_valid[:, 1].astype(np.int64)
        b2b_w_raw = b2b_valid[:, 2]
    else:
        b2b_i = np.zeros(0, dtype=np.int64)
        b2b_j = np.zeros(0, dtype=np.int64)
        b2b_w_raw = np.zeros(0, dtype=np.float64)

    if p2b is not None and p2b.size:
        p2b_valid = p2b[p2b[:, 0] != -1]
        p2b_pin = p2b_valid[:, 0].astype(np.int64)
        p2b_blk = p2b_valid[:, 1].astype(np.int64)
        p2b_w_raw = p2b_valid[:, 2]
    else:
        p2b_pin = np.zeros(0, dtype=np.int64)
        p2b_blk = np.zeros(0, dtype=np.int64)
        p2b_w_raw = np.zeros(0, dtype=np.float64)

    # FIX (culprit 1): normalize net weights so the WL objective is O(1).
    w_tot = float(b2b_w_raw.sum() + p2b_w_raw.sum())
    w_tot = w_tot if w_tot > 1e-18 else 1.0
    b2b_w = b2b_w_raw / w_tot
    p2b_w = p2b_w_raw / w_tot

    # pairwise index grid (upper triangle).  ``iu_f/ju_f`` is the exact full
    # set (used for measurement); the inner loop works on a periodically
    # rebuilt neighbor list (``nb_i/nb_j``), which is ~5x smaller.
    iu_f, ju_f = np.triu_indices(n, k=1)
    iu, ju = iu_f, ju_f
    nb_i, nb_j = iu_f, ju_f

    # cluster membership (precomputed flat arrays, culprit 6)
    cl_unique = np.array(sorted(set(int(g) for g in cluster_gid if g > 0)), dtype=np.int64)
    if cl_unique.size:
        cl_all = np.where(cluster_gid > 0)[0]
        cl_lab_all = np.searchsorted(cl_unique, cluster_gid[cl_all])
        counts = np.bincount(cl_lab_all, minlength=cl_unique.size)
        keep = counts[cl_lab_all] >= 2
        cl_idx = cl_all[keep]
        cl_lab = cl_lab_all[keep]
        n_cl = int(cl_unique.size)
        cl_cnt = np.bincount(cl_lab, minlength=n_cl).astype(np.float64)
        cl_cnt[cl_cnt == 0] = 1.0
    else:
        cl_idx = np.zeros(0, dtype=np.int64)
        cl_lab = np.zeros(0, dtype=np.int64)
        cl_cnt = np.zeros(0)
        n_cl = 0

    # boundary tags
    free_pos = ~preplaced
    l_bias = free_pos & (bcode & 1 != 0)
    r_bias = free_pos & (bcode & 2 != 0)
    t_bias = free_pos & (bcode & 4 != 0)
    b_bias = free_pos & (bcode & 8 != 0)
    l_idx = np.where(l_bias)[0]
    r_idx = np.where(r_bias)[0]
    t_idx = np.where(t_bias)[0]
    b_idx = np.where(b_bias)[0]
    n_wall = l_idx.size + r_idx.size + t_idx.size + b_idx.size

    # ---- position state: QP wirelength seed (FIX, culprit 3).
    pre_cx = np.zeros(n)
    pre_cy = np.zeros(n)
    if has_pre:
        pre_cx[preplaced] = tp_s[preplaced, 0] + tp_s[preplaced, 2] / 2.0
        pre_cy[preplaced] = tp_s[preplaced, 1] + tp_s[preplaced, 3] / 2.0
    cx, cy = _qp_seed(n, b2b_i, b2b_j, b2b_w, p2b_pin, p2b_blk, p2b_w,
                      pins_s, preplaced, pre_cx, pre_cy, w_out, h_out)
    jit = SEED_JITTER * diag_len
    cx = cx + rng.normal(0.0, jit, size=n)
    cy = cy + rng.normal(0.0, jit, size=n)
    # boundary-tag bias nudges the seed toward its wall (soft, not a reset).
    if l_idx.size:
        cx[l_idx] = 0.5 * cx[l_idx] + 0.5 * (w[l_idx] / 2.0)
    if r_idx.size:
        cx[r_idx] = 0.5 * cx[r_idx] + 0.5 * (w_out - w[r_idx] / 2.0)
    if t_idx.size:
        cy[t_idx] = 0.5 * cy[t_idx] + 0.5 * (h_out - h[t_idx] / 2.0)
    if b_idx.size:
        cy[b_idx] = 0.5 * cy[b_idx] + 0.5 * (h[b_idx] / 2.0)
    cx[preplaced] = pre_cx[preplaced]
    cy[preplaced] = pre_cy[preplaced]

    def project(cx_: np.ndarray, cy_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hw = w * 0.5
        hh = h * 0.5
        np.clip(cx_, hw, np.maximum(w_out - hw, hw), out=cx_)
        np.clip(cy_, hh, np.maximum(h_out - hh, hh), out=cy_)
        if has_pre:
            cx_[preplaced] = pre_cx[preplaced]
            cy_[preplaced] = pre_cy[preplaced]
        return cx_, cy_

    cx, cy = project(cx, cy)

    lr_center = LR_CENTER * diag_len
    lr_s = S_LR
    eps_wl = SOFT_ABS_EPS * diag_len
    tr_step = TRUST_REGION * diag_len

    # per-block mobility: papers precondition by block area; Adam whitens
    # that away, so re-apply it multiplicatively after the Adam step.
    w_ref = float(np.sqrt(area_s).mean())
    mobility = np.clip(np.sqrt(w_ref / np.maximum(np.sqrt(area_s), 1e-12)), 0.5, 2.0)

    m_cx = np.zeros(n)
    v_cx = np.zeros(n)
    m_cy = np.zeros(n)
    v_cy = np.zeros(n)
    m_s = np.zeros(n)
    v_s = np.zeros(n)

    obj_trace: List[float] = []
    free_center_mask = ~preplaced
    free_mult = free_center_mask.astype(np.float64)

    # ---------------- objective helpers -----------------------------------
    def wl_of(cx_: np.ndarray, cy_: np.ndarray) -> float:
        """Normalized (weights sum to 1, unit-area coords) HPWL."""
        val = 0.0
        if b2b_i.size:
            val += float(np.sum(b2b_w * (np.abs(cx_[b2b_i] - cx_[b2b_j])
                                         + np.abs(cy_[b2b_i] - cy_[b2b_j]))))
        if p2b_pin.size:
            val += float(np.sum(p2b_w * (np.abs(cx_[p2b_blk] - pins_s[p2b_pin, 0])
                                         + np.abs(cy_[p2b_blk] - pins_s[p2b_pin, 1]))))
        return val

    def overlap_of(cx_: np.ndarray, cy_: np.ndarray,
                   w_: np.ndarray, h_: np.ndarray) -> float:
        if not iu_f.size:
            return 0.0
        ox = np.maximum(0.0, (w_[iu_f] + w_[ju_f]) / 2.0 - np.abs(cx_[iu_f] - cx_[ju_f]))
        oy = np.maximum(0.0, (h_[iu_f] + h_[ju_f]) / 2.0 - np.abs(cy_[iu_f] - cy_[ju_f]))
        return float(np.sum(ox * oy))

    def rebuild_neighbors(cx_: np.ndarray, cy_: np.ndarray, w_: np.ndarray,
                          h_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prune the O(n^2) pair set to pairs that can overlap soon."""
        if not iu_f.size:
            return iu_f, ju_f
        m = NBR_MARGIN * diag_len
        near = ((np.abs(cx_[iu_f] - cx_[ju_f]) < (w_[iu_f] + w_[ju_f]) / 2.0 + m)
                & (np.abs(cy_[iu_f] - cy_[ju_f]) < (h_[iu_f] + h_[ju_f]) / 2.0 + m))
        return iu_f[near], ju_f[near]

    def wall_of(cx_: np.ndarray, cy_: np.ndarray,
                w_: np.ndarray, h_: np.ndarray) -> float:
        val = 0.0
        if l_idx.size:
            val += float(np.sum(np.abs(cx_[l_idx] - w_[l_idx] / 2.0)))
        if r_idx.size:
            val += float(np.sum(np.abs(w_out - (cx_[r_idx] + w_[r_idx] / 2.0))))
        if t_idx.size:
            val += float(np.sum(np.abs(h_out - (cy_[t_idx] + h_[t_idx] / 2.0))))
        if b_idx.size:
            val += float(np.sum(np.abs(cy_[b_idx] - h_[b_idx] / 2.0)))
        return val

    def cluster_of(cx_: np.ndarray, cy_: np.ndarray) -> float:
        if not cl_idx.size:
            return 0.0
        gx = np.bincount(cl_lab, weights=cx_[cl_idx], minlength=n_cl) / cl_cnt
        gy = np.bincount(cl_lab, weights=cy_[cl_idx], minlength=n_cl) / cl_cnt
        return float(np.sum(np.abs(cx_[cl_idx] - gx[cl_lab])
                            + np.abs(cy_[cl_idx] - gy[cl_lab])))

    # FIX (culprit 5): a single reference objective, evaluated at IDENTICAL
    # weights for the initial and final states.
    ref_lo = LAMBDA_O_END
    ref_lb = LAMBDA_B_END

    def objective(cx_: np.ndarray, cy_: np.ndarray, w_: np.ndarray, h_: np.ndarray,
                  lambda_o: float = ref_lo, lambda_b: float = ref_lb) -> float:
        return (wl_of(cx_, cy_)
                + lambda_o * overlap_of(cx_, cy_, w_, h_)
                + lambda_b * wall_of(cx_, cy_, w_, h_) / max(n_wall, 1)
                + CLUSTER_BETA * cluster_of(cx_, cy_) / max(cl_idx.size, 1))

    obj0 = objective(cx, cy, w, h)
    obj_trace.append(obj0)

    ov_seed = max(overlap_of(cx, cy, w, h) / a_tot, 1e-6)
    alpha = LAMBDA_O_START
    # Controller gains are per-iteration, so rescale them by the iteration
    # budget: otherwise a short run (iters << ITER_REF) cannot climb alpha far
    # enough to spread the placement and terminates with huge overlap.
    gain_k = ITER_REF / float(max(iters, 1))
    a_up = ALPHA_UP ** gain_k
    a_dn = ALPHA_DOWN ** gain_k
    snap_every = max(4, iters // 10)

    best_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    best_wl = float("inf")
    # Fallback used when NO iterate ever reaches the overlap band (e.g. a
    # geometrically tight wall-pinned outline): return the least-overlapping
    # iterate rather than whatever the last iteration happened to be.
    tight_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    tight_ov = float("inf")
    last_good = (cx.copy(), cy.copy(), w.copy(), h.copy())
    diverged = False

    for it in range(iters):
        frac = it / float(max(iters - 1, 1))
        lr_c = _cosine_lr(lr_center, it, iters)
        lr_s_now = _cosine_lr(lr_s, it, iters)
        lambda_b = _ramp(it, iters, LAMBDA_B_START, LAMBDA_B_END)

        grad_cx = np.zeros(n)
        grad_cy = np.zeros(n)

        # ---- HPWL, soft-abs smoothed (FIX, culprit 4) --------------------
        if b2b_i.size:
            dxv = cx[b2b_i] - cx[b2b_j]
            dyv = cy[b2b_i] - cy[b2b_j]
            gx = b2b_w * dxv / np.sqrt(dxv * dxv + eps_wl * eps_wl)
            gy = b2b_w * dyv / np.sqrt(dyv * dyv + eps_wl * eps_wl)
            grad_cx += _scatter(b2b_i, gx, n) - _scatter(b2b_j, gx, n)
            grad_cy += _scatter(b2b_i, gy, n) - _scatter(b2b_j, gy, n)
        if p2b_pin.size:
            dxv = cx[p2b_blk] - pins_s[p2b_pin, 0]
            dyv = cy[p2b_blk] - pins_s[p2b_pin, 1]
            gx = p2b_w * dxv / np.sqrt(dxv * dxv + eps_wl * eps_wl)
            gy = p2b_w * dyv / np.sqrt(dyv * dyv + eps_wl * eps_wl)
            grad_cx += _scatter(p2b_blk, gx, n)
            grad_cy += _scatter(p2b_blk, gy, n)

        # Floor the wirelength scale: with zero connectivity wl_l1 == 0, which
        # would zero lambda_o and leave the blocks piled on top of each other.
        wl_l1 = max(float(np.abs(grad_cx).sum() + np.abs(grad_cy).sum()), WL_FLOOR)
        wl_ref = wl_l1 / max(n, 1)

        # ---- overlap repulsion, raw (lambda_o = 1) -----------------------
        ov_frac = 0.0
        grad_s = np.zeros(n)
        if it % NBR_REFRESH == 0:
            nb_i, nb_j = rebuild_neighbors(cx, cy, w, h)
        if nb_i.size:
            dxv = cx[nb_i] - cx[nb_j]
            dyv = cy[nb_i] - cy[nb_j]
            ox = np.maximum(0.0, (w[nb_i] + w[nb_j]) / 2.0 - np.abs(dxv))
            oy = np.maximum(0.0, (h[nb_i] + h[nb_j]) / 2.0 - np.abs(dyv))
            act = (ox > 0.0) & (oy > 0.0)
            ov_frac = float(np.sum(ox * oy)) / a_tot
            if act.any():
                sx = np.where(dxv >= 0.0, 1.0, -1.0)
                sy = np.where(dyv >= 0.0, 1.0, -1.0)
                gcx = -oy * sx * act
                gcy = -ox * sy * act
                gov_x = _scatter(nb_i, gcx, n) - _scatter(nb_j, gcx, n)
                gov_y = _scatter(nb_i, gcy, n) - _scatter(nb_j, gcy, n)
                ov_l1 = float(np.abs(gov_x).sum() + np.abs(gov_y).sum())

                # ---- controller on measured overlap (FIX, culprit 1) -----
                if frac < WL_PHASE:
                    ov_target = ov_seed
                else:
                    t = min(1.0, (frac - WL_PHASE) / max(SPREAD_END - WL_PHASE, 1e-9))
                    ov_target = ov_seed ** (1.0 - t) * OV_TARGET ** t
                alpha *= a_up if ov_frac > ov_target else a_dn
                alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
                lambda_o = alpha * wl_l1 / max(ov_l1, 1e-18)

                grad_cx += lambda_o * gov_x
                grad_cy += lambda_o * gov_y

                # ---- soft-shape gradient WITH the dh/dw term (culprit 2) -
                if s_var_idx.size:
                    gwi = 0.5 * (oy * w[nb_i] - ox * h[nb_i]) * act
                    gwj = 0.5 * (oy * w[nb_j] - ox * h[nb_j]) * act
                    grad_s = lambda_o * (_scatter(nb_i, gwi, n) + _scatter(nb_j, gwj, n))
            else:
                alpha = max(alpha * a_dn, ALPHA_MIN)

        # ---- boundary-wall attraction (normalized to per-block WL force) --
        if n_wall and wl_ref > 0.0:
            kb = lambda_b * wl_ref
            if l_idx.size:
                grad_cx[l_idx] += kb * np.sign(cx[l_idx] - w[l_idx] / 2.0)
            if r_idx.size:
                grad_cx[r_idx] += -kb * np.sign(w_out - (cx[r_idx] + w[r_idx] / 2.0))
            if t_idx.size:
                grad_cy[t_idx] += -kb * np.sign(h_out - (cy[t_idx] + h[t_idx] / 2.0))
            if b_idx.size:
                grad_cy[b_idx] += kb * np.sign(cy[b_idx] - h[b_idx] / 2.0)

        # ---- cluster star attraction (vectorized, culprit 6) --------------
        if cl_idx.size and wl_ref > 0.0:
            kg = CLUSTER_BETA * wl_ref
            gx_c = np.bincount(cl_lab, weights=cx[cl_idx], minlength=n_cl) / cl_cnt
            gy_c = np.bincount(cl_lab, weights=cy[cl_idx], minlength=n_cl) / cl_cnt
            grad_cx[cl_idx] += kg * np.sign(cx[cl_idx] - gx_c[cl_lab])
            grad_cy[cl_idx] += kg * np.sign(cy[cl_idx] - gy_c[cl_lab])

        # ---- divergence guard: outlier gradient clipping ------------------
        grad_cx = _guard_grad(grad_cx)
        grad_cy = _guard_grad(grad_cy)

        # ---- Adam + area mobility + trust region -------------------------
        cx_new = _adam_step(cx, grad_cx, m_cx, v_cx, it + 1, lr_c)
        cy_new = _adam_step(cy, grad_cy, m_cy, v_cy, it + 1, lr_c)
        dxs = (cx_new - cx) * mobility
        dys = (cy_new - cy) * mobility
        step = np.sqrt(dxs * dxs + dys * dys)
        big = step > tr_step
        if big.any():
            shrink = np.where(big, tr_step / np.maximum(step, 1e-18), 1.0)
            dxs = dxs * shrink
            dys = dys * shrink
        cx += dxs * free_mult
        cy += dys * free_mult

        # ---- log-width update --------------------------------------------
        if s_var_idx.size:
            if ASPECT_REG > 0.0:
                s_scale = float(np.abs(grad_s[s_var_idx]).mean()) if s_var_idx.size else 0.0
                dev = np.clip((s - s0) / (0.5 * math.log(ASPECT_MAX)), -1.0, 1.0)
                grad_s = grad_s + ASPECT_REG * s_scale * dev
            grad_s = _guard_grad(grad_s)
            s_new = _adam_step(s, grad_s, m_s, v_s, it + 1, lr_s_now)
            s = np.where(s_var_mask, s_new, s)
            lo = 0.5 * (math.log(ASPECT_MIN) + np.log(np.maximum(area_s, 1e-18)))
            hi = 0.5 * (math.log(ASPECT_MAX) + np.log(np.maximum(area_s, 1e-18)))
            s = np.clip(s, lo, hi)
            ww_all, hh_all = dims_from_s(s)
            w = np.where(~frozen, ww_all, w)
            h = np.where(~frozen, hh_all, h)
            _mirror_groups()

        cx, cy = project(cx, cy)

        # ---- non-finite rollback (checked periodically; 8 numpy calls per
        # iteration is ~10% of the inner loop at n=120) ---------------------
        if it % ROLLBACK_EVERY == 0:
            if not (np.isfinite(cx).all() and np.isfinite(cy).all()
                    and np.isfinite(w).all() and np.isfinite(h).all()):
                cx, cy, w, h = (arr.copy() for arr in last_good)
                diverged = True
                break
            last_good = (cx.copy(), cy.copy(), w.copy(), h.copy())

        # ---- anytime best-iterate snapshot (FIX, culprit 5) ---------------
        if frac >= 0.30 and (it % snap_every == 0 or it == iters - 1):
            ov_now = overlap_of(cx, cy, w, h) / a_tot
            if ov_now <= OV_SNAP_MAX:
                wl_now = wl_of(cx, cy)
                if wl_now < best_wl:
                    best_wl = wl_now
                    best_state = (cx.copy(), cy.copy(), w.copy(), h.copy())
            elif best_state is None and ov_now < tight_ov:
                tight_ov = ov_now
                tight_state = (cx.copy(), cy.copy(), w.copy(), h.copy())

        if (it + 1) % 50 == 0 or it == iters - 1:
            obj_trace.append(objective(cx, cy, w, h))

    if best_state is not None:
        cx, cy, w, h = best_state
    elif tight_state is not None:
        cx, cy, w, h = tight_state

    obj_final = objective(cx, cy, w, h)

    # ---- back to physical units ------------------------------------------
    rects = np.stack([(cx - w / 2.0) * scale, (cy - h / 2.0) * scale,
                      w * scale, h * scale], axis=1)
    rects[preplaced] = tp[preplaced]

    cxp = rects[:, 0] + rects[:, 2] / 2.0
    cyp = rects[:, 1] + rects[:, 3] / 2.0
    wp = rects[:, 2]
    hp = rects[:, 3]

    hpwl = 0.0
    if b2b_i.size:
        hpwl += float(np.sum(b2b_w_raw * (np.abs(cxp[b2b_i] - cxp[b2b_j])
                                          + np.abs(cyp[b2b_i] - cyp[b2b_j]))))
    if p2b_pin.size:
        hpwl += float(np.sum(p2b_w_raw * (np.abs(cxp[p2b_blk] - pins[p2b_pin, 0])
                                          + np.abs(cyp[p2b_blk] - pins[p2b_pin, 1]))))

    if iu.size:
        ox = np.maximum(0.0, (wp[iu] + wp[ju]) / 2.0 - np.abs(cxp[iu] - cxp[ju]))
        oy = np.maximum(0.0, (hp[iu] + hp[ju]) / 2.0 - np.abs(cyp[iu] - cyp[ju]))
        overlap_area = float(np.sum(ox * oy))
    else:
        overlap_area = 0.0
    overlap_frac = overlap_area / max(a_tot_phys, 1e-9)

    valid = ((not np.isnan(rects).any()) and math.isfinite(obj_final)
             and (obj_final < obj0) and not diverged
             and overlap_frac <= OV_REJECT)

    diag = {
        "hpwl": hpwl,
        "overlap_frac": overlap_frac,
        "obj_trace": obj_trace,
        "obj0": obj0,
        "obj_final": obj_final,
        "alpha_final": alpha,
        "ov_seed": ov_seed,
        "ws": ws,
        "gamma": gamma,
        "w_out": w_out * scale,
        "h_out": h_out * scale,
        "valid": valid,
    }
    return {"rects": rects, "diag": diag}


def generate_candidates(area_targets: Any, b2b: Any, p2b: Any, pins_pos: Any,
                         constraints: Any, target_positions: Any,
                         n_configs: int = 6, iters: int = 90,
                         seed: int = 0) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Generate ``n_configs`` analytical global-placement candidates.

    Returns ``(candidates, diagnostics)``: ``candidates`` is a list of
    ``[n, 4]`` (x, y, w, h) arrays (one per successful config; failed
    configs -- objective did not decrease, or NaNs -- are excluded, never
    raised). ``diagnostics`` is a same-length list of per-config dicts with
    keys ``hpwl``, ``overlap_frac``, ``obj_trace`` (every-50-iter objective
    values), ``ws``, ``gamma``, ``config_index``.
    """
    area = _to_numpy(area_targets).reshape(-1)
    cons = _to_numpy(constraints)
    tp = _to_numpy(target_positions)
    b2b_np = _to_numpy(b2b) if b2b is not None else np.zeros((0, 3))
    p2b_np = _to_numpy(p2b) if p2b is not None else np.zeros((0, 3))
    pins_np = _to_numpy(pins_pos) if pins_pos is not None else np.zeros((0, 2))

    n = area.shape[0]
    if cons is None or cons.shape[1] < 5:
        padded = np.zeros((n, 5))
        if cons is not None:
            padded[:, : cons.shape[1]] = cons
        cons = padded
    if tp is None or tp.shape != (n, 4):
        tp = np.full((n, 4), -1.0)

    if b2b_np.size:
        valid_b2b = b2b_np[b2b_np[:, 0] != -1]
        mean_b2b_weight = float(valid_b2b[:, 2].mean()) if valid_b2b.size else 1.0
    else:
        mean_b2b_weight = 1.0

    configs = _configs(n_configs)
    candidates: List[np.ndarray] = []
    diagnostics: List[Dict[str, Any]] = []
    salvage: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
    for cfg_idx, (ws, gamma) in enumerate(configs):
        cfg_seed = seed + cfg_idx * 7919
        try:
            result = _run_config(area, cons, tp, b2b_np, p2b_np, pins_np,
                                  ws, gamma, iters, cfg_seed, mean_b2b_weight)
        except Exception:
            continue
        diag = result["diag"]
        rects = result["rects"]
        if np.isnan(rects).any():
            continue
        if not diag["valid"]:
            # keep as a salvage candidate so we never return an empty list
            salvage.append((diag["overlap_frac"], rects, {
                "hpwl": diag["hpwl"],
                "overlap_frac": diag["overlap_frac"],
                "obj_trace": diag["obj_trace"],
                "alpha_final": diag["alpha_final"],
                "ws": ws, "gamma": gamma, "config_index": cfg_idx,
                "salvaged": True,
            }))
            continue
        candidates.append(rects)
        diagnostics.append({
            "hpwl": diag["hpwl"],
            "overlap_frac": diag["overlap_frac"],
            "obj_trace": diag["obj_trace"],
            "alpha_final": diag["alpha_final"],
            "ws": ws,
            "gamma": gamma,
            "config_index": cfg_idx,
        })
    if not candidates and salvage:
        salvage.sort(key=lambda t: t[0])
        candidates.append(salvage[0][1])
        diagnostics.append(salvage[0][2])
    return candidates, diagnostics
