#!/usr/bin/env python3
"""Column-slicing legalizer + simulated-annealing HPWL optimizer.

Layout model ("variant A" column slicing):
  * Pick a frame height H (from total area / target utilization and the
    pin-bounding-box aspect ratio).
  * Blocks are grouped into *units* (cluster groups and soft-MIB groups,
    merged via union-find).  Each unit is a vertical stack of slices.
  * Columns are laid out left to right.  Every soft slice in a column has
    width == column width and height == area / width, so soft-block areas
    are exact and the column fills its height with zero dead space.
    The column width is solved so the content height equals H:
        w_c = soft_area / (H - rigid_height - obstacle_height)
  * Preplaced blocks stay at their exact target position and act as
    obstacles; stacking skips their y-intervals.  A cluster that contains
    a preplaced member is placed flush against that block so the group
    stays connected.
  * Fixed-shape blocks keep their exact (w, h); the column is widened to
    at least the widest rigid member.
  * Boundary handling: left/right-tagged units are pinned to the first /
    last column (every slice there touches x_min / x_max), bottom-tagged
    units stack first (y = 0), top-tagged units are lifted flush to the
    global top edge.

Hard guarantees by construction: no overlaps, exact soft areas, exact
fixed/preplaced dimensions and positions.

A time-budgeted simulated annealing then permutes units across columns to
minimize a proxy of the contest cost (HPWL + bbox area + soft violations).
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import os as _os
_RDEBUG = bool(_os.environ.get("REFINER_DEBUG"))

Rect = Tuple[float, float, float, float]

EPS = 1e-9
TOUCH_TOL = 1e-7
UTIL_TARGET_FRAME = 0.96   # utilization used to size the frame height
UTIL_TARGET_REF = 0.97     # utilization used for the area cost reference
MIB_AREA_GUARD = 0.0095    # stay under the 1% hard area tolerance

_TRUE = ("1", "true", "True", "on", "ON")


def _flag_on(name: str) -> bool:
    """Shared `PARTNER_*` boolean env reader (same spelling set as the
    existing inline checks)."""
    return _os.environ.get(name, "0") in _TRUE


def early_exit_on() -> bool:
    """PARTNER_EARLY_EXIT=1 (default off): the fork is deadline-bounded, so
    a converged stage must RETURN its unspent share to the case instead of
    handing it to the next opportunistic consumer.  See
    docs/design/2026-08-04-early-exit-true-time-reduction.md."""
    return _flag_on("PARTNER_EARLY_EXIT")


def fast_setup_on() -> bool:
    """PARTNER_FAST_SETUP=1 (default off): make the per-case FIXED costs
    budget-aware.

    At the goal operating point (b(n) = 5e-5 * exp(n/12) clamped to
    [0.05, 0.75]) a small case asks for 0.05 s and measures 0.12-0.17 s of
    wall clock.  The gap is not payload marshalling and not process dispatch
    -- both are sub-millisecond, see
    docs/design/2026-08-05-fast-setup-floor.md -- it is the flat 0.1 s
    minimum anneal span in `finish`, which every one of the ~24 pool workers
    pays no matter how short its deadline is.  The flag clamps that floor to
    the span the chain was actually given, so a case honours its own budget.
    Off: bit-identical (the clamp is a no-op whenever the planned span
    already exceeds 0.1 s, i.e. every pre-goal-tier budget)."""
    return _flag_on("PARTNER_FAST_SETUP")


# --- PARTNER_FRAME_WPIN -------------------------------------------------
# `_choose_frame` pins the frame HEIGHT to a top-tagged preplaced block but
# has no width counterpart, so on instances whose preplaced blocks carry an
# L/R tag the frame width is whatever the aspect heuristic guessed and the
# tag is unsatisfiable by construction.  Rather than hard-pinning W (which
# would override the height pin and lose on instances where the guess was
# right), the derived frame enters the restart pool as an EXTRA ARM and
# competes under the same true cost as every other restart.
FRAME_WPIN_UTIL = 0.96    # frame utilisation used to derive H from W*


def frame_wpin_on() -> bool:
    """PARTNER_FRAME_WPIN=1 (default off): add W*-derived frame arms to the
    restart portfolio on instances with an L/R-tagged preplaced block.

    Gate is a reusable instance statistic (does a preplaced block carry a
    right-wall tag?), never a case id.  Off: the portfolio, the payload
    arity and `_choose_frame` are all byte-identical."""
    return _flag_on("PARTNER_FRAME_WPIN")


def col_narrow_on() -> bool:
    """PARTNER_COL_NARROW=1 (default off): let the column-width solve shrink,
    not just grow.

    `_layout` solves each column's width from its soft area and then retries
    WIDER when the stack overflows the frame -- but never retries NARROWER
    when it underflows.  An underflowing column is dead area inside the
    frame (measured: 96.8% of dead area sits inside columns), and dead area
    inflates the bbox, which is the area_gap term of the score.  The narrow
    loop is the exact mirror of the widen loop, with the same overflow guard
    as its acceptance test.  Off: the loop never runs."""
    return _flag_on("PARTNER_COL_NARROW")


def early_exit_window(specific: str, default: float = 0.25) -> float:
    """Stall-window fraction with the documented precedence:

      1. the phase's own `PARTNER_*_STALL_WINDOW` when explicitly set,
      2. `PARTNER_EARLY_EXIT_WINDOW` when PARTNER_EARLY_EXIT is on,
      3. `default` (the historical stall-stop value).

    The window is always a FRACTION of the phase's own span, so it scales
    itself down with `PARTNER_BUDGET_MAX` (a 0.2 s/case regime gets a 0.2 s
    -scaled window without any new tuning)."""
    raw = _os.environ.get(specific)
    if raw is None and early_exit_on():
        raw = _os.environ.get("PARTNER_EARLY_EXIT_WINDOW")
        default = 0.15
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if 0.0 < val < 1.0 else default


def early_exit_min_window() -> float:
    """Absolute floor (seconds) under the fractional stall window.  At very
    small budgets a fraction of a ~0.1 s span would stop a phase before its
    first productive round; the floor costs nothing at the 3.5 s tier."""
    try:
        val = float(_os.environ.get("PARTNER_EARLY_EXIT_MIN_WINDOW", "0.05"))
    except ValueError:
        return 0.05
    return val if val >= 0.0 else 0.05


# =============================================================================
# PARTNER_GPU_ARM -- turn the idle accelerator into a second candidate wave
# =============================================================================
# The fork is CPU-bound after the first sampling wave: the GPU draws the
# direct/flow batch in ~0.1-0.3 s at the head of a case and then idles for the
# entire SA/refine span (more so on the official A100 than on the local L4).
# PARTNER_GPU_ARM=1 (default off) spends that idle window:
#
#   phase A   all column restarts + the existing NREF refine slots, exactly as
#             today.  While they run, the parent fires a SECOND sampling wave
#             with a different generator seed -- the latency is masked by the
#             workers, so it does not delay the start of the case (the 0723
#             "sample at the head" variant was killed precisely because it did).
#   phase B   the reclaimed tail of the budget: the fresh GPU candidates are
#             refined by the (RK-accelerated) pool alongside the phase-A winner
#             variants, and everything competes under the same true cost.
#
# The arm does NOT take pool slots away from phase A -- it uses a time-axis
# carve, which is what separates it from the PARTNER_NREF increment that was
# convicted at the goal tier for eating column-restart breadth.
#
# The honest risk is the carve itself (phase A loses `1 - frac` of its span,
# the 0723 `dmoff` carve paid -0.010 for nothing).  The arm therefore refuses
# the carve unless the MEASURED sampler latency says both waves fit and phase A
# still retains PARTNER_GPU_ARM_MIN_A of the budget as real SA time.
#
# Env (all default-off / default-neutral):
#   PARTNER_GPU_ARM         1 -> enable the arm (implies an auto phase-B carve)
#   PARTNER_GPU_ARM_TARGET  target phase-B slice in seconds (default 0.80)
#   PARTNER_GPU_ARM_MIN_A   min share of the budget phase A must keep (0.35)
#   PARTNER_GPU_ARM_K       cap on second-wave candidates (default 12)
#   PARTNER_GPU_ARM_TS0     prior sampler latency before the first measurement
#   PARTNER_GPU_ARM_SEED    generator-seed offset for the second wave (8117)
#   PARTNER_GPU_ARM_DEBUG   per-case stderr anatomy line

# number of winner-variant payloads the phase-B round always dispatches
_PHASE_B_VARIANTS = 8

# measured wave-1 sampler latency, EMA per block-count decade.  A reusable
# instance statistic (block count), never a case id -- see CLAUDE.md.
_GPU_ARM_TS: Dict[int, float] = {}


def _env_num(name: str, default: float) -> float:
    """`PARTNER_*` float env reader that never raises."""
    try:
        return float(_os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def gpu_arm_on() -> bool:
    """PARTNER_GPU_ARM=1 (default off): consume the idle accelerator with a
    second sampling wave feeding an auto-sized phase-B round."""
    return _flag_on("PARTNER_GPU_ARM")


def phase_b_min_budget_default(gpu_arm: bool) -> float:
    """Minimum remaining budget for the phase-B carve.

    The historical floor is 8 s, which no promoted tier ever reaches (3.5 s
    and 0.44 s), so `PARTNER_PHASE_B` has been unreachable in practice.  The
    floor exists because a phase-B slice must hold at least one refine rung;
    `PARTNER_REFINE_KERNEL=numba` cut a rung from 0.36-0.8 s to ~0.035 s
    (12-23x), so under the kernel a 0.4 s case can afford the round.  Only the
    arm lowers it -- with the arm off the legacy lever keeps its 8 s floor.
    """
    if not gpu_arm:
        return 8.0
    if _os.environ.get("PARTNER_REFINE_KERNEL", "") == "numba":
        return 0.35
    return 1.5


def gpu_arm_sample_estimate(n: int) -> float:
    """Expected wave-1 sampler latency (s) for an instance of `n` blocks."""
    return float(_GPU_ARM_TS.get(int(n) // 10,
                                 _env_num("PARTNER_GPU_ARM_TS0", 0.25)))


def gpu_arm_record_sample(n: int, dt: float) -> None:
    """Fold a measured wave-1 latency into the per-decade EMA."""
    if not (dt >= 0.0):
        return
    key = int(n) // 10
    prev = _GPU_ARM_TS.get(key)
    _GPU_ARM_TS[key] = dt if prev is None else 0.5 * prev + 0.5 * dt


def gpu_arm_slice(rem: float) -> float:
    """Phase-B slice length (s) for a remaining budget of `rem`.

        slice = clamp(0.20*rem, 0.35*rem, TARGET)

    The absolute target (0.80 s) is what the promoted 3.5 s tier can spend
    without gutting column depth (~20 RK rungs); the 35 % ceiling is what
    keeps the goal tier's 0.44 s span usable (~0.15 s, ~4 rungs); the 20 %
    floor stops the slice from collapsing on long budgets.
    """
    rem = max(0.0, float(rem))
    return min(max(_env_num("PARTNER_GPU_ARM_TARGET", 0.80), 0.20 * rem),
               0.35 * rem)


def gpu_arm_wave2_k(slice_s: float, pool_size: int) -> int:
    """How many second-wave candidates to draw.

    Phase B is *under-subscribed* today (it dispatches 8 winner variants onto
    a pool of up to 24), so the CPU cost of the extra candidates is zero --
    the binding limits are the free pool slots, the batch latency that has to
    hide inside phase A, and how much refine a short slice can actually do.
    One candidate per ~60 ms of slice, plus a floor of 2.
    """
    cap = int(_env_num("PARTNER_GPU_ARM_K", 12))
    return max(0, min(int(pool_size), cap,
                      2 + int(max(0.0, slice_s) / 0.06)))


def gpu_arm_phase_b_split(slice_s: float, pool_size: int) -> Tuple[int, int]:
    """Split the phase-B pool between (new GPU candidates, winner variants).

    On the production pool (24-46 workers) the variants keep all 8 slots and
    the GPU wave takes idle ones -- nothing is displaced.  Only when the pool
    cannot hold both does the split bite, and then the incumbent's basin hops
    keep at least half of it: exploitation of a known-good layout is the
    round's proven job, the new supply is the experiment.
    """
    pool_size = max(0, int(pool_size))
    k2 = gpu_arm_wave2_k(slice_s, pool_size)
    n_var = _PHASE_B_VARIANTS
    if k2 + n_var > pool_size:
        k2 = min(k2, max(0, pool_size // 2))
        n_var = min(n_var, max(1, pool_size - k2))
    return k2, n_var


def gpu_arm_sampler_takes_seed(fn) -> bool:
    """True when `fn` accepts the `gen_seed` keyword.

    The legalizer must never call a legacy one-argument sampler twice: its
    generator seed is pinned, so wave 2 would re-draw wave 1 verbatim and burn
    the carve for nothing.
    """
    if fn is None:
        return False
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "gen_seed" in params or any(
        p.kind == p.VAR_KEYWORD for p in params.values())


def gpu_arm_second_wave(sample_fn, k: int, gen_seed: int, t_wave1: float,
                        deadline_a: float) -> List[np.ndarray]:
    """Draw the masked second wave, or return `[]` for any reason at all.

    Containment is the whole contract here: a GPU failure, an OOM, a legacy
    sampler or a wave that would overrun phase A must all degrade to the plain
    winner-variant round rather than harm the case.
    """
    if k <= 0 or not gpu_arm_sampler_takes_seed(sample_fn):
        return []
    # wave 2 costs about what wave 1 cost (the batch is capped and the step
    # count dominates); refuse it unless it provably lands inside phase A
    if time.time() + 1.20 * max(0.0, t_wave1) + 0.03 >= deadline_a:
        return []
    try:
        out = list(sample_fn(k, gen_seed=int(gen_seed)))[:k]
    except Exception:
        return []
    return [np.asarray(P, dtype=np.float64) for P in out]


# =============================================================================
# Constraint parsing helpers (API kept for contest_optimizer.py)
# =============================================================================
def _pin_centroids_np(p2b: torch.Tensor, pins: torch.Tensor,
                      n: int, n_pins: int):
    """Vectorised twin of the `for edge in p2b` accumulation in
    `_heuristic_init` (PARTNER_FAST_SETUP).

    Bit-exact by construction: the same float32 -> float64 widening, the same
    `(w * px)` product order, and the same accumulation order -- `np.add.at`
    is unbuffered, so repeated target indices are summed in index order, which
    is the loop's edge order.  Returns None on any unexpected layout so the
    caller keeps the reference loop."""
    e = p2b.detach().cpu().numpy()
    pn = pins.detach().cpu().numpy()
    if e.ndim != 2 or e.shape[1] < 3 or pn.ndim != 2 or pn.shape[1] < 2:
        return None
    sx = [0.0] * n
    sy = [0.0] * n
    wsum = [0.0] * n
    if e.shape[0] == 0:
        return sx, sy, wsum
    pi = e[:, 0].astype(np.int64)
    bi = e[:, 1].astype(np.int64)
    keep = (e[:, 0] != -1) & (bi >= 0) & (bi < n) & (pi >= 0) & (pi < n_pins)
    pi = pi[keep]
    bi = bi[keep]
    w = e[:, 2].astype(np.float64)[keep]
    px = pn[pi, 0].astype(np.float64)
    py = pn[pi, 1].astype(np.float64)
    ok = (px != -1.0) & (py != -1.0)
    bi = bi[ok]
    px = px[ok]
    py = py[ok]
    w = np.maximum(w[ok], 0.0)
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    aw = np.zeros(n, dtype=np.float64)
    np.add.at(ax, bi, w * px)
    np.add.at(ay, bi, w * py)
    np.add.at(aw, bi, w)
    return ax.tolist(), ay.tolist(), aw.tolist()


def _b2b_smooth_np(b2b: torch.Tensor, cx, cy, n: int):
    """Vectorised twin of the `for edge in b2b` smoothing pass in
    `_heuristic_init` (PARTNER_FAST_SETUP).

    The loop writes nx[i] before nx[j] for each edge, so the flattened index
    stream is INTERLEAVED (i0, j0, i1, j1, ...) rather than concatenated --
    that is what keeps the unbuffered `np.add.at` accumulation order equal to
    the loop's.  Returns None on any unexpected layout."""
    e = b2b.detach().cpu().numpy()
    if e.ndim != 2 or e.shape[1] < 3:
        return None
    nx = np.array(cx, dtype=np.float64)
    ny = np.array(cy, dtype=np.float64)
    deg = np.zeros(n, dtype=np.float64)
    if e.shape[0] == 0:
        return nx.tolist(), ny.tolist(), deg.tolist()
    ii = e[:, 0].astype(np.int64)
    jj = e[:, 1].astype(np.int64)
    keep = (e[:, 0] != -1) & (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
    ii = ii[keep]
    jj = jj[keep]
    hw = 0.25 * np.maximum(e[:, 2].astype(np.float64)[keep], 0.0)
    m = ii.shape[0]
    # `acx`/`acy` are the READ side: the loop always reads the original
    # centroid, never the partially accumulated `nx`/`ny`.
    acx = np.array(cx, dtype=np.float64)
    acy = np.array(cy, dtype=np.float64)
    idx = np.empty(2 * m, dtype=np.int64)
    idx[0::2] = ii
    idx[1::2] = jj
    vx = np.empty(2 * m, dtype=np.float64)
    vx[0::2] = hw * acx[jj]
    vx[1::2] = hw * acx[ii]
    vy = np.empty(2 * m, dtype=np.float64)
    vy[0::2] = hw * acy[jj]
    vy[1::2] = hw * acy[ii]
    vd = np.empty(2 * m, dtype=np.float64)
    vd[0::2] = hw
    vd[1::2] = hw
    np.add.at(nx, idx, vx)
    np.add.at(ny, idx, vy)
    np.add.at(deg, idx, vd)
    return nx.tolist(), ny.tolist(), deg.tolist()


def _col(constraints: Optional[torch.Tensor], n: int, idx: int) -> List[float]:
    if constraints is None or constraints.dim() < 2 or constraints.shape[1] <= idx:
        return [0.0] * n
    return [float(v) for v in constraints[:n, idx]]


def _parse_constraints(constraints: Optional[torch.Tensor], n: int):
    fixed = [v != 0 for v in _col(constraints, n, 0)]
    preplaced = [v != 0 for v in _col(constraints, n, 1)]
    mib = [int(round(v)) for v in _col(constraints, n, 2)]
    cluster = [int(round(v)) for v in _col(constraints, n, 3)]
    boundary = [int(round(v)) for v in _col(constraints, n, 4)]
    return fixed, preplaced, mib, cluster, boundary


def _target(target_positions: Optional[torch.Tensor], i: int) -> Tuple[float, float, float, float]:
    if target_positions is None:
        return (-1.0, -1.0, -1.0, -1.0)
    tx, ty, tw, th = [float(v) for v in target_positions[i]]
    return tx, ty, tw, th


def _fastsa_temp(frac: float, t0: float, t1: float,
                 k: float, c: float, steps: float) -> float:
    """Fast-SA three-stage temperature law (Chen-Chang ISPD'05), mapped onto
    our wall-clock fraction axis and ANCHORED to the caller's geometric
    endpoints t0 (hot) / t1 (cold) so the acceptance regime stays in the range
    our Metropolis criterion + cost scale are already tuned for. Shape (not
    absolute scale) is what the port reproduces.

    Ported verbatim from src/floorset_arch/legalizer/column_slicing.py
    (`_fastsa_temp`). The paper indexes temperature steps by an integer n:
        T_1 = Davg / ln P                       (stage 1, hot random walk)
        T_n = T_1*|Dcost| / (n*c),  2 <= n <= k (stage 2, fast quench ~greedy)
        T_n = T_1*|Dcost| / n,      n > k       (stage 3, reheat + 1/n cooling)
    We (a) map frac in [0,1] linearly to a continuous index n in [1, steps],
    (b) replace the online Davg/|Dcost| scale by an anchor A = t1*steps so the
    stage-3 kernel A/n lands exactly on t1 at frac=1, and (c) hold stage 1 at
    the hot anchor t0. This preserves the three invariants that DEFINE the
    schedule -- the xc deep quench of stage 2, the ~c-fold reheat jump at n=k,
    and the 1/n stage-3 tail -- while leaving t0/t1 (hence the accept curve)
    untouched. With steps=41, k=7 the reheat lands at frac~0.15, so ~15% of
    the run is hot exploration + quench and ~85% is the low-T 1/n hill-climb,
    the paper's iteration split. Pure function; called once per outer loop,
    exactly like the geometric law it replaces."""
    n = 1.0 + (steps - 1.0) * frac
    A = t1 * steps
    if n < 2.0:
        return t0                    # stage 1: high-T random exploration
    if n <= k:
        return A / (n * c)           # stage 2: fast quench to pseudo-greedy
    return A / n                     # stage 3: reheat, then 1/n cooling to t1


# =============================================================================
# Rectangle reconstruction from diffusion z output (kept for contest_optimizer.py)
# =============================================================================
def rectangles_from_z(
    z: torch.Tensor,
    area_targets: torch.Tensor,
    constraints: Optional[torch.Tensor] = None,
    target_positions: Optional[torch.Tensor] = None,
    z_repr: str = "xylogwh",
) -> List[Rect]:
    n = int((area_targets != -1).sum().item())
    scale = math.sqrt(max(1.0, sum(float(a) for a in area_targets[:n] if float(a) > 0)))
    rects: List[Rect] = []
    for i in range(n):
        area = max(float(area_targets[i]), 1e-6)
        x = float(z[i, 0]) * scale
        y = float(z[i, 1]) * scale
        if z.shape[-1] >= 4:
            if z_repr == "xyaspect":
                log_aspect = max(-3.0, min(3.0, float(z[i, 2])))
            else:
                log_aspect = max(-3.0, min(3.0, float(z[i, 2] - z[i, 3])))
        else:
            log_aspect = max(-3.0, min(3.0, float(z[i, 2])))
        aspect = math.exp(log_aspect)
        w = math.sqrt(area * aspect)
        h = math.sqrt(area / aspect)
        rects.append((x, y, max(w, 1e-6), max(h, 1e-6)))
    return rects


# =============================================================================
# Units
# =============================================================================
class _Unit:
    __slots__ = ("uid", "subgroups", "blocks", "soft_area", "rigid_h", "max_rigid_w",
                 "area_total", "force", "anchors", "seed_x", "seed_y", "hasB", "hasT",
                 "bands", "eff_soft", "eff_rigid_h", "banded", "hcache", "dyn", "pairable")

    def __init__(self, uid: int):
        self.uid = uid
        self.subgroups: List[List[int]] = []
        self.blocks: List[int] = []
        self.soft_area = 0.0
        self.rigid_h = 0.0
        self.max_rigid_w = 0.0
        self.area_total = 0.0
        self.force: Optional[str] = None   # 'L' or 'R'
        self.anchors: List[Rect] = []
        self.seed_x = 0.0
        self.seed_y = 0.0
        self.hasB = False
        self.hasT = False

    def flatten(self):
        self.blocks = [b for sg in self.subgroups for b in sg]


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# =============================================================================
# Main optimizer
# =============================================================================
class _ColumnOptimizer:
    def __init__(
        self,
        rects: Sequence[Rect],
        area_targets: torch.Tensor,
        constraints: Optional[torch.Tensor],
        target_positions: Optional[torch.Tensor],
        b2b: Optional[torch.Tensor],
        p2b: Optional[torch.Tensor],
        pins: Optional[torch.Tensor],
        deadline: Optional[float],
        seed: int = 0,
        v_weight: float = 1.0,
        h_scale: float = 1.0,
        pinned: Optional[Dict[int, Rect]] = None,
        w_star: Optional[float] = None,
    ):
        # PARTNER_FRAME_WPIN portfolio arm: the frame WIDTH implied by the
        # L/R-tagged preplaced blocks.  None (the default, and the only
        # value any pre-flag caller passes) leaves `_choose_frame` untouched.
        self.w_star = w_star
        self.n = len(rects)
        self.rects = rects
        self.deadline = deadline if deadline is not None else time.time() + 3.0
        self.rng = random.Random(seed)
        self.v_weight = v_weight  # >1: search prioritizes killing violations
        self.h_scale = h_scale    # portfolio diversity: stretch/squash frame H

        n = self.n
        self.areas = [max(float(area_targets[i]), 1e-9) for i in range(n)]
        self.fixed, self.preplaced, self.mib, self.cluster, self.boundary = \
            _parse_constraints(constraints, n)
        self.tpos = target_positions

        # kind: 0 = soft, 1 = rigid (exact w,h), 2 = locked (exact x,y,w,h)
        self.kind = [0] * n
        self.rw = [0.0] * n
        self.rh = [0.0] * n
        self.lx = [0.0] * n
        self.ly = [0.0] * n
        self._pinned = pinned or {}
        self._resolve_shapes()

        self.locked_rects: List[Rect] = [
            (self.lx[i], self.ly[i], self.rw[i], self.rh[i])
            for i in range(n) if self.kind[i] == 2
        ]

        self._build_hpwl_arrays(b2b, p2b, pins)
        self._build_soft_norm()
        self._choose_frame(pins)
        self._build_units()

        # -- column-cache delta-evaluation state (PARTNER_COL_CACHE=1) --------
        # Skips re-running _stack_column for the left prefix of columns that are
        # bit-identical to the previous _layout call. See _layout_delta for the
        # correctness argument. Default OFF: bare defaults are unchanged.
        self._dc_enabled = _os.environ.get("PARTNER_COL_CACHE", "0") in (
            "1", "true", "True", "on", "ON")
        self._dc_prev_cols: Optional[List[List[int]]] = None
        self._dc_prev_sig: Optional[List[tuple]] = None
        self._dc_cache: List[Optional[tuple]] = []
        # units whose internal order a _random_move can permute (subgroup
        # reorder). Only these need a per-column signature check; everything
        # else is captured by column membership.
        self._dc_mutable = frozenset(
            k for k, u in enumerate(self.units)
            if len(u.subgroups) >= 2 or any(len(sg) >= 2 for sg in u.subgroups))
        self._dc_hits = 0
        self._dc_recompute = 0

        # PARTNER_COL_NARROW (default off): read once per optimizer, never in
        # the `_layout` inner loop.
        self._col_narrow = col_narrow_on()

        # -- Fast-SA cooling schedule (PARTNER_FASTSA_TEMP=1, default off) ---
        # Cherry-pick #1 from src/floorset_arch/legalizer/column_slicing.py.
        # "geometric" (default) keeps the historical wall-clock geometric law
        # in `_anneal` bit-identical -- the schedule branch is a single string
        # compare per outer loop and never touches the rng stream. "fastsa"
        # swaps ONLY the T(frac) curve for the Chen-Chang three-stage law
        # (`_fastsa_temp`); move set, Metropolis accept, recalibration, the
        # partner-only late-phase violation-weight annealing and finish/polish
        # are all untouched. k / c / steps mirror the paper's (k=7, c=100)
        # plus the frac->index span; malformed values fall back silently.
        self._sa_schedule = "fastsa" if _os.environ.get(
            "PARTNER_FASTSA_TEMP", "0") in ("1", "true", "True", "on", "ON") \
            else "geometric"
        try:
            self._fastsa_k = float(_os.environ.get("PARTNER_FASTSA_K", "7"))
        except ValueError:
            self._fastsa_k = 7.0
        try:
            self._fastsa_c = float(_os.environ.get("PARTNER_FASTSA_C", "100"))
        except ValueError:
            self._fastsa_c = 100.0
        try:
            self._fastsa_steps = float(
                _os.environ.get("PARTNER_FASTSA_STEPS", "41"))
        except ValueError:
            self._fastsa_steps = 41.0
        # guard degenerate params so the law stays well-formed (steps > k >= 2)
        if self._fastsa_k < 2.0:
            self._fastsa_k = 2.0
        if self._fastsa_steps <= self._fastsa_k:
            self._fastsa_steps = self._fastsa_k + 1.0
        if self._fastsa_c <= 0.0:
            self._fastsa_c = 100.0

        # -- SA stall early stop (PARTNER_SA_STALL_STOP=1, default off) ------
        # Cherry-pick #2 from column_slicing.py ("E2-adaptive early stop").
        # An `_anneal` chain breaks once best_cost has not improved by >=
        # `stall_eps` (relative) within one stall window.
        #
        # DEVIATION FROM SRC (deliberate): src derives the window as a
        # fraction of the case BASE budget and threads it from the main
        # process into the forked workers, because its adaptive mode inflates
        # the deadline to base*CAP and the early stop is what reins the
        # runtime back under the ceiling. This fork is deadline-bounded (the
        # official runtime IS the budget), there is no inflated ceiling, and
        # chain spans vary by an order of magnitude between `probe` and
        # `finish`. So the window here is a fraction of the CURRENT chain's
        # own span -- self-scaling, no payload plumbing, and at the default
        # 0.25 it lands in the same band as src's effective window/span ratio
        # (~0.12-0.23). Consequently src's `finish` wrap-up cap (which exists
        # only to stop the polish from eating the ceiling) is NOT ported:
        # here the reclaimed time flows naturally to the next restart, the
        # greedy polish and the refine stage.
        #
        # -- true time reduction (PARTNER_EARLY_EXIT=1, default off) ---------
        # `PARTNER_SA_STALL_STOP` was measured quality-negative because the
        # time it reclaimed had nowhere to go (finish() simply handed it to
        # the greedy polish and the stage-2 refiner, and the case still ran
        # to `worker_deadline`).  EARLY_EXIT keeps the same detector but
        # changes the SEMANTICS: `finish` claws the unspent share back out of
        # its own deadline, so a converged worker RETURNS early and the case
        # wall clock drops below the budget.  EARLY_EXIT therefore implies
        # the SA stall stop; an explicit PARTNER_SA_STALL_STOP=0 cannot turn
        # it back off (the stop is what makes the exit possible), but the
        # window/eps overrides still apply.
        self.stall_eps = 0.003
        self._stall_frac = 0.0
        self._anneal_stalled = False
        self._early_exit = early_exit_on()
        # unspent tail of the probe chains, reclaimed by `run` /
        # `legalize_rectangles` so it cannot inflate `finish` (EARLY_EXIT only)
        self._probe_unspent = 0.0
        if self._early_exit or _flag_on("PARTNER_SA_STALL_STOP"):
            self._stall_frac = early_exit_window("PARTNER_SA_STALL_WINDOW",
                                                 0.25)
            try:
                self.stall_eps = float(_os.environ.get(
                    "PARTNER_SA_STALL_EPS", "0.003"))
            except ValueError:
                self.stall_eps = 0.003

        # -- flat-array numba layout kernel (PARTNER_SA_KERNEL=numba) --------
        # Bit-exact transcription of `_layout_full` over structure-of-arrays
        # buffers; see partner/sa_numeric_kernel.py. Default off.
        self._sa_kernel = None
        if _os.environ.get("PARTNER_SA_KERNEL", "") == "numba":
            try:
                from sa_numeric_kernel import try_attach
                self._sa_kernel = try_attach(self)
            except Exception:
                self._sa_kernel = None
            if self._sa_kernel is not None:
                self._dc_enabled = False   # the kernel supersedes the colcache

    # ------------------------------------------------------------------
    def _resolve_shapes(self):
        n = self.n
        for i in range(n):
            tx, ty, tw, th = _target(self.tpos, i)
            if self.preplaced[i] and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                self.kind[i] = 2
                self.lx[i], self.ly[i], self.rw[i], self.rh[i] = tx, ty, tw, th
            elif (self.fixed[i] or self.preplaced[i]) and tw > 0 and th > 0:
                self.kind[i] = 1
                self.rw[i], self.rh[i] = tw, th

        # MIB groups: if a member has a hard shape, all soft members must copy
        # it exactly to reach zero MIB violations (areas in a group are equal).
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            if self.mib[i] > 0:
                groups[self.mib[i]].append(i)
        self.mib_groups = groups
        for idxs in groups.values():
            ref = next((i for i in idxs if self.kind[i] != 0), None)
            if ref is None:
                continue
            w, h = self.rw[ref], self.rh[ref]
            for i in idxs:
                if self.kind[i] == 0 and abs(w * h - self.areas[i]) / self.areas[i] <= MIB_AREA_GUARD:
                    self.kind[i] = 1
                    self.rw[i], self.rh[i] = w, h
        # perimeter-pack pins (see _perimeter_pack): lock the packed
        # boundary blocks flush on their walls
        for i, (px, py, pw, ph) in self._pinned.items():
            if self.kind[i] == 2:
                continue
            self.kind[i] = 2
            self.lx[i], self.ly[i], self.rw[i], self.rh[i] = px, py, pw, ph

    # ------------------------------------------------------------------
    def _build_hpwl_arrays(self, b2b, p2b, pins):
        n = self.n
        if b2b is not None and len(b2b) > 0 and b2b.dim() == 2 and b2b.shape[1] >= 3:
            arr = b2b.detach().cpu().numpy()
            m = (arr[:, 0] != -1) & (arr[:, 0] < n) & (arr[:, 1] < n) & (arr[:, 0] >= 0) & (arr[:, 1] >= 0)
            self.eI = arr[m, 0].astype(np.int64)
            self.eJ = arr[m, 1].astype(np.int64)
            self.eW = arr[m, 2].astype(np.float64)
        else:
            self.eI = np.zeros(0, dtype=np.int64)
            self.eJ = np.zeros(0, dtype=np.int64)
            self.eW = np.zeros(0)
        if (p2b is not None and pins is not None and len(p2b) > 0
                and p2b.dim() == 2 and p2b.shape[1] >= 3 and len(pins) > 0):
            arr = p2b.detach().cpu().numpy()
            pn = pins.detach().cpu().numpy()
            m = (arr[:, 0] != -1) & (arr[:, 0] >= 0) & (arr[:, 0] < len(pn)) \
                & (arr[:, 1] >= 0) & (arr[:, 1] < n)
            pidx = arr[m, 0].astype(np.int64)
            self.pB = arr[m, 1].astype(np.int64)
            self.pW = arr[m, 2].astype(np.float64)
            self.pX = pn[pidx, 0].astype(np.float64)
            self.pY = pn[pidx, 1].astype(np.float64)
        else:
            self.pB = np.zeros(0, dtype=np.int64)
            self.pW = np.zeros(0)
            self.pX = np.zeros(0)
            self.pY = np.zeros(0)

    def _build_soft_norm(self):
        n_soft = sum(1 for i in range(self.n) if self.boundary[i] > 0)
        for idxs in self.mib_groups.values():
            n_soft += max(0, len(idxs) - 1)
        clus: Dict[int, List[int]] = defaultdict(list)
        for i in range(self.n):
            if self.cluster[i] > 0:
                clus[self.cluster[i]].append(i)
        self.cluster_groups = clus
        for idxs in clus.values():
            n_soft += max(0, len(idxs) - 1)
        self.n_soft_den = max(n_soft, 1)

        bnd = [(i, self.boundary[i]) for i in range(self.n) if self.boundary[i] > 0]
        self._bnd_idx = np.array([i for i, _ in bnd], dtype=np.int64)
        self._bnd_codes = np.array([c for _, c in bnd], dtype=np.int64)
        # clusters whose members are all movable form one contiguous stack by
        # construction — only groups containing locked blocks need checking
        self._clu_arrays = [np.array(idxs, dtype=np.int64)
                            for idxs in clus.values()
                            if len(idxs) >= 2 and any(self.kind[i] == 2 for i in idxs)]
        self._mib_arrays = [np.array(idxs, dtype=np.int64)
                            for idxs in self.mib_groups.values() if len(idxs) >= 2]

    # ------------------------------------------------------------------
    def _choose_frame(self, pins):
        total = 0.0
        for i in range(self.n):
            if self.kind[i] == 0:
                total += self.areas[i]
            else:
                total += self.rw[i] * self.rh[i]
        self.total_area = max(total, 1.0)
        aspect = 1.0
        if pins is not None and len(pins) > 1:
            pn = pins.detach().cpu().numpy()
            valid = pn[(pn[:, 0] != -1) & (pn[:, 1] != -1)]
            if len(valid) >= 2:
                dx = float(valid[:, 0].max() - valid[:, 0].min())
                dy = float(valid[:, 1].max() - valid[:, 1].min())
                if dx > 1.0 and dy > 1.0:
                    aspect = max(0.35, min(2.8, dx / dy))
        frame_area = self.total_area / UTIL_TARGET_FRAME
        H = math.sqrt(frame_area / aspect) * self.h_scale
        # A preplaced block tagged "touch top" reveals the intended frame
        # height exactly — pin H to it so the tag is satisfiable.
        pinned_H = None
        for i in range(self.n):
            if self.kind[i] == 2 and (self.boundary[i] & 4):
                top = self.ly[i] + self.rh[i]
                pinned_H = top if pinned_H is None else max(pinned_H, top)
        if pinned_H is not None:
            H = pinned_H
        # PARTNER_FRAME_WPIN arm: an L/R-tagged preplaced block reveals the
        # intended frame WIDTH the same way a top-tagged one reveals its
        # height.  We do not pin W directly -- the column solve owns width --
        # so we express the same frame through its height and let the
        # existing per-column width solve converge onto W*.  This
        # deliberately overrides `pinned_H`: an arm that pins both is
        # over-constrained, and the height-pinned arms are still in the pool
        # alongside this one.
        if self.w_star is not None and self.w_star > 1.0:
            H = self.total_area / (FRAME_WPIN_UTIL * self.w_star)
        for (x, y, w, h) in self.locked_rects:
            H = max(H, y + h)
        for i in range(self.n):
            if self.kind[i] == 1:
                H = max(H, self.rh[i])
        self.H = H
        self.W_est = frame_area / H
        self.area_ref = self.total_area / UTIL_TARGET_REF

    # ------------------------------------------------------------------
    def _build_units(self):
        n = self.n
        movable = [i for i in range(n) if self.kind[i] != 2]
        dsu = _DSU(n)
        for idxs in self.cluster_groups.values():
            mv = [i for i in idxs if self.kind[i] != 2]
            for a, b in zip(mv, mv[1:]):
                dsu.union(a, b)
        for idxs in self.mib_groups.values():
            # soft (no hard-shape reference) MIB members must share a column
            if any(self.kind[i] != 0 for i in idxs):
                continue
            for a, b in zip(idxs, idxs[1:]):
                dsu.union(a, b)

        roots: Dict[int, List[int]] = defaultdict(list)
        for i in movable:
            roots[dsu.find(i)].append(i)

        units: List[_Unit] = []
        for uid, (_, ids) in enumerate(sorted(roots.items())):
            u = _Unit(uid)
            # subgroups: one per cluster id (contiguity requirement); blocks
            # without a cluster are their own singleton subgroup.
            sub: Dict[int, List[int]] = defaultdict(list)
            singles: List[List[int]] = []
            for i in ids:
                if self.cluster[i] > 0:
                    sub[self.cluster[i]].append(i)
                else:
                    singles.append([i])
            subgroups = list(sub.values()) + singles

            def blk_key(i):
                b = self.boundary[i]
                rank = 0 if (b & 8) else (2 if (b & 4) else 1)
                return (rank, self.rects[i][1] if i < len(self.rects) else 0.0)

            for sg in subgroups:
                sg.sort(key=blk_key)

            def sg_key(sg):
                has_b = any(self.boundary[i] & 8 for i in sg)
                has_t = any(self.boundary[i] & 4 for i in sg)
                rank = 0 if has_b else (2 if has_t else 1)
                my = sum(self.rects[i][1] for i in sg) / len(sg)
                return (rank, my)

            subgroups.sort(key=sg_key)
            u.subgroups = subgroups
            u.flatten()

            for i in u.blocks:
                if self.kind[i] == 0:
                    u.soft_area += self.areas[i]
                    u.area_total += self.areas[i]
                else:
                    u.rigid_h += self.rh[i]
                    u.max_rigid_w = max(u.max_rigid_w, self.rw[i])
                    u.area_total += self.rw[i] * self.rh[i]
                b = self.boundary[i]
                if b & 1:
                    u.force = 'L'
                elif b & 2 and u.force is None:
                    u.force = 'R'
                if b & 8:
                    u.hasB = True
                if b & 4:
                    u.hasT = True

            # anchors: preplaced members of any cluster group in this unit
            anchor_ids = set()
            for i in u.blocks:
                g = self.cluster[i]
                if g > 0:
                    for j in self.cluster_groups[g]:
                        if self.kind[j] == 2:
                            anchor_ids.add(j)
            u.anchors = [(self.lx[j], self.ly[j], self.rw[j], self.rh[j]) for j in sorted(anchor_ids)]

            wsum = max(u.area_total, 1e-9)
            sx = sy = 0.0
            for i in u.blocks:
                a = self.areas[i] if self.kind[i] == 0 else self.rw[i] * self.rh[i]
                r = self.rects[i] if i < len(self.rects) else (0, 0, 1, 1)
                sx += a * (r[0] + r[2] * 0.5)
                sy += a * (r[1] + r[3] * 0.5)
            u.seed_x = sx / wsum
            u.seed_y = sy / wsum
            units.append(u)

        self.units = units
        for u in units:
            self._refresh_unit(u)
            u.pairable = (len(u.blocks) == 1 and self.kind[u.blocks[0]] == 0
                          and u.force is None and not u.hasB and not u.hasT
                          and not u.anchors)
        self.blk_unit = [-1] * n
        for k, u in enumerate(units):
            for i in u.blocks:
                self.blk_unit[i] = k
        self._unit_col = [0] * len(units)
        self._col_spans: List[Optional[Tuple[float, float]]] = []

    def _col_of_x(self, xq: float, C: int) -> int:
        spans = self._col_spans
        best = None
        best_d = None
        for ci, sp in enumerate(spans[:C]):
            if sp is None:
                continue
            if sp[0] - 1e-9 <= xq <= sp[1] + 1e-9:
                return ci
            d = min(abs(xq - sp[0]), abs(xq - sp[1]))
            if best_d is None or d < best_d:
                best_d = d
                best = ci
        if best is None:
            return self.rng.randrange(C)
        return best

    # ------------------------------------------------------------------
    def _init_columns(self, C: int) -> List[List[int]]:
        cols: List[List[int]] = [[] for _ in range(C)]
        units = self.units
        total = sum(u.area_total for u in units)
        cap = [total / C] * C          # area capacity per column
        rigid_cap = 0.55 * self.H      # keep columns solvable (content <= H)
        rigid_used = [0.0] * C

        def put(k: int, ci: int):
            u = units[k]
            cols[ci].append(k)
            cap[ci] -= u.area_total
            rigid_used[ci] += u.rigid_h

        forced = []
        free = []
        for k, u in enumerate(units):
            if u.force == 'L':
                forced.append((k, 0))
            elif u.force == 'R':
                forced.append((k, C - 1))
            elif u.anchors:
                ax = u.anchors[0][0] + u.anchors[0][2] * 0.5
                forced.append((k, max(0, min(C - 1, int(ax / max(self.W_est, 1e-6) * C)))))
            else:
                free.append(k)
        for k, ci in forced:
            put(k, ci)

        free.sort(key=lambda k: units[k].seed_x)
        ci = 0
        for k in free:
            u = units[k]
            while ci < C - 1 and (cap[ci] <= 0.0 or rigid_used[ci] + u.rigid_h > rigid_cap):
                ci += 1
            # spill rigid-heavy units to the least-loaded feasible column
            if rigid_used[ci] + u.rigid_h > rigid_cap:
                ci2 = min(range(C), key=lambda c: rigid_used[c])
                put(k, ci2)
            else:
                put(k, ci)

        def unit_key(k):
            u = units[k]
            has_b = any(self.boundary[i] & 8 for i in u.blocks)
            has_t = any(self.boundary[i] & 4 for i in u.blocks)
            rank = 0 if has_b else (2 if has_t else 1)
            return (rank, u.seed_y)

        for c in cols:
            c.sort(key=unit_key)
        return cols

    # ------------------------------------------------------------------
    def _obstacles_in(self, x0: float, x1: float) -> List[List[float]]:
        iv = []
        for (ox, oy, ow, oh) in self.locked_rects:
            if ox < x1 - 1e-9 and ox + ow > x0 + 1e-9:
                iv.append([oy, oy + oh])
        if not iv:
            return []
        iv.sort()
        merged = [iv[0][:]]
        for s, e in iv[1:]:
            if s <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def _unit_height(self, u: _Unit, w: float) -> float:
        return self._unit_h(u, w)

    def _partition_subgroup(self, sg: List[int]) -> List[List[int]]:
        """Split a cluster with several bottom/top-tagged members into
        side-by-side chunks so each tag can touch its edge. Connectivity is
        preserved: chunks stand shoulder to shoulder inside the unit."""
        if len(sg) < 2:
            return [sg]
        bnd = self.boundary
        B = [i for i in sg if bnd[i] & 8]
        T = [i for i in sg if (bnd[i] & 4) and not (bnd[i] & 8)]
        hasL = any(bnd[i] & 1 for i in sg)
        hasR = any(bnd[i] & 2 for i in sg)
        m = max(len(B), len(T))
        # a single left/right-tagged member should not force the whole
        # cluster into one full-width stack — flatten it into chunks and put
        # that member on the outer edge
        if (hasL or hasR) and len(sg) >= 3:
            m = max(m, min(3, len(sg) - 1))
        if m < 2:
            return [sg]
        if any(self.kind[i] == 0 and self.mib[i] > 0 for i in sg):
            return [sg]
        m = min(m, len(sg), 4)

        def blk_area(i):
            return self.areas[i] if self.kind[i] == 0 else self.rw[i] * self.rh[i]

        chunks: List[List[int]] = [[] for _ in range(m)]
        areas = [0.0] * m
        assigned = set()
        for i in sg:
            if bnd[i] & 1:
                chunks[0].append(i)
                areas[0] += blk_area(i)
                assigned.add(i)
            elif bnd[i] & 2:
                chunks[m - 1].append(i)
                areas[m - 1] += blk_area(i)
                assigned.add(i)
        b_seen = [any(bnd[i] & 8 for i in ch) for ch in chunks]
        for i in B:
            if i in assigned:
                continue
            free = [c for c in range(m) if not b_seen[c]]
            c = free[0] if free else min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)
            b_seen[c] = True
            assigned.add(i)
        t_seen = [any(bnd[i] & 4 for i in ch) for ch in chunks]
        for i in T:
            if i in assigned:
                continue
            free = [c for c in range(m) if not t_seen[c]]
            c = free[0] if free else min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)
            t_seen[c] = True
            assigned.add(i)
        for i in sorted((j for j in sg if j not in assigned), key=lambda j: -blk_area(j)):
            c = min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)

        def blk_rank(i):
            b = bnd[i]
            return (0 if (b & 8) else (2 if (b & 4) else 1))

        for ch in chunks:
            ch.sort(key=blk_rank)
        return [c for c in chunks if c]

    def _refresh_unit(self, u: _Unit):
        u.flatten()
        bands = []
        eff_s = 0.0
        eff_r = 0.0
        for sg in u.subgroups:
            band = []
            for part in self._partition_subgroup(sg):
                pl = [(i, self.kind[i] == 0, self.areas[i], self.rw[i], self.rh[i])
                      for i in part]
                sa = sum(self.areas[i] for i in part if self.kind[i] == 0)
                rh = sum(self.rh[i] for i in part if self.kind[i] != 0)
                rmw = max((self.rw[i] for i in part if self.kind[i] != 0), default=0.0)
                band.append((pl, sa, rh, rmw))
            bands.append(band)
            if len(band) == 1:
                eff_s += band[0][1]
                eff_r += band[0][2]
            else:
                eff_s += sum(ch[1] for ch in band)
                eff_s += sum(bh * bw for ch in band
                             for (_i, soft, _a, bw, bh) in ch[0] if not soft)
        u.bands = bands
        u.eff_soft = eff_s
        u.eff_rigid_h = eff_r
        # bands eligible for width-dependent splitting at layout time: a
        # single stack of >=2 blocks without soft-MIB members (those need one
        # shared width for identical shapes)
        u.dyn = [len(b) == 1 and len(b[0][0]) >= 2
                 and not any(self.kind[e[0]] == 0 and self.mib[e[0]] > 0 for e in b[0][0])
                 for b in bands]
        u.banded = any(len(b) > 1 for b in bands) or any(u.dyn)
        u.hcache = None
        _k = getattr(self, "_sa_kernel", None)
        if _k is not None:
            _k.mark_dirty()

    def _solve_band(self, band, w: float):
        """Find the band height h so the chunk widths sum to w. Mixed chunks
        satisfy w_j = soft_j / (h - rigid_h_j); pure-rigid chunks have fixed
        width. Returns (h, widths) or None if infeasible at this width."""
        pure_w = 0.0
        lo = 0.0
        all_soft = True
        for (_pl, sa, rh, rmw) in band:
            if sa <= 0.0:
                pure_w += rmw
                all_soft = False
            elif rh > 0.0:
                all_soft = False
            if rh > lo:
                lo = rh
        avail = w - pure_w
        if avail <= 1e-6:
            return None
        if all_soft:
            total = sum(ch[1] for ch in band)
            h = total / w
            return h, [ch[1] / h for ch in band]
        lo += 1e-9

        def width_at(h):
            s = 0.0
            for (_pl, sa, rh, _rmw) in band:
                if sa > 0.0:
                    s += sa / (h - rh)
            return s

        hi = lo + 1.0
        for _ in range(80):
            if width_at(hi) <= avail:
                break
            hi *= 2.0
        else:
            return None
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if width_at(mid) > avail:
                lo = mid
            else:
                hi = mid
        h = hi
        widths = []
        for (_pl, sa, rh, rmw) in band:
            if sa > 0.0:
                wj = sa / (h - rh)
                if wj < rmw - 1e-9:
                    return None
                widths.append(wj)
            else:
                widths.append(rmw)
        return h, widths

    def _merge_band_once(self, band):
        """Merge the two adjacent chunks with the smallest combined area."""
        def chunk_area(ch):
            a = ch[1]
            for (_i, soft, _a, bw, bh) in ch[0]:
                if not soft:
                    a += bw * bh
            return a
        areas = [chunk_area(ch) for ch in band]
        j = min(range(len(band) - 1), key=lambda t: areas[t] + areas[t + 1])
        a, b = band[j], band[j + 1]
        bnd = self.boundary

        def rank(entry):
            code = bnd[entry[0]]
            return 0 if (code & 8) else (2 if (code & 4) else 1)

        pl = sorted(a[0] + b[0], key=rank)
        merged = (pl, a[1] + b[1], a[2] + b[2], max(a[3], b[3]))
        return band[:j] + [merged] + band[j + 2:]

    def _band_solutions(self, u: _Unit, w: float):
        """(w, unit_height, per-band placement plan) with a one-slot cache.
        Each plan entry is (effective_band, solution): when a chunk layout is
        infeasible at this width, chunks are merged progressively instead of
        collapsing the whole band into one stack."""
        if u.hcache is not None and u.hcache[0] == w:
            return u.hcache
        plan = []
        h = 0.0
        for band, dyn_ok in zip(u.bands, u.dyn):
            eff = band
            sol = None
            if len(eff) == 1 and dyn_ok:
                d = self._dyn_split(eff[0], w)
                if d is not None:
                    eff = d
            if len(eff) > 1:
                sol = self._solve_band(eff, w)
                while sol is None and len(eff) > 2:
                    eff = self._merge_band_once(eff)
                    sol = self._solve_band(eff, w)
            if sol is None:
                if len(eff) > 1:
                    eff = [(sorted((e for ch in eff for e in ch[0]),
                                   key=lambda t: 0 if (self.boundary[t[0]] & 8)
                                   else (2 if (self.boundary[t[0]] & 4) else 1)),
                            sum(ch[1] for ch in eff),
                            sum(ch[2] for ch in eff),
                            max(ch[3] for ch in eff))]
                ch = eff[0]
                hb = ch[2] + (ch[1] / w if ch[1] > 0.0 else 0.0)
            else:
                hb = sol[0]
            plan.append((eff, sol))
            h += hb
        u.hcache = (w, h, plan)
        return u.hcache

    def _dyn_split(self, ch, w: float):
        """Split a wide single-stack chunk into side-by-side chunks so slice
        aspect stays reasonable. Order-preserving, so cluster chains remain
        connected (adjacent chunks touch along their full shared edge)."""
        pl = ch[0]
        total = ch[1]
        for (_i, soft, _a, bw, bh) in pl:
            if not soft:
                total += bw * bh
        avg = total / len(pl)
        m = int(w / max(1.35 * math.sqrt(max(avg, 1e-9)), 1e-6))
        if m < 2:
            return None
        m = min(m, len(pl), 4)
        target = total / m
        parts = []
        cur = []
        acc = 0.0
        for e in pl:
            cur.append(e)
            acc += e[2] if e[1] else e[3] * e[4]
            if acc >= target - 1e-9 and len(parts) < m - 1:
                parts.append(cur)
                cur = []
                acc = 0.0
        if cur:
            parts.append(cur)
        if len(parts) < 2:
            return None
        chunks = []
        for part in parts:
            sa = sum(e[2] for e in part if e[1])
            rh = sum(e[4] for e in part if not e[1])
            rmw = max((e[3] for e in part if not e[1]), default=0.0)
            chunks.append((part, sa, rh, rmw))
        return chunks

    def _unit_h(self, u: _Unit, w: float) -> float:
        if not u.banded:
            return u.eff_rigid_h + u.eff_soft / w
        return self._band_solutions(u, w)[1]

    def _place_chunk_up(self, pl, xj, wj, y0, pos, full_w=None):
        y = y0
        for (i, soft, a, bw, bh) in pl:
            if soft:
                bw = wj if full_w is None else full_w
                bh = a / bw
            pos[i, 0] = xj
            pos[i, 1] = y
            pos[i, 2] = bw
            pos[i, 3] = bh
            y += bh
        return y

    def _place_band_up(self, band, x0, w, y, pos, sol=None):
        if len(band) == 1:
            return self._place_chunk_up(band[0][0], x0, w, y, pos, full_w=w)
        if sol is None:
            yy = y
            for ch in band:
                yy = self._place_chunk_up(ch[0], x0, w, yy, pos, full_w=w)
            return yy
        h_b, widths = sol
        xj = x0
        for ch, wj in zip(band, widths):
            self._place_chunk_up(ch[0], xj, wj, y, pos)
            xj += wj
        return y + h_b

    def _place_band_down(self, band, x0, w, ytop, pos, sol=None):
        if len(band) == 1:
            y = ytop
            for (i, soft, a, bw, bh) in reversed(band[0][0]):
                if soft:
                    bw = w
                    bh = a / w
                y -= bh
                pos[i, 0] = x0
                pos[i, 1] = y
                pos[i, 2] = bw
                pos[i, 3] = bh
            return y
        if sol is None:
            y = ytop
            for ch in reversed(band):
                for (i, soft, a, bw, bh) in reversed(ch[0]):
                    if soft:
                        bw = w
                        bh = a / w
                    y -= bh
                    pos[i, 0] = x0
                    pos[i, 1] = y
                    pos[i, 2] = bw
                    pos[i, 3] = bh
            return y
        h_b, widths = sol
        xj = x0
        for ch, wj in zip(band, widths):
            y = ytop
            for (i, soft, a, bw, bh) in reversed(ch[0]):
                if soft:
                    bw = wj
                    bh = a / wj
                y -= bh
                pos[i, 0] = xj
                pos[i, 1] = y
                pos[i, 2] = bw
                pos[i, 3] = bh
            xj += wj
        return ytop - h_b

    def _place_unit_up(self, u: _Unit, x0: float, w: float, y0: float, pos: np.ndarray) -> float:
        if not u.banded:
            y = y0
            for band in u.bands:
                y = self._place_band_up(band, x0, w, y, pos)
            return y
        plan = self._band_solutions(u, w)[2]
        y = y0
        for band_eff, sol in plan:
            y = self._place_band_up(band_eff, x0, w, y, pos, sol)
        return y

    def _place_unit_down(self, u: _Unit, x0: float, w: float, ytop: float, pos: np.ndarray) -> float:
        if not u.banded:
            y = ytop
            for band in reversed(u.bands):
                y = self._place_band_down(band, x0, w, y, pos)
            return y
        plan = self._band_solutions(u, w)[2]
        y = ytop
        for band_eff, sol in reversed(plan):
            y = self._place_band_down(band_eff, x0, w, y, pos, sol)
        return y

    def _stack_column(self, ulist, x, w, pos):
        """Stack a column's units at width w; returns (placed, occupied, col_top)."""
        units = self.units
        has_locked = bool(self.locked_rects)
        occupied = [iv[:] for iv in self._obstacles_in(x, x + w)] if has_locked else []

        placed = []
        normal = []
        # anchored clusters: glue to their preplaced member if it is an
        # obstacle of this column
        for k in ulist:
            u = units[k]
            done = False
            if u.anchors and has_locked:
                uh = self._unit_h(u, w)
                for (ax, ay, aw, ah) in u.anchors:
                    if not (ax < x + w - 1e-9 and ax + aw > x + 1e-9):
                        continue
                    y0 = ay - uh
                    if y0 >= -1e-9 and self._interval_free(occupied, y0, ay):
                        yb = self._place_unit_down(u, x, w, ay, pos)
                        self._add_interval(occupied, yb, ay)
                        placed.append((k, yb, ay))
                        done = True
                        break
                    y1 = ay + ah
                    if self._interval_free(occupied, y1, y1 + uh):
                        yt = self._place_unit_up(u, x, w, y1, pos)
                        self._add_interval(occupied, y1, yt)
                        placed.append((k, y1, yt))
                        done = True
                        break
            if not done:
                normal.append(k)

        # enforce boundary-friendly stacking order regardless of the SA
        # permutation: bottom-tagged units first, top-tagged last
        if normal:
            bottoms = []
            mids = []
            tops = []
            for k in normal:
                u = units[k]
                if u.hasB and not u.hasT:
                    bottoms.append(k)
                elif u.hasT:
                    tops.append(k)
                else:
                    mids.append(k)
            normal = bottoms + mids + tops

        # bottom pre-pass: when the column bottom is blocked by an obstacle, a
        # bottom-tagged unit can still stand at y=0 in the strip beside it
        if occupied and normal:
            for k in list(normal):
                u = units[k]
                if not u.hasB or u.hasT:
                    break
                uh = self._unit_h(u, w)
                if self._interval_free(occupied, 0.0, uh):
                    break  # bottom is open; the normal flow handles it
                if u.force is not None:
                    break
                sw = None
                probe_h = uh
                for _ in range(3):
                    strip = self._band_strip(x, w, 0.0, probe_h, placed)
                    if strip is None:
                        sw = None
                        break
                    sw = strip
                    nh = self._unit_h(u, strip[1])
                    if abs(nh - probe_h) < 1e-6:
                        probe_h = nh
                        break
                    probe_h = nh
                if sw is None or u.max_rigid_w > sw[1] + 1e-9:
                    break
                final = self._band_strip(x, w, 0.0, probe_h, placed)
                if final is None or final[1] < sw[1] - 1e-6:
                    break
                yt = self._place_unit_up(u, sw[0], sw[1], 0.0, pos)
                self._add_interval(occupied, 0.0, yt)
                placed.append((k, 0.0, yt))
                normal.remove(k)
                break

        col_top = 0.0
        if not occupied:
            cy = 0.0
            i = 0
            nn = len(normal)
            while i < nn:
                k = normal[i]
                u = units[k]
                # pair two flat unconstrained soft singles side by side:
                # exact fill, near-square blocks, half the stack height
                if i + 1 < nn:
                    k2 = normal[i + 1]
                    u2 = units[k2]
                    if (len(u.blocks) == 1 and len(u2.blocks) == 1
                            and u.pairable and u2.pairable):
                        a1 = u.soft_area
                        a2 = u2.soft_area
                        hp = (a1 + a2) / w
                        if hp <= 0.85 * w:
                            w1 = a1 / hp
                            if 1.0 <= w1 <= w - 1.0:
                                i1 = u.blocks[0]
                                i2 = u2.blocks[0]
                                pos[i1, 0] = x
                                pos[i1, 1] = cy
                                pos[i1, 2] = w1
                                pos[i1, 3] = a1 / w1
                                pos[i2, 0] = x + w1
                                pos[i2, 1] = cy
                                pos[i2, 2] = w - w1
                                pos[i2, 3] = a2 / (w - w1)
                                top = cy + hp
                                placed.append((k, cy, top))
                                placed.append((k2, cy, top))
                                cy = top
                                i += 2
                                continue
                yt = self._place_unit_up(u, x, w, cy, pos)
                placed.append((k, cy, yt))
                cy = yt
                i += 1
            col_top = cy
        else:
            # Free segments, including "narrow" segments beside obstacles that
            # only partially cover the column (the strip next to a preplaced
            # block is usable at reduced width instead of being wasted).
            segs = []  # (start, end, xoff, width)
            cur = 0.0
            for s, e in occupied:
                if s > cur + 1e-9:
                    segs.append((cur, s, x, w))
                band_lo, band_hi = max(cur, s), e
                strip = self._band_strip(x, w, band_lo, band_hi, placed)
                if strip is not None:
                    segs.append((band_lo, band_hi, strip[0], strip[1]))
                cur = max(cur, e)
            segs.append((cur, float('inf'), x, w))

            si = 0
            cy = segs[0][0]
            pending = list(normal)
            while pending:
                _s, seg_end, seg_x, seg_w = segs[si]
                pick = None
                # first unit that fits the current gap (look ahead a few)
                narrow = seg_x > x + 1e-9 or seg_w < w - 1e-9
                for t in range(min(len(pending), 10)):
                    u = units[pending[t]]
                    if u.max_rigid_w > seg_w + 1e-9:
                        continue
                    # left/right-forced units must keep the column's edge x
                    if narrow and u.force is not None:
                        continue
                    if cy + self._unit_h(u, seg_w) <= seg_end + 1e-9:
                        pick = t
                        break
                if pick is None:
                    si += 1
                    cy = segs[si][0]
                    continue
                k = pending.pop(pick)
                u = units[k]
                yt = self._place_unit_up(u, seg_x, seg_w, cy, pos)
                placed.append((k, cy, yt))
                cy = yt
            for (_k, _yb, yt) in placed:
                if yt > col_top:
                    col_top = yt
        return placed, occupied, col_top

    def _band_strip(self, x, w, lo, hi, placed_units):
        """Widest free x-strip inside column [x, x+w] over the y-band
        [lo, hi). Anchored units already placed span the full column width,
        so any overlap with them blocks the band entirely. Returns
        (xoff, width) or None when the strip is too narrow to be useful."""
        if hi <= lo + 1e-6:
            return None
        for (_k, yb, yt) in placed_units:
            if yb < hi - 1e-9 and yt > lo + 1e-9:
                return None
        left = x + w
        right = x
        for (ox, oy, ow, oh) in self.locked_rects:
            if ox < x + w - 1e-9 and ox + ow > x + 1e-9 and oy < hi - 1e-9 and oy + oh > lo + 1e-9:
                if ox < left:
                    left = ox
                if ox + ow > right:
                    right = ox + ow
        if right <= x:
            return None
        left_w = left - x
        right_w = (x + w) - right
        if right_w >= left_w:
            xoff, sw = right, right_w
        else:
            xoff, sw = x, left_w
        if sw < 0.30 * w or sw < 2.0:
            return None
        return (xoff, sw)

    @staticmethod
    def _interval_free(occ: List[List[float]], s: float, e: float) -> bool:
        for a, b in occ:
            if s < b - 1e-9 and a < e - 1e-9:
                return False
        return True

    @staticmethod
    def _add_interval(occ: List[List[float]], s: float, e: float):
        occ.append([s, e])
        occ.sort()
        merged = [occ[0][:]]
        for a, b in occ[1:]:
            if a <= merged[-1][1] + 1e-12:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        occ[:] = merged

    # ------------------------------------------------------------------
    def _layout(self, cols: List[List[int]]) -> Tuple[np.ndarray, float, float]:
        if self._sa_kernel is not None:
            out = self._sa_kernel.layout(cols)
            if out is not None:
                return out
        if self._dc_enabled:
            return self._layout_delta(cols)
        return self._layout_full(cols)

    def _layout_full(self, cols: List[List[int]]) -> Tuple[np.ndarray, float, float]:
        n = self.n
        pos = np.zeros((n, 4))
        for i in range(n):
            if self.kind[i] == 2:
                pos[i] = (self.lx[i], self.ly[i], self.rw[i], self.rh[i])

        H = self.H
        x = 0.0
        col_records = []  # (x0, w, [(unit, y_bot, y_top)], occupied)

        units = self.units
        boundary = self.boundary
        has_locked = bool(self.locked_rects)
        unit_col = self._unit_col
        col_spans = []
        for ci, ulist in enumerate(cols):
            if not ulist:
                col_records.append(None)
                col_spans.append(None)
                continue
            for k in ulist:
                unit_col[k] = ci
            soft_a = 0.0
            rigid_h = 0.0
            max_w = 0.0
            for k in ulist:
                u = units[k]
                soft_a += u.eff_soft
                rigid_h += u.eff_rigid_h
                if u.max_rigid_w > max_w:
                    max_w = u.max_rigid_w
            avail = H - rigid_h
            if avail < 0.05 * H:
                avail = 0.05 * H
            w = soft_a / avail
            if w < max_w:
                w = max_w
            if w < 0.5:
                w = 0.5
            if has_locked:
                for _ in range(3):
                    obs = self._obstacles_in(x, x + w)
                    obs_h = 0.0
                    for s, e in obs:
                        if e > 0.0 and s < H:
                            span = (e if e < H else H) - (s if s > 0.0 else 0.0)
                            strip = self._band_strip(x, w, s, e, ())
                            if strip is not None:
                                span *= 1.0 - strip[1] / w
                            obs_h += span
                    avail = H - rigid_h - obs_h
                    if avail < 0.05 * H:
                        avail = 0.05 * H
                    w2 = soft_a / avail
                    if w2 < max_w:
                        w2 = max_w
                    if w2 < 0.5:
                        w2 = 0.5
                    if abs(w2 - w) < 5e-3:
                        w = w2
                        break
                    w = w2

            placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
            # widen and retry when fragmentation pushed content past the frame;
            # only the soft part shrinks with width, so solve for it exactly
            if soft_a > 0:
                tries = 0
                while col_top > H * 1.0005 and tries < 3:
                    overhead = col_top - soft_a / w
                    if overhead < H * 0.98:
                        w2 = soft_a / (H - overhead)
                    else:
                        w2 = w * 1.25
                    w = min(max(w2, w * 1.01), w * 2.0)
                    placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
                    tries += 1

            # PARTNER_COL_NARROW: the mirror of the widen loop.  A column that
            # stacks SHORT of the frame is carrying dead area that the bbox
            # pays for; narrowing it in proportion to the shortfall trades
            # that dead area for frame width the neighbours can use.  The
            # acceptance test is the widen loop's own overflow guard -- if the
            # narrower width overflows, the previous width is restacked and
            # kept, so this can only ever return a legal column.
            if soft_a > 0 and self._col_narrow:
                tries = 0
                while col_top < H * 0.995 and tries < 2:
                    w2 = w * max(0.85, col_top / H)
                    if w2 < max_w:
                        w2 = max_w
                    if w2 < 0.5:
                        w2 = 0.5
                    if w2 >= w - 1e-9:
                        break
                    p2, o2, ct2 = self._stack_column(ulist, x, w2, pos)
                    if ct2 > H * 1.0005:
                        # overflowed: restore the accepted width and stop
                        placed, occupied, col_top = self._stack_column(
                            ulist, x, w, pos)
                        break
                    w, placed, occupied, col_top = w2, p2, o2, ct2
                    tries += 1

            col_records.append((x, w, placed, occupied))
            col_spans.append((x, x + w))
            x += w
        self._col_spans = col_spans

        x_right = x
        y_top = 0.0
        for rec in col_records:
            if rec is None:
                continue
            for (_k, _yb, yt) in rec[2]:
                y_top = max(y_top, yt)
        for (ox, oy, ow, oh) in self.locked_rects:
            x_right = max(x_right, ox + ow)
            y_top = max(y_top, oy + oh)

        # lift top-tagged top units flush to the global top edge
        for rec in col_records:
            if rec is None or not rec[2]:
                continue
            cx0, cw, placed, occupied = rec
            k, yb, yt = max(placed, key=lambda t: t[2])
            u = self.units[k]
            top_block = u.blocks[-1]
            if not (self.boundary[top_block] & 4):
                continue
            dy = y_top - yt
            if dy <= 1e-9:
                continue
            occ_wo = [iv for iv in occupied if not (abs(iv[0] - yb) < 1e-6 and abs(iv[1] - yt) < 1e-6)]
            # occupied intervals were merged; conservatively test only against
            # raw obstacles + other placed units above the current top
            blockers = self._obstacles_in(cx0, cx0 + cw)
            for (k2, yb2, yt2) in placed:
                if k2 != k:
                    blockers.append([yb2, yt2])
            if self._interval_free(blockers, yb + dy, yt + dy):
                for i in u.blocks:
                    pos[i, 1] += dy

        # right-align right-tagged blocks in the last non-empty column
        last = None
        for rec in reversed(col_records):
            if rec is not None and rec[2]:
                last = rec
                break
        if last is not None:
            for (k, _yb, _yt) in last[2]:
                u = self.units[k]
                for i in u.blocks:
                    if self.boundary[i] & 2:
                        nx = x_right - pos[i, 2]
                        if nx <= pos[i, 0] + 1e-9:
                            continue
                        # collision check against every other block (chunked
                        # units may have siblings to the right)
                        nx0, ny0 = nx, pos[i, 1]
                        nx1, ny1 = x_right, pos[i, 1] + pos[i, 3]
                        px0 = pos[:, 0]
                        py0 = pos[:, 1]
                        clash = ((px0 < nx1 - 1e-7)
                                 & (px0 + pos[:, 2] > nx0 + 1e-7)
                                 & (py0 < ny1 - 1e-7)
                                 & (py0 + pos[:, 3] > ny0 + 1e-7))
                        clash[i] = False
                        if not clash.any():
                            pos[i, 0] = nx

        return pos, x_right, y_top

    # ------------------------------------------------------------------
    # Column-cache delta-evaluation (PARTNER_COL_CACHE=1)
    # ------------------------------------------------------------------
    # Equivalence design (bit-exact; must match _layout_full byte-for-byte):
    #
    # A column ci is placed at x-offset = sum of widths of columns 0..ci-1.
    # _stack_column(ulist, x, w, pos) is a *deterministic* function of
    # (ulist, x, w, self.units' internal state, self.locked_rects). Between two
    # consecutive _layout calls the SA loop mutates only `cols` (and, for the
    # subgroup-reorder move, one unit's .subgroups via _refresh_unit); the
    # locked obstacles never move. Therefore a column reproduces bit-identical
    # output iff ALL of its inputs are bit-identical:
    #   (1) same membership+order  -> ulist == prev ulist
    #   (2) same unit internals    -> subgroup signature of its mutable units
    #                                 unchanged (single-block / fixed-order
    #                                 units cannot be reordered, so only
    #                                 self._dc_mutable members need checking)
    #   (3) same x-offset          -> x is bit-identical to the cached x
    #   (4) same width             -> follows from (1)+(3): w is a deterministic
    #                                 fn of ulist and (for obstacle columns) x.
    # Obstacles are handled implicitly: _stack_column's segment logic depends on
    # x only, so (3) already guarantees an obstacle-touching column reproduces
    # exactly. No separate obstacle test is needed.
    #
    # x propagates left-to-right, so once any column diverges (fails (1)/(2), or
    # is empty-after-nonempty, or is itself past the first change) every later
    # column's x shifts by a non-representable delta and can never re-align
    # bit-exactly -> the whole suffix is recomputed. Reuse is thus a *prefix*:
    # columns 0..(first change - 1). We never translate stored coordinates
    # (x_new + rel is NOT bit-equal to a fresh accumulation from x_new because
    # float addition is non-associative); we only reuse when x is unchanged, so
    # the cached rows are re-emitted verbatim.
    #
    # u.hcache (per-unit band-solve memo) is safe to leave stale across reused
    # columns: it is keyed on (u.bands, w) and _refresh_unit nulls it on
    # mutation, so any later read returns the same value a fresh solve would.
    #
    # The two global post-passes (top-tag lift, right-tag align) read global
    # state (y_top / x_right / all-block collision), so they are NEVER cached:
    # the cache stores the *pre-post-pass* rows, and the post-passes are re-run
    # every call on the fully assembled pos -- identical to _layout_full.
    def _layout_delta(self, cols: List[List[int]]) -> Tuple[np.ndarray, float, float]:
        n = self.n
        pos = np.zeros((n, 4))
        for i in range(n):
            if self.kind[i] == 2:
                pos[i] = (self.lx[i], self.ly[i], self.rw[i], self.rh[i])

        H = self.H
        x = 0.0
        col_records = []
        units = self.units
        has_locked = bool(self.locked_rects)
        unit_col = self._unit_col
        col_spans = []

        prev_cols = self._dc_prev_cols
        prev_sig = self._dc_prev_sig
        cache = self._dc_cache
        if len(cache) != len(cols):
            cache = [None] * len(cols)
        mutable = self._dc_mutable
        cur_sig: List[tuple] = []
        # can only reuse against a previous call with the same column count
        diverged = prev_cols is None or len(prev_cols) != len(cols)

        for ci, ulist in enumerate(cols):
            if mutable:
                sig = tuple((k, tuple(tuple(sg) for sg in units[k].subgroups))
                            for k in ulist if k in mutable)
            else:
                sig = ()
            cur_sig.append(sig)

            if not ulist:
                col_records.append(None)
                col_spans.append(None)
                cache[ci] = None
                # empty column contributes zero width; x is unchanged, but an
                # emptied (was non-empty) column dropped its width -> the
                # downstream x shifts, so the suffix must be recomputed.
                if not diverged and prev_cols[ci] != []:
                    diverged = True
                continue

            reuse = (not diverged
                     and ulist == prev_cols[ci]
                     and sig == prev_sig[ci]
                     and cache[ci] is not None
                     and cache[ci][0] == x)
            if reuse:
                cx, cw, pos_block, record, span = cache[ci]
                for (i, px, py, pw, ph) in pos_block:
                    pos[i, 0] = px
                    pos[i, 1] = py
                    pos[i, 2] = pw
                    pos[i, 3] = ph
                for k in ulist:
                    unit_col[k] = ci
                col_records.append(record)
                col_spans.append(span)
                x = cx + cw
                self._dc_hits += 1
                continue

            # -- recompute this column: verbatim _layout_full per-column body --
            diverged = True
            self._dc_recompute += 1
            for k in ulist:
                unit_col[k] = ci
            soft_a = 0.0
            rigid_h = 0.0
            max_w = 0.0
            for k in ulist:
                u = units[k]
                soft_a += u.eff_soft
                rigid_h += u.eff_rigid_h
                if u.max_rigid_w > max_w:
                    max_w = u.max_rigid_w
            avail = H - rigid_h
            if avail < 0.05 * H:
                avail = 0.05 * H
            w = soft_a / avail
            if w < max_w:
                w = max_w
            if w < 0.5:
                w = 0.5
            if has_locked:
                for _ in range(3):
                    obs = self._obstacles_in(x, x + w)
                    obs_h = 0.0
                    for s, e in obs:
                        if e > 0.0 and s < H:
                            span2 = (e if e < H else H) - (s if s > 0.0 else 0.0)
                            strip = self._band_strip(x, w, s, e, ())
                            if strip is not None:
                                span2 *= 1.0 - strip[1] / w
                            obs_h += span2
                    avail = H - rigid_h - obs_h
                    if avail < 0.05 * H:
                        avail = 0.05 * H
                    w2 = soft_a / avail
                    if w2 < max_w:
                        w2 = max_w
                    if w2 < 0.5:
                        w2 = 0.5
                    if abs(w2 - w) < 5e-3:
                        w = w2
                        break
                    w = w2

            placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
            if soft_a > 0:
                tries = 0
                while col_top > H * 1.0005 and tries < 3:
                    overhead = col_top - soft_a / w
                    if overhead < H * 0.98:
                        w2 = soft_a / (H - overhead)
                    else:
                        w2 = w * 1.25
                    w = min(max(w2, w * 1.01), w * 2.0)
                    placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
                    tries += 1

            # PARTNER_COL_NARROW: mirror of the narrow loop in `_layout_full`
            # (kept in sync so the delta cache and the full path agree).  w
            # stays a deterministic function of (ulist, x, H), which is what
            # the cache's correctness argument needs.
            if soft_a > 0 and self._col_narrow:
                tries = 0
                while col_top < H * 0.995 and tries < 2:
                    w2 = w * max(0.85, col_top / H)
                    if w2 < max_w:
                        w2 = max_w
                    if w2 < 0.5:
                        w2 = 0.5
                    if w2 >= w - 1e-9:
                        break
                    p2, o2, ct2 = self._stack_column(ulist, x, w2, pos)
                    if ct2 > H * 1.0005:
                        placed, occupied, col_top = self._stack_column(
                            ulist, x, w, pos)
                        break
                    w, placed, occupied, col_top = w2, p2, o2, ct2
                    tries += 1

            record = (x, w, placed, occupied)
            span = (x, x + w)
            col_records.append(record)
            col_spans.append(span)
            # cache the pre-post-pass rows for every block this column placed
            pos_block = [(i, pos[i, 0], pos[i, 1], pos[i, 2], pos[i, 3])
                         for k in ulist for i in units[k].blocks]
            cache[ci] = (x, w, pos_block, record, span)
            x += w

        self._col_spans = col_spans
        self._dc_cache = cache
        self._dc_prev_cols = [list(c) for c in cols]
        self._dc_prev_sig = cur_sig

        # -- global post-passes: verbatim _layout_full tail (never cached) --
        x_right = x
        y_top = 0.0
        for rec in col_records:
            if rec is None:
                continue
            for (_k, _yb, yt) in rec[2]:
                y_top = max(y_top, yt)
        for (ox, oy, ow, oh) in self.locked_rects:
            x_right = max(x_right, ox + ow)
            y_top = max(y_top, oy + oh)

        for rec in col_records:
            if rec is None or not rec[2]:
                continue
            cx0, cw, placed, occupied = rec
            k, yb, yt = max(placed, key=lambda t: t[2])
            u = self.units[k]
            top_block = u.blocks[-1]
            if not (self.boundary[top_block] & 4):
                continue
            dy = y_top - yt
            if dy <= 1e-9:
                continue
            occ_wo = [iv for iv in occupied if not (abs(iv[0] - yb) < 1e-6 and abs(iv[1] - yt) < 1e-6)]
            blockers = self._obstacles_in(cx0, cx0 + cw)
            for (k2, yb2, yt2) in placed:
                if k2 != k:
                    blockers.append([yb2, yt2])
            if self._interval_free(blockers, yb + dy, yt + dy):
                for i in u.blocks:
                    pos[i, 1] += dy

        last = None
        for rec in reversed(col_records):
            if rec is not None and rec[2]:
                last = rec
                break
        if last is not None:
            for (k, _yb, _yt) in last[2]:
                u = self.units[k]
                for i in u.blocks:
                    if self.boundary[i] & 2:
                        nx = x_right - pos[i, 2]
                        if nx <= pos[i, 0] + 1e-9:
                            continue
                        nx0, ny0 = nx, pos[i, 1]
                        nx1, ny1 = x_right, pos[i, 1] + pos[i, 3]
                        px0 = pos[:, 0]
                        py0 = pos[:, 1]
                        clash = ((px0 < nx1 - 1e-7)
                                 & (px0 + pos[:, 2] > nx0 + 1e-7)
                                 & (py0 < ny1 - 1e-7)
                                 & (py0 + pos[:, 3] > ny0 + 1e-7))
                        clash[i] = False
                        if not clash.any():
                            pos[i, 0] = nx

        return pos, x_right, y_top

    # ------------------------------------------------------------------
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

    def _violations(self, pos: np.ndarray) -> int:
        if self._sa_kernel is not None:
            return self._sa_kernel.violations(pos)
        V = 0
        px0 = pos[:, 0]
        py0 = pos[:, 1]
        px1 = px0 + pos[:, 2]
        py1 = py0 + pos[:, 3]
        x_min = px0.min()
        y_min = py0.min()
        x_max = px1.max()
        y_max = py1.max()
        eps = 1e-6
        b = self._bnd_idx
        if len(b):
            codes = self._bnd_codes
            bad = ((codes & 1) != 0) & (np.abs(px0[b] - x_min) >= eps)
            bad |= ((codes & 2) != 0) & (np.abs(px1[b] - x_max) >= eps)
            bad |= ((codes & 4) != 0) & (np.abs(py1[b] - y_max) >= eps)
            bad |= ((codes & 8) != 0) & (np.abs(py0[b] - y_min) >= eps)
            V += int(bad.sum())

        for g in self._clu_arrays:
            gx0 = px0[g]
            gy0 = py0[g]
            gx1 = px1[g]
            gy1 = py1[g]
            ox = np.minimum(gx1[:, None], gx1[None, :]) - np.maximum(gx0[:, None], gx0[None, :])
            oy = np.minimum(gy1[:, None], gy1[None, :]) - np.maximum(gy0[:, None], gy0[None, :])
            adj = ((ox > TOUCH_TOL) & (oy >= -TOUCH_TOL)) | ((oy > TOUCH_TOL) & (ox >= -TOUCH_TOL))
            m = len(g)
            seen = np.zeros(m, dtype=bool)
            comps = 0
            for s in range(m):
                if seen[s]:
                    continue
                comps += 1
                frontier = np.zeros(m, dtype=bool)
                frontier[s] = True
                seen[s] = True
                while frontier.any():
                    nxt = adj[frontier].any(axis=0) & ~seen
                    seen |= nxt
                    frontier = nxt
            V += comps - 1

        for g in self._mib_arrays:
            shapes = {(round(float(pos[i, 2]), 4), round(float(pos[i, 3]), 4)) for i in g}
            V += len(shapes) - 1
        return V

    def _cost(self, pos: np.ndarray, x_right: float, y_top: float):
        hp = self._hpwl(pos)
        x_min = float(pos[:, 0].min())
        y_min = float(pos[:, 1].min())
        area = (x_right - x_min) * (y_top - y_min)
        V = self._violations(pos)
        c = (1.0 + 0.5 * ((hp / self.hp_ref - 1.0) + max(0.0, area / self.area_ref - 1.0))) \
            * math.exp(2.0 * self.v_weight * V / self.n_soft_den)
        return c, hp, area, V

    def _evaluate(self, cols):
        pos, x_right, y_top = self._layout(cols)
        c, hp, area, V = self._cost(pos, x_right, y_top)
        return c, pos

    # ------------------------------------------------------------------
    def _random_move(self, cols: List[List[int]]):
        """Mutate cols in place; return an undo callable, or None."""
        rng = self.rng
        C = len(cols)
        nonempty = [i for i in range(C) if cols[i]]
        if not nonempty:
            return None
        r = rng.random()
        if r < 0.55:
            sc = rng.choice(nonempty)
            si = rng.randrange(len(cols[sc]))
            k = cols[sc][si]
            u = self.units[k]
            if u.force == 'L':
                tc = 0
            elif u.force == 'R':
                tc = C - 1
            elif u.anchors and rng.random() < 0.6:
                # usually pull anchored clusters back over their preplaced
                # member, but sometimes let them escape (trading one grouping
                # violation can satisfy several boundary tags)
                ax = u.anchors[0][0] + u.anchors[0][2] * 0.5
                tc = self._col_of_x(ax, C)
            else:
                tc = None
                rr = rng.random()
                # connectivity-guided relocation: pull a unit toward the
                # column of a randomly sampled neighbor / pin
                if rr < 0.35 and len(self.eI):
                    e = rng.randrange(len(self.eI))
                    a, b = int(self.eI[e]), int(self.eJ[e])
                    if self.blk_unit[a] != k and self.blk_unit[b] == k:
                        a, b = b, a
                    if self.blk_unit[a] == k:
                        ku = self.blk_unit[b]
                        if ku >= 0:
                            tc = self._unit_col[ku]
                        else:
                            tc = self._col_of_x(self.lx[b] + self.rw[b] * 0.5, C)
                elif rr < 0.50 and len(self.pB):
                    e = rng.randrange(len(self.pB))
                    if self.blk_unit[int(self.pB[e])] == k:
                        tc = self._col_of_x(float(self.pX[e]), C)
                if tc is None:
                    tc = rng.randrange(C)
                # spread bottom/top-tagged units across columns (only the
                # first / last unit of a column can touch that edge)
                if u.hasB or u.hasT:
                    for _try in range(3):
                        clash = any((self.units[k2].hasB if u.hasB else self.units[k2].hasT)
                                    for k2 in cols[tc] if k2 != k)
                        if not clash:
                            break
                        tc = rng.randrange(C)
            if tc == sc and len(cols[sc]) == 1:
                return None
            cols[sc].pop(si)
            ti = rng.randint(0, len(cols[tc]))
            cols[tc].insert(ti, k)

            def undo():
                cols[tc].pop(ti)
                cols[sc].insert(si, k)
            return undo
        elif r < 0.80:
            if len(nonempty) < 2:
                return None
            c1, c2 = rng.sample(nonempty, 2)
            i1 = rng.randrange(len(cols[c1]))
            i2 = rng.randrange(len(cols[c2]))
            k1, k2 = cols[c1][i1], cols[c2][i2]
            u1, u2 = self.units[k1], self.units[k2]
            if u1.force == 'L' and c2 != 0:
                return None
            if u1.force == 'R' and c2 != C - 1:
                return None
            if u2.force == 'L' and c1 != 0:
                return None
            if u2.force == 'R' and c1 != C - 1:
                return None
            cols[c1][i1], cols[c2][i2] = k2, k1

            def undo():
                cols[c1][i1], cols[c2][i2] = k1, k2
            return undo
        elif r < 0.92:
            cands = [i for i in nonempty if len(cols[i]) >= 2]
            if not cands:
                return None
            c1 = rng.choice(cands)
            si = rng.randrange(len(cols[c1]))
            ti = rng.randrange(len(cols[c1]))
            if si == ti:
                return None
            k = cols[c1].pop(si)
            cols[c1].insert(ti, k)

            def undo():
                cols[c1].pop(ti)
                cols[c1].insert(si, k)
            return undo
        else:
            # internal reorder of a multi-part unit
            multi = [k for k, u in enumerate(self.units) if len(u.subgroups) >= 2
                     or any(len(sg) >= 2 for sg in u.subgroups)]
            if not multi:
                return None
            k = rng.choice(multi)
            u = self.units[k]
            if len(u.subgroups) >= 2 and rng.random() < 0.5:
                a = rng.randrange(len(u.subgroups) - 1)
                u.subgroups[a], u.subgroups[a + 1] = u.subgroups[a + 1], u.subgroups[a]
                self._refresh_unit(u)

                def undo():
                    u.subgroups[a], u.subgroups[a + 1] = u.subgroups[a + 1], u.subgroups[a]
                    self._refresh_unit(u)
                return undo
            sgs = [g for g in u.subgroups if len(g) >= 2]
            if not sgs:
                return None
            sg = rng.choice(sgs)
            a = rng.randrange(len(sg) - 1)
            sg[a], sg[a + 1] = sg[a + 1], sg[a]
            self._refresh_unit(u)

            def undo():
                sg[a], sg[a + 1] = sg[a + 1], sg[a]
                self._refresh_unit(u)
            return undo

    def _snapshot(self, cols):
        return ([list(c) for c in cols],
                {k: [sg[:] for sg in self.units[k].subgroups]
                 for k in range(len(self.units)) if len(self.units[k].blocks) > 1})

    def _restore(self, snap):
        cols = [list(c) for c in snap[0]]
        for k, sgs in snap[1].items():
            u = self.units[k]
            u.subgroups = [sg[:] for sg in sgs]
            self._refresh_unit(u)
        return cols

    def _anneal(self, cols, deadline, cur_cost, t0=0.02, t1=0.0008, recalibrate=False):
        rng = self.rng
        best_cost = cur_cost
        best = self._snapshot(cols)
        best_hp = None
        start = time.time()
        span = max(deadline - start, 1e-6)
        recal_at = [start + 0.25 * span, start + 0.55 * span] if recalibrate else []
        # Late-phase violation-weight annealing (PARTNER_VW_ANNEAL > current
        # v_weight enables): run most of the anneal at the true cost, then
        # reprice violations upward near the end to squeeze the survivors
        # out while there is still time to recover wirelength.  The weight
        # is restored (and best_cost re-scored) before returning, so
        # cross-restart selection stays weight-consistent.
        vw0 = self.v_weight
        try:
            vw_end = float(_os.environ.get("PARTNER_VW_ANNEAL", "0") or 0)
            vw_frac = float(_os.environ.get("PARTNER_VW_ANNEAL_AT", "0.7"))
        except ValueError:
            vw_end, vw_frac = 0.0, 0.7
        vw_at = (start + vw_frac * span) if vw_end > vw0 else None
        # Stall early stop (PARTNER_SA_STALL_STOP): break out of this chain
        # once best_cost has not improved by >= stall_eps (relative) within
        # the last stall_win seconds. The check hangs on the existing
        # per-outer-loop time read, so it adds no per-move cost and consumes
        # no rng. Disabled (bit-identical) when _stall_frac == 0.
        self._anneal_stalled = False
        stall_win = self._stall_frac * span if self._stall_frac > 0.0 else None
        if stall_win is not None and self._early_exit:
            # keep one productive outer iteration inside the window even when
            # the case budget is tiny (0.2 s/case endgame)
            stall_win = max(stall_win, early_exit_min_window())
        stall_eps = self.stall_eps
        stall_ref_cost = best_cost   # best_cost at the last window reset
        stall_ref_time = start
        while True:
            now = time.time()
            if now >= deadline:
                break
            if vw_at is not None and now >= vw_at:
                vw_at = None
                pos, xr, yt = self._layout(cols)
                if self._violations(pos) > 0:
                    self.v_weight = vw_end
                    cur_cost, _, _, _ = self._cost(pos, xr, yt)
                    bcols = self._restore(best)
                    bpos, bxr, byt = self._layout(bcols)
                    best_cost, _, _, _ = self._cost(bpos, bxr, byt)
                    if best_cost < cur_cost:
                        cols = bcols
                        cur_cost = best_cost
                    # repricing violations rescales the cost (best_cost jumps
                    # UP); re-anchor the stall window so the jump is not read
                    # as a stall and does not kill the very phase this feature
                    # exists to run.
                    stall_ref_cost = best_cost
                    stall_ref_time = now
            if recal_at and now >= recal_at[0]:
                recal_at.pop(0)
                # tighten the HPWL normalizer toward the (unknown) baseline so
                # the annealer keeps real pressure on wirelength
                pos, xr, yt = self._layout(cols)
                hp_now = self._hpwl(pos)
                self.hp_ref = max(hp_now * 0.75, 1e-6)
                cur_cost, _, _, _ = self._cost(pos, xr, yt)
                bcols = self._restore(best)
                bpos, bxr, byt = self._layout(bcols)
                best_cost, _, _, _ = self._cost(bpos, bxr, byt)
                if best_cost < cur_cost:
                    cols = bcols
                    cur_cost = best_cost
                # a recalibration rescales the cost; re-anchor the stall
                # window so the shift is not misread as an improvement or a
                # stall.
                stall_ref_cost = best_cost
                stall_ref_time = now
            if stall_win is not None:
                # improvement is measured from the last reset; a sign-safe
                # absolute test avoids the relative form breaking near zero.
                if stall_ref_cost - best_cost >= stall_eps * abs(stall_ref_cost):
                    stall_ref_cost = best_cost
                    stall_ref_time = now
                elif now - stall_ref_time >= stall_win:
                    self._anneal_stalled = True
                    break
            frac = min((now - start) / span, 1.0)
            if self._sa_schedule == "fastsa":
                T = _fastsa_temp(frac, t0, t1, self._fastsa_k,
                                 self._fastsa_c, self._fastsa_steps)
            else:
                T = t0 * (t1 / t0) ** frac
            for _ in range(24):
                undo = self._random_move(cols)
                if undo is None:
                    continue
                new_cost, _pos = self._evaluate(cols)
                if new_cost <= cur_cost or rng.random() < math.exp((cur_cost - new_cost) / T):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = self._snapshot(cols)
                else:
                    undo()
        if self.v_weight != vw0:
            # restore the true-cost weight and re-score the returned best so
            # callers compare restarts on a consistent objective
            self.v_weight = vw0
            bcols = self._restore(best)
            bpos, bxr, byt = self._layout(bcols)
            best_cost, _, _, _ = self._cost(bpos, bxr, byt)
        return best, best_cost

    # ------------------------------------------------------------------
    def locked_only(self) -> bool:
        return not self.units

    def locked_positions(self) -> List[Rect]:
        return [(self.lx[i], self.ly[i], self.rw[i], self.rh[i]) for i in range(self.n)]

    def prepare(self) -> float:
        avg_area = self.total_area / max(self.n, 1)
        target_w = max(math.sqrt(avg_area), 2.0)
        self.C0 = max(2, min(18, int(round(self.W_est / target_w))))
        cols = self._init_columns(self.C0)
        c0, _ = self._evaluate_bootstrap(cols)
        self._cols = cols
        self._cost0 = c0
        self._best_probe = None
        return c0

    def probe(self, t_each: float) -> float:
        cands = sorted({max(2, self.C0 - 1), self.C0, min(18, self.C0 + 1)})
        results = []
        self._probe_unspent = 0.0
        for C in cands:
            pc = self._init_columns(C)
            pcost, _ = self._evaluate(pc)
            chain_end = time.time() + t_each
            snap, bcost = self._anneal(pc, chain_end, pcost,
                                       t0=0.06, t1=0.01)
            if self._early_exit:
                self._probe_unspent += max(0.0, chain_end - time.time())
            results.append((bcost, snap))
        results.sort(key=lambda t: t[0])
        self._best_probe = results[0]
        return results[0][0]

    def finish(self, deadline: float, max_runs: int = 2) -> List[Rect]:
        n = self.n
        if self._best_probe is not None:
            start_snap = self._best_probe[1]
        else:
            start_snap = self._snapshot(self._cols)
        rem = max(deadline - time.time(), 0.0)
        import os as _os
        _rf = float(_os.environ.get('REFINE_FRAC', 0.30))
        _rc = float(_os.environ.get('REFINE_CAP', 7.0))
        refine_t = min(_rf * rem, _rc) if rem > 1.0 else 0.0
        polish_t = min(0.28 * max(rem - refine_t, 0.0), 6.5)
        t_end = deadline - polish_t - refine_t
        now = time.time()
        # Minimum anneal span.  The historical floor is a flat 0.1 s, which
        # is longer than the WHOLE case budget at the goal operating point
        # (0.05 s): every pool worker then annealed ~100 ms regardless of
        # the deadline it was handed, and that -- not marshalling, not
        # dispatch -- is the measured 0.12-0.17 s per-case wall floor.
        # PARTNER_FAST_SETUP clamps the floor to the span this chain was
        # actually given (see `fast_setup_on`); with the flag off the clamp
        # cannot bind, because the floor only ever mattered when the planned
        # span was already below 0.1 s.
        _span_floor = 0.1
        if fast_setup_on():
            _span_floor = min(_span_floor, max(t_end - now, 0.0))
        total = max(t_end - now, _span_floor)
        runs = 2 if (total > 6.0 and max_runs >= 2) else 1
        snaps = []
        # EARLY_EXIT clawback: every stage keeps its PLANNED share and returns
        # the rest to the caller.  `saved` accumulates the unspent tail of the
        # anneal chains; with the flag off it stays 0.0 and every expression
        # below is bit-identical to the pre-port fork.
        saved = 0.0
        for r in range(runs):
            cols_r = self._restore(start_snap)
            c_r, _ = self._evaluate(cols_r)
            sub_end = now + total * (r + 1) / runs - saved
            snap, _bc = self._anneal(cols_r, sub_end, c_r, recalibrate=True)
            if self._early_exit:
                saved += max(0.0, sub_end - time.time())
            snaps.append(snap)
        if self._early_exit and saved > 0.0:
            deadline = max(time.time(), deadline - saved)
        # pick the better run under one common (final) normalizer
        best_cols = None
        best_c = None
        for snap in snaps:
            cols_c = self._restore(snap)
            c_c, _ = self._evaluate(cols_c)
            if best_c is None or c_c < best_c:
                best_c = c_c
                best_cols = [list(c) for c in cols_c]
                best_snap = snap
        cols = self._restore(best_snap)
        polish_end = deadline - refine_t
        cols = self._greedy_polish(cols, best_c, polish_end)
        # `_greedy_polish` is already convergence-bounded (`while improved`),
        # so with EARLY_EXIT off its unspent tail silently inflates the
        # stage-2 refiner slice; claw it back instead.
        if self._early_exit:
            deadline = max(time.time(),
                           deadline - max(0.0, polish_end - time.time()))
        pos, x_right, y_top = self._layout(cols)
        # stage-2 continuous refinement (constraint-graph / weighted-median);
        # accepted only if the shared proxy cost does not regress
        if refine_t > 0.05 and time.time() < deadline - 0.1:
            try:
                from layout_refiner import refine_positions
                pos2 = refine_positions(self, pos, deadline - 0.05,
                                        seed=self.rng.randrange(1 << 30))
                if pos2 is not None:
                    c_old, _, _, _ = self._cost(pos, x_right, y_top)
                    xr2 = float((pos2[:, 0] + pos2[:, 2]).max())
                    yt2 = float((pos2[:, 1] + pos2[:, 3]).max())
                    c_new, _, _, _ = self._cost(pos2, xr2, yt2)
                    if c_new <= c_old:
                        pos, x_right, y_top = pos2, xr2, yt2
            except Exception:
                pass
        hp = self._hpwl(pos)
        V = self._violations(pos)
        area = (x_right - float(pos[:, 0].min())) * (y_top - float(pos[:, 1].min()))
        self.final_metrics = (hp, area, V)
        out = [(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]), float(pos[i, 3]))
               for i in range(n)]
        return _ensure_no_overlap(out, [self.kind[i] == 2 for i in range(n)])

    def run(self) -> List[Rect]:
        if self.locked_only():
            return self.locked_positions()
        self.prepare()
        budget = self.deadline - time.time()
        deadline = self.deadline
        if budget > 2.5:
            self.probe(min(0.12 * budget, 2.0) / 3)
            if self._early_exit and self._probe_unspent > 0.0:
                deadline = max(time.time(), deadline - self._probe_unspent)
        return self.finish(deadline)

    def _spread_tagged(self, cols, cur_cost, deadline):
        """Move surplus bottom/top-tagged units (only the first / last unit
        of a column can touch that edge) into columns that lack one."""
        for tag in ('B', 'T'):
            for sc in range(len(cols)):
                if time.time() >= deadline:
                    return cur_cost
                tagged = [k for k in cols[sc]
                          if (self.units[k].hasB if tag == 'B' else self.units[k].hasT)]
                for k in tagged[1:]:
                    u = self.units[k]
                    si = cols[sc].index(k)
                    best = None
                    for tc in range(len(cols)):
                        if tc == sc:
                            continue
                        if u.force == 'L' and tc != 0:
                            continue
                        if u.force == 'R' and tc != len(cols) - 1:
                            continue
                        if any((self.units[k2].hasB if tag == 'B' else self.units[k2].hasT)
                               for k2 in cols[tc]):
                            continue
                        cols[sc].pop(si)
                        ti = 0 if tag == 'B' else len(cols[tc])
                        cols[tc].insert(ti, k)
                        c, _ = self._evaluate(cols)
                        if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                            best = (c, tc, ti)
                        cols[tc].pop(ti)
                        cols[sc].insert(si, k)
                    if best is not None:
                        c, tc, ti = best
                        cols[sc].pop(si)
                        cols[tc].insert(ti, k)
                        cur_cost = c
        return cur_cost

    def _repair_boundary(self, cols, cur_cost, deadline):
        """Relocate units whose bottom/top-tagged blocks miss their edge."""
        pos, _xr, _yt = self._layout(cols)
        py0 = pos[:, 1]
        py1 = py0 + pos[:, 3]
        y_min = py0.min()
        y_max = py1.max()
        bad_units = []
        for i, code in zip(self._bnd_idx, self._bnd_codes):
            k = self.blk_unit[int(i)]
            if k < 0:
                continue
            if (code & 8) and abs(py0[i] - y_min) >= 1e-6:
                bad_units.append((k, 0))
            elif (code & 4) and abs(py1[i] - y_max) >= 1e-6:
                bad_units.append((k, -1))
        seen = set()
        for k, where in bad_units:
            if k in seen or time.time() >= deadline:
                continue
            seen.add(k)
            u = self.units[k]
            sc = self._unit_col[k]
            if sc >= len(cols):
                continue
            try:
                si = cols[sc].index(k)
            except ValueError:
                continue
            best = None
            for tc in range(len(cols)):
                if u.force == 'L' and tc != 0:
                    continue
                if u.force == 'R' and tc != len(cols) - 1:
                    continue
                cols[sc].pop(si)
                ti = 0 if where == 0 else len(cols[tc])
                cols[tc].insert(ti, k)
                c, _ = self._evaluate(cols)
                cols[tc].pop(ti)
                cols[sc].insert(si, k)
                if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                    best = (c, tc, ti)
            if best is not None:
                c, tc, ti = best
                cols[sc].pop(si)
                cols[tc].insert(ti, k)
                cur_cost = c
        return cur_cost

    def _order_pass(self, cols, cur_cost, deadline):
        """Reorder each column by the connectivity-weighted target y of its
        units, and try swapping adjacent columns; keep strict improvements."""
        pos, _xr, _yt = self._layout(cols)
        n = self.n
        cy = pos[:, 1] + pos[:, 3] * 0.5
        num = np.zeros(n)
        den = np.zeros(n)
        if len(self.eI):
            np.add.at(num, self.eI, self.eW * cy[self.eJ])
            np.add.at(den, self.eI, self.eW)
            np.add.at(num, self.eJ, self.eW * cy[self.eI])
            np.add.at(den, self.eJ, self.eW)
        if len(self.pB):
            np.add.at(num, self.pB, self.pW * self.pY)
            np.add.at(den, self.pB, self.pW)
        pull = np.where(den > 1e-12, num / np.maximum(den, 1e-12), cy)
        for ci in range(len(cols)):
            if len(cols[ci]) < 3 or time.time() >= deadline:
                continue
            def tkey(k):
                u = self.units[k]
                s = 0.0
                a = 0.0
                for i in u.blocks:
                    ai = self.areas[i]
                    s += ai * pull[i]
                    a += ai
                return s / max(a, 1e-9)
            new_order = sorted(cols[ci], key=tkey)
            if new_order != cols[ci]:
                old = cols[ci][:]
                cols[ci][:] = new_order
                c, _ = self._evaluate(cols)
                if c < cur_cost - 1e-9:
                    cur_cost = c
                else:
                    cols[ci][:] = old
        for ci in range(len(cols) - 1):
            if time.time() >= deadline:
                break
            # never displace edge-forced units from their edge column
            if any(self.units[k].force for k in cols[ci]) or \
                    any(self.units[k].force for k in cols[ci + 1]):
                continue
            cols[ci], cols[ci + 1] = cols[ci + 1], cols[ci]
            c, _ = self._evaluate(cols)
            if c < cur_cost - 1e-9:
                cur_cost = c
            else:
                cols[ci], cols[ci + 1] = cols[ci + 1], cols[ci]
        return cur_cost

    def _reduce_overflow(self, cols, cur_cost, deadline):
        """When columns rise past the (possibly pinned) frame height, move
        their topmost unit somewhere cheaper."""
        units = self.units
        for _round in range(6):
            if time.time() >= deadline:
                break
            pos, _xr, yt = self._layout(cols)
            if yt <= self.H * 1.0001:
                break
            moved = False
            tops = []
            for ci, ulist in enumerate(cols):
                if not ulist:
                    continue
                top = max(pos[i, 1] + pos[i, 3] for k in ulist for i in units[k].blocks)
                if top > self.H * 1.0001:
                    tops.append((top, ci))
            tops.sort(reverse=True)
            for _t, sc in tops[:2]:
                ktop = max(cols[sc],
                           key=lambda k: max(pos[i, 1] + pos[i, 3] for i in units[k].blocks))
                si = cols[sc].index(ktop)
                u = units[ktop]
                best = None
                for tc in range(len(cols)):
                    if tc == sc:
                        continue
                    if u.force == 'L' and tc != 0:
                        continue
                    if u.force == 'R' and tc != len(cols) - 1:
                        continue
                    for ti in (0, len(cols[tc])):
                        cols[sc].pop(si)
                        ti2 = min(ti, len(cols[tc]))
                        cols[tc].insert(ti2, ktop)
                        c, _ = self._evaluate(cols)
                        cols[tc].pop(ti2)
                        cols[sc].insert(si, ktop)
                        if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                            best = (c, tc, ti2)
                if best is not None:
                    c, tc, ti2 = best
                    cols[sc].pop(si)
                    cols[tc].insert(ti2, ktop)
                    cur_cost = c
                    moved = True
            if not moved:
                break
        return cur_cost

    def _greedy_polish(self, cols, cur_cost, deadline):
        """Final best-improvement sweeps: for every unit try a handful of
        (column, position) relocations and keep strict improvements."""
        cur_cost, _ = self._evaluate(cols)
        cur_cost = self._reduce_overflow(cols, cur_cost, deadline)
        cur_cost = self._repair_boundary(cols, cur_cost, deadline)
        cur_cost = self._spread_tagged(cols, cur_cost, deadline)
        for _ in range(3):
            before = cur_cost
            cur_cost = self._order_pass(cols, cur_cost, deadline)
            if cur_cost > before - 1e-9 or time.time() >= deadline:
                break
        rng = self.rng
        order = list(range(len(self.units)))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(order)
            for k in order:
                if time.time() >= deadline:
                    break
                u = self.units[k]
                sc = self._unit_col[k]
                if sc >= len(cols):
                    continue
                try:
                    si = cols[sc].index(k)
                except ValueError:
                    continue
                best = None
                for tc in range(len(cols)):
                    if u.force == 'L' and tc != 0:
                        continue
                    if u.force == 'R' and tc != len(cols) - 1:
                        continue
                    ncand = len(cols[tc]) - (1 if tc == sc else 0)
                    for ti in {0, ncand // 2, ncand}:
                        if tc == sc and ti == si:
                            continue
                        cols[sc].pop(si)
                        ti2 = min(ti, len(cols[tc]))
                        cols[tc].insert(ti2, k)
                        c, _ = self._evaluate(cols)
                        cols[tc].pop(ti2)
                        cols[sc].insert(si, k)
                        if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                            best = (c, tc, ti2)
                if best is not None:
                    c, tc, ti2 = best
                    cols[sc].pop(si)
                    cols[tc].insert(ti2, k)
                    cur_cost = c
                    improved = True
        return cols

    def _evaluate_bootstrap(self, cols):
        """First evaluation; establishes the HPWL normalizer."""
        pos, x_right, y_top = self._layout(cols)
        self.hp_ref = max(self._hpwl(pos), 1e-6)
        c, hp, area, V = self._cost(pos, x_right, y_top)
        return c, pos


# =============================================================================
# Safety net
# =============================================================================
def _ensure_no_overlap(rects: List[Rect], locked: List[bool]) -> List[Rect]:
    """Verify and, if needed, nudge apart residual overlaps (position-only
    moves, so hard shape/area guarantees are preserved). No-op in the normal
    path — the column construction is overlap-free."""
    n = len(rects)
    out = [list(r) for r in rects]
    for _ in range(50):
        changed = False
        for i in range(n):
            x1, y1, w1, h1 = out[i]
            for j in range(i + 1, n):
                x2, y2, w2, h2 = out[j]
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if ox <= 1e-6 or oy <= 1e-6:
                    continue
                move = i if locked[j] else j
                if locked[move]:
                    move = j if move == i else i
                    if locked[move]:
                        continue
                mx, my, mw, mh = out[move]
                if ox <= oy:
                    out[move] = [mx + ox + 1e-6, my, mw, mh] if move == j else [mx - ox - 1e-6, my, mw, mh]
                else:
                    out[move] = [mx, my + oy + 1e-6, mw, mh] if move == j else [mx, my - oy - 1e-6, mw, mh]
                x1, y1, w1, h1 = out[i]
                changed = True
        if not changed:
            break
    return [tuple(r) for r in out]


# =============================================================================
# Parallel restart pool
# =============================================================================
_POOL = None
_POOL_SIZE = 0
_POOL_READY = False


def _worker_init():
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _worker_ping(_i):
    time.sleep(0.2)
    return True


def init_worker_pool(n_workers: int = 8, warmup_timeout: float = 180.0):
    """Create the restart pool once and wait until every worker has finished
    importing (call from optimizer __init__ so the spawn/import cost is not
    charged to any test case)."""
    global _POOL, _POOL_SIZE, _POOL_READY
    if _POOL is not None or n_workers < 2:
        return
    try:
        import multiprocessing as mp
        # Warm the opt-in numba refine kernel in the parent BEFORE forking:
        # children inherit the compiled functions, so no worker pays the
        # ~0.17s first-call cache load inside a case span (at 0.4s worker
        # spans that would be ~40% of the slice).  No-op when the flag is off.
        if _os.environ.get("PARTNER_REFINE_KERNEL", "") == "numba":
            try:
                from refine_numeric_kernel import warm_process
                warm_process()
            except Exception:
                pass
        # fork: no re-import of __main__, no per-worker torch import cost.
        # Workers never touch CUDA, so forking a CUDA-initialized parent is
        # safe (same pattern as torch DataLoader workers).
        ctx = mp.get_context('fork')
        _POOL = ctx.Pool(n_workers, initializer=_worker_init)
        _POOL_SIZE = n_workers
        r = _POOL.map_async(_worker_ping, range(n_workers * 3), chunksize=1)
        r.get(timeout=warmup_timeout)
        _POOL_READY = True
    except Exception:
        _shutdown_pool()


def _shutdown_pool():
    global _POOL, _POOL_SIZE, _POOL_READY
    try:
        if _POOL is not None:
            _POOL.terminate()
    except Exception:
        pass
    _POOL = None
    _POOL_SIZE = 0
    _POOL_READY = False


def _perimeter_pack(opt, seed_rects, max_aspect=8.0):
    """Explicit perimeter packing for boundary-coded blocks.

    The worst validation cases are boundary-dense and wall-saturated
    (25-30% of blocks carry wall codes, wall demand at 85-95% of
    capacity); the legalization ladder drops tag pins there and the SA
    move set cannot recover.  This solves each wall line explicitly:
    corner-coded blocks sit exactly in their corners, single-wall blocks
    pack along the wall in seed order (net pull), soft blocks shrink
    along the wall (exact area, aspect-capped) when a line overflows,
    and the frame grows minimally when shrinking is not enough (unless a
    preplaced boundary block pins that dimension).

    Returns ({i: (x, y, w, h)}, W, H, n_unpacked) or None when the
    instance has no useful perimeter structure.
    """
    n = opt.n
    movable_bnd = [(i, opt.boundary[i]) for i in range(n)
                   if opt.boundary[i] > 0 and opt.kind[i] != 2]
    if len(movable_bnd) < 4:
        return None
    W = float(opt.W_est)
    H = float(opt.H)
    # preplaced boundary blocks reveal the intended frame exactly: snap to
    # the majority extent so their own wall codes stay satisfiable (an
    # off-majority preplaced code is structurally unsatisfiable and is a
    # violation floor baked into the instance)
    fx = [float(opt.lx[i] + opt.rw[i]) for i in range(n)
          if opt.kind[i] == 2 and (opt.boundary[i] & 2)]
    fy = [float(opt.ly[i] + opt.rh[i]) for i in range(n)
          if opt.kind[i] == 2 and (opt.boundary[i] & 4)]
    if fx:
        W = max(fx)
    if fy:
        H = max(fy)
    for (x, y, w, h) in opt.locked_rects:
        W = max(W, x + w)
        H = max(H, y + h)
    h_pinned = bool(fy)
    w_pinned = bool(fx)

    def shape(i):
        if opt.kind[i] == 1:
            return float(opt.rw[i]), float(opt.rh[i])
        x, y, w, h = seed_rects[i]
        if w > 0 and h > 0 and abs(w * h - opt.areas[i]) < 0.5 * opt.areas[i]:
            return float(w), float(h)
        s = math.sqrt(opt.areas[i])
        return s, s

    def seed_c(i, axis):
        x, y, w, h = seed_rects[i]
        return float(x + 0.5 * w) if axis == 0 else float(y + 0.5 * h)

    # --- corners: at most one block per corner (largest area wins) -----
    corner_of = {}   # (xbit, ybit) -> block
    unpacked = 0
    for i, code in movable_bnd:
        xb = code & 3
        yb = code & 12
        if xb in (1, 2) and yb in (4, 8):
            key = (xb, yb)
            j = corner_of.get(key)
            if j is None or opt.areas[i] > opt.areas[j]:
                if j is not None:
                    unpacked += 1
                corner_of[key] = i
            else:
                unpacked += 1
    corner_ids = set(corner_of.values())

    # --- wall membership -------------------------------------------------
    def wall_items(bit):
        out = []
        for i, code in movable_bnd:
            if i in corner_ids:
                continue
            if (code & 3) == bit or (code & 12) == bit:
                # pure single-wall blocks only (odd double-x/-y codes are
                # left to the generic machinery)
                if code in (1, 2, 4, 8):
                    out.append(i)
        return out

    walls = {'L': wall_items(1), 'R': wall_items(2),
             'T': wall_items(4), 'B': wall_items(8)}

    def along(i, vertical):
        w, h = shape(i)
        return h if vertical else w

    def min_along(i, vertical):
        if opt.kind[i] == 1:
            return along(i, vertical)
        return math.sqrt(opt.areas[i] / max_aspect)

    # --- required span per wall (corners occupy both their walls) -------
    def corner_span(wall):
        s = 0.0
        for (xb, yb), i in corner_of.items():
            w, h = shape(i)
            if wall == 'L' and xb == 1:
                s += h
            elif wall == 'R' and xb == 2:
                s += h
            elif wall == 'T' and yb == 4:
                s += w
            elif wall == 'B' and yb == 8:
                s += w
        return s

    def locked_span(wall):
        s = 0.0
        eps = 1e-6
        for i in range(n):
            if opt.kind[i] != 2:
                continue
            x, y, w, h = opt.lx[i], opt.ly[i], opt.rw[i], opt.rh[i]
            if wall == 'L' and abs(x) < eps:
                s += h
            elif wall == 'R' and abs(x + w - W) < eps:
                s += h
            elif wall == 'B' and abs(y) < eps:
                s += w
            elif wall == 'T' and abs(y + h - H) < eps:
                s += w
        return s

    def line_need(wall, shrink):
        vertical = wall in ('L', 'R')
        tot = corner_span(wall) + locked_span(wall)
        for i in walls[wall]:
            tot += (min_along(i, vertical) if shrink
                    else along(i, vertical))
        return tot

    # grow H for L/R demand, W for T/B demand (respect pins)
    need_h = max(line_need('L', True), line_need('R', True))
    if need_h > H + 1e-9:
        if h_pinned:
            return None
        H = need_h * 1.001
    need_w = max(line_need('T', True), line_need('B', True))
    if need_w > W + 1e-9:
        if w_pinned:
            return None
        W = need_w * 1.001

    # --- place each wall line -------------------------------------------
    pack = {}

    def place_wall(wall):
        vertical = wall in ('L', 'R')
        span = H if vertical else W
        # fixed occupations: corners + locked-on-wall
        occ = []
        for (xb, yb), i in corner_of.items():
            w, h = shape(i)
            on = ((wall == 'L' and xb == 1) or (wall == 'R' and xb == 2)
                  or (wall == 'T' and yb == 4) or (wall == 'B' and yb == 8))
            if not on:
                continue
            a = h if vertical else w
            lo = 0.0 if (yb == 8 if vertical else xb == 1) else span - a
            occ.append((lo, lo + a))
            if i not in pack:
                x = (0.0 if xb == 1 else W - w)
                y = (0.0 if yb == 8 else H - h)
                pack[i] = (x, y, w, h)
        # anything intruding into the wall band blocks that stretch of the
        # line: locked blocks AND ring blocks packed on earlier walls
        # (corner regions overlap between adjacent walls)
        depth = 0.0
        for i in walls[wall]:
            w0, h0 = shape(i)
            depth = max(depth, w0 if vertical else h0)
        obstacles = []
        for j in range(n):
            if opt.kind[j] == 2:
                obstacles.append((opt.lx[j], opt.ly[j],
                                  opt.rw[j], opt.rh[j]))
            elif j in pack and j not in walls[wall]:
                obstacles.append(pack[j])
        for (x, y, w, h) in obstacles:
            if vertical:
                lo_p, hi_p = (0.0, depth) if wall == 'L' else (W - depth, W)
                if min(x + w, hi_p) - max(x, lo_p) > 1e-9:
                    occ.append((y, y + h))
            else:
                lo_p, hi_p = (0.0, depth) if wall == 'B' else (H - depth, H)
                if min(y + h, hi_p) - max(y, lo_p) > 1e-9:
                    occ.append((x, x + w))
        occ.sort()
        segs = []
        cur = 0.0
        for s, e in occ:
            if s > cur + 1e-12:
                segs.append([cur, min(s, span)])
            cur = max(cur, e)
        if cur < span - 1e-12:
            segs.append([cur, span])
        items = sorted(walls[wall], key=lambda i: seed_c(i, 1 if vertical else 0))
        free = sum(e - s for s, e in segs)
        base = sum(along(i, vertical) for i in items)
        ratio = 1.0
        if base > free - 1e-9:
            lo = sum(min_along(i, vertical) for i in items)
            if lo > free + 1e-9:
                if _os.environ.get("PERI_DEBUG"):
                    print(f"[peri] wall {wall}: min sizes {lo:.1f} > free "
                          f"{free:.1f}", flush=True)
                return False
            # proportional shrink of the soft slack
            slack = base - lo
            ratio = 0.0 if slack <= 1e-12 else (free * 0.999 - lo) / slack

        def eff_of(i, r):
            a = along(i, vertical)
            m = min_along(i, vertical)
            return m + (a - m) * max(0.0, min(r, 1.0))

        def first_fit(r):
            si = 0
            cursor = segs[0][0] if segs else 0.0
            got = {}
            for i in items:
                a = eff_of(i, r)
                while si < len(segs) and cursor + a > segs[si][1] + 1e-9:
                    si += 1
                    if si < len(segs):
                        cursor = segs[si][0]
                if si >= len(segs):
                    return None
                got[i] = (si, cursor)
                cursor += a
            return got

        # fragmentation can defeat seed-order first-fit even when the
        # totals fit; retry down a shrink ladder (sizes floor at the
        # aspect-capped minimum)
        placed = None
        r_used = min(ratio, 1.0)
        for r in (r_used, 0.92 * r_used, 0.82 * r_used, 0.7 * r_used,
                  0.5 * r_used, 0.0):
            placed = first_fit(r)
            if placed is not None:
                r_used = r
                break
        if placed is None:
            if _os.environ.get("PERI_DEBUG"):
                print(f"[peri] wall {wall}: first-fit fail "
                      f"segs={[(round(s, 1), round(e, 1)) for s, e in segs]}",
                      flush=True)
            return False

        def eff(i):
            return eff_of(i, r_used)
        by_seg = {}
        for i, (sj, pos) in placed.items():
            by_seg.setdefault(sj, []).append(i)
        for sj, js in by_seg.items():
            js.sort(key=lambda i: placed[i][1])
            limit = segs[sj][1]
            for i in reversed(js):
                a = eff(i)
                desired = seed_c(i, 1 if vertical else 0) - 0.5 * a
                pos = min(max(desired, placed[i][1]), limit - a)
                placed[i] = (sj, pos)
                limit = pos
        for i in items:
            a = eff(i)
            if opt.kind[i] == 0:
                perp = opt.areas[i] / a
            else:
                w, h = shape(i)
                perp = w if vertical else h
            pos = placed[i][1]
            if wall == 'L':
                pack[i] = (0.0, pos, perp, a)
            elif wall == 'R':
                pack[i] = (W - perp, pos, perp, a)
            elif wall == 'B':
                pack[i] = (pos, 0.0, a, perp)
            else:
                pack[i] = (pos, H - perp, a, perp)
        return True

    for wall in ('L', 'R', 'B', 'T'):
        if not place_wall(wall):
            return None
    return pack, W, H, unpacked


def _perimeter_seed_layouts(opt, seed_rects, variants=2):
    """Composed seed layouts for the direct-refine channel: the perimeter
    ring solved exactly by _perimeter_pack, the interior shelf-packed into
    the inner rectangle (seed order = net pull).  The compositions are
    imperfect on purpose — refine_prediction's tag seeding pins the
    already-satisfied wall blocks and its min-displacement legalization
    cleans the interior, which is exactly the job it does for
    direct-model predictions."""
    got = _perimeter_pack(opt, seed_rects)
    if got is None:
        return []
    pk, W, H, _un = got
    n = opt.n
    base = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        if opt.kind[i] == 2:
            base[i] = (opt.lx[i], opt.ly[i], opt.rw[i], opt.rh[i])
    for i, r in pk.items():
        base[i] = r
    # inner rectangle: clear of every ring block's depth on its wall
    dL = dR = dB = dT = 0.0
    for i, (x, y, w, h) in pk.items():
        if x <= 1e-9:
            dL = max(dL, w)
        if x + w >= W - 1e-6:
            dR = max(dR, w)
        if y <= 1e-9:
            dB = max(dB, h)
        if y + h >= H - 1e-6:
            dT = max(dT, h)
    ix0, ix1 = dL, W - dR
    iy0, iy1 = dB, H - dT
    if ix1 - ix0 < 1.0 or iy1 - iy0 < 1.0:
        return []
    interior = [i for i in range(n) if opt.kind[i] != 2 and i not in pk]

    def seed_c(i, axis):
        x, y, w, h = seed_rects[i]
        return float(x + 0.5 * w) if axis == 0 else float(y + 0.5 * h)

    def shp(i):
        x, y, w, h = seed_rects[i]
        if w > 0 and h > 0 and abs(w * h - opt.areas[i]) < 0.5 * opt.areas[i]:
            return float(w), float(h)
        s = math.sqrt(opt.areas[i])
        return s, s

    # cluster-aware entities: movable cluster members travel together as a
    # meta-square (refine_prediction's cluster reassembly only needs the
    # members NEAR each other to rebuild exact contact); groups that have a
    # locked/ring member are anchored at that member instead of shelf flow
    interior_set = set(interior)
    entities = []       # (members, area, cx, cy)
    anchored = []       # (members, area, ax, ay)
    used = set()
    for idxs in opt.cluster_groups.values():
        mem = [i for i in idxs if i in interior_set]
        if len(mem) < 2:
            continue
        used.update(mem)
        area = sum(opt.areas[i] for i in mem)
        mates = [i for i in idxs if i not in interior_set]
        if mates:
            ax = sum(float(base[i, 0] + 0.5 * base[i, 2]) for i in mates) / len(mates)
            ay = sum(float(base[i, 1] + 0.5 * base[i, 3]) for i in mates) / len(mates)
            anchored.append((mem, area, ax, ay))
        else:
            cx = sum(seed_c(i, 0) for i in mem) / len(mem)
            cy = sum(seed_c(i, 1) for i in mem) / len(mem)
            entities.append((mem, area, cx, cy))
    for i in interior:
        if i not in used:
            entities.append(([i], opt.areas[i], seed_c(i, 0), seed_c(i, 1)))

    def fill_members(P, mem, x0, y0, side):
        """Row-fill members inside a meta-square at (x0, y0)."""
        order = sorted(mem, key=lambda i: (seed_c(i, 1), seed_c(i, 0)))
        cx, cy, row_h = x0, y0, 0.0
        for i in order:
            w, h = shp(i)
            if cx + w > x0 + side + 1e-9 and cx > x0 + 1e-9:
                cy += row_h
                cx, row_h = x0, 0.0
            P[i] = (cx, cy, w, h)
            cx += w
            row_h = max(row_h, h)

    outs = []
    for v in range(variants):
        P = base.copy()
        axis = v % 2
        items = sorted(entities, key=lambda e: (e[2 + axis], e[3 - axis]))
        cx, cy, run = ix0, iy0, 0.0
        for mem, area, _sx, _sy in items:
            side = math.sqrt(max(area, 1e-9)) * (1.15 if len(mem) > 1 else 1.0)
            if len(mem) == 1:
                side_w, side_h = shp(mem[0])
            else:
                side_w = side_h = side
            if axis == 0:   # horizontal shelves
                if cx + side_w > ix1 + 1e-9 and cx > ix0 + 1e-9:
                    cy += run
                    cx, run = ix0, 0.0
                px, py = cx, cy
                cx += side_w
                run = max(run, side_h)
            else:           # vertical shelves
                if cy + side_h > iy1 + 1e-9 and cy > iy0 + 1e-9:
                    cx += run
                    cy, run = iy0, 0.0
                px, py = cx, cy
                cy += side_h
                run = max(run, side_w)
            if len(mem) == 1:
                w, h = shp(mem[0])
                P[mem[0]] = (px, py, w, h)
            else:
                fill_members(P, mem, px, py, side)
        for mem, area, ax, ay in anchored:
            side = math.sqrt(max(area, 1e-9)) * 1.15
            px = min(max(ax - 0.5 * side, ix0), max(ix1 - side, ix0))
            py = min(max(ay - 0.5 * side, iy0), max(iy1 - side, iy0))
            fill_members(P, mem, px, py, side)
        outs.append(P)
    return outs


def _w_star_from_tags(opt) -> Optional[float]:
    """Frame width implied by the L/R-tagged PREPLACED blocks, or None.

    Reusable instance statistic, never a case id: the trigger is "a preplaced
    block carries a right-wall tag" (bit 2).  A right-tagged preplaced block
    can only satisfy its tag if the frame's right edge is its own right edge,
    so max(x + w) over the locked rects IS the intended frame width -- the
    max is over ALL locked rects, not just the tagged one, because a locked
    block sticking out further would move the bbox edge past the tag.
    The left side is the packing origin (x = 0) unless a locked rect starts
    left of it, which is the symmetric L case.

    Returns None when the tag is absent or the derived width is degenerate."""
    if not opt.locked_rects:
        return None
    has_r = any(opt.kind[i] == 2 and (int(opt.boundary[i]) & 2)
                for i in range(opt.n))
    if not has_r:
        return None
    x_hi = max(x + w for (x, _y, w, _h) in opt.locked_rects)
    x_lo = min(0.0, min(x for (x, _y, _w, _h) in opt.locked_rects))
    w_star = x_hi - x_lo
    if not (w_star > 1.0) or not math.isfinite(w_star):
        return None
    return float(w_star)


def _worker_solve(args):
    """One independent (orientation, column count, seed) restart."""
    try:
        # PARTNER_FRAME_WPIN appends a 14th field (the arm's W*, or None).
        # Pre-flag payloads are 13-wide and take the historical branch, so
        # the off path is byte-identical.
        if len(args) == 14:
            (rects, areas_np, cons_np, tpos_np, b2b_np, p2b_np, pins_np,
             orient, c_force, seed, deadline, v_weight, h_scale,
             w_star) = args
        else:
            (rects, areas_np, cons_np, tpos_np, b2b_np, p2b_np, pins_np,
             orient, c_force, seed, deadline, v_weight, h_scale) = args
            w_star = None
        at = torch.from_numpy(areas_np)
        cons = torch.from_numpy(cons_np) if cons_np is not None else None
        tpos = torch.from_numpy(tpos_np) if tpos_np is not None else None
        b2b = torch.from_numpy(b2b_np) if b2b_np is not None else None
        p2b = torch.from_numpy(p2b_np) if p2b_np is not None else None
        pins = torch.from_numpy(pins_np) if pins_np is not None else None
        rs = rects
        if orient == 'T':
            rs, cons, tpos, pins = _transpose_inputs(rects, cons, tpos, pins)
        opt = _ColumnOptimizer(rs, at, cons, tpos, b2b, p2b, pins, deadline,
                               seed=seed, v_weight=v_weight, h_scale=h_scale,
                               w_star=w_star)
        if orient == 'P':
            got = _perimeter_pack(opt, rs)
            if got is not None and got[0]:
                opt = _ColumnOptimizer(rs, at, cons, tpos, b2b, p2b, pins,
                                       deadline, seed=seed,
                                       v_weight=v_weight, h_scale=h_scale,
                                       pinned=got[0], w_star=w_star)
        if opt.locked_only():
            out = opt.locked_positions()
            return (out, 0.0, 0.0, 0)
        opt.prepare()
        if c_force is not None:
            c_use = max(2, min(18, c_force))
            if c_use != opt.C0:
                opt.C0 = c_use
                cols = opt._init_columns(c_use)
                c0, _ = opt._evaluate(cols)
                opt._cols = cols
                opt._cost0 = c0
        out = opt.finish(deadline, max_runs=1)
        hp, area, V = opt.final_metrics
        if orient == 'T':
            out = [(y, x, h, w) for (x, y, w, h) in out]
        return (out, hp, area, V)
    except Exception:
        return None


def _worker_refine(args):
    """Legalize+refine one direct-model prediction with a full worker
    budget (the in-process thread used to slice one budget across
    candidates, so most predictions never got refined at all)."""
    try:
        (pred, areas_np, cons_np, tpos_np, b2b_np, p2b_np, pins_np,
         deadline, seed, v_weight, tag_anchor) = args
        from layout_refiner import refine_prediction, full_violations
        at = torch.from_numpy(areas_np)
        cons = torch.from_numpy(cons_np) if cons_np is not None else None
        tpos = torch.from_numpy(tpos_np) if tpos_np is not None else None
        b2b = torch.from_numpy(b2b_np) if b2b_np is not None else None
        p2b = torch.from_numpy(p2b_np) if p2b_np is not None else None
        pins = torch.from_numpy(pins_np) if pins_np is not None else None
        rect_list = [tuple(map(float, pred[i])) for i in range(len(pred))]
        opt = _ColumnOptimizer(rect_list, at, cons, tpos, b2b, p2b, pins,
                               deadline, seed=seed, v_weight=v_weight)
        opt._tag_anchor = bool(tag_anchor)
        out = refine_prediction(opt, pred, deadline, seed=seed + 10)
        if out is None:
            return None
        pos = np.asarray(out, dtype=np.float64)
        hp = opt._hpwl(pos)
        area = float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                     * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min()))
        V = full_violations(opt, pos)
        lst = [tuple(map(float, pos[i])) for i in range(len(pos))]
        return (lst, hp, area, V)
    except Exception:
        return None


def _parallel_solve(opt1, rects, area_targets, constraints, target_positions,
                    b2b, p2b, pins, deadline, seed, sample_fn=None):
    avg_area = opt1.total_area / max(opt1.n, 1)
    C0 = max(2, min(18, int(round(opt1.W_est / max(math.sqrt(avg_area), 2.0)))))
    hard_n = sum(1 for u in opt1.units if u.hasB) + sum(1 for u in opt1.units if u.hasT)
    hard_t = sum(1 for u in opt1.units if u.force == 'L') + \
        sum(1 for u in opt1.units if u.force == 'R')
    allow_t = hard_t <= hard_n + 2

    # portfolio: mixed column counts / seeds / frame heights, plus
    # violation-averse searchers (layouts compete under true cost at selection)
    configs = [('N', None, seed + 11, 1.0, 1.0), ('N', C0 - 1, seed + 22, 1.0, 1.0),
               ('N', C0 + 1, seed + 33, 1.0, 1.0), ('N', None, seed + 44, 2.5, 1.0),
               ('N', C0 - 2, seed + 55, 1.0, 1.0), ('N', C0 + 2, seed + 66, 1.0, 1.0),
               ('N', None, seed + 77, 2.5, 1.0), ('N', C0 - 1, seed + 88, 1.0, 1.0),
               ('N', None, seed + 99, 1.0, 1.0), ('N', C0 + 1, seed + 110, 1.0, 1.0),
               ('N', C0, seed + 121, 2.5, 1.0), ('N', C0 - 1, seed + 132, 1.0, 1.0)]
    if allow_t:
        configs[5] = ('T', None, seed + 66, 1.0, 1.0)
        configs[7] = ('T', None, seed + 88, 2.5, 1.0)
        configs[9] = ('T', C0 + 1, seed + 110, 1.0, 1.0)
    extra = [('N', None, seed + 143, 1.0, 0.90), ('N', C0, seed + 154, 1.0, 1.10),
             ('N', C0 + 1, seed + 165, 2.5, 1.0), ('N', C0 - 1, seed + 176, 1.0, 0.90),
             ('N', None, seed + 187, 1.0, 1.10), ('N', C0 + 2, seed + 198, 1.0, 1.0),
             ('N', C0 - 2, seed + 209, 2.5, 1.0), ('N', None, seed + 220, 1.0, 0.82),
             ('N', C0 + 1, seed + 231, 1.0, 1.20), ('N', None, seed + 242, 2.5, 1.0),
             ('N', C0 - 1, seed + 253, 1.0, 1.0), ('N', None, seed + 264, 1.0, 1.0)]
    if allow_t:
        extra[3] = ('T', C0 - 1, seed + 176, 1.0, 1.0)
        extra[7] = ('T', None, seed + 220, 1.0, 1.0)

    # PARTNER_PERIMETER=1: on boundary-dense instances, spend three of the
    # extra restarts on perimeter-first searchers (_perimeter_pack) — the
    # wall lines are solved explicitly and locked, the SA only places the
    # interior.  Competes under true cost like every other restart.
    if _os.environ.get("PARTNER_PERIMETER_COL", "0") != "0":
        n_bnd_mv = sum(1 for i in range(opt1.n)
                       if opt1.boundary[i] > 0 and opt1.kind[i] != 2)
        if n_bnd_mv >= max(6, int(0.08 * opt1.n)):
            extra[9] = ('P', None, seed + 242, 1.0, 1.0)
            extra[10] = ('P', C0, seed + 253, 1.0, 1.0)
            extra[11] = ('P', None, seed + 264, 2.5, 1.0)

    # reserve pool workers for direct-prediction refinement: each refine
    # then gets a full worker budget instead of a slice of one thread
    n_ref = 0
    if sample_fn is not None:
        n_ref = min(8, max(3, _POOL_SIZE // 3))
        # PARTNER_NREF: rebalance pool slots toward direct-prediction
        # refinement (the direct channel dominates tail winners); always
        # keeps at least 6 column restarts, and only kicks in on big cases
        # (n >= PARTNER_NREF_MIN_N, default 95) where the tail weight
        # lives.  Same pool, same wall-clock.
        try:
            _nref_env = int(float(_os.environ.get("PARTNER_NREF", "0") or 0))
            _nref_min_n = int(float(_os.environ.get(
                "PARTNER_NREF_MIN_N", "95")))
        except ValueError:
            _nref_env = 0
            _nref_min_n = 95
        if _nref_env > 0 and opt1.n >= _nref_min_n:
            n_ref = max(3, min(_nref_env, _POOL_SIZE - 6))
    configs = (configs + extra)[:max(2, _POOL_SIZE - n_ref)]
    # The hand-written list holds 24 entries, so pools larger than 24+n_ref
    # historically idled the surplus workers.  Extend procedurally with seed
    # variants (same orient/column/weight grid, fresh rng streams) so restart
    # breadth scales with PARTNER_POOL.  Bit-exact for POOL<=24: the slice
    # above already truncated to want<=24 and the loop body never runs.
    _want = max(2, _POOL_SIZE - n_ref)
    if len(configs) < _want:
        _base = list(configs)
        _k = 0
        while len(configs) < _want:
            _orient, _cf, _sd, _vw, _hs = _base[_k % len(_base)]
            _k += 1
            configs.append((_orient, _cf, _sd + 275 + 11 * _k, _vw, _hs))

    # PARTNER_FRAME_WPIN: hand the tail restart slots a frame derived from
    # the L/R-tagged preplaced blocks instead of the pin-aspect guess.  They
    # are ADDITIONAL arms, not replacements -- the h_scale arms all stay --
    # and they compete under the same true cost at selection, so an instance
    # where the aspect guess was already right loses nothing but the tail
    # slots' marginal breadth.  Only 'N' slots qualify: W* is measured in the
    # original frame, and a 'T' restart solves the transposed one.
    _ws_map: Dict[int, float] = {}
    if frame_wpin_on():
        _ws = _w_star_from_tags(opt1)
        if _ws is not None:
            try:
                _ws_arms = int(float(_os.environ.get(
                    "PARTNER_FRAME_WPIN_ARMS", "3")))
            except ValueError:
                _ws_arms = 3
            for _idx in range(len(configs) - 1, -1, -1):
                if len(_ws_map) >= max(0, _ws_arms):
                    break
                if configs[_idx][0] == 'N':
                    _ws_map[_idx] = _ws

    def np_of(t):
        return None if t is None else t.detach().cpu().numpy()

    # PARTNER_PHASE_B (fraction, default 0=off): reserve the tail of the
    # case budget for a parallel exploitation round that re-refines the
    # phase-A winner with seed/v_weight/anchor variants.  The serial form
    # of this (vkill stage2) loses to SA's own tail; eight CONCURRENT
    # basin-hops around the incumbent are a different proposition.  Only
    # engages on budgets long enough for a meaningful refine round.
    #
    # PARTNER_GPU_ARM=1 drives the same carve from the idle-accelerator side
    # and sizes it itself (see the module header): phase A keeps every column
    # slot it has today and only gives up wall clock, and it only gives that
    # up when the measured sampler latency says the masked second wave fits.
    t_pb = time.time()
    rem_pb = max(0.0, deadline - t_pb)
    gpu_arm = (gpu_arm_on() and sample_fn is not None and _POOL_READY
               and _os.environ.get("PARTNER_FUSION", "0") == "0")
    _pb_raw = _os.environ.get("PARTNER_PHASE_B")
    try:
        _pb_frac = float(_pb_raw or 0)
        _pb_min = float(_os.environ.get(
            "PARTNER_PHASE_B_MIN_BUDGET", phase_b_min_budget_default(gpu_arm)))
    except ValueError:
        _pb_frac, _pb_min = 0.0, 8.0
    # auto carve: only when PARTNER_PHASE_B was not pinned by hand
    gpu_auto = False
    _pb_est = 0.0
    if gpu_arm and not _pb_frac:
        _pb_est = gpu_arm_sample_estimate(opt1.n)
        _pb_slice = gpu_arm_slice(rem_pb)
        # phase A must still hold BOTH sampling waves and keep a real SA span
        if (rem_pb >= _pb_min and _pb_slice > 0.0
                and rem_pb - _pb_slice - 2.0 * _pb_est
                >= _env_num("PARTNER_GPU_ARM_MIN_A", 0.35) * rem_pb):
            _pb_frac = _pb_slice / rem_pb
            gpu_auto = True
    phase_b = (0.0 < _pb_frac < 0.9) and (deadline - t_pb) >= _pb_min \
        and _POOL_READY
    gpu_arm = gpu_arm and phase_b
    if _os.environ.get("PARTNER_GPU_ARM_DEBUG") and gpu_arm_on():
        # carve decision, printed for DECLINED cases too -- a goal-tier run
        # where `auto=0` everywhere means the A/B is measuring "arm never
        # engaged", not "arm did not help"
        import sys as _sys
        print(f"[gpuarm] gate n={opt1.n} rem={rem_pb:.3f} est={_pb_est:.3f} "
              f"frac={_pb_frac:.3f} auto={int(gpu_auto)} "
              f"phase_b={int(phase_b)}", file=_sys.stderr, flush=True)
    deadline_A = (t_pb + (deadline - t_pb) * (1.0 - _pb_frac)) \
        if phase_b else deadline
    # Historical margin was a flat 0.30 s; below ~2 s of remaining budget that
    # would eat the whole worker window, so scale it down proportionally
    # (bit-identical for remaining >= 2.0 s, i.e. every pre-POOL_GATE case).
    worker_deadline = deadline_A - min(0.30, 0.15 * max(0.0, deadline_A - t_pb))
    payloads = [(list(rects), np_of(area_targets), np_of(constraints),
                 np_of(target_positions), np_of(b2b), np_of(p2b), np_of(pins),
                 orient, cf, sd, worker_deadline, vw, hs)
                for (orient, cf, sd, vw, hs) in configs]
    if _ws_map:
        # uniform arity so the worker's length test is a single branch
        payloads = [p + (_ws_map.get(k),) for k, p in enumerate(payloads)]
        if _os.environ.get("PARTNER_FRAME_WPIN_DEBUG"):
            import sys as _sys
            print(f"[wpin] n={opt1.n} W*={next(iter(_ws_map.values())):.2f} "
                  f"arms={sorted(_ws_map)} of {len(configs)}",
                  file=_sys.stderr, flush=True)
    res = _POOL.map_async(_worker_solve, payloads)

    # column restarts are already running; sample on the GPU now and hand
    # the predictions to the reserved (idle) workers
    ref_res = None
    preds2: List[np.ndarray] = []
    _pb_nvar = _PHASE_B_VARIANTS
    if n_ref:
        _t_s0 = time.time()
        try:
            preds = list(sample_fn(n_ref))
        except Exception:
            preds = []
        _t_s1 = time.time() - _t_s0
        if gpu_arm:
            gpu_arm_record_sample(opt1.n, _t_s1)
        wave1 = list(preds)
        # boundary-dense instances: overlay the exactly-packed perimeter
        # ring onto the top predictions — the model supplies the global
        # arrangement (its strength), the ring supplies wall exactness
        # (the model is frame-blind); the refine ladder then starts with
        # its wall tags already satisfied and never drops the pins
        if _os.environ.get("PARTNER_PERIMETER", "0") != "0" and preds:
            n_bnd_mv = sum(1 for i in range(opt1.n)
                           if opt1.boundary[i] > 0 and opt1.kind[i] != 2)
            if n_bnd_mv >= max(6, int(0.08 * opt1.n)):
                try:
                    got = _perimeter_pack(opt1, list(rects))
                except Exception:
                    got = None
                if got is not None and got[0]:
                    over = []
                    for P in preds[:2]:
                        Q = np.asarray(P, dtype=np.float64).copy()
                        for i, r in got[0].items():
                            Q[i] = r
                        over.append(Q)
                    preds = over + preds
        if preds:
            # PARTNER_REFINE_VW_MIX > 0: every 3rd refine worker runs
            # violation-averse (v_weight boosted) — portfolio diversity in
            # the direct channel, mirroring the column restarts' 2.5-weight
            # searchers; selection still compares raw (hp, area, V).
            try:
                _vw_mix = float(_os.environ.get(
                    "PARTNER_REFINE_VW_MIX", "0") or 0)
            except ValueError:
                _vw_mix = 0.0
            # PARTNER_TAG_ANCHOR_MIX=1: half the refine workers seed their
            # wall tags against the preplaced-implied frame, half against
            # the prediction's own extents — the two are each right on a
            # different instance class (anchored wins when the content
            # fits the intended frame, emergent wins when it cannot), and
            # true-cost selection picks per case.
            _anchor_mix = _os.environ.get(
                "PARTNER_TAG_ANCHOR_MIX", "0") != "0"
            # PARTNER_TAG_ANCHOR_EXTRA=k: additionally re-refine the top-k
            # predictions with anchored tag seeding (extra draws, so the
            # emergent-strategy draw count is untouched; the slots come out
            # of PARTNER_NREF, i.e. the column-restart pool)
            try:
                _anchor_extra = int(float(_os.environ.get(
                    "PARTNER_TAG_ANCHOR_EXTRA", "0") or 0))
            except ValueError:
                _anchor_extra = 0
            _anchor_extra = max(0, min(_anchor_extra, n_ref - 1, len(preds)))
            specs = [(P, bool(_anchor_mix and k % 2 == 1))
                     for k, P in enumerate(preds[:n_ref - _anchor_extra])]
            specs += [(preds[j], True) for j in range(_anchor_extra)]
            ref_payloads = [(np.asarray(P, dtype=np.float64),
                             np_of(area_targets), np_of(constraints),
                             np_of(target_positions), np_of(b2b),
                             np_of(p2b), np_of(pins),
                             worker_deadline, seed + 301 + 7 * k,
                             (_vw_mix if (_vw_mix > 0.0 and k % 3 == 2)
                              else 1.0),
                             anchor)
                            for k, (P, anchor) in enumerate(specs)]
            ref_res = _POOL.map_async(_worker_refine, ref_payloads)

        # --- GPU idle arm: second sampling wave, masked by phase A -------
        # The parent is about to block in `res.get`; the accelerator is idle
        # from here until the case ends.  Draw a fresh batch (different
        # generator seed) now -- the workers keep running through it, so the
        # latency costs the case nothing as long as it lands before
        # `deadline_A`, which `gpu_arm_second_wave` verifies against the
        # latency just measured for wave 1.
        if gpu_arm:
            _pb_k2, _pb_nvar = gpu_arm_phase_b_split(
                max(0.0, deadline - deadline_A), _POOL_SIZE)
            preds2 = gpu_arm_second_wave(
                sample_fn, _pb_k2,
                int(_env_num("PARTNER_GPU_ARM_SEED", 8117)) + (seed % 1000),
                _t_s1, deadline_A)
            # opt-in flow sub-samplers (ZORDER / NOPT) keep their pinned
            # seeds, so a wave-2 draw can repeat a wave-1 candidate; a
            # duplicate would waste a phase-B slot, so drop it here
            if preds2 and wave1:
                preds2 = [Q for Q in preds2
                          if not any(Q.shape == np.shape(P)
                                     and np.array_equal(Q, np.asarray(P))
                                     for P in wave1)]
    try:
        outs = res.get(timeout=max(deadline_A - time.time(), 0.1) + 2.5)
        ref_outs = []
        if ref_res is not None:
            ref_outs = ref_res.get(
                timeout=max(deadline_A - time.time(), 0.1) + 2.5)
    except Exception:
        # workers overran their deadline — the pool now has stragglers that
        # would delay every later case, so drop it entirely
        _shutdown_pool()
        raise
    outs = [o for o in outs if o is not None]
    ref_outs = [o for o in ref_outs if o is not None]
    if not outs and not ref_outs:
        raise RuntimeError("all parallel workers failed")

    area_ref = opt1.area_ref
    n_soft = opt1.n_soft_den
    hp_ref = max(min(o[1] for o in outs + ref_outs), 1e-9)

    def score(o):
        _out, hp, area, V = o
        return (1.0 + 0.5 * ((hp - hp_ref) / hp_ref
                             + max(0.0, area / area_ref - 1.0))) \
            * math.exp(2.0 * V / n_soft)

    if _RDEBUG:
        for tag, lst in (("col", outs), ("dir", ref_outs)):
            for o in sorted(lst, key=score)[:3]:
                print(f"[psel] {tag} hp={o[1]:.1f} area={o[2]:.0f} "
                      f"V={o[3]} score={score(o):.4f}", flush=True)
    best_col = min(outs, key=score) if outs else None
    best_dir = min(ref_outs, key=score) if ref_outs else None
    # a direct candidate must beat the column result by a clear margin —
    # marginal swaps are proxy-noise coin flips
    if best_dir is not None and (
            best_col is None
            or score(best_dir) < 0.985 * score(best_col)):
        win, win_is_ref = best_dir, True
    else:
        win, win_is_ref = best_col, False

    # PARTNER_FUSION=1 (uses the phase-B carve): regional crossover of the
    # elite candidates.  Every case computes 20-40 diverse layouts and the
    # selection keeps ONE — candidates are routinely complementary by
    # region (one wins the left half, another the right), and that
    # information is free.  Cut the die, take each side from a different
    # parent, repair the seam with the existing machinery, accept by the
    # same true-cost score.
    if (phase_b and _os.environ.get("PARTNER_FUSION", "0") != "0"
            and time.time() < deadline - 1.0):
        elites = sorted(outs + ref_outs, key=score)[:3]
        locked = [opt1.kind[i] == 2 for i in range(opt1.n)]
        fused_cands = []
        for pi in range(min(2, len(elites) - 1)):
            A, B = elites[0], elites[pi + 1]
            PA = np.asarray([list(r) for r in A[0]], dtype=np.float64)
            PB = np.asarray([list(r) for r in B[0]], dtype=np.float64)
            for axis in (0, 1):
                if time.time() >= deadline - 0.4:
                    break
                ca = PA[:, axis] + 0.5 * PA[:, 2 + axis]
                lo = float(PA[:, axis].min())
                hi = float((PA[:, axis] + PA[:, 2 + axis]).max())
                for frac in (0.5,):
                    cut = lo + frac * (hi - lo)
                    take_a = ca < cut
                    if take_a.all() or not take_a.any():
                        continue
                    Q = np.where(take_a[:, None], PA, PB)
                    for i in range(opt1.n):     # locked stay canonical
                        if locked[i]:
                            Q[i] = (opt1.lx[i], opt1.ly[i],
                                    opt1.rw[i], opt1.rh[i])
                    try:
                        rep = _ensure_no_overlap(
                            [tuple(map(float, r)) for r in Q], locked)
                    except Exception:
                        continue
                    Qr = np.asarray([list(r) for r in rep],
                                    dtype=np.float64)
                    from layout_refiner import full_violations as _fv
                    hp2 = opt1._hpwl(Qr)
                    a2 = float(((Qr[:, 0] + Qr[:, 2]).max()
                                - Qr[:, 0].min())
                               * ((Qr[:, 1] + Qr[:, 3]).max()
                                  - Qr[:, 1].min()))
                    fused_cands.append((rep, hp2, a2, _fv(opt1, Qr)))
        if fused_cands:
            bestf = min(fused_cands, key=score)
            if _RDEBUG:
                print(f"[psel] fusion win={score(win):.4f} "
                      f"bestf={score(bestf):.4f} ({len(fused_cands)})",
                      flush=True)
            if score(bestf) < score(win) - 1e-12:
                win, win_is_ref = bestf, True

    # the flat 1.5 s entry gate and 0.30 s worker margin were sized for the
    # legacy >= 8 s carve; under the arm's auto carve both scale with the
    # slice (a 0.15 s slice would otherwise be gated out, or handed a
    # worker deadline in the past).  Byte-identical on the legacy path.
    _pb_enter = 0.5 * (rem_pb * _pb_frac) if gpu_auto else 1.5
    if phase_b and time.time() < deadline - _pb_enter \
            and _os.environ.get("PARTNER_FUSION", "0") == "0":
        Wnp = np.asarray([list(r) for r in win[0]], dtype=np.float64)
        # `_pb_nvar` is `_PHASE_B_VARIANTS` unless the arm had to share a
        # pool too small to hold both rounds -> no-op off the arm path
        variants = ((1.0, False), (1.0, True), (2.5, False), (3.0, False),
                    (1.0, False), (2.5, True), (1.0, True),
                    (1.0, False))[:_pb_nvar]
        wd2 = deadline - (min(0.30, 0.15 * max(0.0, deadline - time.time()))
                          if gpu_auto else 0.30)
        pb_payloads = [(Wnp.copy(), np_of(area_targets),
                        np_of(constraints), np_of(target_positions),
                        np_of(b2b), np_of(p2b), np_of(pins),
                        wd2, seed + 401 + 11 * k, vw, anc)
                       for k, (vw, anc) in enumerate(variants)]
        # second-wave GPU candidates ride the SAME round on the pool slots
        # phase B has always left idle (8 payloads on a pool of up to 24).
        # They enter as raw predictions, so they get their own light
        # v_weight / anchor mix -- selection below is still true cost.
        pb_payloads += [(np.asarray(P, dtype=np.float64),
                         np_of(area_targets), np_of(constraints),
                         np_of(target_positions), np_of(b2b), np_of(p2b),
                         np_of(pins), wd2, seed + 601 + 13 * k,
                         (2.5 if k % 4 == 3 else 1.0), bool(k % 3 == 2))
                        for k, P in enumerate(preds2)]
        if _os.environ.get("PARTNER_GPU_ARM_DEBUG"):
            import sys as _sys
            print(f"[gpuarm] n={opt1.n} rem={rem_pb:.3f} "
                  f"frac={_pb_frac:.3f} slice={rem_pb * _pb_frac:.3f} "
                  f"est={_pb_est:.3f} auto={int(gpu_auto)} "
                  f"wave2={len(preds2)} payloads={len(pb_payloads)}",
                  file=_sys.stderr, flush=True)
        try:
            res2 = _POOL.map_async(_worker_refine, pb_payloads)
            outs2 = [o for o in res2.get(
                timeout=max(deadline - time.time(), 0.1) + 2.5)
                if o is not None]
        except Exception:
            _shutdown_pool()
            raise
        if outs2:
            best2 = min(outs2, key=score)
            if _RDEBUG:
                print(f"[psel] phaseB win={score(win):.4f} "
                      f"best2={score(best2):.4f} ({len(outs2)} cands)",
                      flush=True)
            if score(best2) < score(win) - 1e-12:
                win, win_is_ref = best2, True

    if win_is_ref:
        locked = [opt1.kind[i] == 2 for i in range(opt1.n)]
        return _ensure_no_overlap(list(win[0]), locked)
    return win[0]


# =============================================================================
# Main entry point
# =============================================================================
_BND_TRANSPOSE = {0: 0}
for _c in range(1, 16):
    _t = 0
    if _c & 1:   # left -> bottom
        _t |= 8
    if _c & 2:   # right -> top
        _t |= 4
    if _c & 4:   # top -> right
        _t |= 2
    if _c & 8:   # bottom -> left
        _t |= 1
    _BND_TRANSPOSE[_c] = _t


def _transpose_inputs(rects, constraints, target_positions, pins):
    rects_t = [(y, x, h, w) for (x, y, w, h) in rects]
    cons_t = None
    if constraints is not None:
        cons_t = constraints.clone()
        if cons_t.dim() == 2 and cons_t.shape[1] > 4:
            for i in range(cons_t.shape[0]):
                cons_t[i, 4] = float(_BND_TRANSPOSE.get(int(round(float(cons_t[i, 4]))) & 15, 0))
    tpos_t = None
    if target_positions is not None:
        tpos_t = target_positions.clone()
        tpos_t[:, 0] = target_positions[:, 1]
        tpos_t[:, 1] = target_positions[:, 0]
        tpos_t[:, 2] = target_positions[:, 3]
        tpos_t[:, 3] = target_positions[:, 2]
    pins_t = None
    if pins is not None and pins.dim() == 2 and pins.shape[1] >= 2:
        pins_t = pins.clone()
        pins_t[:, 0] = pins[:, 1]
        pins_t[:, 1] = pins[:, 0]
    return rects_t, cons_t, tpos_t, pins_t


def legalize_rectangles(
    rects: List[Rect],
    area_targets: torch.Tensor,
    constraints: Optional[torch.Tensor] = None,
    target_positions: Optional[torch.Tensor] = None,
    b2b_connectivity: Optional[torch.Tensor] = None,
    p2b_connectivity: Optional[torch.Tensor] = None,
    pins_pos: Optional[torch.Tensor] = None,
    deadline: Optional[float] = None,
    seed: int = 0,
    sample_fn=None,
) -> List[Rect]:
    n = len(rects)
    if n == 0:
        return []
    if deadline is None:
        deadline = time.time() + 3.0
    opt1 = _ColumnOptimizer(
        rects, area_targets, constraints, target_positions,
        b2b_connectivity, p2b_connectivity, pins_pos, deadline, seed=seed,
    )
    if opt1.locked_only():
        return opt1.locked_positions()
    budget = deadline - time.time()

    # PARTNER_POOL_GATE (default 3.0 = historical behavior): minimum remaining
    # budget for the parallel restart portfolio.  The hard 3.0 gate was
    # measured (2026-08-04 low-budget frontier) to be the entire +0.19 quality
    # cliff between BUDGET_MAX=3.5 and 3.0 — below it every case ran the
    # single-threaded chain and the direct/flow channels were never consumed.
    # Set to 0 to let the pool engage at any budget (worker margin scales).
    try:
        _pool_gate = float(_os.environ.get("PARTNER_POOL_GATE", "3.0"))
    except ValueError:
        _pool_gate = 3.0
    if _POOL is not None and _POOL_READY and budget > _pool_gate:
        try:
            return _parallel_solve(opt1, rects, area_targets, constraints,
                                   target_positions, b2b_connectivity,
                                   p2b_connectivity, pins_pos, deadline, seed,
                                   sample_fn=sample_fn)
        except Exception:
            pass  # fall back to the sequential path below

    if budget < 4.0:
        return opt1.run()

    # count units that need a column bottom/top of their own per orientation:
    # L/R tags are structurally cheap (edge columns have unlimited capacity),
    # B/T tags are scarce (one per column). Transposing swaps the two.
    hard_n = sum(1 for u in opt1.units if u.hasB) + sum(1 for u in opt1.units if u.hasT)
    hard_t = sum(1 for u in opt1.units if u.force == 'L') + \
        sum(1 for u in opt1.units if u.force == 'R')
    if hard_t > hard_n + 2:
        return opt1.run()

    # try both orientations (HPWL / area / violations are orientation
    # invariant, so probe costs are directly comparable)
    rects_t, cons_t, tpos_t, pins_t = _transpose_inputs(
        rects, constraints, target_positions, pins_pos)
    opt2 = _ColumnOptimizer(
        rects_t, area_targets, cons_t, tpos_t,
        b2b_connectivity, p2b_connectivity, pins_t, deadline, seed=seed + 1,
    )
    opt1.prepare()
    opt2.prepare()
    opt2.hp_ref = opt1.hp_ref
    t_probe = min(0.20 * budget, 4.2) / 6.0
    p1 = opt1.probe(t_probe)
    p2 = opt2.probe(t_probe)
    if opt1._early_exit:
        deadline = max(time.time(),
                       deadline - opt1._probe_unspent - opt2._probe_unspent)
    # probes are noisy; the normal orientation handles the common
    # many-left-tags pattern structurally better, so require a clear win
    if p2 < p1 * 0.96:
        out_t = opt2.finish(deadline)
        return [(y, x, h, w) for (x, y, w, h) in out_t]
    return opt1.finish(deadline)
