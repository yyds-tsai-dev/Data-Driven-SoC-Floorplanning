"""Opt-in production backbone (FLOORSET_COLUMN_BACKBONE).

Ported from partner/my_opt_claude_v2.py: the pure-heuristic seed layout and
row fallback, plus the column-slicing legalizer call, with the diffusion
refinement step removed (heuristic seed is passed straight into
legalize_rectangles).
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import torch

from .column_slicing import _parse_constraints, _target, init_worker_pool, legalize_rectangles

Rect = Tuple[float, float, float, float]

BUDGET_SCALE = 0.06
BUDGET_TAU = 20.0
BUDGET_MIN = 0.8
BUDGET_MAX = 24.0


def _base_budget(block_count: int) -> float:
    """The unscaled, clamped exponential per-case budget (no tail multiplier).
    ~0.8 s for the smallest cases up to ~24 s for n=120; averages about 5 s
    over the validation set."""
    b = BUDGET_SCALE * math.exp(block_count / BUDGET_TAU)
    return max(BUDGET_MIN, min(BUDGET_MAX, b))


def _tail_n() -> int:
    try:
        return int(os.environ.get("FLOORSET_TAIL_BUDGET_N", "116"))
    except ValueError:
        return 116


def _time_budget(block_count: int) -> float:
    """Per-case wall-clock budget for the SA legalizer.

    Dormant E2 tail-budget escape hatch (default neutral, mirrors the N1
    FLOORSET_AREA_SCALE convention): FLOORSET_TAIL_BUDGET_SCALE (float,
    default 1.0) multiplies the clamped budget for cases with
    block_count >= FLOORSET_TAIL_BUDGET_N (int, default 116), applied AFTER
    the clamp so it can lift the 24s ceiling on the tail. Malformed env
    values fall back to the default silently. With scale == 1.0 (the
    default) this is byte-identical to the old formula for every n.

    E2-adaptive (FLOORSET_TAIL_BUDGET_ADAPTIVE=1, default 0): instead of the
    flat SCALE, give the tail a *generous ceiling* base * FLOORSET_TAIL_BUDGET_CAP
    (float, default 3.0) and let each SA worker self-stop when its own best-cost
    trajectory stalls (see `_anneal` early-stop in column_slicing.py). ADAPTIVE
    takes PRECEDENCE over FLOORSET_TAIL_BUDGET_SCALE: when it is on, CAP replaces
    SCALE as the sole tail multiplier here, and the per-worker stall window
    (see `_tail_stall_window`) is what actually reins runtime back in below the
    ceiling. With ADAPTIVE unset this branch is skipped entirely, so the flat-
    SCALE path is byte-identical to the pre-adaptive formula.
    """
    budget = _base_budget(block_count)

    try:
        tail_scale = float(os.environ.get("FLOORSET_TAIL_BUDGET_SCALE", "1.0"))
    except ValueError:
        tail_scale = 1.0
    tail_n = _tail_n()

    if os.environ.get("FLOORSET_TAIL_BUDGET_ADAPTIVE", "0") == "1":
        try:
            tail_cap = float(os.environ.get("FLOORSET_TAIL_BUDGET_CAP", "3.0"))
        except ValueError:
            tail_cap = 3.0
        if block_count >= tail_n and tail_cap != 1.0:
            budget *= tail_cap
    elif block_count >= tail_n and tail_scale != 1.0:
        budget *= tail_scale
    return budget


def _tail_stall_window(block_count: int) -> Optional[float]:
    """E2-adaptive per-worker stall window (seconds) for this case, or None
    when adaptive is off / the case is below the tail threshold (so the SA
    early-stop stays disabled and byte-identical).

    The window is a fraction (FLOORSET_TAIL_STALL_WINDOW, default 0.25) of the
    BASE (unscaled, clamped) per-case budget, so it scales with case size the
    same way the budget does: ~2.2 s at n=100 up to ~6 s at n=120. A worker
    stops its chain once its best cost has not improved by >= the relative eps
    (FLOORSET_TAIL_STALL_EPS) within one window. Malformed values fall back to
    the default silently."""
    if os.environ.get("FLOORSET_TAIL_BUDGET_ADAPTIVE", "0") != "1":
        return None
    if block_count < _tail_n():
        return None
    try:
        frac = float(os.environ.get("FLOORSET_TAIL_STALL_WINDOW", "0.25"))
    except ValueError:
        frac = 0.25
    return max(frac * _base_budget(block_count), 0.05)


def _tail_stall_eps() -> float:
    """Relative best-cost improvement below which a stall window counts as
    'stalled' (FLOORSET_TAIL_STALL_EPS, default 0.003 == 0.3%). The SA cost is
    positive by construction (>= ~1.0), so a relative threshold is well posed;
    this is the primary knob for the adaptive stop and should be swept in the
    paired measurement."""
    try:
        return float(os.environ.get("FLOORSET_TAIL_STALL_EPS", "0.003"))
    except ValueError:
        return 0.003


def _heuristic_init(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
) -> List[Rect]:
    """Pin/graph-weighted centroid layout used only as a relative-position
    seed for the diffusion model / column optimizer."""
    n = len(area_targets)
    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)

    shapes: List[Tuple[float, float]] = []
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            shapes.append((tw, th))
        else:
            area = max(float(area_targets[i]), 1e-6)
            shapes.append((math.sqrt(area), math.sqrt(area)))

    total_area = sum(w * h for w, h in shapes)
    scale = math.sqrt(max(total_area, 1.0))

    sx = [0.0] * n
    sy = [0.0] * n
    wsum = [0.0] * n
    n_pins = pins.shape[0]
    for edge in p2b:
        if edge[0] == -1:
            continue
        p = int(edge[0].item())
        b = int(edge[1].item())
        if not (0 <= b < n and 0 <= p < n_pins):
            continue
        px, py = float(pins[p, 0]), float(pins[p, 1])
        if px == -1.0 or py == -1.0:
            continue
        w = max(float(edge[2].item()), 0.0)
        sx[b] += w * px
        sy[b] += w * py
        wsum[b] += w

    no_pin = [i for i in range(n) if wsum[i] <= 1e-9]
    cols = max(1, int(math.sqrt(max(len(no_pin), 1))))
    cursor = 0
    cx = [0.0] * n
    cy = [0.0] * n
    for i in range(n):
        if wsum[i] > 1e-9:
            cx[i] = sx[i] / wsum[i]
            cy[i] = sy[i] / wsum[i]
        else:
            row, col = divmod(cursor, cols)
            cx[i] = (col + 0.5) * scale / cols
            cy[i] = (row + 0.5) * scale / cols
            cursor += 1

    nx = list(cx)
    ny = list(cy)
    deg = [0.0] * n
    for edge in b2b:
        if edge[0] == -1:
            continue
        i = int(edge[0].item())
        j = int(edge[1].item())
        if not (0 <= i < n and 0 <= j < n):
            continue
        w = max(float(edge[2].item()), 0.0)
        nx[i] += 0.25 * w * cx[j]
        ny[i] += 0.25 * w * cy[j]
        nx[j] += 0.25 * w * cx[i]
        ny[j] += 0.25 * w * cy[i]
        deg[i] += 0.25 * w
        deg[j] += 0.25 * w
    fcx = [nx[i] / (1.0 + deg[i]) for i in range(n)]
    fcy = [ny[i] / (1.0 + deg[i]) for i in range(n)]

    out: List[Rect] = []
    for i in range(n):
        w, h = shapes[i]
        tx, ty, _tw, _th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0:
            out.append((tx, ty, w, h))
        else:
            out.append((fcx[i] - 0.5 * w, fcy[i] - 0.5 * h, w, h))
    return out


def _fallback_row(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
) -> List[Rect]:
    """Guaranteed-feasible fallback: preplaced blocks stay put; every other
    block is placed in a single row strictly to the right of everything."""
    n = len(area_targets)
    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)
    out: List[Optional[Rect]] = [None] * n
    x_cursor = 0.0
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
            out[i] = (tx, ty, tw, th)
            x_cursor = max(x_cursor, tx + tw)
    x_cursor += 1.0
    for i in range(n):
        if out[i] is not None:
            continue
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            w, h = tw, th
        else:
            area = max(float(area_targets[i]), 1e-9)
            w = math.sqrt(area)
            h = area / w
        out[i] = (x_cursor, 0.0, w, h)
        x_cursor += w
    return [r for r in out]


def warm_worker_pool() -> None:
    """Spawn the parallel-restart worker pool ahead of time so worker
    startup cost is not charged to any test case.

    Dormant W1 widening flag: FLOORSET_SA_WORKERS (int, default 0 = keep the
    historical formula min(12, cores//2)). The hidden-test machine has 48
    cores (official QA A3) while the formula caps the pool at 12 workers;
    setting FLOORSET_SA_WORKERS raises the pool so a wider restart portfolio
    (FLOORSET_SA_CONFIGS in column_slicing._parallel_solve) can actually run.
    Malformed values fall back to the default silently."""
    try:
        w = int(os.environ.get("FLOORSET_SA_WORKERS", "0"))
    except ValueError:
        w = 0
    if w > 0:
        init_worker_pool(max(2, w))
    else:
        init_worker_pool(max(2, min(12, (os.cpu_count() or 4) // 2)))


def solve_with_column_backbone(
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor] = None,
) -> List[Rect]:
    start = time.time()
    budget = _time_budget(block_count)
    deadline = start + budget
    # When the slack refiner is on, carve its time out of the SA share so the
    # total per-case runtime is unchanged; without a reserve the refiner is
    # deadline-starved on large cases (phase1+aspect alone ~0.3s at n=120).
    refine_reserve = 0.0
    if os.environ.get("FLOORSET_SLACK_REFINE", "0") == "1":
        refine_reserve = min(2.0, 0.3 + 0.012 * block_count, 0.4 * budget)

    # Topology-search reserve (M2, FLOORSET_TOPO_SEARCH, default OFF). For
    # larger cases (n >= 60) carve an additional slice out of the SA share so
    # the topo_search stage in the refiner has its own wall-clock allowance.
    # The reserve is measured against the budget REMAINING after the slack
    # refiner's own reserve is set aside, so the two carve-outs never
    # double-count. Total per-case wall clock is unchanged: SA is simply
    # handed a shorter deadline and the freed time is spent inside the refine
    # window instead. When the flag is off, topo_reserve stays 0.0 and every
    # downstream deadline is byte-identical to the pre-M2 behavior.
    # The topo stage lives INSIDE refine_layout, so it only runs when the
    # slack refiner is also enabled; carving its reserve without the refiner on
    # would just waste SA time. Require both flags.
    # Shared with M5 windowed re-pack (FLOORSET_WINDOW_REPACK): both the
    # topo-search stage and the window-repack stage live inside refine_layout,
    # run before the slack projection, and draw from the SAME `topo_reserve`
    # sub-budget (they never both need it -- window-repack is the successor
    # bet). Activate the reserve when EITHER flag is on.
    topo_reserve = 0.0
    if (
        (
            os.environ.get("FLOORSET_TOPO_SEARCH", "0") == "1"
            or os.environ.get("FLOORSET_WINDOW_REPACK", "0") == "1"
        )
        and os.environ.get("FLOORSET_SLACK_REFINE", "0") == "1"
        and block_count >= 60
    ):
        remaining_budget = max(0.0, budget - refine_reserve)
        if os.environ.get("FLOORSET_TOPO_SEARCH", "0") == "1":
            topo_reserve = min(3.0, 0.25 * remaining_budget)
        else:
            # Window-repack only: the stage is fast (measured <=0.9s at
            # n=120); a 3s carve-out taxes the SA far more than the stage
            # uses (full-run gate: SA avg_rt fell 4.46->4.00 and the stage
            # gains were eaten by the SA-budget loss). Reserve only what
            # the stage actually consumes.
            topo_reserve = min(1.0, 0.10 * remaining_budget)

    area_targets = area_targets[:block_count].detach().float().cpu()
    constraints = constraints[:block_count].detach().float().cpu()
    target_positions = (
        target_positions[:block_count].detach().float().cpu()
        if target_positions is not None
        else torch.full((block_count, 4), -1.0)
    )
    b2b = b2b_connectivity.detach().float().cpu()
    p2b = p2b_connectivity.detach().float().cpu()
    pins = pins_pos.detach().float().cpu()

    try:
        seed_rects = _heuristic_init(area_targets, constraints, target_positions, b2b, p2b, pins)

        # SA gets a deadline shortened by BOTH reserves so the refiner (slack
        # projection + optional topo_search) inherits the freed tail. The
        # `worker_deadline = deadline - 0.30` slack inside legalize_rectangles
        # is preserved automatically -- it is applied to whatever deadline we
        # pass here, and we only ever shrink it, never grow the total.
        out = legalize_rectangles(
            seed_rects, area_targets, constraints, target_positions,
            b2b_connectivity=b2b, p2b_connectivity=p2b, pins_pos=pins,
            deadline=deadline - refine_reserve - topo_reserve,
            stall_window=_tail_stall_window(block_count),
            stall_eps=_tail_stall_eps(),
        )
        if os.environ.get("FLOORSET_SLACK_REFINE", "0") == "1":
            from floorset_arch.refine.api import refine_layout
            # The refiner needs its own micro-budget: by the time the SA
            # legalizer above returns, `deadline` (the backbone's overall
            # per-case budget) is frequently already exhausted or within a
            # few ms of expiry, which would make refine_layout's own
            # deadline-check bail out before doing any work. The refiner is
            # pure numpy, single-core, bounded by MAX_SWEEPS and n<=120, so
            # its own pass costs at most tens of ms -- give it a small
            # dedicated allowance instead of inheriting the spent deadline.
            refine_deadline = time.time() + max(refine_reserve + topo_reserve, 0.1)
            # FLOORSET_SLACK_REFINE_VSNAP=1 (default off) enables a further
            # Phase-V violation-snap post-pass inside refine_layout itself
            # (read directly from os.environ there, not threaded through as a
            # parameter) -- see src/floorset_arch/refine/vsnap.py.
            # topo_search (if enabled) runs FIRST inside refine_layout and gets
            # its own sub-deadline (now + topo_reserve) so it cannot starve the
            # downstream slack projection of the refine window. When
            # FLOORSET_TOPO_SEARCH is off, topo_reserve == 0.0 and
            # topo_deadline is None (no stage runs) -- byte-identical to before.
            topo_deadline = (time.time() + topo_reserve) if topo_reserve > 0.0 else None
            out = refine_layout(out, area_targets, constraints, target_positions,
                                b2b, p2b, pins, deadline=refine_deadline,
                                enable_aspect=os.environ.get("FLOORSET_SLACK_REFINE_ASPECT", "0") == "1",
                                topo_deadline=topo_deadline)
        return out
    except Exception:
        return _fallback_row(area_targets, constraints, target_positions)
