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


def _time_budget(block_count: int) -> float:
    """Exponential per-case budget: ~0.8 s for the smallest cases up to
    ~24 s for n=120; averages about 5 s over the validation set."""
    b = BUDGET_SCALE * math.exp(block_count / BUDGET_TAU)
    return max(BUDGET_MIN, min(BUDGET_MAX, b))


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
    startup cost is not charged to any test case."""
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
    topo_reserve = 0.0
    if (
        os.environ.get("FLOORSET_TOPO_SEARCH", "0") == "1"
        and os.environ.get("FLOORSET_SLACK_REFINE", "0") == "1"
        and block_count >= 60
    ):
        remaining_budget = max(0.0, budget - refine_reserve)
        topo_reserve = min(3.0, 0.25 * remaining_budget)

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
