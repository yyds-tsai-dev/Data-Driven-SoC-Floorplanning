"""Phase 2 -- aspect moves (FLOORSET_SLACK_REFINE_ASPECT).

See docs/design/slack_refiner_spec.md section 3. SOFT blocks only (never
RIGID/LOCKED): propose w' = clamp(w*sqrt(rho)) for rho in a small fixed set,
h' = area/w' exactly (soft area stays machine-exact by construction), then
re-clamp position bounds against the frozen axis-separation DAGs (bbox stays
pinned) and accept only if the guarded HPWL strictly improves and every
soft/hard guard from Phase 1 (section 4/5/6) still holds. MIB groups share
one (w,h) literal, moved jointly. Lazy import from api.py -- this module is
only touched when enable_aspect is True.
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, List, Optional, Sequence, Tuple

from .constraint_graph import build_axis_dags
from .guards import hard_legal, soft_violations
from .wirelength import hpwl

Rect = Tuple[float, float, float, float]

RHOS = (0.8, 0.9, 1.11, 1.25)
AREA_ERROR_CAP = 0.009  # spec section 3: stay under 0.9% (evaluator allows 1%)
MAX_PROPOSALS_PER_BLOCK_FACTOR = 4  # spec section 2 budget note: ~4*n proposals per case
MAX_OUTER_PASSES = 2


def _bbox_area(rects: Sequence[Rect]) -> float:
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return (x_max - x_min) * (y_max - y_min)


def _connectivity_weight(n: int, b2b_edges, p2b_edges) -> List[float]:
    w = [0.0] * n
    for edge in b2b_edges:
        i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1 or not (0 <= i < n and 0 <= j < n):
            continue
        w[i] += weight
        w[j] += weight
    for edge in p2b_edges:
        p, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if p == -1 or not (0 <= b < n):
            continue
        w[b] += weight
    return w


def _neighbor_bounds(
    graph_edges: Sequence[Tuple[int, int, float]],
    coords: Sequence[float],
    i: int,
) -> Tuple[float, float]:
    """lo/hi bound on coords[i] from this block's own DAG edges only
    (single-block move, not a rigid group -- aspect moves are per-block)."""
    lo = float("-inf")
    hi = float("inf")
    for u, v, gap in graph_edges:
        if v == i:
            lo = max(lo, coords[u] + gap)
        elif u == i:
            hi = min(hi, coords[v] - gap)
    return lo, hi


def _try_resize_block(
    candidate: List[Rect],
    i: int,
    new_w: float,
    new_h: float,
    gx_edges,
    gy_edges,
    wall_lo_x: float,
    wall_hi_x: float,
    wall_lo_y: float,
    wall_hi_y: float,
    pinned_x: Optional[float],
    pinned_y: Optional[float],
) -> Optional[Rect]:
    """Attempt to place a resized block i, keeping its centroid as close as
    possible to its current centroid, clamped to its own DAG bounds (edges
    recomputed fresh against the current candidate layout) and the frozen
    bbox walls. Returns the new rect, or None if no feasible placement
    exists for this size (over-constrained)."""
    x, y, w, h = candidate[i]
    cx = x + w / 2.0
    cy = y + h / 2.0

    xs = [r[0] for r in candidate]
    ys = [r[1] for r in candidate]

    lo_x, hi_x = _neighbor_bounds(gx_edges, xs, i)
    lo_x = max(lo_x, wall_lo_x)
    hi_x = min(hi_x, wall_hi_x - new_w)
    if pinned_x is not None:
        lo_x = hi_x = pinned_x

    lo_y, hi_y = _neighbor_bounds(gy_edges, ys, i)
    lo_y = max(lo_y, wall_lo_y)
    hi_y = min(hi_y, wall_hi_y - new_h)
    if pinned_y is not None:
        lo_y = hi_y = pinned_y

    if lo_x > hi_x + 1e-9 or lo_y > hi_y + 1e-9:
        return None

    new_x = min(max(cx - new_w / 2.0, lo_x), hi_x)
    new_y = min(max(cy - new_h / 2.0, lo_y), hi_y)
    return (new_x, new_y, new_w, new_h)


def refine_aspect(
    rects: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges: Sequence[Tuple[int, int, float]],
    p2b_edges: Sequence[Tuple[int, int, float]],
    pins: Sequence[Tuple[float, float]],
    deadline: Optional[float] = None,
) -> List[Rect]:
    """Phase 2 aspect-move pass. Pure function: returns `rects` unchanged on
    any exception, and never accepts a candidate that regresses HPWL, bbox,
    the three soft-violation counts, or hard legality relative to the
    layout it started from (which is Phase 1's already-accepted output)."""
    original = list(rects)
    try:
        from floorset_arch.legalizer.column_slicing import _parse_constraints as parse_constraints
    except Exception:
        return original

    try:
        n = len(original)
        if n == 0:
            return original

        fixed, preplaced, mib, cluster, boundary = parse_constraints(constraints, n)
        movable = [not (fixed[i] or preplaced[i]) for i in range(n)]

        baseline_bbox = _bbox_area(original)
        baseline_hpwl = hpwl(original, b2b_edges, p2b_edges, pins)
        baseline_soft = soft_violations(original, constraints)

        candidate = list(original)
        weights = _connectivity_weight(n, b2b_edges, p2b_edges)
        order = sorted(range(n), key=lambda i: -weights[i])

        # MIB groups: one shared (w, h) literal per group, only touched via
        # the group's lowest-index representative to avoid double-proposing.
        mib_groups: Dict[int, List[int]] = {}
        for i in range(n):
            g = mib[i]
            if g > 0:
                mib_groups.setdefault(g, []).append(i)
        mib_rep_of: Dict[int, int] = {}
        for g, members in mib_groups.items():
            rep = min(members)
            for m in members:
                mib_rep_of[m] = rep

        proposal_budget = MAX_PROPOSALS_PER_BLOCK_FACTOR * n
        proposals_used = 0

        for _outer in range(MAX_OUTER_PASSES):
            any_accepted_this_pass = False
            for i in order:
                if deadline is not None and _time.time() >= deadline:
                    return candidate
                if proposals_used >= proposal_budget:
                    break
                if not movable[i]:
                    continue
                # Only the MIB-group representative proposes; members follow.
                if i in mib_rep_of and mib_rep_of[i] != i:
                    continue

                area_i = float(area_targets[i]) if area_targets is not None and i < len(area_targets) else None
                if area_i is None or area_i <= 0:
                    continue

                x, y, w, h = candidate[i]
                group_members = mib_groups.get(mib[i], [i]) if mib[i] > 0 else [i]

                gx, gy = build_axis_dags(candidate, constraints, target_positions,
                                          b2b_edges, p2b_edges, pins)

                best_candidate_layout = None
                best_hpwl = None

                for rho in RHOS:
                    proposals_used += 1
                    new_w = w * math.sqrt(rho)
                    if new_w <= 1e-9:
                        continue
                    new_h = area_i / new_w
                    if new_h <= 1e-9:
                        continue
                    area_err = abs(new_w * new_h - area_i) / area_i
                    if area_err > AREA_ERROR_CAP:
                        continue

                    trial = list(candidate)
                    ok = True
                    for m in group_members:
                        if not movable[m]:
                            ok = False
                            break
                        pinned_x = gx.pinned.get(m)
                        pinned_y = gy.pinned.get(m)
                        placed = _try_resize_block(
                            trial, m, new_w, new_h,
                            gx.edges, gy.edges,
                            gx.wall_lo, gx.wall_hi, gy.wall_lo, gy.wall_hi,
                            pinned_x, pinned_y,
                        )
                        if placed is None:
                            ok = False
                            break
                        trial[m] = placed
                    if not ok:
                        continue

                    # Guards: dims of non-movable (fixed/preplaced) blocks
                    # must not be touched (movable[] already excludes them
                    # from resize/reposition, this re-verifies no drift).
                    dims_ok = all(
                        trial[k][2] == candidate[k][2] and trial[k][3] == candidate[k][3]
                        for k in range(n) if not movable[k]
                    )
                    if not dims_ok:
                        continue

                    trial_bbox = _bbox_area(trial)
                    if trial_bbox > baseline_bbox + 1e-6:
                        continue

                    trial_soft = soft_violations(trial, constraints)
                    if any(c > b for c, b in zip(trial_soft, baseline_soft)):
                        continue

                    if not hard_legal(trial, area_targets, constraints, target_positions):
                        continue

                    trial_hpwl = hpwl(trial, b2b_edges, p2b_edges, pins)
                    if trial_hpwl >= baseline_hpwl - 1e-9:
                        continue
                    if trial_hpwl >= (best_hpwl if best_hpwl is not None else float("inf")):
                        continue

                    best_hpwl = trial_hpwl
                    best_candidate_layout = trial

                if best_candidate_layout is not None:
                    candidate = best_candidate_layout
                    baseline_hpwl = best_hpwl
                    baseline_bbox = _bbox_area(candidate)
                    baseline_soft = soft_violations(candidate, constraints)
                    any_accepted_this_pass = True

            if not any_accepted_this_pass:
                break

        # Final re-check mirroring the Phase 1 failure-containment policy:
        # any regression vs the original (pre-aspect) layout means "no-op".
        final_bbox = _bbox_area(candidate)
        final_hpwl = hpwl(candidate, b2b_edges, p2b_edges, pins)
        orig_hpwl = hpwl(original, b2b_edges, p2b_edges, pins)
        orig_bbox = _bbox_area(original)
        orig_soft = soft_violations(original, constraints)
        final_soft = soft_violations(candidate, constraints)

        if final_hpwl >= orig_hpwl - 1e-9:
            return original
        if final_bbox > orig_bbox + 1e-6:
            return original
        if any(c > b for c, b in zip(final_soft, orig_soft)):
            return original
        if not hard_legal(candidate, area_targets, constraints, target_positions):
            return original

        return candidate

    except Exception:
        return original
