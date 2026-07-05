"""Evaluator-faithful soft-violation counting and hard-legality check.

Mirrors FloorSet/iccad2026contest/iccad2026_evaluate.py:
  - grouping: lines ~501-506 (shapely unary_union per cluster group)
  - MIB: lines ~511-517 (distinct rounded (w,h) per group)
  - boundary: lines ~519-541 (bbox + tag touch check, 1e-6 tol)
  - overlap: lines ~210-225
  - area tolerance: lines ~228-256
  - fixed/preplaced dims: lines ~259-303
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import box
from shapely.ops import unary_union

from floorset_arch.legalizer.column_slicing import _parse_constraints, _target

Rect = Tuple[float, float, float, float]

BOUND_LEFT = 1
BOUND_RIGHT = 2
BOUND_TOP = 4
BOUND_BOTTOM = 8

BOUNDARY_EPS = 1e-6
OVERLAP_TOL = 1e-6
AREA_TOLERANCE = 0.01
DIM_TOLERANCE = 1e-4


def _soft_violations_impl(positions: Sequence[Rect], constraints) -> Dict[str, object]:
    """Shared internals for `soft_violations` / `soft_violations_detail`.
    Returns a dict with counts plus violating-id detail (evaluator-exact)."""
    n = len(positions)
    fixed, preplaced, mib, cluster, boundary = _parse_constraints(constraints, n)

    v_boundary = 0
    v_grouping = 0
    v_mib = 0
    boundary_violations: List[Tuple[int, int, Dict[int, float]]] = []
    grouping_violation_groups: List[int] = []
    mib_violation_groups: List[int] = []

    # --- boundary ---
    if any(c != 0 for c in boundary):
        x_min_bb = min(p[0] for p in positions)
        y_min_bb = min(p[1] for p in positions)
        x_max_bb = max(p[0] + p[2] for p in positions)
        y_max_bb = max(p[1] + p[3] for p in positions)
        for i in range(n):
            code = boundary[i]
            if code == 0:
                continue
            bx, by, bw, bh = positions[i]
            touches = {
                BOUND_LEFT: abs(bx - x_min_bb) < BOUNDARY_EPS,
                BOUND_RIGHT: abs(bx + bw - x_max_bb) < BOUNDARY_EPS,
                BOUND_TOP: abs(by + bh - y_max_bb) < BOUNDARY_EPS,
                BOUND_BOTTOM: abs(by - y_min_bb) < BOUNDARY_EPS,
            }
            failing_bits = [bit for bit in (BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM)
                            if (code & bit) and not touches[bit]]
            if failing_bits:
                v_boundary += 1
                missing_bits = 0
                distances: Dict[int, float] = {}
                for bit in failing_bits:
                    missing_bits |= bit
                    if bit == BOUND_LEFT:
                        distances[bit] = bx - x_min_bb
                    elif bit == BOUND_RIGHT:
                        distances[bit] = x_max_bb - (bx + bw)
                    elif bit == BOUND_TOP:
                        distances[bit] = y_max_bb - (by + bh)
                    elif bit == BOUND_BOTTOM:
                        distances[bit] = by - y_min_bb
                boundary_violations.append((i, missing_bits, distances))

    # --- grouping (shapely) ---
    n_clust_groups = max(cluster) if cluster else 0
    for g in range(1, n_clust_groups + 1):
        group_indices = [i for i in range(n) if cluster[i] == g]
        if not group_indices:
            continue
        group_polys = [box(x, y, x + w, y + h) for (x, y, w, h) in
                        (positions[i] for i in group_indices)]
        union_result = unary_union(group_polys)
        if union_result.geom_type == "MultiPolygon":
            v_grouping += len(union_result.geoms) - 1
            grouping_violation_groups.append(g)

    # --- MIB ---
    n_mib_groups = max(mib) if mib else 0
    for g in range(1, n_mib_groups + 1):
        group_indices = [i for i in range(n) if mib[i] == g]
        if not group_indices:
            continue
        distinct_shapes = set()
        for i in group_indices:
            bw, bh = round(positions[i][2], 4), round(positions[i][3], 4)
            distinct_shapes.add((bw, bh))
        if len(distinct_shapes) - 1 > 0:
            v_mib += len(distinct_shapes) - 1
            mib_violation_groups.append(g)

    return {
        "v_boundary": v_boundary,
        "v_grouping": v_grouping,
        "v_mib": v_mib,
        "boundary_violations": boundary_violations,
        "grouping_violation_groups": grouping_violation_groups,
        "mib_violation_groups": mib_violation_groups,
    }


def soft_violations(positions: Sequence[Rect], constraints) -> Tuple[int, int, int]:
    """Returns (v_boundary, v_grouping, v_mib), evaluator-exact."""
    detail = _soft_violations_impl(positions, constraints)
    return detail["v_boundary"], detail["v_grouping"], detail["v_mib"]


def soft_violations_detail(positions: Sequence[Rect], constraints) -> Dict[str, object]:
    """Same three counts as `soft_violations` plus violating-id detail:

    - boundary_violations: list of (block_id, missing_bits, {bit: distance})
      where distance is the evaluator-exact gap to the required wall (positive
      when the block is inside the wall, matching the bit's frozen-wall check).
    - grouping_violation_groups: cluster group ids with >1 shapely component.
    - mib_violation_groups: mib group ids with >1 distinct rounded shape.
    """
    return _soft_violations_impl(positions, constraints)


def hard_legal(
    rects: Sequence[Rect],
    area_targets,
    constraints,
    target_positions,
) -> bool:
    """Overlap-, area-, and dimension-hard-constraint check mirroring the
    evaluator. Returns True iff feasible."""
    n = len(rects)

    # --- overlap ---
    for i in range(n):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = rects[j]
            overlap_x = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            overlap_y = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            if overlap_x > OVERLAP_TOL and overlap_y > OVERLAP_TOL:
                return False

    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)
    fixed_or_preplaced = {i for i in range(n) if fixed[i] or preplaced[i]}

    # --- soft area tolerance (non fixed/preplaced blocks) ---
    for i in range(n):
        if i in fixed_or_preplaced:
            continue
        if area_targets is None or i >= len(area_targets):
            continue
        target_area = float(area_targets[i])
        if target_area <= 0:
            continue
        _x, _y, w, h = rects[i]
        actual_area = w * h
        diff = abs(actual_area - target_area) / target_area
        if diff > AREA_TOLERANCE:
            return False

    # --- fixed/preplaced dims (and position for preplaced) ---
    if target_positions is not None:
        for i in fixed_or_preplaced:
            if i >= len(target_positions):
                continue
            tx, ty, tw, th = _target(target_positions, i)
            px, py, pw, ph = rects[i]
            if abs(pw - tw) > DIM_TOLERANCE or abs(ph - th) > DIM_TOLERANCE:
                return False
            if preplaced[i]:
                if abs(px - tx) > DIM_TOLERANCE or abs(py - ty) > DIM_TOLERANCE:
                    return False

    return True
