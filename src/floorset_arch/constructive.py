from __future__ import annotations

import math

from floorset_arch.geometry import candidate_frontier_points, first_non_overlapping, bbox
from floorset_arch.models import Instance, Placement, Rect, SolverConfig
from floorset_arch.scoring import placement_score


def block_dimensions(inst: Instance, block: int) -> tuple[float, float]:
    target = inst.target_rects.get(block)
    if target is not None and (block in inst.fixed or block in inst.preplaced):
        return target.width, target.height
    area = float(inst.area_targets[block])
    if area <= 0:
        area = 1.0
    return math.sqrt(area), math.sqrt(area)


def _priority(inst: Instance, block: int) -> tuple[int, int, float]:
    constraint_score = 0
    if block in inst.boundary:
        constraint_score += 4
    if any(block in members for members in inst.cluster_groups.values()):
        constraint_score += 2
    if any(block in members for members in inst.mib_groups.values()):
        constraint_score += 1
    degree = int((inst.valid_b2b[:, :2] == block).sum().item()) if inst.valid_b2b.numel() else 0
    return (-constraint_score, -degree, -float(inst.area_targets[block]))


def _fallback_point(rects: dict[int, Rect]) -> tuple[float, float]:
    bounds = bbox(list(rects.values()))
    return bounds.right, bounds.y


def _boundary_points(code: int, width: float, height: float, rects: dict[int, Rect]) -> list[tuple[float, float]]:
    if not rects or code == 0:
        return []
    bounds = bbox(list(rects.values()))
    points = []
    if code & 1:
        points.append((bounds.x, bounds.top))
        points.append((bounds.x, bounds.y))
    if code & 2:
        points.append((bounds.right, bounds.y))
        points.append((bounds.right, bounds.top))
    if code & 4:
        points.append((bounds.x, bounds.top))
        points.append((max(bounds.x, bounds.right - width), bounds.top))
    if code & 8:
        points.append((bounds.x, bounds.y))
        points.append((max(bounds.x, bounds.right - width), bounds.y))
    return points


def construct_initial_placement(inst: Instance, config: SolverConfig | None = None) -> Placement:
    config = config or SolverConfig()
    placement = Placement()

    for block in sorted(inst.preplaced):
        target = inst.target_rects.get(block)
        if target is not None:
            placement.rects[block] = target

    remaining = [i for i in range(inst.block_count) if i not in placement.rects]
    remaining.sort(key=lambda block: _priority(inst, block))

    for block in remaining:
        width, height = block_dimensions(inst, block)
        placed_rects = list(placement.rects.values())
        candidates = []
        if inst.model_hints and block in inst.model_hints:
            hint = inst.model_hints[block]
            candidates.append((hint.x, hint.y))
        candidates.extend(_boundary_points(inst.boundary.get(block, 0), width, height, placement.rects))
        candidates.extend(candidate_frontier_points(placed_rects))

        best: Rect | None = None
        best_score = float("inf")
        seen = set()
        for x, y in candidates[: config.max_candidates_per_block * 2]:
            key = (round(float(x), 6), round(float(y), 6))
            if key in seen:
                continue
            seen.add(key)
            rect = Rect(max(0.0, float(x)), max(0.0, float(y)), width, height)
            if not first_non_overlapping(rect, placed_rects):
                continue
            score = placement_score(inst, block, rect, placement.rects)
            if score < best_score:
                best = rect
                best_score = score

        if best is None:
            x, y = _fallback_point(placement.rects)
            best = Rect(x, y, width, height)
            while not first_non_overlapping(best, list(placement.rects.values())):
                y = best.top
                best = Rect(x, y, width, height)
        placement.rects[block] = best

    return placement

