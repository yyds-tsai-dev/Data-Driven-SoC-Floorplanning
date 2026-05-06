from __future__ import annotations

import math

from floorset_arch.constructive import block_dimensions
from floorset_arch.geometry import bbox, candidate_frontier_points, first_non_overlapping, overlaps
from floorset_arch.models import Instance, Placement, Rect, SolverConfig


def _snap_hard(inst: Instance, placement: Placement) -> None:
    for block in inst.fixed | inst.preplaced:
        target = inst.target_rects.get(block)
        if target is None:
            continue
        current = placement.rects.get(block, target)
        if block in inst.preplaced:
            placement.rects[block] = target
        else:
            placement.rects[block] = current.resized(target.width, target.height)


def _mib_compatible(inst: Instance, members: list[int]) -> tuple[float, float] | None:
    areas = [float(inst.area_targets[i]) for i in members if i not in inst.fixed and i not in inst.preplaced]
    fixed_dims = [
        (inst.target_rects[i].width, inst.target_rects[i].height)
        for i in members
        if i in inst.target_rects and (i in inst.fixed or i in inst.preplaced)
    ]
    if fixed_dims:
        ref = fixed_dims[0]
        if any(abs(w - ref[0]) > 1e-4 or abs(h - ref[1]) > 1e-4 for w, h in fixed_dims):
            return None
        if any(abs((ref[0] * ref[1]) - area) / max(area, 1e-6) > 0.01 for area in areas):
            return None
        return ref
    if not areas:
        return None
    ref_area = areas[0]
    if any(abs(area - ref_area) / max(ref_area, 1e-6) > 0.01 for area in areas):
        return None
    side = math.sqrt(ref_area)
    return side, side


def _unify_compatible_mib(inst: Instance, placement: Placement) -> None:
    for members in inst.mib_groups.values():
        dims = _mib_compatible(inst, members)
        if dims is None:
            continue
        width, height = dims
        for block in members:
            if block in inst.preplaced:
                continue
            rect = placement.rects.get(block)
            if rect is not None:
                placement.rects[block] = rect.resized(width, height)


def _move_to_first_slot(inst: Instance, placement: Placement, block: int) -> None:
    rect = placement.rects[block]
    others = [r for i, r in placement.rects.items() if i != block]
    for x, y in candidate_frontier_points(others):
        candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
        if first_non_overlapping(candidate, others):
            placement.rects[block] = candidate
            return
    bounds = bbox(others)
    placement.rects[block] = Rect(bounds.right, bounds.y, rect.width, rect.height)


def _resolve_overlaps(inst: Instance, placement: Placement, config: SolverConfig) -> None:
    for _ in range(config.max_repair_passes):
        changed = False
        for block, rect in list(placement.rects.items()):
            if block in inst.preplaced:
                continue
            if any(overlaps(rect, other) for other_idx, other in placement.rects.items() if other_idx != block):
                _move_to_first_slot(inst, placement, block)
                changed = True
        if not changed:
            return


def _repair_boundary(inst: Instance, placement: Placement) -> None:
    if not inst.boundary:
        return
    bounds = bbox(list(placement.rects.values()))
    for block, code in inst.boundary.items():
        if block in inst.preplaced or block not in placement.rects:
            continue
        rect = placement.rects[block]
        candidates = []
        x, y = rect.x, rect.y
        if code & 1:
            x = bounds.x
        if code & 2:
            x = bounds.right - rect.width
        if code & 8:
            y = bounds.y
        if code & 4:
            y = bounds.top - rect.height
        candidates.append(Rect(max(0.0, x), max(0.0, y), rect.width, rect.height))
        others = [r for i, r in placement.rects.items() if i != block]
        for candidate in candidates:
            if first_non_overlapping(candidate, others):
                placement.rects[block] = candidate
                break


def repair_placement(inst: Instance, placement: Placement, config: SolverConfig | None = None) -> Placement:
    config = config or SolverConfig()
    repaired = placement.copy()
    _snap_hard(inst, repaired)
    _unify_compatible_mib(inst, repaired)
    _resolve_overlaps(inst, repaired, config)
    _repair_boundary(inst, repaired)
    _snap_hard(inst, repaired)
    _resolve_overlaps(inst, repaired, config)

    for block in range(inst.block_count):
        if block not in repaired.rects:
            width, height = block_dimensions(inst, block)
            repaired.rects[block] = Rect(0.0, 0.0, width, height)
            _move_to_first_slot(inst, repaired, block)

    return repaired

