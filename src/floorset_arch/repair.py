from __future__ import annotations

import math

from floorset_arch.constructive import block_dimensions
from floorset_arch.geometry import bbox, candidate_frontier_points, edge_touch_length, first_non_overlapping, overlaps
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
        others = [r for i, r in placement.rects.items() if i != block]
        xs = {rect.x, bounds.x, max(bounds.x, bounds.right - rect.width)}
        ys = {rect.y, bounds.y, max(bounds.y, bounds.top - rect.height)}
        for other in others:
            xs.update({other.x, other.right, other.right - rect.width})
            ys.update({other.y, other.top, other.top - rect.height})
        if code & 1:
            xs = {bounds.x}
        elif code & 2:
            xs = {max(bounds.x, bounds.right - rect.width)}
        if code & 8:
            ys = {bounds.y}
        elif code & 4:
            ys = {max(bounds.y, bounds.top - rect.height)}

        candidates = []
        for x in sorted(xs, key=lambda value: abs(value - rect.x)):
            for y in sorted(ys, key=lambda value: abs(value - rect.y)):
                candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
                trial_bounds = bbox([*others, candidate])
                if boundary_satisfied_local(candidate, trial_bounds, code):
                    candidates.append(candidate)
        best: Rect | None = None
        best_score = float("inf")
        for candidate in candidates:
            if not first_non_overlapping(candidate, others):
                continue
            trial_bounds = bbox([*others, candidate])
            move = abs(candidate.x - rect.x) + abs(candidate.y - rect.y)
            score = trial_bounds.area + 0.05 * move
            if score < best_score:
                best = candidate
                best_score = score
        if best is not None:
            placement.rects[block] = best


def boundary_satisfied_local(rect: Rect, bounds: Rect, code: int) -> bool:
    return (
        (not (code & 1) or abs(rect.x - bounds.x) <= 1e-6)
        and (not (code & 2) or abs(rect.right - bounds.right) <= 1e-6)
        and (not (code & 4) or abs(rect.top - bounds.top) <= 1e-6)
        and (not (code & 8) or abs(rect.y - bounds.y) <= 1e-6)
    )


def _cluster_components(inst: Instance, placement: Placement, members: list[int]) -> list[list[int]]:
    present = [block for block in members if block in placement.rects]
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in present:
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        stack = [start]
        while stack:
            cur = stack.pop()
            cur_rect = placement.rects[cur]
            for other in present:
                if other in seen:
                    continue
                if edge_touch_length(cur_rect, placement.rects[other]) > 0.0:
                    seen.add(other)
                    stack.append(other)
                    comp.append(other)
        components.append(comp)
    return components


def _adjacent_positions(anchor: Rect, width: float, height: float) -> list[tuple[float, float]]:
    return [
        (anchor.right, anchor.y),
        (anchor.x - width, anchor.y),
        (anchor.x, anchor.top),
        (anchor.x, anchor.y - height),
        (anchor.right, anchor.top - height),
        (anchor.right - width, anchor.top),
    ]


def _connect_clusters(inst: Instance, placement: Placement, config: SolverConfig) -> None:
    for members in inst.cluster_groups.values():
        for _ in range(max(1, min(3, config.max_repair_passes))):
            components = _cluster_components(inst, placement, members)
            if len(components) <= 1:
                break
            anchor_component = max(components, key=len)
            moved = False
            anchors = [placement.rects[block] for block in anchor_component]
            for comp in components:
                if comp is anchor_component:
                    continue
                for block in sorted(comp, key=lambda idx: idx in inst.preplaced):
                    if block in inst.preplaced:
                        continue
                    rect = placement.rects[block]
                    others = [r for i, r in placement.rects.items() if i != block]
                    candidates = []
                    for anchor in anchors:
                        candidates.extend(_adjacent_positions(anchor, rect.width, rect.height))
                    best: Rect | None = None
                    best_score = float("inf")
                    for x, y in candidates:
                        candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
                        if not first_non_overlapping(candidate, others):
                            continue
                        touch = max(edge_touch_length(candidate, anchor) for anchor in anchors)
                        bounds = bbox([*others, candidate])
                        score = bounds.area - 1000.0 * touch
                        if score < best_score:
                            best = candidate
                            best_score = score
                    if best is not None:
                        placement.rects[block] = best
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break


def _compact_left_down(inst: Instance, placement: Placement) -> None:
    for axis in ("x", "y"):
        movable = sorted(
            [block for block in placement.rects if block not in inst.preplaced],
            key=lambda block: getattr(placement.rects[block], axis),
        )
        for block in movable:
            rect = placement.rects[block]
            others = [r for i, r in placement.rects.items() if i != block]
            values = {0.0}
            for other in others:
                values.add(other.right if axis == "x" else other.top)
            best = rect
            for value in sorted(v for v in values if v <= getattr(rect, axis) + 1e-6):
                candidate = Rect(value, rect.y, rect.width, rect.height) if axis == "x" else Rect(rect.x, value, rect.width, rect.height)
                if first_non_overlapping(candidate, others):
                    best = candidate
                    break
            placement.rects[block] = best


def repair_placement(inst: Instance, placement: Placement, config: SolverConfig | None = None) -> Placement:
    config = config or SolverConfig()
    repaired = placement.copy()
    _snap_hard(inst, repaired)
    _unify_compatible_mib(inst, repaired)
    _resolve_overlaps(inst, repaired, config)
    _connect_clusters(inst, repaired, config)
    _compact_left_down(inst, repaired)
    _repair_boundary(inst, repaired)
    _snap_hard(inst, repaired)
    _resolve_overlaps(inst, repaired, config)
    _connect_clusters(inst, repaired, config)
    _repair_boundary(inst, repaired)

    for block in range(inst.block_count):
        if block not in repaired.rects:
            width, height = block_dimensions(inst, block)
            repaired.rects[block] = Rect(0.0, 0.0, width, height)
            _move_to_first_slot(inst, repaired, block)

    return repaired
