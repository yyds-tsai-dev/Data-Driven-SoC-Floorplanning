from __future__ import annotations

import math
import os

from floorset_arch.constructive import block_dimensions
from floorset_arch.geometry import bbox, candidate_frontier_points, edge_touch_length, first_non_overlapping, overlaps
from floorset_arch.models import Instance, Placement, Rect, SolverConfig
from floorset_arch.scoring import candidate_hpwl_proxy, hpwl_proxy


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


def _move_to_best_slot(inst: Instance, placement: Placement, block: int) -> None:
    rect = placement.rects[block]
    others = [r for i, r in placement.rects.items() if i != block]
    best: Rect | None = None
    best_score = float("inf")
    limit = int(os.environ.get("FLOORSET_OVERLAP_REPAIR_CANDIDATES", "24"))
    for x, y in candidate_frontier_points(others):
        candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
        if not first_non_overlapping(candidate, others):
            continue
        move = abs(candidate.x - rect.x) + abs(candidate.y - rect.y)
        score = _relocation_proxy(inst, placement, block, candidate) + 0.01 * move
        if score < best_score:
            best = candidate
            best_score = score
        limit -= 1
        if limit <= 0:
            break
    if best is not None:
        placement.rects[block] = best
        return
    bounds = bbox(others)
    placement.rects[block] = Rect(bounds.right, bounds.y, rect.width, rect.height)


def _relocation_proxy(inst: Instance, placement: Placement, block: int, candidate: Rect) -> float:
    others = {idx: rect for idx, rect in placement.rects.items() if idx != block}
    bounds = bbox([*others.values(), candidate])
    score = 0.018 * max(bounds.area, 1.0)
    score += 0.0025 * candidate_hpwl_proxy(inst, block, candidate, others)

    code = inst.boundary.get(block, 0)
    if code and not boundary_satisfied_local(candidate, bounds, code):
        score += 2500.0

    for members in inst.cluster_groups.values():
        if block not in members:
            continue
        placed = [others[idx] for idx in members if idx in others]
        if not placed:
            continue
        touch = max(edge_touch_length(candidate, other) for other in placed)
        if touch <= 0.0:
            score += 2500.0
        else:
            score -= 1000.0 * touch

    for members in inst.mib_groups.values():
        if block not in members:
            continue
        shapes = {
            (round(others[idx].width, 5), round(others[idx].height, 5))
            for idx in members
            if idx in others
        }
        if shapes and (round(candidate.width, 5), round(candidate.height, 5)) not in shapes:
            score += 2500.0
    return score


def _resolve_overlaps(inst: Instance, placement: Placement, config: SolverConfig) -> None:
    for _ in range(config.max_repair_passes):
        changed = False
        for block, rect in list(placement.rects.items()):
            if block in inst.preplaced:
                continue
            if any(overlaps(rect, other) for other_idx, other in placement.rects.items() if other_idx != block):
                _move_to_best_slot(inst, placement, block)
                changed = True
        if not changed:
            return


def _repair_boundary(inst: Instance, placement: Placement) -> None:
    if not inst.boundary:
        return
    bounds = bbox(list(placement.rects.values()))
    cap_override = os.environ.get("FLOORSET_BOUNDARY_AXIS_CAP")
    if cap_override:
        axis_cap = int(cap_override)
    else:
        axis_cap = 24 if len(placement.rects) >= 100 else 160
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

        xs = set(_nearest_axis_values(xs, rect.x, axis_cap))
        ys = set(_nearest_axis_values(ys, rect.y, axis_cap))
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


def _boundary_distance(rect: Rect, bounds: Rect, code: int) -> float:
    distance = 0.0
    if code & 1:
        distance += abs(rect.x - bounds.x)
    if code & 2:
        distance += abs(rect.right - bounds.right)
    if code & 8:
        distance += abs(rect.y - bounds.y)
    if code & 4:
        distance += abs(rect.top - bounds.top)
    return distance


def _same_cluster_component(inst: Instance, placement: Placement, block: int) -> list[int]:
    for members in inst.cluster_groups.values():
        if block not in members:
            continue
        for component in _cluster_components(inst, placement, members):
            if block in component:
                return [] if any(member in inst.preplaced for member in component) else component
    return [] if block in inst.preplaced else [block]


def _boundary_component_shifts(inst: Instance, placement: Placement, component: list[int]) -> list[tuple[float, float]]:
    bounds = bbox(list(placement.rects.values()))
    shifts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def add(dx: float, dy: float) -> None:
        key = (round(dx, 7), round(dy, 7))
        if key not in seen and (abs(dx) > 1e-10 or abs(dy) > 1e-10):
            seen.add(key)
            shifts.append((dx, dy))

    for block in component:
        code = inst.boundary.get(block, 0)
        if not code:
            continue
        rect = placement.rects[block]
        dxs = [0.0]
        dys = [0.0]
        if code & 1:
            dxs.append(bounds.x - rect.x)
        if code & 2:
            dxs.append(bounds.right - rect.right)
        if code & 8:
            dys.append(bounds.y - rect.y)
        if code & 4:
            dys.append(bounds.top - rect.top)
        for dx in dxs:
            for dy in dys:
                add(dx, dy)

    def shift_key(shift: tuple[float, float]) -> tuple[int, float]:
        dx, dy = shift
        satisfied = 0
        for block in component:
            code = inst.boundary.get(block, 0)
            if not code:
                continue
            rect = placement.rects[block]
            moved = Rect(rect.x + dx, rect.y + dy, rect.width, rect.height)
            if boundary_satisfied_local(moved, bounds, code):
                satisfied += 1
        return (-satisfied, abs(dx) + abs(dy))

    return sorted(shifts, key=shift_key)[:24]


def _shifted_component_placement(
    placement: Placement,
    component: list[int],
    dx: float,
    dy: float,
) -> Placement | None:
    trial_rects: dict[int, Rect] = {}
    for block in component:
        rect = placement.rects[block]
        nx = rect.x + dx
        ny = rect.y + dy
        if nx < -1e-8 or ny < -1e-8:
            return None
        trial_rects[block] = Rect(max(0.0, nx), max(0.0, ny), rect.width, rect.height)
    if not _component_shift_is_legal(placement, set(component), trial_rects):
        return None
    trial = placement.copy()
    trial.rects.update(trial_rects)
    return trial


def _snap_boundary_components(inst: Instance, placement: Placement, config: SolverConfig) -> None:
    if not inst.boundary:
        return
    budget = max(0, min(config.max_boundary_component_snaps, 24))
    bounds = bbox(list(placement.rects.values()))
    candidates: list[tuple[float, float, int]] = []
    for block, code in inst.boundary.items():
        rect = placement.rects.get(block)
        if rect is None or block in inst.preplaced or boundary_satisfied_local(rect, bounds, code):
            continue
        candidates.append((-_boundary_distance(rect, bounds, code), -rect.area, block))
    candidates.sort()

    used = 0
    for _neg_distance, _neg_area, block in candidates:
        if used >= budget:
            break
        component = _same_cluster_component(inst, placement, block)
        if not component:
            continue
        current = placement.copy()
        best = current
        for dx, dy in _boundary_component_shifts(inst, placement, component):
            trial = _shifted_component_placement(placement, component, dx, dy)
            if trial is not None and _score_better_soft_fast(inst, config, trial, best):
                best = trial
        used += 1
        if best is not current:
            placement.rects = best.rects


def _nearest_axis_values(values: set[float], current: float, limit: int) -> list[float]:
    if len(values) <= limit:
        return list(values)
    ordered = sorted(values, key=lambda value: (abs(value - current), value))
    return ordered[:limit]


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


def soft_violation_counts(inst: Instance, placement: Placement) -> tuple[int, int, int]:
    bounds = bbox(list(placement.rects.values()))
    boundary = 0
    for block, code in inst.boundary.items():
        rect = placement.rects.get(block)
        if rect is None or not boundary_satisfied_local(rect, bounds, code):
            boundary += 1

    grouping = 0
    for members in inst.cluster_groups.values():
        components = _cluster_components(inst, placement, members)
        grouping += max(0, len(components) - 1)

    mib = 0
    for members in inst.mib_groups.values():
        shapes = {
            (round(placement.rects[block].width, 5), round(placement.rects[block].height, 5))
            for block in members
            if block in placement.rects
        }
        mib += max(0, len(shapes) - 1)
    return boundary, grouping, mib


def _placement_proxy(inst: Instance, placement: Placement) -> float:
    bounds = bbox(list(placement.rects.values()))
    area = max(bounds.area, 1.0)
    soft = sum(soft_violation_counts(inst, placement))
    try:
        hpwl = hpwl_proxy(inst, placement.rects)
    except Exception:
        hpwl = 0.0
    return 0.0025 * hpwl + 0.018 * area + 2500.0 * soft


def _area_soft_proxy(inst: Instance, placement: Placement) -> float:
    bounds = bbox(list(placement.rects.values()))
    area = max(bounds.area, 1.0)
    return 0.018 * area + 2500.0 * sum(soft_violation_counts(inst, placement))


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
                if not any(block in inst.preplaced for block in comp):
                    shifted = _try_connect_component(inst, placement, comp, anchor_component, config)
                    if shifted:
                        moved = True
                        break
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


def _try_connect_component(
    inst: Instance,
    placement: Placement,
    component: list[int],
    anchor_component: list[int],
    config: SolverConfig,
) -> bool:
    shifts: list[tuple[float, float]] = []
    for block in component[: max(1, min(len(component), 12))]:
        rect = placement.rects[block]
        for anchor_block in anchor_component[: max(1, min(len(anchor_component), 12))]:
            anchor = placement.rects[anchor_block]
            for x, y in _adjacent_positions(anchor, rect.width, rect.height):
                shifts.append((x - rect.x, y - rect.y))

    seen: set[tuple[float, float]] = set()
    best_shift: tuple[float, float] | None = None
    best_score = float("inf")
    anchor_rects = [placement.rects[block] for block in anchor_component]
    old_rects = {block: placement.rects[block] for block in component}
    limit = max(1, config.max_pair_candidates_per_component)

    checked = 0
    for dx, dy in sorted(shifts, key=lambda s: abs(s[0]) + abs(s[1])):
        key = (round(dx, 6), round(dy, 6))
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        trial_rects = {
            block: Rect(max(0.0, rect.x + dx), max(0.0, rect.y + dy), rect.width, rect.height)
            for block, rect in old_rects.items()
        }
        if not _component_shift_is_legal(placement, set(component), trial_rects):
            if checked >= limit:
                break
            continue
        touch = 0.0
        for rect in trial_rects.values():
            touch = max(touch, max(edge_touch_length(rect, anchor) for anchor in anchor_rects))
        trial_all = [rect for block, rect in placement.rects.items() if block not in trial_rects]
        trial_all.extend(trial_rects.values())
        bounds = bbox(trial_all)
        score = bounds.area + 0.05 * (abs(dx) + abs(dy)) - 2500.0 * touch
        if score < best_score:
            best_score = score
            best_shift = (dx, dy)
        if checked >= limit:
            break

    if best_shift is None:
        return False
    dx, dy = best_shift
    for block, rect in old_rects.items():
        placement.rects[block] = Rect(max(0.0, rect.x + dx), max(0.0, rect.y + dy), rect.width, rect.height)
    return True


def _component_shift_is_legal(
    placement: Placement,
    component: set[int],
    trial_rects: dict[int, Rect],
) -> bool:
    moved = list(trial_rects.values())
    for i, rect in enumerate(moved):
        for other in moved[i + 1 :]:
            if overlaps(rect, other):
                return False
    for block, rect in trial_rects.items():
        for other_block, other in placement.rects.items():
            if other_block in component:
                continue
            if overlaps(rect, other):
                return False
    return True


def _score_better_soft_first(inst: Instance, config: SolverConfig, candidate: Placement, current: Placement) -> bool:
    cand_counts = soft_violation_counts(inst, candidate)
    cur_counts = soft_violation_counts(inst, current)
    cand_soft = sum(cand_counts)
    cur_soft = sum(cur_counts)
    cand_proxy = _placement_proxy(inst, candidate)
    cur_proxy = _placement_proxy(inst, current)
    if cand_soft < cur_soft:
        return cand_proxy <= cur_proxy * (1.0 + config.soft_proxy_slack)
    if cand_soft == cur_soft:
        if cand_counts[0] < cur_counts[0] and cand_proxy <= cur_proxy * 1.12:
            return True
        if cand_counts[0] == cur_counts[0] and cand_counts[1] < cur_counts[1] and cand_proxy <= cur_proxy * 1.10:
            return True
        return cand_proxy <= cur_proxy * (1.0 - config.equal_soft_proxy_slack)
    return False


def _score_better_soft_fast(inst: Instance, config: SolverConfig, candidate: Placement, current: Placement) -> bool:
    cand_counts = soft_violation_counts(inst, candidate)
    cur_counts = soft_violation_counts(inst, current)
    cand_soft = sum(cand_counts)
    cur_soft = sum(cur_counts)
    cand_proxy = _area_soft_proxy(inst, candidate)
    cur_proxy = _area_soft_proxy(inst, current)
    if cand_soft < cur_soft:
        return cand_proxy <= cur_proxy * (1.0 + config.soft_proxy_slack)
    if cand_soft == cur_soft:
        if cand_counts[0] < cur_counts[0] and cand_proxy <= cur_proxy * 1.12:
            return True
        if cand_counts[0] == cur_counts[0] and cand_counts[1] < cur_counts[1] and cand_proxy <= cur_proxy * 1.10:
            return True
        return cand_proxy <= cur_proxy * (1.0 - config.equal_soft_proxy_slack)
    return False


def _guarded_soft_repair(inst: Instance, placement: Placement, config: SolverConfig) -> Placement:
    best = placement.copy()
    boundary_budget = max(0, config.max_boundary_component_snaps)
    cluster_budget = max(0, config.max_cluster_component_moves)

    for _ in range(max(1, min(config.max_repair_passes, 3))):
        trial = best.copy()
        before = best.copy()
        _repair_boundary(inst, trial)
        boundary_budget -= 1
        _connect_clusters(inst, trial, config)
        cluster_budget -= 1
        _resolve_overlaps(inst, trial, config)
        _repair_boundary(inst, trial)
        if _score_better_soft_first(inst, config, trial, best):
            best = trial
        else:
            best = before
            break
        if boundary_budget <= 0 and cluster_budget <= 0:
            break
    return best


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
    _snap_boundary_components(inst, repaired, config)
    _connect_clusters(inst, repaired, config)
    _compact_left_down(inst, repaired)
    _snap_boundary_components(inst, repaired, config)
    _repair_boundary(inst, repaired)
    _snap_hard(inst, repaired)
    _resolve_overlaps(inst, repaired, config)
    _connect_clusters(inst, repaired, config)
    _snap_boundary_components(inst, repaired, config)
    _repair_boundary(inst, repaired)
    repaired = _guarded_soft_repair(inst, repaired, config)

    for block in range(inst.block_count):
        if block not in repaired.rects:
            width, height = block_dimensions(inst, block)
            repaired.rects[block] = Rect(0.0, 0.0, width, height)
            _move_to_best_slot(inst, repaired, block)

    return repaired
