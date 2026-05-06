from __future__ import annotations

from floorset_arch.geometry import bbox, boundary_satisfied, edge_touch_length
from floorset_arch.models import Instance, Rect


def _center(rect: Rect) -> tuple[float, float]:
    return rect.center_x, rect.center_y


def hpwl_proxy(inst: Instance, rects: dict[int, Rect]) -> float:
    total = 0.0
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if i not in rects or j not in rects:
            continue
        cx_i, cy_i = _center(rects[i])
        cx_j, cy_j = _center(rects[j])
        total += float(weight_f) * (abs(cx_i - cx_j) + abs(cy_i - cy_j))
    for pin_f, block_f, weight_f in inst.valid_p2b.tolist():
        pin_idx, block_idx = int(pin_f), int(block_f)
        if block_idx not in rects or pin_idx >= inst.pins_pos.shape[0]:
            continue
        cx, cy = _center(rects[block_idx])
        px = float(inst.pins_pos[pin_idx, 0])
        py = float(inst.pins_pos[pin_idx, 1])
        total += float(weight_f) * (abs(cx - px) + abs(cy - py))
    return total


def placement_score(inst: Instance, candidate_block: int, candidate: Rect, rects: dict[int, Rect]) -> float:
    trial = dict(rects)
    trial[candidate_block] = candidate
    bounds = bbox(list(trial.values()))
    area_term = bounds.area * 0.02
    hpwl_term = hpwl_proxy(inst, trial)

    soft_bonus = 0.0
    code = inst.boundary.get(candidate_block, 0)
    if code and boundary_satisfied(candidate, bounds, code):
        soft_bonus -= 100.0
    for members in inst.cluster_groups.values():
        if candidate_block not in members:
            continue
        for other_idx in members:
            other = rects.get(other_idx)
            if other is not None:
                soft_bonus -= 10.0 * edge_touch_length(candidate, other)
    return hpwl_term + area_term + soft_bonus

