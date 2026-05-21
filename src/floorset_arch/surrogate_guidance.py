from __future__ import annotations

import math
import os

from floorset_arch.models import AnchorGuidance, Instance, Rect


def _shape_for_anchor(inst: Instance, block: int) -> tuple[float, float]:
    target = inst.target_rects.get(block)
    if target is not None and (block in inst.fixed or block in inst.preplaced):
        return target.width, target.height
    area = max(1.0, float(inst.area_targets[block]))
    return math.sqrt(area), math.sqrt(area)


def _layout_scale(inst: Instance) -> float:
    total_area = sum(max(1.0, float(inst.area_targets[i])) for i in range(inst.block_count))
    scale = math.sqrt(max(total_area, 1.0)) * 1.8
    valid_pins = [
        (float(pin[0]), float(pin[1]))
        for pin in inst.pins_pos.tolist()
        if len(pin) >= 2 and float(pin[0]) >= 0.0 and float(pin[1]) >= 0.0
    ]
    if valid_pins:
        pin_extent = max(max(x for x, _y in valid_pins), max(y for _x, y in valid_pins))
        scale = max(scale, pin_extent)
    return max(scale, 1.0)


def _initial_centers(inst: Instance, scale: float) -> tuple[list[float], list[float], set[int]]:
    cx = [scale * 0.5 for _ in range(inst.block_count)]
    cy = [scale * 0.5 for _ in range(inst.block_count)]
    fixed: set[int] = set()
    for block in range(inst.block_count):
        target = inst.target_rects.get(block)
        if target is not None and block in inst.preplaced:
            cx[block] = target.center_x
            cy[block] = target.center_y
            fixed.add(block)
            continue
        total = 0.0
        sx = 0.0
        sy = 0.0
        for pin, weight in inst.p2b_by_block.get(block, []):
            if 0 <= pin < inst.pins_pos.shape[0]:
                px = float(inst.pins_pos[pin, 0])
                py = float(inst.pins_pos[pin, 1])
                if px >= 0.0 and py >= 0.0:
                    sx += px * weight
                    sy += py * weight
                    total += weight
        if total > 0.0:
            cx[block] = sx / total
            cy[block] = sy / total
            continue
        angle = 2.0 * math.pi * (block / max(inst.block_count, 1))
        radius = scale * (0.18 + 0.16 * ((block % 7) / 6.0))
        cx[block] = scale * 0.5 + radius * math.cos(angle)
        cy[block] = scale * 0.5 + radius * math.sin(angle)
    return cx, cy, fixed


def _boundary_target(code: int, scale: float) -> tuple[float | None, float | None]:
    tx = None
    ty = None
    if code & 1:
        tx = 0.0
    elif code & 2:
        tx = scale
    if code & 8:
        ty = 0.0
    elif code & 4:
        ty = scale
    return tx, ty


def build_surrogate_guidance(inst: Instance) -> AnchorGuidance:
    scale = _layout_scale(inst)
    cx, cy, fixed = _initial_centers(inst, scale)
    iterations = int(os.environ.get("FLOORSET_SURROGATE_GUIDANCE_ITERS", "48"))
    edge_weight_scale = float(os.environ.get("FLOORSET_SURROGATE_EDGE_WEIGHT_SCALE", "1.0"))
    boundary_weight = float(os.environ.get("FLOORSET_SURROGATE_BOUNDARY_WEIGHT", "8.0"))
    pin_weight = float(os.environ.get("FLOORSET_SURROGATE_PIN_WEIGHT", "4.0"))
    cluster_blend = float(os.environ.get("FLOORSET_SURROGATE_CLUSTER_BLEND", "0.22"))

    for _ in range(max(0, iterations)):
        next_x = list(cx)
        next_y = list(cy)
        for block in range(inst.block_count):
            if block in fixed:
                continue
            sx = 0.0
            sy = 0.0
            total = 0.0
            for other, weight in inst.b2b_by_block.get(block, []):
                w = max(0.0, float(weight)) * edge_weight_scale
                sx += cx[other] * w
                sy += cy[other] * w
                total += w
            for pin, weight in inst.p2b_by_block.get(block, []):
                if 0 <= pin < inst.pins_pos.shape[0]:
                    px = float(inst.pins_pos[pin, 0])
                    py = float(inst.pins_pos[pin, 1])
                    if px >= 0.0 and py >= 0.0:
                        w = max(0.0, float(weight)) * pin_weight
                        sx += px * w
                        sy += py * w
                        total += w
            tx, ty = _boundary_target(inst.boundary.get(block, 0), scale)
            if tx is not None:
                sx += tx * boundary_weight
                total += boundary_weight
            if ty is not None:
                sy += ty * boundary_weight
                total += boundary_weight
            if total > 0.0:
                next_x[block] = 0.55 * cx[block] + 0.45 * (sx / total)
                next_y[block] = 0.55 * cy[block] + 0.45 * (sy / total)
        cx, cy = next_x, next_y

        for members in inst.cluster_groups.values():
            movable = [block for block in members if block not in fixed and 0 <= block < inst.block_count]
            if len(movable) <= 1:
                continue
            center_x = sum(cx[block] for block in members if 0 <= block < inst.block_count) / len(members)
            center_y = sum(cy[block] for block in members if 0 <= block < inst.block_count) / len(members)
            for block in movable:
                cx[block] = (1.0 - cluster_blend) * cx[block] + cluster_blend * center_x
                cy[block] = (1.0 - cluster_blend) * cy[block] + cluster_blend * center_y

    guidance = AnchorGuidance(scale=scale, source="surrogate")
    for block in range(inst.block_count):
        width, height = _shape_for_anchor(inst, block)
        x = max(width * 0.5, min(scale - width * 0.5, cx[block]))
        y = max(height * 0.5, min(scale - height * 0.5, cy[block]))
        guidance.rect_priors[block] = Rect(x - width * 0.5, y - height * 0.5, width, height)
        guidance.priority[block] = sum(weight for _other, weight in inst.b2b_by_block.get(block, [])) + sum(
            weight for _pin, weight in inst.p2b_by_block.get(block, [])
        )
    return guidance
