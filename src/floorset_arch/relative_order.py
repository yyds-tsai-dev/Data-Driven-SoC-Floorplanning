from __future__ import annotations

import math
import os

from floorset_arch.models import AnchorGuidance, Instance, Placement, Rect, SolverConfig


def _constraint_id(inst: Instance, block: int, column: int) -> int:
    if inst.constraints is None or inst.constraints.dim() <= 1 or inst.constraints.shape[1] <= column:
        return 0
    return int(float(inst.constraints[block, column]))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _narrow_grouping_pair_bias_enabled() -> bool:
    return os.environ.get("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "").strip().lower() in {"1", "true", "yes", "on"}


def _shape_for_block(inst: Instance, block: int, guidance: AnchorGuidance | None, profile: str) -> tuple[float, float]:
    target = inst.target_rects.get(block)
    if target is not None and (block in inst.fixed or block in inst.preplaced):
        return target.width, target.height

    area = max(1.0, float(inst.area_targets[block]))
    for members in inst.mib_groups.values():
        if block not in members:
            continue
        for other in members:
            target = inst.target_rects.get(other)
            if target is None or other not in (inst.fixed | inst.preplaced):
                continue
            ref_area = target.width * target.height
            if abs(ref_area - area) / max(ref_area, area, 1.0) <= 0.010001:
                return target.width, target.height

    log_aspect = None
    if profile == "wide":
        log_aspect = math.log(float(os.environ.get("FLOORSET_WIDE_PROFILE_ASPECT", "2.0")))
    elif profile == "tall":
        log_aspect = math.log(1.0 / float(os.environ.get("FLOORSET_TALL_PROFILE_ASPECT", "2.0")))
    elif profile != "compact":
        if guidance is not None and block in guidance.log_aspect:
            log_aspect = max(-2.5, min(2.5, guidance.log_aspect[block]))
        elif guidance is not None and block in guidance.rect_priors:
            prior = guidance.rect_priors[block]
            if prior.width > 0 and prior.height > 0:
                log_aspect = math.log(max(0.05, min(20.0, prior.width / prior.height)))
    aspect = 1.0 if log_aspect is None else math.exp(log_aspect)
    return math.sqrt(area * aspect), math.sqrt(area / aspect)


def _fallback_anchor(inst: Instance, block: int, width: float, height: float, scale: float) -> tuple[float, float]:
    weighted_x = 0.0
    weighted_y = 0.0
    total = 0.0
    for pin, weight in inst.p2b_by_block.get(block, []):
        if 0 <= pin < inst.pins_pos.shape[0]:
            px = float(inst.pins_pos[pin, 0])
            py = float(inst.pins_pos[pin, 1])
            if px != -1.0 and py != -1.0:
                weighted_x += px * weight
                weighted_y += py * weight
                total += weight
    if total > 0:
        return weighted_x / total, weighted_y / total
    return scale * 0.5 + width * 0.5, scale * 0.5 + height * 0.5


def _anchors(inst: Instance, widths: list[float], heights: list[float], guidance: AnchorGuidance | None) -> tuple[list[float], list[float]]:
    total_area = sum(widths[i] * heights[i] for i in range(inst.block_count))
    scale = math.sqrt(max(total_area, 1.0))
    raw_x: list[float] = []
    raw_y: list[float] = []
    for block in range(inst.block_count):
        target = inst.target_rects.get(block)
        if target is not None and block in inst.preplaced:
            cx, cy = target.center_x, target.center_y
        elif guidance is not None and block in guidance.rect_priors:
            prior = guidance.rect_priors[block]
            cx, cy = prior.center_x, prior.center_y
        else:
            cx, cy = _fallback_anchor(inst, block, widths[block], heights[block], scale)
        raw_x.append(max(cx, widths[block] * 0.5))
        raw_y.append(max(cy, heights[block] * 0.5))
    return raw_x, raw_y


def _preplaced_frame(inst: Instance) -> tuple[float | None, float | None, float | None, float | None]:
    left = right = bottom = top = None
    for block in inst.preplaced:
        rect = inst.target_rects.get(block)
        code = inst.boundary.get(block, 0)
        if rect is None or code == 0:
            continue
        if code & 1:
            left = rect.x if left is None else min(left, rect.x)
        if code & 2:
            right = rect.right if right is None else max(right, rect.right)
        if code & 8:
            bottom = rect.y if bottom is None else min(bottom, rect.y)
        if code & 4:
            top = rect.top if top is None else max(top, rect.top)
    return left, right, bottom, top


def _clamp_to_preplaced_frame(inst: Instance, raw_x: list[float], raw_y: list[float], widths: list[float], heights: list[float]) -> None:
    left, right, bottom, top = _preplaced_frame(inst)
    for block in range(inst.block_count):
        if block in inst.preplaced:
            continue
        if left is not None and right is not None and right - left >= widths[block]:
            raw_x[block] = min(max(raw_x[block], left + widths[block] * 0.5), right - widths[block] * 0.5)
        elif right is not None:
            raw_x[block] = min(raw_x[block], right - widths[block] * 0.5)
        elif left is not None:
            raw_x[block] = max(raw_x[block], left + widths[block] * 0.5)
        if bottom is not None and top is not None and top - bottom >= heights[block]:
            raw_y[block] = min(max(raw_y[block], bottom + heights[block] * 0.5), top - heights[block] * 0.5)
        elif top is not None:
            raw_y[block] = min(raw_y[block], top - heights[block] * 0.5)
        elif bottom is not None:
            raw_y[block] = max(raw_y[block], bottom + heights[block] * 0.5)


def _bias_order_keys(
    inst: Instance,
    raw_x: list[float],
    raw_y: list[float],
    widths: list[float],
    heights: list[float],
    config: SolverConfig,
    profile: str,
) -> tuple[list[float], list[float]]:
    key_x = list(raw_x)
    key_y = list(raw_y)
    if not key_x:
        return key_x, key_y
    span_x = max(max(raw_x) - min(raw_x), max(widths), 1.0)
    span_y = max(max(raw_y) - min(raw_y), max(heights), 1.0)

    clusters: dict[int, list[int]] = {}
    for block in range(inst.block_count):
        cluster = _constraint_id(inst, block, 3)
        if cluster:
            clusters.setdefault(cluster, []).append(block)
    if profile == "compact" and _grouping_adjacency_bias_enabled(profile) and not _narrow_grouping_pair_bias_enabled():
        blend = max(
            0.0,
            min(
                1.0,
                _env_float("FLOORSET_GROUPING_ADJACENCY_KEY_BLEND", 0.65),
            ),
        )
    else:
        blend = 0.0 if profile == "compact" else 0.18
    for members in clusters.values():
        if len(members) <= 1:
            continue
        cx = sum(key_x[i] for i in members) / len(members)
        cy = sum(key_y[i] for i in members) / len(members)
        for block in members:
            key_x[block] = (1.0 - blend) * key_x[block] + blend * cx
            key_y[block] = (1.0 - blend) * key_y[block] + blend * cy

    offset_x = config.boundary_order_bias * span_x
    offset_y = config.boundary_order_bias * span_y
    edge_counts = {1: 0, 2: 0, 4: 0, 8: 0}
    for block, code in inst.boundary.items():
        if code & 1:
            key_x[block] -= offset_x + edge_counts[1] * max(widths[block], 1.0) * 0.05
            edge_counts[1] += 1
        if code & 2:
            key_x[block] += offset_x + edge_counts[2] * max(widths[block], 1.0) * 0.05
            edge_counts[2] += 1
        if code & 8:
            key_y[block] -= offset_y + edge_counts[8] * max(heights[block], 1.0) * 0.05
            edge_counts[8] += 1
        if code & 4:
            key_y[block] += offset_y + edge_counts[4] * max(heights[block], 1.0) * 0.05
            edge_counts[4] += 1
    return key_x, key_y


def _grouping_adjacency_bias_enabled(profile: str) -> bool:
    enabled = os.environ.get(
        "FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return profile != "compact"
    mode = (
        os.environ.get("FLOORSET_GROUPING_ADJACENCY_BIAS_MODE", "auto")
        .strip()
        .lower()
    )
    if mode == "soft_only":
        return profile != "compact"
    if mode == "compact_only":
        return profile == "compact"
    return True


def _cluster_orientations(inst: Instance, raw_x: list[float], raw_y: list[float]) -> dict[int, str]:
    orientations: dict[int, str] = {}
    for cluster, members in inst.cluster_groups.items():
        if len(members) <= 1:
            orientations[cluster] = "H"
            continue
        span_x = max(raw_x[i] for i in members) - min(raw_x[i] for i in members)
        span_y = max(raw_y[i] for i in members) - min(raw_y[i] for i in members)
        orientations[cluster] = "H" if span_x >= span_y else "V"
    return orientations


def _cluster_grouping_pressure(
    inst: Instance,
    cluster: int,
    members: list[int],
    raw_x: list[float],
    raw_y: list[float],
    widths: list[float],
    heights: list[float],
) -> bool:
    del inst, cluster
    if len(members) < _env_int("FLOORSET_NARROW_GROUPING_MIN_MEMBERS", 3):
        return False
    span_x = max(raw_x[i] for i in members) - min(raw_x[i] for i in members)
    span_y = max(raw_y[i] for i in members) - min(raw_y[i] for i in members)
    avg_w = sum(widths[i] for i in members) / max(len(members), 1)
    avg_h = sum(heights[i] for i in members) / max(len(members), 1)
    pressure = max(span_x / max(avg_w, 1e-6), span_y / max(avg_h, 1e-6))
    return pressure >= _env_float("FLOORSET_NARROW_GROUPING_PRESSURE", 1.4)


def construct_relative_order_placement(inst: Instance, config: SolverConfig | None = None, profile: str = "soft") -> Placement:
    config = config or SolverConfig()
    guidance = inst.anchor_guidance
    widths: list[float] = []
    heights: list[float] = []
    for block in range(inst.block_count):
        width, height = _shape_for_block(inst, block, guidance, profile)
        widths.append(width)
        heights.append(height)

    raw_x, raw_y = _anchors(inst, widths, heights, guidance)
    if os.environ.get("FLOORSET_PREPLACED_FRAME_CLAMP") == "1":
        _clamp_to_preplaced_frame(inst, raw_x, raw_y, widths, heights)
    key_x, key_y = _bias_order_keys(inst, raw_x, raw_y, widths, heights, config, profile)
    movable = [i for i in range(inst.block_count) if i not in inst.preplaced]
    order_x = sorted(movable, key=lambda i: (key_x[i], key_y[i], -sum(w for _, w in inst.b2b_by_block.get(i, [])), i))
    order_y = sorted(movable, key=lambda i: (key_y[i], key_x[i], -sum(w for _, w in inst.b2b_by_block.get(i, [])), i))
    pos_x = {block: idx for idx, block in enumerate(order_x)}
    pos_y = {block: idx for idx, block in enumerate(order_y)}
    h_adj = {block: [] for block in movable}
    v_adj = {block: [] for block in movable}
    orient = _cluster_orientations(inst, raw_x, raw_y)
    narrow_enabled = _narrow_grouping_pair_bias_enabled()
    cluster_pressure = {
        cluster: _cluster_grouping_pressure(inst, cluster, members, raw_x, raw_y, widths, heights)
        for cluster, members in inst.cluster_groups.items()
    }
    narrow_bonus = _env_float("FLOORSET_NARROW_GROUPING_AXIS_BONUS", 0.12)
    ambiguity_margin = _env_float("FLOORSET_NARROW_GROUPING_AMBIGUITY_MARGIN", 0.15)

    def add_h(a: int, b: int) -> None:
        if pos_x[a] <= pos_x[b]:
            h_adj[a].append(b)
        else:
            h_adj[b].append(a)

    def add_v(a: int, b: int) -> None:
        if pos_y[a] <= pos_y[b]:
            v_adj[a].append(b)
        else:
            v_adj[b].append(a)

    for offset, i in enumerate(movable):
        for j in movable[offset + 1 :]:
            dx = abs(raw_x[i] - raw_x[j]) / max((widths[i] + widths[j]) * 0.5, 1e-6)
            dy = abs(raw_y[i] - raw_y[j]) / max((heights[i] + heights[j]) * 0.5, 1e-6)
            h_score = dx
            v_score = dy
            if profile != "compact" and ((inst.boundary.get(i, 0) & 3) or (inst.boundary.get(j, 0) & 3)):
                h_score += 0.25
            if profile != "compact" and ((inst.boundary.get(i, 0) & 12) or (inst.boundary.get(j, 0) & 12)):
                v_score += 0.25
            ci = _constraint_id(inst, i, 3)
            cj = _constraint_id(inst, j, 3)
            skip_pair = False
            if ci and ci == cj:
                if narrow_enabled and cluster_pressure.get(ci, False):
                    if abs(h_score - v_score) <= ambiguity_margin:
                        if orient.get(ci, "H") == "H":
                            h_score += narrow_bonus
                        else:
                            v_score += narrow_bonus
                    elif min(h_score, v_score) > ambiguity_margin:
                        skip_pair = True
                elif orient.get(ci, "H") == "H":
                    h_score += 0.45 if _grouping_adjacency_bias_enabled(profile) else 0.0
                else:
                    v_score += 0.45 if _grouping_adjacency_bias_enabled(profile) else 0.0
            if skip_pair:
                continue
            if guidance is not None and guidance.pairwise_axis:
                key = (i, j) if i < j else (j, i)
                pair_logits = guidance.pairwise_axis.get(key)
                if pair_logits is not None:
                    x_logit, y_logit = pair_logits
                    model_axis_bias = max(-1.0, min(1.0, x_logit - y_logit))
                    h_score -= 0.20 * model_axis_bias
                    v_score += 0.20 * model_axis_bias
            if h_score >= v_score:
                add_h(i, j)
            else:
                add_v(i, j)

    if _grouping_adjacency_bias_enabled(profile) and not narrow_enabled:
        for cluster, members in inst.cluster_groups.items():
            chain = [block for block in members if block in pos_x]
            if len(chain) <= 1:
                continue
            if orient.get(cluster, "H") == "H":
                chain.sort(key=lambda i: (key_x[i], key_y[i], i))
                for a, b in zip(chain, chain[1:]):
                    add_h(a, b)
            else:
                chain.sort(key=lambda i: (key_y[i], key_x[i], i))
                for a, b in zip(chain, chain[1:]):
                    add_v(a, b)

    x_pos = {block: 0.0 for block in movable}
    y_pos = {block: 0.0 for block in movable}
    for block in order_x:
        base = x_pos[block] + widths[block]
        for other in h_adj[block]:
            x_pos[other] = max(x_pos[other], base)
    for block in order_y:
        base = y_pos[block] + heights[block]
        for other in v_adj[block]:
            y_pos[other] = max(y_pos[other], base)

    if movable:
        inv = 1.0 / len(movable)
        pack_cx = sum(x_pos[i] + widths[i] * 0.5 for i in movable) * inv
        pack_cy = sum(y_pos[i] + heights[i] * 0.5 for i in movable) * inv
        raw_cx = sum(raw_x[i] for i in movable) * inv
        raw_cy = sum(raw_y[i] for i in movable) * inv
        dx_shift = (raw_cx - pack_cx) * config.anchor_translation_strength
        dy_shift = (raw_cy - pack_cy) * config.anchor_translation_strength
        dx_shift -= min(0.0, min(x_pos[i] + dx_shift for i in movable))
        dy_shift -= min(0.0, min(y_pos[i] + dy_shift for i in movable))
    else:
        dx_shift = 0.0
        dy_shift = 0.0

    rects: dict[int, Rect] = {}
    for block in range(inst.block_count):
        target = inst.target_rects.get(block)
        if target is not None and block in inst.preplaced:
            rects[block] = target
        else:
            rects[block] = Rect(max(0.0, x_pos.get(block, 0.0) + dx_shift), max(0.0, y_pos.get(block, 0.0) + dy_shift), widths[block], heights[block])
    return Placement(rects)
