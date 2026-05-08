from __future__ import annotations

from dataclasses import dataclass, field
import math

from floorset_arch.geometry import (
    bbox,
    boundary_satisfied,
    candidate_frontier_points,
    edge_touch_length,
    first_non_overlapping,
    overlaps,
)
from floorset_arch.models import Instance, Placement, Rect, SolverConfig
from floorset_arch.scoring import candidate_hpwl_proxy, placement_score


def block_dimensions(inst: Instance, block: int) -> tuple[float, float]:
    target = inst.target_rects.get(block)
    if target is not None and (block in inst.fixed or block in inst.preplaced):
        return target.width, target.height
    area = float(inst.area_targets[block])
    if area <= 0:
        area = 1.0
    return math.sqrt(area), math.sqrt(area)


@dataclass
class PlacementState:
    rects: dict[int, Rect] = field(default_factory=dict)
    placed: list[Rect] = field(default_factory=list)
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    _frontier_cache: list[tuple[float, float]] | None = None

    @classmethod
    def from_rects(cls, rects: dict[int, Rect]) -> "PlacementState":
        state = cls()
        for block, rect in rects.items():
            state.add(block, rect)
        return state

    def copy(self) -> "PlacementState":
        return PlacementState.from_rects(dict(self.rects))

    def add(self, block: int, rect: Rect) -> None:
        self.rects[block] = rect
        self.placed.append(rect)
        if len(self.placed) == 1:
            self.x_min, self.y_min, self.x_max, self.y_max = rect.x, rect.y, rect.right, rect.top
        else:
            self.x_min = min(self.x_min, rect.x)
            self.y_min = min(self.y_min, rect.y)
            self.x_max = max(self.x_max, rect.right)
            self.y_max = max(self.y_max, rect.top)
        self._frontier_cache = None

    def frontier(self) -> list[tuple[float, float]]:
        if self._frontier_cache is None:
            self._frontier_cache = candidate_frontier_points(self.placed)
        return self._frontier_cache

    def overlaps_any(self, rect: Rect) -> bool:
        return any(overlaps(rect, other) for other in self.placed)

    def bounds_with(self, rect: Rect) -> Rect:
        if not self.placed:
            return Rect(rect.x, rect.y, rect.width, rect.height)
        x_min = min(self.x_min, rect.x)
        y_min = min(self.y_min, rect.y)
        x_max = max(self.x_max, rect.right)
        y_max = max(self.y_max, rect.top)
        return Rect(x_min, y_min, x_max - x_min, y_max - y_min)

    @property
    def bbox_area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


def _degree(inst: Instance, block: int) -> float:
    return sum(weight for _, weight in inst.b2b_by_block.get(block, [])) + sum(
        weight for _, weight in inst.p2b_by_block.get(block, [])
    )


def _priority(inst: Instance, block: int, order_mode: str) -> tuple[float, float, float, float]:
    constraint_score = 0.0
    if block in inst.boundary:
        constraint_score += 4.0
    if any(block in members for members in inst.cluster_groups.values()):
        constraint_score += 2.0
    if any(block in members for members in inst.mib_groups.values()):
        constraint_score += 1.0
    degree = _degree(inst, block)
    area = float(inst.area_targets[block])
    if order_mode == "cluster_first":
        cluster = 1.0 if any(block in members for members in inst.cluster_groups.values()) else 0.0
        return (-cluster, -constraint_score, -degree, -area)
    if order_mode == "boundary_first":
        return (0.0 if block in inst.boundary else 1.0, -constraint_score, -degree, -area)
    if order_mode == "degree_first":
        return (-degree, -constraint_score, -area, float(block))
    if order_mode == "model_first" and inst.model_hints and block in inst.model_hints:
        hint = inst.model_hints[block]
        return (hint.x + hint.y, -constraint_score, -degree, -area)
    return (-constraint_score, -degree, -area, float(block))


def _fallback_point(state: PlacementState) -> tuple[float, float]:
    if not state.placed:
        return 0.0, 0.0
    return state.x_max, state.y_min


def _boundary_points(code: int, width: float, height: float, state: PlacementState) -> list[tuple[float, float]]:
    if not state.placed or code == 0:
        return []
    points = []
    if code & 1:
        points.extend([(state.x_min, state.y_min), (state.x_min, state.y_max)])
    if code & 2:
        points.extend([(state.x_max, state.y_min), (state.x_max, state.y_max)])
    if code & 4:
        points.extend([(state.x_min, state.y_max), (max(state.x_min, state.x_max - width), state.y_max)])
    if code & 8:
        points.extend([(state.x_min, state.y_min), (max(state.x_min, state.x_max - width), state.y_min)])
    return points


def _pin_anchor_points(inst: Instance, block: int, width: float, height: float) -> list[tuple[float, float]]:
    weighted_x = 0.0
    weighted_y = 0.0
    total = 0.0
    for pin_idx, weight in inst.p2b_by_block.get(block, []):
        if pin_idx >= inst.pins_pos.shape[0]:
            continue
        weighted_x += float(inst.pins_pos[pin_idx, 0]) * weight
        weighted_y += float(inst.pins_pos[pin_idx, 1]) * weight
        total += weight
    if total <= 0:
        return []
    return [(weighted_x / total - width / 2.0, weighted_y / total - height / 2.0)]


def _group_points(inst: Instance, block: int, width: float, height: float, state: PlacementState) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for members in inst.cluster_groups.values():
        if block not in members:
            continue
        for other_idx in members:
            other = state.rects.get(other_idx)
            if other is None:
                continue
            points.extend(
                [
                    (other.right, other.y),
                    (other.x - width, other.y),
                    (other.x, other.top),
                    (other.x, other.y - height),
                    (other.right, other.top - height),
                    (other.right - width, other.top),
                ]
            )
    return points


def _mib_reference_shape(inst: Instance, block: int, state: PlacementState) -> tuple[float, float] | None:
    for members in inst.mib_groups.values():
        if block not in members:
            continue
        for other_idx in members:
            other = state.rects.get(other_idx)
            if other is not None:
                return other.width, other.height
    return None


def _shape_variants(inst: Instance, block: int, config: SolverConfig, state: PlacementState) -> list[tuple[float, float]]:
    base_w, base_h = block_dimensions(inst, block)
    if block in inst.fixed or block in inst.preplaced:
        return [(base_w, base_h)]

    area = max(1.0, float(inst.area_targets[block]))
    variants: list[tuple[float, float]] = []
    mib_shape = _mib_reference_shape(inst, block, state)
    if mib_shape is not None and abs(mib_shape[0] * mib_shape[1] - area) / area <= 0.01:
        variants.append(mib_shape)
    if inst.model_hints and block in inst.model_hints:
        hint = inst.model_hints[block]
        aspect = max(0.05, min(20.0, hint.width / max(hint.height, 1e-6)))
        variants.append((math.sqrt(area * aspect), math.sqrt(area / aspect)))
    for aspect in (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 4.0, 0.25):
        variants.append((math.sqrt(area * aspect), math.sqrt(area / aspect)))

    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for width, height in variants:
        key = (round(width, 4), round(height, 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((width, height))
        if len(deduped) >= max(1, config.shape_variant_count):
            break
    return deduped


def _candidate_points(inst: Instance, block: int, width: float, height: float, state: PlacementState) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if inst.model_hints and block in inst.model_hints:
        hint = inst.model_hints[block]
        points.extend([(hint.x, hint.y), (hint.center_x - width / 2.0, hint.center_y - height / 2.0)])
    points.extend(_pin_anchor_points(inst, block, width, height))
    points.extend(_group_points(inst, block, width, height, state))
    points.extend(_boundary_points(inst.boundary.get(block, 0), width, height, state))
    points.extend(state.frontier())
    if not points:
        points.append((0.0, 0.0))
    return points


def _placement_score(inst: Instance, config: SolverConfig, block: int, candidate: Rect, state: PlacementState) -> float:
    bounds = state.bounds_with(candidate)
    bbox_delta = bounds.area - state.bbox_area
    score = config.hpwl_weight * candidate_hpwl_proxy(inst, block, candidate, state.rects)
    score += config.bbox_weight * max(0.0, bbox_delta)

    code = inst.boundary.get(block, 0)
    if code and not boundary_satisfied(candidate, bounds, code):
        score += config.boundary_penalty
    elif code:
        score -= config.boundary_penalty * 0.35

    for members in inst.cluster_groups.values():
        if block not in members:
            continue
        placed_members = [state.rects[idx] for idx in members if idx in state.rects]
        if not placed_members:
            continue
        best_touch = max(edge_touch_length(candidate, other) for other in placed_members)
        if best_touch > 0:
            score -= config.group_penalty + 20.0 * best_touch
        else:
            best_distance = min(
                abs(candidate.center_x - other.center_x) + abs(candidate.center_y - other.center_y)
                for other in placed_members
            )
            score += config.group_penalty + 0.2 * best_distance

    mib_shape = _mib_reference_shape(inst, block, state)
    if mib_shape is not None and (
        abs(candidate.width - mib_shape[0]) > 1e-4 or abs(candidate.height - mib_shape[1]) > 1e-4
    ):
        score += config.mib_penalty
    return score


def construct_initial_placement(
    inst: Instance,
    config: SolverConfig | None = None,
    order_mode: str = "default",
) -> Placement:
    config = config or SolverConfig()
    if order_mode == "legacy":
        return _construct_legacy(inst, config)
    state = PlacementState()

    for block in sorted(inst.preplaced):
        target = inst.target_rects.get(block)
        if target is not None:
            state.add(block, target)

    remaining = [i for i in range(inst.block_count) if i not in state.rects]
    remaining.sort(key=lambda block: _priority(inst, block, order_mode))

    for block in remaining:
        best: Rect | None = None
        best_score = float("inf")
        seen: set[tuple[float, float, float, float]] = set()

        for width, height in _shape_variants(inst, block, config, state):
            for x, y in _candidate_points(inst, block, width, height, state):
                rect = Rect(max(0.0, float(x)), max(0.0, float(y)), width, height)
                key = (round(rect.x, 6), round(rect.y, 6), round(width, 6), round(height, 6))
                if key in seen:
                    continue
                seen.add(key)
                if len(seen) > config.max_candidates_per_block:
                    break
                if state.overlaps_any(rect):
                    continue
                score = _placement_score(inst, config, block, rect, state)
                if score < best_score:
                    best = rect
                    best_score = score

        if best is None:
            width, height = block_dimensions(inst, block)
            x, y = _fallback_point(state)
            best = Rect(x, y, width, height)
            while state.overlaps_any(best):
                y = best.top
                best = Rect(x, y, width, height)
        state.add(block, best)

    return Placement(dict(state.rects))


def _legacy_priority(inst: Instance, block: int) -> tuple[int, int, float]:
    constraint_score = 0
    if block in inst.boundary:
        constraint_score += 4
    if any(block in members for members in inst.cluster_groups.values()):
        constraint_score += 2
    if any(block in members for members in inst.mib_groups.values()):
        constraint_score += 1
    degree = int((inst.valid_b2b[:, :2] == block).sum().item()) if inst.valid_b2b.numel() else 0
    return (-constraint_score, -degree, -float(inst.area_targets[block]))


def _legacy_fallback_point(rects: dict[int, Rect]) -> tuple[float, float]:
    bounds = bbox(list(rects.values()))
    return bounds.right, bounds.y


def _legacy_boundary_points(code: int, width: float, height: float, rects: dict[int, Rect]) -> list[tuple[float, float]]:
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


def _construct_legacy(inst: Instance, config: SolverConfig) -> Placement:
    placement = Placement()

    for block in sorted(inst.preplaced):
        target = inst.target_rects.get(block)
        if target is not None:
            placement.rects[block] = target

    remaining = [i for i in range(inst.block_count) if i not in placement.rects]
    remaining.sort(key=lambda block: _legacy_priority(inst, block))

    for block in remaining:
        width, height = block_dimensions(inst, block)
        placed_rects = list(placement.rects.values())
        candidates = []
        if inst.model_hints and block in inst.model_hints:
            hint = inst.model_hints[block]
            candidates.append((hint.x, hint.y))
        candidates.extend(_legacy_boundary_points(inst.boundary.get(block, 0), width, height, placement.rects))
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
            x, y = _legacy_fallback_point(placement.rects)
            best = Rect(x, y, width, height)
            while not first_non_overlapping(best, list(placement.rects.values())):
                y = best.top
                best = Rect(x, y, width, height)
        placement.rects[block] = best

    return placement
