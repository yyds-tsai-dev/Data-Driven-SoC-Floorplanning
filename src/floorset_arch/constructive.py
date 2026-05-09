from __future__ import annotations

from dataclasses import dataclass, field
import math

from floorset_arch.geometry import (
    boundary_satisfied,
    candidate_frontier_points,
    edge_touch_length,
    first_non_overlapping,
    overlaps,
)
from floorset_arch.hetero_graph import HeteroFloorplanGraph
from floorset_arch.models import Instance, Placement, Rect, SolverConfig
from floorset_arch.scoring import candidate_hpwl_proxy


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

    def copy(self) -> "PlacementState":
        return PlacementState.from_rects(dict(self.rects))

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


@dataclass(frozen=True)
class Slot:
    rect: Rect
    kind: str


@dataclass
class BeamState:
    state: PlacementState
    remaining: frozenset[int]
    score: float = 0.0
    order: tuple[int, ...] = ()


def block_dimensions(inst: Instance, block: int) -> tuple[float, float]:
    target = inst.target_rects.get(block)
    if target is not None and (block in inst.fixed or block in inst.preplaced):
        return target.width, target.height
    area = max(1.0, float(inst.area_targets[block]))
    return math.sqrt(area), math.sqrt(area)


def _degree(inst: Instance, block: int) -> float:
    return sum(weight for _, weight in inst.b2b_by_block.get(block, [])) + sum(
        weight for _, weight in inst.p2b_by_block.get(block, [])
    )


def _total_area_scale(inst: Instance) -> float:
    total = sum(max(1.0, float(inst.area_targets[i])) for i in range(inst.block_count))
    return math.sqrt(max(total, 1.0))


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


def _anchor_rect(inst: Instance, block: int) -> Rect | None:
    if inst.anchor_guidance is None:
        return None
    return inst.anchor_guidance.rect_priors.get(block)


def _anchor_center(inst: Instance, block: int, width: float, height: float) -> tuple[float, float]:
    target = inst.target_rects.get(block)
    if target is not None and block in inst.preplaced:
        return target.center_x, target.center_y
    prior = _anchor_rect(inst, block)
    if prior is not None:
        return prior.center_x, prior.center_y
    pin_points = _pin_anchor_points(inst, block, width, height)
    if pin_points:
        x, y = pin_points[0]
        return x + width / 2.0, y + height / 2.0
    scale = _total_area_scale(inst)
    return scale * 0.5, scale * 0.5


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

    if inst.anchor_guidance is not None and block in inst.anchor_guidance.log_aspect:
        aspect = math.exp(max(-2.5, min(2.5, inst.anchor_guidance.log_aspect[block])))
        variants.append((math.sqrt(area * aspect), math.sqrt(area / aspect)))

    prior = _anchor_rect(inst, block)
    if prior is not None and prior.height > 0:
        aspect = max(0.05, min(20.0, prior.width / prior.height))
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


def _abut_positions(anchor: Rect, width: float, height: float) -> list[tuple[float, float]]:
    return [
        (anchor.right, anchor.y),
        (anchor.x - width, anchor.y),
        (anchor.x, anchor.top),
        (anchor.x, anchor.y - height),
        (anchor.right, anchor.top - height),
        (anchor.right - width, anchor.top),
    ]


def _group_points(inst: Instance, block: int, width: float, height: float, state: PlacementState) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for members in inst.cluster_groups.values():
        if block not in members:
            continue
        for other_idx in members:
            other = state.rects.get(other_idx)
            if other is not None:
                points.extend(_abut_positions(other, width, height))
    return points


def _mer_skyline_points(width: float, height: float, state: PlacementState, limit: int) -> list[tuple[float, float]]:
    if not state.placed:
        return [(0.0, 0.0)]
    xs = {0.0, state.x_min, state.x_max}
    ys = {0.0, state.y_min, state.y_max}
    points: set[tuple[float, float]] = set(state.frontier())
    for rect in state.placed:
        xs.update({rect.x, rect.right, max(0.0, rect.x - width), max(0.0, rect.right - width)})
        ys.update({rect.y, rect.top, max(0.0, rect.y - height), max(0.0, rect.top - height)})
        points.update(
            {
                (rect.right, rect.y),
                (rect.x, rect.top),
                (rect.right, rect.top),
                (max(0.0, rect.x - width), rect.y),
                (rect.x, max(0.0, rect.y - height)),
            }
        )

    for x in sorted(xs):
        points.add((x, state.y_max))
        points.add((x, 0.0))
    for y in sorted(ys):
        points.add((state.x_max, y))
        points.add((0.0, y))

    # A small bounded set of obstacle-edge intersections approximates MER
    # corners without the quadratic grid explosion on 100+ block instances.
    xs_sorted = sorted(xs, key=lambda x: (abs(x - state.x_max), x))[: max(8, limit // 4)]
    ys_sorted = sorted(ys, key=lambda y: (abs(y - state.y_max), y))[: max(8, limit // 4)]
    for x in xs_sorted:
        for y in ys_sorted[:4]:
            points.add((x, y))

    def compact_key(point: tuple[float, float]) -> tuple[float, float, float, float]:
        x, y = point
        candidate = Rect(max(0.0, x), max(0.0, y), width, height)
        bounds = state.bounds_with(candidate)
        return (bounds.area - state.bbox_area, bounds.width + bounds.height, candidate.x + candidate.y, candidate.x)

    legal: list[tuple[float, float]] = []
    ordered = sorted(points, key=compact_key)[: max(limit * 6, limit)]
    for x, y in ordered:
        candidate = Rect(max(0.0, x), max(0.0, y), width, height)
        if first_non_overlapping(candidate, state.placed):
            legal.append((candidate.x, candidate.y))
        if len(legal) >= limit:
            break
    return legal


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


def _slot_candidates(
    inst: Instance,
    config: SolverConfig,
    block: int,
    width: float,
    height: float,
    state: PlacementState,
) -> list[Slot]:
    points: list[tuple[str, float, float]] = []
    prior = _anchor_rect(inst, block)
    if prior is not None:
        points.append(("anchor", prior.x, prior.y))
        points.append(("anchor", prior.center_x - width / 2.0, prior.center_y - height / 2.0))
    for x, y in _pin_anchor_points(inst, block, width, height):
        points.append(("pin", x, y))
    for x, y in _group_points(inst, block, width, height, state):
        points.append(("group", x, y))
    for x, y in _boundary_points(inst.boundary.get(block, 0), width, height, state):
        points.append(("boundary", x, y))
    remaining_budget = max(8, min(32, config.max_candidates_per_block - len(points)))
    for x, y in _mer_skyline_points(width, height, state, remaining_budget):
        points.append(("mer_skyline", x, y))

    slots: list[Slot] = []
    seen: set[tuple[float, float, float, float]] = set()
    legal_checks = 0
    for kind, x, y in points:
        rect = Rect(max(0.0, float(x)), max(0.0, float(y)), width, height)
        key = (round(rect.x, 6), round(rect.y, 6), round(width, 6), round(height, 6))
        if key in seen:
            continue
        seen.add(key)
        legal_checks += 1
        if first_non_overlapping(rect, state.placed):
            slots.append(Slot(rect, kind))
        if len(slots) >= max(1, config.max_candidates_per_block):
            break
        if legal_checks >= max(config.max_candidates_per_block * 4, 12):
            break
    return slots


def _block_rank(
    inst: Instance,
    graph: HeteroFloorplanGraph,
    block: int,
    state: PlacementState,
) -> tuple[float, float, float, float, float]:
    guidance_priority = 0.0
    if inst.anchor_guidance is not None:
        guidance_priority = inst.anchor_guidance.priority.get(block, 0.0)
    width, height = block_dimensions(inst, block)
    cx, cy = _anchor_center(inst, block, width, height)
    placed_neighbor_weight = sum(weight for other, weight in inst.b2b_by_block.get(block, []) if other in state.rects)
    constraint = 0.0
    constraint += 5.0 if block in inst.boundary else 0.0
    constraint += 3.0 if block in graph.block_to_cluster else 0.0
    constraint += 2.0 if block in graph.block_to_mib else 0.0
    area = max(1.0, float(inst.area_targets[block]))
    return (
        -placed_neighbor_weight,
        -constraint,
        -_degree(inst, block),
        -area,
        cx + cy,
        -0.05 * guidance_priority,
    )


def _slot_score(
    inst: Instance,
    config: SolverConfig,
    block: int,
    slot: Slot,
    state: PlacementState,
) -> float:
    score = _placement_score(inst, config, block, slot.rect, state)
    prior = _anchor_rect(inst, block)
    if prior is not None:
        score += config.anchor_weight * (
            abs(slot.rect.center_x - prior.center_x) + abs(slot.rect.center_y - prior.center_y)
        )
    if slot.kind == "group":
        score -= config.group_penalty * 0.35
    elif slot.kind == "boundary":
        score -= config.boundary_penalty * 0.15
    elif slot.kind == "mer_skyline":
        score -= 0.02
    return score


def _initial_state(inst: Instance) -> PlacementState:
    state = PlacementState()
    for block in sorted(inst.preplaced):
        target = inst.target_rects.get(block)
        if target is not None:
            state.add(block, target)
    return state


def _fallback_place(inst: Instance, block: int, state: PlacementState) -> Rect:
    width, height = block_dimensions(inst, block)
    x = state.x_max if state.placed else 0.0
    y = state.y_min if state.placed else 0.0
    rect = Rect(x, y, width, height)
    while state.overlaps_any(rect):
        y = rect.top
        rect = Rect(x, y, width, height)
    return rect


def construct_beam_placement(
    inst: Instance,
    config: SolverConfig | None = None,
    graph: HeteroFloorplanGraph | None = None,
) -> Placement:
    config = config or SolverConfig()
    graph = graph or HeteroFloorplanGraph()
    initial = _initial_state(inst)
    remaining = frozenset(i for i in range(inst.block_count) if i not in initial.rects)
    beams = [BeamState(initial, remaining)]
    width = max(1, int(config.beam_width))

    while beams and beams[0].remaining:
        candidates: list[BeamState] = []
        for beam in beams:
            beam_candidates: list[BeamState] = []
            ranked_blocks = sorted(
                beam.remaining,
                key=lambda block: _block_rank(inst, graph, block, beam.state),
            )[: max(min(width + 1, len(beam.remaining)), 2)]
            for block in ranked_blocks:
                block_best: list[tuple[float, Slot]] = []
                for shape_w, shape_h in _shape_variants(inst, block, config, beam.state):
                    for slot in _slot_candidates(inst, config, block, shape_w, shape_h, beam.state):
                        block_best.append((_slot_score(inst, config, block, slot, beam.state), slot))
                block_best.sort(key=lambda item: item[0])
                for delta, slot in block_best[: max(1, min(width, config.max_start_candidates))]:
                    next_state = beam.state.copy()
                    next_state.add(block, slot.rect)
                    beam_candidates.append(
                        BeamState(
                            state=next_state,
                            remaining=frozenset(b for b in beam.remaining if b != block),
                            score=beam.score + delta,
                            order=beam.order + (block,),
                        )
                    )
            if not beam_candidates:
                block = min(beam.remaining, key=lambda b: _block_rank(inst, graph, b, beam.state))
                next_state = beam.state.copy()
                next_state.add(block, _fallback_place(inst, block, next_state))
                beam_candidates.append(
                    BeamState(
                        state=next_state,
                        remaining=frozenset(b for b in beam.remaining if b != block),
                        score=beam.score + 1e6,
                        order=beam.order + (block,),
                    )
                )
            candidates.extend(beam_candidates)

        candidates.sort(key=lambda beam: (beam.score, beam.state.bbox_area, beam.order))
        diverse: list[BeamState] = []
        seen_prefix: set[tuple[int, ...]] = set()
        for beam in candidates:
            prefix = beam.order[-3:]
            if prefix in seen_prefix and len(diverse) >= width:
                continue
            seen_prefix.add(prefix)
            diverse.append(beam)
            if len(diverse) >= width:
                break
        beams = diverse

    best = min(beams, key=lambda beam: (beam.score, beam.state.bbox_area))
    return Placement(dict(best.state.rects))


def construct_initial_placement(
    inst: Instance,
    config: SolverConfig | None = None,
    graph: HeteroFloorplanGraph | None = None,
) -> Placement:
    return construct_beam_placement(inst, config=config, graph=graph)
