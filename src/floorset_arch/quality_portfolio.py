from __future__ import annotations

import os

from floorset_arch.geometry import Rect, bbox, candidate_frontier_points, first_non_overlapping
from floorset_arch.models import Instance, Placement, SolverConfig
from floorset_arch.repair import soft_violation_counts
from floorset_arch.scoring import hpwl_proxy


QUALITY_PROFILES = ("default", "hpwl_refine", "area_refine", "balanced_refine")


def enabled_quality_profiles() -> list[str]:
    raw = os.environ.get(
        "FLOORSET_QUALITY_PORTFOLIO_PROFILES",
        "default,hpwl_refine",
    )
    profiles = [part.strip() for part in raw.split(",") if part.strip()]
    filtered = [profile for profile in profiles if profile in QUALITY_PROFILES]
    return filtered or ["default"]


def is_quality_portfolio_case(inst: Instance) -> bool:
    mode = os.environ.get("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "0").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes", "always", "force"}:
        return True

    b2b_count = int(inst.valid_b2b.shape[0]) if inst.valid_b2b is not None else 0
    p2b_count = int(inst.valid_p2b.shape[0]) if inst.valid_p2b is not None else 0
    boundary_count = len(inst.boundary)
    grouping_budget = sum(max(0, len(members) - 1) for members in inst.cluster_groups.values())
    mib_budget = sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
    edge_density = (b2b_count + p2b_count) / max(inst.block_count, 1)

    min_blocks = int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_MIN_BLOCKS", "116"))
    extreme_density = float(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_EDGE_DENSITY", "80.0"))
    pin_heavy_min = int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_PIN_HEAVY_MIN", "3000"))
    b2b_heavy_min = int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_B2B_HEAVY_MIN", "5000"))
    if inst.block_count >= min_blocks and edge_density >= extreme_density:
        return True
    if inst.block_count >= min_blocks and p2b_count >= pin_heavy_min and b2b_count >= b2b_heavy_min:
        return True
    if inst.block_count >= 90 and boundary_count + grouping_budget + mib_budget >= 70 and edge_density >= 60.0:
        return True
    return False


def _edge_weight(inst: Instance, block: int) -> float:
    return sum(weight for _other, weight in inst.b2b_by_block.get(block, [])) + sum(
        weight for _pin, weight in inst.p2b_by_block.get(block, [])
    )


def _quality_score(inst: Instance, placement: Placement, profile: str) -> float:
    bounds = bbox(list(placement.rects.values()))
    hpwl = hpwl_proxy(inst, placement.rects)
    if profile == "area_refine":
        return bounds.area + 0.001 * hpwl
    if profile == "hpwl_refine":
        return hpwl + 0.01 * bounds.area
    return hpwl + 0.08 * bounds.area


def refine_quality_candidate(
    inst: Instance,
    placement: Placement,
    config: SolverConfig,
    profile: str,
) -> Placement:
    if profile == "default" or not placement.rects:
        return placement

    best = placement.copy()
    best_soft = sum(soft_violation_counts(inst, best))
    best_score = _quality_score(inst, best, profile)
    max_blocks = max(0, int(os.environ.get("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "20")))
    max_slots = max(0, int(os.environ.get("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "48")))
    if max_blocks == 0 or max_slots == 0:
        return best

    movable = [block for block in best.rects if block not in inst.preplaced]
    if profile == "area_refine":
        center_x = sum(rect.center_x for rect in best.rects.values()) / max(len(best.rects), 1)
        center_y = sum(rect.center_y for rect in best.rects.values()) / max(len(best.rects), 1)
        movable.sort(
            key=lambda block: (
                -(abs(best.rects[block].center_x - center_x) + abs(best.rects[block].center_y - center_y)),
                -_edge_weight(inst, block),
                block,
            )
        )
    else:
        movable.sort(key=lambda block: (-_edge_weight(inst, block), block))

    for block in movable[:max_blocks]:
        rect = best.rects[block]
        others = [other for idx, other in best.rects.items() if idx != block]
        for x, y in candidate_frontier_points(others)[:max_slots]:
            candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
            if abs(candidate.x - rect.x) <= 1e-9 and abs(candidate.y - rect.y) <= 1e-9:
                continue
            if not first_non_overlapping(candidate, others):
                continue
            trial = best.copy()
            trial.rects[block] = candidate
            if sum(soft_violation_counts(inst, trial)) > best_soft:
                continue
            score = _quality_score(inst, trial, profile)
            if score < best_score * (1.0 - 1e-4):
                best = trial
                best_soft = sum(soft_violation_counts(inst, best))
                best_score = score
                rect = candidate
                others = [other for idx, other in best.rects.items() if idx != block]
    return best
