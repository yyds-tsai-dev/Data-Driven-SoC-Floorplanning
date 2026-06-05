from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class BudgetTier(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


@dataclass(frozen=True)
class RiskBudget:
    block_count: int
    score_share: float
    constraint_density: float
    net_density: float
    boundary_count: int
    grouping_budget: int
    mib_budget: int
    hard_shape_count: int
    tier: BudgetTier


def v10_score_share(
    block_count: int,
    min_block_count: int = 21,
    max_block_count: int = 120,
) -> float:
    weights = [
        math.exp((n - max_block_count) / 12)
        for n in range(min_block_count, max_block_count + 1)
    ]
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return math.exp((block_count - max_block_count) / 12) / denom


def _edge_count(tensor) -> int:
    if tensor is None:
        return 0
    shape = getattr(tensor, "shape", None)
    if not shape:
        return 0
    return int(shape[0])


def _group_budget(groups) -> int:
    return sum(max(0, len(members) - 1) for members in groups.values())


def _tier(
    score_share: float,
    constraint_density: float,
    net_density: float,
) -> BudgetTier:
    if score_share >= 0.045 and constraint_density >= 0.45 and net_density >= 55.0:
        return BudgetTier.HEAVY
    if score_share >= 0.020 and (
        net_density >= 60.0 or (constraint_density >= 0.50 and net_density >= 20.0)
    ):
        return BudgetTier.MEDIUM
    if score_share >= 0.006 and constraint_density >= 0.55:
        return BudgetTier.LIGHT
    if score_share >= 0.015 and (constraint_density >= 0.20 or net_density >= 35.0):
        return BudgetTier.LIGHT
    return BudgetTier.NONE


def instance_risk_budget(inst) -> RiskBudget:
    block_count = int(inst.block_count)
    boundary_count = len(getattr(inst, "boundary", {}))
    grouping_budget = _group_budget(getattr(inst, "cluster_groups", {}))
    mib_budget = _group_budget(getattr(inst, "mib_groups", {}))
    hard_shape_count = len(getattr(inst, "fixed", set())) + len(
        getattr(inst, "preplaced", set())
    )
    b2b_count = _edge_count(getattr(inst, "valid_b2b", None))
    p2b_count = _edge_count(getattr(inst, "valid_p2b", None))
    score_share = v10_score_share(block_count)
    constraint_density = (
        boundary_count + grouping_budget + mib_budget + hard_shape_count
    ) / max(block_count, 1)
    net_density = (b2b_count + p2b_count) / max(block_count, 1)

    return RiskBudget(
        block_count=block_count,
        score_share=score_share,
        constraint_density=constraint_density,
        net_density=net_density,
        boundary_count=boundary_count,
        grouping_budget=grouping_budget,
        mib_budget=mib_budget,
        hard_shape_count=hard_shape_count,
        tier=_tier(score_share, constraint_density, net_density),
    )
