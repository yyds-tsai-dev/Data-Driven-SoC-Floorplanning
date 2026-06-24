from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from floorset_arch.risk_budget import BudgetTier, RiskBudget, instance_risk_budget


SoftCounts = tuple[int, int, int]


@dataclass(frozen=True)
class V10BudgetDecision:
    risk: RiskBudget
    candidate_budget_tier: BudgetTier
    quality_portfolio_allowed: bool


@dataclass(frozen=True)
class ConditionalRuntimeLimits:
    max_rejected_attempts: int
    max_elapsed_ms: int


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _mode(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _is_disabled(mode: str) -> bool:
    return mode in {"0", "false", "off", "no"}


def _is_forced(mode: str) -> bool:
    return mode in {"1", "true", "on", "yes", "always", "force"}


def _tier_value(tier: BudgetTier) -> str:
    return str(tier.value if isinstance(tier, BudgetTier) else tier)


def candidate_budget_tier(
    inst,
    *,
    budget: RiskBudget | None = None,
    mode: str | None = None,
) -> BudgetTier:
    budget = budget or instance_risk_budget(inst)
    mode = (mode or _mode("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", "auto")).strip().lower()
    if _is_disabled(mode):
        return BudgetTier.NONE
    if mode in {"1", "true", "on", "yes"}:
        return budget.tier if budget.tier is not BudgetTier.NONE else BudgetTier.NONE
    return budget.tier if budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY} else BudgetTier.NONE


def quality_portfolio_allowed(inst, *, budget: RiskBudget | None = None) -> bool:
    mode = _mode("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "0")
    if _is_disabled(mode):
        return False
    if _is_forced(mode):
        return True
    budget = budget or instance_risk_budget(inst)
    return budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}


def soft_capacity(inst) -> int:
    boundary_budget = len(getattr(inst, "boundary", {}))
    grouping_budget = sum(
        max(0, len(members) - 1)
        for members in getattr(inst, "cluster_groups", {}).values()
    )
    mib_budget = sum(
        max(0, len(members) - 1)
        for members in getattr(inst, "mib_groups", {}).values()
    )
    return max(boundary_budget + grouping_budget + mib_budget, 1)


def _soft_total(soft_counts: Iterable[int]) -> int:
    return sum(int(count) for count in soft_counts)


def v10_soft_repair_allowed(
    inst,
    *,
    soft_counts: SoftCounts | None = None,
    budget: RiskBudget | None = None,
) -> bool:
    if not env_flag("FLOORSET_ENABLE_V10_SOFT_REPAIR"):
        return False
    budget = budget or instance_risk_budget(inst)
    if budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}:
        return True
    if budget.tier is not BudgetTier.LIGHT or soft_counts is None:
        return False

    soft_total = _soft_total(soft_counts)
    soft_relative = soft_total / soft_capacity(inst)
    min_soft = env_int("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_SOFT", 6)
    min_relative = env_float("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_RELATIVE", 0.12)
    return soft_total >= min_soft or soft_relative >= min_relative


def conditional_runtime_budget_enabled() -> bool:
    return env_flag("FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET")


def conditional_runtime_budget_limits(tier: BudgetTier) -> ConditionalRuntimeLimits:
    if tier is BudgetTier.HEAVY:
        attempts_default = 3
        elapsed_default = 4000
        prefix = "HEAVY"
    elif tier is BudgetTier.MEDIUM:
        attempts_default = 2
        elapsed_default = 2500
        prefix = "MEDIUM"
    else:
        attempts_default = 1
        elapsed_default = 1200
        prefix = "LIGHT"
    return ConditionalRuntimeLimits(
        max_rejected_attempts=env_int(
            f"FLOORSET_CONDITIONAL_RUNTIME_{prefix}_ATTEMPTS",
            attempts_default,
        ),
        max_elapsed_ms=env_int(
            f"FLOORSET_CONDITIONAL_RUNTIME_{prefix}_ELAPSED_MS",
            elapsed_default,
        ),
    )


def v10_soft_acceptance_slack(
    current_counts: SoftCounts,
    candidate_counts: SoftCounts,
) -> tuple[str, float] | None:
    cur_boundary, cur_grouping, cur_mib = current_counts
    cand_boundary, cand_grouping, cand_mib = candidate_counts
    if _soft_total(candidate_counts) >= _soft_total(current_counts):
        return None
    if cand_grouping < cur_grouping:
        return ("grouping", env_float("FLOORSET_V10_SOFT_REPAIR_GROUPING_SLACK", 0.12))
    if cand_grouping > cur_grouping:
        return None
    if cand_boundary < cur_boundary and cand_mib >= cur_mib:
        return ("boundary", env_float("FLOORSET_V10_SOFT_REPAIR_BOUNDARY_SLACK", 0.06))
    return ("general", env_float("FLOORSET_V10_SOFT_REPAIR_GENERAL_SLACK", 0.08))


def budget_decision(inst, *, budget: RiskBudget | None = None) -> V10BudgetDecision:
    budget = budget or instance_risk_budget(inst)
    return V10BudgetDecision(
        risk=budget,
        candidate_budget_tier=candidate_budget_tier(inst, budget=budget),
        quality_portfolio_allowed=quality_portfolio_allowed(inst, budget=budget),
    )


def budget_trace_context(inst, *, budget: RiskBudget | None = None) -> dict[str, object]:
    decision = budget_decision(inst, budget=budget)
    risk = decision.risk
    return {
        "risk_tier": _tier_value(risk.tier),
        "candidate_budget_tier": _tier_value(decision.candidate_budget_tier),
        "quality_portfolio_allowed": decision.quality_portfolio_allowed,
        "score_share": risk.score_share,
        "constraint_density": risk.constraint_density,
        "net_density": risk.net_density,
        "boundary_count": risk.boundary_count,
        "grouping_budget": risk.grouping_budget,
        "mib_budget": risk.mib_budget,
        "hard_shape_count": risk.hard_shape_count,
    }
