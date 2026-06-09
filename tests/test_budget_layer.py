from types import SimpleNamespace

import torch

from floorset_arch.budget_layer import (
    budget_trace_context,
    candidate_budget_tier,
    quality_portfolio_allowed,
    v10_soft_acceptance_slack,
    v10_soft_repair_allowed,
)
from floorset_arch.models import Placement, Rect
from floorset_arch.parser import parse_instance
from floorset_arch.risk_budget import BudgetTier


def _inst(block_count=120, *, b2b=7200, p2b=3000, boundary=0):
    constraints = torch.zeros(block_count, 5)
    if boundary:
        constraints[:boundary, 4] = 1.0
    return parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.tensor(
            [
                [float(i % block_count), float((i + 1) % block_count), 1.0]
                for i in range(b2b)
            ]
        )
        if b2b
        else torch.empty(0, 3),
        torch.tensor(
            [[float(i % 64), float(i % block_count), 1.0] for i in range(p2b)]
        )
        if p2b
        else torch.empty(0, 3),
        torch.zeros(64, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )


def test_candidate_budget_tier_uses_same_auto_gate_as_high_risk_portfolio(monkeypatch):
    inst = _inst()
    monkeypatch.delenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", raising=False)

    assert candidate_budget_tier(inst) in {BudgetTier.MEDIUM, BudgetTier.HEAVY}

    monkeypatch.setattr(
        "floorset_arch.budget_layer.instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=BudgetTier.LIGHT),
    )

    assert candidate_budget_tier(inst) is BudgetTier.NONE


def test_quality_portfolio_allowed_uses_shared_budget_layer(monkeypatch):
    inst = _inst()
    monkeypatch.setenv("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "auto")

    assert quality_portfolio_allowed(inst)

    monkeypatch.setenv("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "0")

    assert not quality_portfolio_allowed(inst)


def test_v10_soft_repair_allowed_uses_risk_tier_and_soft_pressure(monkeypatch):
    inst = _inst(block_count=80, b2b=0, p2b=0, boundary=4)
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(15.0, 0.0, 2.0, 2.0),
            3: Rect(20.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")
    monkeypatch.setenv("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_SOFT", "3")
    monkeypatch.setenv("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_RELATIVE", "1.0")
    monkeypatch.setattr(
        "floorset_arch.budget_layer.instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=BudgetTier.LIGHT),
    )

    assert v10_soft_repair_allowed(inst, soft_counts=(4, 0, 0))
    assert not v10_soft_repair_allowed(inst, soft_counts=(1, 0, 0))


def test_budget_trace_context_exposes_unified_decision_fields(monkeypatch):
    inst = _inst()
    monkeypatch.setenv("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "auto")
    monkeypatch.setenv("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP", "1")

    trace = budget_trace_context(inst)

    assert trace["risk_tier"] in {"medium", "heavy"}
    assert trace["candidate_budget_tier"] in {"medium", "heavy"}
    assert trace["quality_portfolio_allowed"] is True
    assert trace["runtime_tail_clamp_enabled"] is True
    assert trace["score_share"] > 0.0
    assert trace["net_density"] > 0.0


def test_v10_soft_acceptance_slack_prefers_grouping_over_boundary():
    assert v10_soft_acceptance_slack((1, 2, 0), (1, 1, 0))[0] == "grouping"
    assert v10_soft_acceptance_slack((2, 1, 0), (1, 1, 0))[0] == "boundary"
    assert v10_soft_acceptance_slack((1, 1, 1), (1, 1, 0))[0] == "general"
    assert v10_soft_acceptance_slack((1, 1, 0), (1, 2, 0)) is None


def test_conditional_runtime_budget_defaults_off(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET", raising=False)

    from floorset_arch.budget_layer import conditional_runtime_budget_enabled

    assert not conditional_runtime_budget_enabled()


def test_conditional_runtime_limits_are_tier_specific(monkeypatch):
    from floorset_arch.budget_layer import conditional_runtime_budget_limits

    monkeypatch.setenv("FLOORSET_CONDITIONAL_RUNTIME_LIGHT_ATTEMPTS", "1")
    monkeypatch.setenv("FLOORSET_CONDITIONAL_RUNTIME_MEDIUM_ATTEMPTS", "2")
    monkeypatch.setenv("FLOORSET_CONDITIONAL_RUNTIME_HEAVY_ATTEMPTS", "3")

    assert conditional_runtime_budget_limits(BudgetTier.LIGHT).max_rejected_attempts == 1
    assert conditional_runtime_budget_limits(BudgetTier.MEDIUM).max_rejected_attempts == 2
    assert conditional_runtime_budget_limits(BudgetTier.HEAVY).max_rejected_attempts == 3
