"""Tests for the dormant E2 tail time-budget escape hatch in
floorset_arch.legalizer.column_backbone._time_budget (FLOORSET_TAIL_BUDGET_SCALE
/ FLOORSET_TAIL_BUDGET_N)."""

import math

import pytest

from floorset_arch.legalizer import column_backbone as cb


def _old_formula(block_count: int) -> float:
    b = cb.BUDGET_SCALE * math.exp(block_count / cb.BUDGET_TAU)
    return max(cb.BUDGET_MIN, min(cb.BUDGET_MAX, b))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLOORSET_TAIL_BUDGET_SCALE", raising=False)
    monkeypatch.delenv("FLOORSET_TAIL_BUDGET_N", raising=False)
    yield
    monkeypatch.delenv("FLOORSET_TAIL_BUDGET_SCALE", raising=False)
    monkeypatch.delenv("FLOORSET_TAIL_BUDGET_N", raising=False)


@pytest.mark.parametrize("n", [21, 68, 116, 120])
def test_default_matches_old_formula(n):
    assert cb._time_budget(n) == pytest.approx(_old_formula(n))


def test_scale_applies_only_at_or_above_n(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_SCALE", "2.0")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "116")

    assert cb._time_budget(115) == pytest.approx(_old_formula(115))
    assert cb._time_budget(116) == pytest.approx(_old_formula(116) * 2.0)
    assert cb._time_budget(120) == pytest.approx(_old_formula(120) * 2.0)


def test_invalid_env_values_fall_back_to_default(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_SCALE", "abc")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "not-an-int")

    for n in (21, 68, 116, 120):
        assert cb._time_budget(n) == pytest.approx(_old_formula(n))
