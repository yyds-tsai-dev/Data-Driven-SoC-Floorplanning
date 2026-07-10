"""Tests for the dormant E2 tail time-budget escape hatch in
floorset_arch.legalizer.column_backbone._time_budget (flat
FLOORSET_TAIL_BUDGET_SCALE / FLOORSET_TAIL_BUDGET_N) and the E2-adaptive
extension (FLOORSET_TAIL_BUDGET_ADAPTIVE / _CAP / _STALL_WINDOW / _STALL_EPS)."""

import math

import pytest

from floorset_arch.legalizer import column_backbone as cb

_TAIL_ENV = (
    "FLOORSET_TAIL_BUDGET_SCALE",
    "FLOORSET_TAIL_BUDGET_N",
    "FLOORSET_TAIL_BUDGET_ADAPTIVE",
    "FLOORSET_TAIL_BUDGET_CAP",
    "FLOORSET_TAIL_STALL_WINDOW",
    "FLOORSET_TAIL_STALL_EPS",
)


def _old_formula(block_count: int) -> float:
    b = cb.BUDGET_SCALE * math.exp(block_count / cb.BUDGET_TAU)
    return max(cb.BUDGET_MIN, min(cb.BUDGET_MAX, b))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for _v in _TAIL_ENV:
        monkeypatch.delenv(_v, raising=False)
    yield
    for _v in _TAIL_ENV:
        monkeypatch.delenv(_v, raising=False)


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


# ---------------------------------------------------------------------------
# E2-adaptive budget ceiling: CAP replaces the flat SCALE at/above N
# ---------------------------------------------------------------------------
def test_adaptive_off_is_byte_identical_to_flat_scale(monkeypatch):
    """With ADAPTIVE unset, the flat-SCALE path must be exactly the pre-adaptive
    formula (the promoted .env default: SCALE=2, N=100)."""
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_SCALE", "2.0")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    assert cb._time_budget(99) == pytest.approx(_old_formula(99))
    assert cb._time_budget(100) == pytest.approx(_old_formula(100) * 2.0)
    assert cb._time_budget(120) == pytest.approx(_old_formula(120) * 2.0)


def test_adaptive_cap_applies_only_at_or_above_n(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_CAP", "3.0")
    assert cb._time_budget(99) == pytest.approx(_old_formula(99))
    assert cb._time_budget(100) == pytest.approx(_old_formula(100) * 3.0)
    assert cb._time_budget(120) == pytest.approx(_old_formula(120) * 3.0)


def test_adaptive_cap_takes_precedence_over_flat_scale(monkeypatch):
    """When both are set, ADAPTIVE's CAP replaces the flat SCALE as the sole
    tail multiplier (documented precedence)."""
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_SCALE", "2.0")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_CAP", "3.0")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    assert cb._time_budget(120) == pytest.approx(_old_formula(120) * 3.0)


def test_adaptive_cap_default_is_3x(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    assert cb._time_budget(120) == pytest.approx(_old_formula(120) * 3.0)


def test_adaptive_invalid_cap_falls_back(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_CAP", "xyz")
    assert cb._time_budget(120) == pytest.approx(_old_formula(120) * 3.0)


# ---------------------------------------------------------------------------
# E2-adaptive stall window / eps helpers
# ---------------------------------------------------------------------------
def test_stall_window_none_when_adaptive_off(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    assert cb._tail_stall_window(120) is None


def test_stall_window_none_below_n(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    assert cb._tail_stall_window(99) is None


def test_stall_window_default_quarter_of_base(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    win = cb._tail_stall_window(120)
    assert win == pytest.approx(0.25 * _old_formula(120))
    # the window is a fraction of the BASE (unscaled) budget, never the ceiling
    assert win < _old_formula(120)


def test_stall_window_custom_fraction_and_fallback(monkeypatch):
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_ADAPTIVE", "1")
    monkeypatch.setenv("FLOORSET_TAIL_BUDGET_N", "100")
    monkeypatch.setenv("FLOORSET_TAIL_STALL_WINDOW", "0.4")
    assert cb._tail_stall_window(120) == pytest.approx(0.4 * _old_formula(120))
    monkeypatch.setenv("FLOORSET_TAIL_STALL_WINDOW", "oops")
    assert cb._tail_stall_window(120) == pytest.approx(0.25 * _old_formula(120))


def test_stall_eps_default_and_fallback(monkeypatch):
    assert cb._tail_stall_eps() == pytest.approx(0.003)
    monkeypatch.setenv("FLOORSET_TAIL_STALL_EPS", "0.01")
    assert cb._tail_stall_eps() == pytest.approx(0.01)
    monkeypatch.setenv("FLOORSET_TAIL_STALL_EPS", "bad")
    assert cb._tail_stall_eps() == pytest.approx(0.003)
