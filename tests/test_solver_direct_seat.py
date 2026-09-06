"""`PARTNER_DIRECT_SEAT_FIX` — the direct-channel seat-reservation gate.

Handing the legalizer a `sample_fn` is not free: `_parallel_solve` reserves
`PARTNER_NREF` pool workers for direct-prediction refinement and truncates
the column-restart portfolio by the same amount (6 of 24 = 25% of restart
breadth at the shipped 0.3 s operating point).  The shipped gate only asks
whether there is budget to SAMPLE (`remaining > PARTNER_DIRECT_MIN`), so in
the band where the budget covers sampling but not `refine_prediction`'s
rung 0 the reserved workers return nothing and the restarts are lost for
free.

What must hold:
  * flag off -> the gate is exactly `remaining > PARTNER_DIRECT_MIN`, for
    every (n, remaining) pair, including the ones the fix would close;
  * the fix can only ever CLOSE the channel (DIRECT_MIN stays a floor), so
    it can add no runtime and cannot open a case that is shut today;
  * with the shipped 0.3 s budget curve the fix closes exactly the
    measured-dead band (n <= 101) and leaves n >= 102 open;
  * at the 3.5 s tier (and any budget >= 2 s) the fix is a no-op: every
    case the shipped gate opens stays open — the flag is cross-tier safe;
  * the projection is monotone in `remaining` at fixed n, so the gate is a
    single threshold per instance and never oscillates.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

co = pytest.importorskip("contest_optimizer")

# shipped 0.3 s operating point (.env "0806 定案")
SCALE, TAU, BMIN, BMAX = 8.498e-5, 12.0, 0.05, 1.22
# serial head measured between the budget start and the gate (heuristic
# seed + parse); 2.2-2.4 ms across the tail band (`[seatgate]` instrument,
# e.g. n=100: budget 0.3535 -> gate remaining 0.3513)
HEAD = 0.0023


def _budget(n: int) -> float:
    return max(BMIN, min(BMAX, SCALE * math.exp(n / TAU)))


def _remaining(n: int) -> float:
    return _budget(n) - HEAD


@pytest.fixture
def seat_env(monkeypatch):
    monkeypatch.setenv("PARTNER_DIRECT_MIN", "0.3")
    monkeypatch.delenv("PARTNER_SEAT_DEBUG", raising=False)
    monkeypatch.delenv("PARTNER_DIRECT_SEAT_TS", raising=False)
    monkeypatch.delenv("PARTNER_DIRECT_SEAT_R0", raising=False)
    return monkeypatch


def test_flag_off_is_the_shipped_expression(seat_env):
    seat_env.delenv("PARTNER_DIRECT_SEAT_FIX", raising=False)
    for n in range(21, 121):
        rem = _remaining(n)
        assert co._direct_seat_ok(n, rem) is (rem > 0.3), n
    # and for budgets the shipped gate closes / opens by a hair
    for rem in (0.0, 0.29999, 0.3, 0.30001, 1.0, 4.0):
        for n in (21, 60, 99, 102, 120):
            assert co._direct_seat_ok(n, rem) is (rem > 0.3), (n, rem)


def test_flag_off_ignores_the_calibration_constants(seat_env):
    """A constant that could only fire through the fix must not leak into
    the off path (bit-exact off is the promotion precondition)."""
    seat_env.delenv("PARTNER_DIRECT_SEAT_FIX", raising=False)
    seat_env.setenv("PARTNER_DIRECT_SEAT_TS", "99.0")
    seat_env.setenv("PARTNER_DIRECT_SEAT_R0", "99.0")
    for n in (99, 100, 101, 102, 110, 120):
        assert co._direct_seat_ok(n, _remaining(n)) is True, n


def test_fix_only_closes_never_opens(seat_env):
    seat_env.setenv("PARTNER_DIRECT_SEAT_FIX", "1")
    for n in range(21, 121):
        for rem in (0.05, 0.2, 0.29, 0.3, _remaining(n), 1.0, 3.2):
            if not co._direct_seat_ok(n, rem):
                continue
            seat_env.delenv("PARTNER_DIRECT_SEAT_FIX", raising=False)
            assert co._direct_seat_ok(n, rem), (n, rem)
            seat_env.setenv("PARTNER_DIRECT_SEAT_FIX", "1")


def test_closes_the_measured_dead_band_at_the_03s_point(seat_env):
    """n=99/100/101 delivered 0 usable direct candidates (rung 0 failed on
    6/6, 6/6 and 4/6 workers); n>=102 legalized 6/6 and won cases from
    n=103 up."""
    seat_env.setenv("PARTNER_DIRECT_SEAT_FIX", "1")
    for n in (99, 100, 101):
        assert co._direct_seat_ok(n, _remaining(n)) is False, n
    for n in range(102, 121):
        assert co._direct_seat_ok(n, _remaining(n)) is True, n
    # below the band the shipped DIRECT_MIN floor already closes the gate,
    # so the fix changes nothing there
    for n in (60, 90, 95, 98):
        assert co._direct_seat_ok(n, _remaining(n)) is False, n


def test_no_op_at_the_35s_tier(seat_env):
    """Cross-tier safety: at BUDGET_MAX=3.5 (and anywhere the remaining
    budget is >= 2 s) the fix must not close a single case."""
    seat_env.setenv("PARTNER_DIRECT_SEAT_FIX", "1")
    for n in range(21, 121):
        rem = max(BMIN, min(3.5, 0.06 * math.exp(n / 20.0))) - HEAD
        if rem <= 0.3:
            continue
        assert co._direct_seat_ok(n, rem) is True, (n, rem)


def test_projection_is_monotone_in_remaining(seat_env):
    seat_env.setenv("PARTNER_DIRECT_SEAT_FIX", "1")
    for n in (99, 105, 120):
        seen_open = False
        for k in range(1, 400):
            rem = 0.01 * k
            ok = co._direct_seat_ok(n, rem)
            if ok:
                seen_open = True
            else:
                assert not seen_open, (n, rem)
        assert seen_open, n


def test_degenerate_budgets_are_safe(seat_env):
    seat_env.setenv("PARTNER_DIRECT_SEAT_FIX", "1")
    for rem in (-1.0, 0.0, float("nan")):
        assert co._direct_seat_ok(100, rem) is False, rem
    span, need = co._direct_rung0_projection(100, 0.31)
    assert span >= 0.0 and need > 0.0
