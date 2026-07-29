"""`PARTNER_REFINE_STALL_STOP=1` -> convergence stop inside `_Refiner.run`.

The partner fork is deadline-bounded (official runtime == budget), so the
only way a case gets cheaper is for the pool workers to RETURN early.
`_Refiner.run` is the terminal time sink of both worker classes:

  * `_worker_refine` -> `refine_prediction` step 5 -> `r2.run(deadline)`
  * `_worker_solve`  -> `finish()` -> `refine_positions` -> `run(deadline)`
    (the stage-2 slice that is what pins an SA worker to `worker_deadline`,
    which is why `PARTNER_SA_STALL_STOP` alone could not shorten a case)

The stop breaks a search phase once `_key()` has not improved by >=
`stall_eps` (relative) inside one stall window, the window being a fraction
of that phase's own remaining span.  Default OFF: `_stall_frac == 0.0` makes
every added predicate a dead branch, so the decision logic and the rng
stream match the pre-port fork.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import legalizer_claude as lg      # noqa: E402
import refiner_claude as rf        # noqa: E402

_ENV = ("PARTNER_REFINE_STALL_STOP", "PARTNER_REFINE_STALL_WINDOW",
        "PARTNER_REFINE_STALL_EPS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _make_opt():
    """Minimal 3-soft-block optimizer -- enough to build a `_Refiner`."""
    import torch
    n = 3
    rects = [(0.0, 0.0, 2.0, 2.0)] * n
    at = torch.tensor([4.0, 4.0, 4.0])
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((n, n))
    p2b = torch.zeros((2, n))
    pins = torch.zeros((2, 2))
    return lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               time.time() + 1.0, seed=7)


def _make_refiner():
    pos = np.array([[0.0, 0.0, 2.0, 2.0],
                    [2.0, 0.0, 2.0, 2.0],
                    [4.0, 0.0, 2.0, 2.0]], dtype=np.float64)
    return rf._Refiner(_make_opt(), pos, seed=3)


# --------------------------------------------------------------------------
# 1. env wiring
# --------------------------------------------------------------------------
def test_default_is_off():
    r = _make_refiner()
    assert r._stall_frac == 0.0
    assert r.stall_eps == 0.002
    assert r._run_stalled is False


def test_flag_and_overrides(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_REFINE_STALL_WINDOW", "0.4")
    monkeypatch.setenv("PARTNER_REFINE_STALL_EPS", "0.01")
    r = _make_refiner()
    assert r._stall_frac == 0.4
    assert r.stall_eps == 0.01


def test_malformed_and_out_of_range_fall_back(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "on")
    monkeypatch.setenv("PARTNER_REFINE_STALL_WINDOW", "1.5")   # not in (0,1)
    monkeypatch.setenv("PARTNER_REFINE_STALL_EPS", "bogus")
    r = _make_refiner()
    assert r._stall_frac == 0.25
    assert r.stall_eps == 0.002


# --------------------------------------------------------------------------
# 2. the discrete phase: micro-accept spinning vs real progress
# --------------------------------------------------------------------------
def _stub_phases(r, key_step, batch_sleep=0.04):
    """Neutralize every move so only the loop control is under test.

    `_discrete_batch` always ACCEPTS a move (acc=1) and improves the key by
    `key_step` (relative) per batch -- i.e. the exact shape the existing
    `stall2 < 4` guard cannot catch, because any accept resets it.
    """
    state = {"key": 1.0}
    r._key = lambda: state["key"]
    r._axis_pass = lambda *a, **k: None
    r._reshape_pass = lambda *a, **k: None
    r._tag_snap = lambda: None
    r._has_overlap = lambda: False
    r._perturb = lambda: None
    r._squeeze_phase = lambda cur, dl: cur
    r._build_swappable = lambda: None

    def _batch(cur_key, deadline, *a, **k):
        time.sleep(batch_sleep)
        state["key"] *= (1.0 - key_step)
        return 1, state["key"]

    r._discrete_batch = _batch
    return state


def test_off_spins_on_micro_accepts_until_the_deadline():
    r = _make_refiner()
    _stub_phases(r, key_step=1e-6)
    t_end = time.time() + 1.2
    r.run(t_end)
    assert r._run_stalled is False
    assert time.time() >= t_end


def test_stall_stop_breaks_out_of_the_micro_accept_spin(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    r = _make_refiner()
    _stub_phases(r, key_step=1e-6)
    t_end = time.time() + 1.2
    r.run(t_end)
    assert r._run_stalled is True
    # window = 0.25 * span = 0.3s; the break must land far short of 1.2s
    assert time.time() < t_end - 0.5


def test_real_progress_is_not_killed(monkeypatch):
    """2%/batch is an order of magnitude above eps -- the anchor re-arms on
    every window check, so the phase must run to its deadline."""
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    r = _make_refiner()
    _stub_phases(r, key_step=0.02)
    t_end = time.time() + 1.0
    r.run(t_end)
    assert r._run_stalled is False
    assert time.time() >= t_end


def test_eps_override_controls_the_verdict(monkeypatch):
    """The same 0.5%/batch trace is 'progress' at eps=0.002 and 'stalled' at
    eps=0.05 -- the knob, not the trace, decides."""
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_REFINE_STALL_EPS", "0.05")
    r = _make_refiner()
    _stub_phases(r, key_step=0.005)
    t_end = time.time() + 1.2
    r.run(t_end)
    assert r._run_stalled is True
    assert time.time() < t_end - 0.5


# --------------------------------------------------------------------------
# 3. phase boundary: a converged continuous phase must not veto the
#    discrete phase (different move class, fresh anchor + fresh window)
# --------------------------------------------------------------------------
def test_continuous_stall_does_not_skip_the_discrete_phase(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_REFINE_STALL_WINDOW", "0.02")
    r = _make_refiner()
    state = _stub_phases(r, key_step=0.02, batch_sleep=0.01)
    calls = {"n": 0}
    inner = r._discrete_batch

    def _counting(cur_key, deadline, *a, **k):
        calls["n"] += 1
        return inner(cur_key, deadline, *a, **k)

    r._discrete_batch = _counting
    # phase 1 sees a constant key (only batches move it) and stalls at once
    r.run(time.time() + 0.6)
    assert calls["n"] > 0
    assert state["key"] < 1.0
