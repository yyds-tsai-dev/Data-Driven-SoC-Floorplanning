"""`PARTNER_EARLY_EXIT=1` -> true per-case time reduction (task #9).

The partner fork is deadline-bounded (official runtime == the self-set
budget `0.06*e^(n/20)` clamped to [BUDGET_MIN, BUDGET_MAX]), so a case only
gets cheaper if every stage RETURNS its unspent share instead of handing it
to the next opportunistic consumer:

  * `PARTNER_SA_STALL_STOP` alone was measured quality-negative because
    `finish()` re-spent the reclaimed anneal time on the greedy polish and
    the stage-2 refiner -- the worker still ran to `worker_deadline`.
  * `PARTNER_REFINE_STALL_STOP` was promoted for exactly that re-spend
    (the refiner's saving flowed into `refine_prediction`'s later passes).

EARLY_EXIT keeps both detectors but adds the *clawback*: every stage shrinks
the enclosing deadline by the share it did not use, so the tail stages keep
their PLANNED slice and nothing more, and the worker returns early.  Default
off -> every added predicate is a dead branch and the rng stream is
untouched (none of the added code draws randomness).

See docs/design/2026-08-04-early-exit-true-time-reduction.md.
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

import column_sa_legalizer as lg      # noqa: E402
import layout_refiner as rf           # noqa: E402

_ENV = ("PARTNER_EARLY_EXIT", "PARTNER_EARLY_EXIT_WINDOW",
        "PARTNER_EARLY_EXIT_MIN_WINDOW", "PARTNER_EARLY_EXIT_DEBUG",
        "PARTNER_SA_STALL_STOP", "PARTNER_SA_STALL_WINDOW",
        "PARTNER_SA_STALL_EPS", "PARTNER_REFINE_STALL_STOP",
        "PARTNER_REFINE_STALL_WINDOW", "PARTNER_REFINE_STALL_EPS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _make_opt(budget: float = 1.0, n: int = 3):
    """Minimal all-soft optimizer -- enough to build a `_Refiner` and to run
    `finish` end to end."""
    import torch
    rects = [(2.0 * i, 0.0, 2.0, 2.0) for i in range(n)]
    at = torch.full((n,), 4.0)
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((n, n))
    p2b = torch.zeros((2, n))
    pins = torch.zeros((2, 2))
    return lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               time.time() + budget, seed=7)


def _make_refiner():
    pos = np.array([[0.0, 0.0, 2.0, 2.0],
                    [2.0, 0.0, 2.0, 2.0],
                    [4.0, 0.0, 2.0, 2.0]], dtype=np.float64)
    return rf._Refiner(_make_opt(), pos, seed=3)


# ==========================================================================
# 1. env wiring -- off is off, on implies both stall detectors
# ==========================================================================
def test_default_is_off_everywhere():
    opt = _make_opt()
    r = _make_refiner()
    assert lg.early_exit_on() is False
    assert rf.early_exit_on() is False
    assert opt._early_exit is False
    assert opt._stall_frac == 0.0          # SA detector stays a dead branch
    assert opt._probe_unspent == 0.0
    assert r._early_exit is False
    assert r._stall_frac == 0.0


def test_flag_implies_both_stall_detectors(monkeypatch):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    opt = _make_opt()
    r = _make_refiner()
    assert opt._early_exit is True and r._early_exit is True
    # EARLY_EXIT's own default window (tighter than the stall-stop default:
    # here the reclaimed time is returned, so detection latency is cost)
    assert opt._stall_frac == 0.15
    assert r._stall_frac == 0.15
    # eps defaults are inherited from the respective stall stops
    assert opt.stall_eps == 0.003
    assert r.stall_eps == 0.002


@pytest.mark.parametrize("value", ["on", "true", "True", "ON"])
def test_flag_spellings(monkeypatch, value):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", value)
    assert lg.early_exit_on() is True
    assert rf.early_exit_on() is True


def test_shared_window_override(monkeypatch):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    monkeypatch.setenv("PARTNER_EARLY_EXIT_WINDOW", "0.30")
    assert _make_opt()._stall_frac == 0.30
    assert _make_refiner()._stall_frac == 0.30


def test_phase_specific_window_wins_over_the_shared_one(monkeypatch):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    monkeypatch.setenv("PARTNER_EARLY_EXIT_WINDOW", "0.30")
    monkeypatch.setenv("PARTNER_SA_STALL_WINDOW", "0.40")
    monkeypatch.setenv("PARTNER_REFINE_STALL_WINDOW", "0.05")
    assert _make_opt()._stall_frac == 0.40
    assert _make_refiner()._stall_frac == 0.05


@pytest.mark.parametrize("bad", ["bogus", "1.5", "0", "-0.2"])
def test_malformed_shared_window_falls_back(monkeypatch, bad):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    monkeypatch.setenv("PARTNER_EARLY_EXIT_WINDOW", bad)
    assert _make_opt()._stall_frac == 0.15
    assert _make_refiner()._stall_frac == 0.15


def test_legacy_stall_stop_defaults_are_untouched(monkeypatch):
    """EARLY_EXIT off + the promoted `PARTNER_REFINE_STALL_STOP=1` must keep
    the 0.25 window that carries the promoted evaluator evidence."""
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_SA_STALL_STOP", "1")
    r = _make_refiner()
    opt = _make_opt()
    assert r._stall_frac == 0.25 and r._early_exit is False
    assert opt._stall_frac == 0.25 and opt._early_exit is False


def test_min_window_floor(monkeypatch):
    assert lg.early_exit_min_window() == 0.05
    monkeypatch.setenv("PARTNER_EARLY_EXIT_MIN_WINDOW", "0.2")
    assert lg.early_exit_min_window() == 0.2
    assert rf.early_exit_min_window() == 0.2
    monkeypatch.setenv("PARTNER_EARLY_EXIT_MIN_WINDOW", "bogus")
    assert lg.early_exit_min_window() == 0.05


# ==========================================================================
# 2. `_Refiner.run`: the discrete phase must not re-spend the reclaimed span
# ==========================================================================
def _stub_phases(r, key_step, batch_sleep=0.02):
    """Neutralize every move so only the loop control is under test (same
    harness as tests/test_partner_refine_stall.py): `_discrete_batch` always
    accepts, improving the key by `key_step` relative per batch."""
    state = {"key": 1.0, "batches": 0}
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
        state["batches"] += 1
        state["key"] *= (1.0 - key_step)
        return 1, state["key"]

    r._discrete_batch = _batch
    return state


def test_run_returns_early_when_converged(monkeypatch):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    r = _make_refiner()
    _stub_phases(r, key_step=1e-6)
    t_end = time.time() + 1.5
    r.run(t_end)
    assert r._run_stalled is True
    assert time.time() < t_end - 0.6


def test_run_returns_a_legal_snapshot_on_early_exit(monkeypatch):
    """The early return hands back the `best` snapshot, never a half-applied
    sweep: same block count, same areas, no overlap."""
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    pos = np.array([[0.0, 0.0, 2.0, 2.0],
                    [2.0, 0.0, 2.0, 2.0],
                    [4.0, 0.0, 2.0, 2.0]], dtype=np.float64)
    r = rf._Refiner(_make_opt(), pos, seed=3)
    out = r.run(time.time() + 0.4)
    assert out.shape == pos.shape
    assert np.all(out[:, 2] > 0) and np.all(out[:, 3] > 0)
    assert np.allclose(out[:, 2] * out[:, 3], pos[:, 2] * pos[:, 3], rtol=1e-6)
    _assert_no_overlap(out)


def test_discrete_window_does_not_inflate_with_the_reclaimed_span(
        monkeypatch):
    """The historical rescale sizes the discrete window off the LEFTOVER
    span, so a continuous phase that converged instantly hands the discrete
    phase a nearly full-size window -- i.e. it re-spends the wall clock.
    EARLY_EXIT sizes it off the continuous phase's own (short) span, so it
    must run strictly fewer stalled batches on the same trace."""
    def _run(early: bool):
        if early:
            monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
            monkeypatch.delenv("PARTNER_REFINE_STALL_STOP", raising=False)
        else:
            monkeypatch.delenv("PARTNER_EARLY_EXIT", raising=False)
            monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
        monkeypatch.setenv("PARTNER_REFINE_STALL_WINDOW", "0.25")
        monkeypatch.setenv("PARTNER_EARLY_EXIT_MIN_WINDOW", "0.001")
        r = _make_refiner()
        # phase 1 sees a constant key (only batches move it) -> it stalls on
        # its very first window check, leaving nearly the whole span unused
        state = _stub_phases(r, key_step=1e-6, batch_sleep=0.02)
        t0 = time.time()
        r.run(t0 + 2.0)
        return time.time() - t0, state["batches"]

    t_legacy, n_legacy = _run(early=False)
    t_early, n_early = _run(early=True)
    assert n_early < n_legacy
    assert t_early < t_legacy


# ==========================================================================
# 3. `finish()` clawback: the SA saving is returned, not re-spent
# ==========================================================================
def _instrument_finish(opt, anneal_cost, refine_sleep=0.0):
    """Replace the two expensive stages with instrumented stubs.

    `_anneal` returns after `anneal_cost` seconds regardless of the deadline
    it is handed (that is what a stall break looks like from `finish`'s point
    of view); `refine_positions` records the span it was given.
    """
    seen = {"anneal_spans": [], "refine_span": None, "polish_span": None}
    real_anneal = opt._anneal

    def _anneal(cols, deadline, cur_cost, **kw):
        seen["anneal_spans"].append(deadline - time.time())
        time.sleep(anneal_cost)
        return opt._snapshot(cols), cur_cost

    def _polish(cols, cur_cost, deadline):
        seen["polish_span"] = deadline - time.time()
        return cols

    opt._anneal = _anneal
    opt._greedy_polish = _polish
    opt._real_anneal = real_anneal

    import layout_refiner as _rf
    real_rp = _rf.refine_positions

    def _refine_positions(o, pos, deadline, seed=0):
        seen["refine_span"] = deadline - time.time()
        if refine_sleep:
            time.sleep(refine_sleep)
        return None

    _rf.refine_positions = _refine_positions
    seen["_restore"] = real_rp
    return seen


def _restore_refine_positions(seen):
    import layout_refiner as _rf
    _rf.refine_positions = seen["_restore"]


@pytest.mark.parametrize("early", [False, True])
def test_finish_clawback_returns_the_unspent_anneal_share(monkeypatch, early):
    """A 10 s planned anneal that converges in 0.05 s: with EARLY_EXIT off
    the stage-2 refiner inherits the ~10 s (it is handed an absolute
    deadline); with EARLY_EXIT on it keeps only its own planned slice."""
    if early:
        monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    opt = _make_opt(budget=10.0)
    opt.prepare()
    seen = _instrument_finish(opt, anneal_cost=0.05)
    try:
        opt.finish(opt.deadline, max_runs=1)
    finally:
        _restore_refine_positions(seen)
    # refine_t = min(0.30*rem, 7.0) ~= 3.0 s for a 10 s budget
    assert seen["refine_span"] is not None
    if early:
        assert seen["refine_span"] < 3.4
    else:
        assert seen["refine_span"] > 6.0


def test_finish_clawback_is_bounded_by_now(monkeypatch):
    """An anneal that OVERRUNS its planned end must not push the deadline
    into the past (clawback is max(now, ...))."""
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    opt = _make_opt(budget=3.0)
    opt.prepare()
    seen = _instrument_finish(opt, anneal_cost=0.30)
    try:
        opt.finish(time.time() + 0.5, max_runs=1)
    finally:
        _restore_refine_positions(seen)
    assert seen["polish_span"] is not None
    # never negative-infinite: the polish window is finite and small
    assert seen["polish_span"] < 1.0


def test_probe_unspent_is_reclaimed_by_run(monkeypatch):
    """`run` = probe + finish.  A probe that converges instantly must not
    lengthen `finish`."""
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    opt = _make_opt(budget=3.0)
    seen = {"finish_span": None}

    def _probe(t_each):
        opt._probe_unspent = 1.2
        opt._best_probe = None
        return 0.0

    def _finish(deadline, max_runs=2):
        seen["finish_span"] = deadline - time.time()
        return [(0.0, 0.0, 2.0, 2.0)] * opt.n

    opt.probe = _probe
    opt.finish = _finish
    opt.run()
    assert seen["finish_span"] is not None
    assert seen["finish_span"] < 3.0 - 1.0


def test_probe_unspent_is_ignored_when_flag_off():
    opt = _make_opt(budget=3.0)
    seen = {"finish_span": None}

    def _probe(t_each):
        opt._probe_unspent = 1.2      # never written with the flag off
        opt._best_probe = None
        return 0.0

    def _finish(deadline, max_runs=2):
        seen["finish_span"] = deadline - time.time()
        return [(0.0, 0.0, 2.0, 2.0)] * opt.n

    opt.probe = _probe
    opt.finish = _finish
    opt.run()
    assert seen["finish_span"] > 2.5


# ==========================================================================
# 4. end-to-end on the real (unstubbed) stages: legality + determinism
# ==========================================================================
def _assert_no_overlap(pos: np.ndarray, tol: float = 1e-7) -> None:
    n = len(pos)
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            assert not (ox > tol and oy > tol), f"blocks {i},{j} overlap"


def _solve_once(budget: float, seed: int = 7):
    import torch
    n = 6
    rects = [(2.0 * i, 0.0, 2.0, 2.0) for i in range(n)]
    at = torch.full((n,), 4.0)
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((n, n))
    b2b[0, 1] = b2b[1, 0] = 1.0
    b2b[2, 3] = b2b[3, 2] = 1.0
    p2b = torch.zeros((2, n))
    pins = torch.zeros((2, 2))
    opt = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                              time.time() + budget, seed=seed)
    return opt.run()


def test_early_exit_solution_is_legal_and_area_exact(monkeypatch):
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    out = np.asarray(_solve_once(1.0), dtype=np.float64)
    assert len(out) == 6
    _assert_no_overlap(out)
    assert np.allclose(out[:, 2] * out[:, 3], 4.0, rtol=2e-3)


def test_early_exit_consumes_no_extra_randomness(monkeypatch):
    """Determinism claim, stated precisely.

    A wall-clock-bounded SA is never bit-reproducible across two runs -- the
    move count depends on how many iterations fit in the span -- so "same
    seed, same layout" is NOT a property of this fork, with or without the
    flag.  What EARLY_EXIT must not do is perturb the rng STREAM: every
    predicate it adds is a time comparison, so for the same sequence of
    stages the rng state after `finish` must be identical on and off.  Any
    divergence here would mean the flag changed the search itself rather
    than only when it stops.
    """
    def _state(early: bool):
        if early:
            monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
        else:
            monkeypatch.delenv("PARTNER_EARLY_EXIT", raising=False)
        opt = _make_opt(budget=10.0)
        opt.prepare()
        seen = _instrument_finish(opt, anneal_cost=0.02)
        try:
            opt.finish(opt.deadline, max_runs=1)
        finally:
            _restore_refine_positions(seen)
        return opt.rng.getstate()

    assert _state(False) == _state(True)


def test_early_exit_repeated_solves_stay_legal(monkeypatch):
    """The stop can fire at any iteration boundary, so legality must hold
    for every trace, not just the lucky one."""
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    for budget in (0.3, 0.5, 0.9):
        out = np.asarray(_solve_once(budget, seed=11), dtype=np.float64)
        assert len(out) == 6
        _assert_no_overlap(out)
        assert np.allclose(out[:, 2] * out[:, 3], 4.0, rtol=2e-3)


def test_off_path_solution_matches_the_pre_flag_fork():
    """With every EARLY_EXIT env absent the solver must behave exactly as it
    did before the port: the SA detector stays disabled, no clawback state
    is ever written, and the layout is legal."""
    opt = _make_opt(budget=0.6, n=6)
    assert opt._early_exit is False and opt._stall_frac == 0.0
    out = np.asarray(opt.run(), dtype=np.float64)
    assert opt._probe_unspent == 0.0
    _assert_no_overlap(out)
