"""`PARTNER_LEGAL_ADMIT=1`: rung (-1) of the direct-channel refine ladder.

The direct channel's coverage boundary is the LADDER, not the sampler: below
n ~ 101 the per-case budget at the 0.3 s operating point cannot pay for rung
0's `_Refiner` build plus fixed-frame legalization, so `refine_prediction`
returns None and the reserved pool worker contributes nothing -- even when the
layout it was handed is *already* legal.  Rung (-1) sits below rung 0: a
hard-legal input is admitted as a candidate rather than discarded.

The invariants worth pinning:

  * OFF is structurally inert -- `_admit_legal_input` returns None before it
    touches numpy, and a full `refine_prediction` on the production input
    (a raw, overlapping prediction) is bit-identical on and off;
  * only a HARD-LEGAL input may be admitted (the same evaluator-faithful
    re-check the guard's input arm uses), which is what makes the flag a
    measured no-op on the production path;
  * admission only ever fires where the shipped path returns None, or --
    on the comparison arm -- where it can only lower the guard cost.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout_refiner as rf  # noqa: E402
from synth_instances import build_instance, make_optimizer  # noqa: E402

_ENV = ("PARTNER_LEGAL_ADMIT", "PARTNER_LEGAL_ADMIT_DEBUG",
        "PARTNER_REFINE_GUARD", "PARTNER_REFINE_GUARD_DEBUG",
        "PARTNER_REFINE_GUARD_MIN_N", "PARTNER_REFINE_KERNEL",
        "PARTNER_ANYTIME_LADDER", "PARTNER_EARLY_EXIT",
        "PARTNER_REFINE_STALL_STOP")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _legal_layout(inst, seed=3, span=10.0):
    """A layout the pipeline itself produced: legal, exact-area, tags seated."""
    dl = time.time() + span
    opt = make_optimizer(inst, seed=seed, deadline=dl)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    out = rf.refine_prediction(opt, pred, dl, seed=11)
    assert out is not None
    return opt, np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# default-off wiring
# ---------------------------------------------------------------------------

def test_legal_admit_defaults_off():
    assert rf._LEGAL_ADMIT is False
    assert rf._LEGAL_ADMIT_DBG is False


def test_legal_admit_flag_tracks_env(monkeypatch):
    import importlib
    monkeypatch.setenv("PARTNER_LEGAL_ADMIT", "1")
    monkeypatch.setenv("PARTNER_LEGAL_ADMIT_DEBUG", "1")
    try:
        importlib.reload(rf)
        assert rf._LEGAL_ADMIT is True
        assert rf._LEGAL_ADMIT_DBG is True
    finally:
        for v in ("PARTNER_LEGAL_ADMIT", "PARTNER_LEGAL_ADMIT_DEBUG"):
            monkeypatch.delenv(v, raising=False)
        importlib.reload(rf)
    assert rf._LEGAL_ADMIT is False


def test_admit_helper_is_inert_when_off():
    """Off, the helper refuses even a provably legal layout, and does so
    without evaluating legality at all."""
    inst = build_instance(n=60, seed=0)
    opt, L = _legal_layout(inst)
    assert rf._guard_hard_ok(opt, L)          # the input IS legal
    assert rf._admit_legal_input(opt, L, "t") is None

    called = []
    orig = rf._guard_hard_ok
    try:
        rf._guard_hard_ok = lambda *a, **k: called.append(1) or True
        assert rf._admit_legal_input(opt, L, "t") is None
    finally:
        rf._guard_hard_ok = orig
    assert called == []


# ---------------------------------------------------------------------------
# what may be admitted
# ---------------------------------------------------------------------------

def test_admit_returns_a_copy_of_a_legal_input(monkeypatch):
    inst = build_instance(n=60, seed=0)
    opt, L = _legal_layout(inst)
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    got = rf._admit_legal_input(opt, L, "t")
    assert got is not None
    assert np.array_equal(got, L)
    assert got is not L                        # a copy, never the caller's array


@pytest.mark.parametrize("break_", ["overlap", "area", "preplaced"])
def test_admit_refuses_an_illegal_input(monkeypatch, break_):
    inst = build_instance(n=60, seed=0, n_preplaced=2)
    opt, L = _legal_layout(inst)
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    P = L.copy()
    kind = np.asarray(opt.kind)
    soft = np.nonzero(kind == 0)[0]
    if break_ == "overlap":
        P[int(soft[1]), :2] = P[int(soft[0]), :2]
    elif break_ == "area":
        j = int(soft[0])
        P[j, 2] *= 1.05
        P[j, 3] *= 1.05
    else:
        idx = np.nonzero(kind == 2)[0]
        if not len(idx):
            pytest.skip("instance has no preplaced block")
        P[int(idx[0]), 0] += 1.0
    assert rf._admit_legal_input(opt, P, "t") is None


def test_raw_prediction_is_never_admitted(monkeypatch):
    """The production input carries raw overlaps, so on the real path rung
    (-1) is a measured no-op -- this is the unit form of that claim."""
    inst = build_instance(n=100, seed=2)
    opt = make_optimizer(inst, seed=5, deadline=time.time() + 10.0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    assert rf._guard_overlap(pred)
    assert rf._admit_legal_input(opt, pred, "t") is None


# ---------------------------------------------------------------------------
# call site 1: no budget at all
# ---------------------------------------------------------------------------

def test_expired_deadline_admits_a_legal_input(monkeypatch):
    inst = build_instance(n=60, seed=0)
    opt, L = _legal_layout(inst)
    dead = time.time() - 1.0

    assert rf.refine_prediction(opt, L, dead, seed=11) is None   # shipped
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    got = rf.refine_prediction(opt, L, dead, seed=11)
    assert got is not None
    assert np.array_equal(np.asarray(got, dtype=np.float64), L)


def test_expired_deadline_still_refuses_an_illegal_input(monkeypatch):
    inst = build_instance(n=100, seed=2)
    opt = make_optimizer(inst, seed=5, deadline=time.time() + 10.0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    assert rf.refine_prediction(opt, pred, time.time() - 1.0, seed=11) is None


def test_recursion_depth_never_admits(monkeypatch):
    """Step 7 re-enters at `_depth=1` with a SCALED layout; rung (-1) is a
    depth-0 device only, so the retry cannot short-circuit into its input."""
    inst = build_instance(n=60, seed=0)
    opt, L = _legal_layout(inst)
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    assert rf.refine_prediction(
        opt, L, time.time() - 1.0, seed=11, _depth=1) is None


# ---------------------------------------------------------------------------
# call site 2: the ladder failed
# ---------------------------------------------------------------------------

def test_ladder_failure_admits_a_legal_input(monkeypatch):
    """A slice far too short for rung 0's `_Refiner` build: shipped -> None
    (the pool slot is wasted), rung (-1) -> the legal input."""
    inst = build_instance(n=100, seed=0)
    opt, L = _legal_layout(inst)

    off = rf.refine_prediction(opt, L, time.time() + 0.004, seed=11)
    if off is not None:
        pytest.skip("ladder unexpectedly closed inside a 4 ms slice")
    monkeypatch.setattr(rf, "_LEGAL_ADMIT", True)
    got = rf.refine_prediction(opt, L, time.time() + 0.004, seed=11)
    assert got is not None
    assert np.array_equal(np.asarray(got, dtype=np.float64), L)


# ---------------------------------------------------------------------------
# call site 3: the comparison arm, and the bit-exact off path
# ---------------------------------------------------------------------------

def _cost(opt, P, hp_ref):
    den = max(getattr(opt, "n_soft_den", 1), 1)
    return rf._guard_cost(opt, np.asarray(P, dtype=np.float64), hp_ref, den)


def test_on_is_never_worse_than_off_on_a_legal_input():
    """The comparison arm is a max under `_guard_cost`, never a heuristic."""
    inst = build_instance(n=60, seed=0)
    opt0, L = _legal_layout(inst)

    def run(on):
        dl = time.time() + 60.0
        opt = make_optimizer(inst, seed=5, deadline=dl)
        prev = rf._LEGAL_ADMIT
        rf._LEGAL_ADMIT = on
        try:
            out = rf.refine_prediction(opt, L, dl, seed=11)
        finally:
            rf._LEGAL_ADMIT = prev
        assert out is not None
        return opt, np.asarray(out, dtype=np.float64)

    opt_a, a = run(False)
    _opt_b, b = run(True)
    hp_ref = max(float(opt_a._hpwl(a)), 1e-9)
    assert _cost(opt_a, b, hp_ref) <= _cost(opt_a, a, hp_ref) + 1e-9


def test_off_path_is_bit_exact_on_the_production_input():
    """The production input overlaps, so rung (-1) can admit nothing and the
    on/off outputs must agree bit for bit."""
    inst = build_instance(n=100, seed=0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(on):
        dl = time.time() + 60.0
        opt = make_optimizer(inst, seed=5, deadline=dl)
        prev = rf._LEGAL_ADMIT
        rf._LEGAL_ADMIT = on
        try:
            out = rf.refine_prediction(opt, pred, dl, seed=11)
        finally:
            rf._LEGAL_ADMIT = prev
        return None if out is None else np.asarray(out, dtype=np.float64)

    a = run(False)
    b = run(True)
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_guard_on_path_is_unchanged_by_the_flag(monkeypatch):
    """With the guard active the input arm already exists; rung (-1) must not
    add a second, different comparison on top of it."""
    inst = build_instance(n=100, seed=0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(on):
        dl = time.time() + 60.0
        opt = make_optimizer(inst, seed=5, deadline=dl)
        prev = rf._LEGAL_ADMIT
        rf._LEGAL_ADMIT = on
        monkeypatch.setattr(rf, "_GUARD_ON", True)
        monkeypatch.setattr(rf, "_GUARD_MIN_N", 95)
        try:
            out = rf.refine_prediction(opt, pred, dl, seed=11)
        finally:
            rf._LEGAL_ADMIT = prev
        return None if out is None else np.asarray(out, dtype=np.float64)

    a = run(False)
    b = run(True)
    assert a is not None and b is not None
    assert np.array_equal(a, b)
