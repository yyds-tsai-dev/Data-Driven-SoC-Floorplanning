"""`PARTNER_REFINE_GUARD=1`: the no-degradation guard on `refine_prediction`.

`refine_prediction` is a pipeline of independently gated stages, and several
of those gates cannot see terms the official cost charges for: `_edge_seat` /
`_cluster_seat` accept on a strict violation drop alone, the step-6 repair
loop arbitrates on `(V, hpwl)` with no area term, and the ladder re-legalizes
into a frame sized from `area_ref * 1.08` which `_seed_tags` then fills.  The
consequence, measured by replaying a repaired-golden layout through the direct
channel (G-T3-1): on tid 87 a feasible input at official cost 1.0000 comes
back at 1.08-1.13, and the ladder's own output (`legal`) is already the worst
point of the whole trajectory.

The guard re-scores the stage snapshots and the input under one
evaluator-style cost and returns the best.  What it may return is only ever a
layout the pipeline already produced (or the caller's own input), so the
invariants worth testing are:

  * OFF (the default) is not merely cheap but structurally inert -- `_snaps`
    is None, so no snapshot is taken and no scoring happens;
  * the guard never returns a layout that is WORSE under its own cost than
    the pipeline output it replaces (it is a max, not a heuristic);
  * a candidate that is not hard-legal can never be returned, which matters
    for exactly one candidate -- the caller's raw prediction, which on the
    production path carries raw overlaps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout_refiner as rf  # noqa: E402
from synth_instances import build_instance, make_optimizer  # noqa: E402

_ENV = ("PARTNER_REFINE_GUARD", "PARTNER_REFINE_GUARD_DEBUG",
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

def test_guard_defaults_off():
    """The shipped import must leave every guard branch dead."""
    assert rf._GUARD_ON is False
    assert rf._GUARD_DBG is False
    assert rf._GUARD_MIN_N == 95


def test_guard_flag_tracks_env(monkeypatch):
    import importlib
    monkeypatch.setenv("PARTNER_REFINE_GUARD", "1")
    monkeypatch.setenv("PARTNER_REFINE_GUARD_MIN_N", "77")
    try:
        importlib.reload(rf)
        assert rf._GUARD_ON is True
        assert rf._GUARD_MIN_N == 77
    finally:
        monkeypatch.delenv("PARTNER_REFINE_GUARD", raising=False)
        monkeypatch.delenv("PARTNER_REFINE_GUARD_MIN_N", raising=False)
        importlib.reload(rf)
    assert rf._GUARD_ON is False


def test_guard_min_n_gate_is_bit_exact(monkeypatch):
    """On an instance below `PARTNER_REFINE_GUARD_MIN_N` the guard is on but
    unreachable, so the output must be bit-identical to the off path."""
    inst = build_instance(n=100, seed=0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(on):
        monkeypatch.setattr(rf, "_GUARD_ON", on)
        monkeypatch.setattr(rf, "_GUARD_MIN_N", 200)
        dl = time.time() + 60.0
        opt = make_optimizer(inst, seed=5, deadline=dl)
        out = rf.refine_prediction(opt, pred, dl, seed=11)
        return None if out is None else np.asarray(out, dtype=np.float64)

    a = run(False)
    b = run(True)          # n=100 < MIN_N=200 -> `_snaps` stays None
    assert a is not None and b is not None
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# `_guard_hard_ok`: what may be handed back
# ---------------------------------------------------------------------------

def test_hard_ok_accepts_a_pipeline_layout():
    inst = build_instance(n=100, seed=0)
    opt, L = _legal_layout(inst)
    assert rf._guard_hard_ok(opt, L)


@pytest.mark.parametrize("break_", ["overlap", "area", "fixed", "preplaced"])
def test_hard_ok_rejects_illegal(break_):
    inst = build_instance(n=100, seed=0, n_preplaced=2)
    opt, L = _legal_layout(inst)
    P = L.copy()
    kind = np.asarray(opt.kind)
    if break_ == "overlap":
        j = int(np.nonzero(kind == 0)[0][1])
        P[j, 0] = P[int(np.nonzero(kind == 0)[0][0]), 0]
        P[j, 1] = P[int(np.nonzero(kind == 0)[0][0]), 1]
    elif break_ == "area":
        j = int(np.nonzero(kind == 0)[0][0])
        P[j, 2] *= 1.05                     # exact-area broken, shape kept
        P[j, 3] *= 1.05
    elif break_ == "fixed":
        idx = np.nonzero(kind == 1)[0]
        if not len(idx):
            pytest.skip("instance has no fixed-shape block")
        j = int(idx[0])
        P[j, 2] += 1.0
    else:
        idx = np.nonzero(kind == 2)[0]
        if not len(idx):
            pytest.skip("instance has no preplaced block")
        j = int(idx[0])
        P[j, 0] += 1.0
    assert not rf._guard_hard_ok(opt, P)


def test_raw_prediction_is_rejected():
    """The production input -- a raw model prediction -- overlaps, so the
    guard can never hand it back.  This is why the guard's realistic value
    lives in the stage snapshots, not in the input arm."""
    inst = build_instance(n=100, seed=2)
    dl = time.time() + 10.0
    opt = make_optimizer(inst, seed=5, deadline=dl)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    assert rf._guard_overlap(pred)
    assert not rf._guard_hard_ok(opt, pred)


# ---------------------------------------------------------------------------
# `_guard_pick`: it is a max, never a heuristic
# ---------------------------------------------------------------------------

def _cost(opt, P):
    den = max(getattr(opt, "n_soft_den", 1), 1)
    return rf._guard_cost(opt, P, max(float(opt._hpwl(P)), 1e-9), den)


def test_pick_prefers_the_better_input():
    inst = build_instance(n=100, seed=0)
    opt, L = _legal_layout(inst)
    # a strictly worse "pipeline output": the same layout blown up, which
    # inflates the bbox and every hpwl term at once
    bad = L.copy()
    bad[:, :2] *= 1.20
    got = rf._guard_pick(opt, bad, [("legal", bad.copy())], L)
    assert np.array_equal(np.asarray(got, dtype=np.float64), L)


def test_pick_keeps_output_when_it_is_best():
    inst = build_instance(n=100, seed=0)
    opt, L = _legal_layout(inst)
    worse = L.copy()
    worse[:, :2] *= 1.20
    got = rf._guard_pick(opt, L, [("legal", worse)], worse)
    assert got is L or np.array_equal(np.asarray(got, dtype=np.float64), L)


def test_pick_never_returns_an_overlapping_snapshot():
    inst = build_instance(n=100, seed=0)
    opt, L = _legal_layout(inst)
    # an overlapping layout that scores BETTER (everything stacked at the
    # origin has a tiny bbox and a tiny hpwl) must still be refused
    cheat = L.copy()
    cheat[:, 0] = float(L[:, 0].min())
    cheat[:, 1] = float(L[:, 1].min())
    assert _cost(opt, cheat) < _cost(opt, L)
    got = rf._guard_pick(opt, L, [("legal", cheat)], None)
    assert not rf._guard_overlap(np.asarray(got, dtype=np.float64))


def test_pick_returns_input_on_failure():
    """Any internal failure degrades to the pipeline's own output."""
    class _Boom:
        n = 4
        area_ref = 1.0
        n_soft_den = 1

        def _hpwl(self, P):
            raise RuntimeError("boom")

    out = np.zeros((4, 4))
    assert rf._guard_pick(_Boom(), out, [], None) is out


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_end_to_end_never_worse_than_a_legal_input(monkeypatch):
    """With a hard-legal input the guard's contract is absolute: the returned
    layout cannot score worse than the input under the guard's own cost."""
    inst = build_instance(n=100, seed=0)
    opt0, L = _legal_layout(inst, seed=9)
    assert rf._guard_hard_ok(opt0, L)

    monkeypatch.setattr(rf, "_GUARD_ON", True)
    dl = time.time() + 6.0
    opt = make_optimizer(inst, seed=9, deadline=dl)
    out = rf.refine_prediction(opt, L.copy(), dl, seed=21)
    assert out is not None
    P = np.asarray(out, dtype=np.float64)
    assert not rf._guard_overlap(P)
    den = max(getattr(opt, "n_soft_den", 1), 1)
    hp_ref = max(float(opt._hpwl(P)), 1e-9)
    assert rf._guard_cost(opt, P, hp_ref, den) \
        <= rf._guard_cost(opt, L, hp_ref, den) + 1e-9


def test_guard_cost_is_cheap():
    """Budget claim: the guard's whole scoring pass is a sub-millisecond
    fixed cost at the tail size it is gated to."""
    inst = build_instance(n=120, seed=7)
    opt, L = _legal_layout(inst, seed=6)
    den = max(getattr(opt, "n_soft_den", 1), 1)
    hp_ref = max(float(opt._hpwl(L)), 1e-9)
    rf._guard_cost(opt, L, hp_ref, den)          # warm
    t0 = time.perf_counter()
    for _ in range(20):
        rf._guard_cost(opt, L, hp_ref, den)
    per = (time.perf_counter() - t0) / 20.0
    assert per < 0.002, f"{per * 1e3:.3f} ms per guard scoring"
