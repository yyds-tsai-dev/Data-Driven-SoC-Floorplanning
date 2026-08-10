"""`PARTNER_RUNG0_TIGHTEN=1`: the frame anneal on the rung-0 success path.

`_tighten` is the legalization ladder's frame anneal -- shrink both max edges
a few percent, drag the edge-seated tag blocks onto the new walls, re-legalize,
and keep the tightest frame that still closes.  Every EXPAND rung runs it on
success.  The rung-0 (fixed-frame) success path does not: it goes straight
from `legalize_soft` to `_assemble_clusters` and banks `r.P`.  Since
`_seed_tags` seats the tagged blocks flush against the rung-0 walls, that
layout's bbox IS the rung-0 frame constant, so a rung-0 candidate ships an
area the expand rungs would have annealed away.  The flag closes that gap; it
is orthogonal to `PARTNER_FRAME_SCALE_LADDER`, which sets the constant.

The invariants worth pinning:

  * OFF is structurally inert -- `rung0_tighten_frac()` is 0.0, the rung-0
    path calls `_tighten` exactly zero times, and two off runs agree bit for
    bit (the flag cannot perturb the shipped path through timing either,
    because nothing on it reads the flag);
  * ON, `_tighten` really does run on the rung-0 path, and it can only
    SHRINK the frame it was handed (it is an anneal with a revert, not a
    search that may inflate);
  * ON, the returned layout is still hard-legal by the evaluator-faithful
    check, and a rung-0 close is never LOST: an anneal whose layout cannot be
    assembled reverts to the shipped one.
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

_ENV = ("PARTNER_RUNG0_TIGHTEN", "PARTNER_RUNG0_TIGHTEN_FRAC",
        "PARTNER_RUNG0_TIGHTEN_MIN_SLACK",
        "PARTNER_RUNG0_TIGHTEN_DEBUG", "PARTNER_FRAME_SCALE_LADDER",
        "PARTNER_FRAME_SCALE_SET", "PARTNER_FRAME_SCALE_DEBUG",
        "PARTNER_FRAME_SCALE_MIN_N", "PARTNER_FRAME_SCALE_TIGHT_FRAC",
        "PARTNER_ANYTIME_LADDER", "PARTNER_REFINE_GUARD", "PARTNER_RUNG05",
        "PARTNER_EARLY_EXIT", "PARTNER_REFINE_STALL_STOP",
        "PARTNER_LEGAL_ADMIT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    del rf._R0TG_STATS[:]
    del rf._FS_STATS[:]
    yield


# The synthetic instances do not legalize into the shipped 1.08 frame (rung 0
# fails and the expand ladder takes over), so the rung-0 success path -- the
# only thing this flag touches -- would never be reached.  `scale` widens the
# rung-0 frame via the (independent, already-shipped) frame-scale ladder until
# it closes; the mechanism under test is unchanged by the constant.
_CLOSING_SCALE = "1.30"


def _run(inst, on, span=60.0, seed=5, frac=None, scale=_CLOSING_SCALE,
         depth=0, monkeypatch=None):
    """One `refine_prediction` at the on/off arm of the rung-0 anneal.

    A generous span keeps every time gate non-binding, which is what makes
    the paired comparison reproducible rather than clock-dependent.

    `depth=1` suppresses the step-7 recompression rerun, which re-enters
    `refine_prediction` and therefore runs the ladder a SECOND time -- the
    call-attribution tests need exactly one ladder per run.
    """
    if monkeypatch is not None:
        if scale is None:
            monkeypatch.delenv("PARTNER_FRAME_SCALE_LADDER", raising=False)
            monkeypatch.delenv("PARTNER_FRAME_SCALE_SET", raising=False)
        else:
            monkeypatch.setenv("PARTNER_FRAME_SCALE_LADDER", "1")
            monkeypatch.setenv("PARTNER_FRAME_SCALE_SET", scale)
        if on:
            monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN", "1")
            if frac is not None:
                monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN_FRAC", frac)
        else:
            monkeypatch.delenv("PARTNER_RUNG0_TIGHTEN", raising=False)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    dl = time.time() + span
    opt = make_optimizer(inst, seed=seed, deadline=dl)
    out = rf.refine_prediction(opt, pred, dl, seed=11, _depth=depth)
    return opt, (None if out is None else np.asarray(out, dtype=np.float64))


def _count_tighten(monkeypatch):
    """Instrument `_Refiner._tighten` with a call counter (calls still run)."""
    calls: list = []
    orig = rf._Refiner._tighten

    def spy(self, deadline, rounds=10):
        calls.append(float(deadline))
        return orig(self, deadline, rounds)

    monkeypatch.setattr(rf._Refiner, "_tighten", spy)
    return calls


def _rung0_closed():
    return any(s[2] for s in rf._FS_STATS)


def _bb(P):
    return float(((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                 * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))


# ---------------------------------------------------------------------------
# default-off wiring
# ---------------------------------------------------------------------------

def test_defaults_off():
    """The shipped import must leave the anneal dead."""
    assert rf._R0TG_DBG is False
    assert rf.rung0_tighten_frac() == 0.0


@pytest.mark.parametrize("raw,want", [
    (None, 0.45),          # documented default == the expand rungs' share
    ("0.2", 0.2),
    ("0.9", 0.9),
    ("0", 0.45),           # out of (0, 1) -> default
    ("1.0", 0.45),
    ("-0.3", 0.45),
    ("bogus", 0.45),
    ("", 0.45),
])
def test_frac_parsing(monkeypatch, raw, want):
    monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN", "1")
    if raw is not None:
        monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN_FRAC", raw)
    assert rf.rung0_tighten_frac() == pytest.approx(want)


def test_frac_is_ignored_while_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN_FRAC", "0.9")
    assert rf.rung0_tighten_frac() == 0.0


@pytest.mark.parametrize("raw,want", [
    (None, 0.0),           # default: ungated
    ("0", 0.0),
    ("0.005", 0.005),
    ("0.5", 0.5),
    ("0.51", 0.0),         # out of range -> ungated
    ("-0.1", 0.0),
    ("bogus", 0.0),
    ("", 0.0),
])
def test_min_slack_parsing(monkeypatch, raw, want):
    if raw is not None:
        monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN_MIN_SLACK", raw)
    assert rf.rung0_tighten_min_slack() == pytest.approx(want)


def test_min_slack_gate_suppresses_a_tight_frame(monkeypatch):
    """A frame with no slack over `scale * area_ref` cannot be annealed by
    the fixed (0.97, 0.988) schedule, so the gate must skip it -- and a
    slack demand nothing can meet turns the whole arm back into the shipped
    path."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN_MIN_SLACK", "0.5")
    inst = build_instance(n=100, seed=0, n_preplaced=3)
    calls = _count_tighten(monkeypatch)
    _o, P = _run(inst, True, depth=1, monkeypatch=monkeypatch)
    assert P is not None
    if not _rung0_closed():
        pytest.skip("rung 0 did not close on this build/machine")
    assert calls == [], "the gate let an unshrinkable frame through"


def test_stats_are_only_collected_under_the_debug_flag(monkeypatch):
    """`_R0TG_STATS` is an instrument, not a production side effect."""
    assert rf._R0TG_DBG is False
    inst = build_instance(n=60, seed=1, n_preplaced=2)
    _o, P = _run(inst, True, monkeypatch=monkeypatch)
    assert P is not None
    assert rf._R0TG_STATS == []


# ---------------------------------------------------------------------------
# off path is inert
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed,npre", [(60, 1, 2), (100, 0, 0), (40, 2, 0)])
def test_off_path_never_anneals_rung_0(monkeypatch, n, seed, npre):
    """The load-bearing off-claim: with the flag unset the rung-0 success
    path calls `_tighten` zero times, so the branch is the shipped one.

    Attribution is by rung: a run whose rung 0 produced the legal layout
    (`_FS_STATS[-1][2]`) broke out of the ladder before any expand rung, so
    every `_tighten` in it would have to be the one this flag adds."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    inst = build_instance(n=n, seed=seed, n_preplaced=npre)
    calls = _count_tighten(monkeypatch)
    _o, P = _run(inst, False, depth=1, monkeypatch=monkeypatch)
    assert P is not None
    assert len(rf._FS_STATS) == 1, "expected exactly one ladder per run"
    if not _rung0_closed():
        pytest.skip("rung 0 did not close on this build/machine")
    assert calls == []


def test_off_path_is_reproducible(monkeypatch):
    """Two off runs agree bit for bit -- the baseline the on-arm is paired
    against is deterministic, not clock-dependent."""
    inst = build_instance(n=100, seed=0)
    _o, a = _run(inst, False, monkeypatch=monkeypatch)
    _o, b = _run(inst, False, monkeypatch=monkeypatch)
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_explicit_zero_is_off(monkeypatch):
    inst = build_instance(n=100, seed=0)
    _o, a = _run(inst, False, monkeypatch=monkeypatch)
    monkeypatch.setenv("PARTNER_RUNG0_TIGHTEN", "0")
    _o, b = _run(inst, False, monkeypatch=monkeypatch)
    assert a is not None and b is not None
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# on path: the anneal runs, and only ever shrinks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed,npre", [(60, 1, 2), (100, 0, 0), (40, 2, 0)])
def test_on_path_anneals_rung_0(monkeypatch, n, seed, npre):
    monkeypatch.setattr(rf, "_FS_DBG", True)
    inst = build_instance(n=n, seed=seed, n_preplaced=npre)
    calls = _count_tighten(monkeypatch)
    _o, P = _run(inst, True, depth=1, monkeypatch=monkeypatch)
    assert P is not None
    assert len(rf._FS_STATS) == 1
    if not _rung0_closed():
        pytest.skip("rung 0 did not close on this build/machine")
    assert len(calls) >= 1, "rung 0 closed but the anneal never ran"


def test_anneal_never_inflates_the_frame(monkeypatch):
    """`_tighten` snapshots and reverts each failed round, so the frame it
    returns is never larger than the one it was handed."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    monkeypatch.setattr(rf, "_R0TG_DBG", True)
    inst = build_instance(n=60, seed=1, n_preplaced=2)
    _o, P = _run(inst, True, monkeypatch=monkeypatch)
    assert P is not None
    if not rf._R0TG_STATS:
        pytest.skip("rung 0 did not close on this build/machine")
    for _n, bb0, bb1, _t in rf._R0TG_STATS:
        assert bb1 <= bb0 + 1e-9


@pytest.mark.parametrize("n,seed,npre", [(60, 1, 2), (40, 2, 0), (100, 0, 3)])
@pytest.mark.parametrize("scale", [_CLOSING_SCALE, None])
def test_on_path_output_is_hard_legal(monkeypatch, n, seed, npre, scale):
    """The hard-legality red line, checked with the same evaluator-faithful
    predicate the refine guard uses.  `scale=None` is the shipped frame, where
    rung 0 usually fails -- the flag must be harmless there too."""
    inst = build_instance(n=n, seed=seed, n_preplaced=npre)
    opt, P = _run(inst, True, scale=scale, monkeypatch=monkeypatch)
    assert P is not None
    assert P.shape == (opt.n, 4)
    assert np.isfinite(P).all()
    assert (P[:, 2] > 0).all() and (P[:, 3] > 0).all()
    assert rf._guard_hard_ok(opt, P)


def test_on_path_never_loses_a_rung0_close(monkeypatch):
    """An anneal that cannot be assembled reverts to the shipped layout, so
    the on-arm returns a layout wherever the off-arm does."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    for n, seed, npre in ((60, 1, 2), (40, 2, 0), (100, 0, 3), (80, 4, 1)):
        inst = build_instance(n=n, seed=seed, n_preplaced=npre)
        _o, off = _run(inst, False, monkeypatch=monkeypatch)
        _o, on = _run(inst, True, monkeypatch=monkeypatch)
        if off is not None:
            assert on is not None, f"n={n} seed={seed}: on-arm lost a close"


# ---------------------------------------------------------------------------
# budget: the anneal is bounded by the ladder deadline, not by wall clock
# ---------------------------------------------------------------------------

def test_anneal_deadline_stays_inside_the_slice(monkeypatch):
    """The `_tighten` deadline must sit inside the carved ladder deadline
    (`worker deadline - reserve`), so the anneal can never spend the
    violation-repair reserve."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    span = 20.0
    inst = build_instance(n=60, seed=1, n_preplaced=2)
    calls = _count_tighten(monkeypatch)
    t0 = time.time()
    _o, P = _run(inst, True, span=span, depth=1, monkeypatch=monkeypatch)
    assert P is not None
    if not _rung0_closed() or not calls:
        pytest.skip("rung 0 did not close on this build/machine")
    # reserve = min(3.5, 0.3 * span) (the extra `slice_ > 8` share is
    # depth-0 only); the ladder deadline is `worker deadline - reserve`
    ladder_end = t0 + span - min(3.5, 0.3 * span)
    for dl in calls:
        assert dl <= ladder_end + 1e-6


def test_a_tiny_frac_leaves_the_close_intact(monkeypatch):
    """A near-zero share buys (almost) no anneal, but must never break the
    rung-0 close -- the arm degrades to the shipped path, not to failure."""
    inst = build_instance(n=60, seed=1, n_preplaced=2)
    _o, P = _run(inst, True, frac="0.001", monkeypatch=monkeypatch)
    opt, _ = _run(inst, False, monkeypatch=monkeypatch)
    assert P is not None
    assert rf._guard_hard_ok(opt, P)
