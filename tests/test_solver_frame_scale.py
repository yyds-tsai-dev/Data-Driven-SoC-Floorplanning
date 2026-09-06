"""`PARTNER_FRAME_SCALE_LADDER=1`: the rung-0 frame-scale ladder.

`refine_prediction`'s rung 0 legalizes into a frame of area
`area_ref * 1.08` and `_seed_tags` seats every boundary-tagged block flush
against that frame's walls, so a rung-0 success returns a bbox of EXACTLY
`scale * area_ref` -- the frame constant IS the output area.  The evaluator's
area baseline is the golden bbox (~1.01 * area_ref on the tail), so the
shipped 1.08 is a structural `area_gap` tax that nothing downstream recovers
at the sub-second tiers.

The ladder makes that constant a descending sequence of attempts: the first
scale whose fixed-frame rung legalizes wins.  The invariants worth testing:

  * OFF (the default) is not merely cheap but structurally inert -- the set
    collapses to the single shipped scale, so the loop runs exactly one
    attempt whose per-attempt deadline IS the shipped `_r0_end`;
  * the set parser never produces a set that is empty, non-ascending, or
    physically impossible (a frame below `area_ref` cannot hold the blocks);
  * the scale really is the output bbox when rung 0 closes;
  * the ladder stops at the first success (it must not keep paying for
    attempts after it has a legal layout), and the `MIN_N` budget band
    collapses a multi-scale set back to the shipped single attempt.
"""

from __future__ import annotations

import os
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

_ENV = ("PARTNER_FRAME_SCALE_LADDER", "PARTNER_FRAME_SCALE_SET",
        "PARTNER_FRAME_SCALE_TIGHT_FRAC", "PARTNER_FRAME_SCALE_MIN_N",
        "PARTNER_FRAME_SCALE_DEBUG", "PARTNER_ANYTIME_LADDER",
        "PARTNER_REFINE_GUARD", "PARTNER_RUNG05", "PARTNER_EARLY_EXIT",
        "PARTNER_REFINE_STALL_STOP")

SHIPPED = 1.08


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _run(inst, scales, span=60.0, seed=5, pred=None, stats=False,
         monkeypatch=None):
    """One `refine_prediction` at a given frame-scale arm.

    `scales=None` -> the shipped path (flag unset).  A generous span keeps
    every time gate non-binding, which is what makes the off-path comparison
    reproducible rather than clock-dependent.
    """
    if scales is None:
        os.environ.pop("PARTNER_FRAME_SCALE_LADDER", None)
        os.environ.pop("PARTNER_FRAME_SCALE_SET", None)
    else:
        os.environ["PARTNER_FRAME_SCALE_LADDER"] = "1"
        os.environ["PARTNER_FRAME_SCALE_SET"] = scales
    if pred is None:
        pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    if stats:
        del rf._FS_STATS[:]
    dl = time.time() + span
    opt = make_optimizer(inst, seed=seed, deadline=dl)
    out = rf.refine_prediction(opt, pred.copy(), dl, seed=11)
    return opt, (None if out is None else np.asarray(out, dtype=np.float64))


def _bbox(P):
    return float(((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                 * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))


# ---------------------------------------------------------------------------
# default-off wiring
# ---------------------------------------------------------------------------

def test_defaults_off():
    """The shipped import must leave the ladder dead: one attempt, 1.08."""
    assert rf._FS_DBG is False
    assert rf.frame_scale_set() == (SHIPPED,)


@pytest.mark.parametrize("raw,want", [
    (None, (1.02, 1.05, 1.08)),               # documented default set
    ("1.05", (1.05,)),                        # pure constant shift
    ("1.02,1.05,1.08", (1.02, 1.05, 1.08)),
    (" 1.02 , 1.08 ", (1.02, 1.08)),          # whitespace
    ("1.08,1.02", (1.08,)),                   # non-ascending -> tail dropped
    ("1.05,1.05", (1.05,)),                   # duplicate -> wasted attempt
    ("0.5,1.05", (1.05,)),                    # below area_ref: cannot hold
    ("3.0", (SHIPPED,)),                      # absurd -> shipped
    ("bogus", (SHIPPED,)),                    # malformed -> shipped
    ("", (SHIPPED,)),                         # empty -> shipped
])
def test_set_parsing(monkeypatch, raw, want):
    monkeypatch.setenv("PARTNER_FRAME_SCALE_LADDER", "1")
    if raw is not None:
        monkeypatch.setenv("PARTNER_FRAME_SCALE_SET", raw)
    got = rf.frame_scale_set()
    assert got == want
    assert all(b > a for a, b in zip(got, got[1:]))   # strictly ascending
    assert all(1.0 <= v <= 2.0 for v in got)


def test_set_is_ignored_while_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("PARTNER_FRAME_SCALE_SET", "1.02,1.05")
    assert rf.frame_scale_set() == (SHIPPED,)


# ---------------------------------------------------------------------------
# off path is bit-exact
# ---------------------------------------------------------------------------

def test_off_path_is_bit_exact():
    """`PARTNER_FRAME_SCALE_SET=1.08` is the shipped rung, so the whole
    pipeline output must be bit-identical to the flag-off path."""
    inst = build_instance(n=100, seed=0)
    _o, a = _run(inst, None)
    _o, b = _run(inst, "1.08")
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_off_path_is_bit_exact_with_preplaced():
    inst = build_instance(n=100, seed=0, n_preplaced=3)
    _o, a = _run(inst, None)
    _o, b = _run(inst, "1.08")
    assert a is not None and b is not None
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# the scale IS the output area
# ---------------------------------------------------------------------------

def test_scale_sets_the_bbox_when_rung0_closes(monkeypatch):
    """The load-bearing claim: on a rung-0 success the returned bbox is the
    frame, i.e. exactly `scale * area_ref`.  1.30 is used because it is a
    scale this synthetic instance can legalize into; the mechanism under
    test (frame constant -> bbox) is the same one the shipped 1.08 pays."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    inst = build_instance(n=60, seed=1, n_preplaced=2)
    opt, P = _run(inst, "1.30", stats=True)
    assert P is not None
    ok = [s for s in rf._FS_STATS if s[2]]
    if not ok:
        pytest.skip("rung 0 did not close on this build/machine")
    assert _bbox(P) / opt.area_ref == pytest.approx(1.30, abs=2e-3)


def test_tighter_scale_never_inflates_the_frame(monkeypatch):
    """A tighter scale can fail to legalize (then the expand ladder below
    decides), but it can never produce a LARGER rung-0 frame than a looser
    one -- the frame is the constant, not a search result."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    inst = build_instance(n=40, seed=2)
    o1, P1 = _run(inst, "1.30", stats=True)
    closed1 = any(s[2] for s in rf._FS_STATS)
    o2, P2 = _run(inst, "1.60", stats=True)
    closed2 = any(s[2] for s in rf._FS_STATS)
    if not (closed1 and closed2):
        pytest.skip("rung 0 did not close on this build/machine")
    assert _bbox(P1) / o1.area_ref < _bbox(P2) / o2.area_ref


# ---------------------------------------------------------------------------
# ladder control flow
# ---------------------------------------------------------------------------

def _groups(stats):
    """Split the flat attempt log into per-`refine_prediction` runs: scales
    ascend strictly inside one run, so a non-increase starts a new one."""
    out = []
    for s in stats:
        if not out or s[1] <= out[-1][-1][1]:
            out.append([s])
        else:
            out[-1].append(s)
    return out


def test_ladder_stops_at_the_first_success(monkeypatch):
    monkeypatch.setattr(rf, "_FS_DBG", True)
    inst = build_instance(n=60, seed=1, n_preplaced=2)
    _o, P = _run(inst, "1.30,1.45,1.60", stats=True)
    assert P is not None
    saw_ok = False
    for grp in _groups(rf._FS_STATS):
        assert [s[1] for s in grp] == sorted({s[1] for s in grp})
        for j, s in enumerate(grp):
            if s[2]:
                saw_ok = True
                assert j == len(grp) - 1, "kept trying after a legal frame"
    assert saw_ok, "no rung-0 success to check the stop rule against"


def test_ladder_tries_every_scale_when_none_closes(monkeypatch):
    """The tight probes must be attempted in order and must fall through to
    the shipped scale, which is what keeps the shipped rung reachable."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    inst = build_instance(n=100, seed=0)
    _o, P = _run(inst, "1.02,1.05,1.08", stats=True)
    grp = _groups(rf._FS_STATS)[0]
    if any(s[2] for s in grp):
        pytest.skip("rung 0 closed early on this build/machine")
    assert [s[1] for s in grp] == [1.02, 1.05, 1.08]


def test_min_n_band_collapses_to_the_shipped_attempt(monkeypatch):
    """Below the budget band a multi-scale set costs extra rung-0 attempts
    it cannot afford, so it must collapse to the shipped single attempt --
    keyed on the block count, a reusable instance statistic."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    monkeypatch.setenv("PARTNER_FRAME_SCALE_MIN_N", "200")
    inst = build_instance(n=100, seed=0)
    _o, P = _run(inst, "1.02,1.05,1.08", stats=True)
    assert P is not None
    assert {s[1] for s in rf._FS_STATS} == {SHIPPED}


def test_min_n_band_does_not_touch_a_single_scale_set(monkeypatch):
    """A single-scale set costs no extra attempt, so the band must not
    disable it."""
    monkeypatch.setattr(rf, "_FS_DBG", True)
    monkeypatch.setenv("PARTNER_FRAME_SCALE_MIN_N", "200")
    inst = build_instance(n=100, seed=0)
    _o, P = _run(inst, "1.05", stats=True)
    assert P is not None
    assert {s[1] for s in rf._FS_STATS} == {1.05}


def test_stats_are_only_collected_under_the_debug_flag():
    """`_FS_STATS` is an instrument, not a production side effect."""
    assert rf._FS_DBG is False
    inst = build_instance(n=40, seed=2)
    del rf._FS_STATS[:]
    _o, P = _run(inst, "1.02,1.05,1.08")
    assert P is not None
    assert rf._FS_STATS == []
