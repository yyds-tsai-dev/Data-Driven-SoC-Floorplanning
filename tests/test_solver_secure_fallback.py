"""`PARTNER_REFINE_SECURE_FALLBACK=1` -> a bounded secure rung where the
shipped direct-refine ladder is about to return None.

`refine_prediction` is all-or-nothing: rung 0 (fixed frame) and the expand
rungs are tried in order and, when none of them legalizes inside the ladder
window, the function returns None -- the reserved pool worker contributes
nothing and the violation-repair reserve carved off the top
(`PARTNER_REFINE_RES_FRAC`, 0.45 of the worker slice) is thrown away unspent.
On the official suite that is 9/9 reserved Flow workers returning None on the
three cases sitting at the measured rung-0 completion boundary, which
therefore ship the COLUMN champion.

ON, and only where the shipped path returns None: spend what is left of the
worker deadline on a bounded loose-frame rung (the same pin-less rung the
ladder runs last, with a generous frame and a deadline), then fall through
into the shipped refiner/repair tail.

What must hold:
  * default off, and the usual flag spellings;
  * off -> a ladder that cannot close still returns None;
  * on  -> the same case comes back with a hard-legal, overlap-free layout;
  * a ladder that DOES close never reaches the fallback (off-path identity
    for every successful candidate, on either arm);
  * no time left -> still None, and the fallback never outlives the worker
    deadline it was handed;
  * the fallback never replays an expansion a shipped rung already refuted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "tests", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout_refiner as rf                                   # noqa: E402
from synth_instances import build_instance, make_optimizer     # noqa: E402

_ENV = ("PARTNER_REFINE_SECURE_FALLBACK", "PARTNER_SECURE_FALLBACK_EXPANDS",
        "PARTNER_SECURE_FALLBACK_FRAC", "PARTNER_SECURE_FALLBACK_RUNG_FRAC",
        "PARTNER_SECURE_FALLBACK_TIGHTEN", "PARTNER_SECURE_FALLBACK_TAIL",
        "PARTNER_ANYTIME_LADDER", "PARTNER_REFINE_RES_FRAC",
        "PARTNER_LEGAL_ADMIT", "PARTNER_REFINE_GUARD")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _case(n: int = 40, seed: int = 0):
    """A structured synthetic instance (fixed shapes, preplaced obstacles,
    boundary tags, clusters, MIB groups) plus a "prediction" the shipped
    ladder CAN close, so the only thing separating the two arms below is the
    frame gate."""
    inst = build_instance(n=n, seed=seed)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    return inst, pred


def _refine(inst, pred, span: float, seed: int = 11):
    """One `_worker_refine`-shaped call: fresh optimizer, fresh deadline."""
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    t0 = time.time()
    out = rf.refine_prediction(opt, pred, deadline, seed=seed)
    return out, time.time() - t0, deadline


def _tight_frames_fail(monkeypatch, min_ratio: float):
    """Make every legalization into a frame below `min_ratio * area_ref`
    report failure, while the loose ones run for real.

    That is the shape of the production failure -- rung 0 (frame
    `1.02..1.08 * area_ref`) and the small expand rungs cannot close inside
    the window they are given -- reproduced deterministically, so the test
    does not depend on this machine's wall clock.

    The fallback is pointed at an escape frame WIDER than any rung the
    shipped ladder owns (its widest is 0.28), so "only the fallback can
    close this" is a property of the frame and not of the clock.  The
    measured default (0.12) is deliberately inside the shipped ladder's
    range, which is why it has to be overridden here.
    """
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_EXPANDS", "0.6")
    real = rf._Refiner.legalize_soft

    def _gated(self_, max_sweeps=14, deadline=None, fine=False):
        ok = real(self_, max_sweeps, deadline=deadline, fine=fine)
        aref = max(float(getattr(self_.opt, "area_ref", 0.0)), 1e-9)
        frame = (self_.xmax - self_.xmin) * (self_.ymax - self_.ymin)
        return bool(ok) if frame >= min_ratio * aref else False

    monkeypatch.setattr(rf._Refiner, "legalize_soft", _gated)


def _assert_no_overlap(pos: np.ndarray, tol: float = 1e-7) -> None:
    n = len(pos)
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            assert not (ox > tol and oy > tol), f"blocks {i},{j} overlap"


# ==========================================================================
# 1. env wiring
# ==========================================================================
def test_default_is_off():
    assert rf.secure_fallback_on() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "True", "ON"])
def test_flag_spellings(monkeypatch, value):
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", value)
    assert rf.secure_fallback_on() is True


def test_expand_defaults_are_ascending():
    assert rf.secure_fallback_expands([]) == (0.12, 0.28)


def test_expands_drop_what_a_shipped_rung_already_refuted():
    assert rf.secure_fallback_expands([0.12]) == (0.28,)
    # ... but never return an empty ladder: a loose candidate beats none
    assert rf.secure_fallback_expands([0.12, 0.28]) == (0.28,)


@pytest.mark.parametrize("raw,want", [
    ("0.5", (0.5,)),
    ("0.2,0.4,0.9", (0.2, 0.4, 0.9)),
    ("0.4,0.2", (0.4,)),          # non-ascending tail dropped
    ("bogus", (0.12, 0.28)),      # malformed -> the shipped default
    ("", (0.12, 0.28)),
])
def test_expand_env_parsing(monkeypatch, raw, want):
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_EXPANDS", raw)
    assert rf.secure_fallback_expands([]) == want


@pytest.mark.parametrize("bad", ["bogus", "1.5", "0", "-0.2", "1.0"])
def test_malformed_fraction_falls_back(monkeypatch, bad):
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_FRAC", bad)
    assert rf.anytime_frac("PARTNER_SECURE_FALLBACK_FRAC", 0.45) == 0.45


# ==========================================================================
# 2. the mechanism: a ladder that cannot close
# ==========================================================================
def test_off_path_returns_none_when_no_tight_frame_closes(monkeypatch):
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    inst, pred = _case()
    out, _, _ = _refine(inst, pred, 4.0)
    assert out is None, "shipped ladder must still give up"


def test_on_path_delivers_a_legal_candidate(monkeypatch):
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    inst, pred = _case()
    out, _, _ = _refine(inst, pred, 4.0)
    assert out is not None, "the secure rung must deliver a candidate"
    pos = np.asarray(out, dtype=np.float64)
    assert pos.shape == pred.shape
    assert np.isfinite(pos).all()
    _assert_no_overlap(pos)


def test_on_path_candidate_is_hard_legal(monkeypatch):
    """The evaluator-faithful re-check the refine guard uses."""
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    inst, pred = _case()
    deadline = time.time() + 4.0
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    out = rf.refine_prediction(opt, pred, deadline, seed=11)
    assert out is not None
    assert rf._guard_hard_ok(opt, np.asarray(out, dtype=np.float64))


def test_fallback_is_skipped_when_a_shipped_rung_closes(monkeypatch):
    """Off-path identity for every candidate the ladder already delivers:
    the fallback is not merely inert there, it is never entered."""
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    calls = []
    real = rf._secure_fallback

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(rf, "_secure_fallback", _spy)
    inst, pred = _case()
    out, _, _ = _refine(inst, pred, 4.0)
    assert out is not None, "the shipped ladder should close on this case"
    assert not calls, "fallback ran on a case the ladder already solved"


def test_fallback_never_runs_with_the_flag_off(monkeypatch):
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    calls = []
    monkeypatch.setattr(rf, "_secure_fallback",
                        lambda *a, **k: (calls.append(1), (None, "x"))[1])
    inst, pred = _case()
    out, _, _ = _refine(inst, pred, 4.0)
    assert out is None
    assert not calls, "off path must not reach the fallback at all"


# ==========================================================================
# 3. budget discipline
# ==========================================================================
def test_deadline_exhaustion_still_returns_none(monkeypatch):
    """No frame closes at all -> the fallback spends its window and returns
    the shipped None rather than an illegal layout."""
    def _never(self_, max_sweeps=14, deadline=None, fine=False):
        return False

    monkeypatch.setattr(rf._Refiner, "legalize_soft", _never)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    inst, pred = _case()
    out, _, _ = _refine(inst, pred, 2.0)
    assert out is None


def test_no_budget_left_short_circuits(monkeypatch):
    """A ladder that consumed the whole worker deadline leaves the fallback
    nothing: it must return immediately, not build a refiner."""
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    inst, _ = _case()
    opt = make_optimizer(inst, seed=3, deadline=time.time())
    builds = []
    real_build = rf._Refiner.__init__

    def _spy(self_, *a, **k):
        builds.append(1)
        return real_build(self_, *a, **k)

    monkeypatch.setattr(rf._Refiner, "__init__", _spy)
    got, tag = rf._secure_fallback(opt, np.zeros((3, 4)), None, {}, 0,
                                   time.time() - 0.5, [])
    assert got is None and tag == "notime"
    assert not builds


def test_fallback_respects_the_worker_deadline(monkeypatch):
    """The whole runtime argument: the fallback is bounded by the SAME
    deadline the failed ladder already owned."""
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    inst, pred = _case()
    span = 2.0
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    rf.refine_prediction(opt, pred, deadline, seed=11)
    # generous slack for a loaded CI box; the point is that the fallback does
    # not add a *multiple* of the span the way an unbounded rung does
    assert time.time() <= deadline + 0.9 * span


def test_every_fallback_legalization_is_bounded(monkeypatch):
    """The shipped loose rung calls `legalize_soft()` with no deadline at
    all -- that is the overrun source; the fallback must never do it."""
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    inside = {"on": False}
    seen = []
    real_ls = rf._Refiner.legalize_soft
    real_fb = rf._secure_fallback

    def _rec(self_, max_sweeps=14, deadline=None, fine=False):
        if inside["on"]:
            seen.append(deadline)
        return real_ls(self_, max_sweeps, deadline=deadline, fine=fine)

    def _fb(*a, **k):
        inside["on"] = True
        try:
            return real_fb(*a, **k)
        finally:
            inside["on"] = False

    monkeypatch.setattr(rf._Refiner, "legalize_soft", _rec)
    monkeypatch.setattr(rf, "_secure_fallback", _fb)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert seen, "the fallback rung never ran"
    assert all(d is not None for d in seen), \
        "fallback must bound every legalization it starts"


def test_single_rung_configuration(monkeypatch):
    """`PARTNER_SECURE_FALLBACK_EXPANDS` reduces the fallback to exactly one
    generous rung -- the literal reading of 'run the secure rung once'."""
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_EXPANDS", "0.6")
    inst, pred = _case()
    out, _, _ = _refine(inst, pred, 4.0)
    assert out is not None
    _assert_no_overlap(np.asarray(out, dtype=np.float64))


def test_fractional_tail_is_opt_in_and_scoped(monkeypatch):
    """`PARTNER_SECURE_FALLBACK_FRAC_TAIL` re-sizes the repair tail's
    absolute gates -- but only on the fallback path, and only when asked.

    `_cluster_seat` is the witness: the shipped tail hands it `t_hard - 0.9`
    and `t_hard - 0.05`, constants that are already in the past whenever the
    reserve is smaller than they are (i.e. on exactly the candidates the
    fallback produces)."""
    _tight_frames_fail(monkeypatch, min_ratio=1.9)
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    seen = []
    real_cs = rf._cluster_seat

    def _cs(opt, out, deadline=None):
        seen.append(deadline)
        return real_cs(opt, out, deadline=deadline)

    monkeypatch.setattr(rf, "_cluster_seat", _cs)
    inst, pred = _case()
    # under 2.5 s: the step-7 recompression rerun (a whole second
    # `refine_prediction` at `t_hard - 0.05`, whose own tail gates would show
    # up here shifted by 0.05) cannot fire, so the offsets are unambiguous
    span = 2.0
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    assert rf.refine_prediction(opt, pred, deadline, seed=11) is not None
    # default: the shipped absolute offsets survive the fallback
    offsets = sorted(round(deadline - d, 4) for d in seen)
    assert offsets[-1] == pytest.approx(0.9, abs=1e-3)
    assert offsets[0] == pytest.approx(0.05, abs=1e-3)

    seen.clear()
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_FRAC_TAIL", "1")
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    assert rf.refine_prediction(opt, pred, deadline, seed=11) is not None
    offsets = sorted(round(deadline - d, 4) for d in seen)
    assert offsets[-1] < 0.9, "the tail gates were not re-sized"


def test_fractional_tail_is_inert_without_the_fallback(monkeypatch):
    """The sub-flag alone must never touch a candidate the ladder delivered."""
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_FRAC_TAIL", "1")
    seen = []
    real_cs = rf._cluster_seat

    def _cs(opt, out, deadline=None):
        seen.append(deadline)
        return real_cs(opt, out, deadline=deadline)

    monkeypatch.setattr(rf, "_cluster_seat", _cs)
    inst, pred = _case()
    # under 2.5 s: the step-7 recompression rerun (a whole second
    # `refine_prediction` at `t_hard - 0.05`, whose own tail gates would show
    # up here shifted by 0.05) cannot fire, so the offsets are unambiguous
    span = 2.0
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    assert rf.refine_prediction(opt, pred, deadline, seed=11) is not None
    offsets = sorted(round(deadline - d, 4) for d in seen)
    assert offsets[-1] == pytest.approx(0.9, abs=1e-3)
    assert offsets[0] == pytest.approx(0.05, abs=1e-3)
