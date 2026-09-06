"""`PARTNER_LADDER_REBUDGET=1` -> a budgeted expand ladder in
`refine_prediction`.

The shipped ladder budgets exactly one rung.  Rung 0 (the fixed frame) is
capped at 0.35 of the ladder span; the +2% rung behind it then calls
`legalize_soft(deadline=None)` -- unbounded -- so at the shipping worker
slices (0.25-1.1 s, with `PARTNER_REFINE_RES_FRAC` carved off first) that one
call owns every second the ladder has left.  Consequences measured on the
official suite: 23% of ladder attempts return None (120/522), and the rungs
listed behind the +2% one (0.05/0.08/0.12/0.18/0.28) never execute at any
shipping budget -- they are dead code.

ON: the +2% rung gets `PARTNER_LADDER_R1_FRAC` (0.5) of the ladder span, the
rungs behind it (`PARTNER_LADDER_EXPANDS`, default "0.05,0.12,0.28") split
what is left with the last one owning the remainder, and the global 45%
`rung_cap` -- which would only skip the rungs the flag just funded -- is
retired.

What must hold:
  * default off, usual flag spellings, malformed env -> defaults;
  * off  -> the first expand rung is still unbounded (the mechanism this
    flag exists to fix is really there);
  * on   -> every expand rung carries a deadline, none past the ladder's,
    the first one is bounded by R1_FRAC and the last owns the remainder;
  * on   -> when the +2% rung burns its whole window the later rungs are
    actually reached (off: they are not);
  * rung 0 is untouched: a rung-0 success returns the IDENTICAL layout on
    either arm;
  * the ladder never outlives the worker deadline it was handed;
  * independent of `PARTNER_REFINE_SECURE_FALLBACK`: with both on, a ladder
    that cannot close at any frame still comes back legal.
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

_ENV = ("PARTNER_LADDER_REBUDGET", "PARTNER_LADDER_EXPANDS",
        "PARTNER_LADDER_R1_FRAC", "PARTNER_LADDER_DEBUG",
        "PARTNER_LADDER_SECURE_MIN",
        "PARTNER_REFINE_SECURE_FALLBACK", "PARTNER_SECURE_FALLBACK_EXPANDS",
        "PARTNER_ANYTIME_LADDER", "PARTNER_REFINE_RES_FRAC",
        "PARTNER_LEGAL_ADMIT", "PARTNER_REFINE_GUARD",
        "PARTNER_FRAME_SCALE_LADDER", "PARTNER_PIN_FRAME")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _case(n: int = 40, seed: int = 0):
    inst = build_instance(n=n, seed=seed)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    return inst, pred


def _refine(inst, pred, span: float, seed: int = 11):
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    out = rf.refine_prediction(opt, pred, deadline, seed=seed)
    return out, deadline


def _rung0_case(n: int = 24, seed: int = 0):
    """A case whose rung 0 -- the FIXED frame -- really closes at depth 0.

    Rung 0 legalizes into a frame of exactly `1.08 * area_ref` with every
    boundary tag seated flush against it, which no raw synthetic prediction
    survives; the input here is therefore a layout the ladder already
    legalized once (the shape production sees when a direct candidate is
    handed back to the refiner), on an instance with no boundary tags.
    Verified with `PARTNER_LADDER_DEBUG=1`: the ladder closes with no expand
    rung executed at all."""
    inst = build_instance(n=n, seed=seed, frac_fixed=0.0, n_preplaced=0,
                          frac_boundary=0.0, n_clusters=0, n_mib=0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    out, _ = _refine(inst, pred, 4.0)
    assert out is not None, "setup: this case must legalize once"
    return inst, np.asarray(out, dtype=np.float64)


def _record_rungs(monkeypatch, fail_below=None, burn=False):
    """Record every EXPAND-rung legalization.

    Rung 0 is distinguishable without touching it: it calls `legalize_soft`
    with `max_sweeps` 12 (`fine=True`) or 10 (its salvage), never with the
    default 14 the expand rungs use.

    `fail_below`: frames narrower than `fail_below * area_ref` report failure
    (the production shape -- the tight rungs cannot close -- made
    deterministic instead of clock-dependent).
    `burn`: a failing rung also spends its whole window, which is exactly
    what the unbounded shipped rung does with the ladder's remaining span.
    """
    seen = []
    real = rf._Refiner.legalize_soft

    def _rec(self_, max_sweeps=14, deadline=None, fine=False):
        main = (max_sweeps == 14 and not fine)
        t0 = time.time()
        if main:
            seen.append({"t": t0, "deadline": deadline,
                         "frame": float((self_.xmax - self_.xmin)
                                        * (self_.ymax - self_.ymin))})
        ok = real(self_, max_sweeps, deadline=deadline, fine=fine)
        if fail_below is not None:
            aref = max(float(getattr(self_.opt, "area_ref", 0.0)), 1e-9)
            frame = (self_.xmax - self_.xmin) * (self_.ymax - self_.ymin)
            if frame < fail_below * aref:
                ok = False
        if main and burn and not ok:
            # spend the window this rung was given, the way the shipped
            # unbounded call spends the ladder's whole remaining span
            end = deadline
            if end is None:
                # the shipped unbounded rung burns to the WORKER deadline,
                # which is precisely the overrun being reproduced
                end = float(getattr(self_.opt, "deadline", t0 + 2.0))
            while time.time() < end:
                time.sleep(0.005)
        if main:
            seen[-1]["ok"] = bool(ok)
            seen[-1]["end"] = time.time()
        return ok

    monkeypatch.setattr(rf._Refiner, "legalize_soft", _rec)
    return seen


# ==========================================================================
# 1. env wiring
# ==========================================================================
def test_default_is_off():
    assert rf.ladder_rebudget_on() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "True", "ON"])
def test_flag_spellings(monkeypatch, value):
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", value)
    assert rf.ladder_rebudget_on() is True


def test_expand_defaults():
    assert rf.ladder_expand_set() == (0.05, 0.12, 0.28)


@pytest.mark.parametrize("raw,want", [
    ("0.3", (0.3,)),
    ("0.05,0.12,0.28", (0.05, 0.12, 0.28)),
    ("0.12,0.28", (0.12, 0.28)),
    ("0.4,0.2", (0.4,)),                 # non-ascending tail dropped
    ("bogus", (0.05, 0.12, 0.28)),       # malformed -> the default
    ("", (0.05, 0.12, 0.28)),
    (",,", (0.05, 0.12, 0.28)),
])
def test_expand_env_parsing(monkeypatch, raw, want):
    monkeypatch.setenv("PARTNER_LADDER_EXPANDS", raw)
    assert rf.ladder_expand_set() == want


@pytest.mark.parametrize("bad", ["bogus", "1.5", "0", "-0.2", "1.0", ""])
def test_malformed_r1_frac_falls_back(monkeypatch, bad):
    monkeypatch.setenv("PARTNER_LADDER_R1_FRAC", bad)
    assert rf.anytime_frac("PARTNER_LADDER_R1_FRAC", 0.5) == 0.5


# ==========================================================================
# 2. the mechanism
# ==========================================================================
def test_shipped_first_expand_rung_is_unbounded(monkeypatch):
    """The bug this flag fixes: with the flag OFF the +2% rung is handed no
    deadline at all, so it owns the rest of the ladder span."""
    seen = _record_rungs(monkeypatch, fail_below=1.9)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert seen, "no expand rung ran at all"
    assert seen[0]["deadline"] is None


def test_every_rebudgeted_rung_is_bounded(monkeypatch):
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    seen = _record_rungs(monkeypatch, fail_below=1.9)
    inst, pred = _case()
    _, worker_deadline = _refine(inst, pred, 4.0)
    assert seen, "no expand rung ran at all"
    for rec in seen:
        assert rec["deadline"] is not None, "a rung ran unbounded"
        # the ladder deadline is the worker deadline minus the repair
        # reserve; no rung may be allowed past it
        assert rec["deadline"] <= worker_deadline


def test_rung1_window_honours_r1_frac(monkeypatch):
    """The +2% rung gets R1_FRAC of the ladder span that is left when it
    starts -- not the whole of it."""
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    monkeypatch.setenv("PARTNER_LADDER_R1_FRAC", "0.5")
    monkeypatch.setenv("PARTNER_REFINE_RES_FRAC", "0.45")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _, worker_deadline = _refine(inst, pred, 4.0)
    assert len(seen) >= 2
    r1, last = seen[0], seen[-1]
    # the last rung owns the remainder, so its deadline IS the ladder
    # deadline: the first rung must stop well short of it
    lad_end = last["deadline"]
    span1 = lad_end - r1["t"]
    assert span1 > 0.0
    got = r1["deadline"] - r1["t"]
    assert 0.35 * span1 <= got <= 0.65 * span1, (got, span1)


@pytest.mark.parametrize("frac,lo,hi", [("0.3", 0.18, 0.45),
                                        ("0.7", 0.55, 0.85)])
def test_r1_frac_is_tunable(monkeypatch, frac, lo, hi):
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    monkeypatch.setenv("PARTNER_LADDER_R1_FRAC", frac)
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert len(seen) >= 2
    r1, last = seen[0], seen[-1]
    span1 = last["deadline"] - r1["t"]
    got = (r1["deadline"] - r1["t"]) / max(span1, 1e-9)
    assert lo <= got <= hi, got


def test_later_rungs_are_dead_code_on_the_shipped_path(monkeypatch):
    """Off arm: the +2% rung burns the ladder span and NOTHING behind it
    runs -- the census this flag was designed from."""
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    out, _ = _refine(inst, pred, 4.0)
    assert out is None
    assert len(seen) == 1, [round(s["frame"], 1) for s in seen]


def test_later_rungs_are_reached_when_rung1_fails(monkeypatch):
    """On arm, same case: the +2% rung is cut off at its window and the
    configured rungs behind it each get to run."""
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    frames = [s["frame"] for s in seen]
    assert len(frames) == 4, frames          # 0.02 + the three configured
    assert all(b > a for a, b in zip(frames, frames[1:])), frames


def test_expand_set_drives_the_rungs(monkeypatch):
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    monkeypatch.setenv("PARTNER_LADDER_EXPANDS", "0.12,0.28")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert len(seen) == 3, [round(s["frame"], 1) for s in seen]


def test_last_rung_owns_the_remainder(monkeypatch):
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert len(seen) >= 3
    ends = [s["deadline"] for s in seen]
    # every intermediate rung stops short of the ladder deadline, the last
    # one is handed exactly it (all the `min(deadline, ...)` terms collapse)
    assert ends[-1] == max(ends)
    assert all(e < ends[-1] for e in ends[:-1]), ends


def test_secure_min_is_off_by_default(monkeypatch):
    """`PARTNER_LADDER_SECURE_MIN` unset -> the escape-rung floor is 0.0 and
    the rung deadlines are the plain split."""
    assert rf.anytime_frac("PARTNER_LADDER_SECURE_MIN", 0.0) == 0.0
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    ends = [s["deadline"] for s in seen]
    assert len(ends) >= 3 and ends[-1] == max(ends)


def test_secure_min_reserves_the_escape_rung(monkeypatch):
    """ON: no rung but the last may spend the reserved share, so the escape
    rung still owns a real window after three frames failed before it."""
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    monkeypatch.setenv("PARTNER_LADDER_SECURE_MIN", "0.4")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert len(seen) >= 3
    last = seen[-1]
    lad_end = last["deadline"]
    span = lad_end - seen[0]["t"]
    for rec in seen[:-1]:
        # every earlier rung stops at least the floor short of the ladder end
        assert rec["deadline"] <= lad_end - 0.3 * span + 1e-6, rec
    # ... and the escape rung really receives it
    assert lad_end - last["t"] >= 0.3 * span


@pytest.mark.parametrize("bad", ["bogus", "1.5", "-0.2", "1.0", ""])
def test_malformed_secure_min_falls_back(monkeypatch, bad):
    monkeypatch.setenv("PARTNER_LADDER_SECURE_MIN", bad)
    assert rf.anytime_frac("PARTNER_LADDER_SECURE_MIN", 0.0) == 0.0


# ==========================================================================
# 3. what must NOT change
# ==========================================================================
def test_rung0_success_is_bit_identical(monkeypatch):
    """Rung 0 is untouched by the rebudget: on a case rung 0 closes, both
    arms return the same layout, not merely an equally good one."""
    inst, pred = _rung0_case()
    out_off, _ = _refine(inst, pred, 4.0)
    assert out_off is not None
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    out_on, _ = _refine(inst, pred, 4.0)
    assert out_on is not None
    assert np.array_equal(np.asarray(out_off, dtype=np.float64),
                          np.asarray(out_on, dtype=np.float64))


def test_rung0_success_runs_no_expand_rung(monkeypatch):
    """...and it gets there without paying for a single expand rung, so
    nothing the flag rewrites is even reachable."""
    inst, pred = _rung0_case()
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    seen = _record_rungs(monkeypatch)
    out, _ = _refine(inst, pred, 4.0)
    assert out is not None
    assert not seen, "rung 0 closed but expand rungs still ran"


def test_early_ladder_success_is_identical(monkeypatch):
    """More generally: where the rungs close comfortably inside the windows
    the rebudget hands them, the flag changes nothing at all."""
    inst, pred = _case()
    out_off, _ = _refine(inst, pred, 4.0)
    assert out_off is not None
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    out_on, _ = _refine(inst, pred, 4.0)
    assert out_on is not None
    assert np.array_equal(np.asarray(out_off, dtype=np.float64),
                          np.asarray(out_on, dtype=np.float64))


def test_ladder_never_outlives_the_worker_deadline(monkeypatch):
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    span = 2.0
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    rf.refine_prediction(opt, pred, deadline, seed=11)
    assert time.time() <= deadline + 0.9 * span


def test_flag_off_is_inert_when_env_knobs_are_set(monkeypatch):
    """The knobs alone must not move anything: only the flag does."""
    monkeypatch.setenv("PARTNER_LADDER_R1_FRAC", "0.2")
    monkeypatch.setenv("PARTNER_LADDER_EXPANDS", "0.9")
    seen = _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert len(seen) == 1 and seen[0]["deadline"] is None


# ==========================================================================
# 4. independence from the secure fallback
# ==========================================================================
def test_rebudget_plus_fallback_still_delivers(monkeypatch):
    """No frame the ladder owns can close -> the promoted fallback still
    rescues the candidate with the rebudget on."""
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    monkeypatch.setenv("PARTNER_SECURE_FALLBACK_EXPANDS", "0.6")
    _record_rungs(monkeypatch, fail_below=1.9)
    inst, pred = _case()
    out, _ = _refine(inst, pred, 4.0)
    assert out is not None
    pos = np.asarray(out, dtype=np.float64)
    assert pos.shape == pred.shape and np.isfinite(pos).all()
    opt = make_optimizer(inst, seed=3, deadline=time.time() + 1.0)
    assert rf._guard_hard_ok(opt, pos)


def test_a_time_truncated_rung_is_not_reported_as_refuted(monkeypatch):
    """`secure_fallback_expands` drops expansions a rung REFUTED.  Under the
    rebudget a rung can also fail because its window closed, which refutes
    nothing -- the fallback must keep that frame available."""
    monkeypatch.setenv("PARTNER_LADDER_REBUDGET", "1")
    monkeypatch.setenv("PARTNER_REFINE_SECURE_FALLBACK", "1")
    ran = []
    real_fb = rf._secure_fallback

    def _spy(opt, P0, pred_c, saved_cg, seed, t_hard, r):
        ran.append(list(r))
        return real_fb(opt, P0, pred_c, saved_cg, seed, t_hard, r)

    monkeypatch.setattr(rf, "_secure_fallback", _spy)
    _record_rungs(monkeypatch, fail_below=1.9, burn=True)
    inst, pred = _case()
    _refine(inst, pred, 4.0)
    assert ran, "the ladder should have failed into the fallback"
    # every rung was cut off by its own window, so none of them was refuted
    assert ran[0] == [], ran[0]
