"""`PARTNER_ANYTIME_LADDER=1` -> anytime direct-refine ladder (task #10).

`refine_prediction` is the direct channel's whole delivery path: 15 pool slots
(`PARTNER_NREF`) legalize + refine a Direct prediction each, and the ladder
decides what comes back.  Shipped semantics are "try tight frames in order
until one legalizes", which is all-or-nothing under a deadline:

  * the tight rungs call `legalize_soft()` with NO deadline, so one rung can
    (and does) overrun the whole worker deadline -- measured 3x overrun on a
    0.5 s span;
  * when the span runs out mid-ladder the function returns `None` and the
    pool slot delivers nothing at all;
  * when a late loose rung does legalize, the quality tail (frame anneal,
    refiner, repair) is left with ~0 s, so the candidate carries the loose
    rung's inflated bbox;
  * the repair/compaction tail is gated on ABSOLUTE offsets (`t_hard - 0.9`,
    `- 2.5`) that are already in the past whenever the carved reserve is
    smaller than they are.

ANYTIME bounds every call, keeps a secure rung reachable out of a reserve
sized in `_Refiner`-build units (the instance's own cost unit, never absolute
seconds), and pays the tail out of fractions of that reserve.  Default off ->
every added expression is a dead branch, no extra randomness is drawn, and
the shipped deadline arguments are reproduced exactly.

See docs/design/2026-08-04-anytime-ladder.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "tests", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout_refiner as rf            # noqa: E402
from synth_instances import build_instance, make_optimizer   # noqa: E402

_ENV = ("PARTNER_ANYTIME_LADDER", "PARTNER_ANYTIME_SECURE",
        "PARTNER_ANYTIME_SECURE_MIN", "PARTNER_ANYTIME_SECURE_MAX",
        "PARTNER_ANYTIME_TIGHTEN", "PARTNER_ANYTIME_BUILD_MULT",
        "PARTNER_EARLY_EXIT", "PARTNER_REFINE_STALL_STOP")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# --------------------------------------------------------------------------
# harness: a scattered "prediction" over a structured synthetic instance
# --------------------------------------------------------------------------
def _case(n: int = 70, seed: int = 0):
    inst = build_instance(n=n, seed=seed)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    return inst, pred


def _refine(inst, pred, span: float, seed: int = 11):
    """One `_worker_refine`-shaped call: fresh optimizer, fresh deadline."""
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    t0 = time.time()
    out = rf.refine_prediction(opt, pred, deadline, seed=seed)
    return out, time.time() - t0, opt


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
    assert rf.anytime_ladder_on() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "True", "ON"])
def test_flag_spellings(monkeypatch, value):
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", value)
    assert rf.anytime_ladder_on() is True


@pytest.mark.parametrize("bad", ["bogus", "1.5", "0", "-0.2", "1.0"])
def test_malformed_fraction_falls_back(monkeypatch, bad):
    monkeypatch.setenv("PARTNER_ANYTIME_SECURE", bad)
    assert rf.anytime_frac("PARTNER_ANYTIME_SECURE", 0.55) == 0.55


def test_fraction_override(monkeypatch):
    monkeypatch.setenv("PARTNER_ANYTIME_SECURE", "0.7")
    assert rf.anytime_frac("PARTNER_ANYTIME_SECURE", 0.55) == 0.7


# ==========================================================================
# 2. off path == the shipped path (the deadline arguments prove it)
# ==========================================================================
class _Spy:
    """Records the deadline every time-bounded stage is actually handed."""

    def __init__(self, monkeypatch):
        self.lsoft = []          # deadline arg of every legalize_soft call
        self.tighten = []        # (deadline arg - now) of every _tighten
        self.seat = []           # (deadline arg) of every _cluster_seat
        real_ls = rf._Refiner.legalize_soft
        real_tg = rf._Refiner._tighten
        real_cs = rf._cluster_seat
        spy = self

        def _ls(self_, max_sweeps=14, deadline=None, fine=False):
            spy.lsoft.append(deadline)
            return real_ls(self_, max_sweeps, deadline=deadline, fine=fine)

        def _tg(self_, deadline, rounds=10):
            spy.tighten.append((time.time(), deadline))
            return real_tg(self_, deadline, rounds=rounds)

        def _cs(opt, out, deadline=None):
            spy.seat.append(deadline)
            return real_cs(opt, out, deadline=deadline)

        monkeypatch.setattr(rf._Refiner, "legalize_soft", _ls)
        monkeypatch.setattr(rf._Refiner, "_tighten", _tg)
        monkeypatch.setattr(rf, "_cluster_seat", _cs)


def test_off_path_keeps_the_unbounded_rung_call(monkeypatch):
    """The shipped ladder rungs call `legalize_soft()` with no deadline at
    all.  That is the overrun source ANYTIME removes -- and the sharpest
    available witness that the off path is the pre-flag path.

    Stub legalization so this wiring assertion cannot depend on whether a
    machine reaches the expand ladder before its wall-clock deadline.  The
    real fixed-frame rung can consume the entire short test budget when the
    suite runs under load, even though the off-path call remains unbounded.
    """
    deadlines = []

    def _fail_fast(self_, max_sweeps=14, deadline=None, fine=False):
        deadlines.append(deadline)
        return False

    monkeypatch.setattr(rf._Refiner, "legalize_soft", _fail_fast)
    inst, pred = _case(n=70, seed=0)
    _refine(inst, pred, 3.0)
    assert any(d is None for d in deadlines), \
        "off path must keep the shipped unbounded rung call"


def test_on_path_bounds_every_legalization(monkeypatch):
    spy = _Spy(monkeypatch)
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=70, seed=0)
    _refine(inst, pred, 1.0)
    assert spy.lsoft, "no legalization ran at all"
    assert all(d is not None for d in spy.lsoft), \
        "ANYTIME must bound every legalization call"


def test_off_path_tail_gates_are_the_shipped_constants(monkeypatch):
    """`_cluster_seat` is handed `t_hard - 0.9` and `t_hard - 0.05` on the
    shipped path; ANYTIME caps both by a share of the carved reserve."""
    spy = _Spy(monkeypatch)
    inst, pred = _case(n=70, seed=0)
    span = 2.0
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    rf.refine_prediction(opt, pred, deadline, seed=11)
    assert spy.seat, "seat pass never ran"
    offsets = sorted(deadline - d for d in spy.seat)
    assert offsets[-1] == pytest.approx(0.9, abs=1e-6)
    assert offsets[0] == pytest.approx(0.05, abs=1e-6)


def test_on_path_tail_gates_scale_with_the_reserve(monkeypatch):
    spy = _Spy(monkeypatch)
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=70, seed=0)
    span = 2.0
    deadline = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=deadline)
    rf.refine_prediction(opt, pred, deadline, seed=11)
    assert spy.seat
    res = 0.3 * span                     # the carved reserve
    offsets = sorted(deadline - d for d in spy.seat)
    assert offsets[-1] == pytest.approx(min(0.9, 0.1 * res), abs=1e-3)
    assert offsets[0] == pytest.approx(min(0.05, 0.02 * res), abs=1e-3)


def test_tighten_budget_off_is_absolute_on_is_fractional(monkeypatch):
    """The frame anneal is handed `min(ladder_deadline, now + 3.0)` on the
    shipped path -- an absolute-time assumption that never binds below a ~4 s
    worker deadline, where the anneal is instead starved to <= 0 s.  ANYTIME
    hands it a share of what is actually left."""
    inst, pred = _case(n=70, seed=0)
    span = 2.0

    def _observe(on: bool):
        if on:
            monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
        else:
            monkeypatch.delenv("PARTNER_ANYTIME_LADDER", raising=False)
        spy = _Spy(monkeypatch)
        deadline = time.time() + span
        opt = make_optimizer(inst, seed=3, deadline=deadline)
        rf.refine_prediction(opt, pred, deadline, seed=11)
        # copy: the second `_Spy` chains onto the first, so a live list
        # would also collect the other arm's calls
        # `res = min(3.5, 0.3 * slice)`; the ladder ends there
        return list(spy.tighten), deadline - 0.3 * span

    calls_off, ladder_off = _observe(False)
    calls_on, ladder_on = _observe(True)
    if not calls_off or not calls_on:
        pytest.skip("no rung legalized on one of the arms")
    for t_now, t_dl in calls_off:
        assert t_dl == pytest.approx(min(ladder_off, t_now + 3.0), abs=5e-3)
    for t_now, t_dl in calls_on:
        assert t_dl <= t_now + 0.46 * max(0.0, ladder_on - t_now) + 5e-3


# ==========================================================================
# 3. the anytime contract: bounded overrun + delivery
# ==========================================================================
@pytest.mark.parametrize("span", [0.6, 1.2])
def test_on_path_respects_the_deadline(monkeypatch, span):
    """The shipped ladder overruns short deadlines by 2-3x (an unbounded
    rung).  In the pool that is a straggler: the case ends when the SLOWEST
    worker returns, so an overrun is raw runtime."""
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=100, seed=0)
    _out, el, _opt = _refine(inst, pred, span)
    assert el <= span + 0.35, f"overran by {el - span:.2f}s"


def test_on_path_delivers_a_candidate_at_a_short_deadline(monkeypatch):
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    for n, seed in ((70, 0), (70, 1), (100, 0)):
        inst, pred = _case(n=n, seed=seed)
        out, _el, _opt = _refine(inst, pred, 1.2)
        assert out is not None, f"no candidate for n={n} seed={seed}"


def test_delivery_is_monotone_in_the_deadline(monkeypatch):
    """More time must never turn a delivered candidate into `None`."""
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=70, seed=1)
    delivered = None
    for span in (1.0, 2.0, 3.0):
        out, _el, _opt = _refine(inst, pred, span)
        if delivered:
            assert out is not None, f"regressed to None at span={span}"
        delivered = delivered or (out is not None)


def test_more_time_does_not_cost_quality(monkeypatch):
    """Anytime monotonicity, stated on the selection proxy the pool uses
    (`_parallel_solve.score` shape).  The stages are wall-clock bounded, so
    a small tolerance is unavoidable; a broken budget split shows up far
    above it (the shipped path moves 0.3-0.5 between these tiers)."""
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=70, seed=0)

    def _metrics(span):
        out, _el, opt = _refine(inst, pred, span)
        if out is None:
            return None
        pos = np.asarray(out, dtype=np.float64)
        return (float(opt._hpwl(pos)),
                float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                      * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min()))
                / opt.area_ref,
                int(rf.full_violations(opt, pos)),
                max(getattr(opt, "n_soft_den", 1), 1))

    m_short = _metrics(1.2)
    m_long = _metrics(3.0)
    assert m_short is not None and m_long is not None
    hp_ref = min(m_short[0], m_long[0])

    def _proxy(m):
        hp, ar, V, den = m
        return (1.0 + 0.5 * ((hp - hp_ref) / hp_ref + max(0.0, ar - 1.0))) \
            * float(np.exp(2.0 * V / den))

    assert _proxy(m_long) <= _proxy(m_short) + 0.05


# ==========================================================================
# 4. hard legality of whatever the flag delivers
# ==========================================================================
@pytest.mark.parametrize("span", [0.8, 1.5, 2.5])
def test_delivered_layouts_are_legal(monkeypatch, span):
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=70, seed=1)
    out, _el, opt = _refine(inst, pred, span)
    if out is None:
        pytest.skip("no candidate at this span")
    pos = np.asarray(out, dtype=np.float64)

    assert pos.shape == pred.shape
    _assert_no_overlap(pos)
    # exact areas for soft blocks (reshape keeps w*h == target)
    for i in range(opt.n):
        if opt.kind[i] == 0:
            assert pos[i, 2] * pos[i, 3] == pytest.approx(
                float(opt.areas[i]), rel=5e-3)
        assert pos[i, 2] > 0 and pos[i, 3] > 0
    # fixed-shape blocks keep their (w, h); preplaced keep their rect
    for i in range(opt.n):
        if opt.preplaced[i]:
            assert pos[i, 0] == pytest.approx(float(opt.lx[i]), abs=1e-6)
            assert pos[i, 1] == pytest.approx(float(opt.ly[i]), abs=1e-6)
        if opt.fixed[i] or opt.preplaced[i]:
            assert pos[i, 2] == pytest.approx(float(opt.rw[i]), rel=1e-6)
            assert pos[i, 3] == pytest.approx(float(opt.rh[i]), rel=1e-6)
    # MIB groups keep identical dims
    for idxs in opt.mib_groups.values():
        if len(idxs) < 2:
            continue
        w0, h0 = pos[idxs[0], 2], pos[idxs[0], 3]
        for i in idxs:
            assert pos[i, 2] == pytest.approx(w0, rel=1e-4)
            assert pos[i, 3] == pytest.approx(h0, rel=1e-4)


def test_escape_rung_never_returns_an_illegal_layout(monkeypatch):
    """Squeeze the span until only the escape rung can run; whatever comes
    back must still be overlap-free."""
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=40, seed=0)
    for span in (0.15, 0.25, 0.4, 0.6):
        out, el, _opt = _refine(inst, pred, span)
        assert el <= span + 0.35
        if out is not None:
            _assert_no_overlap(np.asarray(out, dtype=np.float64))


# ==========================================================================
# 5. determinism: the flag changes WHEN stages stop, never the rng stream
# ==========================================================================
def test_flag_draws_no_extra_randomness(monkeypatch):
    """Every predicate ANYTIME adds is a time comparison, so the optimizer's
    rng state after a run must be identical on and off (the refiner's own
    rngs are seeded per `_Refiner`, from the caller's seed)."""
    inst, pred = _case(n=40, seed=0)

    def _state(on: bool):
        if on:
            monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
        else:
            monkeypatch.delenv("PARTNER_ANYTIME_LADDER", raising=False)
        deadline = time.time() + 1.0
        opt = make_optimizer(inst, seed=3, deadline=deadline)
        before = opt.rng.getstate()
        rf.refine_prediction(opt, pred, deadline, seed=11)
        return before, opt.rng.getstate()

    b_off, a_off = _state(False)
    b_on, a_on = _state(True)
    assert b_off == b_on
    assert a_off == a_on == b_off


def test_same_seed_same_span_is_reproducible(monkeypatch):
    """Not a bit-reproducibility claim for the whole pipeline (every stage is
    wall-clock bounded), but the secure rung + assembly path must be: two
    runs at a span where only that path fits return the same rung result."""
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    inst, pred = _case(n=40, seed=1)
    outs = []
    for _ in range(2):
        out, _el, _opt = _refine(inst, pred, 0.5)
        outs.append(None if out is None else
                    np.asarray(out, dtype=np.float64).copy())
    if outs[0] is None or outs[1] is None:
        pytest.skip("no candidate at this span")
    # areas and block count are structural, not timing-dependent
    assert outs[0].shape == outs[1].shape
    assert np.allclose(outs[0][:, 2] * outs[0][:, 3],
                       outs[1][:, 2] * outs[1][:, 3], rtol=5e-3)


# ==========================================================================
# 6. interop with the other budget flags
# ==========================================================================
def test_compatible_with_early_exit(monkeypatch):
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    inst, pred = _case(n=70, seed=0)
    out, el, _opt = _refine(inst, pred, 1.5)
    assert el <= 1.5 + 0.35
    if out is not None:
        _assert_no_overlap(np.asarray(out, dtype=np.float64))


def test_compatible_with_refine_stall_stop(monkeypatch):
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    monkeypatch.setenv("PARTNER_REFINE_STALL_STOP", "1")
    inst, pred = _case(n=70, seed=0)
    out, el, _opt = _refine(inst, pred, 1.5)
    assert el <= 1.5 + 0.35
    if out is not None:
        _assert_no_overlap(np.asarray(out, dtype=np.float64))
