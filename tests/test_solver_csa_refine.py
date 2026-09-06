"""`PARTNER_CSA_REFINE=1` -> joint conjugate-subgradient coordinate polish.

Surviving descendant of the CSF gate (docs/design/
2026-07-29-csf-analytical-prototype.md sec. 7).  `_Refiner._axis_pass` is a
coordinate descent over the separation DAG: each group takes its own 1-D
weighted-median optimum and never pays for the chain it drags, so it can sit
at a point only a joint move can improve.  `_csa_pass` runs a projected
Polak-Ribiere subgradient sweep over the whole per-group shift vector on the
SAME polytope (`_axis_constraints`, now shared with `_axis_pass`).

What must hold:
  * flag off  -> csa_share == 0.0, the `run` hook is a dead branch, no pass;
  * flag on   -> shapes untouched, zero overlap, pinned/preplaced/tag-
                 satisfied groups frozen, and the pass is reverted unless
                 the full `_key()` proxy strictly improves;
  * the solver itself is monotone on the axis objective (`best_f <= f0`) and
    `ShiftPolytope.project` is feasible AND identity on feasible points.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as lg          # noqa: E402
import csa_coordinate_solver as csa       # noqa: E402
import layout_refiner as rf               # noqa: E402

_ENV = ("PARTNER_CSA_REFINE", "PARTNER_CSA_SHARE", "PARTNER_CSA_ITERS",
        "PARTNER_CSA_STEP", "PARTNER_CSA_DECAY", "PARTNER_CSA_MS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _make_opt(n, b2b, p2b, pins, cons, tpos):
    import torch
    rects = [(0.0, 0.0, 1.0, 1.0)] * n
    at = torch.ones(n)
    return lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               time.time() + 5.0, seed=7)


def _drag_case(tag0=0):
    """The coordinate-descent blind spot, minimal.

    Blocks 0..5 are unit squares packed solid in a row on [0, 6] x [0, 1].
    Blocks 6 and 7 are PREPLACED at [10, 11] and [0, 1] x [3, 4]: constants
    that pin the bounding box to [0, 11] x [0, 4] (so the proxy's area term
    cannot move) and leave the row five units of x slack.

      * block 0 is wired to block 6 with weight 3, so it wants to go right;
      * blocks 1..5 are each pinned to their OWN current center with weight
        1, so they want to stay put.

    A rigid drag of the whole row costs 5 and buys 3, so the joint optimum
    is "do not move".  `_axis_pass` nevertheless moves it: block 0 is first
    in topological order, takes its own median optimum (+10, clipped to the
    frame at +5) and the difference constraints drag 1..5 along.  From the
    dragged state the sweep is STUCK -- block 0, still processed first,
    refuses to come back -- while the joint optimum is to return.  That is
    precisely the residual CSA exists to collect.

    `tag0` optionally puts a boundary bit on block 0 (1 = left wall, which
    the initial layout already satisfies) so the group is pinned on x.
    """
    import torch
    n = 8
    P = np.array([[float(i), 0.0, 1.0, 1.0] for i in range(6)]
                 + [[10.0, 3.0, 1.0, 1.0], [0.0, 3.0, 1.0, 1.0]],
                 dtype=np.float64)
    b2b = torch.tensor([[0.0, 6.0, 3.0]])                 # (i, j, weight)
    p2b = torch.tensor([[float(k), float(k + 1), 1.0]      # (pin, block, w)
                        for k in range(5)])
    pins = torch.tensor([[float(k + 1) + 0.5, 0.5] for k in range(5)])
    cons = torch.zeros((n, 5))
    cons[6, 1] = 1.0                                       # preplaced
    cons[7, 1] = 1.0                                       # preplaced
    cons[0, 4] = float(tag0)                               # boundary bitmask
    tpos = torch.full((n, 4), -1.0)
    tpos[6] = torch.tensor([10.0, 3.0, 1.0, 1.0])
    tpos[7] = torch.tensor([0.0, 3.0, 1.0, 1.0])
    opt = _make_opt(n, b2b, p2b, pins, cons, tpos)
    return opt, P


def _drag_refiner():
    opt, P = _drag_case()
    return rf._Refiner(opt, P, seed=3)


def _frozen_refiner():
    opt, P = _drag_case(tag0=1)
    return rf._Refiner(opt, P, seed=3)


# --------------------------------------------------------------------------
# 1. env wiring
# --------------------------------------------------------------------------
def test_default_is_off():
    r = _drag_refiner()
    assert r.csa_share == 0.0
    assert r.csa_iters == 60
    assert r.csa_step == 0.05
    assert r.csa_decay == 0.95
    assert r.csa_ms == 15.0
    assert r.csa_where == "stall"
    assert r._csa_calls == 0


def test_flag_and_overrides(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.2")
    monkeypatch.setenv("PARTNER_CSA_ITERS", "25")
    monkeypatch.setenv("PARTNER_CSA_STEP", "0.05")
    monkeypatch.setenv("PARTNER_CSA_DECAY", "0.8")
    monkeypatch.setenv("PARTNER_CSA_MS", "40")
    monkeypatch.setenv("PARTNER_CSA_WHERE", "end")
    r = _drag_refiner()
    assert r.csa_where == "end"
    assert (r.csa_share, r.csa_iters, r.csa_step, r.csa_decay, r.csa_ms) \
        == (0.2, 25, 0.05, 0.8, 40.0)


def test_malformed_and_out_of_range_fall_back(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "on")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "3.0")     # > 1
    monkeypatch.setenv("PARTNER_CSA_ITERS", "bogus")
    monkeypatch.setenv("PARTNER_CSA_STEP", "0")        # not > 0
    monkeypatch.setenv("PARTNER_CSA_DECAY", "-1")
    monkeypatch.setenv("PARTNER_CSA_WHERE", "nowhere")
    r = _drag_refiner()
    assert r.csa_where == "stall"
    assert r.csa_share == 0.08
    assert r.csa_iters == 60
    assert r.csa_step == 0.05
    assert r.csa_decay == 0.95


# --------------------------------------------------------------------------
# 2. off => the run hook never fires
# --------------------------------------------------------------------------
def test_off_never_calls_the_pass():
    r = _drag_refiner()
    r.run(time.time() + 0.35)
    assert r._csa_calls == 0
    assert r._csa_spent == 0.0


def test_on_calls_the_pass(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    r = _drag_refiner()
    r.run(time.time() + 0.5)
    assert r._csa_calls > 0


def test_axis_constraints_is_shared_and_well_formed():
    r = _drag_refiner()
    cons = r._axis_constraints(0)
    assert cons.G == len(r.groups)
    assert len(cons.edges_in) == cons.G
    assert len(cons.edges_out) == cons.G
    assert cons.pinned.shape == (cons.G,)
    assert len(cons.order) == cons.G


# --------------------------------------------------------------------------
# 3. hard legality of an applied pass
# --------------------------------------------------------------------------
def test_pass_preserves_shapes_frozen_blocks_and_zero_overlap(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    monkeypatch.setenv("PARTNER_CSA_ITERS", "80")
    r = _drag_refiner()
    before = r.P.copy()
    pinned = r._axis_constraints(0).pinned
    frozen = [int(i) for gi, g in enumerate(r.groups) if pinned[gi]
              for i in g.members]
    frozen += [i for i in range(r.n) if r.kind[i] == 2]
    applied = False
    for _ in range(6):
        applied |= r._csa_pass(0)
    assert not r._has_overlap()
    # w, h bit-identical -> exact area preserved
    assert np.array_equal(r.P[:, 2:], before[:, 2:])
    for i in frozen:
        assert r.P[i, 0] == before[i, 0]
    assert r.P[:, 0].min() >= r.xmin - 1e-9
    assert (r.P[:, 0] + r.P[:, 2]).max() <= r.xmax + 1e-9
    assert applied or np.array_equal(r.P, before)


def test_pass_is_monotone_on_the_proxy(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    r = _drag_refiner()
    for axis in (0, 1, 0, 1):
        k0 = r._key()
        hp0 = r.opt._hpwl(r.P)
        kept = r._csa_pass(axis)
        assert r._key() <= k0 + 1e-12
        if kept:
            assert r._key() < k0
        else:
            assert r.opt._hpwl(r.P) == hp0


def test_run_output_is_legal_with_the_flag_on(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    r = _drag_refiner()
    w0 = r.P[:, 2:].copy()
    out = r.run(time.time() + 0.6)
    r.P[...] = out
    assert not r._has_overlap()
    assert np.array_equal(out[:, 2:], w0)
    assert r.opt._hpwl(out) <= r.hp0 + 1e-9


def test_time_cap_is_respected(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    monkeypatch.setenv("PARTNER_CSA_ITERS", "1000000")
    monkeypatch.setenv("PARTNER_CSA_MS", "12")
    r = _drag_refiner()
    t0 = time.time()
    r._csa_pass(0)
    assert time.time() - t0 < 0.5     # 12 ms cap + gate overhead


def test_exceptions_are_contained(monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    r = _drag_refiner()
    before = r.P.copy()

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(csa, "csa_shifts", _boom)
    assert r._csa_pass(0) is False
    assert np.array_equal(r.P, before)


# --------------------------------------------------------------------------
# 4. the solver itself
# --------------------------------------------------------------------------
def test_objective_matches_hpwl_axis_half():
    r = _drag_refiner()
    cons = r._axis_constraints(0)
    obj, _poly = r._csa_problem(0, cons)
    f0 = obj.value(np.zeros(cons.G))
    step = 0.25
    d = np.where(cons.pinned, 0.0, step)
    P2 = r.P.copy()
    for gi, g in enumerate(r.groups):
        if not cons.pinned[gi]:
            P2[g.members, 0] += step
    delta_true = r.opt._hpwl(P2) - r.opt._hpwl(r.P)
    assert obj.value(d) - f0 == pytest.approx(delta_true, abs=1e-9)


def test_subgradient_matches_finite_differences():
    r = _drag_refiner()
    cons = r._axis_constraints(0)
    obj, _poly = r._csa_problem(0, cons)
    rng = np.random.default_rng(0)
    d = rng.normal(scale=0.3, size=cons.G)      # generic point, no L1 ties
    g = obj.subgrad(d)
    eps = 1e-6
    for k in range(cons.G):
        e = np.zeros(cons.G)
        e[k] = eps
        fd = (obj.value(d + e) - obj.value(d - e)) / (2 * eps)
        assert g[k] == pytest.approx(fd, abs=1e-6)


def test_projection_is_feasible_and_identity_on_feasible_points():
    r = _drag_refiner()
    cons = r._axis_constraints(0)
    _obj, poly = r._csa_problem(0, cons)
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = rng.normal(scale=3.0, size=cons.G)
        d = poly.project(t)
        assert np.all(d[cons.pinned] == 0.0)
        for gj in range(cons.G):
            for gi, c in cons.edges_in[gj]:
                assert d[gj] - d[gi] >= c - 1e-9
        assert np.allclose(poly.project(d), d, atol=1e-12)


def test_csa_shifts_never_returns_a_worse_point():
    r = _drag_refiner()
    cons = r._axis_constraints(0)
    obj, poly = r._csa_problem(0, cons)
    span = r.xmax - r.xmin
    d, f, f0, it = csa.csa_shifts(obj, poly, 0.05 * span, 60, 0.95, None)
    assert it > 0
    assert f <= f0 + 1e-12
    assert obj.value(d) == pytest.approx(f, abs=1e-9)
    assert np.allclose(poly.project(d), d, atol=1e-9)


def test_csa_at_the_coordinate_descent_fixed_point():
    """Load-bearing claim: run the median sweep to its fixed point, then CSA
    on the same polytope must still be no worse -- and on this chain it is
    strictly better, which is the whole reason the pass exists."""
    r = _drag_refiner()
    for _ in range(12):
        r._axis_pass(0)
    cons = r._axis_constraints(0)
    obj, poly = r._csa_problem(0, cons)
    span = r.xmax - r.xmin
    _d, f, f0, _it = csa.csa_shifts(obj, poly, 0.05 * span, 120, 0.95, None)
    # strictly better: the sweep is stuck at 49.0 while the joint optimum is
    # to walk the whole row back to where it started
    assert f < f0 - 1e-6


def test_csa_pass_escapes_the_coordinate_descent_fixed_point(monkeypatch):
    """End to end, on the layout: the median sweep converges to a strictly
    worse point than it started from and cannot come back; `_csa_pass` walks
    the whole packed row back to the joint optimum."""
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.5")
    r = _drag_refiner()
    for _ in range(4):
        r._axis_pass(0)
    hp_stuck = r.opt._hpwl(r.P)
    assert hp_stuck > r.hp0            # the sweep made it WORSE and is stuck
    x_stuck = r.P[:, 0].copy()
    r._axis_pass(0)
    assert np.array_equal(r.P[:, 0], x_stuck)
    kept = [r._csa_pass(0) for _ in range(4)]
    assert kept[0] is True
    hp_out = r.opt._hpwl(r.P)
    assert hp_out < hp_stuck - 1e-6
    assert hp_out == pytest.approx(r.hp0, abs=1e-2)   # recovers the optimum
    assert not r._has_overlap()
    assert kept[-1] is False           # and then it stops


def test_terminal_placement_carves_its_slice_out_of_the_span(monkeypatch):
    """`PARTNER_CSA_WHERE=end` must not extend the caller's deadline: the fork
    is deadline-bounded, so an overrun would surface as raw runtime."""
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_WHERE", "end")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.3")
    r = _drag_refiner()
    t_end = time.time() + 0.5
    out = r.run(t_end)
    assert time.time() <= t_end + 0.15
    assert r._csa_calls > 0
    r.P[...] = out
    assert not r._has_overlap()


def test_terminal_placement_reaches_the_pass_even_when_phase1_exhausts(
        monkeypatch):
    monkeypatch.setenv("PARTNER_CSA_REFINE", "1")
    monkeypatch.setenv("PARTNER_CSA_WHERE", "end")
    monkeypatch.setenv("PARTNER_CSA_SHARE", "0.3")
    r = _drag_refiner()
    seen = {"n": 0}
    inner = r._csa_pass

    def _counting(axis, key0=None):
        seen["n"] += 1
        return inner(axis, key0)

    r._csa_pass = _counting
    r.run(time.time() + 0.4)
    assert seen["n"] >= 2          # both axes at least once, terminal block
