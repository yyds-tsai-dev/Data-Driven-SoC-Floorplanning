"""`PARTNER_COORD_POLISH=1` -> order-preserving simultaneous-axis coordinate
polish of the final layout (partner/coord_polish.py).

Every coordinate stage upstream of this pass is a coordinate descent over
groups, so the pipeline's output is only locally optimal for its own topology.
The pass freezes that topology (one separation constraint per block pair,
oriented by centroid order) and re-solves both axes jointly.

What must hold:
  * flag off -> the optimizer hook returns the SAME list object, the module is
    never imported, and no env other than the flag can wake it;
  * the separation topology covers every pair exactly once, is acyclic on both
    axes, and is satisfied by the layout it was read from;
  * transitive reduction preserves the reachability closure (so it preserves
    the feasible set);
  * on the triggered path: shapes bit-identical (=> exact-area, fixed-shape and
    MIB survive), preplaced origins bit-identical, no overlap above the guard,
    and the layout is returned ONLY when the evaluator-faithful proxy strictly
    improves;
  * `_LiteScorer` reproduces the evaluator's HPWL and N_soft exactly;
  * every failure is contained -- a raising solver, a cyclic graph or a busted
    guard all return the input layout unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import coord_polish as cp                 # noqa: E402

_ENV = ("PARTNER_COORD_POLISH", "PARTNER_COORD_POLISH_MIN_N",
        "PARTNER_COORD_POLISH_BUDGET_MS", "PARTNER_COORD_POLISH_BACKEND",
        "PARTNER_COORD_POLISH_BOUNDARY", "PARTNER_COORD_POLISH_CONTACTS",
        "PARTNER_COORD_POLISH_FREE_FALLBACK", "PARTNER_COORD_POLISH_MAX_ROWS",
        "PARTNER_COORD_POLISH_LP", "PARTNER_COORD_POLISH_DEBUG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# a synthetic instance with room to polish
# ---------------------------------------------------------------------------

def _grid_case(side: int = 10, pitch: float = 2.0):
    """`side * side` unit blocks on a loose `pitch` grid.

    The slack is the point: the blocks start far apart, so the axis LPs have
    something real to compact, and the wall-flush blocks carry boundary codes
    that a free polish would happily break.

    Returns `(rects, area_targets, constraints, target_positions, b2b, p2b,
    pins)` in exactly the contest's tensor shapes.
    """
    n = side * side
    rects = [(pitch * (i % side), pitch * (i // side), 1.0, 1.0)
             for i in range(n)]
    at = torch.ones(n)
    cons = torch.zeros((n, 5))

    # preplaced: one interior block, pinned where it stands
    pre = n // 2
    cons[pre, 1] = 1.0
    tpos = torch.full((n, 4), -1.0)
    tpos[pre] = torch.tensor(list(rects[pre]))

    # MIB group of three identical blocks
    for i in (1, 2, 3):
        cons[i, 2] = 1.0
    # cluster of two horizontally adjacent grid slots
    cons[side + 1, 3] = 1.0
    cons[side + 2, 3] = 1.0
    # boundary codes on blocks that currently ARE flush on that wall
    # (row = i // side, column = i % side)
    cons[0, 4] = 1.0                       # left   (column 0)
    cons[side - 1, 4] = 2.0                # right  (last column)
    cons[1, 4] = 8.0                       # bottom (row 0)
    cons[n - 1, 4] = 4.0                   # top    (last row)

    # a long-range net pulling opposite corners together, plus a chain
    edges = [[0.0, float(n - 1), 5.0]]
    for i in range(0, n - 1, 7):
        edges.append([float(i), float(i + 1), 1.0])
    b2b = torch.tensor(edges, dtype=torch.float32)

    pins = torch.tensor([[0.0, 0.0], [pitch * side, pitch * side]],
                        dtype=torch.float32)
    p2b = torch.tensor([[0.0, 0.0, 2.0], [1.0, float(n - 1), 2.0]],
                       dtype=torch.float32)
    return rects, at, cons, tpos, b2b, p2b, pins


def _P(rects):
    return np.asarray([[float(a) for a in r] for r in rects], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. the off path
# ---------------------------------------------------------------------------

def test_optimizer_hook_is_identity_when_flag_unset():
    """Bare defaults must not move a single byte, and must not even import."""
    import contest_optimizer as co

    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    opt.verbose = False
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case(side=3)
    out = list(rects)
    got = opt._coord_polish(out, at, cons, tpos, b2b, p2b, pins)
    assert got is out          # identity, not merely equal


def test_min_n_gate_blocks_small_instances(monkeypatch):
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case(side=5)   # n = 25 < 95
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert [tuple(r) for r in got] == [tuple(r) for r in rects]


# ---------------------------------------------------------------------------
# 2. the separation topology
# ---------------------------------------------------------------------------

def test_topology_covers_every_pair_exactly_once():
    rects, *_ = _grid_case()
    P = _P(rects)
    n = len(P)
    hor, ver = cp.build_topology(P)
    assert len(hor) + len(ver) == n * (n - 1) // 2
    seen = {frozenset((int(a), int(b))) for a, b in hor}
    seen |= {frozenset((int(a), int(b))) for a, b in ver}
    assert len(seen) == n * (n - 1) // 2


@pytest.mark.parametrize("axis", (0, 1))
def test_topology_is_acyclic_and_satisfied_by_its_own_layout(axis):
    """Centroid order is a total order, so each axis graph is a DAG -- that is
    what keeps the LP feasible even on layouts with micro-overlaps."""
    rects, *_ = _grid_case()
    P = _P(rects)
    edges = cp.build_topology(P)[axis]
    n = len(P)
    indeg = np.zeros(n, dtype=np.int64)
    succ = [[] for _ in range(n)]
    for a, b in edges:
        succ[int(a)].append(int(b))
        indeg[int(b)] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    assert seen == n                       # acyclic
    # the grid is overlap-free, so every oriented edge must already hold
    c = P[:, axis]
    s = P[:, axis + 2]
    for a, b in edges:
        assert c[a] + s[a] <= c[b] + 1e-9


def test_transitive_reduction_preserves_the_closure():
    rects, *_ = _grid_case(side=6)
    P = _P(rects)
    n = len(P)
    for edges in cp.build_topology(P):
        red = cp.transitive_reduce(n, edges)
        assert len(red) <= len(edges)

        def closure(e):
            A = np.zeros((n, n), dtype=bool)
            if len(e):
                A[e[:, 0], e[:, 1]] = True
            C = A.copy()
            for k in range(n):
                C |= np.outer(C[:, k], C[k, :])
            return C

        assert np.array_equal(closure(edges), closure(red))


# ---------------------------------------------------------------------------
# 3. the scoring surface is evaluator-faithful
# ---------------------------------------------------------------------------

def test_lite_scorer_matches_the_evaluator_hpwl_and_nsoft():
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    P = _P(rects)
    sc = cp._LiteScorer(len(P), cons, b2b, p2b, pins)

    # reference HPWL: centroid Manhattan, exactly scripts/iccad2026_evaluate.py
    cx = P[:, 0] + P[:, 2] / 2.0
    cy = P[:, 1] + P[:, 3] / 2.0
    ref = 0.0
    for i, j, w in b2b.tolist():
        ref += w * (abs(cx[int(i)] - cx[int(j)]) + abs(cy[int(i)] - cy[int(j)]))
    pn = pins.tolist()
    for p, bi, w in p2b.tolist():
        ref += w * (abs(pn[int(p)][0] - cx[int(bi)])
                    + abs(pn[int(p)][1] - cy[int(bi)]))
    assert sc._hpwl(P) == pytest.approx(ref, rel=1e-12, abs=1e-9)

    # reference N_soft = |B_boundary| + sum(|G_p| - 1) + sum(|M_q| - 1)
    n_soft = int((cons[:, 4] != 0).sum())
    for col in (2, 3):
        vals = cons[:, col].round().to(torch.int64)
        for g in range(1, int(vals.max().item()) + 1):
            n_soft += max(0, int((vals == g).sum()) - 1)
    assert sc.n_soft_den == max(n_soft, 1)


def test_term_merging_is_an_identity_on_the_objective():
    """Merging parallel terms is a speedup only if it is exact -- it changes
    the LP the solver sees, so it must not change the function it minimises."""
    rng = np.random.default_rng(7)
    n = 12
    eI = rng.integers(0, n, 60)
    eJ = rng.integers(0, n, 60)
    eW = rng.random(60)
    merged = cp._merge_pairs(eI, eJ, eW)
    for _ in range(5):
        u = rng.random(n) * 10.0
        raw = float(np.sum(eW[eI != eJ]
                           * np.abs(u[eI[eI != eJ]] - u[eJ[eI != eJ]])))
        got = sum(w * abs(u[i] - u[j]) for i, j, w in merged)
        assert got == pytest.approx(raw, rel=1e-12, abs=1e-12)

    pB = rng.integers(0, n, 40)
    pC = np.round(rng.random(40) * 3.0, 1)      # forces coincident pins
    pW = rng.random(40)
    mp = cp._merge_pins(pB, pC, pW)
    for _ in range(5):
        u = rng.random(n) * 10.0
        raw = float(np.sum(pW * np.abs(u[pB] - pC)))
        got = sum(w * abs(u[i] - p) for i, p, w in mp)
        assert got == pytest.approx(raw, rel=1e-12, abs=1e-12)


def test_boundary_mode_2_never_asks_the_lp_to_repair_a_wall(monkeypatch):
    """Mode 2 preserves the wall bits the layout already holds; a bit the
    pipeline could not reach is left alone (asking for it made the axis system
    infeasible on 12 of 26 validation-tail cases -- a full LP for nothing)."""
    pytest.importorskip("scipy")
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    monkeypatch.setenv("PARTNER_COORD_POLISH_BOUNDARY", "2")
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    cons = cons.clone()
    # block 4 is nowhere near the left wall, but demand it anyway
    off_wall = 4
    cons[off_wall, 4] = 1.0
    seen = {}
    real = cp.solve_axis_scipy

    def spy(n, size, lo, hi, edges, nets, pin_terms, eq,
            min_blocks=(), max_blocks=(), *a, **k):
        seen.setdefault("min", list(min_blocks))
        return real(n, size, lo, hi, edges, nets, pin_terms, eq,
                    min_blocks, max_blocks, *a, **k)

    monkeypatch.setattr(cp, "solve_axis_scipy", spy)
    cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert seen, "the x axis solve never ran"
    assert 0 in seen["min"]                 # block 0 IS flush left -> held
    assert off_wall not in seen["min"]      # block 4 is not -> not demanded


# ---------------------------------------------------------------------------
# 4. the triggered path
# ---------------------------------------------------------------------------

@pytest.fixture
def _polished(monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    monkeypatch.setenv("PARTNER_COORD_POLISH_BUDGET_MS", "20000")
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    return rects, got, cons, b2b, p2b, pins


def test_triggered_path_actually_fires(_polished):
    """If this ever stops firing the invariant tests below go vacuous."""
    rects, got, *_ = _polished
    assert [tuple(r) for r in got] != [tuple(r) for r in rects]


def test_shapes_are_bit_identical(_polished):
    """Frozen shapes are what carry exact-area, fixed-shape AND MIB through."""
    rects, got, *_ = _polished
    for r0, r1 in zip(rects, got):
        assert r1[2] == r0[2]
        assert r1[3] == r0[3]


def test_preplaced_origin_is_bit_identical(_polished):
    rects, got, cons, *_ = _polished
    for i in torch.nonzero(cons[:, 1]).flatten().tolist():
        assert got[i][0] == rects[i][0]
        assert got[i][1] == rects[i][1]


def test_no_overlap_after_polish(_polished):
    _rects, got, *_ = _polished
    assert cp._overlap_ok(_P(got))


def test_mib_groups_keep_one_shape(_polished):
    _rects, got, cons, *_ = _polished
    P = _P(got)
    mib = cons[:, 2].round().to(torch.int64)
    for g in range(1, int(mib.max().item()) + 1):
        idx = torch.nonzero(mib == g).flatten().tolist()
        shapes = {(round(float(P[i, 2]), 4), round(float(P[i, 3]), 4))
                  for i in idx}
        assert len(shapes) <= 1


def test_bbox_never_grows(_polished):
    """The axis bounds are the input bbox, so `area_gap` can only improve."""
    rects, got, *_ = _polished
    for a in (0, 1):
        b0 = _P(rects)
        b1 = _P(got)
        assert b1[:, a].min() >= b0[:, a].min() - 1e-9
        assert (b1[:, a] + b1[:, a + 2]).max() <= (b0[:, a] + b0[:, a + 2]).max() + 1e-9


def test_accepted_only_when_the_proxy_strictly_improves(_polished):
    rects, got, cons, b2b, p2b, pins = _polished
    from violation_killer import _Ctx
    sc = cp._LiteScorer(len(rects), cons, b2b, p2b, pins)
    P0 = _P(rects)
    ctx = _Ctx(sc, P0)
    assert ctx.score(_P(got))[0] < ctx.score(P0)[0]


# ---------------------------------------------------------------------------
# 5. guards and containment
# ---------------------------------------------------------------------------

def test_hard_guard_rejects_a_reshaped_block():
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case(side=4)
    P0 = _P(rects)
    sc = cp._LiteScorer(len(P0), cons, b2b, p2b, pins)
    bad = P0.copy()
    bad[2, 2] += 1e-9
    assert not cp._hard_ok(P0, bad, sc)


def test_hard_guard_rejects_a_moved_preplaced_block():
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case(side=4)
    cons = cons.clone()
    cons[3, 1] = 1.0
    P0 = _P(rects)
    sc = cp._LiteScorer(len(P0), cons, b2b, p2b, pins)
    bad = P0.copy()
    bad[3, 0] += 1e-9
    assert not cp._hard_ok(P0, bad, sc)


def test_hard_guard_rejects_an_overlap():
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case(side=4)
    P0 = _P(rects)
    sc = cp._LiteScorer(len(P0), cons, b2b, p2b, pins)
    bad = P0.copy()
    bad[1, 0] = bad[0, 0] + 0.5           # slide block 1 onto block 0
    bad[1, 1] = bad[0, 1]
    assert not cp._hard_ok(P0, bad, sc)


def test_a_raising_axis_solver_is_contained(monkeypatch):
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")

    def boom(*a, **k):
        raise RuntimeError("LP exploded")

    monkeypatch.setattr(cp, "solve_axis_scipy", boom)
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert [tuple(r) for r in got] == [tuple(r) for r in rects]


def test_an_infeasible_axis_solve_is_contained(monkeypatch):
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    monkeypatch.setattr(cp, "solve_axis_scipy", lambda *a, **k: None)
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert [tuple(r) for r in got] == [tuple(r) for r in rects]


def test_zero_budget_is_contained(monkeypatch):
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    monkeypatch.setenv("PARTNER_COORD_POLISH_BUDGET_MS", "0")
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert [tuple(r) for r in got] == [tuple(r) for r in rects]


def test_oversized_instance_is_refused_before_the_lp(monkeypatch):
    """The LP call is atomic, so the size predictor has to refuse up front."""
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MAX_ROWS", "1")

    def boom(*a, **k):
        raise AssertionError("predictor let an oversized LP through")

    monkeypatch.setattr(cp, "solve_axis_scipy", boom)
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert [tuple(r) for r in got] == [tuple(r) for r in rects]


def _block_scipy(monkeypatch):
    real_import = __import__

    def no_scipy(name, *a, **k):
        if name.split(".")[0] == "scipy":
            raise ImportError("scipy is not available here")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", no_scipy)


def test_auto_backend_disables_itself_without_scipy(monkeypatch):
    """scipy is NOT in FloorSet/iccad2026contest/requirements.txt, and the
    dependency-free backend cannot express the wall rows that carry the whole
    gain -- so `auto` must go quiet rather than spend runtime for a zero."""
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    _block_scipy(monkeypatch)
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    assert [tuple(r) for r in got] == [tuple(r) for r in rects]


def test_numpy_backend_runs_and_stays_legal_without_scipy(monkeypatch):
    """The experimental fallback must still be legal and never raise."""
    monkeypatch.setenv("PARTNER_COORD_POLISH", "1")
    monkeypatch.setenv("PARTNER_COORD_POLISH_MIN_N", "10")
    monkeypatch.setenv("PARTNER_COORD_POLISH_BACKEND", "numpy")
    _block_scipy(monkeypatch)
    rects, at, cons, tpos, b2b, p2b, pins = _grid_case()
    got = cp.polish_layout(rects, at, cons, tpos, b2b, p2b, pins)
    P0, P1 = _P(rects), _P(got)
    assert np.array_equal(P1[:, 2:], P0[:, 2:])
    assert cp._overlap_ok(P1)
