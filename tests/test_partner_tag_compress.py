"""`PARTNER_TAG_COMPRESS` -- compact the final layout onto preplaced tag lines.

A preplaced block cannot move, so a boundary tag it carries is satisfiable only
if the final bbox edge on that side IS the block's own edge.  `_edge_seat` can
translate, dilate, or rigidly pull at most 8 outliers; the blocks that overshoot
such a line sit in packed chains, so the residual survives to the evaluator.
This pass pushes the whole chain back onto the line, paying the residual out of
the soft blocks' unused 1% area tolerance.

What must hold:
  * flag off -> a single env lookup, the SAME list object, no import;
  * flag on -> a reachable single-block line is seated;
  * flag on -> a two-block CHAIN is seated (the propagation, not just a
    single translate, is what makes the pass different from `_edge_seat`);
  * flag on -> a line whose chain terminates on a second preplaced block is
    ABORTED, not forced (hard legality precedes soft repair);
  * flag on -> randomized layouts stay legal (no overlap, preplaced origin and
    dimensions untouched, fixed shapes untouched, soft areas inside the 1%
    hard tolerance) and the evaluator-faithful boundary+grouping+MIB total
    never rises;
  * any failure is contained.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl          # noqa: E402
import contest_optimizer as co             # noqa: E402
import layout_refiner as lr                # noqa: E402
import tag_compress as tc                  # noqa: E402

_ENV = ("PARTNER_TAG_COMPRESS", "PARTNER_TAG_COMPRESS_DEBUG",
        "PARTNER_GROUP_BRIDGE", "PARTNER_GROUP_BRIDGE_BUDGET",
        "PARTNER_GROUP_BRIDGE_DEBUG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)


def _opt():
    return co.ContestOptimizer() if hasattr(co, "ContestOptimizer") \
        else co.MyOptimizer()


def _inst(areas, rects, pre_idx, tag_bit, tag_idx=None):
    """A tiny instance: `pre_idx` blocks are preplaced (locked where they
    are), `tag_idx` (default the first preplaced) carries `tag_bit`."""
    n = len(rects)
    at = torch.tensor([float(a) for a in areas])
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    for i in pre_idx:
        cons[i, 1] = 1.0
        tpos[i] = torch.tensor(list(rects[i]))
    cons[pre_idx[0] if tag_idx is None else tag_idx, 4] = float(tag_bit)
    return (n, at, cons, tpos, torch.zeros((0, 3)), torch.zeros((0, 3)),
            torch.zeros((0, 2)), rects)


def _scorer(at, cons, tpos, b2b, p2b, pins, rects):
    return csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                time.time() + 60.0, seed=0)


def _single():
    """Block 0 preplaced at x in [0, 10] and tagged RIGHT; block 1 pokes
    0.05 past the line and can pay for it out of its area tolerance."""
    rects = [(0.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 10.05, 10.0),
             (0.0, 20.0, 10.0, 10.0)]
    return _inst([100.0, 100.5, 100.0], rects, [0], 2)


def _chain():
    """Same line, but two blocks abut across it -- seating the outer one
    requires pushing the inner one too."""
    rects = [(0.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 5.0, 10.0),
             (5.0, 10.0, 5.05, 10.0)]
    return _inst([100.0, 50.0, 50.5], rects, [0], 2)


def _blocked():
    """Same line, but a SECOND preplaced block sits past it: the push would
    have to move a hard-locked block, so it must abort."""
    rects = [(0.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 10.05, 10.0),
             (0.0, 20.0, 10.0, 10.0)]
    inst = _inst([100.0, 100.5, 100.0], rects, [0, 1], 2, tag_idx=0)
    return inst


# ------------------------------------------------------------------ off path
def test_off_is_identity(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    boom = []
    monkeypatch.setattr(tc, "tag_compress",
                        lambda *a, **k: boom.append(1) or a[1])
    out = list(rects)
    got = _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    assert got is out                      # same object, no copy
    assert not boom                        # never called


def test_constructor_warms_dependencies_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(tc, "warm_dependencies",
                        lambda: calls.append("warm"))

    _opt()
    assert calls == []

    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    _opt()
    assert calls == ["warm"]


def test_group_bridge_flag_warms_dependencies_in_constructor(monkeypatch):
    calls = []
    monkeypatch.setattr(tc, "warm_dependencies",
                        lambda: calls.append("warm"))
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    _opt()
    assert calls == ["warm"]


def test_both_final_flags_off_are_identity_without_scorer(monkeypatch):
    out = [(0.0, 0.0, 1.0, 1.0)]
    monkeypatch.setattr(co, "_ColumnOptimizer",
                        lambda *a, **k: pytest.fail("scorer constructed"))
    assert _opt()._tag_compress(out, torch.ones(1), torch.zeros((1, 5)),
                                torch.full((1, 4), -1.0), torch.zeros((0, 3)),
                                torch.zeros((0, 3)), torch.zeros((0, 2)), None) is out


def test_group_bridge_works_with_tag_compress_off(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setattr(tc, "tag_compress",
                        lambda *a, **k: pytest.fail("tag pass called"))
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda scorer, value, budget: value)
    out = list(rects)
    assert _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None) is out


def test_bridge_failure_preserves_tag_result(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    accepted = [(1.0, 2.0, 3.0, 4.0)] * len(rects)
    monkeypatch.setattr(tc, "tag_compress", lambda scorer, value: accepted)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins,
                                None) is accepted


def test_group_bridge_debug_reports_self_paired_fields(monkeypatch, capsys):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE_DEBUG", "1")
    out = list(rects)
    _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    err = capsys.readouterr().err
    for field in ("ms=", "grouping=", "V=", "hpwl=", "bbox=", "committed="):
        assert field in err


def test_warm_dependencies_is_idempotent():
    previous = tc._EXACT_VIOL_FN
    try:
        tc._EXACT_VIOL_FN = None
        first = tc.warm_dependencies()
        second = tc.warm_dependencies()
        assert callable(first)
        assert second is first
    finally:
        tc._EXACT_VIOL_FN = previous


def test_failure_is_contained(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")

    def _boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(tc, "tag_compress", _boom)
    out = list(rects)
    assert _opt()._tag_compress(
        out, at, cons, tpos, b2b, p2b, pins, None) == out


def test_pure_function_swallows_a_broken_opt():
    class _Bad:
        n = 3

        def __getattr__(self, k):
            raise RuntimeError("nope")

    out = [(0.0, 0.0, 1.0, 1.0)]
    assert tc.tag_compress(_Bad(), out) is out


# ------------------------------------------------------------------- on path
@pytest.mark.parametrize("case", ["single", "chain"])
def test_on_seats_a_reachable_line(monkeypatch, case):
    n, at, cons, tpos, b2b, p2b, pins, rects = \
        _single() if case == "single" else _chain()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    sc = _scorer(at, cons, tpos, b2b, p2b, pins, rects)
    P = np.asarray(rects, dtype=np.float64)
    v0 = lr.full_violations(sc, P)
    assert v0 >= 1                         # the fixture must actually violate

    got = _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins,
                               None)
    Q = np.asarray([tuple(map(float, r)) for r in got], dtype=np.float64)
    assert lr.full_violations(sc, Q) < v0
    # the tagged block's own right edge IS the bbox right edge now
    assert abs((Q[:, 0] + Q[:, 2]).max() - 10.0) < 1e-6
    # preplaced untouched, soft areas inside the hard tolerance
    assert np.allclose(Q[0], P[0])
    for i in range(1, n):
        assert abs(Q[i, 2] * Q[i, 3] - float(at[i])) / float(at[i]) <= 0.01


def test_on_aborts_when_the_chain_hits_a_locked_block(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _blocked()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    out = list(rects)
    got = _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    assert [tuple(map(float, r)) for r in got] == \
        [tuple(map(float, r)) for r in out]


def test_on_keeps_legality_and_never_raises_violations(monkeypatch):
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    rng = np.random.default_rng(11)
    o = _opt()
    for _t in range(24):
        rows, cols = 3, 3
        n = rows * cols
        rects = [None] * n
        areas = [0.0] * n
        y = 0.0
        for r in range(rows):
            x = 0.0
            rh = 0.0
            for c in range(cols):
                k = r * cols + c
                w = 10.0 + float(rng.integers(0, 30)) / 10.0
                h = 10.0 + float(rng.integers(0, 30)) / 10.0
                rects[k] = (x, y, w, h)
                areas[k] = w * h
                x += w
                rh = max(rh, h)
            y += rh
        pre = int(rng.integers(0, n))
        bit = int(rng.choice([1, 2, 4, 8]))
        n_, at, cons, tpos, b2b, p2b, pins, rects = \
            _inst(areas, rects, [pre], bit)
        sc = _scorer(at, cons, tpos, b2b, p2b, pins, rects)
        P = np.asarray(rects, dtype=np.float64)
        v0 = lr.full_violations(sc, P)
        got = o._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins,
                              None)
        Q = np.asarray([tuple(map(float, r)) for r in got], dtype=np.float64)
        assert Q.shape == P.shape
        assert lr.full_violations(sc, Q) <= v0
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(Q[i, 0] + Q[i, 2], Q[j, 0] + Q[j, 2]) \
                    - max(Q[i, 0], Q[j, 0])
                oy = min(Q[i, 1] + Q[i, 3], Q[j, 1] + Q[j, 3]) \
                    - max(Q[i, 1], Q[j, 1])
                assert not (ox > 1e-6 and oy > 1e-6), (i, j)
        assert np.allclose(Q[pre], P[pre])
        for i in range(n):
            assert abs(Q[i, 2] * Q[i, 3] - float(at[i])) \
                / float(at[i]) <= 0.01 + 1e-9
        # bbox never grows on either axis
        assert (Q[:, 0] + Q[:, 2]).max() - Q[:, 0].min() \
            <= (P[:, 0] + P[:, 2]).max() - P[:, 0].min() + 1e-9
        assert (Q[:, 1] + Q[:, 3]).max() - Q[:, 1].min() \
            <= (P[:, 1] + P[:, 3]).max() - P[:, 1].min() + 1e-9
