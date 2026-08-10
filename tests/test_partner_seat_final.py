"""`PARTNER_SEAT_FINAL` -- run `_edge_seat` once more on the FINAL layout.

Every pre-existing `_edge_seat` call site sits upstream of a stage that can
still move blocks (`_pick_best` arbitration, then `_coord_polish`), so the list
`solve()` returns has never been offered to the pass.  This flag adds one last
call after `_coord_polish`.

What must hold:
  * flag off -> the hook is a single env lookup, returns the SAME list object,
    and never imports the refiner (so the default pipeline is byte-identical);
  * the flag is inert without `PARTNER_EDGE_SEAT_V2` (the narrow historical
    pass has nothing to add at this point);
  * flag on -> the returned layout is legal (no overlap, preplaced origins and
    fixed shapes untouched, soft areas inside the 1% hard tolerance) and its
    evaluator-faithful boundary+grouping+MIB total never rises;
  * flag on -> a layout with a reachable tag is actually seated.
"""

from __future__ import annotations

import os
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

import column_sa_legalizer as csl          # noqa: E402
import contest_optimizer as co             # noqa: E402
import layout_refiner as lr                # noqa: E402

_ENV = ("PARTNER_SEAT_FINAL", "PARTNER_SEAT_FINAL_DEBUG",
        "PARTNER_EDGE_SEAT_V2", "PARTNER_COORD_POLISH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)


def _case():
    """A tiny instance whose right-tagged block hovers off the right wall by
    a hair -- reachable by a sub-1% edge dilate, which is exactly the move the
    final pass exists to catch."""
    n = 4
    at = torch.tensor([100.0, 100.0, 100.0, 100.0])
    cons = torch.zeros((n, 5))
    cons[3, 4] = 2.0                       # block 3 tagged RIGHT
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((0, 3))
    p2b = torch.zeros((0, 3))
    pins = torch.zeros((0, 2))
    #  (0,0) (10,0)      (0,10) (10,10) -- block 3 short of the right wall
    rects = [(0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 10.0, 10.0), (10.0, 10.0, 9.95, 10.05)]
    return n, at, cons, tpos, b2b, p2b, pins, rects


def _scorer(at, cons, tpos, b2b, p2b, pins, rects):
    import time
    return csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                time.time() + 60.0, seed=0)


def _opt():
    return co.ContestOptimizer() if hasattr(co, "ContestOptimizer") \
        else co.MyOptimizer()


def test_off_is_identity_and_imports_nothing(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _case()
    o = _opt()
    boom = []
    monkeypatch.setattr(lr, "_edge_seat",
                        lambda *a, **k: boom.append(1) or a[1])
    out = list(rects)
    got = o._final_seat(out, at, cons, tpos, b2b, p2b, pins, None)
    assert got is out                      # same object, no copy
    assert not boom                        # never called


def test_inert_without_edge_seat_v2(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _case()
    monkeypatch.setenv("PARTNER_SEAT_FINAL", "1")
    o = _opt()
    out = list(rects)
    assert o._final_seat(out, at, cons, tpos, b2b, p2b, pins, None) is out


def test_on_seats_a_reachable_tag(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _case()
    monkeypatch.setenv("PARTNER_SEAT_FINAL", "1")
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")
    sc = _scorer(at, cons, tpos, b2b, p2b, pins, rects)
    v0 = lr.full_violations(sc, np.asarray(rects, dtype=np.float64))
    assert v0 >= 1                         # the fixture must actually violate
    o = _opt()
    got = o._final_seat(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    Q = np.asarray([tuple(map(float, r)) for r in got], dtype=np.float64)
    assert lr.full_violations(sc, Q) < v0


def test_on_never_raises_violations_and_keeps_legality(monkeypatch):
    monkeypatch.setenv("PARTNER_SEAT_FINAL", "1")
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")
    rng = np.random.default_rng(7)
    o = _opt()
    for _t in range(12):
        n = 6
        at = torch.tensor([100.0] * n)
        cons = torch.zeros((n, 5))
        cons[int(rng.integers(0, n)), 4] = float(rng.choice([1, 2, 4, 8]))
        pre = int(rng.integers(0, n))
        cons[pre, 1] = 1.0
        tpos = torch.full((n, 4), -1.0)
        b2b = torch.zeros((0, 3))
        p2b = torch.zeros((0, 3))
        pins = torch.zeros((0, 2))
        # two rows of three, laid out cumulatively so the start layout is
        # overlap-free whatever aspect the draw picks
        rects = [None] * n
        row_y = 0.0
        for row in range(2):
            x = 0.0
            row_h = 0.0
            for col in range(3):
                k = row * 3 + col
                w = 10.0 - 0.1 * float(rng.integers(0, 4))
                h = 100.0 / w
                rects[k] = (x, row_y, w, h)
                x += w
                row_h = max(row_h, h)
            row_y += row_h
        tpos[pre] = torch.tensor(list(rects[pre]))
        sc = _scorer(at, cons, tpos, b2b, p2b, pins, rects)
        P = np.asarray(rects, dtype=np.float64)
        v0 = lr.full_violations(sc, P)
        got = o._final_seat(list(rects), at, cons, tpos, b2b, p2b, pins, None)
        Q = np.asarray([tuple(map(float, r)) for r in got], dtype=np.float64)
        assert Q.shape == P.shape
        assert lr.full_violations(sc, Q) <= v0
        # no overlap
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(Q[i, 0] + Q[i, 2], Q[j, 0] + Q[j, 2]) \
                    - max(Q[i, 0], Q[j, 0])
                oy = min(Q[i, 1] + Q[i, 3], Q[j, 1] + Q[j, 3]) \
                    - max(Q[i, 1], Q[j, 1])
                assert not (ox > 1e-6 and oy > 1e-6), (i, j)
        # preplaced origin untouched, soft areas inside the 1% hard tolerance
        assert abs(Q[pre, 0] - P[pre, 0]) < 1e-9
        assert abs(Q[pre, 1] - P[pre, 1]) < 1e-9
        for i in range(n):
            assert Q[i, 2] * Q[i, 3] <= float(at[i]) * 1.01 + 1e-9


def test_failure_is_contained(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _case()
    monkeypatch.setenv("PARTNER_SEAT_FINAL", "1")
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")

    def _boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(lr, "_edge_seat", _boom)
    o = _opt()
    out = list(rects)
    assert o._final_seat(out, at, cons, tpos, b2b, p2b, pins, None) == out
