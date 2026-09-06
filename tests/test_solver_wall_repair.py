"""`PARTNER_WALL_REPAIR` (default off): the seat + re-weld composite move in
`layout_refiner._wall_repair`.

Why the move exists: classifying every unsatisfied boundary-tag bit in a
shipped official-100 run (`scripts/probes/wall_seat_diag.py`, 2026-08-26)
showed that 28 of the 57 bits at n >= 76 sit on a block that can translate
straight onto its wall with NO clash, and that every one of those 28 is a
cluster member whose translate is exactly V-NEUTRAL (boundary -1, grouping
+1).  Every seat pass in the pipeline commits only on a STRICT drop, so none
of them can take step one, and the re-weld that pays for it never runs.

What must hold:
  * fixable case  -> the tag is seated AND its cluster is re-welded, the
    evaluator-form violation count strictly drops, and the layout stays
    overlap-free with exact areas.
  * blocked case  -> the gap in front of the tag is occupied; nothing is
    committed and the SAME object comes back.
  * off path      -> `wall_repair_on()` is False without the env var, and
    `ContestOptimizer._wall_repair_final` returns the identical list object
    (no import, no scorer build) when the flag is unset.
  * bounds        -> a deadline already in the past returns the input
    untouched; a preplaced (kind 2) tag is never translated.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout_refiner as LR                       # noqa: E402
from column_sa_legalizer import _ColumnOptimizer  # noqa: E402
from violation_killer import _violations_exact    # noqa: E402

_ENV = ("PARTNER_WALL_REPAIR", "PARTNER_WALL_REPAIR_MS",
        "PARTNER_WALL_REPAIR_SITE_MS", "PARTNER_WALL_REPAIR_MAX_SITES",
        "PARTNER_WALL_REPAIR_DEBUG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# synthetic instances
# ---------------------------------------------------------------------------

def _opt(rects, cons, tpos=None):
    n = len(rects)
    at = torch.tensor([float(w * h) for _x, _y, w, h in rects])
    if tpos is None:
        tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((1, 3))
    p2b = torch.zeros((1, 3))
    pins = torch.zeros((1, 2))
    return _ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                            time.time() + 5.0, seed=0)


def _fixable():
    """Block 0 is tagged LEFT and hovers 4 units off the left wall; the strip
    in front of it is empty, so it can translate onto the wall -- but it is
    clustered with block 1 sitting immediately to its right, so the translate
    alone trades a boundary violation for a grouping one.  Block 1 is free to
    follow it (nothing sits between the two), which is exactly the re-weld."""
    rects = [
        (4.0, 0.0, 6.0, 10.0),     # 0: tagged LEFT, hovering
        (10.0, 0.0, 6.0, 10.0),    # 1: its cluster peer, abutting
        (0.0, 10.0, 20.0, 10.0),   # 2: filler defining the left wall (x=0)
        (16.0, 0.0, 4.0, 10.0),    # 3: filler on the right
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 3] = 1.0               # cluster 1
    cons[1, 3] = 1.0
    cons[0, 4] = 1.0               # boundary: left
    return _opt(rects, cons), np.asarray(rects, dtype=np.float64)


def _blocked():
    """Same shape, but the strip in front of the tagged block is occupied, so
    no seat is even attempted."""
    rects = [
        (4.0, 0.0, 6.0, 10.0),     # 0: tagged LEFT, hovering
        (10.0, 0.0, 6.0, 10.0),    # 1: cluster peer
        (0.0, 10.0, 20.0, 10.0),   # 2: filler defining the left wall
        (16.0, 0.0, 4.0, 10.0),    # 3: filler on the right
        (0.0, 0.0, 4.0, 10.0),     # 4: blocker filling the gap
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 3] = 1.0
    cons[1, 3] = 1.0
    cons[0, 4] = 1.0
    return _opt(rects, cons), np.asarray(rects, dtype=np.float64)


def _no_overlap(P):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > 1e-7) & (oy > 1e-7)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_fixable_case_seats_and_rewelds():
    opt, P = _fixable()
    V0 = _violations_exact(opt, P)
    assert V0 >= 1                       # the hovering tag is the violation

    # the trap the move exists for: the seat ALONE is V-neutral
    Q = P.copy()
    Q[0, 0] = float(P[:, 0].min())
    assert _violations_exact(opt, Q) == V0

    R = np.asarray(LR._wall_repair(opt, P, deadline=time.time() + 1.0),
                   dtype=np.float64)
    assert _violations_exact(opt, R) < V0
    assert R[0, 0] == pytest.approx(float(R[:, 0].min()), abs=1e-9)
    assert _no_overlap(R)
    # exact areas preserved (nothing was reshaped past the evaluator's 1%)
    for i in range(opt.n):
        assert abs(R[i, 2] * R[i, 3] - opt.areas[i]) / opt.areas[i] <= 0.01


def test_blocked_case_is_identity():
    opt, P = _blocked()
    out = LR._wall_repair(opt, P, deadline=time.time() + 1.0)
    assert out is P                      # same object, nothing committed


def test_expired_deadline_is_identity():
    opt, P = _fixable()
    out = LR._wall_repair(opt, P, deadline=time.time() - 1.0)
    assert out is P


def test_preplaced_tag_is_never_translated():
    """A preplaced (kind 2) tagged block is immovable -- the dominant class in
    the shipped tail -- and must not be a repair site."""
    rects = [
        (4.0, 0.0, 6.0, 10.0),
        (10.0, 0.0, 6.0, 10.0),
        (0.0, 10.0, 20.0, 10.0),
        (16.0, 0.0, 4.0, 10.0),
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 1] = 1.0                       # preplaced
    cons[0, 3] = 1.0
    cons[1, 3] = 1.0
    cons[0, 4] = 1.0
    tpos = torch.full((len(rects), 4), -1.0)
    tpos[0] = torch.tensor([4.0, 0.0, 6.0, 10.0])
    opt = _opt(rects, cons, tpos)
    P = np.asarray(rects, dtype=np.float64)
    assert opt.kind[0] == 2
    out = LR._wall_repair(opt, P, deadline=time.time() + 1.0)
    assert out is P


def test_flag_defaults_off():
    assert LR.wall_repair_on() is False


def test_flag_on_off(monkeypatch):
    monkeypatch.setenv("PARTNER_WALL_REPAIR", "1")
    assert LR.wall_repair_on() is True
    monkeypatch.setenv("PARTNER_WALL_REPAIR", "0")
    assert LR.wall_repair_on() is False


def test_final_site_is_identity_when_flag_unset(monkeypatch):
    """`ContestOptimizer._wall_repair_final` must return the SAME list object
    with the flag unset -- no import, no scorer build, no copy."""
    monkeypatch.setenv("PARTNER_POOL", "0")
    import contest_optimizer as co
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    rects = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    out = opt._wall_repair_final(rects, None, None, None, None, None, None, [])
    assert out is rects


def test_site_budget_is_bounded():
    """The whole call must respect PARTNER_WALL_REPAIR_MS even when there is
    work to do (the re-weld's dig/evict tail is the unbounded branch)."""
    opt, P = _fixable()
    t0 = time.time()
    LR._wall_repair(opt, P, deadline=time.time() + 5.0)
    assert time.time() - t0 < 1.0
