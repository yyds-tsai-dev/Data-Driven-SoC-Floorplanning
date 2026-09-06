"""`PARTNER_PIN_FRAME` (default off): tag-locked wall lines as HARD frame
edges inside `layout_refiner.refine_prediction`'s tight rungs.

Why the mechanism exists: `scripts/probes/wall_seat_diag.py` on the shipped
official-100 / shadow-v3 runs (2026-08-27) attributes 28 of the ~37 residual
boundary bits at n >= 76 to the `locked` class -- the tag sits on a PREPLACED
(kind 2) block, so the block cannot move and the WALL has to come to it, yet
the layout's wall on that side lies past the block's edge at frame
utilizations of only 0.80-0.97.  `_Refiner._anchor_frame_to_tags` already
computes those wall lines, but the shipped ladder consults them only as a
FLOOR for shrinking; nothing caps a rung's frame at them, and the MIN-side
locks (7 of the 28 bits) are never applied to the frame at all.

What must hold:
  * max-side lock  -> `_pin_frame_to_locks` caps the wall exactly on the line
    and hands the clamped area to a still-free side; the layout legalizes
    overlap-free inside the pinned frame.
  * min-side lock  -> the frame corner is raised onto the line (the class the
    shipped rung 0 never touches) and `refine_prediction` returns a layout
    whose bbox min IS the line.
  * no locks / degenerate clamp -> False, frame untouched.
  * off path       -> `_PIN_FRAME` is False without the env var and the same
    instance keeps the shipped (overshooting) frame.
"""

from __future__ import annotations

import importlib
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ("PARTNER_PIN_FRAME", "PARTNER_PIN_FRAME_RETRY",
              "PARTNER_PINFRAME_DEBUG",
              "PARTNER_TAG_ANCHOR", "PARTNER_TAG_PACK",
              "PARTNER_ANYTIME_LADDER", "PARTNER_RUNG05",
              "PARTNER_REFINE_GUARD", "PARTNER_LEGAL_ADMIT",
              "PARTNER_WALL_REPAIR", "PARTNER_FRAME_SCALE_LADDER",
              "PARTNER_FRAME_SCALE_SET", "REFINER_DEBUG"):
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# synthetic instances
# ---------------------------------------------------------------------------

def _opt(rects, cons, tpos, budget: float = 20.0):
    n = len(rects)
    at = torch.tensor([float(w * h) for _x, _y, w, h in rects])
    b2b = torch.zeros((1, 3))
    p2b = torch.zeros((1, 3))
    pins = torch.zeros((1, 2))
    return _ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                            time.time() + budget, seed=0)


def _left_tagged():
    """Block 0 is PREPLACED at x=[0,20] and tagged LEFT; blocks 1/2 are soft
    and are *predicted* to the left of it, so the prediction's own xmin (-20)
    -- not the tag line (0) -- is what the shipped rung 0 hands to the frame.
    Everything fits inside a frame that starts at the tag line."""
    rects = [
        (0.0, 0.0, 20.0, 40.0),     # 0: preplaced, tagged LEFT
        (-20.0, 0.0, 20.0, 20.0),   # 1: soft, predicted past the tag line
        (-20.0, 20.0, 20.0, 20.0),  # 2: soft, predicted past the tag line
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 1] = 1.0                # preplaced
    cons[0, 4] = 1.0                # boundary: left
    tpos = torch.full((len(rects), 4), -1.0)
    tpos[0] = torch.tensor([0.0, 0.0, 20.0, 40.0])
    return _opt(rects, cons, tpos), np.asarray(rects, dtype=np.float64)


def _right_tagged():
    """Block 0 is PREPLACED at x=[80,100] and tagged RIGHT; block 1 is soft
    and predicted past its wall line, so the layout's right wall sits at 130
    while the tag needs it at 100."""
    rects = [
        (80.0, 0.0, 20.0, 20.0),    # 0: preplaced, tagged RIGHT
        (100.0, 0.0, 20.0, 20.0),   # 1: soft, overshooting the wall
        (0.0, 0.0, 80.0, 20.0),     # 2: soft filler
        (0.0, 30.0, 40.0, 20.0),    # 3: soft, gives the frame its slack
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 1] = 1.0
    cons[0, 4] = 2.0                # boundary: right
    tpos = torch.full((len(rects), 4), -1.0)
    tpos[0] = torch.tensor([80.0, 0.0, 20.0, 20.0])
    return _opt(rects, cons, tpos), np.asarray(rects, dtype=np.float64)


def _untagged():
    rects = [
        (0.0, 0.0, 20.0, 40.0),
        (20.0, 0.0, 20.0, 40.0),
    ]
    cons = torch.zeros((len(rects), 5))
    tpos = torch.full((len(rects), 4), -1.0)
    return _opt(rects, cons, tpos), np.asarray(rects, dtype=np.float64)


def _no_overlap(P):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > 1e-7) & (oy > 1e-7)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# flag plumbing
# ---------------------------------------------------------------------------

def test_flag_defaults_off():
    assert LR._PIN_FRAME is False


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("PARTNER_PIN_FRAME", "1")
    mod = importlib.reload(LR)
    try:
        assert mod._PIN_FRAME is True
        assert mod._PIN_FRAME_RUNGS is True
        assert mod._PIN_FRAME_RETRY is False
    finally:
        monkeypatch.delenv("PARTNER_PIN_FRAME", raising=False)
        importlib.reload(mod)


def test_min_mode_pins_only_the_rung0_corner(monkeypatch):
    """`PARTNER_PIN_FRAME=min` is the arm that adds no ladder attempt and
    removes none: the rung-0 corner moves, the expand rungs stay shipped."""
    monkeypatch.setenv("PARTNER_PIN_FRAME", "min")
    mod = importlib.reload(LR)
    try:
        assert mod._PIN_FRAME is True
        assert mod._PIN_FRAME_RUNGS is False
    finally:
        monkeypatch.delenv("PARTNER_PIN_FRAME", raising=False)
        importlib.reload(mod)


def test_retry_flag_defaults_off():
    assert LR._PIN_FRAME_RETRY is False


# ---------------------------------------------------------------------------
# _pin_frame_to_locks
# ---------------------------------------------------------------------------

def test_max_side_lock_caps_the_wall_and_moves_the_area():
    opt, P = _right_tagged()
    r = LR._Refiner(opt, P, 0)
    r._anchor_frame_to_tags()
    assert r.lock_xmax == pytest.approx(100.0)
    assert r.lock_ymax is None
    a0 = (r.xmax - r.xmin) * (r.ymax - r.ymin)
    assert r.xmax == pytest.approx(120.0)          # the shipped frame

    assert r._pin_frame_to_locks() is True
    assert r.xmax == pytest.approx(100.0)          # capped ON the tag line
    # the clamped area went to the side that is still free (y)
    assert (r.xmax - r.xmin) * (r.ymax - r.ymin) == pytest.approx(a0)

    r._pull_inside_frame()
    assert r.legalize(40) is True
    assert _no_overlap(r.P)
    assert float((r.P[:, 0] + r.P[:, 2]).max()) <= 100.0 + 1e-6


def test_min_side_lock_raises_the_corner():
    opt, P = _left_tagged()
    r = LR._Refiner(opt, P, 0)
    r._anchor_frame_to_tags()
    assert r.lock_xmin == pytest.approx(0.0)
    assert r.xmin == pytest.approx(-20.0)          # the shipped frame
    a0 = (r.xmax - r.xmin) * (r.ymax - r.ymin)

    assert r._pin_frame_to_locks() is True
    assert r.xmin == pytest.approx(0.0)
    assert (r.xmax - r.xmin) * (r.ymax - r.ymin) == pytest.approx(a0)


def test_no_locks_is_identity():
    opt, P = _untagged()
    r = LR._Refiner(opt, P, 0)
    r._anchor_frame_to_tags()
    frame = (r.xmin, r.xmax, r.ymin, r.ymax)
    assert r._pin_frame_to_locks() is False
    assert (r.xmin, r.xmax, r.ymin, r.ymax) == frame


def test_degenerate_clamp_is_refused():
    """A clamp that would collapse the frame leaves the rung as shipped."""
    opt, P = _right_tagged()
    r = LR._Refiner(opt, P, 0)
    r._anchor_frame_to_tags()
    r.lock_xmin = r.lock_xmax          # zero-width frame
    frame = (r.xmin, r.xmax, r.ymin, r.ymax)
    assert r._pin_frame_to_locks() is False
    assert (r.xmin, r.xmax, r.ymin, r.ymax) == frame


# ---------------------------------------------------------------------------
# end to end through the ladder
# ---------------------------------------------------------------------------

def _run_ladder(pin: bool, monkeypatch):
    if pin:
        monkeypatch.setenv("PARTNER_PIN_FRAME", "1")
    else:
        monkeypatch.delenv("PARTNER_PIN_FRAME", raising=False)
    mod = importlib.reload(LR)
    try:
        opt, P = _left_tagged()
        out = mod.refine_prediction(opt, P, time.time() + 10.0, seed=0)
        assert out is not None
        return np.asarray(out, dtype=np.float64), opt
    finally:
        monkeypatch.delenv("PARTNER_PIN_FRAME", raising=False)
        importlib.reload(mod)


def test_ladder_pins_the_min_side_wall(monkeypatch):
    Q, opt = _run_ladder(True, monkeypatch)
    assert _no_overlap(Q)
    # the tagged preplaced block is still exactly where the input put it
    assert Q[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert Q[0, 1] == pytest.approx(0.0, abs=1e-6)
    # ... and the layout's left wall is now ITS edge, i.e. the tag is seated
    assert float(Q[:, 0].min()) == pytest.approx(0.0, abs=1e-6)
    for i in range(opt.n):
        assert abs(Q[i, 2] * Q[i, 3] - opt.areas[i]) / opt.areas[i] <= 0.01


def test_off_path_keeps_the_shipped_overshoot(monkeypatch):
    """Same instance with the flag unset: the prediction's own extent still
    defines the wall, so the preplaced tag stays unsatisfied.  This is the
    behaviour the flag is allowed to change and nothing else."""
    Q, _opt = _run_ladder(False, monkeypatch)
    assert _no_overlap(Q)
    assert float(Q[:, 0].min()) < -1e-6
