"""`PARTNER_FRAME_CL` (default off): closed-loop H targeting for W*-arms.

`_choose_frame` derives H from W* via H = total_area / (0.96 * W*), where
0.96 is an OPEN-LOOP utilisation guess -- realised utilisation runs
0.878-0.971, so the realised right edge (the sum of column widths) misses
W* by 0.1-5.3%.  Column width w_c = soft_area_c / (H - rigid_h_c -
obstacles_c) is monotonically decreasing in H, so a bounded secant on H
(plus a small hard snap redistributing any residual across columns)
closes the loop.  The loop is a property of W*-arms only
(`self.w_star is not None`); every other restart arm is untouched.

`PARTNER_COL_NARROW_PINNED` (default off) is the same narrow retry as
`PARTNER_COL_NARROW`, scoped to arms with a pinned frame dimension
(`self.w_star is not None`).

What must hold:
  * flag off -> the closed-loop helper never runs (spied) and `_layout_full`
    is deterministic across invocations.
  * flag on, W*-arm with realised utilisation far from 0.96 -> the loop
    closes the gap between the realised right edge and W* to within
    tolerance in <= 2 extra evaluations.
  * flag on, over-constrained instance -> reverts to the original-H layout,
    byte-identical to the flag-off output.
  * flag-on outputs stay hard-legal: no overlaps, exact soft areas, kind==2
    rects untouched, MIB dims unchanged.
  * `PARTNER_COL_NARROW_PINNED` only fires on a pinned (w_star) arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl          # noqa: E402

_ENV = ("PARTNER_FRAME_CL", "PARTNER_FRAME_CL_ITERS", "PARTNER_FRAME_CL_TOL",
         "PARTNER_FRAME_CL_DAMP", "PARTNER_FRAME_WPIN", "PARTNER_COL_NARROW",
         "PARTNER_COL_NARROW_PINNED", "PARTNER_COL_CACHE", "PARTNER_SA_KERNEL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# fixture: preplaced, right-tagged (the W*-arm trigger), copied from
# test_solver_seat_frame_narrow._case
# ---------------------------------------------------------------------------

def _case(n: int = 24, s: float = 1.0, util: float = 5.0 * 5.0):
    rects = []
    for i in range(n):
        rects.append((6.0 * s * (i % 6), 6.0 * s * (i // 6), 5.0 * s, 5.0 * s))
    at = torch.full((n,), 25.0 * s * s)
    cons = torch.zeros((n, 5))

    cons[3, 0] = 1.0                      # fixed shape (rigid)
    for i in (5, 6, 7):
        cons[i, 2] = 1.0                  # MIB group
    cons[10, 3] = 1.0                     # cluster
    cons[11, 3] = 1.0

    cons[0, 4] = 9.0                      # left + bottom  -> corner tag
    cons[5, 4] = 6.0                      # right + top    -> corner tag
    cons[12, 4] = 1.0                     # left
    cons[13, 4] = 4.0                     # top

    tpos = torch.full((n, 4), -1.0)
    tpos[3, 2] = 5.0 * s
    tpos[3, 3] = 5.0 * s
    # preplaced, right-tagged: wakes the W*-arm
    pre = 17
    cons[pre, 1] = 1.0
    cons[pre, 4] = 2.0
    tpos[pre] = torch.tensor([30.0 * s, 12.0 * s, 5.0 * s, 5.0 * s])
    rects[pre] = (30.0 * s, 12.0 * s, 5.0 * s, 5.0 * s)

    edges = [[float(i), float((i * 5 + 3) % n), 1.0] for i in range(0, n, 2)]
    b2b = torch.tensor(edges, dtype=torch.float32)
    pins = torch.tensor([[0.0, 0.0], [36.0 * s, 24.0 * s]],
                        dtype=torch.float32)
    p2b = torch.tensor([[0.0, 0.0, 2.0], [1.0, float(n - 1), 2.0]],
                       dtype=torch.float32)
    return rects, at, cons, tpos, b2b, p2b, pins


def _opt(s: float = 1.0, **kw):
    rects, at, cons, tpos, b2b, p2b, pins = _case(s=s)
    return csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                deadline=None, seed=7, **kw)


def _w_opt(s: float = 1.0, **kw):
    """A W*-arm optimizer: w_star derived from the R-tagged preplaced block."""
    base = _opt(s=s)
    ws = csl._w_star_from_tags(base)
    assert ws is not None
    return _opt(s=s, w_star=ws, **kw), ws


def _P(pos):
    return np.asarray(pos, dtype=np.float64)


def _no_overlap(P, tol=1e-6):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > tol) & (oy > tol)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# 1. off path
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert csl.frame_cl_on() is False
    assert csl.col_narrow_pinned_on() is False


def test_off_path_never_runs_the_closed_loop(monkeypatch):
    opt, _ws = _w_opt()
    opt.prepare()
    calls = []
    orig = opt._frame_closed_loop

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(opt, "_frame_closed_loop", spy)
    opt._layout_full(opt._cols)
    assert calls == []


def test_off_path_is_deterministic():
    opt, _ws = _w_opt()
    opt.prepare()
    a = opt._layout_full(opt._cols)
    b = opt._layout_full(opt._cols)
    assert np.array_equal(_P(a[0]), _P(b[0]))
    assert a[1] == b[1] and a[2] == b[2]


# ---------------------------------------------------------------------------
# 2. convergence
# ---------------------------------------------------------------------------

def test_closed_loop_converges_toward_w_star(monkeypatch):
    """FRAME_WPIN_UTIL=0.96 is a guess; this instance's realised utilisation
    is far from it, so the flag-off realised width misses W* -- and the
    flag-on loop must close most of that gap within tolerance."""
    off, ws = _w_opt()
    off.prepare()
    _pos, x_right_off, _yt = off._layout_full(off._cols)
    gap_off = abs(x_right_off - ws)
    assert gap_off > 3e-4 * ws, "fixture does not exercise the open-loop miss"

    monkeypatch.setenv("PARTNER_FRAME_CL", "1")
    on, ws2 = _w_opt()
    assert ws2 == pytest.approx(ws)
    on.prepare()
    pos, x_right_on, _yt2 = on._layout_full(on._cols)

    tol = 3e-4
    snap_tol = csl.FRAME_CL_SNAP_REL
    assert abs(x_right_on - ws) <= max(tol * ws, snap_tol * ws) + 1e-6
    assert abs(x_right_on - ws) < gap_off


def test_closed_loop_uses_at_most_two_extra_evaluations(monkeypatch):
    monkeypatch.setenv("PARTNER_FRAME_CL", "1")
    opt, _ws = _w_opt()
    opt.prepare()
    calls = []
    orig = opt._layout_full_core

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(opt, "_layout_full_core", spy)
    opt._layout_full(opt._cols)
    # 1 base pass + up to FRAME_CL_ITERS (default 2) extra
    assert 1 <= len(calls) <= 1 + csl.FRAME_CL_ITERS_DEFAULT


# ---------------------------------------------------------------------------
# 3. non-convergence revert
# ---------------------------------------------------------------------------

def test_non_convergence_reverts_to_original_h_layout(monkeypatch):
    """DAMP=0 makes every secant step a no-op (H * r**0 == H unchanged), so
    the loop can never converge in <= 2 steps regardless of the fixture --
    the deterministic way to force the over-constrained / non-convergent
    branch without depending on how well a particular instance happens to
    converge.  The loop must revert to exactly the original-H (flag-off)
    layout."""
    off, ws = _w_opt()
    off.prepare()
    want = off._layout_full(off._cols)

    monkeypatch.setenv("PARTNER_FRAME_CL", "1")
    monkeypatch.setenv("PARTNER_FRAME_CL_TOL", "1e-30")
    monkeypatch.setenv("PARTNER_FRAME_CL_DAMP", "0")
    on, ws2 = _w_opt()
    on.prepare()
    got = on._layout_full(on._cols)

    assert np.array_equal(_P(got[0]), _P(want[0]))
    assert got[1] == pytest.approx(want[1], rel=0, abs=0)
    assert got[2] == pytest.approx(want[2], rel=0, abs=0)


# ---------------------------------------------------------------------------
# 4. hard-legality invariants on all flag-on outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("iters", [1, 2, 3])
def test_flag_on_output_is_hard_legal(monkeypatch, iters):
    monkeypatch.setenv("PARTNER_FRAME_CL", "1")
    monkeypatch.setenv("PARTNER_FRAME_CL_ITERS", str(iters))
    opt, _ws = _w_opt()
    opt.prepare()
    pos, x_right, y_top = opt._layout_full(opt._cols)
    P = _P(pos)

    assert _no_overlap(P)

    for i in range(opt.n):
        if opt.kind[i] == 2:
            assert P[i, 0] == pytest.approx(opt.lx[i], abs=1e-9)
            assert P[i, 1] == pytest.approx(opt.ly[i], abs=1e-9)
            assert P[i, 2] == pytest.approx(opt.rw[i], abs=1e-9)
            assert P[i, 3] == pytest.approx(opt.rh[i], abs=1e-9)
        if opt.kind[i] == 1:
            assert P[i, 2] == pytest.approx(opt.rw[i], abs=1e-9)
            assert P[i, 3] == pytest.approx(opt.rh[i], abs=1e-9)
        if opt.kind[i] == 0:
            a = float(P[i, 2] * P[i, 3])
            assert a == pytest.approx(opt.areas[i], rel=1e-6)

    # MIB dims unchanged across the group
    mib_idxs = [i for i in (5, 6, 7)]
    for i in mib_idxs:
        assert P[i, 2] == pytest.approx(P[mib_idxs[0], 2], rel=1e-6)
        assert P[i, 3] == pytest.approx(P[mib_idxs[0], 3], rel=1e-6)

    assert x_right > 0.0
    assert y_top > 0.0


# ---------------------------------------------------------------------------
# 5. PARTNER_COL_NARROW_PINNED gating
# ---------------------------------------------------------------------------

def test_narrow_pinned_runs_only_on_a_pinned_arm(monkeypatch):
    monkeypatch.setenv("PARTNER_COL_NARROW_PINNED", "1")

    pinned, _ws = _w_opt()
    assert pinned.w_star is not None
    assert pinned._col_narrow is True

    unpinned = _opt()
    assert unpinned.w_star is None
    assert unpinned._col_narrow is False


def test_narrow_pinned_alone_does_not_enable_narrow_globally(monkeypatch):
    monkeypatch.setenv("PARTNER_COL_NARROW_PINNED", "1")
    opt = _opt()  # no w_star
    assert csl.col_narrow_on() is False
    assert opt._col_narrow is False


def test_col_narrow_alone_is_unaffected_by_pinned_flag():
    """PARTNER_COL_NARROW=1 alone must behave exactly as before, regardless
    of PARTNER_COL_NARROW_PINNED."""
    import os
    os.environ["PARTNER_COL_NARROW"] = "1"
    try:
        with_pinned_off = _opt()._col_narrow
        os.environ["PARTNER_COL_NARROW_PINNED"] = "1"
        with_pinned_on = _opt()._col_narrow
    finally:
        os.environ.pop("PARTNER_COL_NARROW", None)
        os.environ.pop("PARTNER_COL_NARROW_PINNED", None)
    assert with_pinned_off is True
    assert with_pinned_on is True


def test_both_flags_unset_no_behavior_change():
    opt = _opt()
    assert opt._col_narrow is False
    w_opt, _ws = _w_opt()
    assert w_opt._col_narrow is False
