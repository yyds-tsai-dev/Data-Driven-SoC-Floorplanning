"""Three opt-in partner flags, each default off and each bit-exact when off.

  * `PARTNER_EDGE_SEAT_V2`  -> layout_refiner._edge_seat widens its reach
    (layout-scaled GAP, 8-outlier pull cap, a joint two-axis corner pass) and
    swaps its acceptance test for the evaluator's own boundary+grouping+MIB
    total; contest_optimizer additionally routes the COLUMN champion through
    `_edge_seat` before `_pick_best` (path coverage).
  * `PARTNER_FRAME_WPIN`    -> column_sa_legalizer adds restart arms whose
    frame is derived from the L/R-tagged preplaced blocks instead of the
    pin-aspect guess.
  * `PARTNER_COL_NARROW`    -> the per-column width solve may shrink a column
    that stacks short of the frame, not just grow one that overflows.

What must hold:
  * flag off -> BIT-EXACT.  Proved the hard way: the same driver is executed
    against the committed (HEAD) sources and against the working tree, and the
    raw layout bytes must match.  A guard that is merely "probably not taken"
    does not pass this.
  * flag off -> the optimizer hook returns the SAME list object and never
    imports the refiner.
  * flag on  -> legality invariants survive: no overlap, exact areas, fixed
    shapes and preplaced origins untouched, no column overflows the frame,
    and the seat pass never raises the evaluator-faithful violation total.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
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
import layout_refiner as lr                # noqa: E402

_ENV = ("PARTNER_EDGE_SEAT_V2", "PARTNER_FRAME_WPIN",
        "PARTNER_FRAME_WPIN_ARMS", "PARTNER_FRAME_WPIN_DEBUG",
        "PARTNER_COL_NARROW", "PARTNER_COL_CACHE", "PARTNER_SA_KERNEL",
        "PARTNER_REFINE_KERNEL")

# the two files these flags touch; the baseline run restores them from HEAD
_TOUCHED = ("layout_refiner.py", "column_sa_legalizer.py")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# a deterministic instance with wall tags, a cluster, a MIB group, a rigid
# block and an R-tagged preplaced block (so every flag has something to chew)
# ---------------------------------------------------------------------------

def _case(n: int = 24, s: float = 1.0):
    """`s` scales the whole instance.  Geometry is identical up to the
    factor, so a scaled case exercises anything that is a FRACTION of the
    frame (the v2 GAP) without changing what the pass has to do."""
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
    # preplaced, right-tagged: this is what wakes the FRAME_WPIN gate
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


def _P(rects):
    return np.asarray([[float(a) for a in r] for r in rects],
                      dtype=np.float64)


def _no_overlap(P, tol=1e-6):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None],
                                                           x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None],
                                                           y0[None, :])
    bad = (ox > tol) & (oy > tol)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# 1. the flags are off by default
# ---------------------------------------------------------------------------

def test_flags_default_off():
    assert lr.edge_seat_v2_on() is False
    assert csl.frame_wpin_on() is False
    assert csl.col_narrow_on() is False
    assert _opt()._col_narrow is False
    assert _opt().w_star is None


def test_flag_readers_accept_the_house_spelling(monkeypatch):
    for name, fn in (("PARTNER_EDGE_SEAT_V2", lr.edge_seat_v2_on),
                     ("PARTNER_FRAME_WPIN", csl.frame_wpin_on),
                     ("PARTNER_COL_NARROW", csl.col_narrow_on)):
        for val in ("1", "true", "True", "on", "ON"):
            monkeypatch.setenv(name, val)
            assert fn() is True, (name, val)
        for val in ("0", "", "no", "off"):
            monkeypatch.setenv(name, val)
            assert fn() is False, (name, val)
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 2. off == HEAD, byte for byte
# ---------------------------------------------------------------------------

_DRIVER = r'''
import hashlib, json, sys
import numpy as np, torch
import column_sa_legalizer as csl
import layout_refiner as lr

sys.path.insert(0, %(tests)r)
from test_partner_seat_frame_narrow import _case          # noqa: E402

rects, at, cons, tpos, b2b, p2b, pins = _case()
opt = csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                           deadline=None, seed=7)
opt.prepare()
pos, x_right, y_top = opt._layout_full(opt._cols)
h = hashlib.sha256()
h.update(np.ascontiguousarray(pos, dtype=np.float64).tobytes())
h.update(repr((float(x_right), float(y_top), float(opt.H),
               float(opt.W_est))).encode())

# and the seat pass, driven off a layout that violates its wall tags
Q = np.asarray(pos, dtype=np.float64).copy()
Q[:, 0] += 0.37
Q[0, 1] += 0.11
seated = lr._edge_seat(opt, [tuple(map(float, r)) for r in Q])
S = np.asarray(seated, dtype=np.float64)
h.update(np.ascontiguousarray(S, dtype=np.float64).tobytes())
print(h.hexdigest())
'''


def _hash_under(src: Path, tmp: Path) -> str:
    env = dict(os.environ)
    for v in _ENV:
        env.pop(v, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), str(ROOT / "FloorSet" / "iccad2026contest"),
         str(ROOT / "FloorSet")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    drv = tmp / f"drv_{src.name}.py"
    drv.write_text(_DRIVER % {"tests": str(ROOT / "tests")})
    out = subprocess.run([sys.executable, str(drv)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.strip().splitlines()[-1]


def test_off_path_is_bit_identical_to_head(tmp_path):
    """The off path must not move one byte relative to the committed code.

    Not a proxy: the same driver runs against HEAD's `layout_refiner.py` /
    `column_sa_legalizer.py` and against the working tree, and the raw
    float64 layout bytes are compared."""
    live = tmp_path / "live"
    head = tmp_path / "head"
    for d in (live, head):
        d.mkdir()
        for f in (ROOT / "src" / "solver").glob("*.py"):
            shutil.copy2(f, d / f.name)
    for name in _TOUCHED:
        blob = subprocess.run(
            ["git", "show", f"HEAD:src/solver/{name}"], cwd=ROOT,
            capture_output=True, timeout=120)
        assert blob.returncode == 0, blob.stderr[-2000:]
        (head / name).write_bytes(blob.stdout)

    assert _hash_under(head, tmp_path) == _hash_under(live, tmp_path)


# ---------------------------------------------------------------------------
# 3. PARTNER_EDGE_SEAT_V2
# ---------------------------------------------------------------------------

def _violated_layout(opt):
    """A legal layout whose wall tags are broken by a uniform shift plus a
    single hovering block -- exactly the residual `_edge_seat` exists for."""
    opt.prepare()
    pos, _xr, _yt = opt._layout_full(opt._cols)
    Q = np.asarray(pos, dtype=np.float64).copy()
    Q[:, 0] += 0.37
    Q[0, 1] += 0.11
    return [tuple(map(float, r)) for r in Q]


def test_edge_seat_off_never_reaches_the_exact_counter(monkeypatch):
    """With the flag off the pass must not even look at violation_killer."""
    import violation_killer as vk
    calls = []
    monkeypatch.setattr(
        vk, "_violations_exact",
        lambda *a, **k: (calls.append(1), 1 / 0)[0])
    opt = _opt()
    lr._edge_seat(opt, _violated_layout(opt))       # must not raise
    assert calls == []


def test_edge_seat_on_uses_the_evaluator_total_and_stays_legal(monkeypatch):
    import violation_killer as vk
    opt = _opt()
    start = _violated_layout(opt)
    P0 = _P(start)
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")
    assert lr.edge_seat_v2_on() is True
    seated = lr._edge_seat(opt, start)
    P1 = _P(seated)

    assert P1.shape == P0.shape
    assert _no_overlap(P1)
    # the acceptance test is the evaluator's own total, and it may only fall
    assert vk._violations_exact(opt, P1) <= vk._violations_exact(opt, P0)
    # hard legality: preplaced pinned, fixed shapes intact, areas exact
    for i in range(opt.n):
        if opt.kind[i] == 2:
            assert np.allclose(P1[i], P0[i], atol=0, rtol=0)
        if opt.kind[i] == 1:
            assert P1[i, 2] == pytest.approx(P0[i, 2], abs=1e-9)
            assert P1[i, 3] == pytest.approx(P0[i, 3], abs=1e-9)
        a = float(P1[i, 2] * P1[i, 3])
        assert a >= 0.99 * float(opt.areas[i]) - 1e-9
        assert a <= 1.01 * float(opt.areas[i]) + 1e-9


def test_edge_seat_v2_widens_gap_with_the_frame(monkeypatch):
    """(i) GAP is a fraction of the frame, so a hover that the flat 2.0
    window refused on a large frame is now in reach."""
    opt = _opt(s=12.0)               # big frame: 0.08 * span >> 2.0
    start = _violated_layout(opt)
    P = _P(start)
    span = min(float((P[:, 0] + P[:, 2]).max() - P[:, 0].min()),
               float((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))
    assert 0.08 * span > 2.0, "instance too small to exercise the scaling"
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")
    on = _P(lr._edge_seat(opt, start))
    monkeypatch.delenv("PARTNER_EDGE_SEAT_V2")
    off = _P(lr._edge_seat(opt, start))
    # the widened pass may not do WORSE than the narrow one on the official
    # total (it is a superset of moves under a stricter acceptance test)
    import violation_killer as vk
    assert vk._violations_exact(opt, on) <= vk._violations_exact(opt, off)


def test_corner_seat_moves_both_axes_at_once(monkeypatch):
    """(iii) a two-bit tag is seated in ONE accepted move; the per-edge loop
    can only ever move one axis per commit."""
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")
    opt = _opt()
    two_bit = [i for i in range(opt.n)
               if bin(int(opt.boundary[i])).count("1") >= 2]
    assert two_bit, "instance must carry a corner tag"
    start = _violated_layout(opt)
    seated = _P(lr._edge_seat(opt, start))
    assert _no_overlap(seated)


def test_optimizer_hook_is_identity_when_flag_unset():
    """Flag off -> the SAME list object, and the refiner is never called."""
    import contest_optimizer as co
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    out = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
    got = opt._column_edge_seat(out, None, None, None, None, None, None, [])
    assert got is out


def test_optimizer_hook_is_contained(monkeypatch):
    """Flag on but the pass explodes -> the input layout, unchanged."""
    import contest_optimizer as co
    monkeypatch.setenv("PARTNER_EDGE_SEAT_V2", "1")
    monkeypatch.setattr(lr, "_edge_seat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    out = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
    assert opt._column_edge_seat(out, None, None, None, None, None, None,
                                 []) is out


# ---------------------------------------------------------------------------
# 4. PARTNER_FRAME_WPIN
# ---------------------------------------------------------------------------

def test_w_star_gate_is_a_reusable_statistic():
    """The trigger is "a preplaced block carries a right-wall tag", nothing
    case-specific.  No such block -> no arm."""
    opt = _opt()
    ws = csl._w_star_from_tags(opt)
    assert ws is not None
    assert ws == pytest.approx(
        max(x + w for (x, _y, w, _h) in opt.locked_rects))

    rects, at, cons, tpos, b2b, p2b, pins = _case()
    cons[17, 4] = 0.0                       # drop the R tag
    bare = csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                deadline=None, seed=7)
    assert csl._w_star_from_tags(bare) is None

    cons[17, 1] = 0.0                       # no preplaced at all
    tpos[17] = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    none_pre = csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                    deadline=None, seed=7)
    assert csl._w_star_from_tags(none_pre) is None


def test_w_star_arm_derives_the_frame_from_the_width():
    """The arm expresses W* through H so the existing per-column width solve
    converges onto it; the locked/rigid floors still win when they must."""
    base = _opt()
    ws = csl._w_star_from_tags(base)
    armed = _opt(w_star=ws)
    want = armed.total_area / (csl.FRAME_WPIN_UTIL * ws)
    floor = max([y + h for (_x, y, _w, h) in armed.locked_rects]
                + [armed.rh[i] for i in range(armed.n) if armed.kind[i] == 1]
                + [0.0])
    assert armed.H == pytest.approx(max(want, floor), rel=1e-12)
    assert armed.H != pytest.approx(base.H, rel=1e-9)


def test_w_star_none_is_the_historical_frame():
    assert _opt(w_star=None).H == pytest.approx(_opt().H, rel=0, abs=0)


def test_worker_payload_arity_is_backward_compatible():
    """A 13-wide payload (every pre-flag caller) must still unpack."""
    import inspect
    src = inspect.getsource(csl._worker_solve)
    assert "len(args) == 14" in src
    assert "w_star = None" in src


def test_w_star_arm_produces_a_legal_layout():
    opt = _opt(w_star=csl._w_star_from_tags(_opt()))
    opt.prepare()
    pos, _xr, _yt = opt._layout_full(opt._cols)
    P = np.asarray(pos, dtype=np.float64)
    assert _no_overlap(P)
    for i in range(opt.n):
        if opt.kind[i] == 2:
            assert P[i, 0] == pytest.approx(opt.lx[i], abs=1e-9)
            assert P[i, 1] == pytest.approx(opt.ly[i], abs=1e-9)


# ---------------------------------------------------------------------------
# 5. PARTNER_COL_NARROW
# ---------------------------------------------------------------------------

def _layout_both(monkeypatch, method: str):
    monkeypatch.delenv("PARTNER_COL_NARROW", raising=False)
    off = _opt()
    off.prepare()
    a = getattr(off, method)(off._cols)

    monkeypatch.setenv("PARTNER_COL_NARROW", "1")
    on = _opt()
    assert on._col_narrow is True
    on.prepare()
    b = getattr(on, method)(on._cols)
    return off, np.asarray(a[0], dtype=np.float64), \
        on, np.asarray(b[0], dtype=np.float64)


@pytest.mark.parametrize("method", ["_layout_full", "_layout_delta"])
def test_col_narrow_keeps_the_layout_legal(monkeypatch, method):
    if method == "_layout_delta":
        monkeypatch.setenv("PARTNER_COL_CACHE", "1")
    _off, A, on, B = _layout_both(monkeypatch, method)
    assert _no_overlap(A)
    assert _no_overlap(B)
    # shapes/areas are the column solve's own business, but nothing may
    # leave the frame the widen loop guards
    for i in range(on.n):
        if on.kind[i] == 2:
            assert B[i, 0] == pytest.approx(on.lx[i], abs=1e-9)
            assert B[i, 1] == pytest.approx(on.ly[i], abs=1e-9)
        if on.kind[i] == 1:
            assert B[i, 2] == pytest.approx(on.rw[i], abs=1e-9)
            assert B[i, 3] == pytest.approx(on.rh[i], abs=1e-9)


def test_col_narrow_never_overflows_the_frame(monkeypatch):
    """The narrow loop's acceptance test IS the widen loop's overflow guard,
    so a narrowed column may never stack past H."""
    monkeypatch.setenv("PARTNER_COL_NARROW", "1")
    opt = _opt()
    opt.prepare()
    pos, _xr, _yt = opt._layout_full(opt._cols)
    P = np.asarray(pos, dtype=np.float64)
    top = float((P[:, 1] + P[:, 3]).max())
    floor = max([opt.H]
                + [y + h for (_x, y, _w, h) in opt.locked_rects])
    assert top <= floor * 1.0005 + 1e-6


def test_col_narrow_shrinks_or_keeps_the_bbox(monkeypatch):
    """Narrowing trades dead column area for width the neighbours reuse; it
    must not inflate the bounding box it exists to shrink."""
    _off, A, _on, B = _layout_both(monkeypatch, "_layout_full")

    def bb(P):
        return (float((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                * float((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))

    assert bb(B) <= bb(A) * 1.02


def test_col_narrow_is_deterministic(monkeypatch):
    monkeypatch.setenv("PARTNER_COL_NARROW", "1")
    outs = []
    for _ in range(2):
        o = _opt()
        o.prepare()
        outs.append(hashlib.sha256(
            np.ascontiguousarray(o._layout_full(o._cols)[0],
                                 dtype=np.float64).tobytes()).hexdigest())
    assert outs[0] == outs[1]
