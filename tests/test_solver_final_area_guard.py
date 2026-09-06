"""Last-line area guard in src/solver/contest_optimizer.py (2026-08-26)."""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "partner") not in sys.path:
    sys.path.insert(0, str(REPO / "partner"))

import contest_optimizer as co  # noqa: E402


def _layout(areas, bad=None):
    out = []
    for i, a in enumerate(areas):
        w = float(np.sqrt(a))
        h = a / w
        if bad is not None and i == bad:
            w, h = 24.0, 13.0
        out.append((10.0 * i, 0.0, w, h))
    return out


def test_area_ok_exact_and_tolerance():
    at = np.array([650.0, 200.0, 312.0])
    assert co._area_ok(_layout(at), at, 3)
    # 0.9% off -> ok, 1.5% off -> fail
    lay = _layout(at)
    x, y, w, h = lay[0]
    assert co._area_ok([(x, y, w * 1.009, h)] + lay[1:], at, 3)
    assert not co._area_ok([(x, y, w * 1.015, h)] + lay[1:], at, 3)
    assert not co._area_ok(lay[:2], at, 3)


def test_guard_returns_same_object_when_clean(monkeypatch):
    monkeypatch.delenv("PARTNER_FINAL_AREA_GUARD", raising=False)
    at = [650.0, 200.0, 312.0]
    res = _layout(at)
    assert co._final_area_guard(res, (None, _layout(at)), at, 3) is res


def test_guard_falls_back_on_area_violation(monkeypatch):
    monkeypatch.delenv("PARTNER_FINAL_AREA_GUARD", raising=False)
    at = [650.0, 200.0, 312.0]
    bad = _layout(at, bad=0)          # block 0 shipped as 24x13 = 312
    post_pick = _layout(at, bad=0)    # first fallback also bad
    column = _layout(at)              # exact by construction
    got = co._final_area_guard(bad, (post_pick, column), at, 3)
    assert got == [tuple(map(float, r)) for r in column]
    # no fallback passes -> original returned unchanged
    assert co._final_area_guard(bad, (post_pick, None), at, 3) is bad


def test_guard_kill_switch(monkeypatch):
    monkeypatch.setenv("PARTNER_FINAL_AREA_GUARD", "0")
    at = [650.0, 200.0, 312.0]
    bad = _layout(at, bad=0)
    assert co._final_area_guard(bad, (_layout(at),), at, 3) is bad
