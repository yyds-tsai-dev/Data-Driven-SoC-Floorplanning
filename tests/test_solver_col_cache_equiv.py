"""Bit-exact equivalence for the partner column-cache delta-evaluation.

PARTNER_COL_CACHE=1 routes _layout -> _layout_delta, which reuses the cached
per-column geometry for the *unchanged left prefix* of columns and never
translates stored coordinates (reuse only when the column's x is bit-identical).
It must therefore be BIT-EXACT with the from-scratch _layout_full -- np.array_equal
on positions and exact equality on cost / frame extents, not merely allclose.

The annealer is wall-clock-bounded, so two independent runs never follow the
same move sequence; we cannot compare full run() outputs. Instead this is an
in-run shadow check: one optimizer drives a fixed-seed random-move stream and
after every move we recompose positions via BOTH paths on the identical column
state and assert bit-exact agreement. Cases carry golden-bbox preplaced blocks
(case 99: preplaced + MIB + cluster + boundary) so the locked-obstacle path --
the cache-invalidation trap -- is genuinely exercised.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as lg  # noqa: E402

try:
    from lite_dataset_test import FloorplanDatasetLiteTest
    _DS = FloorplanDatasetLiteTest(str(ROOT / "FloorSet"))
except Exception as exc:  # pragma: no cover - dataset absence
    _DS = None
    _DS_ERR = str(exc)

# case 99: n=120, 6 preplaced + 10 fixed + 7 MIB + 28 cluster + 36 boundary
# (richest constraint mix, obstacle-heavy). case 50 / case 3: 1 locked each.
CASES = [(99, "n120-preplaced-mib"), (50, "n71-1locked"), (3, "n24-1locked")]
SEED = 20260723


def _n_of(sample) -> int:
    return int((sample["input"][0] != -1).sum().item())


def _golden_targets(sample) -> torch.Tensor:
    at, _b2b, _p2b, _pins, cons = sample["input"]
    polys = sample["label"][0]
    n = _n_of(sample)
    tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        isf = nc > 0 and cons[i, 0] != 0
        isp = nc > 1 and cons[i, 1] != 0
        if not (isf or isp):
            continue
        block = polys[i]
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            continue
        mn = valid.min(dim=0).values
        mx = valid.max(dim=0).values
        gx, gy = float(mn[0]), float(mn[1])
        gw, gh = float(mx[0] - mn[0]), float(mx[1] - mn[1])
        if isp:
            tpos[i] = torch.tensor([gx, gy, gw, gh])
        elif isf:
            tpos[i, 2] = gw
            tpos[i, 3] = gh
    return tpos


def _build(idx: int) -> Optional[lg._ColumnOptimizer]:
    sample = _DS[idx]
    at, b2b, p2b, pins, cons = sample["input"]
    n = _n_of(sample)
    at = at[:n].float()
    cons = cons[:n].float()
    tpos = _golden_targets(sample)
    seeds = [(0.0, 0.0, math.sqrt(max(float(at[i]), 1e-9)),
              math.sqrt(max(float(at[i]), 1e-9))) for i in range(n)]
    opt = lg._ColumnOptimizer(seeds, at, cons, tpos, b2b.float(), p2b.float(),
                              pins.float(), time.time() + 5.0, seed=SEED)
    if opt.locked_only():
        return None
    opt.prepare()
    return opt


def _both_layouts(opt, cols):
    fast_cols = [list(c) for c in cols]
    slow_cols = [list(c) for c in cols]
    opt._dc_enabled = True
    pf, xrf, ytf = opt._layout_delta(fast_cols)
    cf, hf, af, vf = opt._cost(pf, xrf, ytf)
    ps, xrs, yts = opt._layout_full(slow_cols)
    cs_, hs, as_, vs = opt._cost(ps, xrs, yts)
    return (pf, cf, vf, xrf, ytf), (ps, cs_, vs, xrs, yts)


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
@pytest.mark.parametrize("idx,label", CASES)
def test_col_cache_bit_exact(idx, label):
    opt = _build(idx)
    if opt is None:
        pytest.skip(f"case {idx} is locked-only")
    cols = [list(c) for c in opt._cols]

    (pf, cf, vf, xrf, ytf), (ps, cs_, vs, xrs, yts) = _both_layouts(opt, cols)
    assert np.array_equal(pf, ps), f"{label}: initial pos not bit-exact"
    assert cf == cs_ and vf == vs, f"{label}: initial cost mismatch"

    n_checked = 0
    for _step in range(600):
        undo = opt._random_move(cols)
        if undo is None:
            continue
        (pf, cf, vf, xrf, ytf), (ps, cs_, vs, xrs, yts) = _both_layouts(opt, cols)
        assert np.array_equal(pf, ps), (
            f"{label}: pos not bit-exact after move "
            f"(max|d|={np.abs(pf - ps).max():.3e}, step {_step})")
        assert cf == cs_, f"{label}: cost not bit-exact ({cf!r} != {cs_!r})"
        assert vf == vs, f"{label}: violation count mismatch ({vf} != {vs})"
        assert xrf == xrs and ytf == yts, f"{label}: frame extent mismatch"
        n_checked += 1
        # accept every move: drive the trajectory deep into non-trivial states

    assert n_checked > 100, f"{label}: too few moves exercised ({n_checked})"
    # the cache must actually have engaged (otherwise the test is vacuous)
    assert opt._dc_hits > 0, f"{label}: column cache never hit ({opt._dc_hits})"


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_col_cache_invalidates_on_subgroup_reorder():
    """A subgroup-reorder move mutates a unit's internal order without changing
    column membership; the per-column signature must catch it so the stale
    cached geometry is not reused."""
    opt = _build(99)
    assert opt is not None
    # find a multi-part (reorderable) unit and the column holding it
    mutable = [k for k in opt._dc_mutable]
    if not mutable:
        pytest.skip("case 99 has no reorderable units")
    cols = [list(c) for c in opt._cols]
    # prime the cache
    opt._dc_enabled = True
    opt._layout_delta([list(c) for c in cols])

    k = mutable[0]
    u = opt.units[k]
    # force a subgroup reorder equivalent to the SA reorder move
    if len(u.subgroups) >= 2:
        u.subgroups[0], u.subgroups[1] = u.subgroups[1], u.subgroups[0]
    else:
        sg = next(sg for sg in u.subgroups if len(sg) >= 2)
        sg[0], sg[1] = sg[1], sg[0]
    opt._refresh_unit(u)

    pf, xrf, ytf = opt._layout_delta([list(c) for c in cols])
    ps, xrs, yts = opt._layout_full([list(c) for c in cols])
    assert np.array_equal(pf, ps), (
        f"stale cache reused after subgroup reorder (max|d|={np.abs(pf - ps).max():.3e})")


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_col_cache_off_matches_full():
    """With the cache disabled, _layout must be identical to _layout_full
    (the dispatcher must not alter the default path)."""
    opt = _build(50)
    assert opt is not None
    opt._dc_enabled = False
    cols = [list(c) for c in opt._cols]
    p1, xr1, yt1 = opt._layout([list(c) for c in cols])
    p2, xr2, yt2 = opt._layout_full([list(c) for c in cols])
    assert np.array_equal(p1, p2) and xr1 == xr2 and yt1 == yt2
