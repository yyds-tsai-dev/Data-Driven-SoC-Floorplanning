"""`PARTNER_COL_BALANCE` (default off): exact-DP free-unit column assignment
in `column_sa_legalizer._ColumnOptimizer._init_columns`, replacing the greedy
monotone-cursor + equal-area-capacity loop that can strand a column empty
while overloading the last one.

What must hold:
  * flag off -> `_init_columns` output is untouched, and the DP helper
    (`_col_balance_partition`) is never invoked (spied).
  * flag on  -> every free unit is assigned exactly once, forced units stay
    in their forced columns, and every column's rigid usage is within
    `rigid_cap` (or the DP declined and the legacy loop ran instead).
  * the DP is optimal against brute force for small instances.
  * the DP's realized total width never exceeds the legacy loop's.
  * the DP runs well under 5 ms at m=120 free units, C=10 columns.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl          # noqa: E402

_ENV = ("PARTNER_COL_BALANCE", "PARTNER_COL_BALANCE_ARMS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# synthetic instances (reusing the shape of test_partner_seat_frame_narrow's
# `_case` / `_opt` builders, varied so free-unit sizing differs per case)
# ---------------------------------------------------------------------------

def _case(n: int = 24, s: float = 1.0, rigid_every: int = 4):
    rects = []
    for i in range(n):
        rects.append((6.0 * s * (i % 6), 6.0 * s * (i // 6), 5.0 * s, 5.0 * s))
    at = torch.full((n,), 25.0 * s * s)
    cons = torch.zeros((n, 5))

    for i in range(0, n, rigid_every):
        cons[i, 0] = 1.0                  # fixed shape (rigid)
    for i in (5, 6, 7):
        if i < n:
            cons[i, 2] = 1.0               # MIB group
    if n > 11:
        cons[10, 3] = 1.0                  # cluster
        cons[11, 3] = 1.0

    if n > 0:
        cons[0, 4] = 9.0                   # left + bottom -> corner tag
    if n > 5:
        cons[5, 4] = 6.0                   # right + top -> corner tag
    if n > 13:
        cons[12, 4] = 1.0                  # left
        cons[13, 4] = 4.0                  # top

    tpos = torch.full((n, 4), -1.0)
    if n > 17:
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


def _opt(n=24, s=1.0, rigid_every=4, **kw):
    rects, at, cons, tpos, b2b, p2b, pins = _case(n=n, s=s, rigid_every=rigid_every)
    return csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                deadline=None, seed=7, **kw)


_INSTANCES = [
    dict(n=24, s=1.0, rigid_every=4),
    dict(n=24, s=1.0, rigid_every=8),
    dict(n=30, s=0.7, rigid_every=5),
    dict(n=21, s=1.3, rigid_every=3),
    dict(n=36, s=1.0, rigid_every=6),
    dict(n=24, s=2.0, rigid_every=4),
    dict(n=42, s=0.5, rigid_every=7),
    dict(n=24, s=1.0, rigid_every=2),
    dict(n=27, s=1.1, rigid_every=9),
    dict(n=48, s=0.9, rigid_every=6),
]


def _realized_width_sum(opt, cols):
    """Same width formula `_layout_full` uses per column."""
    total = 0.0
    for ulist in cols:
        if not ulist:
            continue
        soft_a = sum(opt.units[k].eff_soft for k in ulist)
        rigid_h = sum(opt.units[k].eff_rigid_h for k in ulist)
        max_w = max((opt.units[k].max_rigid_w for k in ulist), default=0.0)
        avail = max(opt.H - rigid_h, 0.05 * opt.H)
        w = max(soft_a / avail, max_w, 0.5)
        total += w
    return total


# ---------------------------------------------------------------------------
# 0. default off, house spelling
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert csl.col_balance_on() is False


def test_flag_readers_accept_the_house_spelling(monkeypatch):
    for val in ("1", "true", "True", "on", "ON"):
        monkeypatch.setenv("PARTNER_COL_BALANCE", val)
        assert csl.col_balance_on() is True, val
    for val in ("0", "", "no", "off"):
        monkeypatch.setenv("PARTNER_COL_BALANCE", val)
        assert csl.col_balance_on() is False, val


# ---------------------------------------------------------------------------
# 1. flag off -> byte-identical `_init_columns`, DP helper never called
# ---------------------------------------------------------------------------

def test_off_path_is_untouched_and_dp_never_called():
    for kw in _INSTANCES:
        opt = _opt(**kw)
        C = opt.C0 if hasattr(opt, "C0") and opt.C0 else max(2, min(18, 4))
        C = 5
        expected = opt._init_columns(C)
        with mock.patch.object(csl, "_col_balance_partition") as spy:
            got = opt._init_columns(C)
        spy.assert_not_called()
        assert got == expected


# ---------------------------------------------------------------------------
# 2. flag on -> valid partition
# ---------------------------------------------------------------------------

def test_on_path_partition_is_valid(monkeypatch):
    monkeypatch.setenv("PARTNER_COL_BALANCE", "1")
    for kw in _INSTANCES:
        opt = _opt(**kw)
        C = 5
        legacy_forced_cols = {}
        for k, u in enumerate(opt.units):
            if u.force == 'L':
                legacy_forced_cols[k] = 0
            elif u.force == 'R':
                legacy_forced_cols[k] = C - 1
            elif u.anchors:
                ax = u.anchors[0][0] + u.anchors[0][2] * 0.5
                legacy_forced_cols[k] = max(0, min(C - 1, int(ax / max(opt.W_est, 1e-6) * C)))

        cols = opt._init_columns(C)
        rigid_cap = 0.55 * opt.H

        seen = set()
        for ci, ulist in enumerate(cols):
            for k in ulist:
                assert k not in seen, "unit assigned twice"
                seen.add(k)
        assert seen == set(range(len(opt.units)))

        # forced units land in their forced column
        col_of = {}
        for ci, ulist in enumerate(cols):
            for k in ulist:
                col_of[k] = ci
        for k, ci in legacy_forced_cols.items():
            assert col_of[k] == ci

        # rigid budget respected per column (legacy fallback guarantees this
        # too, since the DP only commits when every column is feasible)
        for ci, ulist in enumerate(cols):
            rigid_h = sum(opt.units[k].eff_rigid_h for k in ulist)
            assert rigid_h <= rigid_cap + 1e-6


# ---------------------------------------------------------------------------
# 3. DP optimality vs brute force, small instances
# ---------------------------------------------------------------------------

def _brute_force_partition(units, free, C, H, rigid_cap, forced_soft, forced_rigid, forced_maxw):
    m = len(free)
    best_cost = float("inf")
    best_groups = None
    # all ways to choose C-1 cut points in [0, m] (with repeats -> empty groups)
    for cuts in itertools.combinations_with_replacement(range(m + 1), C - 1):
        bounds = [0] + list(cuts) + [m]
        if any(bounds[i] > bounds[i + 1] for i in range(len(bounds) - 1)):
            continue
        cost = 0.0
        feasible = True
        for c in range(C):
            i, j = bounds[c], bounds[c + 1]
            grp = free[i:j]
            A = forced_soft[c] + sum(units[k].eff_soft for k in grp)
            R = forced_rigid[c] + sum(units[k].eff_rigid_h for k in grp)
            MW = max([forced_maxw[c]] + [units[k].max_rigid_w for k in grp])
            if R > rigid_cap + 1e-9:
                feasible = False
                break
            avail = max(H - R, 1e-9)
            cost += max(A / avail, MW)
        if feasible and cost < best_cost:
            best_cost = cost
            best_groups = [free[bounds[c]:bounds[c + 1]] for c in range(C)]
    return best_groups, best_cost


def _rand_units(rng, m):
    return [SimpleNamespace(eff_soft=float(rng.uniform(0.5, 5.0)),
                            eff_rigid_h=float(rng.uniform(0.0, 2.0)),
                            max_rigid_w=float(rng.uniform(0.0, 1.5)))
            for _ in range(m)]


def _cost_of(units, free, groups, H, rigid_cap, forced_soft, forced_rigid, forced_maxw):
    total = 0.0
    for c, grp in enumerate(groups):
        A = forced_soft[c] + sum(units[k].eff_soft for k in grp)
        R = forced_rigid[c] + sum(units[k].eff_rigid_h for k in grp)
        MW = max([forced_maxw[c]] + [units[k].max_rigid_w for k in grp])
        if R > rigid_cap + 1e-9:
            return float("inf")
        avail = max(H - R, 1e-9)
        total += max(A / avail, MW)
    return total


def test_dp_matches_brute_force():
    rng = np.random.RandomState(0)
    H = 12.0
    rigid_cap = 0.55 * H
    for trial in range(12):
        m = rng.randint(0, 9)
        C = rng.randint(1, 5)
        units = _rand_units(rng, m)
        free = list(range(m))
        forced_soft = [0.0] * C
        forced_rigid = [0.0] * C
        forced_maxw = [0.0] * C
        dp_groups = csl._col_balance_partition(
            units, free, C, H, rigid_cap, forced_soft, forced_rigid, forced_maxw)
        bf_groups, bf_cost = _brute_force_partition(
            units, free, C, H, rigid_cap, forced_soft, forced_rigid, forced_maxw)
        if bf_groups is None:
            assert dp_groups is None
            continue
        assert dp_groups is not None
        dp_cost = _cost_of(units, free, dp_groups, H, rigid_cap,
                           forced_soft, forced_rigid, forced_maxw)
        assert dp_cost == pytest.approx(bf_cost, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. DP realized width <= legacy realized width
# ---------------------------------------------------------------------------

def test_dp_width_no_worse_than_legacy(monkeypatch):
    for kw in _INSTANCES:
        opt_off = _opt(**kw)
        C = 5
        legacy_cols = opt_off._init_columns(C)
        legacy_w = _realized_width_sum(opt_off, legacy_cols)

        opt_on = _opt(**kw)
        monkeypatch.setenv("PARTNER_COL_BALANCE", "1")
        dp_cols = opt_on._init_columns(C)
        monkeypatch.delenv("PARTNER_COL_BALANCE", raising=False)
        dp_w = _realized_width_sum(opt_on, dp_cols)

        assert dp_w <= legacy_w + 1e-6, kw


# ---------------------------------------------------------------------------
# 5. runtime at n=120, C=10
# ---------------------------------------------------------------------------

def test_dp_runtime_budget():
    rng = np.random.RandomState(1)
    H = 40.0
    rigid_cap = 0.55 * H
    m, C = 120, 10
    units = _rand_units(rng, m)
    free = list(range(m))
    forced_soft = [0.0] * C
    forced_rigid = [0.0] * C
    forced_maxw = [0.0] * C
    # warm up (first numpy call pays import/allocator overhead)
    csl._col_balance_partition(units, free, C, H, rigid_cap,
                               forced_soft, forced_rigid, forced_maxw)
    t0 = time.perf_counter()
    for _ in range(20):
        csl._col_balance_partition(units, free, C, H, rigid_cap,
                                   forced_soft, forced_rigid, forced_maxw)
    dt = (time.perf_counter() - t0) / 20
    assert dt < 5e-3, f"DP took {dt * 1e3:.3f} ms, budget 5 ms"
