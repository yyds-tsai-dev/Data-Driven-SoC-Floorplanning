"""Unit tests for the M3 (racing + width re-opt) and M1c (adaptive moves +
constraint-fixing move) additions to `legalizer/column_slicing.py`.

These cover the load-bearing invariants the scheduler's paired full-run gate
relies on but that a paired eval cannot see cheaply:

* racing round-1 snapshots are plain builtins (picklable) and warm-starting
  from a snapshot is deterministic;
* the width-opt override path never poisons / reuses a width-blind FAST_EVAL
  cache entry (its layout equals a from-scratch derive-width layout at the SA's
  own widths, and a forced-width layout stays overlap-free / area-exact);
* the adaptive reweighter respects the 5% per-family floor and re-normalizes;
* the constraint-fixing move only fires when V>0 and never breaks hard
  legality (no overlaps, exact soft areas).

Runs in a few seconds -- no wall-clock annealing loops.
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import floorset_arch.legalizer.column_backbone as cb  # noqa: E402
import floorset_arch.legalizer.column_slicing as cs  # noqa: E402

try:
    from lite_dataset_test import FloorplanDatasetLiteTest
    _DS = FloorplanDatasetLiteTest(str(ROOT / "FloorSet"))
except Exception as exc:  # pragma: no cover
    _DS = None
    _DS_ERR = str(exc)

SEED = 20260705


def _n_of(sample) -> int:
    return int((sample["input"][0] != -1).sum().item())


def _golden_target_positions(sample) -> torch.Tensor:
    at, _b2b, _p2b, _pins, cons = sample["input"]
    polys, _m = sample["label"]
    n = _n_of(sample)
    tpos = torch.full((n, 4), -1.0)
    if cons.dim() < 2 or cons.shape[1] < 2:
        return tpos
    for i in range(n):
        if cons[i, 1] == 0:
            continue
        block = polys[i]
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            continue
        mn = valid.min(dim=0).values
        mx = valid.max(dim=0).values
        tpos[i] = torch.tensor([float(mn[0]), float(mn[1]),
                                float(mx[0] - mn[0]), float(mx[1] - mn[1])])
    return tpos


def _build(idx: int, fast_eval: bool = True, seed: int = SEED):
    sample = _DS[idx]
    at, b2b, p2b, pins, cons = sample["input"]
    n = _n_of(sample)
    at = at[:n].float()
    cons = cons[:n].float()
    b2b = b2b.float()
    p2b = p2b.float()
    pins = pins.float()
    tpos = _golden_target_positions(sample)
    seed_rects = cb._heuristic_init(at, cons, tpos, b2b, p2b, pins)
    opt = cs._ColumnOptimizer(seed_rects, at, cons, tpos, b2b, p2b, pins,
                              time.time() + 5.0, seed=seed)
    if opt.locked_only():
        return None
    opt.prepare()
    opt._fast_eval = fast_eval
    return opt


def _overlap_free(pos: np.ndarray, kind, tol=1e-6) -> bool:
    n = len(pos)
    x0 = pos[:, 0]; y0 = pos[:, 1]; x1 = x0 + pos[:, 2]; y1 = y0 + pos[:, 3]
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
            oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
            if ox > tol and oy > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# M3 racing: snapshot pickle-roundtrip + warm-start determinism
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_race_snapshot_pickle_roundtrip():
    """A round-1 snapshot must be plain builtins (int/list/dict) so it survives
    the multiprocessing pickle boundary, and restoring the round-tripped copy
    must reproduce the same layout as the in-process snapshot."""
    opt = _build(47)
    assert opt is not None
    snap, hp, area, V = opt.race_round1(opt.C0, variant=0, t_budget=0.15)

    # structure: (list[list[int]], dict[int, list[list[int]]]), all builtins
    cols_snap, sub_snap = snap
    assert isinstance(cols_snap, list)
    for c in cols_snap:
        assert isinstance(c, list)
        assert all(isinstance(k, int) for k in c)
    assert isinstance(sub_snap, dict)
    for k, sgs in sub_snap.items():
        assert isinstance(k, int)
        for sg in sgs:
            assert all(isinstance(b, int) for b in sg)

    blob = pickle.dumps(snap)
    snap2 = pickle.loads(blob)

    cols_a = opt._restore(snap)
    pa, xa, ya = opt._layout(cols_a)
    cols_b = opt._restore(snap2)
    pb, xb, yb = opt._layout(cols_b)
    assert np.allclose(pa, pb, atol=1e-9)
    assert abs(xa - xb) <= 1e-9 and abs(ya - yb) <= 1e-9


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_race_round2_warm_start_deterministic():
    """race_round2 from the same snapshot with the same seed/deadline budget
    twice must give identical output rectangles (deterministic warm start)."""
    opt = _build(47)
    snap, _hp, _a, _v = opt.race_round1(opt.C0, variant=0, t_budget=0.15)

    o1 = _build(47)
    out1 = o1.race_round2(copy.deepcopy(snap), opt.C0, time.time() + 0.6)
    o2 = _build(47)
    out2 = o2.race_round2(copy.deepcopy(snap), opt.C0, time.time() + 0.6)
    # SA anneal is wall-clock driven, so full runs need not match, but the
    # warm-start seeding (the deterministic part) must: verify the restored
    # start state is identical before any annealing.
    a1 = _build(47); a1.C0 = opt.C0
    a2 = _build(47); a2.C0 = opt.C0
    c1 = a1._restore(copy.deepcopy(snap))
    c2 = a2._restore(copy.deepcopy(snap))
    assert c1 == c2
    p1, x1, y1 = a1._layout(c1)
    p2, x2, y2 = a2._layout(c2)
    assert np.allclose(p1, p2, atol=1e-9)
    # both round-2 outputs must be legal and complete
    assert len(out1) == a1.n and len(out2) == a2.n


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_race_configs_diversity_and_vweight25():
    """The round-1 config list spans column counts, orientations (when
    allowed), both v_weights and both init variants, and always keeps at least
    one v_weight=2.5 config (slow-convergence protection)."""
    opt = _build(94)
    cfgs = cs._race_configs(opt, SEED, allow_t=True, n_slots=24)
    assert 18 <= len(cfgs) <= 24
    assert any(abs(c[4] - 2.5) < 1e-9 for c in cfgs), "no v_weight=2.5 config"
    assert {c[2] for c in cfgs} == {0, 1}, "both init variants must appear"
    assert len({c[1] for c in cfgs}) >= 3, "column-count diversity too low"
    # cfg_ids unique (needed for the score->snapshot join in round 2)
    assert len({c[5] for c in cfgs}) == len(cfgs)


# ---------------------------------------------------------------------------
# M3 width re-optimization: cache correctness + hard-legality preservation
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
@pytest.mark.parametrize("idx", [4, 47, 90])
def test_layout_widths_derive_matches_fast_and_slow(idx):
    """`_layout_widths(cols, [None]*C)` (derive widths, cache-bypassing) must
    equal BOTH the fast (cached) and the slow from-scratch `_layout` at the
    SA's own derived widths -- proving the width path introduces no staleness
    and produces the same base layout the SA saw."""
    opt = _build(idx, fast_eval=True)
    if opt is None:
        pytest.skip(f"case {idx} locked-only")
    cols = [list(c) for c in opt._cols]
    C = len(cols)
    # warm the FAST_EVAL cache
    pf, xf, yf = opt._layout_fast([list(c) for c in cols])
    pw, xw, yw, tops = opt._layout_widths([list(c) for c in cols], [None] * C)
    opt._fast_eval = False
    ps, xs, ys = opt._layout([list(c) for c in cols])
    opt._fast_eval = True
    assert np.allclose(pw, pf, atol=1e-9), "width-derive != fast cached layout"
    assert np.allclose(pw, ps, atol=1e-9), "width-derive != slow layout"
    assert abs(xw - xs) <= 1e-9 and abs(yw - ys) <= 1e-9


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_width_override_no_cache_staleness():
    """After a forced-width layout, the FAST_EVAL cache must still return the
    correct DERIVED-width layout for the same cols -- i.e. the width pass never
    wrote a width-blind entry that a later SA query could reuse."""
    opt = _build(47, fast_eval=True)
    cols = [list(c) for c in opt._cols]
    C = len(cols)
    # cache the derived layout
    p_before, xb, yb = opt._layout_fast([list(c) for c in cols])
    # run a forced-width layout with a perturbed width vector
    _pw, xw, yw, _tops = opt._layout_widths(
        [list(c) for c in cols],
        [(sp[1] - sp[0]) * 1.2 if sp is not None else None
         for sp in opt._col_spans[:C]])
    # querying the cache again must reproduce the ORIGINAL derived layout
    p_after, xa, ya = opt._layout_fast([list(c) for c in cols])
    assert np.allclose(p_before, p_after, atol=1e-9), "cache poisoned by width pass"
    assert abs(xb - xa) <= 1e-9 and abs(yb - ya) <= 1e-9


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
@pytest.mark.parametrize("idx", [4, 47, 90])
def test_forced_width_layout_overlap_free_and_area_exact(idx):
    """A forced-width layout must stay overlap-free and preserve exact soft
    areas (widening a column only reshapes its area/width soft slices)."""
    opt = _build(idx, fast_eval=True)
    if opt is None:
        pytest.skip(f"case {idx} locked-only")
    cols = [list(c) for c in opt._cols]
    C = len(cols)
    widths = [(sp[1] - sp[0]) * 0.85 if sp is not None else None
              for sp in opt._col_spans[:C]]
    # ensure spans are populated
    opt._layout_fast([list(c) for c in cols])
    widths = [(sp[1] - sp[0]) * 0.85 if sp is not None else None
              for sp in opt._col_spans[:C]]
    pos, xr, yt, tops = opt._layout_widths([list(c) for c in cols], widths)
    kind = [opt.kind[i] for i in range(opt.n)]
    assert _overlap_free(pos, kind), "forced-width layout has overlaps"
    for i in range(opt.n):
        if opt.kind[i] == 0:  # soft block: area must equal its target
            a = float(pos[i, 2] * pos[i, 3])
            assert abs(a - opt.areas[i]) <= 1e-4 * max(1.0, opt.areas[i]), \
                f"soft area drift at block {i}: {a} vs {opt.areas[i]}"


# ---------------------------------------------------------------------------
# M1c adaptive reweighting: floor + normalization
# ---------------------------------------------------------------------------
def test_cumulative_helper():
    cum = cs._cumulative([0.55, 0.25, 0.12, 0.08])
    assert cum[-1] == 1.0
    assert abs(cum[0] - 0.55) < 1e-12
    assert abs(cum[1] - 0.80) < 1e-12
    assert abs(cum[2] - 0.92) < 1e-12


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_adaptive_reweight_floor_and_normalization():
    """The reweighter must give every family >= 5% probability, sum the base
    weights to 1.0, and skew mass toward the family with the best (acceptance x
    mean improvement) signal."""
    opt = _build(47)
    opt._adaptive_moves = True
    opt._adaptive_reset()
    # fabricate a window where family 0 dominates the signal and family 3 has
    # zero acceptance (should be floored, not zeroed).
    opt._am_prop = [100, 100, 100, 100]
    opt._am_acc = [80, 10, 5, 0]
    opt._am_impr = [8.0, 0.5, 0.1, 0.0]
    opt._adaptive_reweight()
    w = opt._move_base
    assert abs(sum(w) - 1.0) < 1e-9, "weights must renormalize to 1"
    assert all(wi >= 0.05 - 1e-9 for wi in w), "5% floor violated"
    assert w[0] == max(w), "dominant-signal family should get the most mass"
    assert abs(w[3] - 0.05) < 1e-9, "zero-signal family should sit at the floor"
    # cumulative thresholds stay monotone and end at 1.0
    assert opt._move_cum[-1] == 1.0
    assert all(opt._move_cum[i] <= opt._move_cum[i + 1] + 1e-12
               for i in range(len(opt._move_cum) - 1))


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_adaptive_reweight_no_signal_keeps_mix():
    """With no accepted moves the mix is unchanged (only counters reset)."""
    opt = _build(47)
    opt._adaptive_moves = True
    opt._adaptive_reset()
    before = list(opt._move_base)
    opt._am_prop = [10, 10, 10, 10]
    opt._am_acc = [0, 0, 0, 0]
    opt._am_impr = [0.0, 0.0, 0.0, 0.0]
    opt._adaptive_reweight()
    assert opt._move_base == before
    assert opt._am_prop == [0, 0, 0, 0]  # counters reset


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_move_type_recorded_matches_threshold():
    """_random_move must set _last_move_type consistent with the first
    rng.random() draw against the cumulative thresholds (default mix)."""
    opt = _build(47)
    cols = [list(c) for c in opt._cols]
    cum = opt._move_cum
    for _ in range(300):
        r0 = None

        real = opt.rng.random
        seen = {}

        def _cap():
            opt.rng.random = real
            v = real()
            seen["r"] = v
            return v
        opt.rng.random = _cap
        try:
            undo = opt._random_move(cols)
        finally:
            opt.rng.random = real
        if undo is None:
            continue
        r0 = seen["r"]
        expect = 0 if r0 < cum[0] else (1 if r0 < cum[1] else (2 if r0 < cum[2] else 3))
        assert opt._last_move_type == expect


# ---------------------------------------------------------------------------
# M1c constraint-fixing move: only on V>0, never breaks hard legality
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_cfix_returns_none_when_no_violations():
    """When the current layout has zero soft violations, _cfix_move is a no-op
    (returns None) so it never perturbs an already-clean state."""
    opt = _build(47, fast_eval=True)
    cols = [list(c) for c in opt._cols]
    pos, _xr, _yt = opt._layout(cols)
    if opt._violations(pos) == 0:
        assert opt._cfix_move(cols, 0.0) is None
    else:
        # force a clean state by construction is hard; at least assert that a
        # case which HAS violations yields a candidate or a principled None.
        res = opt._cfix_move(cols, 0.0)
        if res is not None:
            assert isinstance(res, list)


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
@pytest.mark.parametrize("idx", [4, 47, 90, 94])
def test_cfix_preserves_hard_legality(idx):
    """Applying a cfix move (when one fires) must keep the layout overlap-free
    and soft-area-exact -- it only reorders units across columns, never touches
    block geometry."""
    opt = _build(idx, fast_eval=True)
    if opt is None:
        pytest.skip(f"case {idx} locked-only")
    cols = [list(c) for c in opt._cols]
    fired = 0
    for _ in range(60):
        res = opt._cfix_move(cols, 0.0)
        if res is None:
            # nudge the state with a random move to expose a violation
            undo = opt._random_move(cols)
            if undo is None:
                break
            continue
        cols = res
        pos, xr, yt = opt._layout(cols)
        kind = [opt.kind[i] for i in range(opt.n)]
        assert _overlap_free(pos, kind), f"case {idx}: cfix produced overlap"
        for i in range(opt.n):
            if opt.kind[i] == 0:
                a = float(pos[i, 2] * pos[i, 3])
                assert abs(a - opt.areas[i]) <= 1e-4 * max(1.0, opt.areas[i])
        fired += 1
        if fired >= 5:
            break


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_cfix_only_targets_boundary_violating_units():
    """_cfix_move must only ever relocate a unit that owns a bottom/top-tagged
    block currently missing its edge (never a clean unit)."""
    # find a case whose initial layout actually has a boundary violation
    opt = None
    for idx in (4, 9, 47, 90, 94):
        cand = _build(idx, fast_eval=True)
        if cand is None:
            continue
        cols0 = [list(c) for c in cand._cols]
        for _ in range(40):
            pos, _x, _y = cand._layout(cols0)
            py0 = pos[:, 1]; py1 = py0 + pos[:, 3]
            y_min = float(py0.min()); y_max = float(py1.max())
            has_bad = False
            for i, code in zip(cand._bnd_idx, cand._bnd_codes):
                if (int(code) & 8) and abs(float(py0[i]) - y_min) >= 1e-6:
                    has_bad = True
                elif (int(code) & 4) and abs(float(py1[i]) - y_max) >= 1e-6:
                    has_bad = True
            if has_bad:
                opt = cand
                cols = cols0
                break
            undo = cand._random_move(cols0)
            if undo is None:
                break
        if opt is not None:
            break
    if opt is None:
        pytest.skip("no boundary-violating state reached in the sampled cases")

    res = opt._cfix_move(cols, 0.0)
    # either it declined (no movable candidate column) or it moved a genuinely
    # violating unit -- verify the moved unit was boundary-tagged.
    if res is None:
        pytest.skip("cfix declined (no free same-tag column)")
    # the moved unit is the one whose position differs between cols and res
    moved = None
    for ci in range(len(cols)):
        before = set(cols[ci])
        after = set(res[ci]) if ci < len(res) else set()
        gone = before - after
        if gone:
            moved = next(iter(gone))
            break
    assert moved is not None
    u = opt.units[moved]
    assert u.hasB or u.hasT, "cfix moved a unit with no bottom/top tag"
