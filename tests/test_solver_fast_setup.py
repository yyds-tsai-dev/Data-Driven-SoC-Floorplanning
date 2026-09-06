"""`PARTNER_FAST_SETUP=1`: budget-aware per-case fixed cost.

Two independent surgeries, one flag (default OFF, off-path bit-identical):

1. `finish`'s minimum anneal span was a flat 0.1 s.  At the goal operating
   point (per-case budget 5e-5 * exp(n/12) clamped to [0.05, 0.75]) that
   floor is 2-3x the WHOLE case budget, so all ~24 pool workers annealed
   ~100 ms regardless of the deadline they were handed -- the measured
   0.12-0.17 s per-case wall floor.  The flag clamps the floor to the span
   the chain was actually given.  It can only bind when the planned span is
   already below 0.1 s, so a case at any pre-goal-tier budget stays
   bit-identical even with the flag ON.

2. `_heuristic_init`'s two connectivity loops walked the PADDED b2b / p2b
   tensors row by row with `.item()` calls (4.7 ms at n=25, 136 ms at
   n=120, all serial in the parent).  `_pin_centroids_np` / `_b2b_smooth_np`
   are numpy transcriptions; the contract asserted here is BIT equality, not
   tolerance -- the seed feeds every restart, so anything less would make
   the flag a search-trajectory change instead of a pure speed change.

See docs/design/2026-08-05-fast-setup-floor.md.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as lg  # noqa: E402
from synth_instances import build_instance  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PARTNER_FAST_SETUP", raising=False)
    yield


# --------------------------------------------------------------------------
# reference transcriptions (the loops the twins replace, verbatim)
# --------------------------------------------------------------------------
def _pin_centroids_ref(p2b, pins, n, n_pins):
    sx = [0.0] * n
    sy = [0.0] * n
    wsum = [0.0] * n
    for edge in p2b:
        if edge[0] == -1:
            continue
        p = int(edge[0].item())
        b = int(edge[1].item())
        if not (0 <= b < n and 0 <= p < n_pins):
            continue
        px, py = float(pins[p, 0]), float(pins[p, 1])
        if px == -1.0 or py == -1.0:
            continue
        w = max(float(edge[2].item()), 0.0)
        sx[b] += w * px
        sy[b] += w * py
        wsum[b] += w
    return sx, sy, wsum


def _b2b_smooth_ref(b2b, cx, cy, n):
    nx = list(cx)
    ny = list(cy)
    deg = [0.0] * n
    for edge in b2b:
        if edge[0] == -1:
            continue
        i = int(edge[0].item())
        j = int(edge[1].item())
        if not (0 <= i < n and 0 <= j < n):
            continue
        w = max(float(edge[2].item()), 0.0)
        nx[i] += 0.25 * w * cx[j]
        ny[i] += 0.25 * w * cy[j]
        nx[j] += 0.25 * w * cx[i]
        ny[j] += 0.25 * w * cy[i]
        deg[i] += 0.25 * w
        deg[j] += 0.25 * w
    return nx, ny, deg


def _centroids(n, seed):
    rng = random.Random(seed)
    return ([rng.uniform(-30.0, 300.0) for _ in range(n)],
            [rng.uniform(-30.0, 300.0) for _ in range(n)])


# --------------------------------------------------------------------------
# flag plumbing
# --------------------------------------------------------------------------
def test_flag_defaults_off(monkeypatch):
    assert lg.fast_setup_on() is False
    monkeypatch.setenv("PARTNER_FAST_SETUP", "0")
    assert lg.fast_setup_on() is False
    monkeypatch.setenv("PARTNER_FAST_SETUP", "1")
    assert lg.fast_setup_on() is True


# --------------------------------------------------------------------------
# seed twins: bit equality
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n,seed,n_pins", [(21, 0, 12), (60, 3, 40),
                                           (100, 7, 90), (120, 11, 200)])
def test_pin_centroid_twin_is_bit_exact(n, seed, n_pins):
    inst = build_instance(n=n, seed=seed, n_pins=n_pins)
    ref = _pin_centroids_ref(inst.p2b, inst.pins, n, inst.pins.shape[0])
    got = lg._pin_centroids_np(inst.p2b, inst.pins, n, inst.pins.shape[0])
    assert got is not None
    for a, b in zip(ref, got):
        assert a == b


@pytest.mark.parametrize("n,seed", [(21, 0), (60, 3), (100, 7), (120, 11)])
def test_b2b_twin_is_bit_exact(n, seed):
    inst = build_instance(n=n, seed=seed, edge_mult=4.0)
    cx, cy = _centroids(n, seed)
    ref = _b2b_smooth_ref(inst.b2b, cx, cy, n)
    got = lg._b2b_smooth_np(inst.b2b, cx, cy, n)
    assert got is not None
    for a, b in zip(ref, got):
        assert a == b


def test_twins_decline_unexpected_layouts():
    """A shape the loops would not understand returns None so the caller
    keeps the reference path instead of guessing."""
    assert lg._pin_centroids_np(torch.zeros((4, 2)), torch.zeros((4, 2)),
                                4, 4) is None
    assert lg._b2b_smooth_np(torch.zeros(5), [0.0] * 4, [0.0] * 4, 4) is None


def test_twins_handle_degenerate_edge_tables():
    """Padding rows, out-of-range endpoints, self loops, duplicate pairs,
    negative weights and empty tables must all reproduce the loops."""
    n = 6
    cx, cy = _centroids(n, 5)
    pins = torch.tensor([[1.0, 2.0], [-1.0, 4.0], [5.0, -1.0], [7.0, 8.0]])
    p2b = torch.tensor([
        [-1.0, -1.0, -1.0],      # padding
        [0.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],         # pin with x == -1 -> skipped
        [2.0, 3.0, 4.0],         # pin with y == -1 -> skipped
        [3.0, 99.0, 1.0],        # block out of range
        [99.0, 1.0, 1.0],        # pin out of range
        [3.0, 2.0, -2.0],        # negative weight -> clamped to 0
        [0.0, 2.0, 5.0],         # duplicate target block
        [3.0, 5.0, 0.5],
    ])
    b2b = torch.tensor([
        [-1.0, -1.0, -1.0],      # padding
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 3.0],         # self loop
        [1.0, 0.0, 4.0],         # reversed duplicate (ordering matters)
        [0.0, 99.0, 1.0],        # out of range
        [2.0, 3.0, -1.0],        # negative weight
        [3.0, 2.0, 7.0],
    ])
    assert _pin_centroids_ref(p2b, pins, n, pins.shape[0]) == \
        lg._pin_centroids_np(p2b, pins, n, pins.shape[0])
    assert _b2b_smooth_ref(b2b, cx, cy, n) == \
        lg._b2b_smooth_np(b2b, cx, cy, n)

    empty = torch.zeros((0, 3))
    assert _pin_centroids_ref(empty, pins, n, pins.shape[0]) == \
        lg._pin_centroids_np(empty, pins, n, pins.shape[0])
    assert _b2b_smooth_ref(empty, cx, cy, n) == \
        lg._b2b_smooth_np(empty, cx, cy, n)


# --------------------------------------------------------------------------
# anneal-span floor
#
# The SA is wall-clock bounded, so two runs of the same instance are NOT
# expected to return the same layout (with or without this flag).  What is
# decidable -- and what the flag actually changes -- is the SPAN handed to
# `_anneal`, so that is what these tests assert.
# --------------------------------------------------------------------------
def _run_finish(inst, span, seed=0, spans=None):
    opt = lg._ColumnOptimizer(inst.rects, inst.area_targets,
                              inst.constraints, inst.target_positions,
                              inst.b2b, inst.p2b, inst.pins,
                              time.time() + span, seed=seed)
    opt.prepare()
    real = lg._ColumnOptimizer._anneal
    if spans is not None:
        def spy(self, cols, deadline, cur_cost, **kw):
            spans.append(deadline - time.time())
            return real(self, cols, deadline, cur_cost, **kw)
        lg._ColumnOptimizer._anneal = spy
    try:
        t0 = time.perf_counter()
        out = opt.finish(time.time() + span, max_runs=1)
        dt = time.perf_counter() - t0
    finally:
        lg._ColumnOptimizer._anneal = real
    return out, dt


def test_micro_budget_overshoots_by_default():
    """The behaviour this flag exists to fix: a 0.05 s span still costs
    ~0.1 s because the anneal floor ignores the deadline."""
    inst = build_instance(n=40, seed=2)
    spans = []
    _out, dt = _run_finish(inst, 0.05, spans=spans)
    assert spans and spans[0] == pytest.approx(0.1, abs=0.01)
    assert dt > 0.09


def test_micro_budget_respects_deadline_when_on(monkeypatch):
    monkeypatch.setenv("PARTNER_FAST_SETUP", "1")
    inst = build_instance(n=40, seed=2)
    spans = []
    _out, dt = _run_finish(inst, 0.05, spans=spans)
    # planned share of a 0.05 s window: everything but the polish reserve
    assert spans and spans[0] < 0.05
    assert dt < 0.05 + 0.02       # planned span + one outer-loop grain


def test_flag_is_a_no_op_above_the_floor(monkeypatch):
    """The clamp is `min(0.1, planned)`, so it cannot bind once the planned
    span exceeds 0.1 s: every pre-goal-tier budget keeps its exact anneal
    span with the flag ON."""
    inst = build_instance(n=50, seed=4)
    off, on = [], []
    monkeypatch.delenv("PARTNER_FAST_SETUP", raising=False)
    _run_finish(inst, 0.6, seed=9, spans=off)
    monkeypatch.setenv("PARTNER_FAST_SETUP", "1")
    _run_finish(inst, 0.6, seed=9, spans=on)
    assert off[0] == pytest.approx(on[0], abs=2e-3)


def test_expired_deadline_returns_promptly(monkeypatch):
    """Zero-span edge of the clamp: a deadline that is already gone must
    return a layout immediately instead of buying another 0.1 s."""
    monkeypatch.setenv("PARTNER_FAST_SETUP", "1")
    inst = build_instance(n=40, seed=2)
    out, dt = _run_finish(inst, 0.0)
    assert len(out) == 40
    assert dt < 0.05
    out2, _ = _run_finish(inst, 0.0)
    assert out == out2            # no anneal moves -> deterministic

# --------------------------------------------------------------------------
# legality invariants at the clamped budget
# --------------------------------------------------------------------------
def _overlap_area(r1, r2):
    ox = min(r1[0] + r1[2], r2[0] + r2[2]) - max(r1[0], r2[0])
    oy = min(r1[1] + r1[3], r2[1] + r2[3]) - max(r1[1], r2[1])
    return max(ox, 0.0) * max(oy, 0.0)


@pytest.mark.parametrize("n,seed", [(30, 1), (60, 5), (90, 8)])
def test_layout_stays_legal_at_micro_budget(monkeypatch, n, seed):
    monkeypatch.setenv("PARTNER_FAST_SETUP", "1")
    inst = build_instance(n=n, seed=seed)
    out, _dt = _run_finish(inst, 0.05, seed=seed)
    assert len(out) == n
    for (x, y, w, h) in out:
        assert w > 0.0 and h > 0.0
        assert all(np.isfinite(v) for v in (x, y, w, h))
    for i in range(n):
        for j in range(i + 1, n):
            assert _overlap_area(out[i], out[j]) <= 1e-6
    tp = inst.target_positions
    cons = inst.constraints
    for i in range(n):
        if float(cons[i, 1]) != 0.0 and float(tp[i, 0]) >= 0.0:
            assert out[i] == pytest.approx(
                (float(tp[i, 0]), float(tp[i, 1]),
                 float(tp[i, 2]), float(tp[i, 3])), abs=1e-9)
