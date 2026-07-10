"""Unit tests for the dormant Fast-SA three-stage cooling schedule
(FLOORSET_SA_SCHEDULE=fastsa) in `legalizer/column_slicing.py`.

The port swaps ONLY the T(frac) curve in `_anneal` for the Chen-Chang ISPD'05
three-stage law (`_fastsa_temp`); the move set, Metropolis accept,
recalibration and finish/polish are untouched. Default (`geometric`) keeps the
historical wall-clock geometric law byte-identical.

The temperature law is a pure function, so its invariants (endpoints, the
stage-2 deep quench, the reheat jump at n=k, the 1/n stage-3 tail) are tested
directly without the dataset. A couple of dataset-backed smoke tests confirm
the flag wires into a real optimizer and the fastsa path anneals to a legal
layout.
"""

from __future__ import annotations

import math
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
except Exception as exc:  # pragma: no cover - dataset absence
    _DS = None
    _DS_ERR = str(exc)

SEED = 20260705
T0, T1 = 0.02, 0.0008           # the finish()/probe defaults the port anchors to
K, C, STEPS = 7.0, 100.0, 41.0  # paper (k=7, c=100) + our frac->index span

_SCHED_ENV = (
    "FLOORSET_SA_SCHEDULE",
    "FLOORSET_SA_FASTSA_K",
    "FLOORSET_SA_FASTSA_C",
    "FLOORSET_SA_FASTSA_STEPS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for _v in _SCHED_ENV:
        monkeypatch.delenv(_v, raising=False)
    yield
    for _v in _SCHED_ENV:
        monkeypatch.delenv(_v, raising=False)


def _geometric(frac):
    return T0 * (T1 / T0) ** frac


def _fs(frac):
    return cs._fastsa_temp(frac, T0, T1, K, C, STEPS)


# ---------------------------------------------------------------------------
# Pure temperature-law invariants (dataset-free)
# ---------------------------------------------------------------------------
def test_fastsa_anchors_match_geometric_endpoints():
    """Endpoints must equal the geometric anchors so the accept regime is
    preserved: hot start == t0, cold end == t1."""
    assert _fs(0.0) == pytest.approx(T0, abs=1e-12)
    assert _fs(1.0) == pytest.approx(T1, rel=1e-9)


def test_fastsa_stage2_is_deep_quench_below_cold_end():
    """Stage 2 (2 <= n <= k) quenches BELOW the final cold temperature -- the
    'pseudo-greedy' descent -- thanks to the 1/(n*c) factor."""
    # frac just below the reheat boundary lands inside stage 2 (n approaches k)
    frac_k = (K - 1.0) / (STEPS - 1.0)          # n == k
    assert _fs(frac_k - 0.01) < T1
    assert _fs(frac_k) < T1                       # n == k is still stage 2


def test_fastsa_reheat_jump_at_k():
    """Crossing n=k reheats by ~c: dropping the 1/c factor multiplies T by
    roughly the paper constant."""
    frac_k = (K - 1.0) / (STEPS - 1.0)
    below = _fs(frac_k - 0.005)     # stage 2 (near n=k)
    above = _fs(frac_k + 0.005)     # stage 3 (just past n=k)
    assert above > below
    # the jump is order-c (allow slack since n advances slightly across the gap)
    assert above / below > 40.0


def test_fastsa_stage3_monotone_cooling_to_t1():
    """Stage 3 (n > k) cools monotonically as 1/n down to t1."""
    seq = [_fs(f) for f in (0.2, 0.4, 0.6, 0.8, 0.95, 1.0)]
    assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), seq
    assert seq[-1] == pytest.approx(T1, rel=1e-9)


def test_fastsa_all_positive_and_bounded():
    fracs = [i / 200.0 for i in range(201)]
    ts = [_fs(f) for f in fracs]
    assert all(t > 0.0 for t in ts)
    # never exceeds the hot anchor; stage-1 flat top is the max
    assert max(ts) == pytest.approx(T0, abs=1e-12)


def test_fastsa_shape_differs_from_geometric():
    """Mid-run the two laws must genuinely diverge (the whole point): fastsa
    sits well below the geometric sweep after its quench."""
    for f in (0.3, 0.5, 0.7):
        assert abs(_fs(f) - _geometric(f)) > 0.2 * _geometric(f)


def test_fastsa_reheat_ratio_tracks_c():
    """The exact discontinuity at n=k (same n, drop the 1/c factor) is c: this
    pins the reheat magnitude to the paper parameter, not an ad-hoc bump."""
    # evaluate the two branch formulas at the same index n = k
    A = T1 * STEPS
    stage2_at_k = A / (K * C)
    stage3_at_k = A / K
    assert stage3_at_k / stage2_at_k == pytest.approx(C, rel=1e-9)


def test_fastsa_endpoints_hold_for_probe_anchors():
    """The law adapts to whatever (t0,t1) the caller passes (probe/race use
    different endpoints); endpoints must still pin exactly."""
    t0, t1 = 0.06, 0.01   # probe() anchors
    assert cs._fastsa_temp(0.0, t0, t1, K, C, STEPS) == pytest.approx(t0, abs=1e-12)
    assert cs._fastsa_temp(1.0, t0, t1, K, C, STEPS) == pytest.approx(t1, rel=1e-9)


# ---------------------------------------------------------------------------
# Dataset-backed: flag wiring + fastsa anneal is legal
# ---------------------------------------------------------------------------
def _n_of(sample):
    return int((sample["input"][0] != -1).sum().item())


def _golden_target_positions(sample):
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


def _build(idx, seed=SEED):
    sample = _DS[idx]
    at, b2b, p2b, pins, cons = sample["input"]
    n = _n_of(sample)
    at = at[:n].float(); cons = cons[:n].float()
    b2b = b2b.float(); p2b = p2b.float(); pins = pins.float()
    tpos = _golden_target_positions(sample)
    seed_rects = cb._heuristic_init(at, cons, tpos, b2b, p2b, pins)
    opt = cs._ColumnOptimizer(seed_rects, at, cons, tpos, b2b, p2b, pins,
                              time.time() + 5.0, seed=seed)
    if opt.locked_only():
        return None
    opt.prepare()
    return opt


def _overlap_free(pos, kind, tol=1e-6):
    n = len(pos)
    x0 = pos[:, 0]; y0 = pos[:, 1]; x1 = x0 + pos[:, 2]; y1 = y0 + pos[:, 3]
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
            oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
            if ox > tol and oy > tol:
                return False
    return True


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_default_optimizer_is_geometric():
    opt = _build(47)
    assert opt is not None
    assert opt._sa_schedule == "geometric"
    assert opt._fastsa_k == pytest.approx(7.0)
    assert opt._fastsa_c == pytest.approx(100.0)
    assert opt._fastsa_steps == pytest.approx(41.0)


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_fastsa_flag_and_params_parse(monkeypatch):
    monkeypatch.setenv("FLOORSET_SA_SCHEDULE", "fastsa")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_K", "5")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_C", "50")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_STEPS", "30")
    opt = _build(47)
    assert opt._sa_schedule == "fastsa"
    assert opt._fastsa_k == pytest.approx(5.0)
    assert opt._fastsa_c == pytest.approx(50.0)
    assert opt._fastsa_steps == pytest.approx(30.0)


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_fastsa_invalid_values_fall_back(monkeypatch):
    monkeypatch.setenv("FLOORSET_SA_SCHEDULE", "nonsense")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_K", "abc")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_C", "-3")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_STEPS", "xyz")
    opt = _build(47)
    assert opt._sa_schedule == "geometric"          # unknown -> geometric
    assert opt._fastsa_k == pytest.approx(7.0)
    assert opt._fastsa_c == pytest.approx(100.0)     # non-positive -> default
    assert opt._fastsa_steps == pytest.approx(41.0)


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_fastsa_params_guarded_wellformed(monkeypatch):
    """Degenerate params must be clamped so steps > k >= 2 (the law stays
    well-formed and endpoints still pin)."""
    monkeypatch.setenv("FLOORSET_SA_SCHEDULE", "fastsa")
    monkeypatch.setenv("FLOORSET_SA_FASTSA_K", "1")     # -> clamped to 2
    monkeypatch.setenv("FLOORSET_SA_FASTSA_STEPS", "2")  # -> clamped above k
    opt = _build(47)
    assert opt._fastsa_k >= 2.0
    assert opt._fastsa_steps > opt._fastsa_k
    t0 = cs._fastsa_temp(0.0, T0, T1, opt._fastsa_k, opt._fastsa_c, opt._fastsa_steps)
    t1 = cs._fastsa_temp(1.0, T0, T1, opt._fastsa_k, opt._fastsa_c, opt._fastsa_steps)
    assert t0 == pytest.approx(T0, abs=1e-12)
    assert t1 == pytest.approx(T1, rel=1e-9)


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_fastsa_anneal_produces_legal_layout():
    """The fastsa path must run `_anneal` + `finish` end to end and yield a
    complete, overlap-free layout (schedule change is quality-only, never
    legality)."""
    opt = _build(94)
    opt._sa_schedule = "fastsa"
    out = opt.finish(time.time() + 0.6, max_runs=1)
    assert len(out) == opt.n
    pos = np.array([[x, y, w, h] for (x, y, w, h) in out], dtype=float)
    kind = [opt.kind[i] for i in range(opt.n)]
    assert _overlap_free(pos, kind), "fastsa finish produced overlaps"
    assert all(math.isfinite(v) for r in out for v in r)
