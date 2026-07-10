"""Unit tests for the E2-adaptive per-chain early stop in
`legalizer/column_slicing.py` (mounted from `column_backbone`).

The feature gives the tail a generous budget ceiling (base * CAP) and lets each
SA worker self-stop when its own best-cost trajectory stalls. These tests pin
the load-bearing invariants a paired eval cannot see cheaply:

* the early stop is fully OFF (byte-identical) when `stall_window is None`;
* a configured stall window breaks the chain and flags `_anneal_stalled`,
  well before the (generous) deadline;
* a window larger than the deadline never triggers -- the chain runs to the
  deadline exactly as before;
* `finish` caps its whole wrap-up (polish/width-opt) once the chain stalled, so
  the episode ends ~one window past the anneal instead of burning the ceiling,
  and still returns a complete overlap-free layout.

The stall detector hangs on the existing per-outer-loop wall-clock read, so the
tests drive it purely through `stall_window` / `stall_eps` (a huge eps makes
"no meaningful improvement" certain, so the trigger is deterministic without
mocking the RNG or the clock). Runs in a couple of seconds.
"""

from __future__ import annotations

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


def _build(idx: int, seed: int = SEED):
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
# Default optimizer state: early stop disabled
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_default_stall_window_is_none():
    """A freshly built optimizer has the early stop disabled -- the production
    default and the guarantee that untouched paths are byte-identical."""
    opt = _build(94)
    assert opt is not None
    assert opt.stall_window is None
    assert opt.stall_eps == pytest.approx(0.003)
    assert opt._anneal_stalled is False


# ---------------------------------------------------------------------------
# _anneal early-stop behavior
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_anneal_disabled_runs_to_deadline():
    """stall_window=None: the chain must run to the deadline and never flag a
    stall (byte-identical control)."""
    opt = _build(94)
    cols = [list(c) for c in opt._cols]
    cur_cost, _ = opt._evaluate(cols)
    opt.stall_window = None
    t0 = time.time()
    opt._anneal(cols, time.time() + 0.3, cur_cost)
    elapsed = time.time() - t0
    assert opt._anneal_stalled is False
    assert elapsed >= 0.25, f"stopped early with no window: {elapsed:.3f}s"


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_anneal_large_window_never_triggers():
    """A window larger than the whole run must never fire; behaves exactly like
    the disabled path (runs to the deadline, no stall flag)."""
    opt = _build(94)
    cols = [list(c) for c in opt._cols]
    cur_cost, _ = opt._evaluate(cols)
    opt.stall_window = 1e9
    opt.stall_eps = 0.003
    t0 = time.time()
    opt._anneal(cols, time.time() + 0.3, cur_cost)
    elapsed = time.time() - t0
    assert opt._anneal_stalled is False
    assert elapsed >= 0.25, f"large window stopped early: {elapsed:.3f}s"


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_anneal_stalls_and_returns_early():
    """A zero window with an unreachable eps (so no window ever counts as
    improving) must break on the first outer iteration, flag the stall, and
    return far before the generous deadline."""
    opt = _build(94)
    cols = [list(c) for c in opt._cols]
    cur_cost, _ = opt._evaluate(cols)
    opt.stall_window = 0.0
    opt.stall_eps = 10.0  # requires best_cost -> <= 0 to reset; never happens
    t0 = time.time()
    snap, bc = opt._anneal(cols, time.time() + 10.0, cur_cost)
    elapsed = time.time() - t0
    assert opt._anneal_stalled is True
    assert elapsed < 1.0, f"stall did not cut the 10s deadline: {elapsed:.3f}s"
    # a valid snapshot is still returned (the initial best)
    assert snap is not None


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_anneal_stalls_after_one_window():
    """A small nonzero window with unreachable eps runs ~one window then stops,
    still well under the deadline."""
    opt = _build(94)
    cols = [list(c) for c in opt._cols]
    cur_cost, _ = opt._evaluate(cols)
    opt.stall_window = 0.1
    opt.stall_eps = 10.0
    t0 = time.time()
    opt._anneal(cols, time.time() + 10.0, cur_cost)
    elapsed = time.time() - t0
    assert opt._anneal_stalled is True
    # one window (0.1s) plus at most one 24-move outer batch of slack
    assert elapsed < 2.0, f"ran far past one window: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# finish caps the whole wrap-up once the chain stalls
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_finish_caps_episode_when_stalled():
    """`finish` given a generous ceiling must return quickly once the chain
    stalls -- the polish/width-opt wrap-up is capped to one stall window, not
    the full deadline -- and still yield a complete, overlap-free layout."""
    opt = _build(94)
    opt.stall_window = 0.1
    opt.stall_eps = 10.0
    t0 = time.time()
    out = opt.finish(time.time() + 30.0, max_runs=1)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"episode not capped below the 30s ceiling: {elapsed:.2f}s"
    assert len(out) == opt.n
    pos = np.array([[x, y, w, h] for (x, y, w, h) in out], dtype=float)
    kind = [opt.kind[i] for i in range(opt.n)]
    assert _overlap_free(pos, kind), "capped finish produced overlaps"


@pytest.mark.skipif(_DS is None, reason="LiteTensorDataTest unavailable")
def test_finish_uses_full_deadline_when_disabled():
    """Control: with the early stop off, finish runs to the (short) deadline as
    before -- confirming the cap is gated strictly on the adaptive flag."""
    opt = _build(94)
    assert opt.stall_window is None
    t0 = time.time()
    out = opt.finish(time.time() + 1.0, max_runs=1)
    elapsed = time.time() - t0
    assert len(out) == opt.n
    assert elapsed >= 0.8, f"disabled finish returned early: {elapsed:.3f}s"
