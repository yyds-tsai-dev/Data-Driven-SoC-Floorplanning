"""Tests for the M5 v3 sequence-pair window packer
(src/floorset_arch/refine/sp_pack.py) and its FLOORSET_WINDOW_SP_PACK wiring
into window_repack._repack_window.

Covers: SP relation semantics (left-of / below reachable arrangements), the
fit filter under a tight window, invariants on every emitted candidate (no
overlap, inside window, shapes passed through unchanged), HPWL-sorted output,
ALAP justification reaching the far wall, k caps, determinism, gating (SP
never runs with the flag off; can only improve with it on), and the packer
attribution field in the per-window JSONL log.
"""

import json
import random

import torch

import floorset_arch.refine.sp_pack as sp_pack
from floorset_arch.refine.guards import hard_legal, soft_violations
from floorset_arch.refine.sp_pack import sp_enumerate
from floorset_arch.refine.window_repack import _window_hpwl, refine_window


def _no_overlap(boxes):
    n = len(boxes)
    for i in range(n):
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = boxes[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6), f"{i},{j} overlap: {boxes}"


def _in_window(boxes, win, tol=1e-6):
    wx0, wy0, wx1, wy1 = win
    for (bx, by, bw, bh) in boxes:
        assert bx >= wx0 - tol and by >= wy0 - tol
        assert bx + bw <= wx1 + tol and by + bh <= wy1 + tol


def _boxes(cand, k):
    return [cand[i] for i in range(k)]


# --- SP semantics -----------------------------------------------------------


def test_two_blocks_horizontal_and_stacked_reachable():
    shapes = [(2.0, 1.0), (1.0, 1.0)]
    win = (0.0, 0.0, 3.0, 2.0)
    cands = sp_enumerate(shapes, win, internal=[], external={}, top_n=64)
    assert cands, "expected candidates for a loose window"
    keys = {tuple((round(c[i][0], 6), round(c[i][1], 6)) for i in range(2))
            for c in cands}
    # a left-of b, both bottom-left justified.
    assert ((0.0, 0.0), (2.0, 0.0)) in keys
    # a below b (stacked), bottom-left justified.
    assert ((0.0, 0.0), (0.0, 1.0)) in keys


def test_narrow_window_admits_only_stacking():
    shapes = [(2.0, 1.0), (2.0, 1.0)]
    win = (0.0, 0.0, 2.0, 2.0)
    cands = sp_enumerate(shapes, win, internal=[], external={}, top_n=64)
    assert cands
    for c in cands:
        pos = sorted((c[i][0], c[i][1]) for i in range(2))
        assert pos == [(0.0, 0.0), (0.0, 1.0)], f"non-stacked candidate {c}"


def test_unfittable_returns_empty():
    # Each block alone exceeds the window height.
    shapes = [(1.0, 3.0), (1.0, 3.0)]
    win = (0.0, 0.0, 4.0, 2.0)
    assert sp_enumerate(shapes, win, internal=[], external={}, top_n=8) == []


# --- invariants on every candidate ------------------------------------------


def test_fuzz_invariants_no_overlap_in_window_shapes_unchanged():
    rng = random.Random(7)
    for _ in range(30):
        k = rng.randint(2, 5)
        win = (rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), 0.0, 0.0)
        wx0, wy0 = win[0], win[1]
        win = (wx0, wy0, wx0 + 6.0, wy0 + 6.0)
        shapes = [(rng.uniform(0.5, 6.0 / k), rng.uniform(0.5, 5.5))
                  for _ in range(k)]
        internal = [(a, b, rng.uniform(0.5, 3.0))
                    for a in range(k) for b in range(a + 1, k)
                    if rng.random() < 0.4]
        external = {i: [(wx0 + rng.uniform(0, 6), wy0 + rng.uniform(0, 6),
                         rng.uniform(0.5, 2.0))]
                    for i in range(k) if rng.random() < 0.5}
        cands = sp_enumerate(shapes, win, internal, external, top_n=16)
        assert cands, "a single-row arrangement always fits this window"
        for c in cands:
            boxes = _boxes(c, k)
            _no_overlap(boxes)
            _in_window(boxes, win)
            for i in range(k):
                assert abs(boxes[i][2] - shapes[i][0]) < 1e-12
                assert abs(boxes[i][3] - shapes[i][1]) < 1e-12


def test_candidates_sorted_by_window_hpwl():
    shapes = [(1.0, 1.0)] * 3
    win = (0.0, 0.0, 3.0, 3.0)
    internal = [(0, 1, 2.0)]
    external = {0: [(0.5, 0.5, 10.0)], 2: [(2.5, 2.5, 1.0)]}
    cands = sp_enumerate(shapes, win, internal, external, top_n=24)
    assert len(cands) >= 2
    scores = [_window_hpwl(_boxes(c, 3), internal, external) for c in cands]
    assert all(scores[i] <= scores[i + 1] + 1e-9 for i in range(len(scores) - 1))


def test_alap_justification_reaches_far_wall():
    shapes = [(1.0, 1.0), (1.0, 1.0)]
    win = (0.0, 0.0, 10.0, 1.0)
    external = {0: [(10.0, 0.5, 1.0)], 1: [(10.0, 0.5, 1.0)]}
    cands = sp_enumerate(shapes, win, internal=[], external=external, top_n=4)
    assert cands
    best = _boxes(cands[0], 2)
    assert max(b[0] + b[2] for b in best) > 10.0 - 1e-6, (
        f"best candidate should be right-justified, got {best}")


# --- caps + determinism -------------------------------------------------------


def test_k_bounds():
    win = (0.0, 0.0, 10.0, 10.0)
    assert sp_enumerate([(1.0, 1.0)], win, [], {}, top_n=8) == []
    six = [(1.0, 1.0)] * 6
    assert sp_enumerate(six, win, [], {}, top_n=8, max_k=5) == []
    nine = [(1.0, 1.0)] * 9
    assert sp_enumerate(nine, win, [], {}, top_n=8, max_k=9) == []  # hard cap 6


def test_deterministic():
    shapes = [(2.0, 1.0), (1.0, 2.0), (1.5, 1.0), (1.0, 1.0)]
    win = (0.0, 0.0, 4.0, 3.0)
    internal = [(0, 3, 5.0), (1, 2, 1.0)]
    external = {1: [(0.0, 3.0, 2.0)]}
    a = sp_enumerate(shapes, win, internal, external, top_n=12)
    b = sp_enumerate(shapes, win, internal, external, top_n=12)
    assert a == b


# --- window_repack wiring -----------------------------------------------------


def _constraints(n):
    return torch.tensor([[0.0] * 5 for _ in range(n)], dtype=torch.float32)


def _targets(n):
    return torch.tensor([[-1.0] * 4 for _ in range(n)], dtype=torch.float32)


def _improving_window_layout():
    """2x2 grid of unit squares with a heavy diagonal net (same shape as the
    fixture in test_window_repack.py)."""
    rects = [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]
    area = torch.tensor([1.0, 1.0, 1.0, 1.0])
    b2b = [(0, 3, 20.0), (0, 1, 1.0), (2, 3, 1.0)]
    return rects, area, _constraints(4), _targets(4), b2b, [], []


def test_flag_off_sp_never_called(monkeypatch):
    monkeypatch.delenv("FLOORSET_WINDOW_SP_PACK", raising=False)
    calls = {"n": 0}
    real = sp_pack.sp_enumerate

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sp_pack, "sp_enumerate", counting)
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    refine_window(rects, area, cons, tpos, b2b, p2b, pins)
    assert calls["n"] == 0


def test_flag_on_sp_called_and_monotone(monkeypatch):
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()

    monkeypatch.delenv("FLOORSET_WINDOW_SP_PACK", raising=False)
    out_off, detail_off = refine_window(rects, area, cons, tpos, b2b, p2b, pins)

    calls = {"n": 0}
    real = sp_pack.sp_enumerate

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sp_pack, "sp_enumerate", counting)
    monkeypatch.setenv("FLOORSET_WINDOW_SP_PACK", "1")
    out_on, detail_on = refine_window(rects, area, cons, tpos, b2b, p2b, pins)

    assert calls["n"] > 0, "flag on must reach the SP generator"
    # SP only ADDS candidates through the same strict-better selection: the
    # final HPWL can never be worse than the classic-only stage.
    assert detail_on["post_hpwl"] <= detail_off["post_hpwl"] + 1e-9
    _no_overlap(out_on)
    assert hard_legal(out_on, area, cons, tpos)
    base_soft = soft_violations(rects, cons)
    assert all(c <= b for c, b in
               zip(soft_violations(out_on, cons), base_soft))


def test_packer_attribution_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOORSET_WINDOW_SP_PACK", "1")
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    log = tmp_path / "win.jsonl"
    _out, detail = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                                 log_path=str(log))
    assert "n_sp_accepted" in detail
    recs = [json.loads(line) for line in log.read_text().splitlines() if line]
    win_recs = [r for r in recs if r.get("k")]
    repacked = [r for r in win_recs if r.get("guard_result") != "no_repack"]
    assert all(r.get("packer") in ("sp", "classic") for r in repacked)
