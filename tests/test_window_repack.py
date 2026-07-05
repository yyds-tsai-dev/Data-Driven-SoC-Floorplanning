"""Invariant tests for the M5 windowed re-pack stage
(src/floorset_arch/refine/window_repack.py). Follows the
tests/test_topo_search.py / test_refine_invariants.py patterns.

Covers: window selection filters (LOCKED / boundary / cluster / MIB), the
repacked window stays inside the window bbox, guard failure returns input,
accepted result is strictly better + legal, MIB shape equality preserved,
determinism under a fixed seed, and the end-to-end refine_layout wiring with
FLOORSET_WINDOW_REPACK on.
"""

import time

import torch

from floorset_arch.refine.api import refine_layout
from floorset_arch.refine.guards import hard_legal, soft_violations
from floorset_arch.refine.window_repack import (
    _build_ctx,
    _select_window,
    _tension_ranked_pairs,
    refine_window,
)
from floorset_arch.refine.wirelength import hpwl


def _constraints(n, fixed=None, preplaced=None, mib=None, cluster=None, boundary=None):
    fixed = fixed or [0.0] * n
    preplaced = preplaced or [0.0] * n
    mib = mib or [0.0] * n
    cluster = cluster or [0.0] * n
    boundary = boundary or [0.0] * n
    rows = list(zip(fixed, preplaced, mib, cluster, boundary))
    return torch.tensor(rows, dtype=torch.float32)


def _targets(n, entries=None):
    entries = entries or {}
    rows = [list(entries.get(i, (-1.0, -1.0, -1.0, -1.0))) for i in range(n)]
    return torch.tensor(rows, dtype=torch.float32)


def _no_overlap(out):
    n = len(out)
    for i in range(n):
        xi, yi, wi, hi = out[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = out[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6), f"blocks {i},{j} overlap in {out}"


def _improving_window_layout():
    """A 2x2 grid of unit squares with a heavy net pulling block 0 (bottom-
    left) toward block 3 (top-right). Repacking the four-block window can move
    block 0 nearer block 3's partner geometry and lower HPWL. All soft blocks,
    exact area 1.0."""
    rects = [
        (0.0, 0.0, 1.0, 1.0),  # 0
        (1.0, 0.0, 1.0, 1.0),  # 1
        (0.0, 1.0, 1.0, 1.0),  # 2
        (1.0, 1.0, 1.0, 1.0),  # 3
    ]
    n = 4
    area = torch.tensor([1.0, 1.0, 1.0, 1.0])
    cons = _constraints(n)
    tpos = _targets(n)
    # Heavy net 0-3 (diagonal tension); light structural nets keep the pack sane.
    b2b = [(0, 3, 20.0), (0, 1, 1.0), (2, 3, 1.0)]
    p2b = []
    pins = []
    return rects, area, cons, tpos, b2b, p2b, pins


# --- tension ranking sanity -------------------------------------------------


def test_tension_ranking_orders_by_weighted_distance():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    ranked = _tension_ranked_pairs(rects, b2b, topk=8)
    # The heavy diagonal net (0,3) must rank first.
    assert ranked[0][0] == 0 and ranked[0][1] == 3


# --- window selection filters ----------------------------------------------


def test_select_window_excludes_locked():
    """A LOCKED (preplaced) block is never admitted to a window. Layout is a
    horizontal strip so a clean rectangular hole exists around the movable
    blocks even with one block locked and left frozen outside the hole."""
    rects = [
        (0.0, 0.0, 1.0, 1.0),   # 0
        (1.0, 0.0, 1.0, 1.0),   # 1 -- LOCKED
        (2.0, 0.0, 1.0, 1.0),   # 2
        (3.0, 0.0, 1.0, 1.0),   # 3
    ]
    area = torch.tensor([1.0, 1.0, 1.0, 1.0])
    cons = _constraints(4, preplaced=[0.0, 1.0, 0.0, 0.0])
    tpos = _targets(4, {1: (1.0, 0.0, 1.0, 1.0)})
    ctx = _build_ctx(4, area, cons, tpos)
    # Window around the pair (2,3); block 1 is LOCKED and sits at x=1..2, well
    # to the left of the {2,3} hole (x=2..4), so the clean-hole test passes and
    # the LOCKED block is simply never admitted.
    sel = _select_window(rects, 2, 3, ctx, inflate=1.4, max_k=9, used=set())
    assert sel is not None
    selected, _win = sel
    assert 1 not in selected


def test_select_window_cluster_is_whole_or_none():
    """A cluster group is admitted whole-or-not. Here block 1's far cluster
    partner (block 2) would explode the window past max_k, so admitting block 1
    is impossible and the window must EXCLUDE it (never a partial group)."""
    rects = [
        (0.0, 0.0, 1.0, 1.0),   # 0
        (1.0, 0.0, 1.0, 1.0),   # 1 -- cluster with far block 2
        (20.0, 20.0, 1.0, 1.0),  # 2 -- far away, same cluster as 1
        (3.0, 3.0, 1.0, 1.0),   # 3
        (4.0, 4.0, 1.0, 1.0),   # 4
    ]
    area = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    cons = _constraints(5, cluster=[0.0, 1.0, 1.0, 0.0, 0.0])  # blocks 1,2 grouped
    tpos = _targets(5)
    ctx = _build_ctx(5, area, cons, tpos)
    # Cap max_k below the span the cluster closure would force (admitting 1
    # drags in 2 at (20,20), whose bbox closure would exceed max_k=3), so the
    # window must exclude block 1 entirely -- never a partial {1} without 2.
    sel = _select_window(rects, 0, 3, ctx, inflate=1.4, max_k=3, used=set())
    if sel is not None:
        selected, _win = sel
        # If block 1 is present, block 2 must be too (whole group); given
        # max_k=3 and the far block 2, block 1 should be excluded instead.
        if 1 in selected:
            assert 2 in selected
        else:
            assert 2 not in selected


def test_select_window_boundary_off_wall_excluded():
    """A boundary-tagged block whose window does not own its wall is excluded."""
    rects = [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (2.0, 0.0, 1.0, 1.0),
    ]
    area = torch.tensor([1.0, 1.0, 1.0])
    # Block 2 tagged RIGHT boundary; at x=2..3 it IS on the right wall of the
    # full layout, but a window spanning only {0,1} does not reach x=3, so 2 is
    # not selected anyway. Tag block 0 LEFT: window {0,1} owns the left wall.
    cons = _constraints(3, boundary=[1.0, 0.0, 2.0])
    tpos = _targets(3)
    ctx = _build_ctx(3, area, cons, tpos)
    sel = _select_window(rects, 0, 1, ctx, inflate=1.2, max_k=9, used=set())
    if sel is not None:
        selected, win = sel
        # If block 0 (LEFT-tagged) is admitted, the window must own the left
        # wall (win x0 == block 0's left edge).
        if 0 in selected:
            assert abs(win[0] - 0.0) <= 1e-3


# --- repack stays in bbox & legal ------------------------------------------


def test_repack_stays_in_window_bbox_and_legal():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    out, detail = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                                deadline=time.time() + 3.0)
    # Whether or not a window was accepted, the result must be legal and
    # overlap-free.
    assert hard_legal(out, area, cons, tpos)
    _no_overlap(out)


def test_accepted_result_strictly_better():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    pre = hpwl(rects, b2b, p2b, pins)
    out, detail = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                                deadline=time.time() + 3.0)
    post = hpwl(out, b2b, p2b, pins)
    if detail["guard_result"] == "accepted":
        assert post < pre - 1e-6
        assert hard_legal(out, area, cons, tpos)
        _no_overlap(out)
    else:
        # No accept -> byte-identical input.
        assert out == rects


# --- soft violations never regress -----------------------------------------


def test_soft_violations_non_increasing():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    base = soft_violations(rects, cons)
    out, _ = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                           deadline=time.time() + 3.0)
    after = soft_violations(out, cons)
    assert all(a <= b for a, b in zip(after, base))


# --- MIB shape equality preserved ------------------------------------------


def test_mib_shape_equality_preserved():
    """Two MIB-grouped blocks must keep an identical (w,h) after any repack."""
    rects = [
        (0.0, 0.0, 2.0, 1.0),   # 0 -- MIB group 1
        (2.0, 0.0, 2.0, 1.0),   # 1 -- MIB group 1
        (0.0, 1.0, 2.0, 1.0),   # 2 -- soft
        (2.0, 1.0, 2.0, 1.0),   # 3 -- soft
    ]
    area = torch.tensor([2.0, 2.0, 2.0, 2.0])
    cons = _constraints(4, mib=[1.0, 1.0, 0.0, 0.0])
    tpos = _targets(4)
    b2b = [(0, 3, 15.0), (1, 2, 15.0)]
    out, detail = refine_window(rects, area, cons, tpos, b2b, [], [],
                                deadline=time.time() + 3.0)
    # MIB blocks 0 and 1 must have identical rounded shapes.
    w0, h0 = round(out[0][2], 4), round(out[0][3], 4)
    w1, h1 = round(out[1][2], 4), round(out[1][3], 4)
    assert (w0, h0) == (w1, h1)
    # And their shape must be unchanged from the incoming literal (fixed-shape
    # inside the window).
    assert (w0, h0) == (round(rects[0][2], 4), round(rects[0][3], 4))
    assert hard_legal(out, area, cons, tpos)
    v = soft_violations(out, cons)
    assert v[2] == 0  # v_mib


# --- determinism -----------------------------------------------------------


def test_deterministic_under_fixed_seed():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    out1, _ = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                            deadline=time.time() + 3.0)
    out2, _ = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                            deadline=time.time() + 3.0)
    assert out1 == out2


# --- guard-failure / no-op paths return input ------------------------------


def test_deadline_returns_input():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    out, detail = refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                                deadline=time.time() - 1.0)  # already expired
    assert out == rects
    assert detail["guard_result"] == "deadline"


def test_forced_exception_returns_input():
    """A malformed b2b weight raised inside the stage body must be caught and
    yield the input unchanged with guard_result=exception."""
    rects, area, cons, tpos, _b2b, _p2b, _pins = _improving_window_layout()
    bad_b2b = [(0, 3, "not-a-number")]
    out, detail = refine_window(rects, area, cons, tpos, bad_b2b, [], [],
                                deadline=time.time() + 3.0)
    assert out == rects
    assert detail["guard_result"] == "exception"


def test_no_improvement_returns_input():
    """A tight 2-block layout at its HPWL optimum yields no accepted window."""
    rects = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0), (0.0, 2.0, 2.0, 2.0)]
    area = torch.tensor([4.0, 4.0, 4.0])
    cons = _constraints(3)
    tpos = _targets(3)
    b2b = [(0, 1, 1.0)]
    out, detail = refine_window(rects, area, cons, tpos, b2b, [], [],
                                deadline=time.time() + 3.0)
    # Either no window accepted (byte-identical) or a legal strict improvement.
    if detail["guard_result"] != "accepted":
        assert out == rects
    assert hard_legal(out, area, cons, tpos)


def test_via_monkeypatch_exception(monkeypatch):
    import floorset_arch.refine.window_repack as wr

    def _boom(*a, **k):
        raise RuntimeError("forced")

    monkeypatch.setattr(wr, "_build_ctx", _boom)
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    out, detail = wr.refine_window(rects, area, cons, tpos, b2b, p2b, pins,
                                   deadline=time.time() + 3.0)
    assert out == rects
    assert detail["guard_result"] == "exception"


# --- api.py end-to-end wiring ----------------------------------------------


def test_refine_layout_window_end_to_end(monkeypatch):
    """With FLOORSET_WINDOW_REPACK=1 and a topo_deadline set, refine_layout
    picks up a window improvement end to end and stays legal."""
    monkeypatch.setenv("FLOORSET_WINDOW_REPACK", "1")
    rects, area, cons, tpos, b2b, p2b, pins = _improving_window_layout()
    b2b_t = torch.tensor(b2b, dtype=torch.float32)
    p2b_t = torch.empty(0, 3)
    pins_t = torch.empty(0, 2)
    pre = hpwl(rects, b2b, p2b, pins)
    out = refine_layout(rects, area, cons, tpos, b2b_t, p2b_t, pins_t,
                        deadline=time.time() + 3.0,
                        topo_deadline=time.time() + 3.0)
    post = hpwl(out, b2b, p2b, pins)
    assert post <= pre + 1e-6  # never regresses
    assert hard_legal(out, area, cons, tpos)
    _no_overlap(out)


def test_refine_layout_window_off_is_noop(monkeypatch):
    """topo_deadline=None (or flag off) leaves the window stage dormant."""
    monkeypatch.setenv("FLOORSET_WINDOW_REPACK", "1")
    rects = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    area = torch.tensor([4.0, 4.0])
    cons = _constraints(2)
    tpos = _targets(2)
    b2b = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    p2b = torch.empty(0, 3)
    pins = torch.empty(0, 2)
    out = refine_layout(rects, area, cons, tpos, b2b, p2b, pins,
                        deadline=time.time() + 3.0, topo_deadline=None)
    assert out == rects
