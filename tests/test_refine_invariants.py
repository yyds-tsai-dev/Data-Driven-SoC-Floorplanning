"""Invariant tests for the Phase-1 slack-redistribution refiner
(src/floorset_arch/refine/). See docs/design/slack_refiner_spec.md
"Tests" section for the numbered list this covers.
"""

import networkx as nx
import torch

from floorset_arch.refine.api import refine_layout
from floorset_arch.refine.constraint_graph import build_axis_dags
from floorset_arch.refine.guards import hard_legal


def _constraints(n, fixed=None, preplaced=None, mib=None, cluster=None, boundary=None):
    fixed = fixed or [0.0] * n
    preplaced = preplaced or [0.0] * n
    mib = mib or [0.0] * n
    cluster = cluster or [0.0] * n
    boundary = boundary or [0.0] * n
    rows = list(zip(fixed, preplaced, mib, cluster, boundary))
    return torch.tensor(rows, dtype=torch.float32)


def _targets(n, entries=None):
    """entries: dict i -> (x, y, w, h); defaults to -1 (unset)."""
    entries = entries or {}
    rows = []
    for i in range(n):
        rows.append(list(entries.get(i, (-1.0, -1.0, -1.0, -1.0))))
    return torch.tensor(rows, dtype=torch.float32)


def _synthetic_layout():
    """5 blocks: block 0 LOCKED (preplaced), block 1 boundary-left tagged,
    blocks 2,3 form a cluster group, block 4 free with slack (off its
    weighted-median target so refine should pull it in and reduce HPWL).

    Layout (x, y, w, h):
      0: LOCKED at (0, 0, 2, 2)
      1: (2, 0, 2, 2)          boundary-left (will be pinned to x_min of bbox)
      2: (4, 0, 2, 2)          cluster group 1
      3: (6, 0, 2, 2)          cluster group 1 (abutting block 2)
      4: (20, 0, 2, 2)         free block, far from its b2b partner (block 3)
    """
    rects = [
        (0.0, 0.0, 2.0, 2.0),
        (0.0, 4.0, 2.0, 2.0),
        (4.0, 0.0, 2.0, 2.0),
        (6.0, 0.0, 2.0, 2.0),
        (20.0, 0.0, 2.0, 2.0),
    ]
    n = 5
    area_targets = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0])
    constraints = _constraints(
        n,
        preplaced=[1.0, 0.0, 0.0, 0.0, 0.0],
        cluster=[0.0, 0.0, 1.0, 1.0, 0.0],
        boundary=[0.0, 1.0, 0.0, 0.0, 0.0],  # block 1: left tag (shares x_min=0 with block 0)
    )
    target_positions = _targets(n, {0: (0.0, 0.0, 2.0, 2.0)})
    # b2b: block 4 <-> block 3 with high weight so its weighted-median
    # target (centroid of block 3) pulls it far left, creating slack that
    # project_axis should exploit within the DAG's bounds.
    b2b = torch.tensor([[4.0, 3.0, 10.0]], dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def test_phase1_never_changes_dims():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins)
    for orig, new in zip(rects, out):
        assert new[2] == orig[2]
        assert new[3] == orig[3]


def test_no_overlap_on_refined_output():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins)
    n = len(out)
    for i in range(n):
        xi, yi, wi, hi = out[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = out[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6)


def test_locked_block_byte_identical_to_target():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins)
    tx, ty, tw, th = target_positions[0].tolist()
    ox, oy, ow, oh = out[0]
    assert ox == tx
    assert oy == ty
    assert ow == tw
    assert oh == th


def test_hpwl_monotone_accept_iff_strictly_decreased():
    from floorset_arch.refine.wirelength import hpwl

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    b2b_edges = [(int(r[0]), int(r[1]), float(r[2])) for r in b2b.tolist()]
    p2b_edges = []
    pin_list = []

    hpwl_before = hpwl(rects, b2b_edges, p2b_edges, pin_list)
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins)
    hpwl_after = hpwl(out, b2b_edges, p2b_edges, pin_list)

    if out != rects:
        assert hpwl_after < hpwl_before - 1e-9
    else:
        assert hpwl_after >= hpwl_before - 1e-9

    # This synthetic layout has deliberate slack (block 4 far from its only
    # b2b partner, block 3, with open space to move left) so refine SHOULD
    # find a strict improvement.
    assert hpwl_after < hpwl_before


def test_forced_exception_returns_original_rects_exactly():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    # Malformed constraints (wrong shape) should be caught by the try/except
    # and yield the original rects unchanged.
    out = refine_layout(rects, area_targets, None, None, None, None, None)
    assert out == rects


def test_forced_exception_via_monkeypatch(monkeypatch):
    import floorset_arch.refine.api as api_module

    def _boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(api_module, "build_axis_dags", _boom)
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    out = api_module.refine_layout(
        rects, area_targets, constraints, target_positions, b2b, p2b, pins)
    assert out == rects


def test_axis_dags_acyclic():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    gx, gy = build_axis_dags(rects, constraints, target_positions, b2b, p2b, pins)

    for graph in (gx, gy):
        g = nx.DiGraph()
        g.add_nodes_from(range(graph.n))
        g.add_edges_from((u, v) for u, v, _gap in graph.edges)
        assert nx.is_directed_acyclic_graph(g)


def test_post_projection_edge_constraints_hold():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    gx, gy = build_axis_dags(rects, constraints, target_positions, b2b, p2b, pins)
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins)

    xs = [r[0] for r in out]
    ys = [r[1] for r in out]
    for u, v, gap in gx.edges:
        assert xs[v] - xs[u] >= gap - 1e-9
    for u, v, gap in gy.edges:
        assert ys[v] - ys[u] >= gap - 1e-9


def test_hard_legal_true_on_synthetic_layout():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _synthetic_layout()
    assert hard_legal(rects, area_targets, constraints, target_positions)


def _group_overlap_layout():
    """Reproduces the group-translation overlap bug: a 2-member rigid
    cluster group (blocks 1, 2) where the group's LEADING member (block 1,
    smaller x) has a tight neighbor immediately to its right (block 3, at
    x=10) while the group's TRAILING member (block 2) has open space to its
    right. A representative-only bound (using only block 1's own
    predecessor/successor edges) would let the group shift further right
    than block 1's neighbor allows if the successor bound were mistakenly
    keyed off block 2 instead of block 1 -- or, conversely, if only block
    2's neighbors were consulted, the shift could be too permissive and
    block 1 would run into block 3. The union-extent bound (aggregating
    every member's own edges into the shared-shift clamp) must catch this
    and keep both members legal.

    Layout (x, y, w, h):
      0: (0, 0, 4, 4)     anchor pulling the group left (heavy b2b to block 1)
      1: (6, 0, 2, 2)     cluster group 1 (leading/leftmost member)
      2: (8, 0, 2, 2)     cluster group 1 (trailing member, abuts block 1)
      3: (10, 0, 2, 2)    fixed neighbor immediately to the right of block 2
                          (tight: only 0 slack between block 2 and block 3)
    """
    rects = [
        (0.0, 0.0, 4.0, 4.0),
        (6.0, 0.0, 2.0, 2.0),
        (8.0, 0.0, 2.0, 2.0),
        (10.0, 0.0, 2.0, 2.0),
    ]
    n = 4
    area_targets = torch.tensor([16.0, 4.0, 4.0, 4.0])
    constraints = _constraints(
        n,
        cluster=[0.0, 1.0, 1.0, 0.0],
    )
    target_positions = _targets(n)
    # Strong pull on block 1 (group leader) toward block 0's centroid (x=2),
    # which is to the LEFT -- but the group has zero slack on the right
    # (block 2 abuts block 3 with no gap), so the group must not move right,
    # and moving left is bounded by block 0's own right edge (x=4).
    b2b = torch.tensor([[1.0, 0.0, 10.0]], dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def test_group_translation_does_not_overlap_tight_neighbor():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _group_overlap_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins)
    n = len(out)
    for i in range(n):
        xi, yi, wi, hi = out[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = out[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6), f"blocks {i},{j} overlap in {out}"
    assert hard_legal(out, area_targets, constraints, target_positions)


# --- Phase 2: aspect moves (FLOORSET_SLACK_REFINE_ASPECT) ---


def _aspect_layout():
    """5 blocks with generous slack so an aspect move can reduce HPWL:
      0: LOCKED (preplaced) anchor at (0, 0, 4, 4)
      1: RIGID (fixed-shape) block at (10, 0, 3, 3) -- must never be resized
      2, 3: MIB group (share one (w,h) literal), currently square 4x4,
            both connect strongly to block 0 (to the left) with plenty of
            open space to the right so widening (rho<1, taller/narrower is
            not what we want -- rho>1 widens and shortens, pulling the
            centroid closer while keeping area exact) can reduce HPWL.
      4: SOFT free block, no constraints, isolated (no pull) so it acts as
         a control that should remain untouched by aspect (no HPWL benefit).
    """
    rects = [
        (0.0, 0.0, 4.0, 4.0),
        (10.0, 0.0, 3.0, 3.0),
        (20.0, 0.0, 4.0, 4.0),
        (20.0, 6.0, 4.0, 4.0),
        (40.0, 40.0, 2.0, 2.0),
    ]
    n = 5
    area_targets = torch.tensor([16.0, 9.0, 16.0, 16.0, 4.0])
    constraints = _constraints(
        n,
        fixed=[0.0, 1.0, 0.0, 0.0, 0.0],
        preplaced=[1.0, 0.0, 0.0, 0.0, 0.0],
        mib=[0.0, 0.0, 1.0, 1.0, 0.0],
    )
    target_positions = _targets(n, {0: (0.0, 0.0, 4.0, 4.0), 1: (-1.0, -1.0, 3.0, 3.0)})
    # Strong pull from blocks 2 and 3 toward block 0's centroid (x=2),
    # which is far to the left -- widening (larger w, smaller h) moves the
    # centroid closer for the same area, reducing HPWL without touching bbox
    # (block 2/3 have open space to their right/below to absorb the size
    # change without pushing into any neighbor or the frozen wall).
    b2b = torch.tensor([[2.0, 0.0, 10.0], [3.0, 0.0, 10.0]], dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def test_aspect_disabled_by_default_no_dim_change():
    """Sanity: without enable_aspect, Phase 1 alone never changes dims (this
    is already covered generally, but re-affirm on the aspect fixture)."""
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _aspect_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=False)
    for orig, new in zip(rects, out):
        assert new[2] == orig[2]
        assert new[3] == orig[3]


def test_aspect_preserves_area_within_tolerance():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _aspect_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=True)
    for i, (x, y, w, h) in enumerate(out):
        target_area = float(area_targets[i])
        if target_area <= 0:
            continue
        actual_area = w * h
        rel_err = abs(actual_area - target_area) / target_area
        assert rel_err <= 0.009 + 1e-9, f"block {i} area error {rel_err} exceeds 0.9% cap"


def test_aspect_mib_group_shares_identical_dims():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _aspect_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=True)
    w2, h2 = round(out[2][2], 4), round(out[2][3], 4)
    w3, h3 = round(out[3][2], 4), round(out[3][3], 4)
    assert (w2, h2) == (w3, h3)


def test_aspect_does_not_change_bbox():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _aspect_layout()
    bbox_before = _bbox_area_helper(rects)
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=True)
    bbox_after = _bbox_area_helper(out)
    assert bbox_after <= bbox_before + 1e-6


def test_aspect_no_overlap():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _aspect_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=True)
    n = len(out)
    for i in range(n):
        xi, yi, wi, hi = out[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = out[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6)


def test_aspect_leaves_locked_and_rigid_untouched():
    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _aspect_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=True)
    # block 0: LOCKED -- byte-identical to target_positions.
    tx, ty, tw, th = target_positions[0].tolist()
    ox, oy, ow, oh = out[0]
    assert (ox, oy, ow, oh) == (tx, ty, tw, th)
    # block 1: RIGID (fixed-shape) -- dims must be byte-identical to input.
    assert out[1][2] == rects[1][2]
    assert out[1][3] == rects[1][3]


def _bbox_area_helper(rects):
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return (x_max - x_min) * (y_max - y_min)


# --- Phase-V: violation snap (FLOORSET_SLACK_REFINE_VSNAP) ---


def _vsnap_free_layout():
    """3 blocks, bbox is (0,0)-(10,6). Block 1 is left-tagged but sits 2
    units off the left wall (x=2) on its OWN row (y=4..6, clear of block 0
    at y=0..2) so nothing blocks its path back to x=0 -- vsnap's boundary
    sub-pass should snap it exactly onto x=0 and drop v_boundary to 0. No
    b2b/p2b pulls so phase1/aspect are no-ops and any HPWL change is purely
    from the snap."""
    rects = [
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 4.0, 2.0, 2.0),   # left-tagged, 2 units of free slack to x=0, own row
        (8.0, 0.0, 2.0, 6.0),   # sets the right/top bbox wall, untagged
    ]
    n = 3
    area_targets = torch.tensor([4.0, 4.0, 12.0])
    constraints = _constraints(
        n,
        boundary=[0.0, 1.0, 0.0],  # block 1: left tag
    )
    target_positions = _targets(n)
    b2b = torch.empty(0, 3, dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def _vsnap_blocked_layout():
    """Same bbox, but block 0 sits directly between block 1 and the left
    wall (block 0 spans x in [0,2], block 1 sits at x=2 abutting it) so
    v1's fixed-order pin-and-project cannot reach x=0 for block 1 (v1's
    reachability check fails / hard_legal rejects the naive pin). v2's
    bounded local reorder SHOULD then successfully swap block 0 and block 1
    (block 0 is unconstrained, has room to move right into the bbox) and
    resolve the violation -- this fixture now exercises the v2 success path
    (see test_vsnap_v2_reorders_movable_neighbor), not a permanent skip."""
    rects = [
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 2.0, 2.0),   # left-tagged; block 0 blocks v1's direct path to x=0
        (8.0, 0.0, 2.0, 4.0),
    ]
    n = 3
    area_targets = torch.tensor([4.0, 4.0, 8.0])
    constraints = _constraints(
        n,
        boundary=[0.0, 1.0, 0.0],
    )
    target_positions = _targets(n)
    b2b = torch.empty(0, 3, dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def _vsnap_v2_locked_blocker_layout():
    """Block 0 (the blocker between block 1 and the left wall) is LOCKED at
    a position that overlaps the wall-adjacent slot block 1 would need to
    occupy after a swap -- v2 must detect this via the locked-blocker check
    and skip block 1 entirely (unfixable), leaving the layout unchanged."""
    rects = [
        (0.0, 0.0, 2.0, 2.0),   # LOCKED, occupies exactly the wall slot block 1 needs
        (2.0, 0.0, 2.0, 2.0),   # left-tagged, blocked by LOCKED block 0
        (8.0, 0.0, 2.0, 4.0),
    ]
    n = 3
    area_targets = torch.tensor([4.0, 4.0, 8.0])
    constraints = _constraints(
        n,
        preplaced=[1.0, 0.0, 0.0],
        boundary=[0.0, 1.0, 0.0],
    )
    target_positions = _targets(n, {0: (0.0, 0.0, 2.0, 2.0)})
    b2b = torch.empty(0, 3, dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def _vsnap_v2_no_room_layout():
    """Block 0 (the blocker) is sandwiched tightly: block -1-equivalent is
    the frozen left wall itself (block 0 already touches x=0), and directly
    to block 0's right sits a LOCKED block 2 (preplaced, exact target, zero
    gap to block 0) which occupies the only slot block 0 could swap into.
    Block 1 is further right, left-tagged, blocked by the combination of
    block 0 then LOCKED block 2 -- with LOCKED block 2 immovable and
    already abutting block 0 with zero gap, block 0 has no reachable slot
    to its right within the blocking-set displacement, so v2 must roll
    back (hard_legal / edge check trips) and leave the layout unchanged."""
    rects = [
        (0.0, 0.0, 2.0, 2.0),   # blocker, already at x=0, no slack to its left
        (2.0, 0.0, 2.0, 2.0),   # LOCKED, preplaced exactly here, zero gap to block 0
        (4.0, 0.0, 2.0, 2.0),   # left-tagged, blocked by block 0 (and LOCKED block 1 beyond it)
    ]
    n = 3
    area_targets = torch.tensor([4.0, 4.0, 4.0])
    constraints = _constraints(
        n,
        preplaced=[0.0, 1.0, 0.0],
        boundary=[0.0, 0.0, 1.0],
    )
    target_positions = _targets(n, {1: (2.0, 0.0, 2.0, 2.0)})
    b2b = torch.empty(0, 3, dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def _vsnap_locked_layout():
    """Block 1 is LOCKED (preplaced with a valid target) AND boundary-tagged
    but sits off its wall on its own row (otherwise reachable, same free
    slack as _vsnap_free_layout) -- per contest QA this pre-existing
    violation is unavoidable and vsnap must explicitly skip it (reason
    'locked'), never moving a LOCKED block even though it COULD reach the
    wall geometrically."""
    rects = [
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 4.0, 2.0, 2.0),   # LOCKED, left-tagged, off-wall (pre-existing violation)
        (8.0, 0.0, 2.0, 6.0),
    ]
    n = 3
    area_targets = torch.tensor([4.0, 4.0, 12.0])
    constraints = _constraints(
        n,
        preplaced=[0.0, 1.0, 0.0],
        boundary=[0.0, 1.0, 0.0],
    )
    target_positions = _targets(n, {1: (2.0, 4.0, 2.0, 2.0)})
    b2b = torch.empty(0, 3, dtype=torch.float32)
    p2b = torch.empty(0, 3, dtype=torch.float32)
    pins = torch.empty(0, 2, dtype=torch.float32)
    return rects, area_targets, constraints, target_positions, b2b, p2b, pins


def _env_vsnap_only(monkeypatch):
    monkeypatch.setenv("FLOORSET_SLACK_REFINE_VSNAP", "1")


def test_vsnap_snaps_block_exactly_to_wall(monkeypatch):
    from floorset_arch.refine.vsnap import refine_vsnap
    from floorset_arch.refine.guards import soft_violations

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_free_layout()
    b2b_edges = []
    p2b_edges = []
    pin_list = []

    v_boundary_before, _vg, _vm = soft_violations(rects, constraints)
    assert v_boundary_before == 1

    out, detail = refine_vsnap(
        rects, area_targets, constraints, target_positions,
        b2b_edges, p2b_edges, pin_list,
    )

    assert abs(out[1][0] - 0.0) < 1e-9
    v_boundary_after, _vg2, _vm2 = soft_violations(out, constraints)
    assert v_boundary_after == 0
    assert detail["snaps_accepted"] >= 1


def test_vsnap_v2_reorders_movable_neighbor():
    """v1 alone cannot fix this (fixed-order projection can't move block 0
    out of the way); v2's bounded local reorder must snap block 1 to the
    wall by rigidly displacing block 0 rightward, with no resulting overlap
    and v_boundary dropping to 0."""
    from floorset_arch.refine.vsnap import refine_vsnap
    from floorset_arch.refine.guards import soft_violations

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_blocked_layout()
    b2b_edges = []
    p2b_edges = []
    pin_list = []

    v_boundary_before, _vg, _vm = soft_violations(rects, constraints)
    assert v_boundary_before == 1

    out, detail = refine_vsnap(
        rects, area_targets, constraints, target_positions,
        b2b_edges, p2b_edges, pin_list,
    )

    assert abs(out[1][0] - 0.0) < 1e-9
    # Block 0 (the neighbor) must have shifted right, out of block 1's way.
    assert out[0][0] > rects[0][0]
    v_boundary_after, _vg2, _vm2 = soft_violations(out, constraints)
    assert v_boundary_after == 0
    assert detail["snaps_accepted"] >= 1

    n = len(out)
    for i in range(n):
        xi, yi, wi, hi = out[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = out[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6)


def test_vsnap_v2_skips_locked_blocker():
    from floorset_arch.refine.vsnap import refine_vsnap

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_v2_locked_blocker_layout()
    b2b_edges = []
    p2b_edges = []
    pin_list = []

    out, detail = refine_vsnap(
        rects, area_targets, constraints, target_positions,
        b2b_edges, p2b_edges, pin_list,
    )

    assert out == rects
    assert detail["snaps_accepted"] == 0
    a2_attempts = [a for a in detail["attempts"] if a.get("pass") == "A2"]
    assert a2_attempts, f"expected at least one A2 attempt, got {detail['attempts']}"
    assert any(a.get("reason") == "locked_blocker" for a in a2_attempts)


def test_vsnap_v2_rolls_back_when_no_room_for_neighbor():
    from floorset_arch.refine.vsnap import refine_vsnap

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_v2_no_room_layout()
    b2b_edges = []
    p2b_edges = []
    pin_list = []

    out, detail = refine_vsnap(
        rects, area_targets, constraints, target_positions,
        b2b_edges, p2b_edges, pin_list,
    )

    assert out == rects
    assert detail["snaps_accepted"] == 0


def test_vsnap_skips_locked_block():
    from floorset_arch.refine.vsnap import refine_vsnap

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_locked_layout()
    b2b_edges = []
    p2b_edges = []
    pin_list = []

    out, detail = refine_vsnap(
        rects, area_targets, constraints, target_positions,
        b2b_edges, p2b_edges, pin_list,
    )

    # LOCKED block must never move.
    assert out[1] == rects[1]
    locked_attempts = [a for a in detail["attempts"]
                        if a.get("block") == 1 and a.get("status") == "skipped"]
    assert locked_attempts, f"expected a skipped/locked attempt for block 1, got {detail['attempts']}"
    assert locked_attempts[0]["reason"] == "locked"


def test_vsnap_rolls_back_on_manufactured_overlap(monkeypatch):
    """Force hard_legal to report False for any candidate so every snap
    attempt must be rolled back, leaving the layout byte-identical."""
    import floorset_arch.refine.vsnap as vsnap_module

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_free_layout()
    b2b_edges = []
    p2b_edges = []
    pin_list = []

    monkeypatch.setattr(vsnap_module, "hard_legal", lambda *a, **k: False)

    out, detail = vsnap_module.refine_vsnap(
        rects, area_targets, constraints, target_positions,
        b2b_edges, p2b_edges, pin_list,
    )

    assert out == rects
    assert detail["snaps_accepted"] == 0
    rolled_back = [a for a in detail["attempts"] if a.get("status") == "rolled_back"]
    assert rolled_back, f"expected at least one rolled_back attempt, got {detail['attempts']}"
    assert rolled_back[0]["reason"] == "hard_legal"


def test_refine_layout_vsnap_end_to_end(monkeypatch):
    """Full refine_layout with FLOORSET_SLACK_REFINE_VSNAP=1 must also pick up
    the snap (not just the standalone refine_vsnap call)."""
    _env_vsnap_only(monkeypatch)
    from floorset_arch.refine.guards import soft_violations

    rects, area_targets, constraints, target_positions, b2b, p2b, pins = _vsnap_free_layout()
    out = refine_layout(rects, area_targets, constraints, target_positions, b2b, p2b, pins,
                         enable_aspect=False)

    v_boundary_after, _vg, _vm = soft_violations(out, constraints)
    assert v_boundary_after == 0


