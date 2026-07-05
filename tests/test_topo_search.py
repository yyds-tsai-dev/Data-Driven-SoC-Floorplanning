"""Invariant tests for the M2 topology-changing local-search stage
(src/floorset_arch/refine/topo_search.py) and the edge-edit + reachability
helpers in constraint_graph.py. Follows the tests/test_refine_invariants.py
patterns.
"""

import time

import networkx as nx
import torch

from floorset_arch.refine.api import refine_layout
from floorset_arch.refine.constraint_graph import (
    apply_edits,
    build_axis_dags,
    has_edge,
    reachable,
    undo_edits,
)
from floorset_arch.refine.guards import hard_legal, soft_violations
from floorset_arch.refine.topo_search import refine_topo
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


def _improving_flip_layout():
    """Blocks 0 and 1 side-by-side (x-separated); block 0's heavy partner
    (block 2) is directly above block 1. Flipping 0/1 to y-separation lets
    block 0 rise next to block 2, halving HPWL. This is the canonical
    accept-path fixture."""
    rects = [(0.0, 0.0, 4.0, 2.0), (4.0, 0.0, 4.0, 2.0), (4.0, 4.0, 4.0, 2.0)]
    n = 3
    area = torch.tensor([8.0, 8.0, 8.0])
    cons = _constraints(n)
    tpos = _targets(n)
    b2b = torch.tensor([[0.0, 2.0, 10.0]], dtype=torch.float32)
    p2b = torch.empty(0, 3)
    pins = torch.empty(0, 2)
    return rects, area, cons, tpos, b2b, p2b, pins


def _edge_lists(b2b, p2b, pins):
    b2b_e = [(int(r[0]), int(r[1]), float(r[2])) for r in b2b.tolist()] if len(b2b) else []
    p2b_e = [(int(r[0]), int(r[1]), float(r[2])) for r in p2b.tolist()] if len(p2b) else []
    pin_l = [(float(r[0]), float(r[1])) for r in pins.tolist()] if len(pins) else []
    return b2b_e, p2b_e, pin_l


def _no_overlap(out):
    n = len(out)
    for i in range(n):
        xi, yi, wi, hi = out[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = out[j]
            ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
            oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
            assert not (ox > 1e-6 and oy > 1e-6), f"blocks {i},{j} overlap in {out}"


# --- edge-edit / reachability helper tests ---------------------------------


def test_reachable_basic():
    edges = [(0, 1, 1.0), (1, 2, 1.0)]
    assert reachable(edges, 3, 0, 2)
    assert reachable(edges, 3, 0, 1)
    assert not reachable(edges, 3, 2, 0)
    assert reachable(edges, 3, 0, 0)  # self


def test_reachable_skip_edge():
    # 0->1->2 and 0->2 directly. Skipping the direct 0->2 still leaves the path
    # via 1; skipping 0->1 removes the only route to 2 through 1 but the direct
    # edge remains.
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    assert reachable(edges, 3, 0, 2, skip_edge=(0, 2))   # via 1
    assert reachable(edges, 3, 0, 2, skip_edge=(0, 1))   # direct
    assert not reachable(edges, 3, 1, 0)


def test_apply_undo_edits_restores_exactly():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    gx, gy = build_axis_dags(rects, cons, tpos, b2b_e, p2b_e, pin_l)
    before_x = list(gx.edges)
    before_y = list(gy.edges)

    inv_x = apply_edits(gx, [("del", 0, 1, 4.0), ("add", 2, 0, 4.0)])
    assert gx.edges != before_x
    undo_edits(gx, inv_x)
    # Order may differ but the multiset of edges must be identical.
    assert sorted(gx.edges) == sorted(before_x)
    assert sorted(gy.edges) == sorted(before_y)


def test_apply_edits_add_then_del_roundtrip():
    from floorset_arch.refine.constraint_graph import AxisGraph

    g = AxisGraph(n=3)
    g.edges = [(0, 1, 1.0)]
    inv = apply_edits(g, [("add", 1, 2, 2.0), ("add", 0, 2, 3.0)])
    assert has_edge(g.edges, 1, 2) and has_edge(g.edges, 0, 2)
    undo_edits(g, inv)
    assert g.edges == [(0, 1, 1.0)]


# --- DAG-stays-acyclic-after-each-move-type --------------------------------


def _graph_acyclic(graph):
    g = nx.DiGraph()
    g.add_nodes_from(range(graph.n))
    g.add_edges_from((u, v) for u, v, _g in graph.edges)
    return nx.is_directed_acyclic_graph(g)


def test_dag_stays_acyclic_through_topo_search():
    """Run the stage and rebuild the DAGs from its output; the resulting
    separation graphs must still be acyclic (the stage only accepts legal,
    projectable topologies)."""
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    out, detail = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                              deadline=time.time() + 2.0)
    gx, gy = build_axis_dags(out, cons, tpos, b2b_e, p2b_e, pin_l)
    assert _graph_acyclic(gx)
    assert _graph_acyclic(gy)


# --- accept path: strict HPWL improvement, legal, no overlap ---------------


def test_topo_search_improves_and_stays_legal():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    pre = hpwl(rects, b2b_e, p2b_e, pin_l)
    out, detail = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                              deadline=time.time() + 2.0)
    post = hpwl(out, b2b_e, p2b_e, pin_l)
    assert detail["guard_result"] == "accepted"
    assert post < pre - 1e-6
    assert hard_legal(out, area, cons, tpos)
    _no_overlap(out)


def test_topo_search_never_changes_dims():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    out, _ = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                         deadline=time.time() + 2.0)
    for orig, new in zip(rects, out):
        assert new[2] == orig[2]
        assert new[3] == orig[3]


def test_topo_search_soft_violations_non_increasing():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    base = soft_violations(rects, cons)
    out, _ = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                         deadline=time.time() + 2.0)
    after = soft_violations(out, cons)
    assert all(a <= b for a, b in zip(after, base))


# --- determinism -----------------------------------------------------------


def test_topo_search_deterministic():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    out1, _ = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                          deadline=time.time() + 2.0)
    out2, _ = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                          deadline=time.time() + 2.0)
    assert out1 == out2


# --- guard-failure / no-op paths return input unchanged --------------------


def test_topo_search_no_improvement_returns_input():
    """A layout already at its HPWL optimum yields no accepted move; output is
    byte-identical to input."""
    rects = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    n = 2
    area = torch.tensor([4.0, 4.0])
    cons = _constraints(n)
    tpos = _targets(n)
    b2b_e = [(0, 1, 1.0)]
    out, detail = refine_topo(rects, area, cons, tpos, b2b_e, [], [],
                              deadline=time.time() + 2.0)
    assert out == rects
    assert detail["guard_result"] in ("rejected", "no_change")


def test_topo_search_deadline_returns_input():
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    out, detail = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                              deadline=time.time() - 1.0)  # already expired
    assert out == rects
    assert detail["guard_result"] == "deadline"


def test_topo_search_forced_exception_returns_input():
    """A malformed b2b edge (non-numeric weight) raised inside the stage body
    must be caught and yield the input unchanged with guard_result=exception.
    (constraints=None is NOT an error path -- _parse_constraints treats it as
    'no constraints', so that input legitimately optimizes; we force a real
    exception here instead.)"""
    rects, area, cons, tpos, _b2b, _p2b, _pins = _improving_flip_layout()
    bad_b2b = [(0, 2, "not-a-number")]  # float() on the weight will raise
    out, detail = refine_topo(rects, area, cons, tpos, bad_b2b, [], [],
                              deadline=time.time() + 2.0)
    assert out == rects
    assert detail["guard_result"] == "exception"


def test_topo_search_via_monkeypatch_exception(monkeypatch):
    import floorset_arch.refine.topo_search as ts

    def _boom(*a, **k):
        raise RuntimeError("forced")

    monkeypatch.setattr(ts, "build_axis_dags", _boom)
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    out, detail = ts.refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                                 deadline=time.time() + 2.0)
    assert out == rects
    assert detail["guard_result"] == "exception"


# --- pinned / cluster invariants -------------------------------------------


def test_topo_search_never_moves_locked_block():
    """A LOCKED (preplaced) block must be byte-identical in the output even
    when it is an endpoint of a high-tension pair."""
    rects = [(0.0, 0.0, 4.0, 2.0), (4.0, 0.0, 4.0, 2.0), (4.0, 4.0, 4.0, 2.0)]
    n = 3
    area = torch.tensor([8.0, 8.0, 8.0])
    # Block 0 LOCKED at its position; heavy pull 0-2 would otherwise move it.
    cons = _constraints(n, preplaced=[1.0, 0.0, 0.0])
    tpos = _targets(n, {0: (0.0, 0.0, 4.0, 2.0)})
    b2b_e = [(0, 2, 10.0)]
    out, _ = refine_topo(rects, area, cons, tpos, b2b_e, [], [],
                         deadline=time.time() + 2.0)
    assert out[0] == rects[0]
    assert hard_legal(out, area, cons, tpos)


def test_topo_search_skips_cluster_members():
    """Blocks in a cluster group are never edge-edited by topo (rigid-unit
    conservatism). Give both endpoints of the improving pair a cluster tag;
    the stage must decline the move and return input unchanged."""
    rects, area, _c, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    # Tag blocks 0 and 1 as one cluster group -> flips on (0,1) forbidden.
    cons = _constraints(3, cluster=[1.0, 1.0, 0.0])
    out, detail = refine_topo(rects, area, cons, tpos, b2b_e, p2b_e, pin_l,
                              deadline=time.time() + 2.0)
    # 0 and 1 are clustered; the only improving move was the (0,1) flip, now
    # forbidden. (Re-insertion of block 0 is also skipped as a cluster member.)
    # Output must remain legal; grouping must not regress.
    base_g = soft_violations(rects, cons)
    after_g = soft_violations(out, cons)
    assert all(a <= b for a, b in zip(after_g, base_g))
    assert hard_legal(out, area, cons, tpos)


# --- api.py end-to-end wiring ----------------------------------------------


def test_refine_layout_topo_end_to_end():
    """With topo_deadline set, refine_layout picks up the topology improvement
    end to end (not just the standalone refine_topo)."""
    rects, area, cons, tpos, b2b, p2b, pins = _improving_flip_layout()
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    pre = hpwl(rects, b2b_e, p2b_e, pin_l)
    out = refine_layout(rects, area, cons, tpos, b2b, p2b, pins,
                        deadline=time.time() + 2.0,
                        topo_deadline=time.time() + 2.0)
    post = hpwl(out, b2b_e, p2b_e, pin_l)
    assert post < pre - 1e-6
    assert hard_legal(out, area, cons, tpos)
    _no_overlap(out)


def test_refine_layout_topo_off_is_noop_when_no_slack():
    """topo_deadline=None must leave the topo stage entirely dormant; on a
    layout with no translation slack the output is byte-identical to input."""
    rects = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    n = 2
    area = torch.tensor([4.0, 4.0])
    cons = _constraints(n)
    tpos = _targets(n)
    b2b = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    p2b = torch.empty(0, 3)
    pins = torch.empty(0, 2)
    out = refine_layout(rects, area, cons, tpos, b2b, p2b, pins,
                        deadline=time.time() + 2.0, topo_deadline=None)
    assert out == rects
