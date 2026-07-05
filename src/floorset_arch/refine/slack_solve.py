"""Projected Gauss-Seidel sweeps over one axis's separation DAG.

See docs/design/slack_refiner_spec.md section 2 for the pseudocode this
implements exactly. MAX_ROUNDS=1: x and y are solved independently, once
each, with the DAG fixed at extraction time (never re-derived after moves).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .constraint_graph import AxisGraph
from .wirelength import weighted_median_target

MAX_SWEEPS = 12
BIG = 1e12


def _topo_order(n: int, edges: Sequence[Tuple[int, int, float]]) -> List[int]:
    """Kahn's algorithm topo sort. Edges are (u, v, gap) meaning u -> v
    (u must be placed at or before v on the axis). The constraint-graph
    construction guarantees acyclicity via index tie-breaking."""
    preds: Dict[int, List[int]] = {i: [] for i in range(n)}
    succs: Dict[int, List[int]] = {i: [] for i in range(n)}
    indeg = [0] * n
    for u, v, _g in edges:
        succs[u].append(v)
        preds[v].append(u)
        indeg[v] += 1

    order: List[int] = []
    stack = [i for i in range(n) if indeg[i] == 0]
    # Deterministic order among zero-indegree nodes.
    stack.sort()
    indeg_work = list(indeg)
    idx = 0
    frontier = list(stack)
    while frontier:
        frontier.sort()
        u = frontier.pop(0)
        order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                frontier.append(v)
    if len(order) != n:
        # Should not happen given acyclicity guarantee; fall back to index
        # order rather than raising (caller's try/except in api.py is the
        # real safety net, but avoid infinite loops here).
        seen = set(order)
        order.extend(i for i in range(n) if i not in seen)
    return order


def project_axis(
    graph: AxisGraph,
    rects_coord: List[float],
    dims: List[float],
    anchors_by_block: Dict[int, Tuple[List[float], List[float]]],
) -> Tuple[List[float], int]:
    """Solve one axis via projected Gauss-Seidel sweeps.

    graph: AxisGraph with edges (u, v, gap) meaning coord[v] >= coord[u] + gap.
    rects_coord: current coordinate (x or y) per block index, length n.
    dims: block extent along this axis (w or h), length n.
    anchors_by_block: block index -> (anchor_centroid_list, weight_list) for
        that block's b2b/p2b partners, used for the weighted-median target.

    Returns (new_coords, sweeps_run).
    """
    n = graph.n
    coords = list(rects_coord)

    preds: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    succs: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for u, v, g in graph.edges:
        succs[u].append((v, g))
        preds[v].append((u, g))

    order = _topo_order(n, graph.edges)

    # Group members move together: representative offset per group relative
    # to the group's anchor block (block with min index in the group), so a
    # single scalar shift can be applied to the whole rigid sub-assembly.
    group_members: Dict[int, List[int]] = {}
    for i, g in graph.group_of.items():
        group_members.setdefault(g, []).append(i)
    group_rep: Dict[int, int] = {g: min(members) for g, members in group_members.items()}

    moved_any = True
    sweeps = 0
    direction_forward = True
    while sweeps < MAX_SWEEPS and moved_any:
        moved_any = False
        seq = order if direction_forward else list(reversed(order))
        for i in seq:
            g = graph.group_of.get(i)

            if i in graph.pinned and g is None:
                # Only clamp directly when this node is NOT part of a rigid
                # cluster-group sub-assembly. A pinned node inside a group
                # must move exactly with its group (single shared shift) --
                # its pin is enforced as a bound on that shift below, not as
                # an independent direct assignment, otherwise the group's
                # internal rigid offsets desync (spec section 4: cluster
                # groups translate together as one rigid sub-assembly).
                if coords[i] != graph.pinned[i]:
                    coords[i] = graph.pinned[i]
                    moved_any = True
                continue

            if g is not None and group_rep[g] != i:
                # Non-representative group members are handled when their
                # representative moves (see below); skip direct updates.
                continue

            lo = -BIG
            hi = BIG
            if g is not None:
                members = group_members[g]
            else:
                members = [i]

            # Bounds: aggregate over every member's own predecessor/successor
            # edges (rigid sub-assembly moves as one, so all member
            # constraints apply to the shared shift). Also aggregate every
            # member's universal wall bound (spec section 1: virtual wall
            # nodes WL/WR/WB/WT apply to ALL blocks, not just boundary-tagged
            # ones -- otherwise untagged hull blocks can drift outward and
            # grow the frozen bbox). For a rigid group this yields the
            # group's union extent against the walls, since each member's
            # own wall bound is expressed relative to the representative's
            # shift via the same per-member offset used for edges/pins.
            for m in members:
                offset = coords[m] - coords[i]
                if m in graph.pinned:
                    lo = max(lo, graph.pinned[m] - offset)
                    hi = min(hi, graph.pinned[m] - offset)
                for u, gap in preds.get(m, []):
                    u_coord = coords[u]
                    lo = max(lo, u_coord + gap - offset)
                for v, gap in succs.get(m, []):
                    v_coord = coords[v]
                    hi = min(hi, v_coord - gap - offset)
                # Universal wall bounds: coord[m] >= wall_lo, coord[m] +
                # extent[m] <= wall_hi, for every block (not just boundary-
                # tagged ones).
                lo = max(lo, graph.wall_lo - offset)
                extent_m = graph.dim_extent.get(m, dims[i])
                hi = min(hi, graph.wall_hi - extent_m - offset)

            if lo > hi:
                # Infeasible bound ordering (should not happen on a legal
                # input layout); skip this block rather than corrupt state.
                continue

            anchors, weights = anchors_by_block.get(i, ([], []))
            # Target expressed for the representative's own coordinate:
            # weighted median of anchor centroids minus half this block's
            # own extent (anchors are centroids; coord is the low edge).
            if anchors:
                t_centroid = weighted_median_target(anchors, weights)
                t = t_centroid - dims[i] / 2.0
            else:
                t = coords[i]

            new_i = min(max(t, lo), hi)
            if abs(new_i - coords[i]) > 1e-12:
                moved_any = True
            shift = new_i - coords[i]
            if shift != 0.0:
                for m in members:
                    coords[m] += shift

        sweeps += 1
        direction_forward = not direction_forward

    return coords, sweeps
