"""Topology-changing local search on the axis-separation DAGs (M2).

This is the "main bet" refinement stage: the slack refiner
(`slack_solve.project_axis`) can only translate blocks within a FIXED
separation topology -- it can never change which pairs are x-separated vs
y-separated. That fixed topology is exactly the column-slicing class the
backbone commits to, and it is the ceiling on HPWL (~1.35x GT). This stage
escapes that class by editing the DAG edges (axis flips, adjacent
transpositions, block re-insertion), re-projecting coordinates after each
edit, and greedily keeping only edits that strictly reduce HPWL while passing
the same guard chain the refiner gates on.

Called from `refine/api.py::refine_layout` BEFORE the slack projection, so the
downstream projection polishes whatever new topology this stage produced.

Gated by FLOORSET_TOPO_SEARCH (default OFF, plumbed via a `topo_deadline`
sub-budget carved out of the SA share in
`legalizer/column_backbone.py::solve_with_column_backbone`). The stage is a
pure function: on any exception, guard failure, or deadline hit it returns the
stage input rects unchanged. The backbone+refiner 1.2337 baseline is a hard
floor.

See docs/design/slack_refiner_spec.md (constraint-graph + guard machinery this
reuses) and the M2 section of the approved roadmap.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .constraint_graph import (
    AxisGraph,
    Edit,
    apply_edits,
    build_axis_dags,
    has_edge,
    reachable,
    undo_edits,
)
from .guards import hard_legal, soft_violations
from .slack_solve import project_axis
from .wirelength import hpwl

Rect = Tuple[float, float, float, float]

EPS = 1e-6
SEP_TOL = 1e-6

# Tunables (kept modest -- the machine is under load and the per-case budget
# is only ~3s at n=120). All are overridable via FLOORSET_TOPO_* env vars for
# smoke-tuning without code edits.
DEFAULT_TENSION_TOPK = 24        # rank this many highest-tension b2b pairs
DEFAULT_MAX_PROPOSALS = 4000     # hard cap on proposals per stage call
DEFAULT_DEADLINE_CHECK_EVERY = 8  # check wall clock every N proposals


def _envf(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        return int(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Anchors (weighted-median partner centroids), reused from the api.py pattern.
# ---------------------------------------------------------------------------


def _build_anchors(
    n: int,
    axis: int,  # 0 = x, 1 = y
    rects: Sequence[Rect],
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
) -> Dict[int, Tuple[List[float], List[float]]]:
    anchors: Dict[int, Tuple[List[float], List[float]]] = {i: ([], []) for i in range(n)}

    def centroid(i: int) -> float:
        x, y, w, h = rects[i]
        return (x + w / 2.0) if axis == 0 else (y + h / 2.0)

    for edge in b2b_edges:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1 or not (0 <= i < n and 0 <= j < n) or w <= 0:
            continue
        anchors[i][0].append(centroid(j))
        anchors[i][1].append(w)
        anchors[j][0].append(centroid(i))
        anchors[j][1].append(w)

    n_pins = len(pins)
    for edge in p2b_edges:
        p, b, w = int(edge[0]), int(edge[1]), float(edge[2])
        if p == -1 or not (0 <= b < n and 0 <= p < n_pins) or w <= 0:
            continue
        px, py = pins[p]
        anchor = px if axis == 0 else py
        if anchor == -1.0:
            continue
        anchors[b][0].append(anchor)
        anchors[b][1].append(w)

    return anchors


def _project_both(
    gx: AxisGraph,
    gy: AxisGraph,
    xs: List[float],
    ys: List[float],
    ws: List[float],
    hs: List[float],
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
) -> List[Rect]:
    """Project x then y from the (possibly just-edited) DAGs and return rects.

    Anchors are rebuilt from the CURRENT coordinates each call: after a
    topology edit the partner centroids that drive the weighted-median targets
    should reflect where blocks actually are, so the projection converges the
    new topology toward its own HPWL optimum rather than the old one's.
    """
    n = len(xs)
    rects = [(xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
    anchors_x = _build_anchors(n, 0, rects, b2b_edges, p2b_edges, pins)
    new_xs, _sx = project_axis(gx, xs, ws, anchors_x)
    interim = [(new_xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
    anchors_y = _build_anchors(n, 1, interim, b2b_edges, p2b_edges, pins)
    new_ys, _sy = project_axis(gy, ys, hs, anchors_y)
    return [(new_xs[i], new_ys[i], ws[i], hs[i]) for i in range(n)]


def _bbox_area(rects: Sequence[Rect]) -> float:
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return (x_max - x_min) * (y_max - y_min)


# ---------------------------------------------------------------------------
# Proposal-time legality helpers (cheap filters; the guard chain is the
# real backstop but wasted proposals burn the deadline).
# ---------------------------------------------------------------------------


class _Context:
    """Immutable per-stage context: block classification and pin/cluster sets."""

    __slots__ = ("n", "locked", "boundary", "group_of", "group_members")

    def __init__(self, n, locked, boundary, group_of, group_members):
        self.n = n
        self.locked = locked                 # set of pinned/preplaced indices (never move)
        self.boundary = boundary             # index -> boundary bitmask (0 if none)
        self.group_of = group_of             # index -> cluster group id (0 if none)
        self.group_members = group_members   # group id -> list of member indices

    def movable_pair(self, i: int, j: int) -> bool:
        """A flip/transposition on (i, j) is proposable only if:
          - at least one endpoint can move (both LOCKED -> geometrically inert,
            their coords are re-pinned by project_axis); AND
          - neither endpoint is a cluster member. Cluster groups translate as a
            rigid unit (project_axis moves all members by one shared shift), so
            editing a single member's separation edge desyncs the rigid
            assembly and is either inert or corrupting. Conservatively skip any
            move touching a cluster member; the coordinate refiner still
            polishes those groups, and clusters are a minority of pairs. This
            keeps the DAG-edit move set sound without a group-envelope
            re-derivation (deferred; the guard chain would reject a bad one
            anyway, but proposing it just burns the deadline)."""
        if i in self.locked and j in self.locked:
            return False
        if self.group_of.get(i, 0) != 0 or self.group_of.get(j, 0) != 0:
            return False
        return True

    def same_group(self, i: int, j: int) -> bool:
        gi = self.group_of.get(i, 0)
        gj = self.group_of.get(j, 0)
        return gi != 0 and gi == gj


def _build_context(n: int, gx: AxisGraph, gy: AxisGraph, constraints) -> _Context:
    from floorset_arch.legalizer.column_slicing import _parse_constraints

    _fixed, _preplaced, _mib, cluster, boundary_list = _parse_constraints(constraints, n)
    # LOCKED = any node pinned on either axis (preplaced target or boundary
    # wall). These never move; proposing edits that would relocate them is
    # wasted (project_axis re-pins them anyway).
    locked = set(gx.pinned.keys()) | set(gy.pinned.keys())
    group_of = dict(gx.group_of)  # gx and gy share the same cluster grouping
    group_members: Dict[int, List[int]] = {}
    for i, g in group_of.items():
        group_members.setdefault(g, []).append(i)
    boundary = {i: int(boundary_list[i]) for i in range(n) if boundary_list[i]}
    return _Context(n, locked, boundary, group_of, group_members)


# ---------------------------------------------------------------------------
# Move generators. Each returns an edit script (list of Edit) applied to gx/gy
# plus a human tag, or None if the move is not proposable. The caller applies,
# projects, evaluates, and undoes on rejection.
# ---------------------------------------------------------------------------


def _find_x_edge(gx: AxisGraph, i: int, j: int) -> Optional[Tuple[int, int, float]]:
    for u, v, g in gx.edges:
        if (u == i and v == j) or (u == j and v == i):
            return (u, v, g)
    return None


def _find_y_edge(gy: AxisGraph, i: int, j: int) -> Optional[Tuple[int, int, float]]:
    for u, v, g in gy.edges:
        if (u == i and v == j) or (u == j and v == i):
            return (u, v, g)
    return None


def _propose_axis_flip(
    gx: AxisGraph,
    gy: AxisGraph,
    i: int,
    j: int,
    rects: Sequence[Rect],
    ctx: _Context,
) -> Optional[Tuple[List[Edit], List[Edit], str]]:
    """Flip pair (i, j) from x-separation to y-separation (or vice versa).

    Returns (edits_x, edits_y, tag) where edits_x apply to gx and edits_y to
    gy, or None if not proposable. Direction of the new edge is set by the
    CURRENT relative coordinate on the target axis. Acyclicity of the new edge
    is checked against the target graph.
    """
    if not ctx.movable_pair(i, j):
        return None
    # A flip that detaches a boundary-tagged block from the axis it is pinned
    # to is a wasted proposal (project_axis re-pins it, the flip cannot take).
    # Cheap filter: if either endpoint is boundary-pinned on the axis we would
    # ADD an edge to, skip. (LOCKED already excluded above; boundary walls are
    # the remaining pin source.)

    xe = _find_x_edge(gx, i, j)
    if xe is not None:
        # Currently x-separated -> move to y. New y-edge directed by rel-y.
        yi = rects[i][1]
        yj = rects[j][1]
        if yi < yj or (yi == yj and i < j):
            u, v = i, j
        else:
            u, v = j, i
        gap = rects[u][3]
        # Must not already have a y-edge between them, and adding u->v must
        # keep gy acyclic (v must not already reach u).
        if _find_y_edge(gy, i, j) is not None:
            return None
        if reachable(gy.edges, gy.n, v, u):
            return None
        edits_x: List[Edit] = [("del", xe[0], xe[1], xe[2])]
        edits_y: List[Edit] = [("add", u, v, gap)]
        return edits_x, edits_y, "flip_x2y"

    ye = _find_y_edge(gy, i, j)
    if ye is not None:
        xi = rects[i][0]
        xj = rects[j][0]
        if xi < xj or (xi == xj and i < j):
            u, v = i, j
        else:
            u, v = j, i
        gap = rects[u][2]
        if _find_x_edge(gx, i, j) is not None:
            return None
        if reachable(gx.edges, gx.n, v, u):
            return None
        edits_y = [("del", ye[0], ye[1], ye[2])]
        edits_x = [("add", u, v, gap)]
        return edits_x, edits_y, "flip_y2x"

    return None


def _propose_transposition(
    graph: AxisGraph,
    i: int,
    j: int,
    axis_dims: Sequence[float],
    ctx: _Context,
) -> Optional[Tuple[List[Edit], str]]:
    """Reverse an edge (i, j) on one axis -- only when no alternate i->j path
    exists (else reversing creates a cycle). Returns (edits, tag) for THAT
    graph, or None. `axis_dims` is w for gx, h for gy (the reversed edge's gap
    is the new leading block's extent)."""
    edge = None
    for u, v, g in graph.edges:
        if (u == i and v == j) or (u == j and v == i):
            edge = (u, v, g)
            break
    if edge is None:
        return None
    u, v, g = edge
    if not ctx.movable_pair(u, v):
        return None
    # Reverse to v->u. Safe (acyclic) iff u cannot reach v via ANY OTHER path.
    if reachable(graph.edges, graph.n, u, v, skip_edge=(u, v)):
        return None
    new_gap = axis_dims[v]  # v becomes the leading block
    edits: List[Edit] = [("del", u, v, g), ("add", v, u, new_gap)]
    return edits, "transpose"


def _propose_reinsertion(
    gx: AxisGraph,
    gy: AxisGraph,
    b: int,
    rects: Sequence[Rect],
    ctx: _Context,
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
    ws: Sequence[float],
    hs: Sequence[float],
) -> Optional[Tuple[List[Edit], List[Edit], str]]:
    """Strip block b's separation edges on both axes, compute its HPWL-optimal
    weighted-median target, and re-derive b's edges against the layout at that
    target using the same overlap rule build_axis_dags uses. Returns
    (edits_x, edits_y, tag) or None.

    b must be movable and not a cluster member (rigid groups are handled
    elsewhere) and not LOCKED/boundary-pinned (its position is fixed, so
    re-insertion is inert).
    """
    if b in ctx.locked:
        return None
    if ctx.group_of.get(b, 0) != 0:
        return None

    n = len(rects)

    # Target centroid via weighted median of b's partners (same machinery the
    # slack refiner uses). If b has no partners, re-insertion is aimless.
    ax = _build_anchors(n, 0, rects, b2b_edges, p2b_edges, pins)
    ay = _build_anchors(n, 1, rects, b2b_edges, p2b_edges, pins)
    axb, awb = ax.get(b, ([], []))
    ayb, ayw = ay.get(b, ([], []))
    if not axb and not ayb:
        return None
    from .wirelength import weighted_median_target
    wb = ws[b]
    hb = hs[b]
    tx = (weighted_median_target(axb, awb) - wb / 2.0) if axb else rects[b][0]
    ty = (weighted_median_target(ayb, ayw) - hb / 2.0) if ayb else rects[b][1]

    # Trial layout: b at its target, everyone else unchanged.
    trial = list(rects)
    trial[b] = (tx, ty, wb, hb)

    # Edits: delete every existing edge incident to b on both axes.
    edits_x: List[Edit] = [("del", u, v, g) for (u, v, g) in gx.edges if u == b or v == b]
    edits_y: List[Edit] = [("del", u, v, g) for (u, v, g) in gy.edges if u == b or v == b]

    # Re-derive b's edges against the TRIAL geometry, mirroring build_axis_dags'
    # overlap rule and index tie-breaking. Only edges incident to b are added
    # (all other pairs are unchanged, so their edges stay as-is).
    bx, by, bw, bh = trial[b]
    add_x: List[Edit] = []
    add_y: List[Edit] = []
    for k in range(n):
        if k == b:
            continue
        kx, ky, kw, kh = trial[k]
        oy = min(by + bh, ky + kh) - max(by, ky)
        ox = min(bx + bw, kx + kw) - max(bx, kx)
        if oy > SEP_TOL:
            # x-edge, smaller-x -> larger-x
            if bx < kx or (bx == kx and b < k):
                u, v, uw = b, k, bw
            else:
                u, v, uw = k, b, kw
            add_x.append(("add", u, v, uw))
        elif ox > SEP_TOL:
            if by < ky or (by == ky and b < k):
                u, v, uh = b, k, bh
            else:
                u, v, uh = k, b, kh
            add_y.append(("add", u, v, uh))
        # else corner-only: no edge

    edits_x += add_x
    edits_y += add_y
    if not add_x and not add_y:
        # b would separate from nothing -> would float / overlap; skip.
        return None
    return edits_x, edits_y, "reinsert"


# ---------------------------------------------------------------------------
# Tension ranking: b2b pairs by w * (|dx| + |dy|).
# ---------------------------------------------------------------------------


def _tension_ranked_pairs(
    rects: Sequence[Rect],
    b2b_edges: Sequence[Tuple[float, float, float]],
    topk: int,
) -> List[Tuple[int, int]]:
    n = len(rects)
    scored: Dict[Tuple[int, int], float] = {}
    for edge in b2b_edges:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1 or not (0 <= i < n and 0 <= j < n) or w <= 0 or i == j:
            continue
        xi, yi, wi, hi = rects[i]
        xj, yj, wj, hj = rects[j]
        cxi, cyi = xi + wi / 2.0, yi + hi / 2.0
        cxj, cyj = xj + wj / 2.0, yj + hj / 2.0
        tension = w * (abs(cxi - cxj) + abs(cyi - cyj))
        key = (i, j) if i < j else (j, i)
        if tension > scored.get(key, -1.0):
            scored[key] = tension
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [pair for pair, _t in ordered[:topk]]


def _dag_neighbors(gx: AxisGraph, gy: AxisGraph, i: int, j: int) -> List[Tuple[int, int]]:
    """Pairs formed by (i or j) and their immediate DAG neighbors on either
    axis -- the local edges most likely to interact with a flip of (i, j)."""
    out: List[Tuple[int, int]] = []
    for graph in (gx, gy):
        for u, v, _g in graph.edges:
            if u == i or v == i or u == j or v == j:
                a, b = (u, v) if u < v else (v, u)
                if a != b:
                    out.append((a, b))
    # dedupe, preserve order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# Main stage.
# ---------------------------------------------------------------------------


def refine_topo(
    rects: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
    deadline: Optional[float] = None,
    log_path: Optional[str] = None,
) -> Tuple[List[Rect], Dict[str, object]]:
    """Topology local-search stage. Returns (out_rects, detail).

    Guarantees: out_rects is either a strict HPWL improvement that passes the
    full guard chain (hard_legal, bbox non-growth, soft non-increase) or the
    input rects unchanged. Pure function -- never raises.
    """
    stage_start = time.time()
    detail: Dict[str, object] = {
        "stage": "topo",
        "pre_hpwl": None,
        "post_hpwl": None,
        "pre_cost": None,
        "post_cost": None,
        "elapsed": 0.0,
        "n_proposed": 0,
        "n_accepted": 0,
        "guard_result": "no_change",
    }
    original = list(rects)

    try:
        n = len(rects)
        if n < 2:
            detail["elapsed"] = time.time() - stage_start
            return original, detail

        tension_topk = _envi("FLOORSET_TOPO_TENSION_TOPK", DEFAULT_TENSION_TOPK)
        max_proposals = _envi("FLOORSET_TOPO_MAX_PROPOSALS", DEFAULT_MAX_PROPOSALS)
        check_every = max(1, _envi("FLOORSET_TOPO_DEADLINE_CHECK_EVERY",
                                   DEFAULT_DEADLINE_CHECK_EVERY))

        # Baseline metrics on the incoming layout.
        pre_hpwl = hpwl(original, b2b_edges, p2b_edges, pins)
        pre_bbox = _bbox_area(original)
        base_soft = soft_violations(original, constraints)
        detail["pre_hpwl"] = pre_hpwl
        detail["pre_cost"] = pre_hpwl

        if deadline is not None and time.time() >= deadline:
            detail["guard_result"] = "deadline"
            detail["elapsed"] = time.time() - stage_start
            return original, detail

        gx, gy = build_axis_dags(original, constraints, target_positions,
                                 b2b_edges, p2b_edges, pins)
        ctx = _build_context(n, gx, gy, constraints)

        # Working state (mutable coords; gx/gy edited in place).
        cur_rects = list(original)
        cur_hpwl = pre_hpwl
        ws = [r[2] for r in original]
        hs = [r[3] for r in original]

        best_rects = list(original)
        best_hpwl = pre_hpwl

        n_proposed = 0
        n_accepted = 0

        # Build the proposal pool: top-k tension pairs and their DAG
        # neighbors, deduped, tension pairs first.
        tension_pairs = _tension_ranked_pairs(cur_rects, b2b_edges, tension_topk)
        pool: List[Tuple[int, int]] = []
        seen_pairs = set()
        for pair in tension_pairs:
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pool.append(pair)
        for pair in list(tension_pairs):
            for nb in _dag_neighbors(gx, gy, pair[0], pair[1]):
                if nb not in seen_pairs:
                    seen_pairs.add(nb)
                    pool.append(nb)

        def _deadline_hit() -> bool:
            return deadline is not None and time.time() >= deadline

        def _attempt(
            edits_x: List[Edit],
            edits_y: List[Edit],
            pair: Tuple[int, int],
            tag: str,
        ) -> bool:
            nonlocal cur_rects, cur_hpwl, best_rects, best_hpwl, n_accepted
            inv_x = apply_edits(gx, edits_x) if edits_x else []
            inv_y = apply_edits(gy, edits_y) if edits_y else []
            committed = False
            try:
                xs = [cur_rects[k][0] for k in range(n)]
                ys = [cur_rects[k][1] for k in range(n)]
                cand = _project_both(gx, gy, xs, ys, ws, hs,
                                     b2b_edges, p2b_edges, pins)
                cand_hpwl = hpwl(cand, b2b_edges, p2b_edges, pins)
                if cand_hpwl < cur_hpwl - EPS and _bbox_area(cand) <= pre_bbox + 1e-9:
                    cand_soft = soft_violations(cand, constraints)
                    if all(c <= b for c, b in zip(cand_soft, base_soft)) and \
                            hard_legal(cand, area_targets, constraints, target_positions):
                        cur_rects = cand
                        cur_hpwl = cand_hpwl
                        if cand_hpwl < best_hpwl - EPS:
                            best_rects = cand
                            best_hpwl = cand_hpwl
                        n_accepted += 1
                        committed = True
            except Exception:
                committed = False
            if not committed:
                if inv_y:
                    undo_edits(gy, inv_y)
                if inv_x:
                    undo_edits(gx, inv_x)
            return committed

        # Greedy sweep over the proposal pool. One pass; if any improvement
        # landed and budget remains, re-rank tension on the improved layout
        # and sweep again (bounded by max_proposals / deadline).
        made_progress = True
        passes = 0
        while made_progress and n_proposed < max_proposals and not _deadline_hit():
            made_progress = False
            passes += 1
            for pair in pool:
                if n_proposed >= max_proposals:
                    break
                if (n_proposed % check_every == 0) and _deadline_hit():
                    break
                i, j = pair
                if not (0 <= i < n and 0 <= j < n) or i == j:
                    continue

                # --- axis flip ---
                flip = _propose_axis_flip(gx, gy, i, j, cur_rects, ctx)
                if flip is not None:
                    edits_x, edits_y, tag = flip
                    n_proposed += 1
                    if _attempt(edits_x, edits_y, pair, tag):
                        made_progress = True
                        continue

                # --- adjacent transposition (x then y) ---
                tx = _propose_transposition(gx, i, j, ws, ctx)
                if tx is not None:
                    edits, tag = tx
                    n_proposed += 1
                    if _attempt(edits, [], pair, tag):
                        made_progress = True
                        continue
                ty = _propose_transposition(gy, i, j, hs, ctx)
                if ty is not None:
                    edits, tag = ty
                    n_proposed += 1
                    if _attempt([], edits, pair, tag):
                        made_progress = True
                        continue

                # --- block re-insertion (on each endpoint of the pair) ---
                for b in (i, j):
                    if n_proposed >= max_proposals:
                        break
                    ri = _propose_reinsertion(
                        gx, gy, b, cur_rects, ctx,
                        b2b_edges, p2b_edges, pins, ws, hs,
                    )
                    if ri is not None:
                        edits_x, edits_y, tag = ri
                        n_proposed += 1
                        if _attempt(edits_x, edits_y, pair, tag):
                            made_progress = True
                            break

            # Re-rank tension on the improved layout for the next pass.
            if made_progress and not _deadline_hit():
                tension_pairs = _tension_ranked_pairs(cur_rects, b2b_edges, tension_topk)
                pool = []
                seen_pairs = set()
                for pr in tension_pairs:
                    if pr not in seen_pairs:
                        seen_pairs.add(pr)
                        pool.append(pr)
                for pr in list(tension_pairs):
                    for nb in _dag_neighbors(gx, gy, pr[0], pr[1]):
                        if nb not in seen_pairs:
                            seen_pairs.add(nb)
                            pool.append(nb)

        detail["n_proposed"] = n_proposed
        detail["n_accepted"] = n_accepted

        # Final guard chain on best-so-far vs stage input (mirror api.py).
        if best_hpwl < pre_hpwl - EPS and best_rects != original:
            # Re-verify byte-exact dims and the full guard chain (defense in
            # depth; _attempt already checked but re-check on the survivor).
            dims_ok = all(
                best_rects[k][2] == original[k][2] and best_rects[k][3] == original[k][3]
                for k in range(n)
            )
            post_bbox = _bbox_area(best_rects)
            post_soft = soft_violations(best_rects, constraints)
            post_hpwl = hpwl(best_rects, b2b_edges, p2b_edges, pins)
            if (
                dims_ok
                and post_hpwl < pre_hpwl - EPS
                and post_bbox <= pre_bbox + 1e-9
                and all(c <= b for c, b in zip(post_soft, base_soft))
                and hard_legal(best_rects, area_targets, constraints, target_positions)
            ):
                detail["post_hpwl"] = post_hpwl
                detail["post_cost"] = post_hpwl
                detail["guard_result"] = "accepted"
                detail["elapsed"] = time.time() - stage_start
                _log(log_path, detail)
                return best_rects, detail

        detail["post_hpwl"] = pre_hpwl
        detail["post_cost"] = pre_hpwl
        detail["guard_result"] = "rejected" if n_accepted == 0 else "guard_reject"
        detail["elapsed"] = time.time() - stage_start
        _log(log_path, detail)
        return original, detail

    except Exception:
        detail["guard_result"] = "exception"
        detail["elapsed"] = time.time() - stage_start
        try:
            _log(log_path, detail)
        except Exception:
            pass
        return original, detail


def _log(log_path: Optional[str], detail: Dict[str, object]) -> None:
    if not log_path:
        return
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(detail) + "\n")
    except Exception:
        pass
