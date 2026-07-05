"""Phase-V violation snap: per-block wall snap + grouping snap.

See docs/design/slack_refiner_spec.md for the surrounding Phase-1/Phase-2
refiner this runs after. `refine_vsnap` is called from api.py AFTER the
existing phase1 slack-solve + aspect passes, gated by
`FLOORSET_SLACK_REFINE_VSNAP=1`.

Sub-pass A: per-block boundary-tag wall snap (re-pin a violating block to
its required wall coordinate and re-project that axis's DAG).
Sub-pass B: grouping snap (rigid-translate the smallest disconnected shapely
component of a cluster group toward the largest component).

Both sub-passes only ever ACCEPT a candidate that: keeps hard legality, does
not regress any of the three soft-violation counts (other than the one being
targeted, which must strictly improve), keeps the bbox frozen, and keeps HPWL
within a `1 + 1/N_soft` incremental guard band. Anything else is rolled back.
This module never raises out to its caller: `refine_vsnap` returns the input
rects unchanged on any exception.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import box
from shapely.ops import unary_union

from floorset_arch.legalizer.column_slicing import _parse_constraints, _target

from .constraint_graph import (
    build_axis_dags,
    AxisGraph,
    BOUND_LEFT,
    BOUND_RIGHT,
    BOUND_TOP,
    BOUND_BOTTOM,
    SEP_TOL,
)
from .guards import hard_legal, soft_violations, soft_violations_detail
from .slack_solve import project_axis
from .wirelength import hpwl

Rect = Tuple[float, float, float, float]

REACH_TOL = 1e-9
BBOX_TOL = 1e-9


def _bbox_area(rects: Sequence[Rect]) -> float:
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return (x_max - x_min) * (y_max - y_min)


def _build_anchors(
    n: int,
    axis: int,
    rects: Sequence[Rect],
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
) -> Dict[int, Tuple[List[float], List[float]]]:
    """Duplicated from api.py::_build_anchors (kept import-free to avoid a
    circular import between api.py and vsnap.py; logic must stay identical)."""
    anchors: Dict[int, Tuple[List[float], List[float]]] = {i: ([], []) for i in range(n)}

    def centroid(i: int) -> float:
        x, y, w, h = rects[i]
        return (x + w / 2.0) if axis == 0 else (y + h / 2.0)

    for edge in b2b_edges:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1 or not (0 <= i < n and 0 <= j < n) or w <= 0:
            continue
        ci = centroid(j)
        cj = centroid(i)
        anchors[i][0].append(ci)
        anchors[i][1].append(w)
        anchors[j][0].append(cj)
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


def _n_soft(constraints, n: int) -> int:
    """Same formula as column_slicing.py::_build_soft_norm's n_soft_den:
    count(boundary>0) + sum(max(0,len(mib_group)-1)) +
    sum(max(0,len(cluster_group)-1)), floored at 1."""
    _fixed, _preplaced, mib, cluster, boundary = _parse_constraints(constraints, n)
    n_soft = sum(1 for i in range(n) if boundary[i] > 0)

    mib_groups: Dict[int, List[int]] = {}
    for i in range(n):
        if mib[i] > 0:
            mib_groups.setdefault(mib[i], []).append(i)
    for idxs in mib_groups.values():
        n_soft += max(0, len(idxs) - 1)

    cluster_groups: Dict[int, List[int]] = {}
    for i in range(n):
        if cluster[i] > 0:
            cluster_groups.setdefault(cluster[i], []).append(i)
    for idxs in cluster_groups.values():
        n_soft += max(0, len(idxs) - 1)

    return max(n_soft, 1)


def _log_attempt(log_path: Optional[str], record: Dict[str, object]) -> None:
    if not log_path:
        return
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _is_locked(preplaced: Sequence[bool], target_positions, i: int) -> bool:
    if not preplaced[i]:
        return False
    tx, ty, tw, th = _target(target_positions, i)
    return tx >= 0 and ty >= 0 and tw > 0 and th > 0


def _project_full(
    rects: List[Rect],
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
    extra_pin_x: Optional[Tuple[int, float]] = None,
    extra_pin_y: Optional[Tuple[int, float]] = None,
) -> List[Rect]:
    """Rebuild fresh axis DAGs on `rects`, optionally overwrite one extra pin
    per axis, then re-run project_axis on the implicated axis/axes only (the
    axis with no extra pin is left as-is, since nothing there changed)."""
    n = len(rects)
    gx, gy = build_axis_dags(rects, constraints, target_positions, b2b_edges, p2b_edges, pin_list)

    xs = [r[0] for r in rects]
    ys = [r[1] for r in rects]
    ws = [r[2] for r in rects]
    hs = [r[3] for r in rects]

    new_xs, new_ys = xs, ys

    if extra_pin_x is not None:
        b, val = extra_pin_x
        gx.pinned[b] = val
        anchors_x = _build_anchors(n, 0, rects, b2b_edges, p2b_edges, pin_list)
        new_xs, _sweeps = project_axis(gx, xs, ws, anchors_x)

    if extra_pin_y is not None:
        b, val = extra_pin_y
        gy.pinned[b] = val
        interim = [(new_xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
        anchors_y = _build_anchors(n, 1, interim, b2b_edges, p2b_edges, pin_list)
        new_ys, _sweeps = project_axis(gy, ys, hs, anchors_y)

    return [(new_xs[i], new_ys[i], ws[i], hs[i]) for i in range(n)], gx, gy


def _full_phase1_repolish(
    rects: List[Rect],
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
) -> List[Rect]:
    """Rebuild fresh x/y axis DAGs from `rects` (same extraction as the
    initial phase-1 solve) and run one full projected-Gauss-Seidel pass on
    each axis, independently, to re-polish HPWL after a v2 reorder. This
    mirrors api.py::refine_layout's phase-1 x-then-y solve exactly."""
    n = len(rects)
    gx, gy = build_axis_dags(rects, constraints, target_positions, b2b_edges, p2b_edges, pin_list)

    xs = [r[0] for r in rects]
    ys = [r[1] for r in rects]
    ws = [r[2] for r in rects]
    hs = [r[3] for r in rects]

    anchors_x = _build_anchors(n, 0, rects, b2b_edges, p2b_edges, pin_list)
    new_xs, _sweeps_x = project_axis(gx, xs, ws, anchors_x)

    interim = [(new_xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
    anchors_y = _build_anchors(n, 1, interim, b2b_edges, p2b_edges, pin_list)
    new_ys, _sweeps_y = project_axis(gy, ys, hs, anchors_y)

    return [(new_xs[i], new_ys[i], ws[i], hs[i]) for i in range(n)]


def _blocking_set_for_edge(
    working: List[Rect],
    block_id: int,
    bit: int,
    gx: AxisGraph,
    gy: AxisGraph,
) -> Optional[List[int]]:
    """Blocking set C for a single missing bit (BOUND_LEFT/RIGHT/TOP/BOTTOM):
    blocks whose orthogonal-band interval overlaps block_id's band by
    > SEP_TOL, and whose position along the tagged axis lies strictly
    between the wall and block_id's current position (i.e. literally in the
    way of block_id reaching that wall). Returns None if block_id's own
    band can't be computed (defensive, should not happen)."""
    n = len(working)
    bx, by, bw, bh = working[block_id]

    if bit in (BOUND_LEFT, BOUND_RIGHT):
        # Tagged axis = x. Orthogonal band = y-interval [by, by+bh].
        b_y0, b_y1 = by, by + bh
        blocking = []
        for c in range(n):
            if c == block_id:
                continue
            cx, cy, cw, ch = working[c]
            oy = min(b_y1, cy + ch) - max(b_y0, cy)
            if oy <= SEP_TOL:
                continue
            if bit == BOUND_LEFT:
                # Between wall (x0) and block_id: cx in (x0, bx).
                if gx.wall_lo < cx < bx - SEP_TOL or (cx < bx - SEP_TOL and cx + cw > gx.wall_lo + SEP_TOL):
                    blocking.append(c)
            else:  # BOUND_RIGHT
                if bx + bw + SEP_TOL < cx + cw < gx.wall_hi + SEP_TOL or (cx + cw > bx + bw + SEP_TOL and cx < gx.wall_hi - SEP_TOL):
                    blocking.append(c)
        return blocking
    else:
        # Tagged axis = y. Orthogonal band = x-interval [bx, bx+bw].
        b_x0, b_x1 = bx, bx + bw
        blocking = []
        for c in range(n):
            if c == block_id:
                continue
            cx, cy, cw, ch = working[c]
            ox = min(b_x1, cx + cw) - max(b_x0, cx)
            if ox <= SEP_TOL:
                continue
            if bit == BOUND_BOTTOM:
                if gy.wall_lo < cy < by - SEP_TOL or (cy < by - SEP_TOL and cy + ch > gy.wall_lo + SEP_TOL):
                    blocking.append(c)
            else:  # BOUND_TOP
                if by + bh + SEP_TOL < cy + ch < gy.wall_hi + SEP_TOL or (cy + ch > by + bh + SEP_TOL and cy < gy.wall_hi - SEP_TOL):
                    blocking.append(c)
        return blocking


def _try_v2_reorder_snap(
    working: List[Rect],
    block_id: int,
    bit: int,
    area_targets,
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
    n_soft: int,
    attempts: List[Dict[str, object]],
) -> Tuple[List[Rect], bool]:
    """v2 bounded local reorder for a single missing bit. Pins block_id at
    its wall coordinate, removes block_id's existing axis edges on the
    tagged axis, adds block_id -> c (gap = block_id's own extent) for every
    c in the blocking set C (so C is forced to the far side of block_id),
    re-projects that axis, then rebuilds BOTH axis DAGs from the resulting
    geometry and runs one full phase-1 re-polish pass before the final
    acceptance gate."""
    n = len(working)
    _fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)

    old_v_boundary, old_v_grouping, old_v_mib = soft_violations(working, constraints)
    hpwl_old = hpwl(working, b2b_edges, p2b_edges, pin_list)
    bbox_old = _bbox_area(working)

    gx0, gy0 = build_axis_dags(working, constraints, target_positions, b2b_edges, p2b_edges, pin_list)

    axis = 0 if bit in (BOUND_LEFT, BOUND_RIGHT) else 1
    g0 = gx0 if axis == 0 else gy0

    blocking = _blocking_set_for_edge(working, block_id, bit, gx0, gy0)
    if blocking is None:
        attempts.append({"block": block_id, "pass": "A2", "tag": bit, "status": "skipped", "reason": "no_blocking_set"})
        return working, False

    if not blocking:
        # Nothing actually blocks the path; v1 should have handled this
        # (or it's genuinely unreachable for another reason). Nothing new
        # for v2 to do.
        attempts.append({"block": block_id, "pass": "A2", "tag": bit, "status": "skipped", "reason": "no_blockers"})
        return working, False

    bw_val = working[block_id][2] if axis == 0 else working[block_id][3]
    wall_val = g0.wall_lo if bit in (BOUND_LEFT, BOUND_BOTTOM) else g0.wall_hi - bw_val

    # LOCKED blocker occupying the target wall slot -> unfixable, skip.
    for c in blocking:
        if _is_locked(preplaced, target_positions, c):
            cx, cy, cw, ch = working[c]
            bx, by, bw, bh = working[block_id]
            if axis == 0:
                slot_x0, slot_x1 = (wall_val, wall_val + bw) if bit == BOUND_LEFT else (wall_val, wall_val + bw)
                ox = min(slot_x1, cx + cw) - max(slot_x0, cx)
                oy = min(by + bh, cy + ch) - max(by, cy)
                overlaps = ox > SEP_TOL and oy > SEP_TOL
            else:
                slot_y0, slot_y1 = (wall_val, wall_val + bh) if bit == BOUND_BOTTOM else (wall_val, wall_val + bh)
                oy = min(slot_y1, cy + ch) - max(slot_y0, cy)
                ox = min(bx + bw, cx + cw) - max(bx, cx)
                overlaps = ox > SEP_TOL and oy > SEP_TOL
            if overlaps:
                attempts.append({"block": block_id, "pass": "A2", "tag": bit, "status": "skipped", "reason": "locked_blocker"})
                return working, False

    try:
        # Build the modified axis DAG: same graph, but block_id's existing
        # edges on this axis are removed and replaced with block_id -> c
        # (gap = block_id's own extent) for every c in the blocking set.
        g_mod = AxisGraph(
            n=g0.n,
            edges=[e for e in g0.edges if e[0] != block_id and e[1] != block_id],
            pinned=dict(g0.pinned),
            group_of=dict(g0.group_of),
            wall_lo=g0.wall_lo,
            wall_hi=g0.wall_hi,
            dim_extent=dict(g0.dim_extent),
        )
        g_mod.pinned[block_id] = wall_val
        # A blocker that is itself pinned to this SAME wall (boundary-tagged
        # and already touching it, per build_axis_dags's boundary-pin logic)
        # must be free to vacate that wall slot for block_id -- otherwise
        # the swap is trivially infeasible by construction (two nodes both
        # pinned to the identical coordinate). LOCKED blockers are already
        # excluded above (unfixable, skipped before reaching here); any
        # remaining pin on a blocker at this specific wall value can only be
        # the boundary-tag pin, which this reorder is explicitly meant to
        # relocate, so it is safe to drop.
        for c in blocking:
            if c in g_mod.pinned and abs(g_mod.pinned[c] - wall_val) <= REACH_TOL:
                del g_mod.pinned[c]

        # Edge direction depends on which wall block_id is snapping to: for
        # the LOW walls (left/bottom) block_id must end up BEFORE every
        # blocker on this axis (edge block_id -> c). For the HIGH walls
        # (right/top) block_id must end up AFTER every blocker instead
        # (edge c -> block_id), since block_id is moving to the far end.
        low_wall = bit in (BOUND_LEFT, BOUND_BOTTOM)
        for c in blocking:
            c_extent = g0.dim_extent.get(c, working[c][2] if axis == 0 else working[c][3])
            if low_wall:
                g_mod.edges.append((block_id, c, bw_val))
            else:
                g_mod.edges.append((c, block_id, c_extent))

        coord = [r[0] if axis == 0 else r[1] for r in working]
        dims = [r[2] if axis == 0 else r[3] for r in working]
        anchors = _build_anchors(n, axis, working, b2b_edges, p2b_edges, pin_list)
        new_coord, _sweeps = project_axis(g_mod, coord, dims, anchors)

        if abs(new_coord[block_id] - wall_val) > REACH_TOL:
            attempts.append({"block": block_id, "pass": "A2", "tag": bit, "status": "rolled_back", "reason": "unreachable_after_reorder"})
            return working, False

        # Check every edge in the modified DAG still holds (no lo>hi
        # violation slipped through as a silent skip in project_axis).
        for u, v, gap in g_mod.edges:
            if new_coord[v] - new_coord[u] < gap - 1e-7:
                attempts.append({"block": block_id, "pass": "A2", "tag": bit, "status": "rolled_back", "reason": "edge_violated"})
                return working, False

        if axis == 0:
            interim = [(new_coord[i], working[i][1], working[i][2], working[i][3]) for i in range(n)]
        else:
            interim = [(working[i][0], new_coord[i], working[i][2], working[i][3]) for i in range(n)]

        # Rebuild both axis DAGs from the resulting geometry and run one
        # full phase-1 re-polish pass (step 4 of the spec).
        candidate = _full_phase1_repolish(
            interim, constraints, target_positions, b2b_edges, p2b_edges, pin_list,
        )
    except Exception:
        attempts.append({"block": block_id, "pass": "A2", "tag": bit, "status": "rolled_back", "reason": "exception"})
        return working, False

    bbox_new = _bbox_area(candidate)
    hpwl_new = hpwl(candidate, b2b_edges, p2b_edges, pin_list)
    new_v_boundary, new_v_grouping, new_v_mib = soft_violations(candidate, constraints)
    hpwl_ratio = (hpwl_new / hpwl_old) if hpwl_old > 0 else 1.0

    rejection = None
    if not hard_legal(candidate, area_targets, constraints, target_positions):
        rejection = "hard_legal"
    elif abs(bbox_new - bbox_old) > BBOX_TOL:
        rejection = "bbox_changed"
    elif not (new_v_boundary <= old_v_boundary - 1):
        rejection = "v_boundary_not_improved"
    elif new_v_grouping > old_v_grouping:
        rejection = "v_grouping_regressed"
    elif new_v_mib > old_v_mib:
        rejection = "v_mib_regressed"
    elif not (hpwl_new <= hpwl_old * (1.0 + 1.0 / n_soft) + 1e-9):
        rejection = "hpwl_guard"

    if rejection is None:
        attempts.append({
            "block": block_id, "pass": "A2", "tag": bit, "status": "snapped",
            "hpwl_ratio": hpwl_ratio, "blocking_set": blocking,
            "v_boundary_before": old_v_boundary, "v_boundary_after": new_v_boundary,
        })
        return candidate, True

    attempts.append({
        "block": block_id, "pass": "A2", "tag": bit, "status": "rolled_back", "reason": rejection,
        "hpwl_ratio": hpwl_ratio, "blocking_set": blocking,
        "v_boundary_before": old_v_boundary, "v_boundary_after": new_v_boundary,
    })
    return working, False


def _pass_a_boundary_snap(
    working: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
    n_soft: int,
    attempts: List[Dict[str, object]],
) -> Tuple[List[Rect], int]:
    n = len(working)
    _fixed, preplaced, _mib, _cluster, boundary = _parse_constraints(constraints, n)
    snaps_accepted = 0

    detail0 = soft_violations_detail(working, constraints)
    boundary_violations = detail0["boundary_violations"]

    def total_dist(entry):
        _i, _bits, dist = entry
        return sum(abs(d) for d in dist.values())

    ordered = sorted(boundary_violations, key=total_dist)

    for block_id, missing_bits, _dist in ordered:
        if _is_locked(preplaced, target_positions, block_id):
            attempts.append({"block": block_id, "pass": "A", "status": "skipped", "reason": "locked"})
            continue

        old_v_boundary, old_v_grouping, old_v_mib = soft_violations(working, constraints)
        hpwl_old = hpwl(working, b2b_edges, p2b_edges, pin_list)
        bbox_old = _bbox_area(working)

        gx0, gy0 = build_axis_dags(working, constraints, target_positions, b2b_edges, p2b_edges, pin_list)
        x0, x1 = gx0.wall_lo, gx0.wall_hi
        y0, y1 = gy0.wall_lo, gy0.wall_hi

        bx, by, bw, bh = working[block_id]
        extra_pin_x = None
        extra_pin_y = None
        if missing_bits & BOUND_LEFT:
            extra_pin_x = (block_id, x0)
        if missing_bits & BOUND_RIGHT:
            extra_pin_x = (block_id, x1 - bw)
        if missing_bits & BOUND_TOP:
            extra_pin_y = (block_id, y1 - bh)
        if missing_bits & BOUND_BOTTOM:
            extra_pin_y = (block_id, y0)

        candidate, gx, gy = _project_full(
            working, constraints, target_positions, b2b_edges, p2b_edges, pin_list,
            extra_pin_x=extra_pin_x, extra_pin_y=extra_pin_y,
        )

        # Reachability check.
        reachable = True
        if extra_pin_x is not None:
            _b, val = extra_pin_x
            if abs(candidate[block_id][0] - val) > REACH_TOL:
                reachable = False
        if extra_pin_y is not None:
            _b, val = extra_pin_y
            if abs(candidate[block_id][1] - val) > REACH_TOL:
                reachable = False

        if not reachable:
            attempts.append({"block": block_id, "pass": "A", "status": "skipped", "reason": "unreachable"})
            continue

        bbox_new = _bbox_area(candidate)
        hpwl_new = hpwl(candidate, b2b_edges, p2b_edges, pin_list)
        new_v_boundary, new_v_grouping, new_v_mib = soft_violations(candidate, constraints)

        rejection = None
        if not hard_legal(candidate, area_targets, constraints, target_positions):
            rejection = "hard_legal"
        elif abs(bbox_new - bbox_old) > BBOX_TOL:
            rejection = "bbox_changed"
        elif not (new_v_boundary <= old_v_boundary - 1):
            rejection = "v_boundary_not_improved"
        elif new_v_grouping > old_v_grouping:
            rejection = "v_grouping_regressed"
        elif new_v_mib > old_v_mib:
            rejection = "v_mib_regressed"
        elif not (hpwl_new <= hpwl_old * (1.0 + 1.0 / n_soft) + 1e-9):
            rejection = "hpwl_guard"

        hpwl_ratio = (hpwl_new / hpwl_old) if hpwl_old > 0 else 1.0

        if rejection is None:
            working = candidate
            snaps_accepted += 1
            attempts.append({
                "block": block_id, "pass": "A", "status": "snapped",
                "hpwl_ratio": hpwl_ratio,
                "v_boundary_before": old_v_boundary, "v_boundary_after": new_v_boundary,
            })
        else:
            attempts.append({
                "block": block_id, "pass": "A", "status": "rolled_back", "reason": rejection,
                "hpwl_ratio": hpwl_ratio,
                "v_boundary_before": old_v_boundary, "v_boundary_after": new_v_boundary,
            })

    return working, snaps_accepted


def _pass_a2_reorder_snap(
    working: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
    n_soft: int,
    attempts: List[Dict[str, object]],
) -> Tuple[List[Rect], int]:
    """v2: bounded local reorder. Runs AFTER v1 (per-block fixed-order snap)
    on whatever boundary violations remain. For each still-violating,
    non-LOCKED tagged block, attempt `_try_v2_reorder_snap` per missing bit
    (corner tags run the two edges sequentially, each accepted/rolled-back
    independently), ordered by block area ascending (small blocks displace
    less), capped at one v2 attempt per violating block per tagged edge."""
    n = len(working)
    _fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)
    snaps_accepted = 0

    detail0 = soft_violations_detail(working, constraints)
    boundary_violations = detail0["boundary_violations"]

    def block_area(entry):
        block_id, _bits, _dist = entry
        _x, _y, w, h = working[block_id]
        return w * h

    ordered = sorted(boundary_violations, key=block_area)

    for block_id, missing_bits, _dist in ordered:
        if _is_locked(preplaced, target_positions, block_id):
            continue

        for bit in (BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM):
            if not (missing_bits & bit):
                continue

            # Re-check this specific bit is still violated (an earlier bit
            # or an earlier block's v2 snap in this same pass may already
            # have fixed it via the full re-polish).
            cur_detail = soft_violations_detail(working, constraints)
            still_missing = False
            for bid, bits, _d in cur_detail["boundary_violations"]:
                if bid == block_id and (bits & bit):
                    still_missing = True
                    break
            if not still_missing:
                continue

            working, accepted = _try_v2_reorder_snap(
                working, block_id, bit, area_targets, constraints, target_positions,
                b2b_edges, p2b_edges, pin_list, n_soft, attempts,
            )
            if accepted:
                snaps_accepted += 1

    return working, snaps_accepted


def _pass_b_grouping_snap(
    working: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
    n_soft: int,
    attempts: List[Dict[str, object]],
) -> Tuple[List[Rect], int]:
    n = len(working)
    _fixed, preplaced, _mib, cluster, _boundary = _parse_constraints(constraints, n)
    snaps_accepted = 0

    n_clust_groups = max(cluster) if cluster else 0
    for g in range(1, n_clust_groups + 1):
        group_indices = [i for i in range(n) if cluster[i] == g]
        if len(group_indices) <= 1:
            continue

        polys = [box(x, y, x + w, y + h) for (x, y, w, h) in (working[i] for i in group_indices)]
        union_result = unary_union(polys)
        if union_result.geom_type != "MultiPolygon":
            continue

        # Map each connected component to its member block indices.
        components: List[List[int]] = []
        for geom in union_result.geoms:
            members = []
            for i in group_indices:
                x, y, w, h = working[i]
                p = box(x, y, x + w, y + h)
                if geom.intersects(p) and geom.intersection(p).area > 1e-9:
                    members.append(i)
            if members:
                components.append(members)

        if len(components) < 2:
            continue

        def comp_area(members):
            return sum(working[i][2] * working[i][3] for i in members)

        components.sort(key=lambda m: (len(m), comp_area(m)))
        smallest = components[0]
        largest = components[-1]

        if any(_is_locked(preplaced, target_positions, i) for i in smallest):
            attempts.append({"group": g, "pass": "B", "status": "skipped", "reason": "no_candidate"})
            continue

        old_v_boundary, old_v_grouping, old_v_mib = soft_violations(working, constraints)
        hpwl_old = hpwl(working, b2b_edges, p2b_edges, pin_list)
        bbox_old = _bbox_area(working)

        smallest_rects = [working[i] for i in smallest]
        largest_rects = [working[i] for i in largest]
        s_x0 = min(r[0] for r in smallest_rects)
        s_x1 = max(r[0] + r[2] for r in smallest_rects)
        s_y0 = min(r[1] for r in smallest_rects)
        s_y1 = max(r[1] + r[3] for r in smallest_rects)
        l_x0 = min(r[0] for r in largest_rects)
        l_x1 = max(r[0] + r[2] for r in largest_rects)
        l_y0 = min(r[1] for r in largest_rects)
        l_y1 = max(r[1] + r[3] for r in largest_rects)

        # Candidate (a): slide along x to abut nearest largest-component edge.
        if s_x1 <= l_x0:
            dx_a = l_x0 - s_x1
        elif s_x0 >= l_x1:
            dx_a = l_x1 - s_x0
        else:
            dx_a = 0.0
        dy_a = 0.0

        # Candidate (b): slide along y to abut nearest largest-component edge.
        dx_b = 0.0
        if s_y1 <= l_y0:
            dy_b = l_y0 - s_y1
        elif s_y0 >= l_y1:
            dy_b = l_y1 - s_y0
        else:
            dy_b = 0.0

        accepted_this_group = False
        for (dx, dy) in ((dx_a, dy_a), (dx_b, dy_b)):
            if dx == 0.0 and dy == 0.0:
                continue
            candidate = list(working)
            for i in smallest:
                x, y, w, h = working[i]
                candidate[i] = (x + dx, y + dy, w, h)

            bbox_new = _bbox_area(candidate)
            hpwl_new = hpwl(candidate, b2b_edges, p2b_edges, pin_list)
            new_v_boundary, new_v_grouping, new_v_mib = soft_violations(candidate, constraints)

            rejection = None
            if not hard_legal(candidate, area_targets, constraints, target_positions):
                rejection = "hard_legal"
            elif abs(bbox_new - bbox_old) > BBOX_TOL:
                rejection = "bbox_changed"
            elif not (new_v_grouping <= old_v_grouping - 1):
                rejection = "v_grouping_not_improved"
            elif new_v_boundary > old_v_boundary:
                rejection = "v_boundary_regressed"
            elif new_v_mib > old_v_mib:
                rejection = "v_mib_regressed"
            elif not (hpwl_new <= hpwl_old * (1.0 + 1.0 / n_soft) + 1e-9):
                rejection = "hpwl_guard"

            hpwl_ratio = (hpwl_new / hpwl_old) if hpwl_old > 0 else 1.0

            if rejection is None:
                working = candidate
                snaps_accepted += 1
                accepted_this_group = True
                attempts.append({
                    "group": g, "pass": "B", "status": "snapped",
                    "hpwl_ratio": hpwl_ratio,
                    "v_grouping_before": old_v_grouping, "v_grouping_after": new_v_grouping,
                })
                break
            else:
                attempts.append({
                    "group": g, "pass": "B", "status": "rolled_back", "reason": rejection,
                    "hpwl_ratio": hpwl_ratio,
                    "v_grouping_before": old_v_grouping, "v_grouping_after": new_v_grouping,
                })

        if not accepted_this_group:
            attempts.append({"group": g, "pass": "B", "status": "skipped", "reason": "no_candidate"})

    return working, snaps_accepted


def refine_vsnap(
    rects: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges,
    p2b_edges,
    pin_list,
    deadline: Optional[float] = None,
    log_path: Optional[str] = None,
) -> Tuple[List[Rect], Dict[str, object]]:
    """Phase-V violation snap: per-block boundary wall snap (sub-pass A) then
    grouping snap (sub-pass B). Pure best-effort: on any exception returns the
    input rects unchanged with an empty-ish detail dict."""
    original = list(rects)
    attempts: List[Dict[str, object]] = []
    try:
        n = len(original)
        v_boundary_before, v_grouping_before, v_mib_before = soft_violations(original, constraints)
        n_soft = _n_soft(constraints, n)

        working = original
        total_snaps = 0

        working, snaps_a = _pass_a_boundary_snap(
            working, area_targets, constraints, target_positions,
            b2b_edges, p2b_edges, pin_list, n_soft, attempts,
        )
        total_snaps += snaps_a

        if deadline is None or time.time() < deadline:
            working, snaps_a2 = _pass_a2_reorder_snap(
                working, area_targets, constraints, target_positions,
                b2b_edges, p2b_edges, pin_list, n_soft, attempts,
            )
            total_snaps += snaps_a2

        if deadline is None or time.time() < deadline:
            working, snaps_b = _pass_b_grouping_snap(
                working, area_targets, constraints, target_positions,
                b2b_edges, p2b_edges, pin_list, n_soft, attempts,
            )
            total_snaps += snaps_b

        v_boundary_after, v_grouping_after, v_mib_after = soft_violations(working, constraints)

        detail = {
            "v_boundary_before": v_boundary_before,
            "v_boundary_after": v_boundary_after,
            "v_grouping_before": v_grouping_before,
            "v_grouping_after": v_grouping_after,
            "v_mib_before": v_mib_before,
            "v_mib_after": v_mib_after,
            "snaps_accepted": total_snaps,
            "attempts": attempts,
        }

        for rec in attempts:
            _log_attempt(log_path, dict(rec))

        return working, detail
    except Exception:
        return original, {
            "v_boundary_before": None,
            "v_boundary_after": None,
            "v_grouping_before": None,
            "v_grouping_after": None,
            "v_mib_before": None,
            "v_mib_after": None,
            "snaps_accepted": 0,
            "attempts": attempts,
        }
