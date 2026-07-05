"""Orchestration + failure containment for the slack-redistribution refiner.

See docs/design/slack_refiner_spec.md sections 5-6. `refine_layout` is a
pure function: on any exception, failed hard-legality re-check, or deadline
hit, it returns the ORIGINAL rects unchanged (the backbone's score is the
floor, never regressed).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .constraint_graph import build_axis_dags
from .guards import hard_legal, soft_violations
from .slack_solve import project_axis
from .wirelength import hpwl

Rect = Tuple[float, float, float, float]

EPS = 1e-6


def _bbox_area(rects: Sequence[Rect]) -> float:
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return (x_max - x_min) * (y_max - y_min)


def _to_edge_list(t) -> List[Tuple[float, float, float]]:
    if t is None:
        return []
    out = []
    for row in t:
        out.append((float(row[0]), float(row[1]), float(row[2])))
    return out


def _to_pin_list(t) -> List[Tuple[float, float]]:
    if t is None:
        return []
    return [(float(row[0]), float(row[1])) for row in t]


def _diagnose_hard_legal_failure(
    original: List[Rect],
    candidate: List[Rect],
    gx,
    gy,
) -> Optional[Dict[str, object]]:
    """Temporary diagnostic (spec-debug aid): find the first overlapping
    pair in `candidate` and report their rects before/after, which axis DAG
    (if any) relates them, and their cluster-group membership. Best-effort;
    returns None on any failure so it never affects control flow."""
    try:
        n = len(candidate)
        edge_x = {(u, v): gap for u, v, gap in gx.edges}
        edge_y = {(u, v): gap for u, v, gap in gy.edges}
        for i in range(n):
            xi, yi, wi, hi = candidate[i]
            for j in range(i + 1, n):
                xj, yj, wj, hj = candidate[j]
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                if ox > 1e-6 and oy > 1e-6:
                    relation = None
                    if (i, j) in edge_x:
                        relation = f"x-edge {i}->{j} gap={edge_x[(i, j)]}"
                    elif (j, i) in edge_x:
                        relation = f"x-edge {j}->{i} gap={edge_x[(j, i)]}"
                    elif (i, j) in edge_y:
                        relation = f"y-edge {i}->{j} gap={edge_y[(i, j)]}"
                    elif (j, i) in edge_y:
                        relation = f"y-edge {j}->{i} gap={edge_y[(j, i)]}"
                    else:
                        relation = "no-edge (corner-only or missing)"
                    return {
                        "overlap_pair": [i, j],
                        "rect_i_before": list(original[i]),
                        "rect_j_before": list(original[j]),
                        "rect_i_after": list(candidate[i]),
                        "rect_j_after": list(candidate[j]),
                        "axis_relation": relation,
                        "group_i": gx.group_of.get(i),
                        "group_j": gx.group_of.get(j),
                        "pinned_i_x": gx.pinned.get(i),
                        "pinned_j_x": gx.pinned.get(j),
                        "pinned_i_y": gy.pinned.get(i),
                        "pinned_j_y": gy.pinned.get(j),
                    }
        return None
    except Exception:
        return None


def _build_anchors(
    n: int,
    axis: int,  # 0 = x, 1 = y
    rects: Sequence[Rect],
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
) -> Dict[int, Tuple[List[float], List[float]]]:
    """Per-block (anchor_centroid_list, weight_list) along one axis, built
    from that block's b2b/p2b partners' *current* centroids."""
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


def _log_call(
    log_path: Optional[str],
    block_count: int,
    hpwl_before: float,
    hpwl_after: float,
    bbox_before: float,
    bbox_after: float,
    sweeps: int,
    accepted: bool,
    rejection_reason: Optional[str] = None,
    detail: Optional[Dict[str, object]] = None,
) -> None:
    if not log_path:
        return
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = {
            "block_count": block_count,
            "hpwl_before": hpwl_before,
            "hpwl_after": hpwl_after,
            "bbox_before": bbox_before,
            "bbox_after": bbox_after,
            "sweeps": sweeps,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
        }
        if detail:
            record.update(detail)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def refine_layout(
    rects: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b,
    p2b,
    pins,
    deadline: Optional[float] = None,
    enable_aspect: bool = False,
) -> List[Rect]:
    """Phase-1 slack-redistribution refiner. Translates blocks only (dims
    and bbox frozen); returns the ORIGINAL rects on any exception, failed
    re-check, or deadline hit. `enable_aspect` is accepted for forward
    compatibility with Phase 2 but is a no-op in Step 1."""
    original = list(rects)
    log_path = os.environ.get("FLOORSET_SLACK_REFINE_LOG")
    block_count = len(rects)

    hpwl_before = None
    bbox_before = None
    sweeps_total = 0
    accepted = False

    try:
        if deadline is not None and time.time() >= deadline:
            _log_call(log_path, block_count, float("nan"), float("nan"),
                       float("nan"), float("nan"), 0, False,
                       rejection_reason="deadline")
            return original

        b2b_edges = _to_edge_list(b2b)
        p2b_edges = _to_edge_list(p2b)
        pin_list = _to_pin_list(pins)

        hpwl_before = hpwl(original, b2b_edges, p2b_edges, pin_list)
        bbox_before = _bbox_area(original)
        v_boundary_before, v_grouping_before, v_mib_before = soft_violations(
            original, constraints)

        gx, gy = build_axis_dags(original, constraints, target_positions, b2b, p2b, pins)

        if os.environ.get("FLOORSET_SLACK_REFINE_DUMP"):
            try:
                dump_path = os.environ["FLOORSET_SLACK_REFINE_DUMP"]
                with open(dump_path, "a") as f:
                    f.write(json.dumps({
                        "block_count": block_count,
                        "rects": [list(r) for r in original],
                        "b2b": b2b_edges,
                        "p2b": p2b_edges,
                        "pins": pin_list,
                        "constraints": constraints.tolist() if hasattr(constraints, "tolist") else constraints,
                        "target_positions": target_positions.tolist() if hasattr(target_positions, "tolist") else target_positions,
                    }) + "\n")
            except Exception:
                pass

        if deadline is not None and time.time() >= deadline:
            _log_call(log_path, block_count, hpwl_before, hpwl_before,
                       bbox_before, bbox_before, 0, False,
                       rejection_reason="deadline")
            return original

        n = len(original)
        xs = [r[0] for r in original]
        ys = [r[1] for r in original]
        ws = [r[2] for r in original]
        hs = [r[3] for r in original]

        anchors_x = _build_anchors(n, 0, original, b2b_edges, p2b_edges, pin_list)
        new_xs, sweeps_x = project_axis(gx, xs, ws, anchors_x)

        # Recompute anchor centroids for the y-solve using the just-updated
        # x coordinates (axes are independent per spec; y-partners' x does
        # not affect the y target, so this is purely for correctness of any
        # future coupling and is safe either way).
        interim_rects = [(new_xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
        anchors_y = _build_anchors(n, 1, interim_rects, b2b_edges, p2b_edges, pin_list)
        new_ys, sweeps_y = project_axis(gy, ys, hs, anchors_y)

        sweeps_total = sweeps_x + sweeps_y

        candidate = [(new_xs[i], new_ys[i], ws[i], hs[i]) for i in range(n)]

        if deadline is not None and time.time() >= deadline:
            _log_call(log_path, block_count, hpwl_before, hpwl_before,
                       bbox_before, bbox_before, sweeps_total, False,
                       rejection_reason="deadline")
            return original

        hpwl_after = hpwl(candidate, b2b_edges, p2b_edges, pin_list)
        bbox_after = _bbox_area(candidate)

        # Dims must be byte-identical to input (Phase 1 invariant).
        dims_ok = all(
            candidate[i][2] == original[i][2] and candidate[i][3] == original[i][3]
            for i in range(n)
        )

        v_boundary_after, v_grouping_after, v_mib_after = soft_violations(
            candidate, constraints)

        detail = {
            "v_boundary_before": v_boundary_before,
            "v_boundary_after": v_boundary_after,
            "v_grouping_before": v_grouping_before,
            "v_grouping_after": v_grouping_after,
            "v_mib_before": v_mib_before,
            "v_mib_after": v_mib_after,
        }

        # Ordered acceptance-clause checks (first failing clause is reported
        # as the rejection_reason, per spec section 5 acceptance criteria).
        rejection_reason = None
        if not dims_ok:
            rejection_reason = "dims_changed"
        elif not (hpwl_after < hpwl_before - EPS):
            rejection_reason = "hpwl_not_lower"
        elif not (bbox_after <= bbox_before + 1e-9):
            rejection_reason = "bbox_grew"
        elif v_boundary_after > v_boundary_before:
            rejection_reason = "v_boundary"
        elif v_grouping_after > v_grouping_before:
            rejection_reason = "v_grouping"
        elif v_mib_after > v_mib_before:
            rejection_reason = "v_mib"
        elif not hard_legal(candidate, area_targets, constraints, target_positions):
            rejection_reason = "hard_legal"
            diag = _diagnose_hard_legal_failure(original, candidate, gx, gy)
            if diag:
                detail.update(diag)

        phase1_accepted = rejection_reason is None
        phase1_out = candidate if phase1_accepted else original
        phase1_hpwl = hpwl_after if phase1_accepted else hpwl_before

        aspect_applied = False
        final_out = phase1_out
        final_hpwl = phase1_hpwl
        if enable_aspect:
            try:
                from .aspect import refine_aspect

                aspect_deadline = deadline
                aspect_out = refine_aspect(
                    phase1_out, area_targets, constraints, target_positions,
                    b2b_edges, p2b_edges, pin_list, deadline=aspect_deadline,
                )
                if aspect_out != phase1_out:
                    aspect_hpwl = hpwl(aspect_out, b2b_edges, p2b_edges, pin_list)
                    aspect_bbox = _bbox_area(aspect_out)
                    aspect_soft = soft_violations(aspect_out, constraints)
                    phase1_soft = (
                        (v_boundary_after, v_grouping_after, v_mib_after)
                        if phase1_accepted
                        else (v_boundary_before, v_grouping_before, v_mib_before)
                    )
                    phase1_bbox = bbox_after if phase1_accepted else bbox_before
                    if (
                        aspect_hpwl < phase1_hpwl - EPS
                        and aspect_bbox <= phase1_bbox + 1e-6
                        and all(c <= b for c, b in zip(aspect_soft, phase1_soft))
                        and hard_legal(aspect_out, area_targets, constraints, target_positions)
                    ):
                        final_out = aspect_out
                        final_hpwl = aspect_hpwl
                        aspect_applied = True
            except Exception:
                pass

        detail["aspect_applied"] = aspect_applied
        detail["phase1_accepted"] = phase1_accepted

        # Phase-V violation snap. FLOORSET_SLACK_REFINE_VSNAP=1 (default off;
        # see FLOORSET_SLACK_REFINE_ASPECT in
        # legalizer/column_backbone.py::solve_with_column_backbone for the
        # sibling Phase-2 gate) enables a post-pass that runs AFTER
        # phase1/aspect on `final_out`: per-block boundary wall snap
        # (sub-pass A) + grouping snap (sub-pass B), see
        # src/floorset_arch/refine/vsnap.py. Failure-contained: never lets an
        # exception propagate or corrupt final_out.
        vsnap_applied = False
        if os.environ.get("FLOORSET_SLACK_REFINE_VSNAP", "0") == "1":
            try:
                from .vsnap import refine_vsnap

                vsnap_out, vsnap_detail = refine_vsnap(
                    final_out, area_targets, constraints, target_positions,
                    b2b_edges, p2b_edges, pin_list, deadline=deadline,
                    log_path=log_path,
                )
                if vsnap_out != final_out:
                    final_out = vsnap_out
                    final_hpwl = hpwl(final_out, b2b_edges, p2b_edges, pin_list)
                    vsnap_applied = True
                detail.update(vsnap_detail)
            except Exception:
                pass
        detail["vsnap_applied"] = vsnap_applied

        if phase1_accepted or aspect_applied or vsnap_applied:
            accepted = True
            _log_call(log_path, block_count, hpwl_before, final_hpwl,
                       bbox_before, _bbox_area(final_out), sweeps_total, True,
                       detail=detail)
            return final_out

        _log_call(log_path, block_count, hpwl_before, hpwl_after,
                   bbox_before, bbox_after, sweeps_total, False,
                   rejection_reason=rejection_reason, detail=detail)
        return original

    except Exception:
        try:
            _log_call(log_path, block_count,
                       hpwl_before if hpwl_before is not None else float("nan"),
                       float("nan"),
                       bbox_before if bbox_before is not None else float("nan"),
                       float("nan"), sweeps_total, False,
                       rejection_reason="exception")
        except Exception:
            pass
        return original
