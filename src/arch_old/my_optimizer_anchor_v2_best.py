#!/usr/bin/env python3

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer

try:
    from gnn_model import FloorplanGNN, build_node_features, build_edge_tensors
    GNN_AVAILABLE = True
except Exception:
    GNN_AVAILABLE = False


class MyOptimizer(FloorplanOptimizer):
    """
    Experimental Version: conservative relative-order + guarded soft repair.

    Main flow:
      1. Get raw centers from GNN anchor, or fallback pin-based anchor.
      2. Build relative horizontal / vertical constraints from raw centers.
      3. Solve x/y by longest-path packing. This gives an overlap-free compact
         movable-only placement.
      4. Insert preplaced blocks as fixed obstacles and repair using a small
         push legalizer.
      5. Run light boundary / cluster repair.

    This is intended to test whether sequence-pair / constraint-graph style
    placement gives better speed-quality tradeoff than analytical + push.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

        # GNN switches.
        self.use_gnn_anchor = True
        self.use_gnn_priority = False
        self.gnn_aspect_strength = 0.0

        # Relative-order packing switches.
        self.use_sequence_pack = True
        self.anchor_translation_strength = 0.58

        # Soft-aware relative-order bias.
        # This version does NOT simply push boundary blocks to extreme raw
        # coordinates.  Instead, it uses mild raw-center nudging plus pairwise
        # orientation bias while building the constraint graph.  This keeps the
        # packing compact but still encourages boundary / cluster constraints.
        self.boundary_order_bias = 0.72
        self.boundary_pair_bias = 0.00
        self.cluster_raw_blend = 0.00
        self.cluster_pair_bias = 0.00
        self.enable_cluster_chain = False

        # Guarded final repair: accept a repair pass only if a global proxy is not worsened.
        self.guarded_repair_slack = 0.025
        self.guarded_soft_bonus = 0.08

        # Push legalizer controls.
        self.max_start_candidates = 10
        self.max_push_iters = 32
        self.max_overlap_push_choices = 3

        # Repair controls.
        self.enable_boundary_repair = True
        self.enable_cluster_repair = True
        self.repair_passes = 2

        # Candidate / repair cost weights.
        self.hpwl_weight = 1.0
        self.area_weight = 0.045
        self.aspect_weight = 0.25
        self.boundary_weight = 950.0
        self.grouping_weight = 110.0
        self.anchor_weight = 0.35
        self.raw_weight = 0.30

        self.device = torch.device("cpu")
        self.gnn_model = self._load_gnn_model()

    # ================================================================
    # Top-level solve
    # ================================================================

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:

        n = block_count

        adj, pin_adj, degree = self._build_graph(n, b2b_connectivity, p2b_connectivity)

        gnn_priority, gnn_anchors, gnn_log_aspects = self._predict_gnn_guidance(
            block_count=n,
            area_targets=area_targets,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
            constraints=constraints,
        )

        widths, heights = self._init_dimensions(
            n=n,
            area_targets=area_targets,
            constraints=constraints,
            target_positions=target_positions,
            gnn_log_aspects=gnn_log_aspects,
        )

        if gnn_anchors is None or not self.use_gnn_anchor:
            anchors = self._fallback_anchors(n, widths, heights, area_targets, pin_adj, pins_pos)
        else:
            anchors = gnn_anchors

        if self.use_sequence_pack:
            raw_positions = self._relative_order_pack(
                n=n,
                widths=widths,
                heights=heights,
                anchors=anchors,
                constraints=constraints,
                target_positions=target_positions,
                degree=degree,
            )
        else:
            raw_positions = []
            for i in range(n):
                ax, ay = anchors[i]
                raw_positions.append((max(0.0, ax - widths[i] / 2.0), max(0.0, ay - heights[i] / 2.0), widths[i], heights[i]))

        positions = self._legalize_from_raw(
            n=n,
            widths=widths,
            heights=heights,
            raw_positions=raw_positions,
            anchors=anchors,
            gnn_priority=gnn_priority,
            area_targets=area_targets,
            pins_pos=pins_pos,
            constraints=constraints,
            target_positions=target_positions,
            adj=adj,
            pin_adj=pin_adj,
            degree=degree,
        )

        positions = self._final_repair(
            n=n,
            positions=positions,
            raw_positions=raw_positions,
            anchors=anchors,
            constraints=constraints,
            adj=adj,
            pin_adj=pin_adj,
            pins_pos=pins_pos,
        )

        return [tuple(float(v) for v in positions[i]) for i in range(n)]

    # ================================================================
    # GNN helpers
    # ================================================================

    def _load_checkpoint(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _load_gnn_model(self):
        if not GNN_AVAILABLE:
            return None

        ckpt_path = Path(__file__).parent / "checkpoints" / "gnn_best.pt"
        if not ckpt_path.exists():
            return None

        try:
            ckpt = self._load_checkpoint(ckpt_path)
            model = FloorplanGNN(
                node_feat_dim=int(ckpt["node_feat_dim"]),
                hidden_dim=int(ckpt.get("hidden_dim", 128)),
                num_layers=int(ckpt.get("layers", 4)),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model
        except Exception:
            return None

    def _predict_gnn_guidance(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
    ):
        if self.gnn_model is None:
            return None, None, None

        try:
            with torch.no_grad():
                node_feat, scale = build_node_features(
                    area_targets=area_targets,
                    b2b_connectivity=b2b_connectivity,
                    p2b_connectivity=p2b_connectivity,
                    pins_pos=pins_pos,
                    constraints=constraints,
                    block_count=block_count,
                    device=self.device,
                )

                if node_feat.shape[1] != self.gnn_model.node_feat_dim:
                    return None, None, None

                edge_index, edge_attr = build_edge_tensors(
                    block_count=block_count,
                    b2b_connectivity=b2b_connectivity,
                    device=self.device,
                )

                pred = self.gnn_model(node_feat, edge_index, edge_attr)

                priority = pred["priority"].detach().cpu().tolist()
                anchors = (pred["anchor"].detach().cpu() * float(scale)).tolist()
                log_aspects = pred["log_aspect"].detach().cpu().clamp(min=-2.0, max=2.0).tolist()

                return priority, anchors, log_aspects
        except Exception:
            return None, None, None

    # ================================================================
    # Relative-order / constraint-graph packer
    # ================================================================

    def _relative_order_pack(
        self,
        n,
        widths,
        heights,
        anchors,
        constraints,
        target_positions,
        degree,
    ):
        """
        Build horizontal/vertical precedence constraints from raw anchor order.

        For each movable pair (i, j):
          - If x separation dominates, put the left raw-center block on the left.
          - Otherwise, put the lower raw-center block below.

        Since all horizontal edges follow increasing raw x, and all vertical
        edges follow increasing raw y, the two graphs are DAGs. Longest path
        then gives a compact legal placement for movable blocks.
        """

        raw_cx = [0.0] * n
        raw_cy = [0.0] * n

        for i in range(n):
            if self._is_preplaced(i, constraints) and target_positions is not None:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
                raw_cx[i] = x + w / 2.0
                raw_cy[i] = y + h / 2.0
            else:
                ax, ay = anchors[i]
                raw_cx[i] = max(float(ax), widths[i] / 2.0)
                raw_cy[i] = max(float(ay), heights[i] / 2.0)

        # Apply soft-constraint-aware raw-center bias before building
        # relative-order graphs. This is the main improvement in this version.
        raw_cx, raw_cy = self._bias_raw_centers_for_constraints(
            n=n,
            raw_cx=raw_cx,
            raw_cy=raw_cy,
            widths=widths,
            heights=heights,
            constraints=constraints,
        )

        movable = [i for i in range(n) if not self._is_preplaced(i, constraints)]

        if not movable:
            raw_positions = []
            for i in range(n):
                if target_positions is not None:
                    x = float(target_positions[i, 0])
                    y = float(target_positions[i, 1])
                    w = float(target_positions[i, 2])
                    h = float(target_positions[i, 3])
                    raw_positions.append((x, y, w, h))
                else:
                    raw_positions.append((0.0, 0.0, widths[i], heights[i]))
            return raw_positions

        # Effective ordering keys.  These are only used to orient edges;
        # the raw coordinates themselves stay relatively compact.
        order_key_x, order_key_y = self._make_soft_order_keys(
            n=n,
            raw_cx=raw_cx,
            raw_cy=raw_cy,
            widths=widths,
            heights=heights,
            constraints=constraints,
        )

        order_x = sorted(movable, key=lambda i: (order_key_x[i], order_key_y[i], -degree[i], i))
        order_y = sorted(movable, key=lambda i: (order_key_y[i], order_key_x[i], -degree[i], i))

        pos_x = {i: k for k, i in enumerate(order_x)}
        pos_y = {i: k for k, i in enumerate(order_y)}

        # Dominant orientation for each cluster.  A horizontal cluster should
        # be packed as a local horizontal chain; a vertical cluster as a local
        # vertical chain.  This handles grouping before the graph is solved.
        cluster_orient = self._cluster_orientations(
            n=n,
            raw_cx=raw_cx,
            raw_cy=raw_cy,
            constraints=constraints,
        )

        h_adj = {i: [] for i in movable}
        v_adj = {i: [] for i in movable}

        def add_h_edge(a, b):
            if pos_x[a] <= pos_x[b]:
                h_adj[a].append(b)
            else:
                h_adj[b].append(a)

        def add_v_edge(a, b):
            if pos_y[a] <= pos_y[b]:
                v_adj[a].append(b)
            else:
                v_adj[b].append(a)

        m = len(movable)
        for a in range(m):
            i = movable[a]
            for b in range(a + 1, m):
                j = movable[b]

                dx = abs(raw_cx[i] - raw_cx[j]) / max((widths[i] + widths[j]) * 0.5, 1e-6)
                dy = abs(raw_cy[i] - raw_cy[j]) / max((heights[i] + heights[j]) * 0.5, 1e-6)

                h_score = dx
                v_score = dy

                ci = self._boundary_code(i, constraints)
                cj = self._boundary_code(j, constraints)

                # Boundary should affect orientation, not just raw location.
                # Left/right constraints prefer horizontal precedence;
                # top/bottom constraints prefer vertical precedence.
                if (ci & 3) or (cj & 3):
                    h_score += self.boundary_pair_bias
                if (ci & 12) or (cj & 12):
                    v_score += self.boundary_pair_bias

                gi = self._cluster_id(i, constraints)
                gj = self._cluster_id(j, constraints)

                if gi != 0 and gi == gj:
                    if cluster_orient.get(gi, 'H') == 'H':
                        h_score += self.cluster_pair_bias
                    else:
                        v_score += self.cluster_pair_bias

                if h_score >= v_score:
                    add_h_edge(i, j)
                else:
                    add_v_edge(i, j)

        # Extra local chain for each cluster.  This is not a full superblock,
        # but it prevents cluster members from being completely interleaved
        # with unrelated blocks in the longest-path packing.
        clusters = {}
        for i in movable:
            gid = self._cluster_id(i, constraints)
            if gid != 0:
                clusters.setdefault(gid, []).append(i)

        if self.enable_cluster_chain:
            for gid, members in clusters.items():
                if len(members) <= 1:
                    continue
                if cluster_orient.get(gid, 'H') == 'H':
                    members = sorted(members, key=lambda i: (order_key_x[i], order_key_y[i], i))
                    for u, v in zip(members, members[1:]):
                        add_h_edge(u, v)
                else:
                    members = sorted(members, key=lambda i: (order_key_y[i], order_key_x[i], i))
                    for u, v in zip(members, members[1:]):
                        add_v_edge(u, v)

        x_pos = {i: 0.0 for i in movable}
        y_pos = {i: 0.0 for i in movable}

        for u in order_x:
            base = x_pos[u] + widths[u]
            for v in h_adj[u]:
                if base > x_pos[v]:
                    x_pos[v] = base

        for u in order_y:
            base = y_pos[u] + heights[u]
            for v in v_adj[u]:
                if base > y_pos[v]:
                    y_pos[v] = base

        # Translate compact pack toward raw anchor centroid. This preserves
        # no-overlap among movable blocks, and later push handles preplaced.
        pack_cx_sum = 0.0
        pack_cy_sum = 0.0
        raw_cx_sum = 0.0
        raw_cy_sum = 0.0

        for i in movable:
            pack_cx_sum += x_pos[i] + widths[i] / 2.0
            pack_cy_sum += y_pos[i] + heights[i] / 2.0
            raw_cx_sum += raw_cx[i]
            raw_cy_sum += raw_cy[i]

        inv_m = 1.0 / max(len(movable), 1)
        dx_shift = (raw_cx_sum - pack_cx_sum) * inv_m * self.anchor_translation_strength
        dy_shift = (raw_cy_sum - pack_cy_sum) * inv_m * self.anchor_translation_strength

        min_x_after = min(x_pos[i] + dx_shift for i in movable)
        min_y_after = min(y_pos[i] + dy_shift for i in movable)

        if min_x_after < 0.0:
            dx_shift -= min_x_after
        if min_y_after < 0.0:
            dy_shift -= min_y_after

        raw_positions = [None] * n
        for i in range(n):
            if self._is_preplaced(i, constraints) and target_positions is not None:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
                raw_positions[i] = (x, y, w, h)
            else:
                x = max(0.0, x_pos.get(i, 0.0) + dx_shift)
                y = max(0.0, y_pos.get(i, 0.0) + dy_shift)
                raw_positions[i] = (x, y, widths[i], heights[i])

        return raw_positions

    def _bias_raw_centers_for_constraints(
        self,
        n,
        raw_cx,
        raw_cy,
        widths,
        heights,
        constraints,
    ):
        """
        Mild raw-center preprocessing.

        Important: do not force boundary blocks to huge extreme positions here.
        Extreme raw bias made some cases better and some cases worse.  This
        version only nudges boundary blocks slightly and leaves the stronger
        soft-constraint behavior to pairwise graph orientation.
        """

        raw_cx = list(raw_cx)
        raw_cy = list(raw_cy)

        if n == 0:
            return raw_cx, raw_cy

        min_x = min(raw_cx)
        max_x = max(raw_cx)
        min_y = min(raw_cy)
        max_y = max(raw_cy)

        span_x = max(max_x - min_x, max(widths) if widths else 1.0, 1.0)
        span_y = max(max_y - min_y, max(heights) if heights else 1.0, 1.0)

        # Cluster preprocessing: keep members closer, but do not collapse them.
        clusters = {}
        for i in range(n):
            gid = self._cluster_id(i, constraints)
            if gid != 0:
                clusters.setdefault(gid, []).append(i)

        blend = max(0.0, min(1.0, self.cluster_raw_blend))
        if blend > 0.0:
            for members in clusters.values():
                if len(members) <= 1:
                    continue
                cx = sum(raw_cx[i] for i in members) / len(members)
                cy = sum(raw_cy[i] for i in members) / len(members)
                for i in members:
                    raw_cx[i] = (1.0 - blend) * raw_cx[i] + blend * cx
                    raw_cy[i] = (1.0 - blend) * raw_cy[i] + blend * cy

        # Mild boundary nudge.  The actual boundary-aware behavior is applied
        # in _make_soft_order_keys() and pairwise edge scoring.
        nudge_x = 0.12 * self.boundary_order_bias * span_x
        nudge_y = 0.12 * self.boundary_order_bias * span_y

        for i in range(n):
            code = self._boundary_code(i, constraints)
            if code == 0:
                continue

            if code & 1:   # left
                raw_cx[i] -= nudge_x
            if code & 2:   # right
                raw_cx[i] += nudge_x
            if code & 8:   # bottom
                raw_cy[i] -= nudge_y
            if code & 4:   # top
                raw_cy[i] += nudge_y

        return raw_cx, raw_cy

    def _make_soft_order_keys(
        self,
        n,
        raw_cx,
        raw_cy,
        widths,
        heights,
        constraints,
    ):
        """
        Create ordering keys for graph construction.

        These keys can be more biased than the raw coordinates because they
        decide pairwise precedence only.  This is safer than moving actual raw
        centers too far away, which can inflate area.
        """

        key_x = list(raw_cx)
        key_y = list(raw_cy)

        if n == 0:
            return key_x, key_y

        span_x = max(max(raw_cx) - min(raw_cx), max(widths) if widths else 1.0, 1.0)
        span_y = max(max(raw_cy) - min(raw_cy), max(heights) if heights else 1.0, 1.0)

        offset_x = self.boundary_order_bias * span_x
        offset_y = self.boundary_order_bias * span_y

        left_k = 0
        right_k = 0
        bottom_k = 0
        top_k = 0

        for i in range(n):
            code = self._boundary_code(i, constraints)
            if code == 0:
                continue

            if code & 1:
                key_x[i] -= offset_x + left_k * max(widths[i], 1.0) * 0.05
                left_k += 1
            if code & 2:
                key_x[i] += offset_x + right_k * max(widths[i], 1.0) * 0.05
                right_k += 1
            if code & 8:
                key_y[i] -= offset_y + bottom_k * max(heights[i], 1.0) * 0.05
                bottom_k += 1
            if code & 4:
                key_y[i] += offset_y + top_k * max(heights[i], 1.0) * 0.05
                top_k += 1

        return key_x, key_y

    def _cluster_orientations(self, n, raw_cx, raw_cy, constraints):
        """
        Decide whether each cluster should be packed mainly horizontally or
        vertically, based on its raw bbox aspect.
        """

        clusters = {}
        for i in range(n):
            gid = self._cluster_id(i, constraints)
            if gid != 0:
                clusters.setdefault(gid, []).append(i)

        orient = {}
        for gid, members in clusters.items():
            if len(members) <= 1:
                orient[gid] = 'H'
                continue
            sx = max(raw_cx[i] for i in members) - min(raw_cx[i] for i in members)
            sy = max(raw_cy[i] for i in members) - min(raw_cy[i] for i in members)
            orient[gid] = 'H' if sx >= sy else 'V'

        return orient

    # ================================================================
    # Legalization from raw positions
    # ================================================================

    def _legalize_from_raw(
        self,
        n,
        widths,
        heights,
        raw_positions,
        anchors,
        gnn_priority,
        area_targets,
        pins_pos,
        constraints,
        target_positions,
        adj,
        pin_adj,
        degree,
    ):
        positions = [None] * n
        occupied = []

        for i in range(n):
            if self._is_preplaced(i, constraints):
                if target_positions is not None:
                    x = float(target_positions[i, 0])
                    y = float(target_positions[i, 1])
                    w = float(target_positions[i, 2])
                    h = float(target_positions[i, 3])
                else:
                    x = raw_positions[i][0]
                    y = raw_positions[i][1]
                    w = widths[i]
                    h = heights[i]

                positions[i] = (x, y, w, h)
                occupied.append(positions[i])

        movable = [i for i in range(n) if not self._is_preplaced(i, constraints)]

        max_degree = max(max(degree), 1.0)

        def priority_value(i):
            deg_score = degree[i] / max_degree
            if self.use_gnn_priority and gnn_priority is not None:
                return 0.5 * float(gnn_priority[i]) + 0.5 * deg_score
            return deg_score

        # Raw order from constraint graph is important, but boundary/cluster
        # terms are still placed early enough to reduce soft violations.
        movable.sort(
            key=lambda i: (
                raw_positions[i][0] + raw_positions[i][1],
                1 if self._boundary_code(i, constraints) != 0 else 0,
                0 if self._cluster_id(i, constraints) != 0 else 1,
                self._cluster_id(i, constraints),
                -priority_value(i),
                i,
            )
        )

        for i in movable:
            w = widths[i]
            h = heights[i]

            starts = self._generate_start_positions(
                idx=i,
                w=w,
                h=h,
                positions=positions,
                occupied=occupied,
                adj=adj,
                pin_adj=pin_adj,
                pins_pos=pins_pos,
                constraints=constraints,
                raw_positions=raw_positions,
                anchors=anchors,
            )

            best_pos = None
            best_cost = float("inf")

            for sx, sy in starts:
                rect = (max(0.0, sx), max(0.0, sy), w, h)
                legal_rect = self._legalize_by_push(
                    idx=i,
                    rect=rect,
                    positions=positions,
                    occupied=occupied,
                    adj=adj,
                    pin_adj=pin_adj,
                    pins_pos=pins_pos,
                    constraints=constraints,
                    anchors=anchors,
                    raw_positions=raw_positions,
                )

                if legal_rect is None:
                    continue

                cost = self._candidate_cost(
                    idx=i,
                    cand=legal_rect,
                    positions=positions,
                    occupied=occupied,
                    adj=adj,
                    pin_adj=pin_adj,
                    pins_pos=pins_pos,
                    constraints=constraints,
                    anchors=anchors,
                    raw_positions=raw_positions,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_pos = legal_rect

            if best_pos is None:
                best_pos = self._fallback_place_best(
                    idx=i,
                    w=w,
                    h=h,
                    positions=positions,
                    occupied=occupied,
                    adj=adj,
                    pin_adj=pin_adj,
                    pins_pos=pins_pos,
                    constraints=constraints,
                    anchors=anchors,
                    raw_positions=raw_positions,
                )

            positions[i] = best_pos
            occupied.append(best_pos)

        return positions

    def _generate_start_positions(
        self,
        idx,
        w,
        h,
        positions,
        occupied,
        adj,
        pin_adj,
        pins_pos,
        constraints,
        raw_positions,
        anchors,
    ):
        starts = set()

        rx, ry, _, _ = raw_positions[idx]
        starts.add((rx, ry))

        # Boundary-aware starts. The relative-order graph already biases the
        # raw position, but when preplaced obstacles push a block away, these
        # starts give the legalizer another chance to sit exactly on the
        # current outer boundary.
        code = self._boundary_code(idx, constraints)
        if code != 0 and occupied:
            ox_min = min(x for x, y, rw, rh in occupied)
            oy_min = min(y for x, y, rw, rh in occupied)
            ox_max = max(x + rw for x, y, rw, rh in occupied)
            oy_max = max(y + rh for x, y, rw, rh in occupied)
            if code & 1:
                starts.add((ox_min, ry))
            if code & 2:
                starts.add((max(0.0, ox_max - w), ry))
            if code & 8:
                starts.add((rx, oy_min))
            if code & 4:
                starts.add((rx, max(0.0, oy_max - h)))

        if anchors is not None:
            ax, ay = anchors[idx]
            starts.add((float(ax) - w / 2.0, float(ay) - h / 2.0))

        if occupied:
            x_edges = {0.0}
            y_edges = {0.0}
            for x, y, rw, rh in occupied:
                x_edges.add(max(0.0, x))
                x_edges.add(max(0.0, x + rw))
                x_edges.add(max(0.0, x - w))
                y_edges.add(max(0.0, y))
                y_edges.add(max(0.0, y + rh))
                y_edges.add(max(0.0, y - h))

            nearest_x = sorted(x_edges, key=lambda xx: abs(xx - rx))[:5]
            nearest_y = sorted(y_edges, key=lambda yy: abs(yy - ry))[:5]
            for xx in nearest_x:
                for yy in nearest_y:
                    starts.add((xx, yy))

            x_max = max(x + rw for x, y, rw, rh in occupied)
            y_max = max(y + rh for x, y, rw, rh in occupied)
            starts.add((x_max, 0.0))
            starts.add((0.0, y_max))

        gid = self._cluster_id(idx, constraints)
        if gid != 0:
            for j, pos in enumerate(positions):
                if pos is None:
                    continue
                if self._cluster_id(j, constraints) != gid:
                    continue
                for c in self._abut_candidates_around(pos, w, h):
                    starts.add(c)

        neighbor_rects = []
        for j, weight in adj[idx]:
            if positions[j] is not None:
                neighbor_rects.append((weight, positions[j]))
        neighbor_rects.sort(reverse=True, key=lambda t: t[0])

        for _, pos in neighbor_rects[:4]:
            x2, y2, w2, h2 = pos
            starts.add((x2 + w2, y2))
            starts.add((x2 - w, y2))
            starts.add((x2, y2 + h2))
            starts.add((x2, y2 - h))

        if idx < len(pin_adj):
            pins = sorted(pin_adj[idx], reverse=True, key=lambda t: t[1])
            for pin_idx, _ in pins[:2]:
                if 0 <= pin_idx < pins_pos.shape[0]:
                    px = float(pins_pos[pin_idx, 0])
                    py = float(pins_pos[pin_idx, 1])
                    if px != -1.0 and py != -1.0:
                        starts.add((px - w / 2.0, py - h / 2.0))

        starts = [(max(0.0, x), max(0.0, y)) for x, y in starts]
        starts.sort(key=lambda p: self._start_sort_cost(idx, p, w, h, occupied, raw_positions, anchors))
        return starts[: self.max_start_candidates]

    def _start_sort_cost(self, idx, start, w, h, occupied, raw_positions, anchors):
        x, y = start
        cx = x + w / 2.0
        cy = y + h / 2.0

        rx, ry, _, _ = raw_positions[idx]
        rcx = rx + w / 2.0
        rcy = ry + h / 2.0
        cost = self.raw_weight * (abs(cx - rcx) + abs(cy - rcy))

        if anchors is not None:
            ax, ay = anchors[idx]
            cost += self.anchor_weight * 0.5 * (abs(cx - float(ax)) + abs(cy - float(ay)))

        cost += 0.05 * (x + y)

        if occupied:
            rects = occupied + [(x, y, w, h)]
            x_min = min(rx2 for rx2, ry2, rw2, rh2 in rects)
            y_min = min(ry2 for rx2, ry2, rw2, rh2 in rects)
            x_max = max(rx2 + rw2 for rx2, ry2, rw2, rh2 in rects)
            y_max = max(ry2 + rh2 for rx2, ry2, rw2, rh2 in rects)
            cost += 0.001 * (x_max - x_min) * (y_max - y_min)

        return cost

    def _legalize_by_push(
        self,
        idx,
        rect,
        positions,
        occupied,
        adj,
        pin_adj,
        pins_pos,
        constraints,
        anchors,
        raw_positions,
    ):
        x, y, w, h = rect
        rect = (max(0.0, x), max(0.0, y), w, h)
        visited = set()

        if occupied:
            occ_x_min = min(rx for rx, ry, rw, rh in occupied)
            occ_y_min = min(ry for rx, ry, rw, rh in occupied)
            occ_x_max = max(rx + rw for rx, ry, rw, rh in occupied)
            occ_y_max = max(ry + rh for rx, ry, rw, rh in occupied)
        else:
            occ_x_min = 0.0
            occ_y_min = 0.0
            occ_x_max = 0.0
            occ_y_max = 0.0

        for _ in range(self.max_push_iters):
            overlaps = self._find_overlaps(rect, occupied)
            if not overlaps:
                return rect

            key = (round(rect[0], 5), round(rect[1], 5), round(rect[2], 5), round(rect[3], 5))
            if key in visited:
                break
            visited.add(key)

            overlaps.sort(reverse=True, key=lambda item: item[0])
            cx, cy, cw, ch = rect
            move_set = set()

            for _, ox, oy, ow, oh in overlaps[: self.max_overlap_push_choices]:
                move_set.add((ox + ow, cy, cw, ch))
                move_set.add((cx, oy + oh, cw, ch))
                if ox - cw >= 0.0:
                    move_set.add((ox - cw, cy, cw, ch))
                if oy - ch >= 0.0:
                    move_set.add((cx, oy - ch, cw, ch))
                move_set.add((ox + ow, oy + oh, cw, ch))

            best = None
            best_score = float("inf")

            rx, ry, _, _ = raw_positions[idx]
            rcx = rx + w / 2.0
            rcy = ry + h / 2.0

            for cand in move_set:
                cand = (max(0.0, cand[0]), max(0.0, cand[1]), cand[2], cand[3])
                overlap_count, overlap_area = self._overlap_stats(cand, occupied)

                bx_min = min(occ_x_min, cand[0])
                by_min = min(occ_y_min, cand[1])
                bx_max = max(occ_x_max, cand[0] + cand[2])
                by_max = max(occ_y_max, cand[1] + cand[3])
                bbox_area = (bx_max - bx_min) * (by_max - by_min)

                ccx = cand[0] + cand[2] / 2.0
                ccy = cand[1] + cand[3] / 2.0
                raw_dist = abs(ccx - rcx) + abs(ccy - rcy)

                if anchors is not None:
                    ax, ay = anchors[idx]
                    anchor_dist = abs(ccx - float(ax)) + abs(ccy - float(ay))
                else:
                    anchor_dist = cand[0] + cand[1]

                score = (
                    1e9 * overlap_count
                    + 1e6 * overlap_area
                    + 0.015 * bbox_area
                    + 0.25 * raw_dist
                    + 0.15 * anchor_dist
                    + 0.08 * (cand[0] + cand[1])
                )

                if score < best_score:
                    best_score = score
                    best = cand

            if best is None:
                break
            if abs(best[0] - rect[0]) < 1e-9 and abs(best[1] - rect[1]) < 1e-9:
                break
            rect = best

        return None

    def _fallback_place_best(
        self,
        idx,
        w,
        h,
        positions,
        occupied,
        adj,
        pin_adj,
        pins_pos,
        constraints,
        anchors,
        raw_positions,
    ):
        if not occupied:
            return (0.0, 0.0, w, h)

        max_x = max(x + rw for x, y, rw, rh in occupied)
        max_y = max(y + rh for x, y, rw, rh in occupied)

        candidates = [(max_x, 0.0, w, h), (0.0, max_y, w, h), (max_x, max_y, w, h)]
        best = None
        best_cost = float("inf")

        for cand in candidates:
            if not self._is_legal(cand, occupied):
                continue
            cost = self._candidate_cost(idx, cand, positions, occupied, adj, pin_adj, pins_pos, constraints, anchors, raw_positions)
            if cost < best_cost:
                best_cost = cost
                best = cand

        if best is not None:
            return best
        return (max_x, 0.0, w, h)

    # ================================================================
    # Final repair
    # ================================================================

    def _final_repair(self, n, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos):
        # Conservative global guard.  The previous soft-aware graph/repair versions
        # sometimes reduced one violation but destroyed compactness.  Here a repair
        # pass is accepted only when a full-placement proxy remains competitive.
        positions = list(positions)
        best_positions = list(positions)
        best_proxy = self._total_proxy(best_positions, constraints, adj, pin_adj, pins_pos)
        best_soft = self._soft_violation_proxy(best_positions, constraints)

        for _ in range(self.repair_passes):
            trial = list(best_positions)

            if self.enable_cluster_repair:
                trial = self._repair_clusters(n, trial, raw_positions, anchors, constraints, adj, pin_adj, pins_pos)
            if self.enable_boundary_repair:
                trial = self._repair_boundaries(n, trial, raw_positions, anchors, constraints, adj, pin_adj, pins_pos)

            trial_proxy = self._total_proxy(trial, constraints, adj, pin_adj, pins_pos)
            trial_soft = self._soft_violation_proxy(trial, constraints)

            # Accept if proxy improves.  Also accept a small proxy regression if
            # soft violations decrease; this mirrors the exponential V_rel penalty
            # without allowing bbox / HPWL to explode.
            allowed = best_proxy * (1.0 + self.guarded_repair_slack)
            if trial_proxy <= best_proxy or (trial_soft < best_soft and trial_proxy <= allowed):
                best_positions = trial
                best_proxy = min(best_proxy, trial_proxy)
                best_soft = trial_soft
            else:
                break

        return best_positions


    def _total_proxy(self, positions, constraints, adj, pin_adj, pins_pos):
        bbox = self._bbox(positions)
        if bbox is None:
            return 1e30

        x_min, y_min, x_max, y_max = bbox
        area = max((x_max - x_min) * (y_max - y_min), 1.0)

        hpwl = 0.0
        for i, pos in enumerate(positions):
            if pos is None:
                continue
            x, y, w, h = pos
            cx = x + w / 2.0
            cy = y + h / 2.0

            for j, wt in adj[i]:
                if j <= i or positions[j] is None:
                    continue
                x2, y2, w2, h2 = positions[j]
                hpwl += wt * (abs(cx - (x2 + w2 / 2.0)) + abs(cy - (y2 + h2 / 2.0)))

            for pin_idx, wt in pin_adj[i]:
                if 0 <= pin_idx < pins_pos.shape[0]:
                    px = float(pins_pos[pin_idx, 0])
                    py = float(pins_pos[pin_idx, 1])
                    if px != -1.0 and py != -1.0:
                        hpwl += wt * (abs(cx - px) + abs(cy - py))

        soft = self._soft_violation_proxy(positions, constraints)
        # Scale-free enough for comparing candidates in the same testcase.
        return 0.0025 * hpwl + 0.018 * area + 2500.0 * soft

    def _soft_violation_proxy(self, positions, constraints):
        bbox = self._bbox(positions)
        if bbox is None:
            return 1e9

        soft = 0.0
        for i, pos in enumerate(positions):
            if pos is None:
                continue
            soft += self._boundary_violation_score(i, pos, bbox, constraints)

        clusters = {}
        mibs = {}
        for i, pos in enumerate(positions):
            if pos is None:
                continue
            gid = self._cluster_id(i, constraints)
            if gid != 0:
                clusters.setdefault(gid, []).append(i)
            mid = self._mib_id(i, constraints)
            if mid != 0:
                mibs.setdefault(mid, []).append(i)

        # Approximate grouping by counting non-abutting blocks in each group.
        for members in clusters.values():
            if len(members) <= 1:
                continue
            for i in members:
                ok = False
                for j in members:
                    if i != j and self._abut(positions[i], positions[j]):
                        ok = True
                        break
                if not ok:
                    soft += 1.0

        # MIB: count distinct rounded dimensions minus one.
        for members in mibs.values():
            shapes = set()
            for i in members:
                x, y, w, h = positions[i]
                shapes.add((round(w, 5), round(h, 5)))
            if len(shapes) > 1:
                soft += len(shapes) - 1

        return soft

    def _repair_boundaries(self, n, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos):
        bbox = self._bbox(positions)
        if bbox is None:
            return positions

        order = [
            i for i in range(n)
            if (not self._is_preplaced(i, constraints)) and self._boundary_code(i, constraints) != 0
        ]
        order.sort(key=lambda i: (-self._boundary_violation_score(i, positions[i], bbox, constraints), i))

        for i in order:
            cur = positions[i]
            old_v = self._boundary_violation_score(i, cur, bbox, constraints)
            if old_v <= 0:
                continue

            candidates = self._boundary_repair_candidates(i, cur, bbox, positions, constraints)
            best = cur
            best_score = self._repair_objective(i, cur, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos)

            for cand in candidates:
                cand = (max(0.0, cand[0]), max(0.0, cand[1]), cand[2], cand[3])
                if not self._is_legal_except(i, cand, positions):
                    continue

                new_v = self._boundary_violation_score(i, cand, bbox, constraints)
                score = self._repair_objective(i, cand, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos)

                if new_v < old_v:
                    score -= 6000.0 * (old_v - new_v)

                if score < best_score:
                    best_score = score
                    best = cand

            positions[i] = best
            bbox = self._bbox(positions)
            if bbox is None:
                break

        return positions

    def _boundary_repair_candidates(self, idx, rect, bbox, positions, constraints):
        x_min, y_min, x_max, y_max = bbox
        x, y, w, h = rect
        code = self._boundary_code(idx, constraints)

        candidates = set()

        x_edges = {0.0, x_min, max(0.0, x_max - w)}
        y_edges = {0.0, y_min, max(0.0, y_max - h)}

        for j, pos in enumerate(positions):
            if j == idx or pos is None:
                continue
            px, py, pw, ph = pos
            x_edges.add(max(0.0, px))
            x_edges.add(max(0.0, px + pw))
            x_edges.add(max(0.0, px - w))
            y_edges.add(max(0.0, py))
            y_edges.add(max(0.0, py + ph))
            y_edges.add(max(0.0, py - h))

        if code & 1:
            for yy in y_edges:
                candidates.add((x_min, yy, w, h))
        if code & 2:
            for yy in y_edges:
                candidates.add((x_max - w, yy, w, h))
        if code & 8:
            for xx in x_edges:
                candidates.add((xx, y_min, w, h))
        if code & 4:
            for xx in x_edges:
                candidates.add((xx, y_max - h, w, h))

        # Also keep original, and direct projection to all requested sides.
        x_proj = x
        y_proj = y
        if code & 1:
            x_proj = x_min
        if code & 2:
            x_proj = x_max - w
        if code & 8:
            y_proj = y_min
        if code & 4:
            y_proj = y_max - h
        candidates.add((x_proj, y_proj, w, h))

        out = [(max(0.0, cx), max(0.0, cy), cw, ch) for cx, cy, cw, ch in candidates]
        out.sort(key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        return out[:70]

    def _repair_clusters(self, n, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos):
        clusters = {}
        for i in range(n):
            gid = self._cluster_id(i, constraints)
            if gid != 0:
                clusters.setdefault(gid, []).append(i)

        for gid, members in clusters.items():
            if len(members) <= 1:
                continue

            for i in members:
                if self._is_preplaced(i, constraints):
                    continue

                cur = positions[i]
                old_dist = self._cluster_distance(i, cur, positions, constraints)
                if old_dist == 0.0:
                    continue

                x, y, w, h = cur
                candidates = set()
                for j in members:
                    if j == i or positions[j] is None:
                        continue
                    for c in self._abut_candidates_around(positions[j], w, h):
                        candidates.add((c[0], c[1], w, h))

                candidates = [(max(0.0, cx), max(0.0, cy), cw, ch) for cx, cy, cw, ch in candidates]
                candidates.sort(key=lambda c: self._cluster_distance(i, c, positions, constraints))

                best = cur
                best_score = self._repair_objective(i, cur, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos)

                for cand in candidates[:70]:
                    if not self._is_legal_except(i, cand, positions):
                        continue
                    new_dist = self._cluster_distance(i, cand, positions, constraints)
                    if new_dist > old_dist:
                        continue

                    score = self._repair_objective(i, cand, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos)
                    if new_dist < old_dist:
                        score -= 4000.0 * (old_dist - new_dist)
                    if score < best_score:
                        best_score = score
                        best = cand

                positions[i] = best

        return positions

    def _repair_objective(self, idx, cand, positions, raw_positions, anchors, constraints, adj, pin_adj, pins_pos):
        occupied = [pos for j, pos in enumerate(positions) if j != idx and pos is not None]
        fake_positions = list(positions)
        fake_positions[idx] = cand

        cost = self._candidate_cost(idx, cand, fake_positions, occupied, adj, pin_adj, pins_pos, constraints, anchors, raw_positions)
        bbox = self._bbox(fake_positions)
        if bbox is not None:
            cost += 7000.0 * self._boundary_violation_score(idx, cand, bbox, constraints)
        cost += 80.0 * self._cluster_distance(idx, cand, fake_positions, constraints)
        return cost

    # ================================================================
    # Constraint / dimension helpers
    # ================================================================

    def _is_fixed(self, i, constraints) -> bool:
        return constraints is not None and constraints.dim() > 1 and constraints.shape[1] > 0 and float(constraints[i, 0]) != 0.0

    def _is_preplaced(self, i, constraints) -> bool:
        return constraints is not None and constraints.dim() > 1 and constraints.shape[1] > 1 and float(constraints[i, 1]) != 0.0

    def _mib_id(self, i, constraints) -> int:
        if constraints is None or constraints.dim() <= 1 or constraints.shape[1] <= 2:
            return 0
        return int(float(constraints[i, 2]))

    def _cluster_id(self, i, constraints) -> int:
        if constraints is None or constraints.dim() <= 1 or constraints.shape[1] <= 3:
            return 0
        return int(float(constraints[i, 3]))

    def _boundary_code(self, i, constraints) -> int:
        if constraints is None or constraints.dim() <= 1 or constraints.shape[1] <= 4:
            return 0
        return int(float(constraints[i, 4]))

    def _init_dimensions(self, n, area_targets, constraints, target_positions, gnn_log_aspects):
        widths = []
        heights = []

        for i in range(n):
            use_target_dim = False
            if target_positions is not None:
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tw != -1.0 and th != -1.0:
                    use_target_dim = True

            if use_target_dim and (self._is_fixed(i, constraints) or self._is_preplaced(i, constraints)):
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            else:
                area = float(area_targets[i]) if float(area_targets[i]) > 0.0 else 1.0
                if gnn_log_aspects is not None and self.gnn_aspect_strength > 0.0:
                    log_r = max(-2.0, min(2.0, float(gnn_log_aspects[i])))
                    log_r = self.gnn_aspect_strength * log_r
                    ratio = math.exp(log_r)
                    w = math.sqrt(area * ratio)
                    h = math.sqrt(area / ratio)
                else:
                    w = math.sqrt(area)
                    h = math.sqrt(area)

            widths.append(w)
            heights.append(h)

        # MIB dimension repair: only when soft movable members have nearly equal area.
        groups = {}
        for i in range(n):
            gid = self._mib_id(i, constraints)
            if gid == 0:
                continue
            if self._is_fixed(i, constraints) or self._is_preplaced(i, constraints):
                continue
            groups.setdefault(gid, []).append(i)

        for _, members in groups.items():
            if len(members) <= 1:
                continue
            areas = [float(area_targets[i]) for i in members if float(area_targets[i]) > 0.0]
            if not areas:
                continue
            amin = min(areas)
            amax = max(areas)
            if amin > 0.0 and (amax - amin) / amin <= 0.01:
                avg_area = sum(areas) / len(areas)
                common_w = math.sqrt(avg_area)
                common_h = math.sqrt(avg_area)
                for i in members:
                    widths[i] = common_w
                    heights[i] = common_h

        return widths, heights

    # ================================================================
    # Graph / fallback anchors
    # ================================================================

    def _build_graph(self, n, b2b_connectivity, p2b_connectivity):
        adj = [[] for _ in range(n)]
        pin_adj = [[] for _ in range(n)]
        degree = [0.0 for _ in range(n)]

        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if float(edge[0]) < 0.0:
                    continue
                i = int(edge[0])
                j = int(edge[1])
                w = float(edge[2])
                if 0 <= i < n and 0 <= j < n:
                    adj[i].append((j, w))
                    adj[j].append((i, w))
                    degree[i] += w
                    degree[j] += w

        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if float(edge[0]) < 0.0:
                    continue
                pin_idx = int(edge[0])
                block_idx = int(edge[1])
                w = float(edge[2])
                if 0 <= block_idx < n:
                    pin_adj[block_idx].append((pin_idx, w))
                    degree[block_idx] += w

        return adj, pin_adj, degree

    def _fallback_anchors(self, n, widths, heights, area_targets, pin_adj, pins_pos):
        total_area = 0.0
        for i in range(n):
            if float(area_targets[i]) > 0.0:
                total_area += float(area_targets[i])
        scale = math.sqrt(max(total_area, 1.0))
        default = [scale * 0.5, scale * 0.5]

        anchors = []
        for i in range(n):
            if i < len(pin_adj) and pin_adj[i]:
                sx = 0.0
                sy = 0.0
                sw = 0.0
                for pin_idx, weight in pin_adj[i]:
                    if 0 <= pin_idx < pins_pos.shape[0]:
                        px = float(pins_pos[pin_idx, 0])
                        py = float(pins_pos[pin_idx, 1])
                        if px != -1.0 and py != -1.0:
                            sx += weight * px
                            sy += weight * py
                            sw += weight
                if sw > 0.0:
                    anchors.append([sx / sw, sy / sw])
                else:
                    anchors.append(default)
            else:
                anchors.append(default)
        return anchors

    # ================================================================
    # Geometry helpers
    # ================================================================

    def _bbox(self, positions):
        rects = [p for p in positions if p is not None]
        if not rects:
            return None
        x_min = min(x for x, y, w, h in rects)
        y_min = min(y for x, y, w, h in rects)
        x_max = max(x + w for x, y, w, h in rects)
        y_max = max(y + h for x, y, w, h in rects)
        return x_min, y_min, x_max, y_max

    def _find_overlaps(self, rect, occupied):
        result = []
        for ox, oy, ow, oh in occupied:
            area = self._overlap_area(rect, (ox, oy, ow, oh))
            if area > 1e-9:
                result.append((area, ox, oy, ow, oh))
        return result

    def _overlap_stats(self, rect, occupied):
        count = 0
        area = 0.0
        for other in occupied:
            a = self._overlap_area(rect, other)
            if a > 1e-9:
                count += 1
                area += a
        return count, area

    def _overlap_area(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
        overlap_y = min(ay + ah, by + bh) - max(ay, by)
        if overlap_x > 1e-9 and overlap_y > 1e-9:
            return overlap_x * overlap_y
        return 0.0

    def _is_legal(self, cand, occupied) -> bool:
        x, y, w, h = cand
        if x < -1e-9 or y < -1e-9:
            return False
        if w <= 0.0 or h <= 0.0:
            return False
        return self._overlap_stats(cand, occupied)[0] == 0

    def _is_legal_except(self, idx, cand, positions):
        x, y, w, h = cand
        if x < -1e-9 or y < -1e-9:
            return False
        if w <= 0.0 or h <= 0.0:
            return False
        for j, other in enumerate(positions):
            if j == idx or other is None:
                continue
            if self._overlap_area(cand, other) > 1e-9:
                return False
        return True

    def _abut_candidates_around(self, rect, w, h):
        x2, y2, w2, h2 = rect
        return [
            (x2 + w2, y2),
            (x2 - w, y2),
            (x2, y2 + h2),
            (x2, y2 - h),
            (x2 + w2, y2 + h2 - h),
            (x2 - w, y2 + h2 - h),
            (x2 + w2 - w, y2 + h2),
            (x2 + w2 - w, y2 - h),
        ]

    def _abut(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax1 = ax
        ax2 = ax + aw
        ay1 = ay
        ay2 = ay + ah
        bx1 = bx
        bx2 = bx + bw
        by1 = by
        by2 = by + bh
        eps = 1e-6
        vertical_touch = (abs(ax2 - bx1) < eps or abs(bx2 - ax1) < eps) and min(ay2, by2) - max(ay1, by1) > eps
        horizontal_touch = (abs(ay2 - by1) < eps or abs(by2 - ay1) < eps) and min(ax2, bx2) - max(ax1, bx1) > eps
        return vertical_touch or horizontal_touch

    def _rect_distance(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax1 = ax
        ax2 = ax + aw
        ay1 = ay
        ay2 = ay + ah
        bx1 = bx
        bx2 = bx + bw
        by1 = by
        by2 = by + bh
        dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
        dy = max(0.0, max(by1 - ay2, ay1 - by2))
        return dx + dy

    # ================================================================
    # Cost / soft proxy helpers
    # ================================================================

    def _candidate_cost(self, idx, cand, positions, occupied, adj, pin_adj, pins_pos, constraints, anchors, raw_positions):
        x, y, w, h = cand
        cx = x + w / 2.0
        cy = y + h / 2.0

        hpwl_cost = 0.0
        for j, weight in adj[idx]:
            if positions[j] is None:
                continue
            x2, y2, w2, h2 = positions[j]
            cx2 = x2 + w2 / 2.0
            cy2 = y2 + h2 / 2.0
            hpwl_cost += weight * (abs(cx - cx2) + abs(cy - cy2))

        for pin_idx, weight in pin_adj[idx]:
            if 0 <= pin_idx < pins_pos.shape[0]:
                px = float(pins_pos[pin_idx, 0])
                py = float(pins_pos[pin_idx, 1])
                if px != -1.0 and py != -1.0:
                    hpwl_cost += weight * (abs(cx - px) + abs(cy - py))

        bbox_area = self._bbox_area_with_candidate(occupied, cand)
        bbox_w, bbox_h = self._bbox_wh_with_candidate(occupied, cand)
        aspect_cost = abs(bbox_w - bbox_h)
        boundary_cost = self._boundary_cost(idx, cand, occupied, constraints)
        grouping_cost = self._grouping_cost(idx, cand, positions, constraints)
        anchor_cost = self._anchor_cost(idx, cand, anchors)
        raw_cost = self._raw_cost(idx, cand, raw_positions)

        return (
            self.hpwl_weight * hpwl_cost
            + self.area_weight * bbox_area
            + self.aspect_weight * aspect_cost
            + self.boundary_weight * boundary_cost
            + self.grouping_weight * grouping_cost
            + self.anchor_weight * anchor_cost
            + self.raw_weight * raw_cost
        )

    def _bbox_area_with_candidate(self, occupied, cand):
        rects = occupied + [cand]
        x_min = min(x for x, y, w, h in rects)
        y_min = min(y for x, y, w, h in rects)
        x_max = max(x + w for x, y, w, h in rects)
        y_max = max(y + h for x, y, w, h in rects)
        return (x_max - x_min) * (y_max - y_min)

    def _bbox_wh_with_candidate(self, occupied, cand):
        rects = occupied + [cand]
        x_min = min(x for x, y, w, h in rects)
        y_min = min(y for x, y, w, h in rects)
        x_max = max(x + w for x, y, w, h in rects)
        y_max = max(y + h for x, y, w, h in rects)
        return (x_max - x_min), (y_max - y_min)

    def _anchor_cost(self, idx, cand, anchors):
        if anchors is None:
            return 0.0
        ax, ay = anchors[idx]
        x, y, w, h = cand
        cx = x + w / 2.0
        cy = y + h / 2.0
        return abs(cx - float(ax)) + abs(cy - float(ay))

    def _raw_cost(self, idx, cand, raw_positions):
        if raw_positions is None:
            return 0.0
        rx, ry, rw, rh = raw_positions[idx]
        rcx = rx + rw / 2.0
        rcy = ry + rh / 2.0
        x, y, w, h = cand
        cx = x + w / 2.0
        cy = y + h / 2.0
        return abs(cx - rcx) + abs(cy - rcy)

    def _boundary_cost(self, idx, cand, occupied, constraints):
        code = self._boundary_code(idx, constraints)
        if code == 0:
            return 0.0
        rects = occupied + [cand]
        x_min = min(x for x, y, w, h in rects)
        y_min = min(y for x, y, w, h in rects)
        x_max = max(x + w for x, y, w, h in rects)
        y_max = max(y + h for x, y, w, h in rects)
        return self._boundary_violation_score(idx, cand, (x_min, y_min, x_max, y_max), constraints)

    def _boundary_violation_score(self, idx, cand, bbox, constraints):
        code = self._boundary_code(idx, constraints)
        if code == 0:
            return 0.0
        x_min, y_min, x_max, y_max = bbox
        x, y, w, h = cand
        eps = 1e-6
        cost = 0.0
        if code & 1:
            if abs(x - x_min) > eps:
                cost += 1.0
        if code & 2:
            if abs(x + w - x_max) > eps:
                cost += 1.0
        if code & 4:
            if abs(y + h - y_max) > eps:
                cost += 1.0
        if code & 8:
            if abs(y - y_min) > eps:
                cost += 1.0
        return cost

    def _grouping_cost(self, idx, cand, positions, constraints):
        return self._cluster_distance(idx, cand, positions, constraints)

    def _cluster_distance(self, idx, cand, positions, constraints):
        gid = self._cluster_id(idx, constraints)
        if gid == 0:
            return 0.0
        best_dist = float("inf")
        found = False
        for j, pos in enumerate(positions):
            if j == idx or pos is None:
                continue
            if self._cluster_id(j, constraints) != gid:
                continue
            found = True
            if self._abut(cand, pos):
                return 0.0
            best_dist = min(best_dist, self._rect_distance(cand, pos))
        if not found:
            return 0.0
        return best_dist
