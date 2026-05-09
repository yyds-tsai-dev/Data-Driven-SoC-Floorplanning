#!/usr/bin/env python3
"""
Frame-first soft repair optimizer wrapper.

This version is intentionally an experiment beyond the previous cluster/boundary-bias
wrappers.  It keeps the best stable global placer as the base, then applies a bounded
frame-first post pass:

  1. MIB fixed/preplaced reference-shape repair in dimension initialization.
  2. After the base placement, try to snap movable boundary blocks/components exactly
     onto the current final bbox frame.
  3. Then repair violated clusters by moving whole connected components, preserving
     internal offsets, so grouping improves without breaking exact abutments.

It does NOT use force-style boundary bias.  Boundary is treated as exact frame contact.

Expected local files in iccad2026contest/:
  - Prefer: my_optimizer_cluster_v1_best.py
  - Fallback: my_optimizer_anchor_v2_best.py

If neither exists, restore your current best optimizer and save it as one of those names.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from my_optimizer_cluster_v1_best000 import MyOptimizer as _BaseOptimizer
except Exception:
    try:
        from my_optimizer_anchor_v2_best import MyOptimizer as _BaseOptimizer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "This version requires either my_optimizer_cluster_v1_best.py or "
            "my_optimizer_anchor_v2_best.py in the same directory."
        ) from exc


GLOBAL_SEED = 2026


def _set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


class MyOptimizer(_BaseOptimizer):
    """Best baseline + MIB repair + bounded frame-first boundary/cluster post repair."""

    def __init__(self, verbose: bool = False):
        _set_global_seed(GLOBAL_SEED)
        super().__init__(verbose)

        # Deterministic soft repair switches.
        self.enable_mib_reference_shape = True
        self.mib_area_tolerance = 0.0100001

        self.enable_frame_first_post_repair = True
        self.frame_repair_passes = 2

        # Hard budgets: keep runtime bounded.  This pass should not become another
        # slow soft-first relegalizer.
        self.max_boundary_component_snaps = 30
        self.max_cluster_component_moves = 26
        self.max_pair_candidates_per_component = 40
        self.max_members_for_pair_candidates = 12

        # Soft-first acceptance.  Soft reduction is allowed to sacrifice some
        # HPWL/area, but not unlimited.
        self.soft_proxy_slack = 0.42
        self.equal_soft_proxy_slack = 0.015

        # Keep baseline legalizer reasonably stable if these knobs exist.
        if hasattr(self, "repair_passes"):
            self.repair_passes = min(max(int(self.repair_passes), 2), 3)
        if hasattr(self, "max_start_candidates"):
            self.max_start_candidates = max(int(self.max_start_candidates), 10)

    # ================================================================
    # Top-level
    # ================================================================

    def solve(self, *args, **kwargs):
        _set_global_seed(GLOBAL_SEED)
        positions = list(super().solve(*args, **kwargs))

        if not self.enable_frame_first_post_repair:
            return positions

        try:
            n, area_targets, b2b, p2b, pins_pos, constraints = self._parse_solve_args(args, kwargs)
            adj, pin_adj, _degree = self._build_graph(n, b2b, p2b)
            positions = self._frame_first_post_repair(
                n=n,
                positions=[tuple(map(float, p)) for p in positions],
                constraints=constraints,
                adj=adj,
                pin_adj=pin_adj,
                pins_pos=pins_pos,
            )
        except Exception:
            # Never risk infeasibility due to debug/repair failure.
            return positions

        return [tuple(float(v) for v in p) for p in positions]

    def _parse_solve_args(self, args, kwargs):
        if len(args) >= 6:
            return int(args[0]), args[1], args[2], args[3], args[4], args[5]
        return (
            int(kwargs["block_count"]),
            kwargs["area_targets"],
            kwargs["b2b_connectivity"],
            kwargs["p2b_connectivity"],
            kwargs["pins_pos"],
            kwargs["constraints"],
        )

    # ================================================================
    # MIB fixed/preplaced reference-shape repair
    # ================================================================

    def _init_dimensions(self, n, area_targets, constraints, target_positions, gnn_log_aspects):
        widths, heights = super()._init_dimensions(
            n=n,
            area_targets=area_targets,
            constraints=constraints,
            target_positions=target_positions,
            gnn_log_aspects=gnn_log_aspects,
        )

        if not self.enable_mib_reference_shape or constraints is None:
            return widths, heights

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            mid = self._mib_id(i, constraints)
            if mid != 0:
                groups.setdefault(mid, []).append(i)

        for _mid, members in groups.items():
            if len(members) <= 1:
                continue

            ref_shape: Optional[Tuple[float, float]] = None
            # Prefer fixed/preplaced shape.  In debug logs, this exactly fixed cases
            # where movable blocks had equal area but were square while a fixed MIB
            # member had rectangular shape.
            for i in members:
                if self._is_fixed(i, constraints) or self._is_preplaced(i, constraints):
                    rw = float(widths[i])
                    rh = float(heights[i])
                    if rw > 0.0 and rh > 0.0:
                        ref_shape = (rw, rh)
                        break

            if ref_shape is None:
                # Fall back: if all areas are very close, use the first member's shape.
                areas = [float(widths[i]) * float(heights[i]) for i in members]
                amin = min(areas)
                amax = max(areas)
                if amin > 0.0 and (amax - amin) / amin <= self.mib_area_tolerance:
                    ref_shape = (float(widths[members[0]]), float(heights[members[0]]))

            if ref_shape is None:
                continue

            rw, rh = ref_shape
            ref_area = rw * rh
            for i in members:
                if self._is_fixed(i, constraints) or self._is_preplaced(i, constraints):
                    continue
                target_area = float(area_targets[i]) if area_targets is not None else float(widths[i]) * float(heights[i])
                if target_area <= 0.0:
                    continue
                if abs(target_area - ref_area) / max(target_area, ref_area, 1e-9) <= self.mib_area_tolerance:
                    widths[i] = rw
                    heights[i] = rh

        return widths, heights

    # ================================================================
    # Frame-first post repair
    # ================================================================

    def _frame_first_post_repair(self, n, positions, constraints, adj, pin_adj, pins_pos):
        if constraints is None:
            return positions

        cur = list(positions)
        best_score = self._placement_score(cur, constraints, adj, pin_adj, pins_pos)

        boundary_budget = self.max_boundary_component_snaps
        cluster_budget = self.max_cluster_component_moves

        for _pass in range(self.frame_repair_passes):
            changed = False

            # 1) Exact boundary frame contact.  Move a single block or its current
            # same-cluster connected component as a rigid object.  This specifically
            # targets boundary, not force-like proximity.
            cur, used_b, changed_b = self._snap_boundary_components(
                n=n,
                positions=cur,
                constraints=constraints,
                adj=adj,
                pin_adj=pin_adj,
                pins_pos=pins_pos,
                budget=boundary_budget,
            )
            boundary_budget -= used_b
            changed = changed or changed_b

            # 2) Connect split cluster components by rigid component moves.
            # This preserves exact abutments already inside each component.
            cur, used_c, changed_c = self._connect_cluster_components(
                n=n,
                positions=cur,
                constraints=constraints,
                adj=adj,
                pin_adj=pin_adj,
                pins_pos=pins_pos,
                budget=cluster_budget,
            )
            cluster_budget -= used_c
            changed = changed or changed_c

            new_score = self._placement_score(cur, constraints, adj, pin_adj, pins_pos)
            if self._score_better(new_score, best_score, allow_equal_soft=True):
                best_score = new_score
            else:
                # If a whole pass made the placement worse in soft-first proxy,
                # stop.  Individual moves are already guarded, so this is safety.
                break

            if not changed or boundary_budget <= 0 or cluster_budget <= 0:
                break

        return cur

    def _snap_boundary_components(self, n, positions, constraints, adj, pin_adj, pins_pos, budget):
        if budget <= 0:
            return positions, 0, False

        cur = list(positions)
        used = 0
        changed = False

        # Recompute order every pass: largest boundary distance first.
        bbox = self._bbox(cur)
        if bbox is None:
            return cur, used, changed

        candidates = []
        for i in range(n):
            code = self._boundary_code(i, constraints)
            if code == 0 or self._is_preplaced(i, constraints):
                continue
            bscore = self._boundary_violation_score(i, cur[i], bbox, constraints)
            if bscore <= 0.0:
                continue
            dist = self._boundary_distance_sum(i, cur[i], bbox, constraints)
            candidates.append((-bscore, -dist, i))
        candidates.sort()

        for _neg_v, _neg_d, idx in candidates:
            if used >= budget:
                break
            before_score = self._placement_score(cur, constraints, adj, pin_adj, pins_pos)
            component = self._movable_same_cluster_component(idx, cur, constraints)
            if not component:
                continue

            shifts = self._boundary_shifts_for_component(component, cur, constraints)
            best = None
            best_score = before_score

            for dx, dy in shifts:
                trial = self._try_shift_component(cur, component, dx, dy)
                if trial is None:
                    continue
                if not self._component_shift_is_legal(cur, trial, component):
                    continue
                score = self._placement_score(trial, constraints, adj, pin_adj, pins_pos)
                if self._score_better(score, best_score, allow_equal_soft=True):
                    best_score = score
                    best = trial

            used += 1
            if best is not None:
                cur = best
                changed = True

        return cur, used, changed

    def _connect_cluster_components(self, n, positions, constraints, adj, pin_adj, pins_pos, budget):
        if budget <= 0:
            return positions, 0, False

        cur = list(positions)
        used = 0
        changed = False

        groups = self._cluster_groups(n, constraints)
        # More split clusters first.
        cluster_items = []
        for gid, members in groups.items():
            comps = self._cluster_components(cur, members)
            if len(comps) > 1:
                cluster_items.append((-len(comps), gid, members, comps))
        cluster_items.sort()

        for _neg_c, gid, members, _old_comps in cluster_items:
            if used >= budget:
                break

            # Keep trying until this cluster is connected or budget stops.
            while used < budget:
                comps = self._cluster_components(cur, members)
                if len(comps) <= 1:
                    break

                anchor = self._choose_anchor_component(comps, cur, constraints)
                movable_comps = [c for c in comps if c is not anchor and not any(self._is_preplaced(i, constraints) for i in c)]
                if not movable_comps:
                    break

                # Move closest / smallest component first.  This gives highest chance
                # of legal exact abut without large bbox damage.
                movable_comps.sort(key=lambda c: (len(c), self._component_distance(c, anchor, cur)))
                comp = movable_comps[0]

                before_score = self._placement_score(cur, constraints, adj, pin_adj, pins_pos)
                shifts = self._cluster_abut_shifts(comp, anchor, cur, constraints)
                best = None
                best_score = before_score

                for dx, dy in shifts[: self.max_pair_candidates_per_component]:
                    trial = self._try_shift_component(cur, comp, dx, dy)
                    if trial is None:
                        continue
                    if not self._component_shift_is_legal(cur, trial, comp):
                        continue
                    score = self._placement_score(trial, constraints, adj, pin_adj, pins_pos)
                    # For grouping, require soft improvement, or same soft with proxy improvement.
                    if self._score_better(score, best_score, allow_equal_soft=True):
                        best_score = score
                        best = trial

                used += 1
                if best is None:
                    break
                cur = best
                changed = True

        return cur, used, changed

    # ================================================================
    # Soft scoring / acceptance
    # ================================================================

    def _placement_score(self, positions, constraints, adj, pin_adj, pins_pos):
        b, g, m = self._soft_counts(positions, constraints)
        soft = b + g + m
        try:
            proxy = float(self._total_proxy(positions, constraints, adj, pin_adj, pins_pos))
        except Exception:
            proxy = self._fallback_proxy(positions)
        return (int(soft), int(b), int(g), int(m), float(proxy))

    def _score_better(self, new_score, old_score, allow_equal_soft: bool) -> bool:
        ns, nb, ng, nm, np = new_score
        os, ob, og, om, op = old_score
        if ns < os:
            # Soft is exponential in the official cost, so allow bounded proxy sacrifice.
            return np <= op * (1.0 + self.soft_proxy_slack)
        if allow_equal_soft and ns == os:
            # Prefer boundary reduction first because boundary often gets worse when
            # connecting clusters; then grouping; then proxy.
            if nb < ob and np <= op * (1.0 + 0.12):
                return True
            if nb == ob and ng < og and np <= op * (1.0 + 0.10):
                return True
            return np <= op * (1.0 - self.equal_soft_proxy_slack)
        return False

    def _soft_counts(self, positions, constraints):
        bbox = self._bbox(positions)
        if bbox is None or constraints is None:
            return 0, 0, 0

        n = len(positions)
        boundary = 0
        for i in range(n):
            if self._boundary_code(i, constraints) == 0:
                continue
            if self._boundary_violation_score(i, positions[i], bbox, constraints) > 0.0:
                boundary += 1

        grouping = 0
        for _gid, members in self._cluster_groups(n, constraints).items():
            comps = self._cluster_components(positions, members)
            if len(comps) > 1:
                grouping += len(comps) - 1

        mib = 0
        mib_groups: Dict[int, List[int]] = {}
        for i in range(n):
            mid = self._mib_id(i, constraints)
            if mid != 0:
                mib_groups.setdefault(mid, []).append(i)
        for _mid, members in mib_groups.items():
            shapes = set()
            for i in members:
                _x, _y, w, h = positions[i]
                shapes.add((round(float(w), 5), round(float(h), 5)))
            if len(shapes) > 1:
                mib += len(shapes) - 1

        return boundary, grouping, mib

    def _fallback_proxy(self, positions):
        bbox = self._bbox(positions)
        if bbox is None:
            return 1e30
        x0, y0, x1, y1 = bbox
        return max((x1 - x0) * (y1 - y0), 1.0)

    # ================================================================
    # Component geometry
    # ================================================================

    def _cluster_groups(self, n, constraints):
        groups: Dict[int, List[int]] = {}
        if constraints is None:
            return groups
        for i in range(n):
            gid = self._cluster_id(i, constraints)
            if gid != 0:
                groups.setdefault(gid, []).append(i)
        return groups

    def _cluster_components(self, positions, members):
        members = list(members)
        remaining = set(members)
        comps = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            comp = [root]
            stack = [root]
            while stack:
                u = stack.pop()
                for v in list(remaining):
                    if self._abut(positions[u], positions[v]):
                        remaining.remove(v)
                        stack.append(v)
                        comp.append(v)
            comps.append(sorted(comp))
        comps.sort(key=lambda c: (min(c), len(c)))
        return comps

    def _movable_same_cluster_component(self, idx, positions, constraints):
        if self._is_preplaced(idx, constraints):
            return []
        gid = self._cluster_id(idx, constraints)
        if gid == 0:
            return [idx]
        members = [i for i in range(len(positions)) if self._cluster_id(i, constraints) == gid]
        for comp in self._cluster_components(positions, members):
            if idx in comp:
                if any(self._is_preplaced(i, constraints) for i in comp):
                    return []
                return comp
        return [idx]

    def _choose_anchor_component(self, comps, positions, constraints):
        def key(comp):
            has_preplaced = any(self._is_preplaced(i, constraints) for i in comp)
            boundary_ok = 0
            boundary_total = 0
            bbox = self._bbox(positions)
            for i in comp:
                if self._boundary_code(i, constraints) != 0:
                    boundary_total += 1
                    if bbox is not None and self._boundary_violation_score(i, positions[i], bbox, constraints) == 0.0:
                        boundary_ok += 1
            area = sum(positions[i][2] * positions[i][3] for i in comp)
            return (
                0 if has_preplaced else 1,
                -boundary_ok,
                -boundary_total,
                -len(comp),
                -area,
                min(comp),
            )
        return sorted(comps, key=key)[0]

    def _component_bbox(self, comp, positions):
        x0 = min(positions[i][0] for i in comp)
        y0 = min(positions[i][1] for i in comp)
        x1 = max(positions[i][0] + positions[i][2] for i in comp)
        y1 = max(positions[i][1] + positions[i][3] for i in comp)
        return x0, y0, x1, y1

    def _component_distance(self, a, b, positions):
        ax0, ay0, ax1, ay1 = self._component_bbox(a, positions)
        bx0, by0, bx1, by1 = self._component_bbox(b, positions)
        dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
        dy = max(0.0, max(by0 - ay1, ay0 - by1))
        return dx + dy

    def _try_shift_component(self, positions, comp, dx, dy):
        dx = float(dx)
        dy = float(dy)
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return None
        trial = list(positions)
        for i in comp:
            x, y, w, h = trial[i]
            nx = x + dx
            ny = y + dy
            if nx < -1e-8 or ny < -1e-8:
                return None
            trial[i] = (max(0.0, nx), max(0.0, ny), w, h)
        return trial

    def _component_shift_is_legal(self, old_positions, new_positions, comp):
        moving = set(comp)
        # moved blocks should not overlap each other; normally already true.
        for a_idx, i in enumerate(comp):
            for j in comp[a_idx + 1:]:
                if self._overlap_area(new_positions[i], new_positions[j]) > 1e-9:
                    return False
        # moved blocks should not overlap fixed/non-moving blocks.
        for i in comp:
            ri = new_positions[i]
            for j, rj in enumerate(new_positions):
                if j in moving:
                    continue
                if self._overlap_area(ri, rj) > 1e-9:
                    return False
        return True

    # ================================================================
    # Candidate shifts
    # ================================================================

    def _boundary_shifts_for_component(self, comp, positions, constraints):
        bbox = self._bbox(positions)
        if bbox is None:
            return []
        bx0, by0, bx1, by1 = bbox

        shifts = []
        seen = set()

        def add(dx, dy):
            key = (round(float(dx), 7), round(float(dy), 7))
            if key not in seen:
                seen.add(key)
                shifts.append((float(dx), float(dy)))

        # Generate shifts that satisfy one or more boundary members in this component.
        for i in comp:
            code = self._boundary_code(i, constraints)
            if code == 0:
                continue
            x, y, w, h = positions[i]
            dx_options = [0.0]
            dy_options = [0.0]
            if code & 1:
                dx_options.append(bx0 - x)
            if code & 2:
                dx_options.append(bx1 - (x + w))
            if code & 8:
                dy_options.append(by0 - y)
            if code & 4:
                dy_options.append(by1 - (y + h))

            # If a block has both horizontal and vertical boundary bits, try combined shift first.
            for dx in dx_options:
                for dy in dy_options:
                    if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                        add(dx, dy)

        # Sort by smallest movement first, but prioritize shifts satisfying more boundary bits.
        def shift_key(s):
            dx, dy = s
            satisfied = 0
            for i in comp:
                code = self._boundary_code(i, constraints)
                if code == 0:
                    continue
                x, y, w, h = positions[i]
                nx = x + dx
                ny = y + dy
                if (code & 1) and abs(nx - bx0) < 1e-6:
                    satisfied += 1
                if (code & 2) and abs(nx + w - bx1) < 1e-6:
                    satisfied += 1
                if (code & 8) and abs(ny - by0) < 1e-6:
                    satisfied += 1
                if (code & 4) and abs(ny + h - by1) < 1e-6:
                    satisfied += 1
            return (-satisfied, abs(dx) + abs(dy))

        shifts.sort(key=shift_key)
        return shifts[:24]

    def _cluster_abut_shifts(self, comp, anchor, positions, constraints):
        shifts = []
        seen = set()

        def add(dx, dy):
            key = (round(float(dx), 7), round(float(dy), 7))
            if key not in seen:
                seen.add(key)
                shifts.append((float(dx), float(dy)))

        # Pair-level exact abut candidates.  Limit if components are large.
        comp_members = sorted(comp)[: self.max_members_for_pair_candidates]
        anchor_members = sorted(anchor)[: self.max_members_for_pair_candidates]
        for i in comp_members:
            ax, ay, aw, ah = positions[i]
            for j in anchor_members:
                bx, by, bw, bh = positions[j]
                # Put i right/left/top/bottom of j with several orthogonal alignments.
                y_aligns = [by, by + bh - ah, by + 0.5 * (bh - ah)]
                x_aligns = [bx, bx + bw - aw, bx + 0.5 * (bw - aw)]
                for yy in y_aligns:
                    add((bx + bw) - ax, yy - ay)
                    add((bx - aw) - ax, yy - ay)
                for xx in x_aligns:
                    add(xx - ax, (by + bh) - ay)
                    add(xx - ax, (by - ah) - ay)

        # Bbox-level candidates: faster and often enough.
        cx0, cy0, cx1, cy1 = self._component_bbox(comp, positions)
        ax0, ay0, ax1, ay1 = self._component_bbox(anchor, positions)
        add(ax1 - cx0, ay0 - cy0)
        add(ax0 - cx1, ay0 - cy0)
        add(ax0 - cx0, ay1 - cy0)
        add(ax0 - cx0, ay0 - cy1)
        add(ax1 - cx0, ay1 - cy1)
        add(ax0 - cx1, ay1 - cy1)

        # Boundary-aware ordering: if moving component has boundary blocks, try to
        # keep/snap them to the current frame after abutment.
        bbox = self._bbox(positions)
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            for i in comp:
                code = self._boundary_code(i, constraints)
                if code == 0:
                    continue
                x, y, w, h = positions[i]
                if code & 1:
                    add(bx0 - x, 0.0)
                if code & 2:
                    add(bx1 - (x + w), 0.0)
                if code & 8:
                    add(0.0, by0 - y)
                if code & 4:
                    add(0.0, by1 - (y + h))

        # Sort by small movement.  The score guard will reject bad moves.
        shifts.sort(key=lambda s: abs(s[0]) + abs(s[1]))
        return shifts

    def _boundary_distance_sum(self, idx, rect, bbox, constraints):
        code = self._boundary_code(idx, constraints)
        x0, y0, x1, y1 = bbox
        x, y, w, h = rect
        d = 0.0
        if code & 1:
            d += abs(x - x0)
        if code & 2:
            d += abs((x + w) - x1)
        if code & 8:
            d += abs(y - y0)
        if code & 4:
            d += abs((y + h) - y1)
        return d


# Backward-compatible alias if any evaluator introspects it.
Optimizer = MyOptimizer
