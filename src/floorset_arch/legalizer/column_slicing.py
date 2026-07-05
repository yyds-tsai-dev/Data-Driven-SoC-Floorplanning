#!/usr/bin/env python3
# Vendored from partner/legalizer_claude_v2.py (column-slicing + SA legalizer,
# teammate contribution, 2026-07-05). Do not hand-edit without eval evidence.
"""Column-slicing legalizer + simulated-annealing HPWL optimizer.

Layout model ("variant A" column slicing):
  * Pick a frame height H (from total area / target utilization and the
    pin-bounding-box aspect ratio).
  * Blocks are grouped into *units* (cluster groups and soft-MIB groups,
    merged via union-find).  Each unit is a vertical stack of slices.
  * Columns are laid out left to right.  Every soft slice in a column has
    width == column width and height == area / width, so soft-block areas
    are exact and the column fills its height with zero dead space.
    The column width is solved so the content height equals H:
        w_c = soft_area / (H - rigid_height - obstacle_height)
  * Preplaced blocks stay at their exact target position and act as
    obstacles; stacking skips their y-intervals.  A cluster that contains
    a preplaced member is placed flush against that block so the group
    stays connected.
  * Fixed-shape blocks keep their exact (w, h); the column is widened to
    at least the widest rigid member.
  * Boundary handling: left/right-tagged units are pinned to the first /
    last column (every slice there touches x_min / x_max), bottom-tagged
    units stack first (y = 0), top-tagged units are lifted flush to the
    global top edge.

Hard guarantees by construction: no overlaps, exact soft areas, exact
fixed/preplaced dimensions and positions.

A time-budgeted simulated annealing then permutes units across columns to
minimize a proxy of the contest cost (HPWL + bbox area + soft violations).
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

Rect = Tuple[float, float, float, float]

EPS = 1e-9
TOUCH_TOL = 1e-7
UTIL_TARGET_FRAME = 0.96   # utilization used to size the frame height
UTIL_TARGET_REF = 0.97     # utilization used for the area cost reference
MIB_AREA_GUARD = 0.0095    # stay under the 1% hard area tolerance


def _cumulative(weights: List[float]) -> List[float]:
    """Cumulative-threshold list for the move selector. The last entry is
    pinned to 1.0 so a uniform draw always lands in some family."""
    acc = 0.0
    cum = []
    for w in weights:
        acc += w
        cum.append(acc)
    if cum:
        cum[-1] = 1.0
    return cum


# =============================================================================
# Constraint parsing helpers (API kept for my_opt_claude.py)
# =============================================================================
def _col(constraints: Optional[torch.Tensor], n: int, idx: int) -> List[float]:
    if constraints is None or constraints.dim() < 2 or constraints.shape[1] <= idx:
        return [0.0] * n
    return [float(v) for v in constraints[:n, idx]]


def _parse_constraints(constraints: Optional[torch.Tensor], n: int):
    fixed = [v != 0 for v in _col(constraints, n, 0)]
    preplaced = [v != 0 for v in _col(constraints, n, 1)]
    mib = [int(round(v)) for v in _col(constraints, n, 2)]
    cluster = [int(round(v)) for v in _col(constraints, n, 3)]
    boundary = [int(round(v)) for v in _col(constraints, n, 4)]
    return fixed, preplaced, mib, cluster, boundary


def _target(target_positions: Optional[torch.Tensor], i: int) -> Tuple[float, float, float, float]:
    if target_positions is None:
        return (-1.0, -1.0, -1.0, -1.0)
    tx, ty, tw, th = [float(v) for v in target_positions[i]]
    return tx, ty, tw, th


# =============================================================================
# Rectangle reconstruction from diffusion z output (kept for my_opt_claude.py)
# =============================================================================
def rectangles_from_z(
    z: torch.Tensor,
    area_targets: torch.Tensor,
    constraints: Optional[torch.Tensor] = None,
    target_positions: Optional[torch.Tensor] = None,
    z_repr: str = "xylogwh",
) -> List[Rect]:
    n = int((area_targets != -1).sum().item())
    scale = math.sqrt(max(1.0, sum(float(a) for a in area_targets[:n] if float(a) > 0)))
    rects: List[Rect] = []
    for i in range(n):
        area = max(float(area_targets[i]), 1e-6)
        x = float(z[i, 0]) * scale
        y = float(z[i, 1]) * scale
        if z.shape[-1] >= 4:
            if z_repr == "xyaspect":
                log_aspect = max(-3.0, min(3.0, float(z[i, 2])))
            else:
                log_aspect = max(-3.0, min(3.0, float(z[i, 2] - z[i, 3])))
        else:
            log_aspect = max(-3.0, min(3.0, float(z[i, 2])))
        aspect = math.exp(log_aspect)
        w = math.sqrt(area * aspect)
        h = math.sqrt(area / aspect)
        rects.append((x, y, max(w, 1e-6), max(h, 1e-6)))
    return rects


# =============================================================================
# Units
# =============================================================================
class _Unit:
    __slots__ = ("uid", "subgroups", "blocks", "soft_area", "rigid_h", "max_rigid_w",
                 "area_total", "force", "anchors", "seed_x", "seed_y", "hasB", "hasT",
                 "bands", "eff_soft", "eff_rigid_h", "banded", "hcache", "dyn", "pairable",
                 "version")

    def __init__(self, uid: int):
        self.uid = uid
        self.subgroups: List[List[int]] = []
        self.blocks: List[int] = []
        self.soft_area = 0.0
        self.rigid_h = 0.0
        self.max_rigid_w = 0.0
        self.area_total = 0.0
        self.force: Optional[str] = None   # 'L' or 'R'
        self.anchors: List[Rect] = []
        self.seed_x = 0.0
        self.seed_y = 0.0
        self.hasB = False
        self.hasT = False
        # monotone counter bumped whenever the unit's internal geometry
        # changes (subgroup reorder etc.); keys the FAST_EVAL layout cache.
        self.version = 0

    def flatten(self):
        self.blocks = [b for sg in self.subgroups for b in sg]


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# =============================================================================
# Main optimizer
# =============================================================================
class _ColumnOptimizer:
    def __init__(
        self,
        rects: Sequence[Rect],
        area_targets: torch.Tensor,
        constraints: Optional[torch.Tensor],
        target_positions: Optional[torch.Tensor],
        b2b: Optional[torch.Tensor],
        p2b: Optional[torch.Tensor],
        pins: Optional[torch.Tensor],
        deadline: Optional[float],
        seed: int = 0,
        v_weight: float = 1.0,
    ):
        self.n = len(rects)
        self.rects = rects
        self.deadline = deadline if deadline is not None else time.time() + 3.0
        self.rng = random.Random(seed)
        self.v_weight = v_weight  # >1: search prioritizes killing violations

        n = self.n
        self.areas = [max(float(area_targets[i]), 1e-9) for i in range(n)]
        self.fixed, self.preplaced, self.mib, self.cluster, self.boundary = \
            _parse_constraints(constraints, n)
        self.tpos = target_positions

        # kind: 0 = soft, 1 = rigid (exact w,h), 2 = locked (exact x,y,w,h)
        self.kind = [0] * n
        self.rw = [0.0] * n
        self.rh = [0.0] * n
        self.lx = [0.0] * n
        self.ly = [0.0] * n
        self._resolve_shapes()

        self.locked_rects: List[Rect] = [
            (self.lx[i], self.ly[i], self.rw[i], self.rh[i])
            for i in range(n) if self.kind[i] == 2
        ]
        # x-projections of the obstacles, for the FAST_EVAL cache: a column
        # whose [x, x+w) span intersects one of these has geometry that
        # depends on its absolute x, so it cannot be rigid-shifted blindly.
        self._obs_x = [(ox, ox + ow) for (ox, _oy, ow, _oh) in self.locked_rects]
        # FAST_EVAL: memoize per-column base geometry keyed by (unit ids,
        # unit versions). Read the flag once so it is stable for this
        # optimizer's lifetime. Cache is per-instance (never shared across
        # workers/pickle).
        self._fast_eval = os.environ.get("FLOORSET_FAST_EVAL", "0") == "1"
        self._col_cache: Dict[tuple, tuple] = {}

        # --- M1c: adaptive move probabilities (FLOORSET_SA_ADAPTIVE) ---------
        # The move selector in `_random_move` picks a family from a single
        # rng.random() draw compared against these cumulative thresholds. With
        # the flag OFF these stay bit-identical to the historical fixed mix
        # (0.55 / 0.25 / 0.12 / 0.08), so the rng stream and branch structure
        # -- and therefore the FAST_EVAL equivalence shadow check -- are
        # unchanged. With the flag ON, `_anneal` reweights them from a sliding
        # window of per-family acceptance x mean accepted improvement.
        self._adaptive_moves = os.environ.get("FLOORSET_SA_ADAPTIVE", "0") == "1"
        self._move_base = [0.55, 0.25, 0.12, 0.08]  # relocate/swap/reorder/subgrp
        self._move_cum = _cumulative(self._move_base)
        self._last_move_type = -1  # family index set by every _random_move call

        # --- M1c: PARSAC-style constraint-fixing move (FLOORSET_SA_CFIX) -----
        # With small probability, when soft violations remain, perform a
        # targeted boundary-repair move and accept it unconditionally. Hard
        # legality is by construction, so this can never break it.
        self._cfix = os.environ.get("FLOORSET_SA_CFIX", "0") == "1"
        self._cfix_p = 0.001

        # --- M3: post-SA column width re-optimization (FLOORSET_WIDTH_OPT) ---
        # After the SA fixes the column assignment, coordinate-descend the
        # per-column widths (golden-section over a bounded multiplier) to trim
        # HPWL, exploiting the exact-area / free-aspect soft blocks. Runs on
        # the from-scratch `_layout_widths` path so it never poisons the
        # FAST_EVAL cache. Optional JSONL pre/post log via FLOORSET_WIDTH_OPT_LOG.
        self._width_opt = os.environ.get("FLOORSET_WIDTH_OPT", "0") == "1"
        self._width_opt_log = os.environ.get("FLOORSET_WIDTH_OPT_LOG", "")

        self._build_hpwl_arrays(b2b, p2b, pins)
        self._build_soft_norm()
        self._choose_frame(pins)
        self._build_units()

    # ------------------------------------------------------------------
    def _resolve_shapes(self):
        n = self.n
        for i in range(n):
            tx, ty, tw, th = _target(self.tpos, i)
            if self.preplaced[i] and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                self.kind[i] = 2
                self.lx[i], self.ly[i], self.rw[i], self.rh[i] = tx, ty, tw, th
            elif (self.fixed[i] or self.preplaced[i]) and tw > 0 and th > 0:
                self.kind[i] = 1
                self.rw[i], self.rh[i] = tw, th

        # MIB groups: if a member has a hard shape, all soft members must copy
        # it exactly to reach zero MIB violations (areas in a group are equal).
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            if self.mib[i] > 0:
                groups[self.mib[i]].append(i)
        self.mib_groups = groups
        for idxs in groups.values():
            ref = next((i for i in idxs if self.kind[i] != 0), None)
            if ref is None:
                continue
            w, h = self.rw[ref], self.rh[ref]
            for i in idxs:
                if self.kind[i] == 0 and abs(w * h - self.areas[i]) / self.areas[i] <= MIB_AREA_GUARD:
                    self.kind[i] = 1
                    self.rw[i], self.rh[i] = w, h

    # ------------------------------------------------------------------
    def _build_hpwl_arrays(self, b2b, p2b, pins):
        n = self.n
        if b2b is not None and len(b2b) > 0 and b2b.dim() == 2 and b2b.shape[1] >= 3:
            arr = b2b.detach().cpu().numpy()
            m = (arr[:, 0] != -1) & (arr[:, 0] < n) & (arr[:, 1] < n) & (arr[:, 0] >= 0) & (arr[:, 1] >= 0)
            self.eI = arr[m, 0].astype(np.int64)
            self.eJ = arr[m, 1].astype(np.int64)
            self.eW = arr[m, 2].astype(np.float64)
        else:
            self.eI = np.zeros(0, dtype=np.int64)
            self.eJ = np.zeros(0, dtype=np.int64)
            self.eW = np.zeros(0)
        if (p2b is not None and pins is not None and len(p2b) > 0
                and p2b.dim() == 2 and p2b.shape[1] >= 3 and len(pins) > 0):
            arr = p2b.detach().cpu().numpy()
            pn = pins.detach().cpu().numpy()
            m = (arr[:, 0] != -1) & (arr[:, 0] >= 0) & (arr[:, 0] < len(pn)) \
                & (arr[:, 1] >= 0) & (arr[:, 1] < n)
            pidx = arr[m, 0].astype(np.int64)
            self.pB = arr[m, 1].astype(np.int64)
            self.pW = arr[m, 2].astype(np.float64)
            self.pX = pn[pidx, 0].astype(np.float64)
            self.pY = pn[pidx, 1].astype(np.float64)
        else:
            self.pB = np.zeros(0, dtype=np.int64)
            self.pW = np.zeros(0)
            self.pX = np.zeros(0)
            self.pY = np.zeros(0)

    def _build_soft_norm(self):
        n_soft = sum(1 for i in range(self.n) if self.boundary[i] > 0)
        for idxs in self.mib_groups.values():
            n_soft += max(0, len(idxs) - 1)
        clus: Dict[int, List[int]] = defaultdict(list)
        for i in range(self.n):
            if self.cluster[i] > 0:
                clus[self.cluster[i]].append(i)
        self.cluster_groups = clus
        for idxs in clus.values():
            n_soft += max(0, len(idxs) - 1)
        self.n_soft_den = max(n_soft, 1)

        bnd = [(i, self.boundary[i]) for i in range(self.n) if self.boundary[i] > 0]
        self._bnd_idx = np.array([i for i, _ in bnd], dtype=np.int64)
        self._bnd_codes = np.array([c for _, c in bnd], dtype=np.int64)
        # clusters whose members are all movable form one contiguous stack by
        # construction — only groups containing locked blocks need checking
        self._clu_arrays = [np.array(idxs, dtype=np.int64)
                            for idxs in clus.values()
                            if len(idxs) >= 2 and any(self.kind[i] == 2 for i in idxs)]
        self._mib_arrays = [np.array(idxs, dtype=np.int64)
                            for idxs in self.mib_groups.values() if len(idxs) >= 2]

    # ------------------------------------------------------------------
    def _choose_frame(self, pins):
        total = 0.0
        for i in range(self.n):
            if self.kind[i] == 0:
                total += self.areas[i]
            else:
                total += self.rw[i] * self.rh[i]
        self.total_area = max(total, 1.0)
        aspect = 1.0
        if pins is not None and len(pins) > 1:
            pn = pins.detach().cpu().numpy()
            valid = pn[(pn[:, 0] != -1) & (pn[:, 1] != -1)]
            if len(valid) >= 2:
                dx = float(valid[:, 0].max() - valid[:, 0].min())
                dy = float(valid[:, 1].max() - valid[:, 1].min())
                if dx > 1.0 and dy > 1.0:
                    aspect = max(0.35, min(2.8, dx / dy))
        frame_area = self.total_area / UTIL_TARGET_FRAME
        H = math.sqrt(frame_area / aspect)
        # A preplaced block tagged "touch top" reveals the intended frame
        # height exactly — pin H to it so the tag is satisfiable.
        pinned_H = None
        for i in range(self.n):
            if self.kind[i] == 2 and (self.boundary[i] & 4):
                top = self.ly[i] + self.rh[i]
                pinned_H = top if pinned_H is None else max(pinned_H, top)
        if pinned_H is not None:
            H = pinned_H
        for (x, y, w, h) in self.locked_rects:
            H = max(H, y + h)
        for i in range(self.n):
            if self.kind[i] == 1:
                H = max(H, self.rh[i])
        self.H = H
        self.W_est = frame_area / H
        self.area_ref = self.total_area / UTIL_TARGET_REF

    # ------------------------------------------------------------------
    def _build_units(self):
        n = self.n
        movable = [i for i in range(n) if self.kind[i] != 2]
        dsu = _DSU(n)
        for idxs in self.cluster_groups.values():
            mv = [i for i in idxs if self.kind[i] != 2]
            for a, b in zip(mv, mv[1:]):
                dsu.union(a, b)
        for idxs in self.mib_groups.values():
            # soft (no hard-shape reference) MIB members must share a column
            if any(self.kind[i] != 0 for i in idxs):
                continue
            for a, b in zip(idxs, idxs[1:]):
                dsu.union(a, b)

        roots: Dict[int, List[int]] = defaultdict(list)
        for i in movable:
            roots[dsu.find(i)].append(i)

        units: List[_Unit] = []
        for uid, (_, ids) in enumerate(sorted(roots.items())):
            u = _Unit(uid)
            # subgroups: one per cluster id (contiguity requirement); blocks
            # without a cluster are their own singleton subgroup.
            sub: Dict[int, List[int]] = defaultdict(list)
            singles: List[List[int]] = []
            for i in ids:
                if self.cluster[i] > 0:
                    sub[self.cluster[i]].append(i)
                else:
                    singles.append([i])
            subgroups = list(sub.values()) + singles

            def blk_key(i):
                b = self.boundary[i]
                rank = 0 if (b & 8) else (2 if (b & 4) else 1)
                return (rank, self.rects[i][1] if i < len(self.rects) else 0.0)

            for sg in subgroups:
                sg.sort(key=blk_key)

            def sg_key(sg):
                has_b = any(self.boundary[i] & 8 for i in sg)
                has_t = any(self.boundary[i] & 4 for i in sg)
                rank = 0 if has_b else (2 if has_t else 1)
                my = sum(self.rects[i][1] for i in sg) / len(sg)
                return (rank, my)

            subgroups.sort(key=sg_key)
            u.subgroups = subgroups
            u.flatten()

            for i in u.blocks:
                if self.kind[i] == 0:
                    u.soft_area += self.areas[i]
                    u.area_total += self.areas[i]
                else:
                    u.rigid_h += self.rh[i]
                    u.max_rigid_w = max(u.max_rigid_w, self.rw[i])
                    u.area_total += self.rw[i] * self.rh[i]
                b = self.boundary[i]
                if b & 1:
                    u.force = 'L'
                elif b & 2 and u.force is None:
                    u.force = 'R'
                if b & 8:
                    u.hasB = True
                if b & 4:
                    u.hasT = True

            # anchors: preplaced members of any cluster group in this unit
            anchor_ids = set()
            for i in u.blocks:
                g = self.cluster[i]
                if g > 0:
                    for j in self.cluster_groups[g]:
                        if self.kind[j] == 2:
                            anchor_ids.add(j)
            u.anchors = [(self.lx[j], self.ly[j], self.rw[j], self.rh[j]) for j in sorted(anchor_ids)]

            wsum = max(u.area_total, 1e-9)
            sx = sy = 0.0
            for i in u.blocks:
                a = self.areas[i] if self.kind[i] == 0 else self.rw[i] * self.rh[i]
                r = self.rects[i] if i < len(self.rects) else (0, 0, 1, 1)
                sx += a * (r[0] + r[2] * 0.5)
                sy += a * (r[1] + r[3] * 0.5)
            u.seed_x = sx / wsum
            u.seed_y = sy / wsum
            units.append(u)

        self.units = units
        for u in units:
            self._refresh_unit(u)
            u.pairable = (len(u.blocks) == 1 and self.kind[u.blocks[0]] == 0
                          and u.force is None and not u.hasB and not u.hasT
                          and not u.anchors)
        self.blk_unit = [-1] * n
        for k, u in enumerate(units):
            for i in u.blocks:
                self.blk_unit[i] = k
        self._unit_col = [0] * len(units)
        self._col_spans: List[Optional[Tuple[float, float]]] = []

    def _col_of_x(self, xq: float, C: int) -> int:
        spans = self._col_spans
        best = None
        best_d = None
        for ci, sp in enumerate(spans[:C]):
            if sp is None:
                continue
            if sp[0] - 1e-9 <= xq <= sp[1] + 1e-9:
                return ci
            d = min(abs(xq - sp[0]), abs(xq - sp[1]))
            if best_d is None or d < best_d:
                best_d = d
                best = ci
        if best is None:
            return self.rng.randrange(C)
        return best

    # ------------------------------------------------------------------
    def _init_columns(self, C: int) -> List[List[int]]:
        cols: List[List[int]] = [[] for _ in range(C)]
        units = self.units
        total = sum(u.area_total for u in units)
        cap = [total / C] * C          # area capacity per column
        rigid_cap = 0.55 * self.H      # keep columns solvable (content <= H)
        rigid_used = [0.0] * C

        def put(k: int, ci: int):
            u = units[k]
            cols[ci].append(k)
            cap[ci] -= u.area_total
            rigid_used[ci] += u.rigid_h

        forced = []
        free = []
        for k, u in enumerate(units):
            if u.force == 'L':
                forced.append((k, 0))
            elif u.force == 'R':
                forced.append((k, C - 1))
            elif u.anchors:
                ax = u.anchors[0][0] + u.anchors[0][2] * 0.5
                forced.append((k, max(0, min(C - 1, int(ax / max(self.W_est, 1e-6) * C)))))
            else:
                free.append(k)
        for k, ci in forced:
            put(k, ci)

        free.sort(key=lambda k: units[k].seed_x)
        ci = 0
        for k in free:
            u = units[k]
            while ci < C - 1 and (cap[ci] <= 0.0 or rigid_used[ci] + u.rigid_h > rigid_cap):
                ci += 1
            # spill rigid-heavy units to the least-loaded feasible column
            if rigid_used[ci] + u.rigid_h > rigid_cap:
                ci2 = min(range(C), key=lambda c: rigid_used[c])
                put(k, ci2)
            else:
                put(k, ci)

        def unit_key(k):
            u = units[k]
            has_b = any(self.boundary[i] & 8 for i in u.blocks)
            has_t = any(self.boundary[i] & 4 for i in u.blocks)
            rank = 0 if has_b else (2 if has_t else 1)
            return (rank, u.seed_y)

        for c in cols:
            c.sort(key=unit_key)
        return cols

    # ------------------------------------------------------------------
    def _obstacles_in(self, x0: float, x1: float) -> List[List[float]]:
        iv = []
        for (ox, oy, ow, oh) in self.locked_rects:
            if ox < x1 - 1e-9 and ox + ow > x0 + 1e-9:
                iv.append([oy, oy + oh])
        if not iv:
            return []
        iv.sort()
        merged = [iv[0][:]]
        for s, e in iv[1:]:
            if s <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def _unit_height(self, u: _Unit, w: float) -> float:
        return self._unit_h(u, w)

    def _partition_subgroup(self, sg: List[int]) -> List[List[int]]:
        """Split a cluster with several bottom/top-tagged members into
        side-by-side chunks so each tag can touch its edge. Connectivity is
        preserved: chunks stand shoulder to shoulder inside the unit."""
        if len(sg) < 2:
            return [sg]
        bnd = self.boundary
        B = [i for i in sg if bnd[i] & 8]
        T = [i for i in sg if (bnd[i] & 4) and not (bnd[i] & 8)]
        hasL = any(bnd[i] & 1 for i in sg)
        hasR = any(bnd[i] & 2 for i in sg)
        m = max(len(B), len(T))
        # a single left/right-tagged member should not force the whole
        # cluster into one full-width stack — flatten it into chunks and put
        # that member on the outer edge
        if (hasL or hasR) and len(sg) >= 3:
            m = max(m, min(3, len(sg) - 1))
        if m < 2:
            return [sg]
        if any(self.kind[i] == 0 and self.mib[i] > 0 for i in sg):
            return [sg]
        m = min(m, len(sg), 4)

        def blk_area(i):
            return self.areas[i] if self.kind[i] == 0 else self.rw[i] * self.rh[i]

        chunks: List[List[int]] = [[] for _ in range(m)]
        areas = [0.0] * m
        assigned = set()
        for i in sg:
            if bnd[i] & 1:
                chunks[0].append(i)
                areas[0] += blk_area(i)
                assigned.add(i)
            elif bnd[i] & 2:
                chunks[m - 1].append(i)
                areas[m - 1] += blk_area(i)
                assigned.add(i)
        b_seen = [any(bnd[i] & 8 for i in ch) for ch in chunks]
        for i in B:
            if i in assigned:
                continue
            free = [c for c in range(m) if not b_seen[c]]
            c = free[0] if free else min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)
            b_seen[c] = True
            assigned.add(i)
        t_seen = [any(bnd[i] & 4 for i in ch) for ch in chunks]
        for i in T:
            if i in assigned:
                continue
            free = [c for c in range(m) if not t_seen[c]]
            c = free[0] if free else min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)
            t_seen[c] = True
            assigned.add(i)
        for i in sorted((j for j in sg if j not in assigned), key=lambda j: -blk_area(j)):
            c = min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)

        def blk_rank(i):
            b = bnd[i]
            return (0 if (b & 8) else (2 if (b & 4) else 1))

        for ch in chunks:
            ch.sort(key=blk_rank)
        return [c for c in chunks if c]

    def _refresh_unit(self, u: _Unit):
        u.flatten()
        bands = []
        eff_s = 0.0
        eff_r = 0.0
        for sg in u.subgroups:
            band = []
            for part in self._partition_subgroup(sg):
                pl = [(i, self.kind[i] == 0, self.areas[i], self.rw[i], self.rh[i])
                      for i in part]
                sa = sum(self.areas[i] for i in part if self.kind[i] == 0)
                rh = sum(self.rh[i] for i in part if self.kind[i] != 0)
                rmw = max((self.rw[i] for i in part if self.kind[i] != 0), default=0.0)
                band.append((pl, sa, rh, rmw))
            bands.append(band)
            if len(band) == 1:
                eff_s += band[0][1]
                eff_r += band[0][2]
            else:
                eff_s += sum(ch[1] for ch in band)
                eff_s += sum(bh * bw for ch in band
                             for (_i, soft, _a, bw, bh) in ch[0] if not soft)
        u.bands = bands
        u.eff_soft = eff_s
        u.eff_rigid_h = eff_r
        # bands eligible for width-dependent splitting at layout time: a
        # single stack of >=2 blocks without soft-MIB members (those need one
        # shared width for identical shapes)
        u.dyn = [len(b) == 1 and len(b[0][0]) >= 2
                 and not any(self.kind[e[0]] == 0 and self.mib[e[0]] > 0 for e in b[0][0])
                 for b in bands]
        u.banded = any(len(b) > 1 for b in bands) or any(u.dyn)
        u.hcache = None
        # invalidate any cached column geometry that includes this unit
        u.version += 1

    def _solve_band(self, band, w: float):
        """Find the band height h so the chunk widths sum to w. Mixed chunks
        satisfy w_j = soft_j / (h - rigid_h_j); pure-rigid chunks have fixed
        width. Returns (h, widths) or None if infeasible at this width."""
        pure_w = 0.0
        lo = 0.0
        all_soft = True
        for (_pl, sa, rh, rmw) in band:
            if sa <= 0.0:
                pure_w += rmw
                all_soft = False
            elif rh > 0.0:
                all_soft = False
            if rh > lo:
                lo = rh
        avail = w - pure_w
        if avail <= 1e-6:
            return None
        if all_soft:
            total = sum(ch[1] for ch in band)
            h = total / w
            return h, [ch[1] / h for ch in band]
        lo += 1e-9

        def width_at(h):
            s = 0.0
            for (_pl, sa, rh, _rmw) in band:
                if sa > 0.0:
                    s += sa / (h - rh)
            return s

        hi = lo + 1.0
        for _ in range(80):
            if width_at(hi) <= avail:
                break
            hi *= 2.0
        else:
            return None
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if width_at(mid) > avail:
                lo = mid
            else:
                hi = mid
        h = hi
        widths = []
        for (_pl, sa, rh, rmw) in band:
            if sa > 0.0:
                wj = sa / (h - rh)
                if wj < rmw - 1e-9:
                    return None
                widths.append(wj)
            else:
                widths.append(rmw)
        return h, widths

    def _merge_band_once(self, band):
        """Merge the two adjacent chunks with the smallest combined area."""
        def chunk_area(ch):
            a = ch[1]
            for (_i, soft, _a, bw, bh) in ch[0]:
                if not soft:
                    a += bw * bh
            return a
        areas = [chunk_area(ch) for ch in band]
        j = min(range(len(band) - 1), key=lambda t: areas[t] + areas[t + 1])
        a, b = band[j], band[j + 1]
        bnd = self.boundary

        def rank(entry):
            code = bnd[entry[0]]
            return 0 if (code & 8) else (2 if (code & 4) else 1)

        pl = sorted(a[0] + b[0], key=rank)
        merged = (pl, a[1] + b[1], a[2] + b[2], max(a[3], b[3]))
        return band[:j] + [merged] + band[j + 2:]

    def _band_solutions(self, u: _Unit, w: float):
        """(w, unit_height, per-band placement plan) with a one-slot cache.
        Each plan entry is (effective_band, solution): when a chunk layout is
        infeasible at this width, chunks are merged progressively instead of
        collapsing the whole band into one stack."""
        if u.hcache is not None and u.hcache[0] == w:
            return u.hcache
        plan = []
        h = 0.0
        for band, dyn_ok in zip(u.bands, u.dyn):
            eff = band
            sol = None
            if len(eff) == 1 and dyn_ok:
                d = self._dyn_split(eff[0], w)
                if d is not None:
                    eff = d
            if len(eff) > 1:
                sol = self._solve_band(eff, w)
                while sol is None and len(eff) > 2:
                    eff = self._merge_band_once(eff)
                    sol = self._solve_band(eff, w)
            if sol is None:
                if len(eff) > 1:
                    eff = [(sorted((e for ch in eff for e in ch[0]),
                                   key=lambda t: 0 if (self.boundary[t[0]] & 8)
                                   else (2 if (self.boundary[t[0]] & 4) else 1)),
                            sum(ch[1] for ch in eff),
                            sum(ch[2] for ch in eff),
                            max(ch[3] for ch in eff))]
                ch = eff[0]
                hb = ch[2] + (ch[1] / w if ch[1] > 0.0 else 0.0)
            else:
                hb = sol[0]
            plan.append((eff, sol))
            h += hb
        u.hcache = (w, h, plan)
        return u.hcache

    def _dyn_split(self, ch, w: float):
        """Split a wide single-stack chunk into side-by-side chunks so slice
        aspect stays reasonable. Order-preserving, so cluster chains remain
        connected (adjacent chunks touch along their full shared edge)."""
        pl = ch[0]
        total = ch[1]
        for (_i, soft, _a, bw, bh) in pl:
            if not soft:
                total += bw * bh
        avg = total / len(pl)
        m = int(w / max(1.35 * math.sqrt(max(avg, 1e-9)), 1e-6))
        if m < 2:
            return None
        m = min(m, len(pl), 4)
        target = total / m
        parts = []
        cur = []
        acc = 0.0
        for e in pl:
            cur.append(e)
            acc += e[2] if e[1] else e[3] * e[4]
            if acc >= target - 1e-9 and len(parts) < m - 1:
                parts.append(cur)
                cur = []
                acc = 0.0
        if cur:
            parts.append(cur)
        if len(parts) < 2:
            return None
        chunks = []
        for part in parts:
            sa = sum(e[2] for e in part if e[1])
            rh = sum(e[4] for e in part if not e[1])
            rmw = max((e[3] for e in part if not e[1]), default=0.0)
            chunks.append((part, sa, rh, rmw))
        return chunks

    def _unit_h(self, u: _Unit, w: float) -> float:
        if not u.banded:
            return u.eff_rigid_h + u.eff_soft / w
        return self._band_solutions(u, w)[1]

    def _place_chunk_up(self, pl, xj, wj, y0, pos, full_w=None):
        y = y0
        for (i, soft, a, bw, bh) in pl:
            if soft:
                bw = wj if full_w is None else full_w
                bh = a / bw
            pos[i, 0] = xj
            pos[i, 1] = y
            pos[i, 2] = bw
            pos[i, 3] = bh
            y += bh
        return y

    def _place_band_up(self, band, x0, w, y, pos, sol=None):
        if len(band) == 1:
            return self._place_chunk_up(band[0][0], x0, w, y, pos, full_w=w)
        if sol is None:
            yy = y
            for ch in band:
                yy = self._place_chunk_up(ch[0], x0, w, yy, pos, full_w=w)
            return yy
        h_b, widths = sol
        xj = x0
        for ch, wj in zip(band, widths):
            self._place_chunk_up(ch[0], xj, wj, y, pos)
            xj += wj
        return y + h_b

    def _place_band_down(self, band, x0, w, ytop, pos, sol=None):
        if len(band) == 1:
            y = ytop
            for (i, soft, a, bw, bh) in reversed(band[0][0]):
                if soft:
                    bw = w
                    bh = a / w
                y -= bh
                pos[i, 0] = x0
                pos[i, 1] = y
                pos[i, 2] = bw
                pos[i, 3] = bh
            return y
        if sol is None:
            y = ytop
            for ch in reversed(band):
                for (i, soft, a, bw, bh) in reversed(ch[0]):
                    if soft:
                        bw = w
                        bh = a / w
                    y -= bh
                    pos[i, 0] = x0
                    pos[i, 1] = y
                    pos[i, 2] = bw
                    pos[i, 3] = bh
            return y
        h_b, widths = sol
        xj = x0
        for ch, wj in zip(band, widths):
            y = ytop
            for (i, soft, a, bw, bh) in reversed(ch[0]):
                if soft:
                    bw = wj
                    bh = a / wj
                y -= bh
                pos[i, 0] = xj
                pos[i, 1] = y
                pos[i, 2] = bw
                pos[i, 3] = bh
            xj += wj
        return ytop - h_b

    def _place_unit_up(self, u: _Unit, x0: float, w: float, y0: float, pos: np.ndarray) -> float:
        if not u.banded:
            y = y0
            for band in u.bands:
                y = self._place_band_up(band, x0, w, y, pos)
            return y
        plan = self._band_solutions(u, w)[2]
        y = y0
        for band_eff, sol in plan:
            y = self._place_band_up(band_eff, x0, w, y, pos, sol)
        return y

    def _place_unit_down(self, u: _Unit, x0: float, w: float, ytop: float, pos: np.ndarray) -> float:
        if not u.banded:
            y = ytop
            for band in reversed(u.bands):
                y = self._place_band_down(band, x0, w, y, pos)
            return y
        plan = self._band_solutions(u, w)[2]
        y = ytop
        for band_eff, sol in reversed(plan):
            y = self._place_band_down(band_eff, x0, w, y, pos, sol)
        return y

    def _place_split_stack_up(self, pl, seg_x, seg_w, y0, seg_end, segs, si, pos):
        """Place a flat pure-soft block stack (area/width-derived heights)
        bottom-up starting in free segment `si`, crossing into later free
        segments when the stack's cumulative height exceeds the current
        segment. Each block keeps its exact target area; only where the
        stack's vertical partition falls (i.e. which segment it lands in)
        changes, so total area placed is invariant. Returns (y_top, si) so
        the caller's segment index stays in sync with where the stack
        actually finished, avoiding stale-segment re-splits."""
        y = y0
        cur_x, cur_w = seg_x, seg_w
        cur_end = seg_end
        for (i, soft, a, bw, bh) in pl:
            bw = cur_w
            bh = a / bw
            while y + bh > cur_end + 1e-9 and cur_end != float('inf') \
                    and si + 1 < len(segs):
                # advance to the next free segment for this column
                si += 1
                nxt_s, cur_end, cur_x, cur_w = segs[si]
                y = max(y, nxt_s)
                bw = cur_w
                bh = a / bw
            pos[i, 0] = cur_x
            pos[i, 1] = y
            pos[i, 2] = bw
            pos[i, 3] = bh
            y += bh
        return y, si

    def _stack_column(self, ulist, x, w, pos):
        """Stack a column's units at width w; returns (placed, occupied, col_top)."""
        units = self.units
        has_locked = bool(self.locked_rects)
        occupied = [iv[:] for iv in self._obstacles_in(x, x + w)] if has_locked else []

        placed = []
        normal = []
        # anchored clusters: glue to their preplaced member if it is an
        # obstacle of this column
        for k in ulist:
            u = units[k]
            done = False
            if u.anchors and has_locked:
                uh = self._unit_h(u, w)
                for (ax, ay, aw, ah) in u.anchors:
                    if not (ax < x + w - 1e-9 and ax + aw > x + 1e-9):
                        continue
                    y0 = ay - uh
                    if y0 >= -1e-9 and self._interval_free(occupied, y0, ay):
                        yb = self._place_unit_down(u, x, w, ay, pos)
                        self._add_interval(occupied, yb, ay)
                        placed.append((k, yb, ay))
                        done = True
                        break
                    y1 = ay + ah
                    if self._interval_free(occupied, y1, y1 + uh):
                        yt = self._place_unit_up(u, x, w, y1, pos)
                        self._add_interval(occupied, y1, yt)
                        placed.append((k, y1, yt))
                        done = True
                        break
            if not done:
                normal.append(k)

        # enforce boundary-friendly stacking order regardless of the SA
        # permutation: bottom-tagged units first, top-tagged last
        if normal:
            bottoms = []
            mids = []
            tops = []
            for k in normal:
                u = units[k]
                if u.hasB and not u.hasT:
                    bottoms.append(k)
                elif u.hasT:
                    tops.append(k)
                else:
                    mids.append(k)
            normal = bottoms + mids + tops

        # bottom pre-pass: when the column bottom is blocked by an obstacle, a
        # bottom-tagged unit can still stand at y=0 in the strip beside it
        if occupied and normal:
            for k in list(normal):
                u = units[k]
                if not u.hasB or u.hasT:
                    break
                uh = self._unit_h(u, w)
                if self._interval_free(occupied, 0.0, uh):
                    break  # bottom is open; the normal flow handles it
                if u.force is not None:
                    break
                sw = None
                probe_h = uh
                for _ in range(3):
                    strip = self._band_strip(x, w, 0.0, probe_h, placed)
                    if strip is None:
                        sw = None
                        break
                    sw = strip
                    nh = self._unit_h(u, strip[1])
                    if abs(nh - probe_h) < 1e-6:
                        probe_h = nh
                        break
                    probe_h = nh
                if sw is None or u.max_rigid_w > sw[1] + 1e-9:
                    break
                final = self._band_strip(x, w, 0.0, probe_h, placed)
                if final is None or final[1] < sw[1] - 1e-6:
                    break
                yt = self._place_unit_up(u, sw[0], sw[1], 0.0, pos)
                self._add_interval(occupied, 0.0, yt)
                placed.append((k, 0.0, yt))
                normal.remove(k)
                break

        col_top = 0.0
        if not occupied:
            cy = 0.0
            i = 0
            nn = len(normal)
            while i < nn:
                k = normal[i]
                u = units[k]
                # pair two flat unconstrained soft singles side by side:
                # exact fill, near-square blocks, half the stack height
                if i + 1 < nn:
                    k2 = normal[i + 1]
                    u2 = units[k2]
                    if (len(u.blocks) == 1 and len(u2.blocks) == 1
                            and u.pairable and u2.pairable):
                        a1 = u.soft_area
                        a2 = u2.soft_area
                        hp = (a1 + a2) / w
                        if hp <= 0.85 * w:
                            w1 = a1 / hp
                            if 1.0 <= w1 <= w - 1.0:
                                i1 = u.blocks[0]
                                i2 = u2.blocks[0]
                                pos[i1, 0] = x
                                pos[i1, 1] = cy
                                pos[i1, 2] = w1
                                pos[i1, 3] = a1 / w1
                                pos[i2, 0] = x + w1
                                pos[i2, 1] = cy
                                pos[i2, 2] = w - w1
                                pos[i2, 3] = a2 / (w - w1)
                                top = cy + hp
                                placed.append((k, cy, top))
                                placed.append((k2, cy, top))
                                cy = top
                                i += 2
                                continue
                yt = self._place_unit_up(u, x, w, cy, pos)
                placed.append((k, cy, yt))
                cy = yt
                i += 1
            col_top = cy
        else:
            # Free segments, including "narrow" segments beside obstacles that
            # only partially cover the column (the strip next to a preplaced
            # block is usable at reduced width instead of being wasted).
            segs = []  # (start, end, xoff, width)
            cur = 0.0
            for s, e in occupied:
                if s > cur + 1e-9:
                    segs.append((cur, s, x, w))
                band_lo, band_hi = max(cur, s), e
                strip = self._band_strip(x, w, band_lo, band_hi, placed)
                if strip is not None:
                    segs.append((band_lo, band_hi, strip[0], strip[1]))
                cur = max(cur, e)
            segs.append((cur, float('inf'), x, w))

            si = 0
            cy = segs[0][0]
            pending = list(normal)
            while pending:
                _s, seg_end, seg_x, seg_w = segs[si]
                pick = None
                # first unit that fits the current gap (look ahead a few)
                narrow = seg_x > x + 1e-9 or seg_w < w - 1e-9
                for t in range(min(len(pending), 10)):
                    u = units[pending[t]]
                    if u.max_rigid_w > seg_w + 1e-9:
                        continue
                    # left/right-forced units must keep the column's edge x
                    if narrow and u.force is not None:
                        continue
                    if cy + self._unit_h(u, seg_w) <= seg_end + 1e-9:
                        pick = t
                        break
                if pick is None:
                    # a pure-soft unit (no rigid/fixed members, single flat
                    # stack) that overshoots every segment can still be split
                    # across the obstacle band that blocks it: fill the
                    # current free segment bottom-up, then resume the same
                    # stack at the top of the obstacle, preserving each
                    # block's exact area (only the vertical partition point
                    # of the stack moves).
                    split_pick = None
                    if seg_end != float('inf') and si + 1 < len(segs):
                        for t in range(min(len(pending), 10)):
                            u = units[pending[t]]
                            if u.eff_rigid_h > 0.0 or u.banded:
                                continue
                            if u.max_rigid_w > seg_w + 1e-9:
                                continue
                            if narrow and u.force is not None:
                                continue
                            if len(u.bands) == 1 and len(u.bands[0]) == 1 \
                                    and cy < seg_end - 1e-9:
                                split_pick = t
                                break
                    if split_pick is None:
                        si += 1
                        cy = segs[si][0]
                        continue
                    k = pending.pop(split_pick)
                    u = units[k]
                    pl = u.bands[0][0][0]
                    yt, si = self._place_split_stack_up(pl, seg_x, seg_w, cy, seg_end,
                                                         segs, si, pos)
                    placed.append((k, cy, yt))
                    cy = yt
                    continue
                k = pending.pop(pick)
                u = units[k]
                yt = self._place_unit_up(u, seg_x, seg_w, cy, pos)
                placed.append((k, cy, yt))
                cy = yt
            for (_k, _yb, yt) in placed:
                if yt > col_top:
                    col_top = yt
        return placed, occupied, col_top

    def _band_strip(self, x, w, lo, hi, placed_units):
        """Widest free x-strip inside column [x, x+w] over the y-band
        [lo, hi). Anchored units already placed span the full column width,
        so any overlap with them blocks the band entirely. Returns
        (xoff, width) or None when the strip is too narrow to be useful."""
        if hi <= lo + 1e-6:
            return None
        for (_k, yb, yt) in placed_units:
            if yb < hi - 1e-9 and yt > lo + 1e-9:
                return None
        left = x + w
        right = x
        for (ox, oy, ow, oh) in self.locked_rects:
            if ox < x + w - 1e-9 and ox + ow > x + 1e-9 and oy < hi - 1e-9 and oy + oh > lo + 1e-9:
                if ox < left:
                    left = ox
                if ox + ow > right:
                    right = ox + ow
        if right <= x:
            return None
        left_w = left - x
        right_w = (x + w) - right
        if right_w >= left_w:
            xoff, sw = right, right_w
        else:
            xoff, sw = x, left_w
        if sw < 0.30 * w or sw < 2.0:
            return None
        return (xoff, sw)

    @staticmethod
    def _interval_free(occ: List[List[float]], s: float, e: float) -> bool:
        for a, b in occ:
            if s < b - 1e-9 and a < e - 1e-9:
                return False
        return True

    @staticmethod
    def _add_interval(occ: List[List[float]], s: float, e: float):
        occ.append([s, e])
        occ.sort()
        merged = [occ[0][:]]
        for a, b in occ[1:]:
            if a <= merged[-1][1] + 1e-12:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        occ[:] = merged

    # ------------------------------------------------------------------
    def _solve_column(self, ulist, x, pos, w_force=None):
        """Compute one column's width and stack its units into `pos` at left
        offset `x`. Returns (w, placed, occupied, col_top). Pure function of
        (ulist, x, units-state, locked_rects); writes only pos[block] for the
        blocks in this column.

        M3 width re-optimization: when `w_force` is given the derived width is
        skipped and the column is stacked at exactly that width (clamped up to
        the widest rigid member so hard shapes still fit). The widen-retry loop
        is also skipped -- the width pass owns feasibility and rejects a width
        whose returned col_top exceeds the frame height."""
        H = self.H
        has_locked = bool(self.locked_rects)
        units = self.units
        soft_a = 0.0
        rigid_h = 0.0
        max_w = 0.0
        for k in ulist:
            u = units[k]
            soft_a += u.eff_soft
            rigid_h += u.eff_rigid_h
            if u.max_rigid_w > max_w:
                max_w = u.max_rigid_w
        if w_force is not None:
            w = max(w_force, max_w, 0.5)
            placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
            return w, placed, occupied, col_top
        avail = H - rigid_h
        if avail < 0.05 * H:
            avail = 0.05 * H
        w = soft_a / avail
        if w < max_w:
            w = max_w
        if w < 0.5:
            w = 0.5
        if has_locked:
            for _ in range(3):
                obs = self._obstacles_in(x, x + w)
                obs_h = 0.0
                for s, e in obs:
                    if e > 0.0 and s < H:
                        span = (e if e < H else H) - (s if s > 0.0 else 0.0)
                        strip = self._band_strip(x, w, s, e, ())
                        if strip is not None:
                            span *= 1.0 - strip[1] / w
                        obs_h += span
                avail = H - rigid_h - obs_h
                if avail < 0.05 * H:
                    avail = 0.05 * H
                w2 = soft_a / avail
                if w2 < max_w:
                    w2 = max_w
                if w2 < 0.5:
                    w2 = 0.5
                if abs(w2 - w) < 5e-3:
                    w = w2
                    break
                w = w2

        placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
        # widen and retry when fragmentation pushed content past the frame;
        # only the soft part shrinks with width, so solve for it exactly
        if soft_a > 0:
            tries = 0
            w_base = w
            prev_overhead = None
            while col_top > H * 1.0005 and tries < 3:
                overhead = col_top - soft_a / w
                # dead (non-soft) space already dominates the column: widening
                # only shrinks the soft part, it cannot recover fixed overhead.
                if overhead >= H * 0.90:
                    break
                # require overhead to shrink by >=1% per try; otherwise the
                # loop is diverging (e.g. widen exposes no more free space).
                if prev_overhead is not None and overhead > prev_overhead * 0.99:
                    break
                prev_overhead = overhead
                if overhead < H * 0.98:
                    w2 = soft_a / (H - overhead)
                else:
                    w2 = w * 1.25
                w = min(max(w2, w * 1.01), w_base * 1.6)
                placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
                tries += 1
        return w, placed, occupied, col_top

    def _layout(self, cols: List[List[int]]) -> Tuple[np.ndarray, float, float]:
        if self._fast_eval:
            return self._layout_fast(cols)
        n = self.n
        pos = np.zeros((n, 4))
        for i in range(n):
            if self.kind[i] == 2:
                pos[i] = (self.lx[i], self.ly[i], self.rw[i], self.rh[i])

        x = 0.0
        col_records = []  # (x0, w, [(unit, y_bot, y_top)], occupied)

        unit_col = self._unit_col
        col_spans = []
        for ci, ulist in enumerate(cols):
            if not ulist:
                col_records.append(None)
                col_spans.append(None)
                continue
            for k in ulist:
                unit_col[k] = ci
            w, placed, occupied, _col_top = self._solve_column(ulist, x, pos)
            col_records.append((x, w, placed, occupied))
            col_spans.append((x, x + w))
            x += w
        self._col_spans = col_spans
        return self._finish_layout(pos, col_records, x)

    def _layout_widths(self, cols, widths):
        """M3: layout that forces each column to widths[ci] (None = derive as
        usual). Always the from-scratch path -- it never touches the FAST_EVAL
        `_col_cache`, so no width-blind cache entry can be written or reused;
        the cache stays correct for the SA's own (derived-width) queries.
        Returns (pos, x_right, y_top, col_tops) where col_tops[ci] is the raw
        stacked height of column ci (before the top-lift post-pass) so the
        width pass can check per-column frame feasibility."""
        n = self.n
        pos = np.zeros((n, 4))
        for i in range(n):
            if self.kind[i] == 2:
                pos[i] = (self.lx[i], self.ly[i], self.rw[i], self.rh[i])
        x = 0.0
        col_records = []
        col_spans = []
        col_tops = [0.0] * len(cols)
        unit_col = self._unit_col
        for ci, ulist in enumerate(cols):
            if not ulist:
                col_records.append(None)
                col_spans.append(None)
                continue
            for k in ulist:
                unit_col[k] = ci
            wf = widths[ci] if ci < len(widths) else None
            w, placed, occupied, col_top = self._solve_column(ulist, x, pos, w_force=wf)
            col_records.append((x, w, placed, occupied))
            col_spans.append((x, x + w))
            col_tops[ci] = col_top
            x += w
        self._col_spans = col_spans
        pos, x_right, y_top = self._finish_layout(pos, col_records, x)
        return pos, x_right, y_top, col_tops

    def _finish_layout(self, pos, col_records, x_right):
        """Global post-passes shared by the slow and fast layout paths:
        compute the frame extent, lift top-tagged units flush to the top
        edge, and right-align right-tagged blocks. Operates on the same
        col_records shape ((x0, w, placed, occupied) or None) and gives
        bit-identical results regardless of how pos/col_records were built."""
        y_top = 0.0
        for rec in col_records:
            if rec is None:
                continue
            for (_k, _yb, yt) in rec[2]:
                y_top = max(y_top, yt)
        for (ox, oy, ow, oh) in self.locked_rects:
            x_right = max(x_right, ox + ow)
            y_top = max(y_top, oy + oh)

        # lift top-tagged top units flush to the global top edge
        for rec in col_records:
            if rec is None or not rec[2]:
                continue
            cx0, cw, placed, occupied = rec
            k, yb, yt = max(placed, key=lambda t: t[2])
            u = self.units[k]
            top_block = u.blocks[-1]
            if not (self.boundary[top_block] & 4):
                continue
            dy = y_top - yt
            if dy <= 1e-9:
                continue
            occ_wo = [iv for iv in occupied if not (abs(iv[0] - yb) < 1e-6 and abs(iv[1] - yt) < 1e-6)]
            # occupied intervals were merged; conservatively test only against
            # raw obstacles + other placed units above the current top
            blockers = self._obstacles_in(cx0, cx0 + cw)
            for (k2, yb2, yt2) in placed:
                if k2 != k:
                    blockers.append([yb2, yt2])
            if self._interval_free(blockers, yb + dy, yt + dy):
                for i in u.blocks:
                    pos[i, 1] += dy

        # right-align right-tagged blocks in the last non-empty column
        last = None
        for rec in reversed(col_records):
            if rec is not None and rec[2]:
                last = rec
                break
        if last is not None:
            for (k, _yb, _yt) in last[2]:
                u = self.units[k]
                for i in u.blocks:
                    if self.boundary[i] & 2:
                        nx = x_right - pos[i, 2]
                        if nx <= pos[i, 0] + 1e-9:
                            continue
                        # collision check against every other block (chunked
                        # units may have siblings to the right)
                        nx0, ny0 = nx, pos[i, 1]
                        nx1, ny1 = x_right, pos[i, 1] + pos[i, 3]
                        px0 = pos[:, 0]
                        py0 = pos[:, 1]
                        clash = ((px0 < nx1 - 1e-7)
                                 & (px0 + pos[:, 2] > nx0 + 1e-7)
                                 & (py0 < ny1 - 1e-7)
                                 & (py0 + pos[:, 3] > ny0 + 1e-7))
                        clash[i] = False
                        if not clash.any():
                            pos[i, 0] = nx

        return pos, x_right, y_top

    # ------------------------------------------------------------------
    # FAST_EVAL: per-column memoized layout. On a cache miss the column is
    # solved with the identical _solve_column body and its base geometry is
    # recorded relative to the column's left edge; on a hit the cached base
    # geometry is rigidly shifted to the current x. A column whose absolute
    # x-span intersects an obstacle strip depends on x, so its cache entry
    # records the x it was solved at and is recomputed when x changes.
    # ------------------------------------------------------------------
    def _col_touches_obstacle(self, x0: float, x1: float) -> bool:
        for (ox0, ox1) in self._obs_x:
            if ox0 < x1 - 1e-9 and ox1 > x0 + 1e-9:
                return True
        return False

    def _layout_fast(self, cols):
        n = self.n
        pos = np.zeros((n, 4))
        for i in range(n):
            if self.kind[i] == 2:
                pos[i] = (self.lx[i], self.ly[i], self.rw[i], self.rh[i])

        has_obs = bool(self._obs_x)
        cache = self._col_cache
        unit_col = self._unit_col
        col_records = []
        col_spans = []
        x = 0.0
        for ci, ulist in enumerate(cols):
            if not ulist:
                col_records.append(None)
                col_spans.append(None)
                continue
            for k in ulist:
                unit_col[k] = ci
            key = (tuple(ulist), tuple(self.units[k].version for k in ulist))
            ent = cache.get(key)
            # An obstacle-touching column depends on absolute x: only reuse a
            # cached entry when it was solved at (nearly) the same x AND the
            # obstacle-intersection status of the old/new span is unchanged.
            reuse = ent is not None
            if reuse and has_obs:
                ent_x = ent[0]
                ent_w = ent[1]
                old_touch = ent[4]
                new_touch = self._col_touches_obstacle(x, x + ent_w)
                if old_touch or new_touch:
                    reuse = abs(ent_x - x) <= 1e-9 and old_touch == new_touch
            if reuse:
                _ex, w, ids, rel, _touch, placed, occupied = ent
                # rigid x-shift + fixed y (base geometry `rel` is relative to
                # the column's left edge); recompose by a single vectorized
                # block copy plus the x offset (rel is never mutated).
                pos[ids] = rel
                pos[ids, 0] += x
            else:
                w, placed, occupied, _col_top = self._solve_column(ulist, x, pos)
                ids = np.array([i for k in ulist for i in self.units[k].blocks],
                               dtype=np.int64)
                rel = pos[ids].copy()
                rel[:, 0] -= x
                touch = self._col_touches_obstacle(x, x + w) if has_obs else False
                cache[key] = (x, w, ids, rel, touch, placed, occupied)
            col_records.append((x, w, placed, occupied))
            col_spans.append((x, x + w))
            x += w
        self._col_spans = col_spans
        return self._finish_layout(pos, col_records, x)

    # ------------------------------------------------------------------
    def _hpwl(self, pos: np.ndarray) -> float:
        cx = pos[:, 0] + pos[:, 2] * 0.5
        cy = pos[:, 1] + pos[:, 3] * 0.5
        total = 0.0
        if len(self.eI):
            total += float(np.sum(self.eW * (np.abs(cx[self.eI] - cx[self.eJ])
                                             + np.abs(cy[self.eI] - cy[self.eJ]))))
        if len(self.pB):
            total += float(np.sum(self.pW * (np.abs(self.pX - cx[self.pB])
                                             + np.abs(self.pY - cy[self.pB]))))
        return total

    @staticmethod
    def _count_components(adj: np.ndarray) -> int:
        """Number of connected components of a small symmetric adjacency
        matrix, computed by boolean transitive closure (repeated squaring to a
        fixpoint) + a vectorized representative count. Bit-exact-equivalent to
        the frontier-expansion BFS it replaces (verified exhaustively for
        m<=6 and over 200k random graphs up to m=9), but O(log m) matmuls
        instead of O(m) frontier steps, so it wins on the larger (chain-like)
        cluster groups. Groups here are <=9 nodes."""
        m = len(adj)
        if m <= 1:
            return m
        r = adj.copy()
        np.fill_diagonal(r, True)
        prev = -1
        cur = int(r.sum())
        while cur != prev:
            r = r | (r @ r)
            prev = cur
            cur = int(r.sum())
        # a node is its component's representative iff no earlier-indexed node
        # reaches it; components == number of representatives
        return int((~np.tril(r, -1).any(axis=1)).sum())

    def _violations(self, pos: np.ndarray) -> int:
        V = 0
        px0 = pos[:, 0]
        py0 = pos[:, 1]
        px1 = px0 + pos[:, 2]
        py1 = py0 + pos[:, 3]
        x_min = px0.min()
        y_min = py0.min()
        x_max = px1.max()
        y_max = py1.max()
        eps = 1e-6
        b = self._bnd_idx
        if len(b):
            codes = self._bnd_codes
            bad = ((codes & 1) != 0) & (np.abs(px0[b] - x_min) >= eps)
            bad |= ((codes & 2) != 0) & (np.abs(px1[b] - x_max) >= eps)
            bad |= ((codes & 4) != 0) & (np.abs(py1[b] - y_max) >= eps)
            bad |= ((codes & 8) != 0) & (np.abs(py0[b] - y_min) >= eps)
            V += int(bad.sum())

        for g in self._clu_arrays:
            gx0 = px0[g]
            gy0 = py0[g]
            gx1 = px1[g]
            gy1 = py1[g]
            ox = np.minimum(gx1[:, None], gx1[None, :]) - np.maximum(gx0[:, None], gx0[None, :])
            oy = np.minimum(gy1[:, None], gy1[None, :]) - np.maximum(gy0[:, None], gy0[None, :])
            adj = ((ox > TOUCH_TOL) & (oy >= -TOUCH_TOL)) | ((oy > TOUCH_TOL) & (ox >= -TOUCH_TOL))
            V += self._count_components(adj) - 1

        for g in self._mib_arrays:
            shapes = {(round(float(pos[i, 2]), 4), round(float(pos[i, 3]), 4)) for i in g}
            V += len(shapes) - 1
        return V

    def _cost(self, pos: np.ndarray, x_right: float, y_top: float):
        hp = self._hpwl(pos)
        x_min = float(pos[:, 0].min())
        y_min = float(pos[:, 1].min())
        area = (x_right - x_min) * (y_top - y_min)
        V = self._violations(pos)
        c = (1.0 + 0.5 * ((hp / self.hp_ref - 1.0) + max(0.0, area / self.area_ref - 1.0))) \
            * math.exp(2.0 * self.v_weight * V / self.n_soft_den)
        return c, hp, area, V

    def _evaluate(self, cols):
        pos, x_right, y_top = self._layout(cols)
        c, hp, area, V = self._cost(pos, x_right, y_top)
        return c, pos

    # ------------------------------------------------------------------
    def _random_move(self, cols: List[List[int]]):
        """Mutate cols in place; return an undo callable, or None."""
        rng = self.rng
        C = len(cols)
        nonempty = [i for i in range(C) if cols[i]]
        if not nonempty:
            return None
        r = rng.random()
        cum = self._move_cum
        if r < cum[0]:
            self._last_move_type = 0
            sc = rng.choice(nonempty)
            si = rng.randrange(len(cols[sc]))
            k = cols[sc][si]
            u = self.units[k]
            if u.force == 'L':
                tc = 0
            elif u.force == 'R':
                tc = C - 1
            elif u.anchors and rng.random() < 0.6:
                # usually pull anchored clusters back over their preplaced
                # member, but sometimes let them escape (trading one grouping
                # violation can satisfy several boundary tags)
                ax = u.anchors[0][0] + u.anchors[0][2] * 0.5
                tc = self._col_of_x(ax, C)
            else:
                tc = None
                rr = rng.random()
                # connectivity-guided relocation: pull a unit toward the
                # column of a randomly sampled neighbor / pin
                if rr < 0.35 and len(self.eI):
                    e = rng.randrange(len(self.eI))
                    a, b = int(self.eI[e]), int(self.eJ[e])
                    if self.blk_unit[a] != k and self.blk_unit[b] == k:
                        a, b = b, a
                    if self.blk_unit[a] == k:
                        ku = self.blk_unit[b]
                        if ku >= 0:
                            tc = self._unit_col[ku]
                        else:
                            tc = self._col_of_x(self.lx[b] + self.rw[b] * 0.5, C)
                elif rr < 0.50 and len(self.pB):
                    e = rng.randrange(len(self.pB))
                    if self.blk_unit[int(self.pB[e])] == k:
                        tc = self._col_of_x(float(self.pX[e]), C)
                if tc is None:
                    tc = rng.randrange(C)
                # spread bottom/top-tagged units across columns (only the
                # first / last unit of a column can touch that edge)
                if u.hasB or u.hasT:
                    for _try in range(3):
                        clash = any((self.units[k2].hasB if u.hasB else self.units[k2].hasT)
                                    for k2 in cols[tc] if k2 != k)
                        if not clash:
                            break
                        tc = rng.randrange(C)
            if tc == sc and len(cols[sc]) == 1:
                return None
            cols[sc].pop(si)
            ti = rng.randint(0, len(cols[tc]))
            cols[tc].insert(ti, k)

            def undo():
                cols[tc].pop(ti)
                cols[sc].insert(si, k)
            return undo
        elif r < cum[1]:
            self._last_move_type = 1
            if len(nonempty) < 2:
                return None
            c1, c2 = rng.sample(nonempty, 2)
            i1 = rng.randrange(len(cols[c1]))
            i2 = rng.randrange(len(cols[c2]))
            k1, k2 = cols[c1][i1], cols[c2][i2]
            u1, u2 = self.units[k1], self.units[k2]
            if u1.force == 'L' and c2 != 0:
                return None
            if u1.force == 'R' and c2 != C - 1:
                return None
            if u2.force == 'L' and c1 != 0:
                return None
            if u2.force == 'R' and c1 != C - 1:
                return None
            cols[c1][i1], cols[c2][i2] = k2, k1

            def undo():
                cols[c1][i1], cols[c2][i2] = k1, k2
            return undo
        elif r < cum[2]:
            self._last_move_type = 2
            cands = [i for i in nonempty if len(cols[i]) >= 2]
            if not cands:
                return None
            c1 = rng.choice(cands)
            si = rng.randrange(len(cols[c1]))
            ti = rng.randrange(len(cols[c1]))
            if si == ti:
                return None
            k = cols[c1].pop(si)
            cols[c1].insert(ti, k)

            def undo():
                cols[c1].pop(ti)
                cols[c1].insert(si, k)
            return undo
        else:
            self._last_move_type = 3
            # internal reorder of a multi-part unit
            multi = [k for k, u in enumerate(self.units) if len(u.subgroups) >= 2
                     or any(len(sg) >= 2 for sg in u.subgroups)]
            if not multi:
                return None
            k = rng.choice(multi)
            u = self.units[k]
            if len(u.subgroups) >= 2 and rng.random() < 0.5:
                a = rng.randrange(len(u.subgroups) - 1)
                u.subgroups[a], u.subgroups[a + 1] = u.subgroups[a + 1], u.subgroups[a]
                self._refresh_unit(u)

                def undo():
                    u.subgroups[a], u.subgroups[a + 1] = u.subgroups[a + 1], u.subgroups[a]
                    self._refresh_unit(u)
                return undo
            sgs = [g for g in u.subgroups if len(g) >= 2]
            if not sgs:
                return None
            sg = rng.choice(sgs)
            a = rng.randrange(len(sg) - 1)
            sg[a], sg[a + 1] = sg[a + 1], sg[a]
            self._refresh_unit(u)

            def undo():
                sg[a], sg[a + 1] = sg[a + 1], sg[a]
                self._refresh_unit(u)
            return undo

    def _snapshot(self, cols):
        return ([list(c) for c in cols],
                {k: [sg[:] for sg in self.units[k].subgroups]
                 for k in range(len(self.units)) if len(self.units[k].blocks) > 1})

    def _restore(self, snap):
        cols = [list(c) for c in snap[0]]
        for k, sgs in snap[1].items():
            u = self.units[k]
            u.subgroups = [sg[:] for sg in sgs]
            self._refresh_unit(u)
        return cols

    # ------------------------------------------------------------------
    # M1c: adaptive-move bookkeeping. Sliding window over recent proposals;
    # every window we reweight the move mix by (acceptance x mean accepted
    # improvement) with a per-family probability floor.
    def _adaptive_reset(self):
        self._am_prop = [0, 0, 0, 0]     # proposals per family this window
        self._am_acc = [0, 0, 0, 0]      # accepted per family this window
        self._am_impr = [0.0, 0.0, 0.0, 0.0]  # summed positive improvement
        self._am_since = 0               # moves since last reweight
        self._am_window = 200            # ~200 total proposals per update

    def _adaptive_reweight(self):
        floor = 0.05
        n_fam = len(self._move_base)
        scores = []
        for t in range(n_fam):
            prop = self._am_prop[t]
            if prop <= 0:
                scores.append(0.0)
                continue
            acc_rate = self._am_acc[t] / prop
            mean_impr = (self._am_impr[t] / self._am_acc[t]) if self._am_acc[t] else 0.0
            scores.append(acc_rate * mean_impr)
        s = sum(scores)
        if s <= 0.0:
            # no signal this window -- keep the current mix, just reset counters
            self._adaptive_reset()
            return
        free = 1.0 - floor * n_fam
        weights = [floor + free * (sc / s) for sc in scores]
        self._move_base = weights
        self._move_cum = _cumulative(weights)
        self._adaptive_reset()

    def _anneal(self, cols, deadline, cur_cost, t0=0.02, t1=0.0008, recalibrate=False):
        rng = self.rng
        best_cost = cur_cost
        best = self._snapshot(cols)
        best_hp = None
        start = time.time()
        span = max(deadline - start, 1e-6)
        recal_at = [start + 0.25 * span, start + 0.55 * span] if recalibrate else []
        if self._adaptive_moves:
            self._adaptive_reset()
        while True:
            now = time.time()
            if now >= deadline:
                break
            if recal_at and now >= recal_at[0]:
                recal_at.pop(0)
                # tighten the HPWL normalizer toward the (unknown) baseline so
                # the annealer keeps real pressure on wirelength
                pos, xr, yt = self._layout(cols)
                hp_now = self._hpwl(pos)
                self.hp_ref = max(hp_now * 0.75, 1e-6)
                cur_cost, _, _, _ = self._cost(pos, xr, yt)
                bcols = self._restore(best)
                bpos, bxr, byt = self._layout(bcols)
                best_cost, _, _, _ = self._cost(bpos, bxr, byt)
                if best_cost < cur_cost:
                    cols = bcols
                    cur_cost = best_cost
            frac = min((now - start) / span, 1.0)
            T = t0 * (t1 / t0) ** frac
            for _ in range(24):
                # M1c: PARSAC-style constraint-fixing move. Rare, targeted,
                # accepted unconditionally; only fires while soft violations
                # remain. Hard legality is by construction so this is safe.
                if self._cfix and rng.random() < self._cfix_p:
                    new_cols = self._cfix_move(cols, cur_cost)
                    if new_cols is not None:
                        cols[:] = new_cols
                        cur_cost, _ = self._evaluate(cols)
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best = self._snapshot(cols)
                    continue
                undo = self._random_move(cols)
                if undo is None:
                    continue
                mt = self._last_move_type
                if self._adaptive_moves:
                    self._am_prop[mt] += 1
                    self._am_since += 1
                new_cost, _pos = self._evaluate(cols)
                if new_cost <= cur_cost or rng.random() < math.exp((cur_cost - new_cost) / T):
                    if self._adaptive_moves:
                        self._am_acc[mt] += 1
                        if new_cost < cur_cost:
                            self._am_impr[mt] += (cur_cost - new_cost)
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = self._snapshot(cols)
                else:
                    undo()
                if self._adaptive_moves and self._am_since >= self._am_window:
                    self._adaptive_reweight()
        return best, best_cost

    # ------------------------------------------------------------------
    def _cfix_move(self, cols, cur_cost):
        """PARSAC-style targeted boundary repair for the SA inner loop.

        Fires only when soft violations remain (V>0). Finds one unit whose
        bottom/top-tagged block misses its edge and relocates that unit to the
        head/tail of a column that currently lacks a same-tag unit (so it can
        actually reach the wall), preferring the column whose current x is
        nearest the unit's connectivity target. Returns the mutated column
        lists (a fresh copy) or None when there is nothing to fix. The caller
        accepts the result unconditionally; only `cols` (unit ordering) is
        mutated, never subgroup geometry, so hard legality is untouched."""
        pos, _xr, _yt = self._layout(cols)
        if self._violations(pos) <= 0:
            return None
        py0 = pos[:, 1]
        py1 = py0 + pos[:, 3]
        y_min = float(py0.min())
        y_max = float(py1.max())
        bad = []  # (unit, tag) tag: 'B' bottom-miss, 'T' top-miss
        for i, code in zip(self._bnd_idx, self._bnd_codes):
            k = self.blk_unit[int(i)]
            if k < 0:
                continue
            if (int(code) & 8) and abs(float(py0[i]) - y_min) >= 1e-6:
                bad.append((k, 'B'))
            elif (int(code) & 4) and abs(float(py1[i]) - y_max) >= 1e-6:
                bad.append((k, 'T'))
        if not bad:
            return None
        k, tag = self.rng.choice(bad)
        u = self.units[k]
        sc = self._unit_col[k]
        C = len(cols)
        if sc >= C:
            return None
        try:
            si = cols[sc].index(k)
        except ValueError:
            return None
        # candidate target columns that currently have no same-tag unit
        cands = []
        for tc in range(C):
            if tc == sc:
                continue
            if u.force == 'L' and tc != 0:
                continue
            if u.force == 'R' and tc != C - 1:
                continue
            has_same = any(
                (self.units[k2].hasB if tag == 'B' else self.units[k2].hasT)
                for k2 in cols[tc])
            if not has_same:
                cands.append(tc)
        if not cands:
            return None
        # nearest column to the unit's seed x (cheap connectivity proxy)
        tc = min(cands, key=lambda c: abs(c - sc))
        new_cols = [list(c) for c in cols]
        new_cols[sc].pop(si)
        ti = 0 if tag == 'B' else len(new_cols[tc])
        new_cols[tc].insert(ti, k)
        return new_cols

    # ------------------------------------------------------------------
    def locked_only(self) -> bool:
        return not self.units

    def locked_positions(self) -> List[Rect]:
        return [(self.lx[i], self.ly[i], self.rw[i], self.rh[i]) for i in range(self.n)]

    def prepare(self) -> float:
        avg_area = self.total_area / max(self.n, 1)
        target_w = max(math.sqrt(avg_area), 2.0)
        self.C0 = max(2, min(18, int(round(self.W_est / target_w))))
        cols = self._init_columns(self.C0)
        c0, _ = self._evaluate_bootstrap(cols)
        self._cols = cols
        self._cost0 = c0
        self._best_probe = None
        return c0

    def probe(self, t_each: float) -> float:
        cands = sorted({max(2, self.C0 - 1), self.C0, min(18, self.C0 + 1)})
        results = []
        for C in cands:
            pc = self._init_columns(C)
            pcost, _ = self._evaluate(pc)
            snap, bcost = self._anneal(pc, time.time() + t_each, pcost,
                                       t0=0.06, t1=0.01)
            results.append((bcost, snap))
        results.sort(key=lambda t: t[0])
        self._best_probe = results[0]
        return results[0][0]

    # ------------------------------------------------------------------
    # M3 racing: short round-1 anneal that returns a cheap picklable snapshot
    # of the column state plus raw metrics, and a round-2 warm-started finish.
    def race_round1(self, C, variant, t_budget):
        """Bootstrap columns at count C (init variant 0 or 1), run a short
        warm anneal for t_budget seconds, and return (snapshot, hp, area, V)
        where snapshot is plain lists/ints (picklable). Does not depend on
        `prepare()` having chosen C0."""
        C = max(2, min(18, int(C)))
        self.C0 = C
        cols = self._init_columns(C)
        if variant == 1:
            # alternate init: reverse each column's unit order (still legal,
            # ordering-only) so the two variants explore different basins
            for c in cols:
                c.reverse()
        c0, _ = self._evaluate_bootstrap(cols)
        deadline = time.time() + max(t_budget, 0.05)
        snap, _bc = self._anneal(cols, deadline, c0, t0=0.05, t1=0.008)
        cols_b = self._restore(snap)
        pos, xr, yt = self._layout(cols_b)
        hp = self._hpwl(pos)
        V = self._violations(pos)
        area = (xr - float(pos[:, 0].min())) * (yt - float(pos[:, 1].min()))
        return snap, hp, area, V

    def race_round2(self, snapshot, C, deadline):
        """Warm-start `finish` from a round-1 snapshot for the rest of the
        budget. Restores the snapshot into `_cols`, disables the probe path,
        and runs a single anneal + polish through `finish`."""
        self.C0 = max(2, min(18, int(C)))
        cols = self._restore(snapshot)
        c0, _ = self._evaluate_bootstrap(cols)
        self._cols = cols
        self._cost0 = c0
        self._best_probe = None
        return self.finish(deadline, max_runs=1)

    def finish(self, deadline: float, max_runs: int = 2) -> List[Rect]:
        n = self.n
        if self._best_probe is not None:
            start_snap = self._best_probe[1]
        else:
            start_snap = self._snapshot(self._cols)
        polish_t = min(0.28 * max(deadline - time.time(), 0.0), 6.5)
        t_end = deadline - polish_t
        now = time.time()
        total = max(t_end - now, 0.1)
        runs = 2 if (total > 6.0 and max_runs >= 2) else 1
        snaps = []
        for r in range(runs):
            cols_r = self._restore(start_snap)
            c_r, _ = self._evaluate(cols_r)
            snap, _bc = self._anneal(cols_r, now + total * (r + 1) / runs, c_r,
                                     recalibrate=True)
            snaps.append(snap)
        # pick the better run under one common (final) normalizer
        best_cols = None
        best_c = None
        for snap in snaps:
            cols_c = self._restore(snap)
            c_c, _ = self._evaluate(cols_c)
            if best_c is None or c_c < best_c:
                best_c = c_c
                best_cols = [list(c) for c in cols_c]
                best_snap = snap
        cols = self._restore(best_snap)
        # M3: carve a small width-opt slice out of the polish budget so the
        # total per-case wall clock is unchanged (only when the flag is on).
        width_reserve = 0.0
        if self._width_opt:
            width_reserve = min(0.20 * max(deadline - time.time(), 0.0), 2.0)
        cols = self._greedy_polish(cols, best_c, deadline - width_reserve)
        pos, x_right, y_top = self._layout(cols)
        hp = self._hpwl(pos)
        V = self._violations(pos)
        area = (x_right - float(pos[:, 0].min())) * (y_top - float(pos[:, 1].min()))
        if self._width_opt and width_reserve > 0.0:
            wres = self._width_optimize(cols, deadline)
            if wres is not None:
                _widths, pos, x_right, y_top, _c, hp = wres
                V = self._violations(pos)
                area = (x_right - float(pos[:, 0].min())) * (y_top - float(pos[:, 1].min()))
        self.final_metrics = (hp, area, V)
        out = [(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]), float(pos[i, 3]))
               for i in range(n)]
        return _ensure_no_overlap(out, [self.kind[i] == 2 for i in range(n)])

    # ------------------------------------------------------------------
    def _width_optimize(self, cols, deadline):
        """M3: coordinate-descent the per-column widths for HPWL after the SA
        fixes the column assignment.

        Returns (widths, pos, x_right, y_top, cost, hp) for the best width
        vector found, or None when nothing beat the SA's own derived widths.
        Objective is the full numpy HPWL of the re-stacked layout; acceptance
        is additionally gated on the full cost (violations + area) not
        regressing, so the pass can never trade a soft violation or a bbox
        blowup for wirelength. Every trial goes through `_layout_widths`, which
        bypasses the FAST_EVAL cache, so no width-blind entry can leak into it.
        Bounded to 2 sweeps and its own deadline."""
        C = len(cols)
        # baseline derived widths + cost
        pos0, xr0, yt0, _tops0 = self._layout_widths(cols, [None] * C)
        base_c, base_hp, _a0, _v0 = self._cost(pos0, xr0, yt0)
        w0 = [(sp[1] - sp[0]) if sp is not None else None
              for sp in self._col_spans[:C]]
        widths = list(w0)
        best_pos, best_xr, best_yt = pos0, xr0, yt0
        best_c, best_hp = base_c, base_hp
        H = self.H
        gr = 0.6180339887498949  # golden ratio conjugate
        lo_mult, hi_mult = 0.7, 1.4

        def trial(ci, w_try):
            saved = widths[ci]
            widths[ci] = w_try
            pos, xr, yt, tops = self._layout_widths(cols, widths)
            widths[ci] = saved
            # feasibility: the re-widthed column must still fit the frame
            if tops[ci] > H * 1.0005:
                return None
            c, hp, area, V = self._cost(pos, xr, yt)
            return c, hp, pos, xr, yt

        improved_any = False
        for _sweep in range(2):
            sweep_improved = False
            for ci in range(C):
                if time.time() >= deadline:
                    break
                if widths[ci] is None:  # empty column
                    continue
                w_base = widths[ci]
                lo = lo_mult * w_base
                hi = hi_mult * w_base
                # golden-section minimize HPWL on [lo, hi]
                a, b = lo, hi
                x1 = b - gr * (b - a)
                x2 = a + gr * (b - a)
                r1 = trial(ci, x1)
                r2 = trial(ci, x2)
                f1 = r1[1] if r1 is not None else float('inf')
                f2 = r2[1] if r2 is not None else float('inf')
                for _it in range(6):
                    if f1 <= f2:
                        b, x2, f2, r2 = x2, x1, f1, r1
                        x1 = b - gr * (b - a)
                        r1 = trial(ci, x1)
                        f1 = r1[1] if r1 is not None else float('inf')
                    else:
                        a, x1, f1, r1 = x1, x2, f2, r2
                        x2 = a + gr * (b - a)
                        r2 = trial(ci, x2)
                        f2 = r2[1] if r2 is not None else float('inf')
                cand = r1 if f1 <= f2 else r2
                if cand is None:
                    continue
                c_c, hp_c, pos_c, xr_c, yt_c = cand
                # accept only a strict full-cost improvement for this column
                if c_c < best_c - 1e-9:
                    widths[ci] = (x1 if f1 <= f2 else x2)
                    best_c, best_hp = c_c, hp_c
                    best_pos, best_xr, best_yt = pos_c, xr_c, yt_c
                    improved_any = True
                    sweep_improved = True
            if not sweep_improved or time.time() >= deadline:
                break
        if self._width_opt_log:
            try:
                import json
                with open(self._width_opt_log, "a") as fh:
                    fh.write(json.dumps({
                        "stage": "width_opt", "n": self.n, "C": C,
                        "pre_hpwl": base_hp, "post_hpwl": best_hp,
                        "pre_cost": base_c, "post_cost": best_c,
                        "improved": improved_any,
                    }) + "\n")
            except Exception:
                pass
        if not improved_any:
            return None
        return widths, best_pos, best_xr, best_yt, best_c, best_hp

    def run(self) -> List[Rect]:
        if self.locked_only():
            return self.locked_positions()
        self.prepare()
        budget = self.deadline - time.time()
        if budget > 2.5:
            self.probe(min(0.12 * budget, 2.0) / 3)
        return self.finish(self.deadline)

    def _spread_tagged(self, cols, cur_cost, deadline):
        """Move surplus bottom/top-tagged units (only the first / last unit
        of a column can touch that edge) into columns that lack one."""
        for tag in ('B', 'T'):
            for sc in range(len(cols)):
                if time.time() >= deadline:
                    return cur_cost
                tagged = [k for k in cols[sc]
                          if (self.units[k].hasB if tag == 'B' else self.units[k].hasT)]
                for k in tagged[1:]:
                    u = self.units[k]
                    si = cols[sc].index(k)
                    best = None
                    for tc in range(len(cols)):
                        if tc == sc:
                            continue
                        if u.force == 'L' and tc != 0:
                            continue
                        if u.force == 'R' and tc != len(cols) - 1:
                            continue
                        if any((self.units[k2].hasB if tag == 'B' else self.units[k2].hasT)
                               for k2 in cols[tc]):
                            continue
                        cols[sc].pop(si)
                        ti = 0 if tag == 'B' else len(cols[tc])
                        cols[tc].insert(ti, k)
                        c, _ = self._evaluate(cols)
                        if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                            best = (c, tc, ti)
                        cols[tc].pop(ti)
                        cols[sc].insert(si, k)
                    if best is not None:
                        c, tc, ti = best
                        cols[sc].pop(si)
                        cols[tc].insert(ti, k)
                        cur_cost = c
        return cur_cost

    def _repair_boundary(self, cols, cur_cost, deadline):
        """Relocate units whose bottom/top-tagged blocks miss their edge."""
        pos, _xr, _yt = self._layout(cols)
        py0 = pos[:, 1]
        py1 = py0 + pos[:, 3]
        y_min = py0.min()
        y_max = py1.max()
        bad_units = []
        for i, code in zip(self._bnd_idx, self._bnd_codes):
            k = self.blk_unit[int(i)]
            if k < 0:
                continue
            if (code & 8) and abs(py0[i] - y_min) >= 1e-6:
                bad_units.append((k, 0))
            elif (code & 4) and abs(py1[i] - y_max) >= 1e-6:
                bad_units.append((k, -1))
        seen = set()
        for k, where in bad_units:
            if k in seen or time.time() >= deadline:
                continue
            seen.add(k)
            u = self.units[k]
            sc = self._unit_col[k]
            if sc >= len(cols):
                continue
            try:
                si = cols[sc].index(k)
            except ValueError:
                continue
            best = None
            for tc in range(len(cols)):
                if u.force == 'L' and tc != 0:
                    continue
                if u.force == 'R' and tc != len(cols) - 1:
                    continue
                cols[sc].pop(si)
                ti = 0 if where == 0 else len(cols[tc])
                cols[tc].insert(ti, k)
                c, _ = self._evaluate(cols)
                cols[tc].pop(ti)
                cols[sc].insert(si, k)
                if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                    best = (c, tc, ti)
            if best is not None:
                c, tc, ti = best
                cols[sc].pop(si)
                cols[tc].insert(ti, k)
                cur_cost = c
        return cur_cost

    def _order_pass(self, cols, cur_cost, deadline):
        """Reorder each column by the connectivity-weighted target y of its
        units, and try swapping adjacent columns; keep strict improvements."""
        pos, _xr, _yt = self._layout(cols)
        n = self.n
        cy = pos[:, 1] + pos[:, 3] * 0.5
        num = np.zeros(n)
        den = np.zeros(n)
        if len(self.eI):
            np.add.at(num, self.eI, self.eW * cy[self.eJ])
            np.add.at(den, self.eI, self.eW)
            np.add.at(num, self.eJ, self.eW * cy[self.eI])
            np.add.at(den, self.eJ, self.eW)
        if len(self.pB):
            np.add.at(num, self.pB, self.pW * self.pY)
            np.add.at(den, self.pB, self.pW)
        pull = np.where(den > 1e-12, num / np.maximum(den, 1e-12), cy)
        for ci in range(len(cols)):
            if len(cols[ci]) < 3 or time.time() >= deadline:
                continue
            def tkey(k):
                u = self.units[k]
                s = 0.0
                a = 0.0
                for i in u.blocks:
                    ai = self.areas[i]
                    s += ai * pull[i]
                    a += ai
                return s / max(a, 1e-9)
            new_order = sorted(cols[ci], key=tkey)
            if new_order != cols[ci]:
                old = cols[ci][:]
                cols[ci][:] = new_order
                c, _ = self._evaluate(cols)
                if c < cur_cost - 1e-9:
                    cur_cost = c
                else:
                    cols[ci][:] = old
        for ci in range(len(cols) - 1):
            if time.time() >= deadline:
                break
            # never displace edge-forced units from their edge column
            if any(self.units[k].force for k in cols[ci]) or \
                    any(self.units[k].force for k in cols[ci + 1]):
                continue
            cols[ci], cols[ci + 1] = cols[ci + 1], cols[ci]
            c, _ = self._evaluate(cols)
            if c < cur_cost - 1e-9:
                cur_cost = c
            else:
                cols[ci], cols[ci + 1] = cols[ci + 1], cols[ci]
        return cur_cost

    def _reduce_overflow(self, cols, cur_cost, deadline):
        """When columns rise past the (possibly pinned) frame height, move
        their topmost unit somewhere cheaper."""
        units = self.units
        for _round in range(6):
            if time.time() >= deadline:
                break
            pos, _xr, yt = self._layout(cols)
            if yt <= self.H * 1.0001:
                break
            moved = False
            tops = []
            for ci, ulist in enumerate(cols):
                if not ulist:
                    continue
                top = max(pos[i, 1] + pos[i, 3] for k in ulist for i in units[k].blocks)
                if top > self.H * 1.0001:
                    tops.append((top, ci))
            tops.sort(reverse=True)
            for _t, sc in tops[:2]:
                ktop = max(cols[sc],
                           key=lambda k: max(pos[i, 1] + pos[i, 3] for i in units[k].blocks))
                si = cols[sc].index(ktop)
                u = units[ktop]
                best = None
                for tc in range(len(cols)):
                    if tc == sc:
                        continue
                    if u.force == 'L' and tc != 0:
                        continue
                    if u.force == 'R' and tc != len(cols) - 1:
                        continue
                    for ti in (0, len(cols[tc])):
                        cols[sc].pop(si)
                        ti2 = min(ti, len(cols[tc]))
                        cols[tc].insert(ti2, ktop)
                        c, _ = self._evaluate(cols)
                        cols[tc].pop(ti2)
                        cols[sc].insert(si, ktop)
                        if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                            best = (c, tc, ti2)
                if best is not None:
                    c, tc, ti2 = best
                    cols[sc].pop(si)
                    cols[tc].insert(ti2, ktop)
                    cur_cost = c
                    moved = True
            if not moved:
                break
        return cur_cost

    def _greedy_polish(self, cols, cur_cost, deadline):
        """Final best-improvement sweeps: for every unit try a handful of
        (column, position) relocations and keep strict improvements."""
        cur_cost, _ = self._evaluate(cols)
        cur_cost = self._reduce_overflow(cols, cur_cost, deadline)
        cur_cost = self._repair_boundary(cols, cur_cost, deadline)
        cur_cost = self._spread_tagged(cols, cur_cost, deadline)
        for _ in range(3):
            before = cur_cost
            cur_cost = self._order_pass(cols, cur_cost, deadline)
            if cur_cost > before - 1e-9 or time.time() >= deadline:
                break
        rng = self.rng
        order = list(range(len(self.units)))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(order)
            for k in order:
                if time.time() >= deadline:
                    break
                u = self.units[k]
                sc = self._unit_col[k]
                if sc >= len(cols):
                    continue
                try:
                    si = cols[sc].index(k)
                except ValueError:
                    continue
                best = None
                for tc in range(len(cols)):
                    if u.force == 'L' and tc != 0:
                        continue
                    if u.force == 'R' and tc != len(cols) - 1:
                        continue
                    ncand = len(cols[tc]) - (1 if tc == sc else 0)
                    for ti in {0, ncand // 2, ncand}:
                        if tc == sc and ti == si:
                            continue
                        cols[sc].pop(si)
                        ti2 = min(ti, len(cols[tc]))
                        cols[tc].insert(ti2, k)
                        c, _ = self._evaluate(cols)
                        cols[tc].pop(ti2)
                        cols[sc].insert(si, k)
                        if c < cur_cost - 1e-9 and (best is None or c < best[0]):
                            best = (c, tc, ti2)
                if best is not None:
                    c, tc, ti2 = best
                    cols[sc].pop(si)
                    cols[tc].insert(ti2, k)
                    cur_cost = c
                    improved = True
        return cols

    def _evaluate_bootstrap(self, cols):
        """First evaluation; establishes the HPWL normalizer."""
        pos, x_right, y_top = self._layout(cols)
        self.hp_ref = max(self._hpwl(pos), 1e-6)
        c, hp, area, V = self._cost(pos, x_right, y_top)
        return c, pos


# =============================================================================
# Safety net
# =============================================================================
def _ensure_no_overlap(rects: List[Rect], locked: List[bool]) -> List[Rect]:
    """Verify and, if needed, nudge apart residual overlaps (position-only
    moves, so hard shape/area guarantees are preserved). No-op in the normal
    path — the column construction is overlap-free."""
    n = len(rects)
    out = [list(r) for r in rects]
    for _ in range(50):
        changed = False
        for i in range(n):
            x1, y1, w1, h1 = out[i]
            for j in range(i + 1, n):
                x2, y2, w2, h2 = out[j]
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if ox <= 1e-6 or oy <= 1e-6:
                    continue
                move = i if locked[j] else j
                if locked[move]:
                    move = j if move == i else i
                    if locked[move]:
                        continue
                mx, my, mw, mh = out[move]
                if ox <= oy:
                    out[move] = [mx + ox + 1e-6, my, mw, mh] if move == j else [mx - ox - 1e-6, my, mw, mh]
                else:
                    out[move] = [mx, my + oy + 1e-6, mw, mh] if move == j else [mx, my - oy - 1e-6, mw, mh]
                x1, y1, w1, h1 = out[i]
                changed = True
        if not changed:
            break
    return [tuple(r) for r in out]


# =============================================================================
# Parallel restart pool
# =============================================================================
_POOL = None
_POOL_SIZE = 0
_POOL_READY = False


def _worker_init():
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _worker_ping(_i):
    time.sleep(0.2)
    return True


def init_worker_pool(n_workers: int = 8, warmup_timeout: float = 180.0):
    """Create the restart pool once and wait until every worker has finished
    importing (call from optimizer __init__ so the spawn/import cost is not
    charged to any test case)."""
    global _POOL, _POOL_SIZE, _POOL_READY
    if _POOL is not None or n_workers < 2:
        return
    try:
        import multiprocessing as mp
        # fork: no re-import of __main__, no per-worker torch import cost.
        # Workers never touch CUDA, so forking a CUDA-initialized parent is
        # safe (same pattern as torch DataLoader workers).
        ctx = mp.get_context('fork')
        _POOL = ctx.Pool(n_workers, initializer=_worker_init)
        _POOL_SIZE = n_workers
        r = _POOL.map_async(_worker_ping, range(n_workers * 3), chunksize=1)
        r.get(timeout=warmup_timeout)
        _POOL_READY = True
    except Exception:
        _shutdown_pool()


def _shutdown_pool():
    global _POOL, _POOL_SIZE, _POOL_READY
    try:
        if _POOL is not None:
            _POOL.terminate()
    except Exception:
        pass
    _POOL = None
    _POOL_SIZE = 0
    _POOL_READY = False


def _worker_solve(args):
    """One independent (orientation, column count, seed) restart."""
    try:
        (rects, areas_np, cons_np, tpos_np, b2b_np, p2b_np, pins_np,
         orient, c_force, seed, deadline, v_weight) = args
        at = torch.from_numpy(areas_np)
        cons = torch.from_numpy(cons_np) if cons_np is not None else None
        tpos = torch.from_numpy(tpos_np) if tpos_np is not None else None
        b2b = torch.from_numpy(b2b_np) if b2b_np is not None else None
        p2b = torch.from_numpy(p2b_np) if p2b_np is not None else None
        pins = torch.from_numpy(pins_np) if pins_np is not None else None
        rs = rects
        if orient == 'T':
            rs, cons, tpos, pins = _transpose_inputs(rects, cons, tpos, pins)
        opt = _ColumnOptimizer(rs, at, cons, tpos, b2b, p2b, pins, deadline,
                               seed=seed, v_weight=v_weight)
        if opt.locked_only():
            out = opt.locked_positions()
            return (out, 0.0, 0.0, 0)
        opt.prepare()
        if c_force is not None:
            c_use = max(2, min(18, c_force))
            if c_use != opt.C0:
                opt.C0 = c_use
                cols = opt._init_columns(c_use)
                c0, _ = opt._evaluate(cols)
                opt._cols = cols
                opt._cost0 = c0
        out = opt.finish(deadline, max_runs=1)
        hp, area, V = opt.final_metrics
        if orient == 'T':
            out = [(y, x, h, w) for (x, y, w, h) in out]
        return (out, hp, area, V)
    except Exception:
        return None


def _build_worker_opt(args, deadline, seed, v_weight):
    """Rebuild a `_ColumnOptimizer` from a numpy payload inside a worker.
    Shared by the racing round-1/round-2 workers. Returns (opt, orient)."""
    (rects, areas_np, cons_np, tpos_np, b2b_np, p2b_np, pins_np, orient) = args
    at = torch.from_numpy(areas_np)
    cons = torch.from_numpy(cons_np) if cons_np is not None else None
    tpos = torch.from_numpy(tpos_np) if tpos_np is not None else None
    b2b = torch.from_numpy(b2b_np) if b2b_np is not None else None
    p2b = torch.from_numpy(p2b_np) if p2b_np is not None else None
    pins = torch.from_numpy(pins_np) if pins_np is not None else None
    rs = rects
    if orient == 'T':
        rs, cons, tpos, pins = _transpose_inputs(rects, cons, tpos, pins)
    opt = _ColumnOptimizer(rs, at, cons, tpos, b2b, p2b, pins, deadline,
                           seed=seed, v_weight=v_weight)
    return opt, orient


def _worker_race_r1(args):
    """M3 racing round 1: a short warm anneal returning a cheap picklable
    snapshot + raw metrics + the config index. `args` = (payload_core, cfg)
    where cfg = (orient, C, variant, seed, v_weight, cfg_id, t_budget)."""
    try:
        payload_core, cfg = args
        orient, C, variant, seed, v_weight, cfg_id, t_budget = cfg
        opt, _o = _build_worker_opt(payload_core + (orient,), time.time() + t_budget + 1.0,
                                    seed, v_weight)
        if opt.locked_only():
            return (cfg_id, None, 0.0, 0.0, 0)
        snap, hp, area, V = opt.race_round1(C, variant, t_budget)
        # snap is ([list[list[int]]], {int: [list[int]]}) -- all builtins
        return (cfg_id, snap, float(hp), float(area), int(V))
    except Exception:
        return None


def _worker_race_r2(args):
    """M3 racing round 2: warm-start `finish` from a round-1 snapshot for the
    rest of the budget. `args` = (payload_core, cfg, snapshot, C, deadline)."""
    try:
        payload_core, cfg, snapshot, C, deadline = args
        orient, _C0, _variant, seed, v_weight, _cfg_id, _t = cfg
        opt, orient = _build_worker_opt(payload_core + (orient,), deadline, seed, v_weight)
        if opt.locked_only():
            out = opt.locked_positions()
            return (out, 0.0, 0.0, 0)
        out = opt.race_round2(snapshot, C, deadline)
        hp, area, V = opt.final_metrics
        if orient == 'T':
            out = [(y, x, h, w) for (x, y, w, h) in out]
        return (out, hp, area, V)
    except Exception:
        return None


def _race_configs(opt1, seed, allow_t, n_slots):
    """Build the round-1 config list: column counts C0-3..C0+3 (clamped) x
    orientations (T only when allow_t) x v_weight {1.0, 2.5} x 2 init variants,
    truncated to n_slots (target 18-24). Returns list of
    (orient, C, variant, seed, v_weight, cfg_id, t_budget=None)."""
    avg_area = opt1.total_area / max(opt1.n, 1)
    C0 = max(2, min(18, int(round(opt1.W_est / max(math.sqrt(avg_area), 2.0)))))
    col_counts = sorted({max(2, min(18, C0 + d)) for d in (-3, -2, -1, 0, 1, 2, 3)})
    orients = ['N', 'T'] if allow_t else ['N']
    v_weights = [1.0, 2.5]
    variants = [0, 1]
    cfgs = []
    cfg_id = 0
    # interleave so the truncation keeps a diverse spread (not all one C)
    for variant in variants:
        for vw in v_weights:
            for orient in orients:
                for C in col_counts:
                    cfgs.append((orient, C, variant, seed + 7 * cfg_id + 3,
                                 vw, cfg_id, None))
                    cfg_id += 1
    # spread by striding so a prefix cut still samples all axes
    stride = max(1, len(cfgs) // max(n_slots, 1))
    picked = cfgs[::stride][:n_slots]
    if len(picked) < min(n_slots, len(cfgs)):
        seen = {c[5] for c in picked}
        for c in cfgs:
            if c[5] not in seen:
                picked.append(c)
                seen.add(c[5])
            if len(picked) >= n_slots:
                break
    # guarantee at least one v_weight=2.5 config is present
    if not any(abs(c[4] - 2.5) < 1e-9 for c in picked) and \
            any(abs(c[4] - 2.5) < 1e-9 for c in cfgs):
        vw25 = next(c for c in cfgs if abs(c[4] - 2.5) < 1e-9)
        if picked:
            picked[-1] = vw25
        else:
            picked.append(vw25)
    return picked


def _parallel_solve_racing(opt1, rects, area_targets, constraints,
                           target_positions, b2b, p2b, pins, deadline, seed):
    """M3 two-round successive-halving racing (FLOORSET_SA_RACING).

    Round 1: 18-24 diverse configs each at ~25% of the worker budget, returning
    a picklable snapshot + raw metrics. Round 2: the top few (warm-started from
    their snapshots) run the remaining budget; at least one v_weight=2.5 config
    is force-promoted (slow-convergence protection). Both rounds preserve the
    straggler `_shutdown_pool` failure path and `worker_deadline = deadline -
    0.30`. Per-worker round costs are logged (JSONL) when FLOORSET_SA_RACING_LOG
    is set so the scheduler can run the selection-quality sign test."""
    hard_n = sum(1 for u in opt1.units if u.hasB) + sum(1 for u in opt1.units if u.hasT)
    hard_t = sum(1 for u in opt1.units if u.force == 'L') + \
        sum(1 for u in opt1.units if u.force == 'R')
    allow_t = hard_t <= hard_n + 2

    n_slots = min(24, max(18, _POOL_SIZE * 3))
    configs = _race_configs(opt1, seed, allow_t, n_slots)
    area_ref = opt1.area_ref
    n_soft = opt1.n_soft_den

    def np_of(t):
        return None if t is None else t.detach().cpu().numpy()

    payload_core = (list(rects), np_of(area_targets), np_of(constraints),
                    np_of(target_positions), np_of(b2b), np_of(p2b), np_of(pins))

    worker_deadline = deadline - 0.30
    total_budget = max(worker_deadline - time.time(), 0.2)
    # Budget accounting: each of the len(configs) round-1 configs runs on the
    # shared pool, so a worker handles ceil(n_configs / n_workers) of them
    # SEQUENTIALLY. Round 1 should consume ~25% of the total wall clock, so a
    # single config's budget is 25% of the total divided by that per-worker
    # config count; round 2 (the survivors) then gets the remaining ~75%.
    n_workers = max(_POOL_SIZE, 1)
    cfg_per_worker = max(1, (len(configs) + n_workers - 1) // n_workers)
    r1_wall = 0.25 * total_budget
    r1_budget = max(r1_wall / cfg_per_worker, 0.05)   # per-config anneal time
    # the pool needs r1_budget x cfg_per_worker wall to drain round 1
    r1_deadline = time.time() + r1_budget * cfg_per_worker + 1.0

    log_path = os.environ.get("FLOORSET_SA_RACING_LOG", "")

    def _log(rows):
        if not log_path:
            return
        try:
            import json
            with open(log_path, "a") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        except Exception:
            pass

    # ------- round 1 -------
    r1_payloads = [(payload_core, (c[0], c[1], c[2], c[3], c[4], c[5], r1_budget))
                   for c in configs]
    res1 = _POOL.map_async(_worker_race_r1, r1_payloads)
    try:
        outs1 = res1.get(timeout=max(r1_deadline - time.time(), 0.1) + 2.5)
    except Exception:
        _shutdown_pool()
        raise
    outs1 = [o for o in outs1 if o is not None]
    if not outs1:
        raise RuntimeError("all racing round-1 workers failed")

    cfg_by_id = {c[5]: c for c in configs}
    # round-1 tuple layout: (cfg_id, snap, hp, area, V) -> indices 0..4
    usable1 = [o for o in outs1 if o[1] is not None]
    if not usable1:
        raise RuntimeError("racing round-1 produced no usable snapshots")
    hp_ref1 = max(min(o[2] for o in usable1), 1e-9)

    def score1(o):
        _cid, _snap, hp, area, V = o
        return (1.0 + 0.5 * ((hp - hp_ref1) / hp_ref1
                             + max(0.0, area / area_ref - 1.0))) \
            * math.exp(2.0 * V / n_soft)

    ranked = sorted(usable1, key=score1)

    _log([{"stage": "race_r1", "cfg_id": o[0], "config": cfg_by_id[o[0]][:5],
           "hp": o[2], "area": o[3], "V": o[4], "score": score1(o)}
          for o in usable1])

    # ------- pick top 4-6, force in a v_weight=2.5 config -------
    n_top = min(len(ranked), max(4, min(6, _POOL_SIZE)))
    top = ranked[:n_top]
    top_ids = {o[0] for o in top}
    if not any(abs(cfg_by_id[o[0]][4] - 2.5) < 1e-9 for o in top):
        vw25 = next((o for o in ranked if abs(cfg_by_id[o[0]][4] - 2.5) < 1e-9), None)
        if vw25 is not None and vw25[0] not in top_ids:
            top[-1] = vw25  # slow-convergence protection
            top_ids = {o[0] for o in top}

    # ------- round 2: warm-start the survivors -------
    r2_payloads = []
    for o in top:
        cid, snap, _hp, _area, _V = o
        cfg = cfg_by_id[cid]
        r2_payloads.append((payload_core, cfg, snap, cfg[1], worker_deadline))
    res2 = _POOL.map_async(_worker_race_r2, r2_payloads)
    try:
        outs2 = res2.get(timeout=max(deadline - time.time(), 0.1) + 2.5)
    except Exception:
        _shutdown_pool()
        raise
    outs2 = [o for o in outs2 if o is not None]
    if not outs2:
        raise RuntimeError("all racing round-2 workers failed")

    hp_ref2 = max(min(o[1] for o in outs2), 1e-9)

    def score2(o):
        _out, hp, area, V = o
        return (1.0 + 0.5 * ((hp - hp_ref2) / hp_ref2
                             + max(0.0, area / area_ref - 1.0))) \
            * math.exp(2.0 * V / n_soft)

    _log([{"stage": "race_r2", "cfg_id": top[i][0],
           "config": cfg_by_id[top[i][0]][:5],
           "hp": outs2[i][1], "area": outs2[i][2], "V": outs2[i][3],
           "score": score2(outs2[i])}
          for i in range(min(len(top), len(outs2)))])

    best = min(outs2, key=score2)
    return best[0]


def _parallel_solve(opt1, rects, area_targets, constraints, target_positions,
                    b2b, p2b, pins, deadline, seed):
    if os.environ.get("FLOORSET_SA_RACING", "0") == "1":
        return _parallel_solve_racing(opt1, rects, area_targets, constraints,
                                      target_positions, b2b, p2b, pins,
                                      deadline, seed)
    avg_area = opt1.total_area / max(opt1.n, 1)
    C0 = max(2, min(18, int(round(opt1.W_est / max(math.sqrt(avg_area), 2.0)))))
    hard_n = sum(1 for u in opt1.units if u.hasB) + sum(1 for u in opt1.units if u.hasT)
    hard_t = sum(1 for u in opt1.units if u.force == 'L') + \
        sum(1 for u in opt1.units if u.force == 'R')
    allow_t = hard_t <= hard_n + 2

    # portfolio: mixed column counts / seeds, plus two violation-averse
    # searchers (their layouts compete under the true cost at selection)
    configs = [('N', None, seed + 11, 1.0), ('N', C0 - 1, seed + 22, 1.0),
               ('N', C0 + 1, seed + 33, 1.0), ('N', None, seed + 44, 2.5),
               ('N', C0 - 2, seed + 55, 1.0), ('N', C0 + 2, seed + 66, 1.0),
               ('N', None, seed + 77, 2.5), ('N', C0 - 1, seed + 88, 1.0),
               ('N', None, seed + 99, 1.0), ('N', C0 + 1, seed + 110, 1.0),
               ('N', C0, seed + 121, 2.5), ('N', C0 - 1, seed + 132, 1.0)]
    if allow_t:
        configs[5] = ('T', None, seed + 66, 1.0)
        configs[7] = ('T', None, seed + 88, 2.5)
        configs[9] = ('T', C0 + 1, seed + 110, 1.0)
    configs = configs[:max(2, _POOL_SIZE)]

    def np_of(t):
        return None if t is None else t.detach().cpu().numpy()

    worker_deadline = deadline - 0.30
    payloads = [(list(rects), np_of(area_targets), np_of(constraints),
                 np_of(target_positions), np_of(b2b), np_of(p2b), np_of(pins),
                 orient, cf, sd, worker_deadline, vw)
                for (orient, cf, sd, vw) in configs]
    res = _POOL.map_async(_worker_solve, payloads)
    try:
        outs = res.get(timeout=max(deadline - time.time(), 0.1) + 2.5)
    except Exception:
        # workers overran their deadline — the pool now has stragglers that
        # would delay every later case, so drop it entirely
        _shutdown_pool()
        raise
    outs = [o for o in outs if o is not None]
    if not outs:
        raise RuntimeError("all parallel workers failed")

    area_ref = opt1.area_ref
    n_soft = opt1.n_soft_den
    hp_ref = max(min(o[1] for o in outs), 1e-9)

    def score(o):
        _out, hp, area, V = o
        return (1.0 + 0.5 * ((hp - hp_ref) / hp_ref
                             + max(0.0, area / area_ref - 1.0))) \
            * math.exp(2.0 * V / n_soft)

    best = min(outs, key=score)
    return best[0]


# =============================================================================
# Main entry point
# =============================================================================
_BND_TRANSPOSE = {0: 0}
for _c in range(1, 16):
    _t = 0
    if _c & 1:   # left -> bottom
        _t |= 8
    if _c & 2:   # right -> top
        _t |= 4
    if _c & 4:   # top -> right
        _t |= 2
    if _c & 8:   # bottom -> left
        _t |= 1
    _BND_TRANSPOSE[_c] = _t


def _transpose_inputs(rects, constraints, target_positions, pins):
    rects_t = [(y, x, h, w) for (x, y, w, h) in rects]
    cons_t = None
    if constraints is not None:
        cons_t = constraints.clone()
        if cons_t.dim() == 2 and cons_t.shape[1] > 4:
            for i in range(cons_t.shape[0]):
                cons_t[i, 4] = float(_BND_TRANSPOSE.get(int(round(float(cons_t[i, 4]))) & 15, 0))
    tpos_t = None
    if target_positions is not None:
        tpos_t = target_positions.clone()
        tpos_t[:, 0] = target_positions[:, 1]
        tpos_t[:, 1] = target_positions[:, 0]
        tpos_t[:, 2] = target_positions[:, 3]
        tpos_t[:, 3] = target_positions[:, 2]
    pins_t = None
    if pins is not None and pins.dim() == 2 and pins.shape[1] >= 2:
        pins_t = pins.clone()
        pins_t[:, 0] = pins[:, 1]
        pins_t[:, 1] = pins[:, 0]
    return rects_t, cons_t, tpos_t, pins_t


def legalize_rectangles(
    rects: List[Rect],
    area_targets: torch.Tensor,
    constraints: Optional[torch.Tensor] = None,
    target_positions: Optional[torch.Tensor] = None,
    b2b_connectivity: Optional[torch.Tensor] = None,
    p2b_connectivity: Optional[torch.Tensor] = None,
    pins_pos: Optional[torch.Tensor] = None,
    deadline: Optional[float] = None,
    seed: int = 0,
) -> List[Rect]:
    n = len(rects)
    if n == 0:
        return []
    if deadline is None:
        deadline = time.time() + 3.0
    opt1 = _ColumnOptimizer(
        rects, area_targets, constraints, target_positions,
        b2b_connectivity, p2b_connectivity, pins_pos, deadline, seed=seed,
    )
    if opt1.locked_only():
        return opt1.locked_positions()
    budget = deadline - time.time()

    if _POOL is not None and _POOL_READY and budget > 3.0:
        try:
            return _parallel_solve(opt1, rects, area_targets, constraints,
                                   target_positions, b2b_connectivity,
                                   p2b_connectivity, pins_pos, deadline, seed)
        except Exception:
            pass  # fall back to the sequential path below

    if budget < 4.0:
        return opt1.run()

    # count units that need a column bottom/top of their own per orientation:
    # L/R tags are structurally cheap (edge columns have unlimited capacity),
    # B/T tags are scarce (one per column). Transposing swaps the two.
    hard_n = sum(1 for u in opt1.units if u.hasB) + sum(1 for u in opt1.units if u.hasT)
    hard_t = sum(1 for u in opt1.units if u.force == 'L') + \
        sum(1 for u in opt1.units if u.force == 'R')
    if hard_t > hard_n + 2:
        return opt1.run()

    # try both orientations (HPWL / area / violations are orientation
    # invariant, so probe costs are directly comparable)
    rects_t, cons_t, tpos_t, pins_t = _transpose_inputs(
        rects, constraints, target_positions, pins_pos)
    opt2 = _ColumnOptimizer(
        rects_t, area_targets, cons_t, tpos_t,
        b2b_connectivity, p2b_connectivity, pins_t, deadline, seed=seed + 1,
    )
    opt1.prepare()
    opt2.prepare()
    opt2.hp_ref = opt1.hp_ref
    t_probe = min(0.20 * budget, 4.2) / 6.0
    p1 = opt1.probe(t_probe)
    p2 = opt2.probe(t_probe)
    # probes are noisy; the normal orientation handles the common
    # many-left-tags pattern structurally better, so require a clear win
    if p2 < p1 * 0.96:
        out_t = opt2.finish(deadline)
        return [(y, x, h, w) for (x, y, w, h) in out_t]
    return opt1.finish(deadline)
