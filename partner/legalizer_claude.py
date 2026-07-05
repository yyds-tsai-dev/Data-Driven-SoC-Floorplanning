#!/usr/bin/env python3
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
                 "bands", "eff_soft", "eff_rigid_h")

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
    ):
        self.n = len(rects)
        self.rects = rects
        self.deadline = deadline if deadline is not None else time.time() + 3.0
        self.rng = random.Random(seed)

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
        return u.rigid_h + u.soft_area / w

    def _partition_subgroup(self, sg: List[int]) -> List[List[int]]:
        """Split a cluster with several bottom/top-tagged members into
        side-by-side chunks so each tag can touch its edge. Connectivity is
        preserved: chunks stand shoulder to shoulder inside the unit."""
        if len(sg) < 2:
            return [sg]
        bnd = self.boundary
        B = [i for i in sg if bnd[i] & 8]
        T = [i for i in sg if (bnd[i] & 4) and not (bnd[i] & 8)]
        m = min(max(len(B), len(T)), len(sg), 4)
        if m < 2:
            return [sg]
        if any(self.kind[i] == 0 and self.mib[i] > 0 for i in sg):
            return [sg]

        def blk_area(i):
            return self.areas[i] if self.kind[i] == 0 else self.rw[i] * self.rh[i]

        chunks: List[List[int]] = [[] for _ in range(m)]
        areas = [0.0] * m
        for c, i in enumerate(B[:m]):
            chunks[c].append(i)
            areas[c] += blk_area(i)
        used = set(B[:m]) | set(T[:m])
        mid = [i for i in sg if i not in used]
        for i in sorted(mid, key=lambda j: -blk_area(j)):
            c = min(range(m), key=lambda cc: areas[cc])
            chunks[c].append(i)
            areas[c] += blk_area(i)
        for c, i in enumerate(T[:m]):
            chunks[c].append(i)
            areas[c] += blk_area(i)
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

    def _solve_band(self, band, w: float):
        """Find the band height h so the chunk widths sum to w. Mixed chunks
        satisfy w_j = soft_j / (h - rigid_h_j); pure-rigid chunks have fixed
        width. Returns (h, widths) or None if infeasible at this width."""
        pure_w = 0.0
        lo = 0.0
        for (_pl, sa, rh, rmw) in band:
            if sa <= 0.0:
                pure_w += rmw
            if rh > lo:
                lo = rh
        avail = w - pure_w
        if avail <= 1e-6:
            return None
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

    def _unit_h(self, u: _Unit, w: float) -> float:
        h = 0.0
        for band in u.bands:
            if len(band) == 1:
                ch = band[0]
                h += ch[2] + (ch[1] / w if ch[1] > 0.0 else 0.0)
            else:
                sol = self._solve_band(band, w)
                if sol is None:
                    for ch in band:
                        h += ch[2] + (ch[1] / w if ch[1] > 0.0 else 0.0)
                else:
                    h += sol[0]
        return h

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

    def _place_band_up(self, band, x0, w, y, pos):
        if len(band) == 1:
            return self._place_chunk_up(band[0][0], x0, w, y, pos, full_w=w)
        sol = self._solve_band(band, w)
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

    def _place_band_down(self, band, x0, w, ytop, pos):
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
        sol = self._solve_band(band, w)
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
        y = y0
        for band in u.bands:
            y = self._place_band_up(band, x0, w, y, pos)
        return y

    def _place_unit_down(self, u: _Unit, x0: float, w: float, ytop: float, pos: np.ndarray) -> float:
        y = ytop
        for band in reversed(u.bands):
            y = self._place_band_down(band, x0, w, y, pos)
        return y

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
                uh = u.rigid_h + u.soft_area / w
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

        col_top = 0.0
        if not occupied:
            cy = 0.0
            for k in normal:
                u = units[k]
                yt = self._place_unit_up(u, x, w, cy, pos)
                placed.append((k, cy, yt))
                cy = yt
            col_top = cy
        else:
            segs = []
            cur = 0.0
            for s, e in occupied:
                if s > cur + 1e-9:
                    segs.append((cur, s))
                cur = max(cur, e)
            segs.append((cur, float('inf')))

            si = 0
            cy = segs[0][0]
            pending = list(normal)
            while pending:
                seg_end = segs[si][1]
                pick = None
                # first unit that fits the current gap (look ahead a few)
                for t in range(min(len(pending), 8)):
                    u = units[pending[t]]
                    if cy + u.rigid_h + u.soft_area / w <= seg_end + 1e-9:
                        pick = t
                        break
                if pick is None:
                    si += 1
                    cy = segs[si][0]
                    continue
                k = pending.pop(pick)
                u = units[k]
                yt = self._place_unit_up(u, x, w, cy, pos)
                placed.append((k, cy, yt))
                cy = yt
            for (_k, _yb, yt) in placed:
                if yt > col_top:
                    col_top = yt
        return placed, occupied, col_top

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
    def _layout(self, cols: List[List[int]]) -> Tuple[np.ndarray, float, float]:
        n = self.n
        pos = np.zeros((n, 4))
        for i in range(n):
            if self.kind[i] == 2:
                pos[i] = (self.lx[i], self.ly[i], self.rw[i], self.rh[i])

        H = self.H
        x = 0.0
        col_records = []  # (x0, w, [(unit, y_bot, y_top)], occupied)

        units = self.units
        boundary = self.boundary
        has_locked = bool(self.locked_rects)
        unit_col = self._unit_col
        col_spans = []
        for ci, ulist in enumerate(cols):
            if not ulist:
                col_records.append(None)
                col_spans.append(None)
                continue
            for k in ulist:
                unit_col[k] = ci
            soft_a = 0.0
            rigid_h = 0.0
            max_w = 0.0
            for k in ulist:
                u = units[k]
                soft_a += u.soft_area
                rigid_h += u.rigid_h
                if u.max_rigid_w > max_w:
                    max_w = u.max_rigid_w
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
                            obs_h += (e if e < H else H) - (s if s > 0.0 else 0.0)
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
                while col_top > H * 1.003 and tries < 2:
                    overhead = col_top - soft_a / w
                    if overhead < H * 0.98:
                        w2 = soft_a / (H - overhead)
                    else:
                        w2 = w * 1.25
                    w = min(max(w2, w * 1.01), w * 2.0)
                    placed, occupied, col_top = self._stack_column(ulist, x, w, pos)
                    tries += 1

            col_records.append((x, w, placed, occupied))
            col_spans.append((x, x + w))
            x += w
        self._col_spans = col_spans

        x_right = x
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
                        ok = True
                        for (ox, oy, ow, oh) in self.locked_rects:
                            if (nx < ox + ow - 1e-7 and ox < nx + pos[i, 2] - 1e-7
                                    and pos[i, 1] < oy + oh - 1e-7 and oy < pos[i, 1] + pos[i, 3] - 1e-7):
                                ok = False
                                break
                        if ok:
                            pos[i, 0] = nx

        return pos, x_right, y_top

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
            m = len(g)
            seen = np.zeros(m, dtype=bool)
            comps = 0
            for s in range(m):
                if seen[s]:
                    continue
                comps += 1
                frontier = np.zeros(m, dtype=bool)
                frontier[s] = True
                seen[s] = True
                while frontier.any():
                    nxt = adj[frontier].any(axis=0) & ~seen
                    seen |= nxt
                    frontier = nxt
            V += comps - 1

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
            * math.exp(2.0 * V / self.n_soft_den)
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
        if r < 0.55:
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
        elif r < 0.80:
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
        elif r < 0.92:
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

    def _anneal(self, cols, deadline, cur_cost, t0=0.02, t1=0.0008, recalibrate=False):
        rng = self.rng
        best_cost = cur_cost
        best = self._snapshot(cols)
        best_hp = None
        start = time.time()
        span = max(deadline - start, 1e-6)
        recal_at = [start + 0.25 * span, start + 0.55 * span] if recalibrate else []
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
                undo = self._random_move(cols)
                if undo is None:
                    continue
                new_cost, _pos = self._evaluate(cols)
                if new_cost <= cur_cost or rng.random() < math.exp((cur_cost - new_cost) / T):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = self._snapshot(cols)
                else:
                    undo()
        return best, best_cost

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

    def finish(self, deadline: float) -> List[Rect]:
        n = self.n
        if self._best_probe is not None:
            cur_cost = self._best_probe[0]
            cols = self._restore(self._best_probe[1])
        else:
            cur_cost = self._cost0
            cols = self._cols
        polish_t = min(0.25 * max(deadline - time.time(), 0.0), 5.0)
        snap, bc = self._anneal(cols, deadline - polish_t, cur_cost, recalibrate=True)
        cols = self._restore(snap)
        cols = self._greedy_polish(cols, bc, deadline)
        pos, _xr, _yt = self._layout(cols)
        out = [(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]), float(pos[i, 3]))
               for i in range(n)]
        return _ensure_no_overlap(out, [self.kind[i] == 2 for i in range(n)])

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

    def _greedy_polish(self, cols, cur_cost, deadline):
        """Final best-improvement sweeps: for every unit try a handful of
        (column, position) relocations and keep strict improvements."""
        cur_cost, _ = self._evaluate(cols)
        cur_cost = self._spread_tagged(cols, cur_cost, deadline)
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
    if budget < 4.0:
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
    t_probe = min(0.18 * budget, 3.6) / 6.0
    p1 = opt1.probe(t_probe)
    p2 = opt2.probe(t_probe)
    if p2 < p1:
        out_t = opt2.finish(deadline)
        return [(y, x, h, w) for (x, y, w, h) in out_t]
    return opt1.finish(deadline)
