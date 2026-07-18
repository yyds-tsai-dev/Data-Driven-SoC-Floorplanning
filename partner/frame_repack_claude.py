"""Frame-constrained repack: legalize a prediction INTO a pinned frame.

The refine ladder's philosophy is "if it does not fit, grow the frame,
and finally drop the tag pins".  That is correct when the frame is free,
and guarantees boundary violations when preplaced tagged blocks pin the
frame exactly (e.g. case-89-class instances: frame utilization 96%, the
pipeline's arrangements reach ~92-93%, the layout overflows by a column
and every wall-coded block on that side breaks).

This module adds the missing branch: a MaxRects free-rectangle packer
that places every movable block INSIDE the pinned frame, in the order and
near the positions the model prediction suggests, reshaping soft blocks
to fit the available free rectangles (exact area, aspect-capped).
Preplaced blocks are obstacles; boundary-coded blocks are pre-seated by
_perimeter_pack.  No case-specific logic anywhere: everything derives
from constraint codes, areas and the prediction.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]
MAX_ASPECT = 10.0


def _split_free(fr: Rect, used: Rect) -> List[Rect]:
    """MaxRects split: free rect minus used rect -> up to 4 free rects."""
    fx, fy, fw, fh = fr
    ux, uy, uw, uh = used
    if ux >= fx + fw or ux + uw <= fx or uy >= fy + fh or uy + uh <= fy:
        return [fr]
    out = []
    if uy > fy:                                   # below
        out.append((fx, fy, fw, uy - fy))
    if uy + uh < fy + fh:                         # above
        out.append((fx, uy + uh, fw, fy + fh - (uy + uh)))
    if ux > fx:                                   # left
        out.append((fx, fy, ux - fx, fh))
    if ux + uw < fx + fw:                         # right
        out.append((ux + uw, fy, fx + fw - (ux + uw), fh))
    return [r for r in out if r[2] > 1e-9 and r[3] > 1e-9]


def _prune(free: List[Rect]) -> List[Rect]:
    """Drop free rects fully contained in another (MaxRects invariant)."""
    out = []
    for i, a in enumerate(free):
        ax, ay, aw, ah = a
        contained = False
        for j, b in enumerate(free):
            if i == j:
                continue
            bx, by, bw, bh = b
            if (bx <= ax + 1e-12 and by <= ay + 1e-12
                    and bx + bw >= ax + aw - 1e-12
                    and by + bh >= ay + ah - 1e-12
                    and (bw * bh > aw * ah or j < i)):
                contained = True
                break
        if not contained:
            out.append(a)
    return out


def frame_repack(opt, pred: np.ndarray, pack: Dict[int, Rect],
                 W: float, H: float,
                 deadline: float,
                 group_mode: str = "scatter") -> Optional[np.ndarray]:
    """Pack every movable non-ring block inside [0,W]x[0,H].

    opt: _ColumnOptimizer for kind/areas/cluster info.  pred: [n,4] model
    prediction (position guidance).  pack: ring placements from
    _perimeter_pack (already flush on the walls).  Returns [n,4] or None.
    """
    n = opt.n
    kind = opt.kind
    areas = opt.areas
    out = np.zeros((n, 4), dtype=np.float64)
    free: List[Rect] = [(0.0, 0.0, W, H)]

    def occupy(r: Rect):
        nonlocal free
        nxt: List[Rect] = []
        for fr in free:
            nxt.extend(_split_free(fr, r))
        free = _prune(nxt)

    # obstacles: locked blocks and the perimeter ring
    for i in range(n):
        if kind[i] == 2:
            out[i] = (opt.lx[i], opt.ly[i], opt.rw[i], opt.rh[i])
            occupy(tuple(out[i]))
    for i, r in pack.items():
        out[i] = r
        occupy(tuple(r))

    placed = set(i for i in range(n) if kind[i] == 2) | set(pack.keys())
    todo = [i for i in range(n) if i not in placed]
    # cluster groups first (members consecutively, so they land adjacent),
    # then the rest; inside each group / the remainder follow the model's
    # row-major order
    def band_key(i):
        # band = the prediction's row (global structure); inside a band
        # big blocks go first (FFD: skeletons before fillers, otherwise
        # the tail of the packing dies on fragmentation)
        cy = float(pred[i, 1] + 0.5 * pred[i, 3])
        return (round(cy / max(H * 0.12, 1.0)), -float(areas[i]))

    groups: List[List[int]] = []
    used_g = set()
    for idxs in opt.cluster_groups.values():
        mem = [i for i in idxs if i in todo]
        if len(mem) >= 2:
            groups.append(sorted(mem, key=band_key))
            used_g.update(mem)
    singles = [i for i in todo if i not in used_g]
    ent = ([(min(band_key(i) for i in g), g) for g in groups]
           + [(band_key(i), [i]) for i in singles])
    entities = [mem for _k, mem in sorted(ent, key=lambda e: e[0])]

    def place_group(mem: List[int]) -> bool:
        """Pack a cluster group as one meta-rect, then shelf-fill the
        members inside: rows are flush and soft members reshape to the
        row height, so the group stays one connected component by
        construction.  Groups with locked/ring mates are placed as close
        to those mates as possible instead of bottom-left."""
        a_g = sum(float(areas[j]) for j in mem) * 1.16
        mates = [j for j in (set(range(n)) - set(mem))
                 if j in placed and any(j in idxs and mem[0] in idxs
                                        for idxs in
                                        opt.cluster_groups.values())]
        anchor = None
        if mates:
            anchor = (sum(out[j, 0] + 0.5 * out[j, 2] for j in mates)
                      / len(mates),
                      sum(out[j, 1] + 0.5 * out[j, 3] for j in mates)
                      / len(mates))
        best = None
        for fx, fy, fw, fh in free:
            for wf in (math.sqrt(a_g), fw, a_g / fh if fh > 0 else 0.0):
                if wf <= 0:
                    continue
                hf = a_g / wf
                if wf <= fw + 1e-9 and hf <= fh + 1e-9 \
                        and max(wf / hf, hf / wf) <= 6.0:
                    if anchor is not None:
                        gx2 = min(max(anchor[0] - 0.5 * wf, fx),
                                  fx + fw - wf)
                        gy2 = min(max(anchor[1] - 0.5 * hf, fy),
                                  fy + fh - hf)
                        key = (abs(gx2 + 0.5 * wf - anchor[0])
                               + abs(gy2 + 0.5 * hf - anchor[1]), 0.0, 0.0)
                    else:
                        gx2, gy2 = fx, fy
                        key = (fy, fx, min(fw - wf, fh - hf))
                    if best is None or key < best[0]:
                        best = (key, gx2, gy2, wf, hf)
        if best is None:
            return False
        _k, gx, gy, gw, gh = best
        # shelf-fill inside the meta rect (two-phase: validate then commit)
        inner = sorted(mem, key=lambda j: -(float(opt.rh[j]) if kind[j] == 1
                                            else math.sqrt(areas[j])))
        put: List[Tuple[int, float, float, float, float]] = []
        cx, cy = gx, gy
        row_h = 0.0
        for j in inner:
            aj = float(areas[j])
            if kind[j] == 1:
                wj, hj = float(opt.rw[j]), float(opt.rh[j])
            else:
                hj = row_h if row_h > 0 else min(math.sqrt(aj) * 1.2, gh)
                hj = min(max(hj, math.sqrt(aj / MAX_ASPECT)), gh)
                wj = aj / hj
                if wj / hj > MAX_ASPECT:
                    wj = math.sqrt(aj * MAX_ASPECT)
                    hj = aj / wj
            if cx + wj > gx + gw + 1e-9 and cx > gx + 1e-9:
                cy += row_h
                cx, row_h = gx, 0.0
            if kind[j] == 0 and row_h > 0:
                hj = min(row_h, max(gh - (cy - gy), 1e-6))
                hj = max(hj, math.sqrt(aj / MAX_ASPECT))
                wj = aj / hj
            if (cx + wj > gx + gw + 1e-6
                    or cy + hj > gy + gh + 1e-6):
                return False   # spills outside the meta rect: abort
            put.append((j, cx, cy, wj, hj))
            cx += wj
            row_h = max(row_h, hj)
        for j, xj, yj, wj, hj in put:
            out[j] = (xj, yj, wj, hj)
            placed.add(j)
        occupy((gx, gy, gw, gh))
        return True

    # groups are NOT reserved as meta rects (any reserved slack is exactly
    # the margin that makes 96%-util frames unpackable); instead their
    # members are placed consecutively and each member prefers the free
    # spot closest to the group's already-placed footprint — BL packs
    # flush, so members glue together without an area tax
    # 'scatter' (default): groups place like singles — the ONLY mode that
    # reliably fits 96%-util frames; the downstream refiner's cluster
    # reassembly is responsible for regrouping.  'chain' glues members via
    # flush positions but fragments the packing (kept for experiments).
    group_of: Dict[int, int] = {}
    if group_mode == "chain":
        for gi, mem in enumerate(g for g in entities if len(g) >= 2):
            for j in mem:
                group_of[j] = gi
    g_last: Dict[int, Tuple[float, float, float, float]] = {}
    for idxs in opt.cluster_groups.values():
        mem_all = sorted(int(j) for j in idxs)
        mv = [j for j in mem_all if j in todo]
        if len(mv) < 2:
            continue
        gi = group_of.get(mv[0])
        if gi is None:
            continue
        for j in mem_all:
            if j in placed:   # a locked / ring mate seeds the chain
                g_last[gi] = tuple(out[j])
                break

    order: List[int] = []
    for mem in entities:
        order.extend(mem)

    t_check = 0
    for i in order:
        t_check += 1
        if (t_check & 15) == 0 and time.time() >= deadline:
            return None
        a = float(areas[i])
        if kind[i] == 1:
            shapes = [(float(opt.rw[i]), float(opt.rh[i]))]
        else:
            w0 = float(pred[i, 2]) if pred[i, 2] > 0 else math.sqrt(a)
            h0 = a / max(w0, 1e-9)
            if max(w0 / h0, h0 / w0) > MAX_ASPECT:
                w0 = math.sqrt(a)
                h0 = w0
            shapes = [(w0, h0)]
        # bottom-left placement discipline: the model's structure is kept
        # by the placement ORDER (row-major bands of the prediction); the
        # POSITION must serve packing tightness or 96%-util frames are
        # unreachable (position-chasing fragments the free space).
        gi = group_of.get(i)
        prev = g_last.get(gi) if gi is not None else None

        def find_best(aspect_cap):
            best = None   # (key, x, y, w, h)
            # chain placement for grouped members: exact flush positions
            # against the previous member (zero fragmentation, exact
            # adjacency); each candidate must sit inside ONE free rect
            if prev is not None:
                px, py, pw, ph = prev
                for (w0, h0) in (shapes if kind[i] != 0 else
                                 shapes + [(math.sqrt(a), math.sqrt(a))]):
                    for (x2, y2) in ((px + pw, py), (px - w0, py),
                                     (px, py + ph), (px, py - h0),
                                     (px + pw, py + ph - h0),
                                     (px + pw - w0, py + ph)):
                        for fx, fy, fw, fh in free:
                            if (x2 >= fx - 1e-9 and y2 >= fy - 1e-9
                                    and x2 + w0 <= fx + fw + 1e-9
                                    and y2 + h0 <= fy + fh + 1e-9):
                                key = (0, abs(x2 - px) + abs(y2 - py), 0.0)
                                if best is None or key < best[0]:
                                    best = (key, x2, y2, w0, h0)
                                break
                if best is not None:
                    return best
            for fx, fy, fw, fh in free:
                cands = []
                for (w0, h0) in shapes:
                    if w0 <= fw + 1e-9 and h0 <= fh + 1e-9:
                        cands.append((w0, h0))
                if kind[i] == 0:
                    for wf in (fw, (a / fh) if fh > 0 else 0.0):
                        if wf <= 0:
                            continue
                        hf = a / wf
                        if (wf <= fw + 1e-9 and hf <= fh + 1e-9
                                and max(wf / hf, hf / wf) <= aspect_cap):
                            cands.append((wf, hf))
                for (w2, h2) in cands:
                    key = (1, fy, fx)
                    if best is None or key < best[0]:
                        best = (key, fx, fy, w2, h2)
            return best

        best = find_best(MAX_ASPECT)
        if best is None and kind[i] == 0:
            # last resort: slivers are legal for soft blocks, and the
            # downstream refiner can reshape them later
            best = find_best(30.0)
        if best is None:
            return None       # does not fit: caller falls back
        _k, x2, y2, w2, h2 = best
        out[i] = (x2, y2, w2, h2)
        placed.add(i)
        occupy((x2, y2, w2, h2))
        if gi is not None:
            g_last[gi] = (x2, y2, w2, h2)

    return out
