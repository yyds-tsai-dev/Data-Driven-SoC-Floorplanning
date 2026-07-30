"""Targeted post-pass that eliminates residual soft violations.

Runs after the column/direct pipeline has produced its final legal layout.
For each remaining boundary / grouping / MIB violation it enumerates a small
set of exact candidate placements (wall slots, wall swaps, soft reshapes,
component re-attachment) and accepts a move only when the number of
violations strictly drops AND an evaluator-faithful cost strictly improves.

Faithfulness notes (mirrors scripts/iccad2026_evaluate.py):
  overlap   hard violation iff BOTH axis overlaps exceed 1e-6
  boundary  |edge - bbox_edge| < 1e-6 (absolute)
  grouping  shapely unary_union: connected iff exact touch with a shared
            edge of positive length (corner-point touch does NOT connect,
            any positive gap does NOT connect) -> we test ox>=0 & oy>=0 &
            (ox>0 | oy>0) in the same float arithmetic the evaluator uses
  MIB       distinct (round(w,4), round(h,4)) per group
  area      hard tolerance 1% -> reshapes keep w*h == area exactly (ulp)
  fixed     never reshaped; preplaced never moved

Containment: any exception, guard failure, or non-improvement returns the
input unchanged.  Deterministic (no RNG).  Debug prints via VKILL_DEBUG=1.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from legalizer_claude import _ColumnOptimizer, MIB_AREA_GUARD

Rect = Tuple[float, float, float, float]

B_EPS = 1e-6          # evaluator boundary-touch epsilon (absolute)
OV_GUARD = 1e-7       # own overlap guard, 10x stricter than evaluator's 1e-6
JOIN = 5e-10          # attach bias: overlap the join so float drift can
                      # never open a gap (legal: << 1e-6; union: connected)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
MAX_ASPECT = 10.0     # cap for reshape slivers
HP_SAFETY = 1.25      # deflate apparent HPWL rewards / inflate HPWL costs
MEMBER_CAP = 10       # attach targets per grouping fix
SWAP_CAP = 12         # wall-swap partners per boundary fix


def kill_violations(out, area_targets, constraints, target_positions,
                    b2b, p2b, pins, budget: float = 6.0,
                    verbose: bool = False):
    """Entry point used by my_opt_claude.MyOptimizer._violation_kill."""
    try:
        res = _kill(list(out), area_targets, constraints, target_positions,
                    b2b, p2b, pins, float(budget), bool(verbose))
        return out if res is None else res
    except Exception:
        if verbose or os.environ.get("VKILL_DEBUG"):
            traceback.print_exc()
        return out


# ---------------------------------------------------------------------------
# evaluator-faithful violation counting
# ---------------------------------------------------------------------------

def _bbox(P: np.ndarray) -> Tuple[float, float, float, float]:
    x0 = P[:, 0]
    y0 = P[:, 1]
    return (float(x0.min()), float(y0.min()),
            float((x0 + P[:, 2]).max()), float((y0 + P[:, 3]).max()))


def _bbox_area(P: np.ndarray) -> float:
    X0, Y0, X1, Y1 = _bbox(P)
    return (X1 - X0) * (Y1 - Y0)


def _boundary_violators(opt, P: np.ndarray) -> List[Tuple[int, int]]:
    if not len(opt._bnd_idx):
        return []
    x0 = P[:, 0]
    y0 = P[:, 1]
    x1 = x0 + P[:, 2]
    y1 = y0 + P[:, 3]
    X0, Y0, X1, Y1 = _bbox(P)
    bad = []
    for i, code in zip(opt._bnd_idx, opt._bnd_codes):
        i = int(i)
        code = int(code)
        ok = True
        if code & 1:
            ok = ok and abs(x0[i] - X0) < B_EPS
        if code & 2:
            ok = ok and abs(x1[i] - X1) < B_EPS
        if code & 4:
            ok = ok and abs(y1[i] - Y1) < B_EPS
        if code & 8:
            ok = ok and abs(y0[i] - Y0) < B_EPS
        if not ok:
            bad.append((i, code))
    return bad


def _components(P: np.ndarray, g: np.ndarray) -> List[List[int]]:
    """Connected components under the evaluator's exact-touch rule."""
    x0 = P[g, 0]
    y0 = P[g, 1]
    x1 = x0 + P[g, 2]
    y1 = y0 + P[g, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    adj = (ox >= 0.0) & (oy >= 0.0) & ((ox > 0.0) | (oy > 0.0))
    m = len(g)
    seen = np.zeros(m, dtype=bool)
    comps: List[List[int]] = []
    for s in range(m):
        if seen[s]:
            continue
        frontier = np.zeros(m, dtype=bool)
        frontier[s] = True
        seen[s] = True
        comp = [s]
        while frontier.any():
            nxt = adj[frontier].any(axis=0) & ~seen
            seen |= nxt
            comp.extend(np.nonzero(nxt)[0].tolist())
            frontier = nxt
        comps.append([int(g[k]) for k in comp])  # back to block ids
    return comps


def _grouping_count(opt, P: np.ndarray) -> int:
    V = 0
    for idxs in opt.cluster_groups.values():
        if len(idxs) < 2:
            continue
        g = np.asarray(sorted(int(i) for i in idxs), dtype=np.int64)
        V += len(_components(P, g)) - 1
    return V


def _mib_count(opt, P: np.ndarray) -> int:
    V = 0
    for g in opt._mib_arrays:
        shapes = {(round(float(P[i, 2]), 4), round(float(P[i, 3]), 4))
                  for i in g}
        V += len(shapes) - 1
    return V


def _violations_exact(opt, P: np.ndarray) -> int:
    return (len(_boundary_violators(opt, P))
            + _grouping_count(opt, P)
            + _mib_count(opt, P))


def _overlap_ok(P: np.ndarray) -> bool:
    """No pair may overlap by more than OV_GUARD on both axes."""
    x0 = P[:, 0]
    y0 = P[:, 1]
    x1 = x0 + P[:, 2]
    y1 = y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > OV_GUARD) & (oy > OV_GUARD)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# scoring / acceptance
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, opt, P0: np.ndarray):
        self.opt = opt
        self.hp0 = max(float(opt._hpwl(P0)), 1e-9)
        self.A0 = max(_bbox_area(P0), 1e-9)
        self.M = float(max(opt.n_soft_den, 1))

    def score(self, P: np.ndarray) -> Tuple[float, int]:
        hp = float(self.opt._hpwl(P))
        dh = (hp - self.hp0) / self.hp0
        dh *= HP_SAFETY if dh > 0.0 else (1.0 / HP_SAFETY)
        dA = (_bbox_area(P) - self.A0) / self.A0
        dA *= HP_SAFETY if dA > 0.0 else (1.0 / HP_SAFETY)
        V = _violations_exact(self.opt, P)
        return (1.0 + 0.5 * (dh + dA)) * math.exp(2.0 * V / self.M), V


def _try_candidates(ctx: _Ctx, P: np.ndarray, cur_score: float, cur_V: int,
                    cands: List[np.ndarray], t_end: float,
                    allow_equal_v: bool = False):
    """Return the best strictly-improving candidate, or None.  V must drop
    (or stay equal when allow_equal_v, for shape/score-only fixers)."""
    best = None
    for Q in cands:
        if time.time() >= t_end:
            break
        if not _overlap_ok(Q):
            continue
        s, V = ctx.score(Q)
        v_ok = (V < cur_V) or (allow_equal_v and V <= cur_V)
        if v_ok and s < cur_score - 1e-12:
            if best is None or s < best[0]:
                best = (s, V, Q)
    return best


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _nudge_le(size: float, edge: float) -> float:
    """Origin o such that o+size lands ON or a few ulp OVER `edge`
    (guarantees exact-touch-or-tiny-overlap, never a gap)."""
    o = edge - size
    for _ in range(4):
        if o + size >= edge:
            return o
        o = float(np.nextafter(o, math.inf))
    return o


def _free_gaps(P: np.ndarray, i: int, axis: int, strip_lo: float,
               strip_hi: float, lo: float, hi: float) -> List[Tuple[float, float]]:
    """Free intervals along `axis` within [lo, hi] inside the perpendicular
    strip [strip_lo, strip_hi], treating every block except i as an obstacle."""
    a0 = P[:, axis]
    a1 = a0 + P[:, 2 + axis]
    p0 = P[:, 1 - axis]
    p1 = p0 + P[:, 3 - axis]
    ivs = []
    for j in range(len(P)):
        if j == i:
            continue
        if min(p1[j], strip_hi) - max(p0[j], strip_lo) > 1e-9:
            ivs.append((float(a0[j]), float(a1[j])))
    ivs.sort()
    gaps = []
    cur = lo
    for s, e in ivs:
        if s > cur:
            gaps.append((cur, min(s, hi)))
        cur = max(cur, e)
        if cur >= hi:
            break
    if cur < hi:
        gaps.append((cur, hi))
    return [(s, e) for s, e in gaps if e - s > 1e-12]


# ---------------------------------------------------------------------------
# fixers
# ---------------------------------------------------------------------------

def _shove_rect(P: np.ndarray, exclude, axis: int, wall_lo: float,
                wall_hi: float, band_lo: float, band_hi: float,
                t: float, size: float, rigid: np.ndarray):
    """Make room for a `size`-long interval near t along `axis` inside the
    perpendicular band [band_lo, band_hi], shifting movable residents
    minimally (order-preserving); rigid residents act as fixed barriers
    that displaced blocks jump over.  Returns (t_final, {j: new_a0}) or
    None when the line overflows [wall_lo, wall_hi]."""
    p0 = P[:, 1 - axis]
    p1 = p0 + P[:, 3 - axis]
    a0 = P[:, axis]
    a1 = a0 + P[:, 2 + axis]
    ex = set(int(e) for e in exclude)
    res = [j for j in range(len(P)) if j not in ex
           and min(p1[j], band_hi) - max(p0[j], band_lo) > 1e-9]
    bars = sorted((float(a0[j]), float(a1[j])) for j in res if rigid[j])
    mov = [j for j in res if not rigid[j]]
    t = min(max(t, wall_lo), wall_hi - size)
    for b0, b1 in bars:  # nudge the target off any barrier (nearer side)
        if t + size > b0 and t < b1:
            below = b0 - size
            t = below if (t - below <= b1 - t and below >= wall_lo) else b1
    if t < wall_lo - 1e-9 or t + size > wall_hi + 1e-9:
        return None
    for b0, b1 in bars:
        if t + size > b0 and t < b1:
            return None
    shifts: Dict[int, float] = {}
    lo_side = sorted((j for j in mov if a0[j] + a1[j] <= 2 * t + size),
                     key=lambda j: -a1[j])
    hi_side = sorted((j for j in mov if a0[j] + a1[j] > 2 * t + size),
                     key=lambda j: a0[j])
    floor = t
    for j in lo_side:
        s = float(P[j, 2 + axis])
        new_end = min(float(a1[j]), floor)
        moved = True
        while moved:  # jump below any barrier the block would straddle
            moved = False
            for b0, b1 in bars:
                if new_end > b0 and new_end - s < b1:
                    new_end = b0
                    moved = True
        if new_end - s < wall_lo - 1e-9:
            return None
        shifts[j] = new_end - s
        floor = new_end - s
    ceil = t + size
    for j in hi_side:
        s = float(P[j, 2 + axis])
        new_start = max(float(a0[j]), ceil)
        moved = True
        while moved:  # jump above any barrier the block would straddle
            moved = False
            for b0, b1 in bars:
                if new_start < b1 and new_start + s > b0:
                    new_start = b1
                    moved = True
        if new_start + s > wall_hi + 1e-9:
            return None
        shifts[j] = new_start
        ceil = new_start + s
    return t, shifts


def _shove_insert(P: np.ndarray, i: int, axis: int, wall_lo: float,
                  wall_hi: float, band_lo: float, band_hi: float,
                  t: float, rigid: np.ndarray) -> Optional[np.ndarray]:
    got = _shove_rect(P, (i,), axis, wall_lo, wall_hi, band_lo, band_hi,
                      t, float(P[i, 2 + axis]), rigid)
    if got is None:
        return None
    t_f, shifts = got
    Q = P.copy()
    Q[i, axis] = t_f
    for j, v in shifts.items():
        Q[j, axis] = v
    return Q


def _weld_comp(P0: np.ndarray, Q: np.ndarray, comp: List[int]) -> None:
    """Re-close internal contacts of `comp` that float drift turned into
    sub-nanometre gaps after a rigid translate (tiny overlaps are fine for
    both the evaluator and the union test).  Mutates Q in place."""
    comp = sorted(comp)
    for _ in range(3):
        moved = False
        for ai in range(len(comp)):
            for bi in range(ai + 1, len(comp)):
                a, b = comp[ai], comp[bi]
                ox0 = (min(P0[a, 0] + P0[a, 2], P0[b, 0] + P0[b, 2])
                       - max(P0[a, 0], P0[b, 0]))
                oy0 = (min(P0[a, 1] + P0[a, 3], P0[b, 1] + P0[b, 3])
                       - max(P0[a, 1], P0[b, 1]))
                if not (ox0 >= 0 and oy0 >= 0 and (ox0 > 0 or oy0 > 0)):
                    continue  # was not a contact pair
                ox = (min(Q[a, 0] + Q[a, 2], Q[b, 0] + Q[b, 2])
                      - max(Q[a, 0], Q[b, 0]))
                oy = (min(Q[a, 1] + Q[a, 3], Q[b, 1] + Q[b, 3])
                      - max(Q[a, 1], Q[b, 1]))
                if -1e-9 < ox < 0 and oy > 0:
                    delta = -ox + 1e-10
                    Q[b, 0] += -delta if Q[b, 0] > Q[a, 0] else delta
                    moved = True
                elif -1e-9 < oy < 0 and ox > 0:
                    delta = -oy + 1e-10
                    Q[b, 1] += -delta if Q[b, 1] > Q[a, 1] else delta
                    moved = True
        if not moved:
            break


def _comp_attach_shove(P: np.ndarray, comp: List[int],
                       target_bbox: Tuple[float, float, float, float],
                       rigid_x: np.ndarray,
                       rigid_y: np.ndarray) -> List[np.ndarray]:
    """Candidates that rigidly translate `comp` onto each flank of
    target_bbox, with shove-made room.  The join is biased JOIN into the
    target so float drift can never open a gap; internal contacts are
    re-welded after the translate."""
    X0, Y0, X1, Y1 = _bbox(P)
    cx0 = min(float(P[b, 0]) for b in comp)
    cy0 = min(float(P[b, 1]) for b in comp)
    cx1 = max(float(P[b, 0] + P[b, 2]) for b in comp)
    cy1 = max(float(P[b, 1] + P[b, 3]) for b in comp)
    cw = cx1 - cx0
    ch = cy1 - cy0
    tx0, ty0, tx1, ty1 = target_bbox
    out: List[np.ndarray] = []
    specs = (
        (1, tx1 - JOIN, (ty0, ty1 - ch, 0.5 * (ty0 + ty1 - ch)), rigid_y),
        (1, tx0 - cw + JOIN, (ty0, ty1 - ch, 0.5 * (ty0 + ty1 - ch)),
         rigid_y),
        (0, ty1 - JOIN, (tx0, tx1 - cw, 0.5 * (tx0 + tx1 - cw)), rigid_x),
        (0, ty0 - ch + JOIN, (tx0, tx1 - cw, 0.5 * (tx0 + tx1 - cw)),
         rigid_x),
    )
    for axis, anchor, t_list, rigid in specs:
        if axis == 1:
            lo, hi = Y0, Y1
            band = (anchor, anchor + cw)
            size = ch
        else:
            lo, hi = X0, X1
            band = (anchor, anchor + ch)
            size = cw
        for t in t_list:
            got = _shove_rect(P, comp, axis, lo, hi, band[0], band[1],
                              t, size, rigid)
            if got is None:
                continue
            t_f, shifts = got
            if axis == 1:
                dx, dy = anchor - cx0, t_f - cy0
            else:
                dx, dy = t_f - cx0, anchor - cy0
            Q = P.copy()
            for b in comp:
                Q[b, 0] += dx
                Q[b, 1] += dy
            for j, v in shifts.items():
                Q[j, axis] = v
            if len(comp) > 1:
                _weld_comp(P, Q, comp)
            out.append(Q)
    return out


def _wall_repack(P: np.ndarray, i: int, wall: str, t: float,
                 kind: List[int], areas: List[float],
                 rigid: np.ndarray) -> Optional[np.ndarray]:
    """Rebuild one wall line: order-preserving repack of the wall band's
    movable residents with i inserted near coordinate t; rigid residents
    act as fixed partitions the line packs around.  If the line overflows,
    soft residents are proportionally shrunk along the wall (exact area
    kept, growing into the die, aspect-capped).  Blocks are then relaxed
    back toward their original coordinates within their segment.  Only
    residents already touching the wall are re-pinned to it.  None on
    failure."""
    X0, Y0, X1, Y1 = _bbox(P)
    n = len(P)
    axis = 1 if wall in ('L', 'R') else 0
    wlo, whi = (Y0, Y1) if axis == 1 else (X0, X1)
    pw = float(P[i, 3 - axis])
    if wall == 'L':
        band = (X0, X0 + pw)
    elif wall == 'R':
        band = (X1 - pw, X1)
    elif wall == 'B':
        band = (Y0, Y0 + pw)
    else:
        band = (Y1 - pw, Y1)
    p0 = P[:, 1 - axis]
    p1 = p0 + P[:, 3 - axis]
    a0 = P[:, axis]
    res = [j for j in range(n)
           if j != i and min(p1[j], band[1]) - max(p0[j], band[0]) > 1e-9]
    bars = sorted((float(a0[j]), float(a0[j] + P[j, 2 + axis]))
                  for j in res if rigid[j])
    movable = [j for j in res if not rigid[j]]
    segs: List[List[float]] = []
    cur = wlo
    for b0, b1 in bars:
        if b0 > cur + 1e-12:
            segs.append([cur, min(b0, whi)])
        cur = max(cur, b1)
    if cur < whi - 1e-12:
        segs.append([cur, whi])
    if not segs:
        return None
    items = sorted(movable, key=lambda j: float(a0[j]))
    sizes = {j: float(P[j, 2 + axis]) for j in items}
    size_i = float(P[i, 2 + axis])
    rank = 0
    for idx, j in enumerate(items):
        if float(a0[j]) + 0.5 * sizes[j] < t + 0.5 * size_i:
            rank = idx + 1
    items.insert(rank, i)
    sizes[i] = size_i
    span = sum(e - s for s, e in segs)
    total = sum(sizes.values())
    shrink: Dict[int, float] = {}
    if total > span + 1e-9:
        cap = {}
        for j in items:
            if j != i and kind[j] == 0:
                s_min = math.sqrt(areas[j] / MAX_ASPECT)
                cap[j] = max(0.0, sizes[j] - s_min)
        avail = sum(cap.values())
        deficit = total - span
        if avail < deficit - 1e-12:
            return None
        k = deficit / avail
        shrink = {j: c * k for j, c in cap.items() if c > 0.0}

    def eff(j):
        return sizes[j] - shrink.get(j, 0.0)

    # first-fit in order across the segments between the rigid partitions
    placed: Dict[int, Tuple[int, float]] = {}
    si = 0
    cursor = segs[0][0]
    for j in items:
        s = eff(j)
        while si < len(segs) and cursor + s > segs[si][1] + 1e-9:
            si += 1
            if si < len(segs):
                cursor = segs[si][0]
        if si >= len(segs):
            return None
        placed[j] = (si, cursor)
        cursor += s
    # relax within each segment: pull blocks back toward their originals
    by_seg: Dict[int, List[int]] = {}
    for j, (sj, _pos) in placed.items():
        by_seg.setdefault(sj, []).append(j)
    final: Dict[int, float] = {}
    for sj, js in by_seg.items():
        js.sort(key=lambda j: placed[j][1])
        limit = segs[sj][1]
        for j in reversed(js):
            s = eff(j)
            desired = t if j == i else float(a0[j])
            pos = min(max(desired, placed[j][1]), limit - s)
            final[j] = pos
            limit = pos
    Q = P.copy()
    for j, pos in final.items():
        s = eff(j)
        Q[j, axis] = pos
        if j != i and j in shrink:
            Q[j, 2 + axis] = s
            Q[j, 3 - axis] = areas[j] / s
        on_wall = (j == i
                   or (wall == 'L' and abs(p0[j] - X0) < B_EPS)
                   or (wall == 'R' and abs(p1[j] - X1) < B_EPS)
                   or (wall == 'B' and abs(p0[j] - Y0) < B_EPS)
                   or (wall == 'T' and abs(p1[j] - Y1) < B_EPS))
        if not on_wall:
            continue
        if wall == 'L':
            Q[j, 0] = X0
        elif wall == 'R':
            Q[j, 0] = X1 - float(Q[j, 2])
        elif wall == 'B':
            Q[j, 1] = Y0
        else:
            Q[j, 1] = Y1 - float(Q[j, 3])
    return Q


def _fix_boundary(ctx: _Ctx, P: np.ndarray, cur_score: float, cur_V: int,
                  kind: List[int], areas: List[float], t_end: float):
    opt = ctx.opt
    changed = True
    while changed and time.time() < t_end:
        changed = False
        for i, code in _boundary_violators(opt, P):
            if time.time() >= t_end:
                break
            if kind[i] == 2:
                continue
            X0, Y0, X1, Y1 = _bbox(P)
            w = float(P[i, 2])
            h = float(P[i, 3])
            shapes = [(w, h)]
            cands: List[np.ndarray] = []

            def add(x, y, ww, hh):
                Q = P.copy()
                Q[i, 0] = x
                Q[i, 1] = y
                Q[i, 2] = ww
                Q[i, 3] = hh
                cands.append(Q)

            need_l = bool(code & 1)
            need_r = bool(code & 2)
            need_t = bool(code & 4)
            need_b = bool(code & 8)

            for ww, hh in list(shapes):
                # fully constrained on x?
                if need_l and need_r:
                    if kind[i] == 0:
                        ww2 = X1 - X0
                        hh2 = areas[i] / ww2
                        if max(ww2 / hh2, hh2 / ww2) <= MAX_ASPECT * 2:
                            for y in (Y0, Y1 - hh2, float(P[i, 1])):
                                add(X0, min(max(y, Y0), Y1 - hh2), ww2, hh2)
                    continue
                if need_t and need_b:
                    if kind[i] == 0:
                        hh2 = Y1 - Y0
                        ww2 = areas[i] / hh2
                        if max(ww2 / hh2, hh2 / ww2) <= MAX_ASPECT * 2:
                            for x in (X0, X1 - ww2, float(P[i, 0])):
                                add(min(max(x, X0), X1 - ww2), Y0, ww2, hh2)
                    continue

                fx = X0 if need_l else (X1 - ww if need_r else None)
                fy = Y0 if need_b else (Y1 - hh if need_t else None)

                if fx is not None and fy is not None:
                    add(fx, fy, ww, hh)
                elif fx is not None:
                    for gs, ge in _free_gaps(P, i, 1, fx, fx + ww, Y0, Y1):
                        room = ge - gs
                        if room >= hh - 1e-9:
                            for y in (gs, ge - hh,
                                      min(max(float(P[i, 1]), gs), ge - hh)):
                                add(fx, y, ww, hh)
                        elif kind[i] == 0 and room > 1e-9:
                            hh2 = room
                            ww2 = areas[i] / hh2
                            if max(ww2 / hh2, hh2 / ww2) <= MAX_ASPECT:
                                x2 = X0 if need_l else X1 - ww2
                                add(x2, gs, ww2, hh2)
                elif fy is not None:
                    for gs, ge in _free_gaps(P, i, 0, fy, fy + hh, X0, X1):
                        room = ge - gs
                        if room >= ww - 1e-9:
                            for x in (gs, ge - ww,
                                      min(max(float(P[i, 0]), gs), ge - ww)):
                                add(x, fy, ww, hh)
                        elif kind[i] == 0 and room > 1e-9:
                            ww2 = room
                            hh2 = areas[i] / ww2
                            if max(ww2 / hh2, hh2 / ww2) <= MAX_ASPECT:
                                y2 = Y0 if need_b else Y1 - hh2
                                add(gs, y2, ww2, hh2)

            # shove / repack insertion along the required wall(s); corner
            # codes get pinned targets on both wall variants
            x_bit = code & 3
            y_bit = code & 12
            if x_bit in (1, 2) or y_bit in (4, 8):
                base_rigid = np.array([k == 2 for k in kind], dtype=bool)
                bnd_code_of = {int(b): int(c) for b, c in
                               zip(opt._bnd_idx, opt._bnd_codes)}
                variants = []
                if x_bit in (1, 2):
                    wall = 'L' if x_bit == 1 else 'R'
                    if y_bit == 8:
                        t_list = (Y0,)
                    elif y_bit == 4:
                        t_list = (Y1 - h,)
                    else:
                        t_list = (float(P[i, 1]), Y0, Y1 - h,
                                  0.5 * (Y0 + Y1 - h))
                    variants.append((wall, 1, Y0, Y1, t_list))
                if y_bit in (4, 8):
                    wall = 'B' if y_bit == 8 else 'T'
                    if x_bit == 1:
                        t_list = (X0,)
                    elif x_bit == 2:
                        t_list = (X1 - w,)
                    else:
                        t_list = (float(P[i, 0]), X0, X1 - w,
                                  0.5 * (X0 + X1 - w))
                    variants.append((wall, 0, X0, X1, t_list))
                for wall, axis, lo, hi, t_list in variants:
                    rigid = base_rigid.copy()
                    for bj, bc_ in bnd_code_of.items():
                        if bj == i:
                            continue
                        if axis == 1 and bc_ & 12:
                            rigid[bj] = True
                        if axis == 0 and bc_ & 3:
                            rigid[bj] = True
                    if wall == 'L':
                        wall_c, band = X0, (X0, X0 + w)
                    elif wall == 'R':
                        wall_c, band = X1 - w, (X1 - w, X1)
                    elif wall == 'B':
                        wall_c, band = Y0, (Y0, Y0 + h)
                    else:
                        wall_c, band = Y1 - h, (Y1 - h, Y1)
                    Q0 = P.copy()
                    if axis == 1:
                        Q0[i, 0] = wall_c
                    else:
                        Q0[i, 1] = wall_c
                    for t in t_list:
                        Q = _shove_insert(Q0, i, axis, lo, hi,
                                          band[0], band[1], t, rigid)
                        if Q is not None:
                            cands.append(Q)
                    Qr = _wall_repack(P, i, wall, t_list[0], kind, areas,
                                      rigid)
                    if Qr is not None:
                        cands.append(Qr)

            # swaps with current wall residents (single-wall codes only)
            if not (need_l and need_r) and not (need_t and need_b):
                x0a = P[:, 0]
                y0a = P[:, 1]
                x1a = x0a + P[:, 2]
                y1a = y0a + P[:, 3]
                on_wall = np.zeros(len(P), dtype=bool)
                if need_l:
                    on_wall = np.abs(x0a - X0) < B_EPS
                elif need_r:
                    on_wall = np.abs(x1a - X1) < B_EPS
                elif need_t:
                    on_wall = np.abs(y1a - Y1) < B_EPS
                elif need_b:
                    on_wall = np.abs(y0a - Y0) < B_EPS
                bnd_set = {int(b) for b in opt._bnd_idx}
                partners = [j for j in np.nonzero(on_wall)[0]
                            if j != i and kind[int(j)] != 2
                            and int(j) not in bnd_set]
                partners.sort(key=lambda j: abs(float(y0a[j] - y0a[i]))
                              + abs(float(x0a[j] - x0a[i])))
                for j in partners[:SWAP_CAP]:
                    j = int(j)
                    wj = float(P[j, 2])
                    hj = float(P[j, 3])
                    if kind[i] == 0:
                        wi2 = wj
                        hi2 = areas[i] / wj
                        if max(wi2 / hi2, hi2 / wi2) > MAX_ASPECT:
                            wi2, hi2 = w, h
                    else:
                        wi2, hi2 = w, h
                    Q = P.copy()
                    Q[i, 0] = P[j, 0]
                    Q[i, 1] = P[j, 1]
                    Q[i, 2] = wi2
                    Q[i, 3] = hi2
                    Q[j, 0] = P[i, 0]
                    Q[j, 1] = P[i, 1]
                    cands.append(Q)

            best = _try_candidates(ctx, P, cur_score, cur_V, cands, t_end)
            if best is not None:
                cur_score, cur_V = best[0], best[1]
                P = best[2]
                changed = True
                if os.environ.get("VKILL_DEBUG"):
                    print(f"[vkill] boundary fix block {i} code {code} "
                          f"-> V={cur_V}", flush=True)
                break  # re-derive violator list on the mutated layout
            if os.environ.get("VKILL_DEBUG"):
                alt = None
                for Q in cands:
                    if not _overlap_ok(Q):
                        continue
                    s, V = ctx.score(Q)
                    if V < cur_V and (alt is None or s < alt[0]):
                        alt = (s, V)
                if alt is None:
                    print(f"[vkill] bnd block {i} code {code}: "
                          f"{len(cands)} cands, none legal+V-dropping",
                          flush=True)
                else:
                    print(f"[vkill] bnd block {i} code {code}: best "
                          f"V-drop score {alt[0]:.5f} vs cur "
                          f"{cur_score:.5f} ({alt[0] - cur_score:+.5f})",
                          flush=True)
    return P, cur_score, cur_V


def _fix_grouping(ctx: _Ctx, P: np.ndarray, cur_score: float, cur_V: int,
                  kind: List[int], areas: List[float], t_end: float):
    opt = ctx.opt
    changed = True
    while changed and time.time() < t_end:
        changed = False
        for idxs in opt.cluster_groups.values():
            if len(idxs) < 2 or time.time() >= t_end:
                continue
            g = np.asarray(sorted(int(i) for i in idxs), dtype=np.int64)
            comps = _components(P, g)
            if len(comps) <= 1:
                continue
            comps.sort(key=lambda c: -sum(float(P[b, 2] * P[b, 3]) for b in c))
            main = comps[0]
            rigid_y = np.array([k == 2 for k in kind], dtype=bool)
            rigid_x = rigid_y.copy()
            for bi_, bc_ in zip(opt._bnd_idx, opt._bnd_codes):
                if int(bc_) & 12:
                    rigid_y[int(bi_)] = True
                if int(bc_) & 3:
                    rigid_x[int(bi_)] = True

            def _bb_of(blocks):
                return (min(float(P[b, 0]) for b in blocks),
                        min(float(P[b, 1]) for b in blocks),
                        max(float(P[b, 0] + P[b, 2]) for b in blocks),
                        max(float(P[b, 1] + P[b, 3]) for b in blocks))

            cands: List[np.ndarray] = []
            for comp in comps[1:]:
                if any(kind[b] == 2 for b in comp):
                    # split piece is anchored: try moving the MAIN component
                    # onto it instead (only if main is fully movable)
                    if all(kind[b] != 2 for b in main):
                        cands.extend(_comp_attach_shove(
                            P, list(main), _bb_of(comp), rigid_x, rigid_y))
                    continue
                if len(comp) == 1:
                    b = comp[0]
                    wb = float(P[b, 2])
                    hb = float(P[b, 3])
                    targets = sorted(main, key=lambda m: -float(P[m, 2] * P[m, 3]))
                    for m in targets[:MEMBER_CAP]:
                        mx0 = float(P[m, 0])
                        my0 = float(P[m, 1])
                        mx1 = mx0 + float(P[m, 2])
                        my1 = my0 + float(P[m, 3])
                        pad = 1e-3 * min(hb, float(P[m, 3]))
                        ys = (my0, my1 - hb,
                              min(max(float(P[b, 1]), my0 - hb + pad), my1 - pad))
                        padx = 1e-3 * min(wb, float(P[m, 2]))
                        xs = (mx0, mx1 - wb,
                              min(max(float(P[b, 0]), mx0 - wb + padx), mx1 - padx))
                        for y in ys:
                            for x in (mx1 - JOIN, mx0 - wb + JOIN):
                                Q = P.copy()
                                Q[b, 0] = x
                                Q[b, 1] = y
                                cands.append(Q)
                        for x in xs:
                            for y in (my1 - JOIN, my0 - hb + JOIN):
                                Q = P.copy()
                                Q[b, 0] = x
                                Q[b, 1] = y
                                cands.append(Q)
                    for m in targets[:4]:
                        mx0 = float(P[m, 0])
                        my0 = float(P[m, 1])
                        mx1 = mx0 + float(P[m, 2])
                        my1 = my0 + float(P[m, 3])
                        pad = 1e-3 * min(hb, float(P[m, 3]))
                        padx = 1e-3 * min(wb, float(P[m, 2]))
                        # gap-based placements along each flank of m
                        for x_a in (mx1 - JOIN, mx0 - wb + JOIN):
                            for gs, ge in _free_gaps(P, b, 1, x_a, x_a + wb,
                                                     my0 - hb + pad,
                                                     my1 - pad + hb):
                                if ge - gs >= hb - 1e-9:
                                    for y in (gs, ge - hb):
                                        Q = P.copy()
                                        Q[b, 0] = x_a
                                        Q[b, 1] = y
                                        cands.append(Q)
                        for y_a in (my1 - JOIN, my0 - hb + JOIN):
                            for gs, ge in _free_gaps(P, b, 0, y_a, y_a + hb,
                                                     mx0 - wb + padx,
                                                     mx1 - padx + wb):
                                if ge - gs >= wb - 1e-9:
                                    for x in (gs, ge - wb):
                                        Q = P.copy()
                                        Q[b, 0] = x
                                        Q[b, 1] = y_a
                                        cands.append(Q)
                        cands.extend(_comp_attach_shove(
                            P, [b], (mx0, my0, mx1, my1), rigid_x, rigid_y))
                else:
                    cands.extend(_comp_attach_shove(
                        P, list(comp), _bb_of(main), rigid_x, rigid_y))
                    if all(kind[b] != 2 for b in main):
                        cands.extend(_comp_attach_shove(
                            P, list(main), _bb_of(comp), rigid_x, rigid_y))
            best = _try_candidates(ctx, P, cur_score, cur_V, cands, t_end)
            if best is not None:
                cur_score, cur_V = best[0], best[1]
                P = best[2]
                changed = True
                if os.environ.get("VKILL_DEBUG"):
                    print(f"[vkill] grouping fix -> V={cur_V}", flush=True)
                break
            if os.environ.get("VKILL_DEBUG"):
                sizes = [(len(cc), any(kind[b] == 2 for b in cc))
                         for cc in comps]
                alt = None
                n_legal = 0
                for Q in cands:
                    if not _overlap_ok(Q):
                        continue
                    n_legal += 1
                    s, V = ctx.score(Q)
                    if V < cur_V and (alt is None or s < alt[0]):
                        alt = (s, V)
                msg = (f"best V-drop {alt[0]:.5f} vs cur {cur_score:.5f} "
                       f"({alt[0] - cur_score:+.5f})" if alt
                       else f"legal={n_legal}, none V-dropping")
                print(f"[vkill] grouping stuck: comps(size,locked)={sizes} "
                      f"cands={len(cands)} {msg}", flush=True)
    return P, cur_score, cur_V


def _fix_mib(ctx: _Ctx, P: np.ndarray, cur_score: float, cur_V: int,
             kind: List[int], areas: List[float], t_end: float):
    opt = ctx.opt
    for g in opt._mib_arrays:
        if time.time() >= t_end:
            break
        shapes: Dict[Tuple[float, float], List[int]] = {}
        for i in g:
            key = (round(float(P[i, 2]), 4), round(float(P[i, 3]), 4))
            shapes.setdefault(key, []).append(int(i))
        if len(shapes) <= 1:
            continue
        rigid_keys = [k for k, members in shapes.items()
                      if any(kind[i] != 0 for i in members)]
        if len(rigid_keys) > 1:
            if os.environ.get("VKILL_DEBUG"):
                print(f"[vkill] mib stuck: {len(rigid_keys)} conflicting "
                      f"rigid shapes", flush=True)
            continue  # conflicting hard shapes: unfixable here
        if rigid_keys:
            ref_i = next((i for i in shapes[rigid_keys[0]] if kind[i] != 0),
                         shapes[rigid_keys[0]][0])
        else:
            ref_i = max(shapes.values(), key=len)[0]
        tw = float(P[ref_i, 2])
        th = float(P[ref_i, 3])
        for key, members in list(shapes.items()):
            for i in members:
                if (float(P[i, 2]) == tw and float(P[i, 3]) == th):
                    continue
                if kind[i] != 0:
                    continue
                if abs(tw * th - areas[i]) / max(areas[i], 1e-9) > MIB_AREA_GUARD:
                    continue
                cx = float(P[i, 0]) + 0.5 * float(P[i, 2])
                cy = float(P[i, 1]) + 0.5 * float(P[i, 3])
                cands = []
                for x, y in ((float(P[i, 0]), float(P[i, 1])),
                             (cx - 0.5 * tw, cy - 0.5 * th),
                             (float(P[i, 0] + P[i, 2]) - tw, float(P[i, 1])),
                             (float(P[i, 0]), float(P[i, 1] + P[i, 3]) - th)):
                    Q = P.copy()
                    Q[i, 0] = x
                    Q[i, 1] = y
                    Q[i, 2] = tw
                    Q[i, 3] = th
                    cands.append(Q)
                best = _try_candidates(ctx, P, cur_score, cur_V, cands, t_end)
                if best is not None:
                    cur_score, cur_V = best[0], best[1]
                    P = best[2]
                    if os.environ.get("VKILL_DEBUG"):
                        print(f"[vkill] mib fix block {i} -> V={cur_V}",
                              flush=True)
    return P, cur_score, cur_V


# ---------------------------------------------------------------------------
# sliver repair: reshape extreme-aspect soft blocks toward square
# ---------------------------------------------------------------------------

def _fix_slivers(ctx: _Ctx, P: np.ndarray, cur_score: float, cur_V: int,
                 kind: List[int], areas: List[float], t_end: float,
                 threshold: float = 4.0, max_fix: int = 24):
    """The column channel quantizes soft widths to column widths, leaving
    long slivers (up to 100:1 on the worst cases) that fragment the
    packing, inflate the bbox and stretch nets.  Worst-first: shrink the
    sliver along its long axis (centered, conflict-free) and shove
    neighbors along the short axis to make room for the regrown side;
    exact-score accept (equal violations allowed — this is a shape/score
    fixer)."""
    opt = ctx.opt
    n = len(P)
    coded = set(int(i) for i in opt._bnd_idx)
    fixed = 0
    order = sorted(range(n),
                   key=lambda i: -max(P[i, 2] / P[i, 3], P[i, 3] / P[i, 2]))
    for i in order:
        if time.time() >= t_end or fixed >= max_fix:
            break
        w0, h0 = float(P[i, 2]), float(P[i, 3])
        asp = max(w0 / h0, h0 / w0)
        if asp <= threshold:
            break
        if kind[i] != 0 or i in coded:
            continue
        a = float(areas[i])
        s = math.sqrt(a)
        r2 = math.sqrt(2.0)
        long_axis = 0 if w0 > h0 else 1
        short_axis = 1 - long_axis
        X0, Y0, X1, Y1 = _bbox(P)
        wall_lo, wall_hi = (Y0, Y1) if short_axis == 1 else (X0, X1)
        rigid = np.array([k == 2 for k in kind], dtype=bool)
        for bi_, bc_ in zip(opt._bnd_idx, opt._bnd_codes):
            if short_axis == 1 and int(bc_) & 12:
                rigid[int(bi_)] = True
            if short_axis == 0 and int(bc_) & 3:
                rigid[int(bi_)] = True
        cands: List[np.ndarray] = []
        for tw, th in ((s, s), (s * r2, s / r2), (s / r2, s * r2)):
            la_new = tw if long_axis == 0 else th
            sa_new = th if long_axis == 0 else tw
            if sa_new <= float(P[i, 2 + short_axis]) + 1e-9:
                continue   # not growing on the short axis: pointless
            Q0 = P.copy()
            Q0[i, long_axis] = (P[i, long_axis]
                                + 0.5 * (P[i, 2 + long_axis] - la_new))
            Q0[i, 2] = tw
            Q0[i, 3] = th
            lo_b = float(Q0[i, long_axis])
            hi_b = lo_b + la_new
            t0 = float(P[i, short_axis]) \
                - 0.5 * (sa_new - float(P[i, 2 + short_axis]))
            Q = _shove_insert(Q0, i, short_axis, wall_lo, wall_hi,
                              lo_b, hi_b, t0, rigid)
            if Q is not None:
                cands.append(Q)
        best = _try_candidates(ctx, P, cur_score, cur_V, cands, t_end,
                               allow_equal_v=True)
        if best is not None:
            cur_score, cur_V = best[0], best[1]
            P = best[2]
            fixed += 1
            if os.environ.get("VKILL_DEBUG"):
                print(f"[vkill] sliver fix block {i} asp {asp:.1f} -> "
                      f"score {cur_score:.5f}", flush=True)
    return P, cur_score, cur_V


# ---------------------------------------------------------------------------
# stage 3: large-neighborhood search (ruin and recreate)
# ---------------------------------------------------------------------------

def _lns_pass(ctx: _Ctx, P: np.ndarray, cur_score: float, cur_V: int,
              kind: List[int], areas: List[float], t_end: float,
              rounds: int = 6):
    """Rip the region around the block farthest from its net pull and
    re-place the ripped blocks inside the freed hole, positions chosen by
    net pull (weighted neighbor/pin centroid).  This is the move class the
    pipeline lacks entirely: the SA moves one unit, the refiner applies
    local ops — neither can re-arrange ten blocks jointly.  Exact-cost
    accept, deterministic seeds (round r rips the r-th worst region)."""
    from frame_repack_claude import _split_free, _prune
    opt = ctx.opt
    n = len(P)
    coded = set(int(i) for i in opt._bnd_idx)

    def desired(Pc):
        cx = Pc[:, 0] + 0.5 * Pc[:, 2]
        cy = Pc[:, 1] + 0.5 * Pc[:, 3]
        sw = np.full(n, 1e-9)
        sx = np.zeros(n)
        sy = np.zeros(n)
        np.add.at(sx, opt.eI, opt.eW * cx[opt.eJ])
        np.add.at(sy, opt.eI, opt.eW * cy[opt.eJ])
        np.add.at(sw, opt.eI, opt.eW)
        np.add.at(sx, opt.eJ, opt.eW * cx[opt.eI])
        np.add.at(sy, opt.eJ, opt.eW * cy[opt.eI])
        np.add.at(sw, opt.eJ, opt.eW)
        np.add.at(sx, opt.pB, opt.pW * opt.pX)
        np.add.at(sy, opt.pB, opt.pW * opt.pY)
        np.add.at(sw, opt.pB, opt.pW)
        return sx / sw, sy / sw, sw

    med = math.sqrt(float(np.median([areas[i] for i in range(n)])))
    for r in range(rounds):
        if time.time() >= t_end - 0.1:
            break
        cx = P[:, 0] + 0.5 * P[:, 2]
        cy = P[:, 1] + 0.5 * P[:, 3]
        dx, dy, wsum = desired(P)
        # connectivity-driven ruin: rip the endpoints of the heaviest
        # stretched edges (they live in DIFFERENT regions of the die, so
        # the freed holes let blocks jump across the die toward their
        # partners — the long-range swap no single-unit move can express)
        if len(opt.eI) == 0:
            break
        stretch = opt.eW * (np.abs(cx[opt.eI] - cx[opt.eJ])
                            + np.abs(cy[opt.eI] - cy[opt.eJ]))
        eorder = np.argsort(-stretch)
        rip_l: List[int] = []
        seen_e = 0
        for eidx in eorder[r::max(1, rounds // 2)]:
            for b in (int(opt.eI[eidx]), int(opt.eJ[eidx])):
                if kind[b] != 2 and b not in coded and b not in rip_l:
                    rip_l.append(b)
            seen_e += 1
            if len(rip_l) >= 10 or seen_e >= 8:
                break
        region = rip_l
        if len(region) < 3:
            continue
        rip = set(region)
        # holes = exactly the ripped footprints (disjoint by legality)
        free = [(float(P[i, 0]), float(P[i, 1]),
                 float(P[i, 2]), float(P[i, 3])) for i in region]
        Q = P.copy()
        ok = True
        for i in sorted(region, key=lambda i: -areas[i]):
            a = float(areas[i])
            best = None
            for fx, fy, fw, fh in free:
                cands = [(float(P[i, 2]), float(P[i, 3]))]
                if kind[i] == 0:
                    for wf in (fw, a / fh if fh > 0 else 0.0):
                        if wf <= 0:
                            continue
                        hf = a / wf
                        if max(wf / hf, hf / wf) <= MAX_ASPECT:
                            cands.append((wf, hf))
                for w2, h2 in cands:
                    if w2 > fw + 1e-9 or h2 > fh + 1e-9:
                        continue
                    x2 = min(max(dx[i] - 0.5 * w2, fx), fx + fw - w2)
                    y2 = min(max(dy[i] - 0.5 * h2, fy), fy + fh - h2)
                    d = abs(x2 + 0.5 * w2 - dx[i]) + abs(y2 + 0.5 * h2 - dy[i])
                    if best is None or d < best[0]:
                        best = (d, x2, y2, w2, h2)
            if best is None:
                ok = False
                break
            _d, x2, y2, w2, h2 = best
            Q[i] = (x2, y2, w2, h2)
            nxt = []
            for fr in free:
                nxt.extend(_split_free(fr, (x2, y2, w2, h2)))
            free = _prune(nxt)
        if not ok or not _overlap_ok(Q):
            continue
        s2, V2 = ctx.score(Q)
        if V2 <= cur_V and s2 < cur_score - 1e-12:
            P, cur_score, cur_V = Q, s2, V2
            if os.environ.get("VKILL_DEBUG"):
                print(f"[vkill] LNS round {r} accepted: score={s2:.5f} "
                      f"V={V2} region={len(region)}", flush=True)
    return P, cur_score, cur_V


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _final_guards_ok(opt, P0: np.ndarray, P: np.ndarray,
                     kind: List[int], areas: List[float]) -> bool:
    if P.shape != P0.shape or not np.isfinite(P).all():
        return False
    if (P[:, 2] <= 0).any() or (P[:, 3] <= 0).any():
        return False
    for i in range(len(P)):
        if kind[i] == 2:
            if not np.array_equal(P[i], P0[i]):
                return False
        elif kind[i] == 1:
            if P[i, 2] != P0[i, 2] or P[i, 3] != P0[i, 3]:
                return False
        else:
            if abs(float(P[i, 2] * P[i, 3]) - areas[i]) \
                    > MIB_AREA_GUARD * areas[i]:
                return False
    return _overlap_ok(P)


def _kill(out: List[Rect], area_targets, constraints, target_positions,
          b2b, p2b, pins, budget: float, verbose: bool):
    t_end = time.time() + max(budget, 0.2)
    opt = _ColumnOptimizer(out, area_targets, constraints, target_positions,
                           b2b, p2b, pins, deadline=t_end, seed=0)
    P0 = np.asarray([[r[0], r[1], r[2], r[3]] for r in out], dtype=np.float64)
    V0 = _violations_exact(opt, P0)
    if V0 <= 0 and not (os.environ.get("VKILL_LNS")
                        or os.environ.get("VKILL_SLIVER")):
        return None   # nothing to fix (LNS/sliver also work on clean ones)
    ctx = _Ctx(opt, P0)
    kind = list(opt.kind)
    areas = list(opt.areas)
    P = P0.copy()
    cur_score, cur_V = ctx.score(P)
    s0 = cur_score
    debug = bool(verbose or os.environ.get("VKILL_DEBUG"))

    for _ in range(4):
        if cur_V <= 0 or time.time() >= t_end:
            break
        before = cur_V
        P, cur_score, cur_V = _fix_boundary(ctx, P, cur_score, cur_V,
                                            kind, areas, t_end)
        P, cur_score, cur_V = _fix_grouping(ctx, P, cur_score, cur_V,
                                            kind, areas, t_end)
        P, cur_score, cur_V = _fix_mib(ctx, P, cur_score, cur_V,
                                       kind, areas, t_end)
        if cur_V >= before:
            break

    # sliver repair (VKILL_SLIVER=1): reshape extreme-aspect soft blocks
    # toward square with shove-made room; exact-score accept.
    if os.environ.get("VKILL_SLIVER") and time.time() < t_end - 0.2:
        try:
            P, cur_score, cur_V = _fix_slivers(ctx, P, cur_score, cur_V,
                                               kind, areas, t_end)
        except Exception:
            if debug:
                traceback.print_exc()

    # stage 3 (VKILL_LNS=1): ruin-and-recreate around the worst net-pull
    # regions — joint re-arrangement of ~10 blocks, the move class both
    # the SA and the refiner lack; exact-cost accept.
    if os.environ.get("VKILL_LNS") and time.time() < t_end - 0.3:
        try:
            P, cur_score, cur_V = _lns_pass(ctx, P, cur_score, cur_V,
                                            kind, areas, t_end)
        except Exception:
            if debug:
                traceback.print_exc()

    # stage 2: violation-weighted deep re-refinement through the partner's
    # own refine_prediction pipeline (v_weight > 1 biases its search toward
    # killing violations); best-of by the evaluator-faithful proxy, never
    # accepting a violation increase.
    n_blocks = len(out)
    # Time-neutral: stage 2 lives inside the same carved budget (t_end).
    # VKILL_STAGE2_BUDGET (default 0) exists only so offline headroom
    # probes can extend it; production configs must leave it at 0.
    if (cur_V > 0
            and n_blocks >= int(_envf("VKILL_STAGE2_MIN_N", 60.0))
            and not os.environ.get("VKILL_STAGE2_OFF")):
        t_end2 = t_end + _envf("VKILL_STAGE2_BUDGET", 0.0)
        try:
            from refiner_claude import refine_prediction
            for vw, seed in ((3.0, 7), (2.0, 11), (4.0, 13),
                             (3.0, 17), (2.0, 19)):
                left = t_end2 - time.time()
                if left < 3.0:
                    break
                sub_end = time.time() + min(left - 0.5, 10.0)
                opt2 = _ColumnOptimizer(
                    [tuple(map(float, r)) for r in P],
                    area_targets, constraints, target_positions,
                    b2b, p2b, pins, deadline=sub_end, seed=seed,
                    v_weight=vw)
                res = refine_prediction(opt2, P.copy(), sub_end, seed=seed)
                if res is None:
                    continue
                Q = np.asarray(res, dtype=np.float64)
                if Q.shape != P.shape or not np.isfinite(Q).all():
                    continue
                if not _overlap_ok(Q):
                    continue
                s, V = ctx.score(Q)
                if V <= cur_V and s < cur_score - 1e-12:
                    P, cur_score, cur_V = Q, s, V
                    if debug:
                        print(f"[vkill] stage2 vw={vw} seed={seed} "
                              f"accepted: V={V} score={s:.5f}", flush=True)
        except Exception:
            if debug:
                traceback.print_exc()

    if cur_V > V0 or cur_score >= s0 - 1e-12:
        return None
    if not _final_guards_ok(opt, P0, P, kind, areas):
        return None
    if debug:
        print(f"[vkill] V {V0} -> {cur_V} "
              f"(hp {ctx.hp0:.1f} -> {float(opt._hpwl(P)):.1f})", flush=True)
    return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in P]
