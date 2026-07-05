"""Noise-tolerant order-repair core for the Track-B exact decoder (iteration 2).

The E1 decoder (gen_decoder_probe.decode) is order-faithful and overlap-free by
construction WHEN the pairwise order is pin-consistent. With noisy model hints
(GNN 0514) it is not: measured on all 100 validation cases, 87/100 decode to an
illegal layout, and in EVERY illegal case 100% of the residual overlaps involve
a PREPLACED block. The failure is a transitive over-constraint: a soft block's
pin-incident order edge is set correctly, but the soft-soft chain upstream of it
forces its coordinate PAST the immovable pin's ceiling, so longest-path
compaction leaves it overlapping the pin.

This module provides `repair_pin_order`: a monotone, acyclicity-guarded,
lock-based local repair over the separation DAGs. For the worst residual
pin-overlap it tries all four candidate separations (x/y x two directions),
keeps the acyclic one that minimizes total pin penetration, and LOCKS that pair
so it is never revisited (guaranteeing termination in <= 4*npre iterations and
eliminating the oscillation that unlocked greedy exhibits). Measured: 87 -> 18
illegal, mean 3 iterations, <=290 ms on n>=100.

Pin geometry, not noisy hints, is authoritative for pin-incident relations: the
preplaced block's TRUE rect (from target_positions) is what the compaction pins
to, and the repair's direction choice is derived from realized post-compaction
geometry (which has already settled the soft-soft chain), not from the hint.

Imports only from gen_decoder_probe (sibling probe) and the stdlib; no src/
edits. Safe to import src freely per the iteration brief.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from gen_decoder_probe import (
    _golden_rects,
    _opt_target_positions,
    _parse_constraints,
    _target,
    build_order_dags,
    decode_shapes,
    longest_path_coords,
)

Rect = Tuple[float, float, float, float]
Edge = Tuple[int, int, float]
SEP_TOL = 1e-6


def _key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _overlap(a: Rect, b: Rect) -> Tuple[float, float]:
    ox = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ox, oy


def _succ(edges: Sequence[Edge]) -> Dict[int, List[int]]:
    s: Dict[int, List[int]] = defaultdict(list)
    for u, v, _g in edges:
        s[u].append(v)
    return s


def _reachable(succ: Dict[int, List[int]], src: int, dst: int) -> bool:
    """True iff dst reachable from src (used to reject cycle-creating edits)."""
    if src == dst:
        return True
    seen: Set[int] = {src}
    dq = deque([src])
    while dq:
        c = dq.popleft()
        for nx in succ.get(c, ()):  # noqa: B905
            if nx == dst:
                return True
            if nx not in seen:
                seen.add(nx)
                dq.append(nx)
    return False


def _pin_overlaps(
    rects: Sequence[Rect], pre: Sequence[int], n: int
) -> List[Tuple[int, int, float, float]]:
    """All (p, k, ox, oy) where preplaced p overlaps block k (ox,oy > tol)."""
    out: List[Tuple[int, int, float, float]] = []
    for p in pre:
        rp = rects[p]
        for k in range(n):
            if k == p:
                continue
            ox, oy = _overlap(rp, rects[k])
            if ox > SEP_TOL and oy > SEP_TOL:
                out.append((p, k, ox, oy))
    return out


def _total_pin_pen(rects: Sequence[Rect], pre: Sequence[int], n: int) -> float:
    return sum(min(ox, oy) for _p, _k, ox, oy in _pin_overlaps(rects, pre, n))


def repair_pin_order(
    x_edges: List[Edge],
    y_edges: List[Edge],
    ws: List[float],
    hs: List[float],
    x_pins: Dict[int, float],
    y_pins: Dict[int, float],
    preplaced: Sequence[bool],
    n: int,
    max_iter: Optional[int] = None,
) -> Tuple[Optional[List[Rect]], Dict[str, object]]:
    """Lock-based monotone pin-order repair.

    Returns (rects, trace). rects is None only if compaction cycles (should not
    happen: every accepted edit is acyclicity-checked) or the residual is
    unresolvable (trace['status'] in {'all_locked','noacyclic','cap'}); on
    success trace['status']=='ok'. The caller decides whether to accept the
    (possibly still-illegal) partial or fall back.
    """
    pre = [i for i in range(n) if preplaced[i]]
    trace: Dict[str, object] = {"status": "ok", "iters": 0, "repairs": 0}
    if not pre:
        xc = longest_path_coords(n, x_edges, x_pins)
        yc = longest_path_coords(n, y_edges, y_pins)
        if xc is None or yc is None:
            trace["status"] = "cycle"
            return None, trace
        return [(xc[i], yc[i], ws[i], hs[i]) for i in range(n)], trace

    xe = list(x_edges)
    ye = list(y_edges)
    locked: Set[Tuple[int, int]] = set()
    cap = max_iter if max_iter is not None else 4 * len(pre) + 8

    for it in range(cap):
        trace["iters"] = it + 1
        xc = longest_path_coords(n, xe, x_pins)
        yc = longest_path_coords(n, ye, y_pins)
        if xc is None or yc is None:
            trace["status"] = "cycle"
            return None, trace
        rects = [(xc[i], yc[i], ws[i], hs[i]) for i in range(n)]
        confl = _pin_overlaps(rects, pre, n)
        if not confl:
            trace["status"] = "ok"
            trace["rects"] = None  # placeholder; caller uses returned rects
            return rects, trace
        # Repair the WORST unlocked conflict (largest min-penetration).
        unlocked = [c for c in confl if _key(c[0], c[1]) not in locked]
        if not unlocked:
            trace["status"] = "all_locked"
            return rects, trace  # still-illegal partial; caller decides
        unlocked.sort(key=lambda c: -min(c[2], c[3]))
        p, k, _ox, _oy = unlocked[0]
        kk = _key(p, k)
        xe2 = [e for e in xe if _key(e[0], e[1]) != kk]
        ye2 = [e for e in ye if _key(e[0], e[1]) != kk]
        # Four candidate separations for the pin pair (p,k).
        cands: List[Tuple[str, int, int, float]] = [
            ("x", p, k, ws[p]),  # k on high-x side of pin
            ("x", k, p, ws[k]),  # k on low-x side of pin
            ("y", p, k, hs[p]),  # k on high-y side of pin
            ("y", k, p, hs[k]),  # k on low-y side of pin
        ]
        best: Optional[Tuple[float, List[Edge], List[Edge]]] = None
        for ax, u, v, g in cands:
            base = xe2 if ax == "x" else ye2
            if _reachable(_succ(base), v, u):
                continue  # adding u->v would close a cycle
            if ax == "x":
                txe, tye = xe2 + [(u, v, g)], ye2
            else:
                txe, tye = xe2, ye2 + [(u, v, g)]
            tx = longest_path_coords(n, txe, x_pins)
            ty = longest_path_coords(n, tye, y_pins)
            if tx is None or ty is None:
                continue
            tr = [(tx[i], ty[i], ws[i], hs[i]) for i in range(n)]
            pen = _total_pin_pen(tr, pre, n)
            if best is None or pen < best[0] - 1e-9:
                best = (pen, txe, tye)
        if best is None:
            trace["status"] = "noacyclic"
            return rects, trace
        xe, ye = best[1], best[2]
        locked.add(kk)
        trace["repairs"] = int(trace["repairs"]) + 1

    trace["status"] = "cap"
    xc = longest_path_coords(n, xe, x_pins)
    yc = longest_path_coords(n, ye, y_pins)
    if xc is None or yc is None:
        trace["status"] = "cycle"
        return None, trace
    return [(xc[i], yc[i], ws[i], hs[i]) for i in range(n)], trace


# ---------------------------------------------------------------------------
# Aspect repair (item 3): the residual `all_locked` conflicts after the order
# repair are NOT order failures -- they are SHAPE failures. Measured on the 18
# residual cases: the trapped soft block's decoded aspect (from the noisy hint)
# is wrong (e.g. golden 8x23 tall-thin, hint-decoded 11x16 near-square), so the
# fatter footprint physically cannot clear the pin at any ordering. Reshaping
# the trapped block within its EXACT AREA (log-aspect stays legal: hard_legal's
# area check is on w*h) to be thin on the penetration axis resolves 16/18.
#
# This is golden-free: the penetration axis and the available gap come from the
# realized (post-order-repair) geometry, not from golden. Best-of-k over a small
# aspect ladder, keeping the legal candidate with the lowest HPWL so we don't
# pay a sliver penalty when a milder aspect already fits.
# ---------------------------------------------------------------------------

# log-aspect ladder, symmetric, ordered mild -> extreme so the first legal fit
# is also the least distorted (we still scan all for HPWL, but this documents
# intent). Clamp matches column_slicing LOG_ASPECT_CLAMP == 3.0.
# Magnitudes only: the sign (which axis to thin) is chosen from the penetration
# axis, so the ladder needs positive magnitudes and 0.0 (the original shape) as
# the "no-distortion first" option. Kept short (5 rungs) to bound the search.
_ASPECT_LADDER = [0.0, 0.7, 1.3, 2.0, 2.7]

# Cap trapped blocks reshaped per round (budget guard; see repair_pin_shapes).
MAX_RESHAPE_PER_ROUND = 6


def _shape_for(area: float, log_aspect: float) -> Tuple[float, float]:
    e = math.exp(log_aspect)
    return math.sqrt(area * e), math.sqrt(area / e)


def repair_pin_shapes(
    hints,
    x_edges: List[Edge],
    y_edges: List[Edge],
    ws: List[float],
    hs: List[float],
    x_pins: Dict[int, float],
    y_pins: Dict[int, float],
    preplaced: Sequence[bool],
    fixed: Sequence[bool],
    mib: Sequence[int],
    area_targets: Sequence[float],
    n: int,
    hpwl_fn=None,
    hpwl_args: Tuple = (),
    max_rounds: int = 4,
) -> Tuple[Optional[List[Rect]], Dict[str, object]]:
    """Order repair, then aspect repair for the residual pin conflicts.

    `hints` is only used to REBUILD the order DAG after a reshape (a reshape
    changes gaps, so the DAG must be regenerated from the same hint centroids).
    `ws`/`hs` are the initial decoded shapes; they are copied, never mutated in
    place for the caller. Returns (rects, trace); rects may still be illegal if
    a block is genuinely un-fittable, in which case the caller falls back.
    """
    ws = list(ws)
    hs = list(hs)
    trace: Dict[str, object] = {"reshaped": 0, "aspect_rounds": 0}

    rects, rtr = repair_pin_order(
        x_edges, y_edges, ws, hs, x_pins, y_pins, preplaced, n
    )
    trace["order_status"] = rtr.get("status")
    trace["order_iters"] = rtr.get("iters")
    trace["order_edits"] = rtr.get("repairs")
    if rects is None:
        trace["status"] = "order_" + str(rtr.get("status"))
        return None, trace

    pre = [i for i in range(n) if preplaced[i]]
    confl = _pin_overlaps(rects, pre, n)
    if not confl:
        trace["status"] = "ok"
        return rects, trace

    # cx/cy from hints for DAG rebuilds after reshape.
    from gen_decoder_probe import build_order_dags as _bod

    for rnd in range(max_rounds):
        confl = _pin_overlaps(rects, pre, n)
        if not confl:
            break
        trace["aspect_rounds"] = rnd + 1
        # collect reshapeable trapped blocks (skip preplaced/fixed/mib: their
        # shape is exact / group-shared and must not change).
        pen_by_k: Dict[int, Tuple[float, float]] = {}
        for p, k, ox, oy in confl:
            if preplaced[k] or fixed[k] or (mib[k] > 0):
                continue
            px, py = pen_by_k.get(k, (0.0, 0.0))
            pen_by_k[k] = (px + ox, py + oy)
        if not pen_by_k:
            break  # only un-reshapeable blocks remain
        changed = False
        # Budget guard: the aspect search runs a full order-repair per candidate
        # aspect per trapped block; on large n that is the dominant cost. Cap the
        # number of trapped blocks reshaped per round to the WORST few (by total
        # penetration) so a pathological case cannot blow the per-case budget.
        ranked_k = sorted(
            pen_by_k.items(), key=lambda kv: -(kv[1][0] + kv[1][1])
        )[:MAX_RESHAPE_PER_ROUND]
        for k, (sumox, sumoy) in ranked_k:
            area = float(area_targets[k])
            thin_axis_x = sumox >= sumoy  # penetration worse on x -> thin width
            best_shape = None
            best_hpwl = None
            for la in _ASPECT_LADDER:
                # sign convention: log_aspect>0 -> wider (bigger w). To thin x we
                # want small w -> negative log_aspect; to thin y -> positive.
                lav = -abs(la) if thin_axis_x else abs(la)
                w, h = _shape_for(area, lav)
                tws = list(ws)
                ths = list(hs)
                tws[k] = w
                ths[k] = h
                xe2, ye2 = _bod(hints, tws, ths, n)
                r2, tr2 = repair_pin_order(
                    xe2, ye2, tws, ths, x_pins, y_pins, preplaced, n
                )
                if r2 is None:
                    continue
                if _pin_overlaps(r2, pre, n):
                    continue  # still conflicts under this aspect
                cost = (
                    hpwl_fn(r2, *hpwl_args) if hpwl_fn is not None else 0.0
                )
                if best_hpwl is None or cost < best_hpwl:
                    best_hpwl = cost
                    best_shape = (w, h)
            if best_shape is not None:
                ws[k], hs[k] = best_shape
                trace["reshaped"] = int(trace["reshaped"]) + 1
                changed = True
        # rebuild with all accepted reshapes committed.
        xe2, ye2 = _bod(hints, ws, hs, n)
        r2, tr2 = repair_pin_order(
            xe2, ye2, ws, hs, x_pins, y_pins, preplaced, n
        )
        if r2 is not None:
            rects = r2
        if not changed:
            break

    trace["status"] = "ok" if not _pin_overlaps(rects, pre, n) else "residual"
    return rects, trace


def evict_overlaps(
    rects: List[Rect], preplaced: Sequence[bool], n: int
) -> Tuple[List[Rect], int]:
    """Non-destructive legality floor for residual cases: keep the whole
    repaired decode and only PARK the soft blocks still overlapping anything
    onto a shelf above the current bbox. Measured on the 6 residual GNN cases
    this scores 3.7-6.8 vs the shelf-pack's flat 10.0, because it preserves the
    ~95% of blocks that decoded well.

    Only soft (non-preplaced) blocks are evicted; preplaced pins never move.
    Returns (rects, n_evicted). Not guaranteed legal if two evicted blocks are
    huge relative to the shelf width, but the shelf is overlap-free by
    construction (row packing) and preplaced blocks are below the shelf, so it
    is legal whenever the kept sub-layout was.
    """
    r = [list(x) for x in rects]
    bad: Set[int] = set()
    for i in range(n):
        for j in range(i + 1, n):
            ox, oy = _overlap(rects[i], rects[j])
            if ox > SEP_TOL and oy > SEP_TOL:
                if not preplaced[i]:
                    bad.add(i)
                if not preplaced[j]:
                    bad.add(j)
    if not bad:
        return [tuple(v) for v in r], 0
    keep = [i for i in range(n) if i not in bad]
    if not keep:
        return [tuple(v) for v in r], 0
    x0 = min(r[i][0] for i in keep)
    x1 = max(r[i][0] + r[i][2] for i in keep)
    y1 = max(r[i][1] + r[i][3] for i in keep)
    x = x0
    y = y1
    row_h = 0.0
    for k in sorted(bad):
        w, h = r[k][2], r[k][3]
        if x > x0 and x + w > x1:
            x = x0
            y += row_h
            row_h = 0.0
        r[k][0] = x
        r[k][1] = y
        x += w
        row_h = max(row_h, h)
    return [tuple(v) for v in r], len(bad)
