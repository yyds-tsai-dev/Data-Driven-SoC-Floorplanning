"""PARTNER_TAG_COMPRESS: directional compaction onto preplaced tag lines.

A preplaced block is hard-locked in place, so a boundary tag it carries can
only be satisfied if the FINAL bbox edge on that side is exactly the block's
own edge.  Nothing downstream of the legalizer enforces that: the shipped
n>=100 layouts overshoot such a line by 0.16%-2.5% of the span, and the tag
is charged even though the block itself is exactly where the instance asked
for it.  `_edge_seat`'s outlier pull is the nearest existing move, but it is
a RIGID translate of at most 8 blocks onto the shelf, and the blocks that
cross these lines sit in packed chains 6-16 deep.

This pass pushes the whole chain instead, as a longest-path solve on the
layout's own separation DAG:

  * the target line is a reusable instance statistic -- a preplaced block's
    own edge on a side it is tagged for -- never a case id;
  * only blocks CROSSING the line move, and only INWARD (one-directional),
    so a seated tag on the same side rides the edge in and stays seated and
    the opposite edge never moves;
  * every original separation ordering between blocks that overlap on the
    other axis is re-imposed, so the result is OVERLAP-FREE BY CONSTRUCTION
    (re-verified anyway before commit);
  * preplaced blocks never move -- a chain that would have to move one
    ABORTS the whole push, which is why this can be run unconditionally;
  * shapes only ever shrink, and only for soft non-MIB blocks, by at most
    `ATOL` of the target area along the push axis.  That 1% tolerance is a
    real unused degree of freedom and it is what makes the pass work at all:
    with rigid shapes NONE of the shipped tag lines is reachable, because
    each chain misses by 0.02-0.7 units on a 130-270-unit span.

Committed only when the evaluator's own boundary+grouping+MIB total strictly
drops, or when it is unchanged and the bbox strictly shrinks without HPWL
regressing.  Both branches are monotone in the official cost.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

EPS = 1e-6
ATOL = 0.008          # area given up along the push axis (hard limit is 1%)
OVL_TOL = 1e-7        # overlap re-verification tolerance
SEP_TOL = 1e-9        # "these two blocks overlap on the other axis"


def _compress(P: np.ndarray, areas: np.ndarray, locked: np.ndarray,
              shrinkable: Optional[np.ndarray], ax: int, side: int,
              line: float, freeze: Optional[np.ndarray] = None
              ) -> Optional[np.ndarray]:
    """One directional push of everything crossing `line` back onto it.

    `side` 1 = the max edge of `ax` (push toward decreasing), 0 = the min
    edge (push toward increasing).  Returns a new (n, 4) array, or None when
    the push is infeasible.
    """
    n = P.shape[0]
    o = 1 - ax
    # mirror so the push is always toward DECREASING coordinate
    if side:
        u = P[:, ax].copy()
        target = float(line)
        far_wall = float(P[:, ax].min())
    else:
        u = -(P[:, ax] + P[:, ax + 2])
        target = -float(line)
        far_wall = -float((P[:, ax] + P[:, ax + 2]).max())
    w = P[:, ax + 2].copy()
    hi = u + w
    olo = P[:, o]
    ohi = P[:, o] + P[:, o + 2]

    frozen = np.asarray(locked, dtype=bool).copy()
    if freeze is not None:
        frozen |= np.asarray(freeze, dtype=bool)

    # minimum axis dimension each block may be squeezed to
    m = w.copy()
    if shrinkable is not None:
        od = P[:, o + 2]
        wmin = np.where(od > SEP_TOL,
                        (1.0 - ATOL) * areas / np.maximum(od, SEP_TOL), w)
        ok = np.asarray(shrinkable, dtype=bool) & ~frozen
        m = np.where(ok, np.minimum(w, np.maximum(wmin, 0.0)), w)
        m = np.maximum(m, EPS)

    upper = hi.copy()                      # upper bound on each high edge
    need = hi > target + EPS
    if not need.any():
        return None
    upper[need] = target

    # reverse topological order: i before j implies hi_i <= u_j < hi_j
    for j in np.argsort(-hi, kind="stable"):
        if frozen[j] and upper[j] < hi[j] - SEP_TOL:
            return None                    # chain reached an immovable block
        squeezed = upper[j] - m[j]
        if squeezed < far_wall - SEP_TOL:
            return None                    # chain would cross the far wall
        ov = (np.minimum(ohi, ohi[j]) - np.maximum(olo, olo[j])) > SEP_TOL
        pred = ov & (hi <= u[j] + SEP_TOL)
        pred[j] = False
        if pred.any():
            upper[pred] = np.minimum(upper[pred], squeezed)

    # assign dimensions: `upper` is final, so one pass suffices.  A block
    # only gives up area where its predecessor really is pressed into it.
    Q = P.copy()
    for j in range(n):
        ov = (np.minimum(ohi, ohi[j]) - np.maximum(olo, olo[j])) > SEP_TOL
        pred = ov & (hi <= u[j] + SEP_TOL)
        pred[j] = False
        room = upper[j] - (float(upper[pred].max()) if pred.any() else far_wall)
        wj = max(min(float(w[j]), room), float(m[j]))
        if upper[j] >= hi[j] - 1e-12 and wj >= float(w[j]) - 1e-12:
            continue                       # untouched: keep bit-identical
        if side:
            Q[j, ax] = upper[j] - wj
        else:
            Q[j, ax] = -upper[j]
        Q[j, ax + 2] = wj

    # one-directional guard (belt and braces; both hold by construction)
    if side:
        if (Q[:, ax] > P[:, ax] + SEP_TOL).any():
            return None
    else:
        if (Q[:, ax] < P[:, ax] - SEP_TOL).any():
            return None
    Q[locked] = P[locked]
    return Q


def _tag_lines(P: np.ndarray, locked: np.ndarray, boundary: Sequence[int],
               n: int):
    """(axis, side, line, block) for every preplaced tag whose own edge the
    current bbox overshoots and which no OTHER locked block crosses.

    The second test is the structural-deadlock filter: when two preplaced
    blocks carry the same tag on different lines only the outermost can ever
    be the wall (pigeonhole), and this keeps exactly that one.
    """
    lk = np.nonzero(np.asarray(locked, dtype=bool))[0]
    if not len(lk):
        return []
    lo = (float(P[:, 0].min()), float(P[:, 1].min()))
    hi = (float((P[:, 0] + P[:, 2]).max()), float((P[:, 1] + P[:, 3]).max()))
    out = []
    for i in lk:
        code = int(boundary[i])
        if not code:
            continue
        for bit, ax, side in ((1, 0, 0), (2, 0, 1), (4, 1, 1), (8, 1, 0)):
            if not (code & bit):
                continue
            e = float(P[i, ax] + (P[i, ax + 2] if side else 0.0))
            if abs(e - (hi[ax] if side else lo[ax])) < EPS:
                continue                   # already seated
            other = P[lk, ax] + (P[lk, ax + 2] if side else 0.0)
            beyond = (other > e + EPS) if side else (other < e - EPS)
            if bool(beyond.any()):
                continue                   # structurally dead line
            out.append((ax, side, e, int(i)))
    return out


def _has_overlap(P: np.ndarray) -> bool:
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    cx = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    cy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    hit = (cx > OVL_TOL) & (cy > OVL_TOL)
    np.fill_diagonal(hit, False)
    return bool(hit.any())


def tag_compress(opt, out, viol_fn=None, hpwl_fn=None) -> List[tuple]:
    """Compact `out` onto every reachable preplaced tag line.

    Pure: returns the input list unchanged on any guard failure, on any
    exception, and whenever no push clears the acceptance test.
    """
    try:
        n = int(opt.n)
        P0 = np.asarray([tuple(map(float, r)) for r in out], dtype=np.float64)
        if P0.shape != (n, 4) or n < 2:
            return out
        kind = np.asarray(opt.kind, dtype=np.int64)
        locked = kind == 2
        if not locked.any():
            return out
        boundary = [int(b) for b in opt.boundary]
        lines = _tag_lines(P0, locked, boundary, n)
        if not lines:
            return out

        areas = np.asarray([float(a) for a in opt.areas], dtype=np.float64)
        cluster = np.asarray([int(c) for c in opt.cluster], dtype=np.int64)
        mib = np.asarray([int(mv) for mv in opt.mib], dtype=np.int64)
        # soft, non-MIB, non-fixed, non-preplaced blocks own the area slack
        shrink = (kind == 0) & (mib <= 0)

        if viol_fn is None:
            from violation_killer import _violations_exact as viol_fn
        if hpwl_fn is None:
            hpwl_fn = opt._hpwl

        def _bbox(P):
            return float(((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                         * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))

        cur = P0
        v_cur = viol_fn(opt, cur)
        a_cur = _bbox(cur)
        h_cur = float(hpwl_fn(cur))
        moved = False

        # two passes: a corner tag needs BOTH of its axes, and the second
        # axis only becomes reachable once the first has been pulled in
        for _pass in range(2):
            for (ax, side, line, _blk) in lines:
                best = None
                for freeze in (None, cluster > 0):
                    for sh in (shrink, None):
                        Q = _compress(cur, areas, locked, sh, ax, side, line,
                                      freeze=freeze)
                        if Q is None:
                            continue
                        # hard legality: shapes, areas, overlaps
                        if not np.array_equal(Q[kind != 0, 2:],
                                              cur[kind != 0, 2:]):
                            continue
                        soft = kind == 0
                        rel = np.abs(Q[soft, 2] * Q[soft, 3] - areas[soft]) \
                            / np.maximum(areas[soft], EPS)
                        if soft.any() and float(rel.max()) > 0.01 - 1e-9:
                            continue
                        if _has_overlap(Q):
                            continue
                        v = viol_fn(opt, Q)
                        a = _bbox(Q)
                        if v > v_cur:
                            continue
                        if v == v_cur and not (a < a_cur - EPS
                                               and float(hpwl_fn(Q))
                                               <= h_cur + EPS):
                            continue
                        if best is None or v < best[1] or (
                                v == best[1] and a < best[2] - EPS):
                            best = (Q, v, a)
                if best is not None:
                    cur, v_cur, a_cur = best
                    h_cur = float(hpwl_fn(cur))
                    moved = True

        if not moved:
            return out
        return [tuple(map(float, r)) for r in cur]
    except Exception:
        return out
