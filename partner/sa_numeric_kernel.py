#!/usr/bin/env python3
"""Flat-array (structure-of-arrays) numba kernel for the column-SA layout pass.

Why this exists
---------------
`_ColumnOptimizer._layout` is ~82% of the SA inner-move cost (measured, see
`scratchpad/sa_split.py`), and its hot leaf `_stack_column` walks a Python
object graph: `_Unit` instances holding lists of subgroups holding lists of
`(block, is_soft, area, w, h)` tuples.  A previous attempt to `njit` that graph
in place was correctly judged infeasible (2026-07-23) -- nopython mode cannot
see through it.

This module takes the other route the 0708 roadmap named ("typed C-ext"): it
re-expresses the *same* algorithm over flat numpy arrays.  The unit -> band ->
chunk -> entry hierarchy is stored CSR-style, so every operation the layout
performs becomes a pure numeric kernel that numba can compile.

Bit-exactness contract
----------------------
The kernel is a *transcription*, not a reimplementation: every arithmetic
expression, comparison tolerance, iteration order, accumulation order,
tie-break and early-exit is reproduced from `column_sa_legalizer.py`.  There
are no vectorised reductions where the Python original accumulates
sequentially, and no fast-math.  `layout()` therefore returns bit-identical
`pos` / `x_right` / `y_top` to `_layout_full`, and `tests/test_partner_sa_kernel.py`
asserts exactly that (`==`, not `allclose`).

Scope
-----
Ported: `_layout` (the whole `_layout_full` including `_stack_column`, the
band solver, the obstacle segment logic and both global post-passes) and
`_violations`.  `_violations` is safe to port because it returns an integer
COUNT -- unlike a float sum it cannot depend on accumulation order.

NOT ported: `_hpwl`.  It is a float reduction, and `np.sum` uses pairwise
summation; a sequential loop would give a different (equally valid) result,
which would perturb every Metropolis accept and turn this toggle from a pure
speed change into a search-trajectory change.  Reproducing numpy's pairwise
kernel is possible but is a separate, evidence-gated increment.

Compile cost
------------
Cold JIT of this module is ~10 s (measured; `scratchpad/compile_cost.py`).
`inline='always'` is applied ONLY to the small hot leaves: inlining the heavy
functions too buys ~10% steady-state but costs ~161 s of compile time, which
no per-case budget can absorb.  `cache=True` cuts a warm re-run to ~0.2 s, but
numba's on-disk cache is keyed on source path/mtime and CPU features, so a
fresh machine pays the cold compile.  Because the kernel is bit-identical to
the Python path, it is safe to bring it online *mid-run*: the compile can be
warmed once in the parent process (forked SA workers then inherit the compiled
code) while `_layout` keeps serving from Python until it is ready.

Activation
----------
`PARTNER_SA_KERNEL=numba`.  Unset (the default) leaves the solver on the
existing Python path with no behavioural change whatsoever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:  # numba is an optional dependency; absence must never break the solver
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on numba-less machines
    NUMBA_AVAILABLE = False

    def njit(*a, **k):  # type: ignore[misc]
        def deco(fn):
            return fn

        return deco if not a or not callable(a[0]) else a[0]


# Maximum effective chunks per band. `_partition_subgroup` caps at 4 and
# `_dyn_split` caps at 4 and only fires when the band has a single chunk, so 4
# is the true bound; 8 leaves headroom.
_CHUNK_CAP = 8

# must match `column_sa_legalizer.TOUCH_TOL`
TOUCH_TOL = 1e-7

_TRUE = ("1", "true", "True", "on", "ON")

# Status codes returned in `outf[2]`.
_OK = 0.0
_FAIL_SEGMENT_OVERRUN = 1.0   # `si` ran past the last segment (Python raises
                              # IndexError here; we fall back so it raises)
_FAIL_CAPACITY = 2.0          # a scratch buffer was too small


# =============================================================================
# njit leaves
# =============================================================================
@njit(cache=True, inline='always')
def _interval_free(occ, n_occ, s, e):
    for t in range(n_occ):
        if s < occ[t, 1] - 1e-9 and occ[t, 0] < e - 1e-9:
            return False
    return True


@njit(cache=True)
def _add_interval(occ, n_occ, s, e):
    """occ.append([s, e]); occ.sort(); merge -- as in `_add_interval`."""
    occ[n_occ, 0] = s
    occ[n_occ, 1] = e
    n = n_occ + 1
    # stable insertion sort on (start, end): matches list.sort() on [s, e]
    for i in range(1, n):
        a0 = occ[i, 0]
        a1 = occ[i, 1]
        j = i - 1
        while j >= 0 and (occ[j, 0] > a0 or (occ[j, 0] == a0 and occ[j, 1] > a1)):
            occ[j + 1, 0] = occ[j, 0]
            occ[j + 1, 1] = occ[j, 1]
            j -= 1
        occ[j + 1, 0] = a0
        occ[j + 1, 1] = a1
    m = 1
    for i in range(1, n):
        if occ[i, 0] <= occ[m - 1, 1] + 1e-12:
            if occ[i, 1] > occ[m - 1, 1]:
                occ[m - 1, 1] = occ[i, 1]
        else:
            occ[m, 0] = occ[i, 0]
            occ[m, 1] = occ[i, 1]
            m += 1
    return m


@njit(cache=True)
def _obstacles_in(lock, x0, x1, out):
    """Merged y-intervals of locked rects spanning [x0, x1]; returns count."""
    n = 0
    for t in range(lock.shape[0]):
        ox = lock[t, 0]
        ow = lock[t, 2]
        if ox < x1 - 1e-9 and ox + ow > x0 + 1e-9:
            out[n, 0] = lock[t, 1]
            out[n, 1] = lock[t, 1] + lock[t, 3]
            n += 1
    if n == 0:
        return 0
    for i in range(1, n):
        a0 = out[i, 0]
        a1 = out[i, 1]
        j = i - 1
        while j >= 0 and (out[j, 0] > a0 or (out[j, 0] == a0 and out[j, 1] > a1)):
            out[j + 1, 0] = out[j, 0]
            out[j + 1, 1] = out[j, 1]
            j -= 1
        out[j + 1, 0] = a0
        out[j + 1, 1] = a1
    m = 1
    for i in range(1, n):
        if out[i, 0] <= out[m - 1, 1] + 1e-9:
            if out[i, 1] > out[m - 1, 1]:
                out[m - 1, 1] = out[i, 1]
        else:
            out[m, 0] = out[i, 0]
            out[m, 1] = out[i, 1]
            m += 1
    return m


@njit(cache=True)
def _band_strip(lock, x, w, lo, hi, pl_yb, pl_yt, pl_lo, pl_hi, res):
    """Widest free x-strip in [x, x+w] over [lo, hi). res <- (xoff, width).

    Returns True when a usable strip exists (Python returns None otherwise)."""
    if hi <= lo + 1e-6:
        return False
    for t in range(pl_lo, pl_hi):
        if pl_yb[t] < hi - 1e-9 and pl_yt[t] > lo + 1e-9:
            return False
    left = x + w
    right = x
    for t in range(lock.shape[0]):
        ox = lock[t, 0]
        oy = lock[t, 1]
        ow = lock[t, 2]
        oh = lock[t, 3]
        if (ox < x + w - 1e-9 and ox + ow > x + 1e-9
                and oy < hi - 1e-9 and oy + oh > lo + 1e-9):
            if ox < left:
                left = ox
            if ox + ow > right:
                right = ox + ow
    if right <= x:
        return False
    left_w = left - x
    right_w = (x + w) - right
    if right_w >= left_w:
        xoff = right
        sw = right_w
    else:
        xoff = x
        sw = left_w
    if sw < 0.30 * w or sw < 2.0:
        return False
    res[0] = xoff
    res[1] = sw
    return True


@njit(cache=True)
def _solve_band(eckf, nck, w, widths):
    """`_solve_band`: band height h with chunk widths summing to w.

    eckf[j] = (sa, rh, rmw, _).  Returns h, or -1.0 when infeasible."""
    pure_w = 0.0
    lo = 0.0
    all_soft = True
    for j in range(nck):
        sa = eckf[j, 0]
        rh = eckf[j, 1]
        if sa <= 0.0:
            pure_w += eckf[j, 2]
            all_soft = False
        elif rh > 0.0:
            all_soft = False
        if rh > lo:
            lo = rh
    avail = w - pure_w
    if avail <= 1e-6:
        return -1.0
    if all_soft:
        total = 0.0
        for j in range(nck):
            total += eckf[j, 0]
        h = total / w
        for j in range(nck):
            widths[j] = eckf[j, 0] / h
        return h
    lo += 1e-9

    hi = lo + 1.0
    found = False
    for _ in range(80):
        s = 0.0
        for j in range(nck):
            if eckf[j, 0] > 0.0:
                s += eckf[j, 0] / (hi - eckf[j, 1])
        if s <= avail:
            found = True
            break
        hi *= 2.0
    if not found:
        return -1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        s = 0.0
        for j in range(nck):
            if eckf[j, 0] > 0.0:
                s += eckf[j, 0] / (mid - eckf[j, 1])
        if s > avail:
            lo = mid
        else:
            hi = mid
    h = hi
    for j in range(nck):
        if eckf[j, 0] > 0.0:
            wj = eckf[j, 0] / (h - eckf[j, 1])
            if wj < eckf[j, 2] - 1e-9:
                return -1.0
            widths[j] = wj
        else:
            widths[j] = eckf[j, 2]
    return h


@njit(cache=True)
def _band_plan(u, w, uf, ui, bandi, chunkf, chunki, entf, enti,
               hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp):
    """`_band_solutions`: fill the one-slot plan cache for unit `u` at width w.

    Layout of the result, per band b of u:
      plan_ent[bandi[b,3]:bandi[b,4]]  entry ids in effective order
      plan_ck[b*_CHUNK_CAP + j]        start index of effective chunk j
                                       (chunk j spans [.. j], [.. j+1]))
      plan_ckw[b*_CHUNK_CAP + j]       width of effective chunk j
      plani[b,0] = n effective chunks, plani[b,1] = 1 when `_solve_band` won
      planf[b]   = band height contribution
    hcf[u] = (w, unit height)."""
    if hcf[u, 0] == w:
        return
    h = 0.0
    for b in range(ui[u, 5], ui[u, 6]):
        ent_lo = bandi[b, 3]
        ent_hi = bandi[b, 4]
        base = b * _CHUNK_CAP
        # -- start from the static chunk partition ------------------------
        p = ent_lo
        nck = 0
        for c in range(bandi[b, 1], bandi[b, 2]):
            plan_ck[base + nck] = p
            for e in range(chunki[c, 0], chunki[c, 1]):
                plan_ent[p] = e
                p += 1
            eckf[nck, 0] = chunkf[c, 0]
            eckf[nck, 1] = chunkf[c, 1]
            eckf[nck, 2] = chunkf[c, 2]
            nck += 1
        plan_ck[base + nck] = p

        # -- `_dyn_split` ---------------------------------------------------
        if nck == 1 and bandi[b, 0] == 1:
            npl = ent_hi - ent_lo
            total = eckf[0, 0]
            for t in range(ent_lo, ent_hi):
                e = plan_ent[t]
                if enti[e, 1] == 0:
                    total += entf[e, 1] * entf[e, 2]
            avg = total / npl
            d = 1.35 * np.sqrt(avg if avg > 1e-9 else 1e-9)
            if d < 1e-6:
                d = 1e-6
            m = int(w / d)
            if m >= 2:
                if m > npl:
                    m = npl
                if m > 4:
                    m = 4
                target = total / m
                # order-preserving greedy split
                nparts = 0
                acc = 0.0
                start = ent_lo
                for t in range(ent_lo, ent_hi):
                    e = plan_ent[t]
                    acc += entf[e, 0] if enti[e, 1] == 1 else entf[e, 1] * entf[e, 2]
                    if acc >= target - 1e-9 and nparts < m - 1:
                        plan_ck[base + nparts] = start
                        nparts += 1
                        start = t + 1
                        acc = 0.0
                if start < ent_hi:
                    plan_ck[base + nparts] = start
                    nparts += 1
                if nparts >= 2:
                    plan_ck[base + nparts] = ent_hi
                    for j in range(nparts):
                        sa = 0.0
                        rh = 0.0
                        rmw = 0.0
                        for t in range(plan_ck[base + j], plan_ck[base + j + 1]):
                            e = plan_ent[t]
                            if enti[e, 1] == 1:
                                sa += entf[e, 0]
                            else:
                                rh += entf[e, 2]
                                if entf[e, 1] > rmw:
                                    rmw = entf[e, 1]
                        eckf[j, 0] = sa
                        eckf[j, 1] = rh
                        eckf[j, 2] = rmw
                    nck = nparts
                else:
                    # restore the single-chunk boundaries
                    plan_ck[base] = ent_lo
                    plan_ck[base + 1] = ent_hi

        # -- solve, merging progressively on failure ------------------------
        has_sol = 0
        hb = 0.0
        if nck > 1:
            hs = _solve_band(eckf, nck, w, plan_ckw[base:base + _CHUNK_CAP])
            while hs < 0.0 and nck > 2:
                # `_merge_band_once`: merge the adjacent pair of least area
                amin = -1.0
                jmin = 0
                for j in range(nck - 1):
                    aj = eckf[j, 0]
                    for t in range(plan_ck[base + j], plan_ck[base + j + 1]):
                        e = plan_ent[t]
                        if enti[e, 1] == 0:
                            aj += entf[e, 1] * entf[e, 2]
                    aj1 = eckf[j + 1, 0]
                    for t in range(plan_ck[base + j + 1], plan_ck[base + j + 2]):
                        e = plan_ent[t]
                        if enti[e, 1] == 0:
                            aj1 += entf[e, 1] * entf[e, 2]
                    if amin < 0.0 or aj + aj1 < amin:
                        amin = aj + aj1
                        jmin = j
                # stable sort-by-rank of the concatenation (counting sort)
                s0 = plan_ck[base + jmin]
                s2 = plan_ck[base + jmin + 2]
                cnt = 0
                for t in range(s0, s2):
                    tmp[cnt] = plan_ent[t]
                    cnt += 1
                p = s0
                for r in range(3):
                    for t in range(cnt):
                        if enti[tmp[t], 2] == r:
                            plan_ent[p] = tmp[t]
                            p += 1
                eckf[jmin, 0] = eckf[jmin, 0] + eckf[jmin + 1, 0]
                eckf[jmin, 1] = eckf[jmin, 1] + eckf[jmin + 1, 1]
                eckf[jmin, 2] = max(eckf[jmin, 2], eckf[jmin + 1, 2])
                for j in range(jmin + 1, nck - 1):
                    eckf[j, 0] = eckf[j + 1, 0]
                    eckf[j, 1] = eckf[j + 1, 1]
                    eckf[j, 2] = eckf[j + 1, 2]
                    plan_ck[base + j] = plan_ck[base + j + 1]
                nck -= 1
                plan_ck[base + nck] = ent_hi
                hs = _solve_band(eckf, nck, w, plan_ckw[base:base + _CHUNK_CAP])
            if hs >= 0.0:
                has_sol = 1
                hb = hs
        if has_sol == 0:
            if nck > 1:
                cnt = 0
                for t in range(ent_lo, ent_hi):
                    tmp[cnt] = plan_ent[t]
                    cnt += 1
                p = ent_lo
                for r in range(3):
                    for t in range(cnt):
                        if enti[tmp[t], 2] == r:
                            plan_ent[p] = tmp[t]
                            p += 1
                sa = 0.0
                rh = 0.0
                rmw = 0.0
                for j in range(nck):
                    sa += eckf[j, 0]
                    rh += eckf[j, 1]
                    if eckf[j, 2] > rmw:
                        rmw = eckf[j, 2]
                eckf[0, 0] = sa
                eckf[0, 1] = rh
                eckf[0, 2] = rmw
                nck = 1
                plan_ck[base] = ent_lo
                plan_ck[base + 1] = ent_hi
            hb = eckf[0, 1] + (eckf[0, 0] / w if eckf[0, 0] > 0.0 else 0.0)
        plani[b, 0] = nck
        plani[b, 1] = has_sol
        planf[b] = hb
        h += hb
    hcf[u, 0] = w
    hcf[u, 1] = h


@njit(cache=True, inline='always')
def _unit_h(u, w, uf, ui, bandi, chunkf, chunki, entf, enti,
            hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp):
    if ui[u, 0] == 0:
        return uf[u, 1] + uf[u, 0] / w
    _band_plan(u, w, uf, ui, bandi, chunkf, chunki, entf, enti,
               hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp)
    return hcf[u, 1]


@njit(cache=True, inline='always')
def _chunk_up(pos, entf, enti, ents, e_lo, e_hi, xj, wj, y0, full_w, use_full):
    y = y0
    for t in range(e_lo, e_hi):
        e = ents[t]
        i = enti[e, 0]
        if enti[e, 1] == 1:
            bw = full_w if use_full else wj
            bh = entf[e, 0] / bw
        else:
            bw = entf[e, 1]
            bh = entf[e, 2]
        pos[i, 0] = xj
        pos[i, 1] = y
        pos[i, 2] = bw
        pos[i, 3] = bh
        y += bh
    return y


@njit(cache=True)
def _place_unit_up(u, x0, w, y0, pos, uf, ui, bandi, chunkf, chunki, entf, enti,
                   hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp,
                   static_ent):
    if ui[u, 0] == 0:
        # every band holds exactly one static chunk
        y = y0
        for b in range(ui[u, 5], ui[u, 6]):
            c = bandi[b, 1]
            y = _chunk_up(pos, entf, enti, static_ent,
                          chunki[c, 0], chunki[c, 1], x0, w, y, w, True)
        return y
    _band_plan(u, w, uf, ui, bandi, chunkf, chunki, entf, enti,
               hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp)
    y = y0
    for b in range(ui[u, 5], ui[u, 6]):
        base = b * _CHUNK_CAP
        nck = plani[b, 0]
        if nck == 1:
            y = _chunk_up(pos, entf, enti, plan_ent,
                          plan_ck[base], plan_ck[base + 1], x0, w, y, w, True)
        elif plani[b, 1] == 0:
            yy = y
            for j in range(nck):
                yy = _chunk_up(pos, entf, enti, plan_ent,
                               plan_ck[base + j], plan_ck[base + j + 1],
                               x0, w, yy, w, True)
            y = yy
        else:
            xj = x0
            for j in range(nck):
                _chunk_up(pos, entf, enti, plan_ent,
                          plan_ck[base + j], plan_ck[base + j + 1],
                          xj, plan_ckw[base + j], y, 0.0, False)
                xj += plan_ckw[base + j]
            y = y + planf[b]
    return y


@njit(cache=True, inline='always')
def _chunk_down(pos, entf, enti, ents, e_lo, e_hi, xj, wj, ytop):
    y = ytop
    for t in range(e_hi - 1, e_lo - 1, -1):
        e = ents[t]
        i = enti[e, 0]
        if enti[e, 1] == 1:
            bw = wj
            bh = entf[e, 0] / wj
        else:
            bw = entf[e, 1]
            bh = entf[e, 2]
        y -= bh
        pos[i, 0] = xj
        pos[i, 1] = y
        pos[i, 2] = bw
        pos[i, 3] = bh
    return y


@njit(cache=True)
def _place_unit_down(u, x0, w, ytop, pos, uf, ui, bandi, chunkf, chunki,
                     entf, enti, hcf, plan_ent, plan_ck, plan_ckw, plani,
                     planf, eckf, tmp, static_ent):
    if ui[u, 0] == 0:
        y = ytop
        for b in range(ui[u, 6] - 1, ui[u, 5] - 1, -1):
            c = bandi[b, 1]
            y = _chunk_down(pos, entf, enti, static_ent,
                            chunki[c, 0], chunki[c, 1], x0, w, y)
        return y
    _band_plan(u, w, uf, ui, bandi, chunkf, chunki, entf, enti,
               hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp)
    y = ytop
    for b in range(ui[u, 6] - 1, ui[u, 5] - 1, -1):
        base = b * _CHUNK_CAP
        nck = plani[b, 0]
        if nck == 1:
            y = _chunk_down(pos, entf, enti, plan_ent,
                            plan_ck[base], plan_ck[base + 1], x0, w, y)
        elif plani[b, 1] == 0:
            yy = y
            for j in range(nck - 1, -1, -1):
                yy = _chunk_down(pos, entf, enti, plan_ent,
                                 plan_ck[base + j], plan_ck[base + j + 1],
                                 x0, w, yy)
            y = yy
        else:
            xj = x0
            for j in range(nck):
                _chunk_down(pos, entf, enti, plan_ent,
                            plan_ck[base + j], plan_ck[base + j + 1],
                            xj, plan_ckw[base + j], y)
                xj += plan_ckw[base + j]
            y = y - planf[b]
    return y


@njit(cache=True)
def _stack_column(col_u, c_lo, c_hi, x, w, pos, pb,
                  uf, ui, bandi, chunkf, chunki, entf, enti, ublk, anc, lock,
                  hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp,
                  static_ent, occ, segs, normal, pending, pl_k, pl_yb, pl_yt,
                  strip):
    """`_stack_column`. Returns (n_placed, col_top, status).

    Placed rows are written to pl_*[pb : pb + n_placed]."""
    has_locked = lock.shape[0] > 0
    n_occ = 0
    if has_locked:
        n_occ = _obstacles_in(lock, x, x + w, occ)

    npl = 0
    nn = 0
    for idx in range(c_lo, c_hi):
        k = col_u[idx]
        done = False
        if ui[k, 9] < ui[k, 10] and has_locked:
            uh = _unit_h(k, w, uf, ui, bandi, chunkf, chunki, entf, enti,
                         hcf, plan_ent, plan_ck, plan_ckw, plani, planf,
                         eckf, tmp)
            for a in range(ui[k, 9], ui[k, 10]):
                ax = anc[a, 0]
                ay = anc[a, 1]
                aw = anc[a, 2]
                ah = anc[a, 3]
                if not (ax < x + w - 1e-9 and ax + aw > x + 1e-9):
                    continue
                y0 = ay - uh
                if y0 >= -1e-9 and _interval_free(occ, n_occ, y0, ay):
                    yb = _place_unit_down(k, x, w, ay, pos, uf, ui, bandi,
                                          chunkf, chunki, entf, enti, hcf,
                                          plan_ent, plan_ck, plan_ckw, plani,
                                          planf, eckf, tmp, static_ent)
                    n_occ = _add_interval(occ, n_occ, yb, ay)
                    pl_k[pb + npl] = k
                    pl_yb[pb + npl] = yb
                    pl_yt[pb + npl] = ay
                    npl += 1
                    done = True
                    break
                y1 = ay + ah
                if _interval_free(occ, n_occ, y1, y1 + uh):
                    yt = _place_unit_up(k, x, w, y1, pos, uf, ui, bandi,
                                        chunkf, chunki, entf, enti, hcf,
                                        plan_ent, plan_ck, plan_ckw, plani,
                                        planf, eckf, tmp, static_ent)
                    n_occ = _add_interval(occ, n_occ, y1, yt)
                    pl_k[pb + npl] = k
                    pl_yb[pb + npl] = y1
                    pl_yt[pb + npl] = yt
                    npl += 1
                    done = True
                    break
        if not done:
            normal[nn] = k
            nn += 1

    # boundary-friendly stacking order: bottom-tagged first, top-tagged last
    if nn > 0:
        m = 0
        for t in range(nn):
            k = normal[t]
            if ui[k, 2] == 1 and ui[k, 3] == 0:
                pending[m] = k
                m += 1
        for t in range(nn):
            k = normal[t]
            if not (ui[k, 2] == 1 and ui[k, 3] == 0) and ui[k, 3] == 0:
                pending[m] = k
                m += 1
        for t in range(nn):
            k = normal[t]
            if not (ui[k, 2] == 1 and ui[k, 3] == 0) and ui[k, 3] == 1:
                pending[m] = k
                m += 1
        for t in range(nn):
            normal[t] = pending[t]

    # bottom pre-pass (the Python loop always breaks -> only normal[0])
    if n_occ > 0 and nn > 0:
        k = normal[0]
        if ui[k, 2] == 1 and ui[k, 3] == 0:
            uh = _unit_h(k, w, uf, ui, bandi, chunkf, chunki, entf, enti,
                         hcf, plan_ent, plan_ck, plan_ckw, plani, planf,
                         eckf, tmp)
            if not _interval_free(occ, n_occ, 0.0, uh):
                if ui[k, 4] == -1:
                    have = False
                    sw_x = 0.0
                    sw_w = 0.0
                    probe_h = uh
                    for _ in range(3):
                        if not _band_strip(lock, x, w, 0.0, probe_h,
                                           pl_yb, pl_yt, pb, pb + npl, strip):
                            have = False
                            break
                        have = True
                        sw_x = strip[0]
                        sw_w = strip[1]
                        nh = _unit_h(k, sw_w, uf, ui, bandi, chunkf, chunki,
                                     entf, enti, hcf, plan_ent, plan_ck,
                                     plan_ckw, plani, planf, eckf, tmp)
                        if abs(nh - probe_h) < 1e-6:
                            probe_h = nh
                            break
                        probe_h = nh
                    if have and not (uf[k, 2] > sw_w + 1e-9):
                        if _band_strip(lock, x, w, 0.0, probe_h,
                                       pl_yb, pl_yt, pb, pb + npl, strip):
                            if not (strip[1] < sw_w - 1e-6):
                                yt = _place_unit_up(
                                    k, sw_x, sw_w, 0.0, pos, uf, ui, bandi,
                                    chunkf, chunki, entf, enti, hcf, plan_ent,
                                    plan_ck, plan_ckw, plani, planf, eckf, tmp,
                                    static_ent)
                                n_occ = _add_interval(occ, n_occ, 0.0, yt)
                                pl_k[pb + npl] = k
                                pl_yb[pb + npl] = 0.0
                                pl_yt[pb + npl] = yt
                                npl += 1
                                for t in range(1, nn):
                                    normal[t - 1] = normal[t]
                                nn -= 1

    col_top = 0.0
    if n_occ == 0:
        cy = 0.0
        i = 0
        while i < nn:
            k = normal[i]
            if i + 1 < nn:
                k2 = normal[i + 1]
                if (ui[k, 7] + 1 == ui[k, 8] and ui[k2, 7] + 1 == ui[k2, 8]
                        and ui[k, 1] == 1 and ui[k2, 1] == 1):
                    a1 = uf[k, 3]
                    a2 = uf[k2, 3]
                    hp = (a1 + a2) / w
                    if hp <= 0.85 * w:
                        w1 = a1 / hp
                        if 1.0 <= w1 and w1 <= w - 1.0:
                            i1 = ublk[ui[k, 7]]
                            i2 = ublk[ui[k2, 7]]
                            pos[i1, 0] = x
                            pos[i1, 1] = cy
                            pos[i1, 2] = w1
                            pos[i1, 3] = a1 / w1
                            pos[i2, 0] = x + w1
                            pos[i2, 1] = cy
                            pos[i2, 2] = w - w1
                            pos[i2, 3] = a2 / (w - w1)
                            top = cy + hp
                            pl_k[pb + npl] = k
                            pl_yb[pb + npl] = cy
                            pl_yt[pb + npl] = top
                            npl += 1
                            pl_k[pb + npl] = k2
                            pl_yb[pb + npl] = cy
                            pl_yt[pb + npl] = top
                            npl += 1
                            cy = top
                            i += 2
                            continue
            yt = _place_unit_up(k, x, w, cy, pos, uf, ui, bandi, chunkf,
                                chunki, entf, enti, hcf, plan_ent, plan_ck,
                                plan_ckw, plani, planf, eckf, tmp, static_ent)
            pl_k[pb + npl] = k
            pl_yb[pb + npl] = cy
            pl_yt[pb + npl] = yt
            npl += 1
            cy = yt
            i += 1
        col_top = cy
    else:
        nseg = 0
        cur = 0.0
        for t in range(n_occ):
            s = occ[t, 0]
            e = occ[t, 1]
            if s > cur + 1e-9:
                segs[nseg, 0] = cur
                segs[nseg, 1] = s
                segs[nseg, 2] = x
                segs[nseg, 3] = w
                nseg += 1
            band_lo = cur if cur > s else s
            band_hi = e
            if _band_strip(lock, x, w, band_lo, band_hi,
                           pl_yb, pl_yt, pb, pb + npl, strip):
                segs[nseg, 0] = band_lo
                segs[nseg, 1] = band_hi
                segs[nseg, 2] = strip[0]
                segs[nseg, 3] = strip[1]
                nseg += 1
            if e > cur:
                cur = e
        segs[nseg, 0] = cur
        segs[nseg, 1] = np.inf
        segs[nseg, 2] = x
        segs[nseg, 3] = w
        nseg += 1

        si = 0
        cy = segs[0, 0]
        npend = nn
        for t in range(nn):
            pending[t] = normal[t]
        while npend > 0:
            seg_end = segs[si, 1]
            seg_x = segs[si, 2]
            seg_w = segs[si, 3]
            pick = -1
            narrow = seg_x > x + 1e-9 or seg_w < w - 1e-9
            lim = npend if npend < 10 else 10
            for t in range(lim):
                k = pending[t]
                if uf[k, 2] > seg_w + 1e-9:
                    continue
                if narrow and ui[k, 4] != -1:
                    continue
                uh = _unit_h(k, seg_w, uf, ui, bandi, chunkf, chunki, entf,
                             enti, hcf, plan_ent, plan_ck, plan_ckw, plani,
                             planf, eckf, tmp)
                if cy + uh <= seg_end + 1e-9:
                    pick = t
                    break
            if pick < 0:
                si += 1
                if si >= nseg:
                    return npl, col_top, _FAIL_SEGMENT_OVERRUN
                cy = segs[si, 0]
                continue
            k = pending[pick]
            for t in range(pick, npend - 1):
                pending[t] = pending[t + 1]
            npend -= 1
            yt = _place_unit_up(k, seg_x, seg_w, cy, pos, uf, ui, bandi,
                                chunkf, chunki, entf, enti, hcf, plan_ent,
                                plan_ck, plan_ckw, plani, planf, eckf, tmp,
                                static_ent)
            pl_k[pb + npl] = k
            pl_yb[pb + npl] = cy
            pl_yt[pb + npl] = yt
            npl += 1
            cy = yt
        for t in range(pb, pb + npl):
            if pl_yt[t] > col_top:
                col_top = pl_yt[t]
    return npl, col_top, _OK


@njit(cache=True)
def _layout(col_u, col_ptr, ncol, H, pos, blkf, blki,
            uf, ui, bandi, chunkf, chunki, entf, enti, ublk, anc, lock,
            hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp,
            static_ent, occ, segs, normal, pending, pl_k, pl_yb, pl_yt,
            strip, ucol, col_x0, col_w, col_valid, col_pl, outf):
    """`_layout_full`, transcribed. Writes pos / ucol / col_* and outf."""
    n = pos.shape[0]
    for i in range(n):
        pos[i, 0] = 0.0
        pos[i, 1] = 0.0
        pos[i, 2] = 0.0
        pos[i, 3] = 0.0
        if blki[i, 0] == 2:
            pos[i, 0] = blkf[i, 0]
            pos[i, 1] = blkf[i, 1]
            pos[i, 2] = blkf[i, 2]
            pos[i, 3] = blkf[i, 3]

    has_locked = lock.shape[0] > 0
    x = 0.0
    pb = 0
    for ci in range(ncol):
        c_lo = col_ptr[ci]
        c_hi = col_ptr[ci + 1]
        col_pl[ci, 0] = pb
        col_pl[ci, 1] = pb
        if c_lo == c_hi:
            col_valid[ci] = 0
            continue
        col_valid[ci] = 1
        for idx in range(c_lo, c_hi):
            ucol[col_u[idx]] = ci
        soft_a = 0.0
        rigid_h = 0.0
        max_w = 0.0
        for idx in range(c_lo, c_hi):
            k = col_u[idx]
            soft_a += uf[k, 0]
            rigid_h += uf[k, 1]
            if uf[k, 2] > max_w:
                max_w = uf[k, 2]
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
                n_ob = _obstacles_in(lock, x, x + w, occ)
                obs_h = 0.0
                for t in range(n_ob):
                    s = occ[t, 0]
                    e = occ[t, 1]
                    if e > 0.0 and s < H:
                        span = (e if e < H else H) - (s if s > 0.0 else 0.0)
                        if _band_strip(lock, x, w, s, e,
                                       pl_yb, pl_yt, 0, 0, strip):
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

        npl, col_top, st = _stack_column(
            col_u, c_lo, c_hi, x, w, pos, pb, uf, ui, bandi, chunkf, chunki,
            entf, enti, ublk, anc, lock, hcf, plan_ent, plan_ck, plan_ckw,
            plani, planf, eckf, tmp, static_ent, occ, segs, normal, pending,
            pl_k, pl_yb, pl_yt, strip)
        if st != _OK:
            outf[2] = st
            return
        if soft_a > 0.0:
            tries = 0
            while col_top > H * 1.0005 and tries < 3:
                overhead = col_top - soft_a / w
                if overhead < H * 0.98:
                    w2 = soft_a / (H - overhead)
                else:
                    w2 = w * 1.25
                lo2 = w * 1.01
                w2 = w2 if w2 > lo2 else lo2
                hi2 = w * 2.0
                w = w2 if w2 < hi2 else hi2
                npl, col_top, st = _stack_column(
                    col_u, c_lo, c_hi, x, w, pos, pb, uf, ui, bandi, chunkf,
                    chunki, entf, enti, ublk, anc, lock, hcf, plan_ent,
                    plan_ck, plan_ckw, plani, planf, eckf, tmp, static_ent,
                    occ, segs, normal, pending, pl_k, pl_yb, pl_yt, strip)
                if st != _OK:
                    outf[2] = st
                    return
                tries += 1
        col_x0[ci] = x
        col_w[ci] = w
        col_pl[ci, 1] = pb + npl
        pb += npl
        x += w

    x_right = x
    y_top = 0.0
    for ci in range(ncol):
        if col_valid[ci] == 0:
            continue
        for t in range(col_pl[ci, 0], col_pl[ci, 1]):
            if pl_yt[t] > y_top:
                y_top = pl_yt[t]
    for t in range(lock.shape[0]):
        v = lock[t, 0] + lock[t, 2]
        if v > x_right:
            x_right = v
        v = lock[t, 1] + lock[t, 3]
        if v > y_top:
            y_top = v

    # lift top-tagged top units flush to the global top edge
    for ci in range(ncol):
        if col_valid[ci] == 0 or col_pl[ci, 1] == col_pl[ci, 0]:
            continue
        best = col_pl[ci, 0]
        for t in range(col_pl[ci, 0] + 1, col_pl[ci, 1]):
            if pl_yt[t] > pl_yt[best]:
                best = t
        k = pl_k[best]
        yb = pl_yb[best]
        yt = pl_yt[best]
        top_block = ublk[ui[k, 8] - 1]
        if (blki[top_block, 1] & 4) == 0:
            continue
        dy = y_top - yt
        if dy <= 1e-9:
            continue
        nbl = _obstacles_in(lock, col_x0[ci], col_x0[ci] + col_w[ci], occ)
        for t in range(col_pl[ci, 0], col_pl[ci, 1]):
            if pl_k[t] != k:
                occ[nbl, 0] = pl_yb[t]
                occ[nbl, 1] = pl_yt[t]
                nbl += 1
        if _interval_free(occ, nbl, yb + dy, yt + dy):
            for t in range(ui[k, 7], ui[k, 8]):
                pos[ublk[t], 1] += dy

    # right-align right-tagged blocks in the last non-empty column
    last = -1
    for ci in range(ncol - 1, -1, -1):
        if col_valid[ci] == 1 and col_pl[ci, 1] > col_pl[ci, 0]:
            last = ci
            break
    if last >= 0:
        for t in range(col_pl[last, 0], col_pl[last, 1]):
            k = pl_k[t]
            for q in range(ui[k, 7], ui[k, 8]):
                i = ublk[q]
                if (blki[i, 1] & 2) != 0:
                    nx = x_right - pos[i, 2]
                    if nx <= pos[i, 0] + 1e-9:
                        continue
                    ny0 = pos[i, 1]
                    ny1 = pos[i, 1] + pos[i, 3]
                    clash = False
                    for j in range(n):
                        if j == i:
                            continue
                        if (pos[j, 0] < x_right - 1e-7
                                and pos[j, 0] + pos[j, 2] > nx + 1e-7
                                and pos[j, 1] < ny1 - 1e-7
                                and pos[j, 1] + pos[j, 3] > ny0 + 1e-7):
                            clash = True
                            break
                    if not clash:
                        pos[i, 0] = nx

    outf[0] = x_right
    outf[1] = y_top
    outf[2] = _OK


@njit(cache=True)
def _violations(pos, bnd_idx, bnd_codes, clu_ptr, clu_idx, mib_ptr, mib_idx,
                comp):
    """`_violations`, transcribed.

    Bit-exact for free: the result is an integer COUNT derived from float
    comparisons, so unlike a floating-point reduction it cannot depend on
    accumulation order. `x_min`/`x_max` are min/max reductions, which are
    exact in any order, and numba's `round(x, 4)` matches CPython's
    correctly-rounded decimal round exactly (verified over 20k random
    doubles), so the MIB distinct-shape count is reproduced too."""
    n = pos.shape[0]
    x_min = pos[0, 0]
    y_min = pos[0, 1]
    x_max = pos[0, 0] + pos[0, 2]
    y_max = pos[0, 1] + pos[0, 3]
    for i in range(1, n):
        if pos[i, 0] < x_min:
            x_min = pos[i, 0]
        if pos[i, 1] < y_min:
            y_min = pos[i, 1]
        v = pos[i, 0] + pos[i, 2]
        if v > x_max:
            x_max = v
        v = pos[i, 1] + pos[i, 3]
        if v > y_max:
            y_max = v

    V = 0
    eps = 1e-6
    for t in range(bnd_idx.shape[0]):
        i = bnd_idx[t]
        c = bnd_codes[t]
        bad = False
        if (c & 1) != 0 and abs(pos[i, 0] - x_min) >= eps:
            bad = True
        if (c & 2) != 0 and abs(pos[i, 0] + pos[i, 2] - x_max) >= eps:
            bad = True
        if (c & 4) != 0 and abs(pos[i, 1] + pos[i, 3] - y_max) >= eps:
            bad = True
        if (c & 8) != 0 and abs(pos[i, 1] - y_min) >= eps:
            bad = True
        if bad:
            V += 1

    # grouping: connected components under the touch/overlap relation
    for g in range(clu_ptr.shape[0] - 1):
        lo = clu_ptr[g]
        hi = clu_ptr[g + 1]
        m = hi - lo
        if m < 2:
            continue
        for s in range(m):
            comp[s] = s
        for a in range(m):
            ia = clu_idx[lo + a]
            ax0 = pos[ia, 0]
            ay0 = pos[ia, 1]
            ax1 = ax0 + pos[ia, 2]
            ay1 = ay0 + pos[ia, 3]
            for b in range(a + 1, m):
                ib = clu_idx[lo + b]
                bx0 = pos[ib, 0]
                by0 = pos[ib, 1]
                ox = (ax1 if ax1 < bx0 + pos[ib, 2] else bx0 + pos[ib, 2]) \
                    - (ax0 if ax0 > bx0 else bx0)
                oy = (ay1 if ay1 < by0 + pos[ib, 3] else by0 + pos[ib, 3]) \
                    - (ay0 if ay0 > by0 else by0)
                if ((ox > TOUCH_TOL and oy >= -TOUCH_TOL)
                        or (oy > TOUCH_TOL and ox >= -TOUCH_TOL)):
                    ra = a
                    while comp[ra] != ra:
                        ra = comp[ra]
                    rb = b
                    while comp[rb] != rb:
                        rb = comp[rb]
                    if ra != rb:
                        comp[rb] = ra
        comps = 0
        for s in range(m):
            if comp[s] == s:
                comps += 1
        V += comps - 1

    # multi-instantiation: distinct rounded shapes within a group
    for g in range(mib_ptr.shape[0] - 1):
        lo = mib_ptr[g]
        hi = mib_ptr[g + 1]
        m = hi - lo
        if m < 2:
            continue
        distinct = 0
        for a in range(m):
            ia = mib_idx[lo + a]
            wa = round(pos[ia, 2], 4)
            ha = round(pos[ia, 3], 4)
            dup = False
            for b in range(a):
                ib = mib_idx[lo + b]
                if round(pos[ib, 2], 4) == wa and round(pos[ib, 3], 4) == ha:
                    dup = True
                    break
            if not dup:
                distinct += 1
        V += distinct - 1
    return V


# =============================================================================
# Flat-array build + Python-side driver
# =============================================================================
@dataclass
class _Static:
    blkf: np.ndarray
    blki: np.ndarray
    uf: np.ndarray
    ui: np.ndarray
    bandi: np.ndarray
    chunkf: np.ndarray
    chunki: np.ndarray
    entf: np.ndarray
    enti: np.ndarray
    ublk: np.ndarray
    anc: np.ndarray
    lock: np.ndarray
    static_ent: np.ndarray
    bnd_idx: np.ndarray
    bnd_codes: np.ndarray
    clu_ptr: np.ndarray
    clu_idx: np.ndarray
    mib_ptr: np.ndarray
    mib_idx: np.ndarray


def _rank(code: int) -> int:
    return 0 if (code & 8) else (2 if (code & 4) else 1)


def build_static(opt) -> _Static:
    """Flatten the `_Unit` object graph into CSR arrays.

    Mirrors `_refresh_unit`'s output exactly: bands = u.bands, chunks = the
    `_partition_subgroup` parts, entries = the (block, is_soft, area, w, h)
    tuples in `pl`."""
    n = opt.n
    units = opt.units

    blkf = np.zeros((n, 4))
    blki = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        blkf[i, 0] = opt.lx[i]
        blkf[i, 1] = opt.ly[i]
        blkf[i, 2] = opt.rw[i]
        blkf[i, 3] = opt.rh[i]
        blki[i, 0] = opt.kind[i]
        blki[i, 1] = opt.boundary[i]

    U = len(units)
    uf = np.zeros((U, 4))
    ui = np.zeros((U, 11), dtype=np.int64)

    bands_f: List[Tuple[int, int, int, int, int]] = []
    chunk_f: List[Tuple[float, float, float]] = []
    chunk_i: List[Tuple[int, int]] = []
    ent_f: List[Tuple[float, float, float]] = []
    ent_i: List[Tuple[int, int, int]] = []
    ublk: List[int] = []
    anc: List[Tuple[float, float, float, float]] = []

    for k, u in enumerate(units):
        uf[k, 0] = u.eff_soft
        uf[k, 1] = u.eff_rigid_h
        uf[k, 2] = u.max_rigid_w
        uf[k, 3] = u.soft_area
        ui[k, 0] = 1 if u.banded else 0
        ui[k, 1] = 1 if u.pairable else 0
        ui[k, 2] = 1 if u.hasB else 0
        ui[k, 3] = 1 if u.hasT else 0
        ui[k, 4] = 0 if u.force == 'L' else (1 if u.force == 'R' else -1)
        ui[k, 5] = len(bands_f)
        for band, dyn_ok in zip(u.bands, u.dyn):
            c_lo = len(chunk_f)
            e_lo = len(ent_f)
            for (pl, sa, rh, rmw) in band:
                ce_lo = len(ent_f)
                for (i, soft, a, bw, bh) in pl:
                    ent_f.append((a, bw, bh))
                    ent_i.append((i, 1 if soft else 0, _rank(opt.boundary[i])))
                chunk_f.append((sa, rh, rmw))
                chunk_i.append((ce_lo, len(ent_f)))
            bands_f.append((1 if dyn_ok else 0, c_lo, len(chunk_f),
                            e_lo, len(ent_f)))
        ui[k, 6] = len(bands_f)
        ui[k, 7] = len(ublk)
        ublk.extend(u.blocks)
        ui[k, 8] = len(ublk)
        ui[k, 9] = len(anc)
        anc.extend(u.anchors)
        ui[k, 10] = len(anc)

    def arr2(rows, cols, dtype=np.float64):
        if not rows:
            return np.zeros((0, cols), dtype=dtype)
        return np.array(rows, dtype=dtype)

    entf = arr2(ent_f, 3)
    enti = arr2(ent_i, 3, np.int64)
    E = entf.shape[0]

    def csr(groups):
        ptr = [0]
        idx: List[int] = []
        for g in groups:
            idx.extend(int(v) for v in g)
            ptr.append(len(idx))
        return (np.array(ptr, dtype=np.int64),
                np.array(idx, dtype=np.int64) if idx else np.zeros(0, np.int64))

    clu_ptr, clu_idx = csr(opt._clu_arrays)
    mib_ptr, mib_idx = csr(opt._mib_arrays)
    return _Static(
        blkf=blkf,
        blki=blki,
        uf=uf,
        ui=ui,
        bandi=arr2(bands_f, 5, np.int64),
        chunkf=arr2(chunk_f, 3),
        chunki=arr2(chunk_i, 2, np.int64),
        entf=entf,
        enti=enti,
        ublk=np.array(ublk, dtype=np.int64) if ublk else np.zeros(0, np.int64),
        anc=arr2(anc, 4),
        lock=arr2(list(opt.locked_rects), 4),
        static_ent=np.arange(E, dtype=np.int64),
        bnd_idx=opt._bnd_idx,
        bnd_codes=opt._bnd_codes,
        clu_ptr=clu_ptr,
        clu_idx=clu_idx,
        mib_ptr=mib_ptr,
        mib_idx=mib_idx,
    )


@dataclass
class SAKernel:
    """Driver: owns the flat arrays and dispatches `_layout` per SA move."""

    opt: object
    stat: _Static
    n_units: int
    dirty: bool = False
    calls: int = 0
    fallbacks: int = 0
    _work: dict = field(default_factory=dict)
    _ncol: int = -1

    # -- lifecycle ------------------------------------------------------
    def mark_dirty(self) -> None:
        """A `_refresh_unit` changed a unit's band structure."""
        self.dirty = True

    def _rebuild(self) -> None:
        self.stat = build_static(self.opt)
        self.dirty = False
        self._alloc(force=True)

    def _alloc(self, force: bool = False) -> None:
        s = self.stat
        U = s.uf.shape[0]
        B = s.bandi.shape[0]
        E = s.entf.shape[0]
        L = s.lock.shape[0]
        n = s.blkf.shape[0]
        if force or not self._work:
            cap_iv = L + U + 8
            self._work = {
                "hcf": np.full((U, 2), -1.0),
                "plan_ent": np.zeros(max(E, 1), dtype=np.int64),
                "plan_ck": np.zeros(max(B * _CHUNK_CAP, 1), dtype=np.int64),
                "plan_ckw": np.zeros(max(B * _CHUNK_CAP, 1)),
                "plani": np.zeros((max(B, 1), 2), dtype=np.int64),
                "planf": np.zeros(max(B, 1)),
                "eckf": np.zeros((_CHUNK_CAP, 4)),
                "tmp": np.zeros(max(E, 1), dtype=np.int64),
                "occ": np.zeros((cap_iv * 2 + 8, 2)),
                "segs": np.zeros((cap_iv * 2 + 8, 4)),
                "normal": np.zeros(max(U, 1), dtype=np.int64),
                "pending": np.zeros(max(U, 1), dtype=np.int64),
                "pl_k": np.zeros(max(U, 1), dtype=np.int64),
                "pl_yb": np.zeros(max(U, 1)),
                "pl_yt": np.zeros(max(U, 1)),
                "strip": np.zeros(2),
                "ucol": np.zeros(max(U, 1), dtype=np.int64),
                "outf": np.zeros(4),
                "col_u": np.zeros(max(U, 1), dtype=np.int64),
                "comp": np.zeros(max(n, 1), dtype=np.int64),
                "n": n,
            }
            # `_unit_col` is read by `_random_move`; hand it the same buffer so
            # the kernel can write it with zero Python-side cost.
            self.opt._unit_col = self._work["ucol"]
            self._ncol = -1   # the per-column buffers went with the old dict

    def _alloc_cols(self, C: int) -> None:
        if C == self._ncol:
            return
        self._ncol = C
        self._work["col_ptr"] = np.zeros(C + 1, dtype=np.int64)
        self._work["col_x0"] = np.zeros(C)
        self._work["col_w"] = np.zeros(C)
        self._work["col_valid"] = np.zeros(C, dtype=np.int64)
        self._work["col_pl"] = np.zeros((C, 2), dtype=np.int64)

    # -- hot path -------------------------------------------------------
    def layout(self, cols: List[List[int]]):
        """Return (pos, x_right, y_top), or None to fall back to Python."""
        if self.dirty:
            self._rebuild()
        C = len(cols)
        self._alloc_cols(C)
        w = self._work
        # Flatten `cols` with two bulk slice-assignments rather than per-unit
        # numpy stores: element-wise `arr[i] = k` costs ~60 ns each, which at
        # 100 units is a visible fraction of the whole kernel call.
        ptr = [0] * (C + 1)
        acc = 0
        for ci in range(C):
            acc += len(cols[ci])
            ptr[ci + 1] = acc
        col_ptr = w["col_ptr"]
        col_u = w["col_u"]
        col_ptr[:] = ptr
        if acc:
            col_u[:acc] = [k for c in cols for k in c]

        # `_layout` writes every row of `pos` (locked rows from blkf, the rest
        # zeroed in its prologue), so an uninitialised buffer is safe here.
        pos = np.empty((w["n"], 4))
        outf = w["outf"]
        outf[2] = _OK
        _layout(col_u, col_ptr, C, self.opt.H, pos,
                self.stat.blkf, self.stat.blki, self.stat.uf, self.stat.ui,
                self.stat.bandi, self.stat.chunkf, self.stat.chunki,
                self.stat.entf, self.stat.enti, self.stat.ublk,
                self.stat.anc, self.stat.lock,
                w["hcf"], w["plan_ent"], w["plan_ck"], w["plan_ckw"],
                w["plani"], w["planf"], w["eckf"], w["tmp"],
                self.stat.static_ent, w["occ"], w["segs"], w["normal"],
                w["pending"], w["pl_k"], w["pl_yb"], w["pl_yt"], w["strip"],
                w["ucol"], w["col_x0"], w["col_w"], w["col_valid"],
                w["col_pl"], outf)
        self.calls += 1
        if outf[2] != _OK:
            self.fallbacks += 1
            return None
        # `.tolist()` once beats C numpy-scalar arithmetic operations
        x0 = w["col_x0"].tolist()
        cw = w["col_w"].tolist()
        valid = w["col_valid"].tolist()
        self.opt._col_spans = [
            (x0[ci], x0[ci] + cw[ci]) if valid[ci] else None for ci in range(C)
        ]
        return pos, outf[0], outf[1]

    def violations(self, pos) -> int:
        s = self.stat
        return int(_violations(pos, s.bnd_idx, s.bnd_codes, s.clu_ptr,
                               s.clu_idx, s.mib_ptr, s.mib_idx,
                               self._work["comp"]))

    def warmup(self) -> None:
        """Force JIT compilation (or a cache load) before the SA starts."""
        cols = [[k] for k in range(min(self.n_units, 2))]
        if not cols:
            return
        try:
            out = self.layout(cols)
            if out is not None:
                self.violations(out[0])
        except Exception:
            pass
        # reset the caches the warmup dirtied
        self._work["hcf"][:, 0] = -1.0


def try_attach(opt) -> Optional[SAKernel]:
    """Build a kernel for `opt`, or return None (caller keeps the Python path)."""
    if not NUMBA_AVAILABLE:
        return None
    if not getattr(opt, "units", None):
        return None
    try:
        stat = build_static(opt)
        k = SAKernel(opt=opt, stat=stat, n_units=len(opt.units))
        k._alloc(force=True)
        if os.environ.get("PARTNER_SA_KERNEL_WARMUP", "1") in _TRUE:
            k.warmup()
        return k
    except Exception:
        return None


def kernel_enabled() -> bool:
    return os.environ.get("PARTNER_SA_KERNEL", "") == "numba"


# =============================================================================
# Not ported (deliberate)
# =============================================================================
# `_hpwl` is ~4-5% of the inner move and stays in numpy: it is a float
# reduction, and `np.sum`'s pairwise summation is not what a sequential loop
# produces. Porting it would perturb every Metropolis accept and turn this
# toggle from a pure speed change into a search-trajectory change, which needs
# its own evaluator evidence. Reproducing numpy's pairwise kernel exactly is
# the way to claim that ~5% without giving up the bit-exactness contract.
