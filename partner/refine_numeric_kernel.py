#!/usr/bin/env python3
"""Flat-array (structure-of-arrays) numba kernel for the direct-refine rung.

Why this exists
---------------
A ladder rung is one `_Refiner.legalize_soft()` call, and `legalize_soft` is
`legalize()` four times over (0.88 / 0.94 / 0.97 / 1.0 regrow steps).  Measured
anatomy of a rung at n=116 (`scratchpad/refine_axis_split.py`, single thread,
cached Direct predictions):

    section                              share of a rung
    _evict           (anchor x variant scan)   35 %
    _axis_constraints B (pair -> edge loop)    25 %
    _axis_constraints C (per-group bounds)     14 %
    _axis_pass        E (forward assign+move)   9 %
    _axis_constraints A (n^2 numpy masks)       8 %
    _axis_constraints D (backward longest path) 6 %
    _has_overlap                                2 %
    everything else                             3 %

Every one of those is *pure numeric* -- the object graph (`_Group`, the
`_AxisCons` edge lists) is only a container for numbers.  So, exactly as
`sa_numeric_kernel.py` did for the column-SA layout pass, this module
re-expresses the same algorithm over flat arrays that numba can compile.
Amdahl bound with `_evict` included: ~1/0.03, i.e. the rung is not
serial-limited at all; the realised speedup is set by how well the kernel
does, not by what is left in Python.

Bit-exactness contract
----------------------
Transcription, not reimplementation: every arithmetic expression, comparison
tolerance, iteration order, tie-break and early exit is reproduced from
`layout_refiner.py`.  `tests/test_partner_refine_kernel.py` asserts `==` (not
`allclose`) on the resulting `P` arrays.

Two places deserve their own argument, because they are where a transcription
could silently drift:

1. **Edge ordering.**  `_axis_constraints` collects pair constraints into a
   `tight` dict and then walks `tight.items()` -- i.e. *first-encounter*
   order -- to fill `edges_in` / `edges_out`.  The kernel reproduces that
   order exactly: `tight_idx` maps `(gi, gj)` to a slot in an append-ordered
   edge list, and the CSR is built by a counting sort that is stable in that
   append order.  (The order is in fact inert -- every consumer reduces the
   edges with `max` / `min` of exactly-representable doubles, never a sum --
   but reproducing it costs nothing and removes the argument.)

2. **`sorted()` inside `_move`.**  The cluster contact re-snap sorts `g.cH` by
   `P[a, 0]` with Python's stable `sorted`, evaluating every key *before* any
   assignment.  The kernel snapshots the keys, runs a stable insertion sort,
   and only then assigns -- same permutation, same reads.

`np.argsort(key_of, kind="stable")` is reproduced with numba's
`kind='mergesort'`: a stable sort's output permutation is uniquely determined
by its input, so the two agree by definition (asserted in the test file).

Scope
-----
Ported under `PARTNER_REFINE_KERNEL=numba`: `_axis_pass` with `hold=True`
(the legalization sweep -- `legalize`, `_legal_check`), the `_evict`
anchor x variant scan, `_has_overlap` and `_overlap_count`.

Ported under the additional sub-switch `PARTNER_REFINE_KERNEL_QSWEEP=1`:
`_axis_pass(hold=False)` -- the HPWL weighted-median sweep that `_Refiner.run`
spends its rounds on (the *quality* half, as opposed to the legalization half
above).  It shares the whole `_axis_constraints` prologue with the `hold=True`
pass (`_axis_build_k`); the only new numerics are `_median_shift_k` and
`_wmedian_k`.

The reason this needed its own switch is `_wmedian`'s `np.argsort(vals)`:
numpy's *default* (introsort) argsort, whose tie permutation is not
reproducible from the algorithm alone, and whose output is then `cumsum`ed.
The invariance argument is:

  * `v = vals[order]` is the sorted value array -- unique, so independent of
    which permutation the sort chose;
  * a different permutation only permutes *within* blocks of equal `vals`, so
    `c = cumsum(wts[order])` differs only at interior positions of such a
    block; the cumulative weight at every block *boundary* is the same sum of
    the same terms;
  * `searchsorted(c, half)` can therefore only land at a different index
    *inside* one block, where every `v` is the same number.

So in exact arithmetic `_wmedian` is permutation-invariant, and the kernel is
free to use its own (stable) sort.  What survives the argument is float
re-association: reordering the weights inside a tie block changes the
intermediate partial sums, hence `c[-1]` and `half`, by up to an ULP, which in
a knife-edge case could move the crossing across a block boundary.  That
residual is why the contract here is **measured**, not proved: with no ties in
`vals` (the overwhelmingly common case, since `des` is a difference of
floating-point centres) the two permutations are *identical* and bit equality
is a theorem; with ties, `tests/test_partner_refine_qsweep.py` asserts `==` on
adversarially tie-dense inputs and on whole `refine_prediction` runs.

Ported under the additional sub-switch `PARTNER_REFINE_KERNEL_DISC=1`: the
discrete-move family.  `PARTNER_REFINE_PROF_DISC` (second-tier `run()` timers,
`_DPROF_KEYS` in `layout_refiner.py`) splits the `discrete` bucket -- 58% of
`run`'s wall clock on the real large-band cases -- as

    lc_axis  (the `_axis_pass(hold=True)` pairs of `_legal_check`)   91.4 %
    screen_delta (`_block_hp` pair deltas)                            3.7 %
    enum_optpt / enum_gain / screen_np / edits / bookkeeping        ~ 4.3 %

so the interesting number is that the 91% is *already numba* and what it
still pays is the BOUNDARY.  Hence two parts, selected with
`PARTNER_REFINE_DISC_PARTS` (default `fuse,hp`):

  * `fuse` -- `_legal_sweeps_k`: `_legal_check`'s `3x(axis 0, axis 1)` +
    overlap loop as ONE njit call.  Removes 6 numba dispatches and 6 O(G)
    Python `pin_x`/`pin_y` re-syncs per swap attempt (the re-sync alone
    measures 28% of one `axis_pass_hold` at n=118/G=99).  Syncing once is
    sound because nothing inside the fused loop can write a `_Group` flag:
    `_axis_pass_hold_k` writes `P`, `_has_overlap_k` reads it.
  * `hp`   -- `_block_hp_k`, `_optimal_point_k`, `_discrete_gains_k`,
    `_swap_delta_k` over a per-block CSR of `badj` / `bpin` built in
    `_build_badj`'s append order, so the float accumulation order (hence the
    sum, bit for bit) is the Python one.

Measured on `_Refiner.run` at converged spans, n = 100-120: `fuse` 1.13-1.18x,
`hp` 1.04-1.05x, together 1.18-1.25x; in the 0.23 s deadline regime the same
change buys +23.6% discrete swap attempts inside the same wall clock.

`hp` inherits `_wmedian_k`'s stable-vs-introsort residual through
`_optimal_point_k`; `tests/test_partner_refine_disc.py` measures it on
tie-saturated inputs, exactly as the QSWEEP file does.

Still NOT ported: nothing of the discrete family that the profile prices
above 1%.  `_key` (`opt._hpwl` + `opt._violations`) is deliberately left
alone -- it is vectorized numpy plus an existing `PARTNER_SA_KERNEL` path,
and in the 0.23 s regime `_legal_check` reaches it on a minority of attempts.

Activation
----------
`PARTNER_REFINE_KERNEL=numba` (+ optional `PARTNER_REFINE_KERNEL_QSWEEP=1`,
`PARTNER_REFINE_KERNEL_DISC=1`).
Unset (the default) leaves the refiner on the existing Python path with no
behavioural change whatsoever: `_Refiner._nk` is `None` and every dispatch
site is a single `is None` test.  With `PARTNER_REFINE_KERNEL=numba` but
`PARTNER_REFINE_KERNEL_QSWEEP` unset, `RefineKernel.qsweep` is `False`, the
median-sweep CSR is not even built, and `_axis_pass(hold=False)` keeps the
Python path exactly as it shipped.
"""

from __future__ import annotations

import os
from typing import Optional

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


# must match `layout_refiner.SEP_TOL`
SEP_TOL = 5e-7
# `_has_overlap` / `_overlap_count` use their own literal
OVL_TOL = 9e-7

_TRUE = ("1", "true", "True", "on", "ON")


# =============================================================================
# njit leaves
# =============================================================================
@njit(cache=True)
def _argsort_stable_k(a):
    """The kernel's stable sort, exposed so the test file can assert it agrees
    with `np.argsort(..., kind="stable")` instead of arguing that it must."""
    return np.argsort(a, kind='mergesort')


@njit(cache=True)
def _has_overlap_k(P, n):
    for i in range(n):
        x0i = P[i, 0]
        x1i = x0i + P[i, 2]
        y0i = P[i, 1]
        y1i = y0i + P[i, 3]
        for j in range(i + 1, n):
            x0j = P[j, 0]
            ox = min(x1i, x0j + P[j, 2]) - max(x0i, x0j)
            if ox <= OVL_TOL:
                continue
            y0j = P[j, 1]
            oy = min(y1i, y0j + P[j, 3]) - max(y0i, y0j)
            if oy > OVL_TOL:
                return True
    return False


@njit(cache=True)
def _overlap_count_k(P, n):
    cnt = 0
    for i in range(n):
        x0i = P[i, 0]
        x1i = x0i + P[i, 2]
        y0i = P[i, 1]
        y1i = y0i + P[i, 3]
        for j in range(i + 1, n):
            x0j = P[j, 0]
            ox = min(x1i, x0j + P[j, 2]) - max(x0i, x0j)
            if ox <= OVL_TOL:
                continue
            y0j = P[j, 1]
            oy = min(y1i, y0j + P[j, 3]) - max(y0i, y0j)
            if oy > OVL_TOL:
                cnt += 1
    return cnt


@njit(cache=True)
def _evict_scan_k(P, n, i, order, max_anchors, is_soft, area,
                  xmin, xmax, ymin, ymax):
    """The `_evict` anchor x variant scan.  Writes `P[i]` and returns 1 on the
    first overlap-free landing, 0 if every candidate clashed."""
    nb = order.shape[0]
    if max_anchors < nb:
        nb = max_anchors
    vx = np.empty(4)
    vy = np.empty(4)
    vw = np.empty(4)
    vh = np.empty(4)
    for oi in range(nb):
        j = order[oi]
        if j == i:
            continue
        xj = P[j, 0]
        yj = P[j, 1]
        wj = P[j, 2]
        hj = P[j, 3]
        if is_soft:
            nw = wj
            nh = area / wj
            vx[0] = xj
            vy[0] = yj + hj
            vw[0] = nw
            vh[0] = nh
            vx[1] = xj
            vy[1] = yj - nh
            vw[1] = nw
            vh[1] = nh
            nh2 = hj
            nw2 = area / hj
            vx[2] = xj + wj
            vy[2] = yj
            vw[2] = nw2
            vh[2] = nh2
            vx[3] = xj - nw2
            vy[3] = yj
            vw[3] = nw2
            vh[3] = nh2
        else:
            nw = P[i, 2]
            nh = P[i, 3]
            vx[0] = xj
            vy[0] = yj + hj
            vx[1] = xj
            vy[1] = yj - nh
            vx[2] = xj + wj
            vy[2] = yj
            vx[3] = xj - nw
            vy[3] = yj
            for v in range(4):
                vw[v] = nw
                vh[v] = nh
        for v in range(4):
            nw_ = vw[v]
            nh_ = vh[v]
            nx = min(max(vx[v], xmin), xmax - nw_)
            ny = min(max(vy[v], ymin), ymax - nh_)
            nx1 = nx + nw_
            ny1 = ny + nh_
            clash = False
            for k in range(n):
                if k == i:
                    continue
                x0k = P[k, 0]
                ox = min(nx1, x0k + P[k, 2]) - max(nx, x0k)
                if ox <= SEP_TOL:
                    continue
                y0k = P[k, 1]
                oy = min(ny1, y0k + P[k, 3]) - max(ny, y0k)
                if oy > SEP_TOL:
                    clash = True
                    break
            if not clash:
                P[i, 0] = nx
                P[i, 1] = ny
                P[i, 2] = nw_
                P[i, 3] = nh_
                return 1
    return 0


@njit(cache=True)
def _move_group_k(P, gi, dv, axis, grp_ptr, grp_mem,
                  ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                  snap_key, snap_ord):
    """`_Refiner._move`: rigid shift of group `gi` plus the cluster contact
    re-snap, `sorted(g.cH, key=lambda ab: P[a, 0])` -- keys snapshotted
    before any assignment, stable insertion sort, then the writes."""
    for t in range(grp_ptr[gi], grp_ptr[gi + 1]):
        P[grp_mem[t], axis] += dv
    if axis == 0:
        lo_c = ch_ptr[gi]
        m = ch_ptr[gi + 1] - lo_c
        for u in range(m):
            snap_key[u] = P[ch_a[lo_c + u], 0]
            snap_ord[u] = u
        for u in range(1, m):
            kk = snap_ord[u]
            kv = snap_key[kk]
            v2 = u - 1
            while v2 >= 0 and snap_key[snap_ord[v2]] > kv:
                snap_ord[v2 + 1] = snap_ord[v2]
                v2 -= 1
            snap_ord[v2 + 1] = kk
        for u in range(m):
            t = lo_c + snap_ord[u]
            a = ch_a[t]
            P[ch_b[t], 0] = P[a, 0] + P[a, 2]
    else:
        lo_c = cv_ptr[gi]
        m = cv_ptr[gi + 1] - lo_c
        for u in range(m):
            snap_key[u] = P[cv_a[lo_c + u], 1]
            snap_ord[u] = u
        for u in range(1, m):
            kk = snap_ord[u]
            kv = snap_key[kk]
            v2 = u - 1
            while v2 >= 0 and snap_key[snap_ord[v2]] > kv:
                snap_ord[v2 + 1] = snap_ord[v2]
                v2 -= 1
            snap_ord[v2 + 1] = kk
        for u in range(m):
            t = lo_c + snap_ord[u]
            a = cv_a[t]
            P[cv_b[t], 1] = P[a, 1] + P[a, 3]


@njit(cache=True)
def _wmedian_k(vals, wts, m, sv, sc):
    """`_Refiner._wmedian` over the first `m` entries of two scratch buffers.

    `np.cumsum` is a plain sequential accumulation, so the running `acc` here
    is bit-identical to it *given the same permutation*; the binary search is
    `np.searchsorted(c, half)` (side='left') for the non-decreasing `c` that
    non-negative weights guarantee.  The sort is stable rather than numpy's
    introsort -- see the module docstring for why that is value-preserving.
    """
    order = np.argsort(vals[:m], kind='mergesort')
    acc = 0.0
    for k in range(m):
        o = order[k]
        sv[k] = vals[o]
        acc += wts[o]
        sc[k] = acc
    half = 0.5 * sc[m - 1]
    lo = 0
    hi = m
    while lo < hi:
        mid = (lo + hi) // 2
        if sc[mid] < half:
            lo = mid + 1
        else:
            hi = mid
    if lo >= m:
        # unreachable for non-negative weights (half <= c[-1]); numpy would
        # raise IndexError here, so clamping cannot mask a live divergence.
        lo = m - 1
    return sv[lo]


@njit(cache=True)
def _median_shift_k(P, axis, gi, ge_ptr, ge_M, ge_J, ge_W,
                    gp_ptr, gp_M, gp_X, gp_Y, gp_W, des, wts, sv, sc):
    """`_Refiner._median_shift`: returns (found, shift).  `found=False` is the
    Python path's `None` (no external edge and no pin for this group)."""
    e0 = ge_ptr[gi]
    ne = ge_ptr[gi + 1] - e0
    p0 = gp_ptr[gi]
    npn = gp_ptr[gi + 1] - p0
    if ne == 0 and npn == 0:
        return False, 0.0
    for k in range(ne):
        m = ge_M[e0 + k]
        j = ge_J[e0 + k]
        des[k] = ((P[j, axis] + 0.5 * P[j, axis + 2])
                  - (P[m, axis] + 0.5 * P[m, axis + 2]))
        wts[k] = ge_W[e0 + k]
    for k in range(npn):
        b = gp_M[p0 + k]
        tgt = gp_X[p0 + k] if axis == 0 else gp_Y[p0 + k]
        des[ne + k] = tgt - (P[b, axis] + 0.5 * P[b, axis + 2])
        wts[ne + k] = gp_W[p0 + k]
    return True, _wmedian_k(des, wts, ne + npn, sv, sc)


# =============================================================================
# the fused axis pass: shared `_axis_constraints` prologue, then one of the
# two forward passes (hold=True legalization / hold=False median sweep)
# =============================================================================
@njit(cache=True)
def _axis_build_k(P, n, G, axis, invert, lim_lo, lim_hi,
                  group_of, kind, grp_ptr, grp_mem, pinx, piny,
                  c0, c1, f0, f1, cen, pin_this, pin_oth,
                  lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                  ein_ptr, ein_e, eout_ptr, eout_e, cur, key_of):
    """`_Refiner._axis_constraints`: fills `lo_f` / `dmax` / the edge CSR and
    returns the group `order`.  Byte-for-byte the prologue that shipped fused
    into `_axis_pass_hold_k`; both forward passes now call it, so the two
    sweeps provably optimize over the same feasible set (a divergence would
    be a legality hole)."""
    o = 1 - axis
    for i in range(n):
        a0 = P[i, axis]
        c0[i] = a0
        c1[i] = a0 + P[i, axis + 2]
        cen[i] = a0 + 0.5 * P[i, axis + 2]
        b0 = P[i, o]
        f0[i] = b0
        f1[i] = b0 + P[i, o + 2]
        pt = 1 if kind[i] == 2 else 0
        pin_this[i] = pt
        pin_oth[i] = pt

    # -- (A) per-block pin flags: `pt[g.members] = True` over every group ----
    for g in range(G):
        this_on = pinx[g] if axis == 0 else piny[g]
        oth_on = piny[g] if axis == 0 else pinx[g]
        if this_on:
            for t in range(grp_ptr[g], grp_ptr[g + 1]):
                pin_this[grp_mem[t]] = 1
        if oth_on:
            for t in range(grp_ptr[g], grp_ptr[g + 1]):
                pin_oth[grp_mem[t]] = 1

    NEG = -1e18
    POS = 1e18
    for g in range(G):
        lo_f[g] = NEG
        hi_f[g] = POS
    for t in range(G * G):
        tight_idx[t] = -1

    # -- (A+B) pair scan -> per-group difference constraints ----------------
    ne = 0
    for i in range(n):
        f1i = f1[i]
        f0i = f0[i]
        c0i = c0[i]
        c1i = c1[i]
        ceni = cen[i]
        gi_ = group_of[i]
        pti = pin_this[i]
        poi = pin_oth[i]
        for j in range(n):
            if i == j:
                continue
            cenj = cen[j]
            if not (ceni < cenj or (ceni == cenj and i < j)):
                continue
            ovf = min(f1i, f1[j]) - max(f0i, f0[j])
            if ovf <= SEP_TOL:
                continue
            ovm = min(c1i, c1[j]) - max(c0i, c0[j])
            if ovm > SEP_TOL:
                if pti == 1 and pin_this[j] == 1:
                    continue
                can_oth = not (poi == 1 and pin_oth[j] == 1)
                if invert:
                    pref = ovm >= ovf
                else:
                    pref = ovm <= ovf
                if can_oth and not pref:
                    continue
            gj_ = group_of[j]
            if gi_ == gj_ and gi_ >= 0:
                continue
            c = c1i - c0[j]
            if gi_ >= 0 and gj_ >= 0:
                key = gi_ * G + gj_
                p = tight_idx[key]
                if p < 0:
                    tight_idx[key] = ne
                    eg_i[ne] = gi_
                    eg_j[ne] = gj_
                    eg_c[ne] = c
                    ne += 1
                elif c > eg_c[p]:
                    eg_c[p] = c
            elif gj_ >= 0:
                if c > lo_f[gj_]:
                    lo_f[gj_] = c
            elif gi_ >= 0:
                if -c < hi_f[gi_]:
                    hi_f[gi_] = -c

    # -- edges_in / edges_out CSR, stable in append order -------------------
    for g in range(G + 1):
        ein_ptr[g] = 0
        eout_ptr[g] = 0
    for e in range(ne):
        ein_ptr[eg_j[e] + 1] += 1
        eout_ptr[eg_i[e] + 1] += 1
    for g in range(G):
        ein_ptr[g + 1] += ein_ptr[g]
        eout_ptr[g + 1] += eout_ptr[g]
    for g in range(G):
        cur[g] = ein_ptr[g]
    for e in range(ne):
        p = cur[eg_j[e]]
        ein_e[p] = e
        cur[eg_j[e]] = p + 1
    for g in range(G):
        cur[g] = eout_ptr[g]
    for e in range(ne):
        p = cur[eg_i[e]]
        eout_e[p] = e
        cur[eg_i[e]] = p + 1

    # -- (C) frame + pin bounds per group -----------------------------------
    for g in range(G):
        mn = 1e300
        mx = -1e300
        for t in range(grp_ptr[g], grp_ptr[g + 1]):
            m = grp_mem[t]
            if c0[m] < mn:
                mn = c0[m]
            if c1[m] > mx:
                mx = c1[m]
        v = lim_lo - mn
        if v > lo_f[g]:
            lo_f[g] = v
        v = lim_hi - mx
        if v < hi_f[g]:
            hi_f[g] = v
        key_of[g] = mn
        if (pinx[g] if axis == 0 else piny[g]):
            lo_f[g] = 0.0
            hi_f[g] = 0.0

    order = np.argsort(key_of[:G], kind='mergesort')

    # -- (D) backward longest path ------------------------------------------
    for g in range(G):
        dmax[g] = hi_f[g]
    for k in range(G - 1, -1, -1):
        gi = order[k]
        for t in range(eout_ptr[gi], eout_ptr[gi + 1]):
            e = eout_e[t]
            v = dmax[eg_j[e]] - eg_c[e]
            if v < dmax[gi]:
                dmax[gi] = v
    return order


@njit(cache=True)
def _axis_pass_hold_k(P, n, G, axis, invert, lim_lo, lim_hi,
                      group_of, kind, grp_ptr, grp_mem,
                      ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                      pinx, piny,
                      c0, c1, f0, f1, cen, pin_this, pin_oth,
                      lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                      ein_ptr, ein_e, eout_ptr, eout_e, cur,
                      key_of, d, assigned, snap_key, snap_ord):
    order = _axis_build_k(P, n, G, axis, invert, lim_lo, lim_hi,
                          group_of, kind, grp_ptr, grp_mem, pinx, piny,
                          c0, c1, f0, f1, cen, pin_this, pin_oth,
                          lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                          ein_ptr, ein_e, eout_ptr, eout_e, cur, key_of)

    # -- (E) forward assignment + rigid moves; every target is "stay put" ----
    for g in range(G):
        d[g] = 0.0
        assigned[g] = 0
    for k in range(G):
        gi = order[k]
        if (pinx[gi] if axis == 0 else piny[gi]):
            assigned[gi] = 1
            continue
        lo = lo_f[gi]
        for t in range(ein_ptr[gi], ein_ptr[gi + 1]):
            e = ein_e[t]
            gp = eg_i[e]
            base = d[gp] if assigned[gp] == 1 else 0.0
            v = base + eg_c[e]
            if v > lo:
                lo = v
        hi = dmax[gi]
        for t in range(eout_ptr[gi], eout_ptr[gi + 1]):
            e = eout_e[t]
            gj = eg_j[e]
            if assigned[gj] == 1:
                v = d[gj] - eg_c[e]
                if v < hi:
                    hi = v
        if lo > hi:
            assigned[gi] = 1
            continue
        dv = min(max(0.0, lo), hi)
        d[gi] = dv
        assigned[gi] = 1
        if abs(dv) > 1e-12:
            _move_group_k(P, gi, dv, axis, grp_ptr, grp_mem,
                          ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                          snap_key, snap_ord)


@njit(cache=True)
def _legal_sweeps_k(P, n, G, lim_lo_x, lim_hi_x, lim_lo_y, lim_hi_y,
                    group_of, kind, grp_ptr, grp_mem,
                    ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                    pinx, piny,
                    c0, c1, f0, f1, cen, pin_this, pin_oth,
                    lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                    ein_ptr, ein_e, eout_ptr, eout_e, cur,
                    key_of, d, assigned, snap_key, snap_ord):
    """`_Refiner._legal_check`'s legalization loop, fused.

    Transcription of

        for _ in range(3):
            self._axis_pass(0, hold=True)
            self._axis_pass(1, hold=True)
            if not self._has_overlap():
                break
        if self._has_overlap():
            ...

    -- the same two `_axis_pass_hold_k` calls in the same order, the same
    early exit, and the trailing re-test (`_has_overlap_k` is a pure function
    of `P`, so returning the loop's last value would be the same number; it is
    recomputed anyway to keep the transcription trivially faithful).  Returns
    the final overlap flag.

    The win is not new numerics -- every arithmetic line already ran in numba.
    It is the *boundary*: the Python path pays 6 numba dispatches and, worse,
    6 O(G) Python `pin_x`/`pin_y` re-syncs (measured 28% of one
    `axis_pass_hold` at n=118/G=99) for a call sequence that provably cannot
    change a pin flag -- `_axis_pass_hold_k` writes `P` only, `_has_overlap_k`
    reads it.  Here the flags are synced once by the caller.
    """
    for _ in range(3):
        _axis_pass_hold_k(P, n, G, 0, False, lim_lo_x, lim_hi_x,
                          group_of, kind, grp_ptr, grp_mem,
                          ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                          pinx, piny,
                          c0, c1, f0, f1, cen, pin_this, pin_oth,
                          lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                          ein_ptr, ein_e, eout_ptr, eout_e, cur,
                          key_of, d, assigned, snap_key, snap_ord)
        _axis_pass_hold_k(P, n, G, 1, False, lim_lo_y, lim_hi_y,
                          group_of, kind, grp_ptr, grp_mem,
                          ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                          pinx, piny,
                          c0, c1, f0, f1, cen, pin_this, pin_oth,
                          lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                          ein_ptr, ein_e, eout_ptr, eout_e, cur,
                          key_of, d, assigned, snap_key, snap_ord)
        if not _has_overlap_k(P, n):
            break
    return _has_overlap_k(P, n)


@njit(cache=True)
def _block_hp_k(P, i, cx, cy, ba_ptr, ba_j, ba_w,
                bp_ptr, bp_X, bp_Y, bp_W, ex, nex):
    """`_Refiner._block_hp`.

    Same accumulation order as the Python loops -- `badj[i]` in CSR order
    (which IS `_build_badj`'s append order), then `bpin[i]` -- so the float
    sum is bit-identical, not merely equal to rounding.  `ex[:nex]` is the
    `exclude` tuple, tested with the same `in` semantics (it is 0 or 2
    elements at the pairwise sites and <= 8 in `_matching_batch`, so the
    linear scan is what Python does too).
    """
    s = 0.0
    for t in range(ba_ptr[i], ba_ptr[i + 1]):
        j = ba_j[t]
        skip = False
        for e in range(nex):
            if ex[e] == j:
                skip = True
                break
        if skip:
            continue
        s += ba_w[t] * (abs(cx - (P[j, 0] + 0.5 * P[j, 2]))
                        + abs(cy - (P[j, 1] + 0.5 * P[j, 3])))
    for t in range(bp_ptr[i], bp_ptr[i + 1]):
        s += bp_W[t] * (abs(cx - bp_X[t]) + abs(cy - bp_Y[t]))
    return s


@njit(cache=True)
def _optimal_point_k(P, i, ba_ptr, ba_j, ba_w, bp_ptr, bp_X, bp_Y, bp_W,
                     xs, ys, ws, sv, sc):
    """`_Refiner._optimal_point`: returns (found, ox, oy).

    `found=False` is the Python path's `None` (block with no incident edge and
    no pin).  The two `_wmedian` calls share one weight buffer, exactly as the
    Python path shares `wa`.  Inherits `_wmedian_k`'s stable-sort caveat (see
    the module docstring); the values fed here are block CENTRES rather than
    the centre *differences* `_median_shift` feeds, so exact ties are if
    anything rarer.
    """
    m = 0
    for t in range(ba_ptr[i], ba_ptr[i + 1]):
        j = ba_j[t]
        xs[m] = P[j, 0] + 0.5 * P[j, 2]
        ys[m] = P[j, 1] + 0.5 * P[j, 3]
        ws[m] = ba_w[t]
        m += 1
    for t in range(bp_ptr[i], bp_ptr[i + 1]):
        xs[m] = bp_X[t]
        ys[m] = bp_Y[t]
        ws[m] = bp_W[t]
        m += 1
    if m == 0:
        return False, 0.0, 0.0
    ox = _wmedian_k(xs, ws, m, sv, sc)
    oy = _wmedian_k(ys, ws, m, sv, sc)
    return True, ox, oy


@njit(cache=True)
def _discrete_gains_k(P, idxs, ba_ptr, ba_j, ba_w, bp_ptr, bp_X, bp_Y, bp_W,
                      xs, ys, ws, sv, sc, gain, opx, opy, keep):
    """The `_discrete_batch` / `_matching_batch` candidate scan, fused.

    Per swappable block: `_optimal_point`, then the two `_block_hp` calls that
    score it, then the `gain > 1e-9` filter -- the identical sequence, so
    `keep`/`gain`/`opx`/`opy` reproduce the Python `cands` list entry for
    entry (the caller still does the `sort(reverse=True)`, which is Python's
    stable tuple sort and has no kernel equivalent worth the risk).
    """
    ex = np.empty(1, dtype=np.int64)
    for t in range(idxs.shape[0]):
        i = idxs[t]
        found, ox, oy = _optimal_point_k(P, i, ba_ptr, ba_j, ba_w,
                                         bp_ptr, bp_X, bp_Y, bp_W,
                                         xs, ys, ws, sv, sc)
        if not found:
            keep[t] = 0
            continue
        cxi = P[i, 0] + 0.5 * P[i, 2]
        cyi = P[i, 1] + 0.5 * P[i, 3]
        g = (_block_hp_k(P, i, cxi, cyi, ba_ptr, ba_j, ba_w,
                         bp_ptr, bp_X, bp_Y, bp_W, ex, 0)
             - _block_hp_k(P, i, ox, oy, ba_ptr, ba_j, ba_w,
                           bp_ptr, bp_X, bp_Y, bp_W, ex, 0))
        gain[t] = g
        opx[t] = ox
        opy[t] = oy
        keep[t] = 1 if g > 1e-9 else 0


@njit(cache=True)
def _swap_delta_k(P, i, j, cxi, cyi, cxj, cyj,
                  ba_ptr, ba_j, ba_w, bp_ptr, bp_X, bp_Y, bp_W, ex):
    """The four-term pair delta of `_discrete_batch`, in the same order."""
    ex[0] = i
    ex[1] = j
    return (_block_hp_k(P, i, cxj, cyj, ba_ptr, ba_j, ba_w,
                        bp_ptr, bp_X, bp_Y, bp_W, ex, 2)
            + _block_hp_k(P, j, cxi, cyi, ba_ptr, ba_j, ba_w,
                          bp_ptr, bp_X, bp_Y, bp_W, ex, 2)
            - _block_hp_k(P, i, cxi, cyi, ba_ptr, ba_j, ba_w,
                          bp_ptr, bp_X, bp_Y, bp_W, ex, 2)
            - _block_hp_k(P, j, cxj, cyj, ba_ptr, ba_j, ba_w,
                          bp_ptr, bp_X, bp_Y, bp_W, ex, 2))


@njit(cache=True)
def _axis_pass_soft_k(P, n, G, axis, invert, lim_lo, lim_hi,
                      group_of, kind, grp_ptr, grp_mem,
                      ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                      pinx, piny,
                      c0, c1, f0, f1, cen, pin_this, pin_oth,
                      lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                      ein_ptr, ein_e, eout_ptr, eout_e, cur,
                      key_of, d, assigned, snap_key, snap_ord,
                      ge_ptr, ge_M, ge_J, ge_W,
                      gp_ptr, gp_M, gp_X, gp_Y, gp_W,
                      des, wts, sv, sc, max_step, has_max_step):
    """`_axis_pass(hold=False)`: same polytope, HPWL weighted-median targets.

    Note where `_median_shift_k` is called -- *after* this group's `lo` / `hi`
    and *after* every earlier group in `order` has already moved `P`.  The
    sweep is Gauss-Seidel, so evaluating the target any earlier would be a
    different algorithm, not a faster one."""
    order = _axis_build_k(P, n, G, axis, invert, lim_lo, lim_hi,
                          group_of, kind, grp_ptr, grp_mem, pinx, piny,
                          c0, c1, f0, f1, cen, pin_this, pin_oth,
                          lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                          ein_ptr, ein_e, eout_ptr, eout_e, cur, key_of)

    for g in range(G):
        d[g] = 0.0
        assigned[g] = 0
    for k in range(G):
        gi = order[k]
        if (pinx[gi] if axis == 0 else piny[gi]):
            assigned[gi] = 1
            continue
        lo = lo_f[gi]
        for t in range(ein_ptr[gi], ein_ptr[gi + 1]):
            e = ein_e[t]
            gp = eg_i[e]
            base = d[gp] if assigned[gp] == 1 else 0.0
            v = base + eg_c[e]
            if v > lo:
                lo = v
        hi = dmax[gi]
        for t in range(eout_ptr[gi], eout_ptr[gi + 1]):
            e = eout_e[t]
            gj = eg_j[e]
            if assigned[gj] == 1:
                v = d[gj] - eg_c[e]
                if v < hi:
                    hi = v
        if lo > hi:
            assigned[gi] = 1
            continue
        found, t_ = _median_shift_k(P, axis, gi, ge_ptr, ge_M, ge_J, ge_W,
                                    gp_ptr, gp_M, gp_X, gp_Y, gp_W,
                                    des, wts, sv, sc)
        if not found:
            t_ = 0.0
        if has_max_step:
            t_ = min(max(t_, -max_step), max_step)
        dv = min(max(t_, lo), hi)
        d[gi] = dv
        assigned[gi] = 1
        if abs(dv) > 1e-12:
            _move_group_k(P, gi, dv, axis, grp_ptr, grp_mem,
                          ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                          snap_key, snap_ord)


# =============================================================================
# host-side wrapper
# =============================================================================
class RefineKernel:
    """Flat mirror of a `_Refiner`'s static structure + its scratch buffers.

    Group membership, `group_of`, `kind`, `in_cluster` and the cluster contact
    lists are all frozen after `_Refiner.__init__` (verified by inspection: the
    only post-init group mutations in `layout_refiner.py` are `pin_x` /
    `pin_y`), so the CSR is built once.  The pin flags are re-synced on every
    call -- the same `for g in self.groups` walk the Python path already did.
    """

    __slots__ = ("ref", "n", "G", "group_of", "kind", "grp_ptr", "grp_mem",
                 "ch_ptr", "ch_a", "ch_b", "cv_ptr", "cv_a", "cv_b",
                 "pinx", "piny", "c0", "c1", "f0", "f1", "cen",
                 "pin_this", "pin_oth", "lo_f", "hi_f", "dmax", "tight_idx",
                 "eg_i", "eg_j", "eg_c", "ein_ptr", "ein_e", "eout_ptr",
                 "eout_e", "cur", "key_of", "d", "assigned", "snap_key",
                 "snap_ord", "calls",
                 "qsweep", "qcalls", "ge_ptr", "ge_M", "ge_J", "ge_W",
                 "gp_ptr", "gp_M", "gp_X", "gp_Y", "gp_W",
                 "des", "wts", "sv", "sc",
                 "disc", "dcalls", "ba_ptr", "ba_j", "ba_w",
                 "bp_ptr", "bp_X", "bp_Y", "bp_W",
                 "b_xs", "b_ys", "b_ws", "b_sv", "b_sc", "b_ex",
                 "d_gain", "d_opx", "d_opy", "d_keep")

    def __init__(self, ref, qsweep: bool = False, disc: bool = False):
        self.ref = ref
        n = int(ref.n)
        groups = ref.groups
        G = len(groups)
        self.n = n
        self.G = G
        self.group_of = np.asarray(ref.group_of, dtype=np.int64)
        self.kind = np.asarray(ref.kind, dtype=np.int64)

        grp_ptr = np.zeros(G + 1, dtype=np.int64)
        for gi, g in enumerate(groups):
            grp_ptr[gi + 1] = grp_ptr[gi] + len(g.members)
        grp_mem = np.zeros(max(int(grp_ptr[G]), 1), dtype=np.int64)
        for gi, g in enumerate(groups):
            grp_mem[grp_ptr[gi]:grp_ptr[gi + 1]] = g.members
        self.grp_ptr = grp_ptr
        self.grp_mem = grp_mem

        self.ch_ptr, self.ch_a, self.ch_b = _contact_csr(groups, "cH")
        self.cv_ptr, self.cv_a, self.cv_b = _contact_csr(groups, "cV")

        self.pinx = np.zeros(max(G, 1), dtype=np.int64)
        self.piny = np.zeros(max(G, 1), dtype=np.int64)
        self.c0 = np.zeros(n)
        self.c1 = np.zeros(n)
        self.f0 = np.zeros(n)
        self.f1 = np.zeros(n)
        self.cen = np.zeros(n)
        self.pin_this = np.zeros(n, dtype=np.int64)
        self.pin_oth = np.zeros(n, dtype=np.int64)
        self.lo_f = np.zeros(max(G, 1))
        self.hi_f = np.zeros(max(G, 1))
        self.dmax = np.zeros(max(G, 1))
        self.tight_idx = np.zeros(max(G * G, 1), dtype=np.int64)
        cap = max(G * G, 1)
        self.eg_i = np.zeros(cap, dtype=np.int64)
        self.eg_j = np.zeros(cap, dtype=np.int64)
        self.eg_c = np.zeros(cap)
        self.ein_ptr = np.zeros(G + 1, dtype=np.int64)
        self.eout_ptr = np.zeros(G + 1, dtype=np.int64)
        self.ein_e = np.zeros(cap, dtype=np.int64)
        self.eout_e = np.zeros(cap, dtype=np.int64)
        self.cur = np.zeros(max(G, 1), dtype=np.int64)
        self.key_of = np.zeros(max(G, 1))
        self.d = np.zeros(max(G, 1))
        self.assigned = np.zeros(max(G, 1), dtype=np.int64)
        mc = max(int(self.ch_ptr[G]), int(self.cv_ptr[G]), 1)
        self.snap_key = np.zeros(mc)
        self.snap_ord = np.zeros(mc, dtype=np.int64)
        self.calls = 0

        # -- PARTNER_REFINE_KERNEL_QSWEEP companions ----------------------
        # `g.eM/eJ/eW/pM/pX/pY/pW` are written once, in `_build_edges`, and
        # never touched again (the only post-init group mutations anywhere in
        # `layout_refiner.py` are `pin_x` / `pin_y`), so this CSR is built
        # once too.  Off by default: not building it keeps the promoted
        # `hold=True` kernel's `__init__` cost exactly where it was.
        self.qsweep = bool(qsweep)
        self.qcalls = 0
        if self.qsweep:
            self.ge_ptr, self.ge_M, self.ge_J, self.ge_W = _edge_csr(
                groups, "eM", "eJ", "eW")
            self.gp_ptr, self.gp_M, self.gp_X, self.gp_Y, self.gp_W = \
                _pin_csr(groups)
            widest = 1
            for gi in range(G):
                w = ((self.ge_ptr[gi + 1] - self.ge_ptr[gi])
                     + (self.gp_ptr[gi + 1] - self.gp_ptr[gi]))
                if w > widest:
                    widest = int(w)
            self.des = np.zeros(widest)
            self.wts = np.zeros(widest)
            self.sv = np.zeros(widest)
            self.sc = np.zeros(widest)
        else:
            z1 = np.zeros(1)
            zi = np.zeros(1, dtype=np.int64)
            self.ge_ptr = np.zeros(G + 1, dtype=np.int64)
            self.ge_M = zi
            self.ge_J = zi
            self.ge_W = z1
            self.gp_ptr = np.zeros(G + 1, dtype=np.int64)
            self.gp_M = zi
            self.gp_X = z1
            self.gp_Y = z1
            self.gp_W = z1
            self.des = z1
            self.wts = z1
            self.sv = z1
            self.sc = z1

        # -- PARTNER_REFINE_KERNEL_DISC companions -------------------------
        # `badj` / `bpin` are written once by `_build_badj` and never mutated
        # (the only post-init mutations anywhere in `layout_refiner.py` are
        # `pin_x` / `pin_y` and `P`), so this CSR is built once.  Off by
        # default: not building it keeps the promoted kernel's `__init__` cost
        # exactly where it was.
        self.disc = bool(disc)
        self.dcalls = 0
        if self.disc:
            badj = ref.badj
            bpin = ref.bpin
            ba_ptr = np.zeros(n + 1, dtype=np.int64)
            bp_ptr = np.zeros(n + 1, dtype=np.int64)
            for i in range(n):
                ba_ptr[i + 1] = ba_ptr[i] + len(badj[i])
                bp_ptr[i + 1] = bp_ptr[i] + len(bpin[i])
            ba_j = np.zeros(max(int(ba_ptr[n]), 1), dtype=np.int64)
            ba_w = np.zeros(max(int(ba_ptr[n]), 1))
            bp_X = np.zeros(max(int(bp_ptr[n]), 1))
            bp_Y = np.zeros(max(int(bp_ptr[n]), 1))
            bp_W = np.zeros(max(int(bp_ptr[n]), 1))
            widest = 1
            for i in range(n):
                t = int(ba_ptr[i])
                for (j, w) in badj[i]:
                    ba_j[t] = j
                    ba_w[t] = w
                    t += 1
                t = int(bp_ptr[i])
                for (px, py, w) in bpin[i]:
                    bp_X[t] = px
                    bp_Y[t] = py
                    bp_W[t] = w
                    t += 1
                deg = len(badj[i]) + len(bpin[i])
                if deg > widest:
                    widest = deg
            self.ba_ptr, self.ba_j, self.ba_w = ba_ptr, ba_j, ba_w
            self.bp_ptr, self.bp_X, self.bp_Y, self.bp_W = \
                bp_ptr, bp_X, bp_Y, bp_W
            self.b_xs = np.zeros(widest)
            self.b_ys = np.zeros(widest)
            self.b_ws = np.zeros(widest)
            self.b_sv = np.zeros(widest)
            self.b_sc = np.zeros(widest)
            # `exclude` is <= 8 (`_matching_batch`'s pool cap K)
            self.b_ex = np.zeros(max(n, 8), dtype=np.int64)
            self.d_gain = np.zeros(max(n, 1))
            self.d_opx = np.zeros(max(n, 1))
            self.d_opy = np.zeros(max(n, 1))
            self.d_keep = np.zeros(max(n, 1), dtype=np.int64)
        else:
            z1 = np.zeros(1)
            zi = np.zeros(1, dtype=np.int64)
            self.ba_ptr = np.zeros(n + 1, dtype=np.int64)
            self.ba_j = zi
            self.ba_w = z1
            self.bp_ptr = np.zeros(n + 1, dtype=np.int64)
            self.bp_X = z1
            self.bp_Y = z1
            self.bp_W = z1
            self.b_xs = self.b_ys = self.b_ws = z1
            self.b_sv = self.b_sc = z1
            self.b_ex = zi
            self.d_gain = self.d_opx = self.d_opy = z1
            self.d_keep = zi

    # -- hot path -------------------------------------------------------
    def sync_pins(self) -> None:
        """Mirror `g.pin_x` / `g.pin_y` into the flat arrays."""
        pinx = self.pinx
        piny = self.piny
        for gi, g in enumerate(self.ref.groups):
            pinx[gi] = 1 if g.pin_x else 0
            piny[gi] = 1 if g.pin_y else 0

    def legal_sweeps(self) -> bool:
        """`_legal_check`'s 3x(axis 0, axis 1) + overlap loop, fused.

        Only reachable with `self.disc`.  Pins are synced ONCE: nothing inside
        `_legal_sweeps_k` can write a `_Group` flag."""
        ref = self.ref
        G = self.G
        self.dcalls += 1
        if G == 0:
            return bool(_has_overlap_k(ref.P, self.n))
        self.sync_pins()
        return bool(_legal_sweeps_k(
            ref.P, self.n, G,
            float(ref.xmin), float(ref.xmax),
            float(ref.ymin), float(ref.ymax),
            self.group_of, self.kind, self.grp_ptr, self.grp_mem,
            self.ch_ptr, self.ch_a, self.ch_b,
            self.cv_ptr, self.cv_a, self.cv_b,
            self.pinx, self.piny,
            self.c0, self.c1, self.f0, self.f1, self.cen,
            self.pin_this, self.pin_oth,
            self.lo_f, self.hi_f, self.dmax, self.tight_idx,
            self.eg_i, self.eg_j, self.eg_c,
            self.ein_ptr, self.ein_e, self.eout_ptr, self.eout_e, self.cur,
            self.key_of, self.d, self.assigned,
            self.snap_key, self.snap_ord))

    def block_hp(self, i: int, cx: float, cy: float, exclude=()) -> float:
        ex = self.b_ex
        nex = 0
        for e in exclude:
            ex[nex] = e
            nex += 1
        return float(_block_hp_k(
            self.ref.P, int(i), float(cx), float(cy),
            self.ba_ptr, self.ba_j, self.ba_w,
            self.bp_ptr, self.bp_X, self.bp_Y, self.bp_W, ex, nex))

    def optimal_point(self, i: int):
        found, ox, oy = _optimal_point_k(
            self.ref.P, int(i), self.ba_ptr, self.ba_j, self.ba_w,
            self.bp_ptr, self.bp_X, self.bp_Y, self.bp_W,
            self.b_xs, self.b_ys, self.b_ws, self.b_sv, self.b_sc)
        return (float(ox), float(oy)) if found else None

    def discrete_gains(self, idxs):
        """Vectorized candidate scan; returns (keep, gain, opx, opy) views."""
        m = len(idxs)
        _discrete_gains_k(
            self.ref.P, idxs, self.ba_ptr, self.ba_j, self.ba_w,
            self.bp_ptr, self.bp_X, self.bp_Y, self.bp_W,
            self.b_xs, self.b_ys, self.b_ws, self.b_sv, self.b_sc,
            self.d_gain, self.d_opx, self.d_opy, self.d_keep)
        return (self.d_keep[:m], self.d_gain[:m],
                self.d_opx[:m], self.d_opy[:m])

    def swap_delta(self, i: int, j: int, cxi: float, cyi: float,
                   cxj: float, cyj: float) -> float:
        return float(_swap_delta_k(
            self.ref.P, int(i), int(j),
            float(cxi), float(cyi), float(cxj), float(cyj),
            self.ba_ptr, self.ba_j, self.ba_w,
            self.bp_ptr, self.bp_X, self.bp_Y, self.bp_W, self.b_ex))

    def axis_pass_hold(self, axis: int, invert: bool) -> None:
        ref = self.ref
        G = self.G
        if G == 0:
            return
        pinx = self.pinx
        piny = self.piny
        for gi, g in enumerate(ref.groups):
            pinx[gi] = 1 if g.pin_x else 0
            piny[gi] = 1 if g.pin_y else 0
        lim_lo = ref.xmin if axis == 0 else ref.ymin
        lim_hi = ref.xmax if axis == 0 else ref.ymax
        self.calls += 1
        _axis_pass_hold_k(
            ref.P, self.n, G, axis, bool(invert),
            float(lim_lo), float(lim_hi),
            self.group_of, self.kind, self.grp_ptr, self.grp_mem,
            self.ch_ptr, self.ch_a, self.ch_b,
            self.cv_ptr, self.cv_a, self.cv_b,
            pinx, piny,
            self.c0, self.c1, self.f0, self.f1, self.cen,
            self.pin_this, self.pin_oth,
            self.lo_f, self.hi_f, self.dmax, self.tight_idx,
            self.eg_i, self.eg_j, self.eg_c,
            self.ein_ptr, self.ein_e, self.eout_ptr, self.eout_e, self.cur,
            self.key_of, self.d, self.assigned,
            self.snap_key, self.snap_ord)

    def axis_pass_soft(self, axis: int, max_step, invert: bool) -> None:
        """`_axis_pass(hold=False)`.  Only reachable with `self.qsweep`."""
        ref = self.ref
        G = self.G
        if G == 0:
            return
        pinx = self.pinx
        piny = self.piny
        for gi, g in enumerate(ref.groups):
            pinx[gi] = 1 if g.pin_x else 0
            piny[gi] = 1 if g.pin_y else 0
        lim_lo = ref.xmin if axis == 0 else ref.ymin
        lim_hi = ref.xmax if axis == 0 else ref.ymax
        self.qcalls += 1
        _axis_pass_soft_k(
            ref.P, self.n, G, axis, bool(invert),
            float(lim_lo), float(lim_hi),
            self.group_of, self.kind, self.grp_ptr, self.grp_mem,
            self.ch_ptr, self.ch_a, self.ch_b,
            self.cv_ptr, self.cv_a, self.cv_b,
            pinx, piny,
            self.c0, self.c1, self.f0, self.f1, self.cen,
            self.pin_this, self.pin_oth,
            self.lo_f, self.hi_f, self.dmax, self.tight_idx,
            self.eg_i, self.eg_j, self.eg_c,
            self.ein_ptr, self.ein_e, self.eout_ptr, self.eout_e, self.cur,
            self.key_of, self.d, self.assigned,
            self.snap_key, self.snap_ord,
            self.ge_ptr, self.ge_M, self.ge_J, self.ge_W,
            self.gp_ptr, self.gp_M, self.gp_X, self.gp_Y, self.gp_W,
            self.des, self.wts, self.sv, self.sc,
            0.0 if max_step is None else float(max_step),
            max_step is not None)

    def evict_scan(self, i: int, order, max_anchors: int, is_soft: bool,
                   area: float) -> bool:
        ref = self.ref
        return bool(_evict_scan_k(
            ref.P, self.n, int(i), np.ascontiguousarray(order, dtype=np.int64),
            int(max_anchors), bool(is_soft), float(area),
            float(ref.xmin), float(ref.xmax), float(ref.ymin), float(ref.ymax)))

    def has_overlap(self) -> bool:
        return bool(_has_overlap_k(self.ref.P, self.n))

    def overlap_count(self) -> int:
        return int(_overlap_count_k(self.ref.P, self.n))

    def warmup(self) -> None:
        """Force JIT compilation (or a cache load) before the ladder starts.

        Delegates to `warm_process()`: numba specializes on argument *types*,
        and the dummy problem has the same ones, so this compiles exactly the
        signatures the hot path will use without touching the real layout.
        """
        warm_process(self.qsweep, self.disc)


def _contact_csr(groups, attr: str):
    G = len(groups)
    ptr = np.zeros(G + 1, dtype=np.int64)
    for gi, g in enumerate(groups):
        ptr[gi + 1] = ptr[gi] + len(getattr(g, attr))
    tot = max(int(ptr[G]), 1)
    a = np.zeros(tot, dtype=np.int64)
    b = np.zeros(tot, dtype=np.int64)
    for gi, g in enumerate(groups):
        p = ptr[gi]
        for k, (aa, bb) in enumerate(getattr(g, attr)):
            a[p + k] = int(aa)
            b[p + k] = int(bb)
    return ptr, a, b


def _edge_csr(groups, a_attr: str, b_attr: str, w_attr: str):
    """CSR over the per-group external-edge arrays (`eM` / `eJ` / `eW`)."""
    G = len(groups)
    ptr = np.zeros(G + 1, dtype=np.int64)
    for gi, g in enumerate(groups):
        ptr[gi + 1] = ptr[gi] + len(getattr(g, a_attr))
    tot = max(int(ptr[G]), 1)
    a = np.zeros(tot, dtype=np.int64)
    b = np.zeros(tot, dtype=np.int64)
    w = np.zeros(tot)
    for gi, g in enumerate(groups):
        lo, hi = int(ptr[gi]), int(ptr[gi + 1])
        if hi > lo:
            a[lo:hi] = getattr(g, a_attr)
            b[lo:hi] = getattr(g, b_attr)
            w[lo:hi] = getattr(g, w_attr)
    return ptr, a, b, w


def _pin_csr(groups):
    """CSR over the per-group pin arrays (`pM` / `pX` / `pY` / `pW`)."""
    G = len(groups)
    ptr = np.zeros(G + 1, dtype=np.int64)
    for gi, g in enumerate(groups):
        ptr[gi + 1] = ptr[gi] + len(g.pM)
    tot = max(int(ptr[G]), 1)
    m = np.zeros(tot, dtype=np.int64)
    x = np.zeros(tot)
    y = np.zeros(tot)
    w = np.zeros(tot)
    for gi, g in enumerate(groups):
        lo, hi = int(ptr[gi]), int(ptr[gi + 1])
        if hi > lo:
            m[lo:hi] = g.pM
            x[lo:hi] = g.pX
            y[lo:hi] = g.pY
            w[lo:hi] = g.pW
    return ptr, m, x, y, w


def kernel_enabled() -> bool:
    return os.environ.get("PARTNER_REFINE_KERNEL", "") == "numba"


def qsweep_enabled() -> bool:
    """The `hold=False` median-sweep port -- a sub-switch of `kernel_enabled`,
    so it can never fire on its own."""
    return (kernel_enabled()
            and os.environ.get("PARTNER_REFINE_KERNEL_QSWEEP", "") in _TRUE)


def disc_enabled() -> bool:
    """The discrete-move port -- likewise a sub-switch of `kernel_enabled`."""
    return (kernel_enabled()
            and os.environ.get("PARTNER_REFINE_KERNEL_DISC", "") in _TRUE)


def disc_parts() -> tuple:
    """Which halves of the DISC port are live, for A/B attribution.

    `PARTNER_REFINE_DISC_PARTS` (default `fuse,hp`):
      * `fuse` -- `_legal_check`'s legalization loop fused into one njit call
        (this is where the wall clock is: 91% of the `discrete` bucket);
      * `hp`   -- `_block_hp` / `_optimal_point` / the candidate scan
        (the Python leaves: ~7% of the bucket).
    Both are output-identical transformations, so the split exists purely so
    a paired run can price them separately.
    """
    raw = os.environ.get("PARTNER_REFINE_DISC_PARTS")
    if raw is None:
        return ("fuse", "hp")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


_WARMED = False
_WARMED_Q = False
_WARMED_D = False


def warm_process(qsweep: Optional[bool] = None,
                 disc: Optional[bool] = None) -> bool:
    """Compile (or load from the on-disk cache) every njit entry point on a
    2-block dummy problem, without needing a `_Refiner`.

    Call this **in the parent, before the restart pool forks**: the children
    then inherit compiled code and pay nothing.  Without it the first
    `_Refiner` in each worker pays the cache load (~0.4 s measured), which on a
    1.7 s worker span is the whole first rung.  Idempotent and never raises.

    `qsweep=None` reads `PARTNER_REFINE_KERNEL_QSWEEP` (what the pool-side
    caller wants); an explicit bool is for `RefineKernel.warmup`, which already
    knows whether its own instance built the median-sweep CSR.  The median
    sweep's entry point is compiled only when asked, so a run with the
    sub-switch off pays not one millisecond of extra JIT.
    """
    global _WARMED, _WARMED_Q, _WARMED_D
    if not NUMBA_AVAILABLE:
        return False
    if qsweep is None:
        qsweep = qsweep_enabled()
    if disc is None:
        disc = disc_enabled()
    if _WARMED and (_WARMED_Q or not qsweep) and (_WARMED_D or not disc):
        return _WARMED
    try:
        n = 2
        G = 2
        P = np.array([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]])
        i64 = np.int64
        z = np.zeros
        _has_overlap_k(P, n)
        _overlap_count_k(P, n)
        _argsort_stable_k(np.zeros(2))
        _axis_pass_hold_k(
            P, n, G, 0, False, 0.0, 4.0,
            np.arange(2, dtype=i64), z(2, dtype=i64),
            np.array([0, 1, 2], dtype=i64), np.arange(2, dtype=i64),
            z(3, dtype=i64), z(1, dtype=i64), z(1, dtype=i64),
            z(3, dtype=i64), z(1, dtype=i64), z(1, dtype=i64),
            z(2, dtype=i64), z(2, dtype=i64),
            z(2), z(2), z(2), z(2), z(2), z(2, dtype=i64), z(2, dtype=i64),
            z(2), z(2), z(2), z(G * G, dtype=i64),
            z(G * G, dtype=i64), z(G * G, dtype=i64), z(G * G),
            z(G + 1, dtype=i64), z(G * G, dtype=i64),
            z(G + 1, dtype=i64), z(G * G, dtype=i64), z(G, dtype=i64),
            z(2), z(2), z(2, dtype=i64), z(1), z(1, dtype=i64))
        _evict_scan_k(P, n, 0, np.arange(2, dtype=i64), 2, True, 1.0,
                      0.0, 4.0, 0.0, 4.0)
        _WARMED = True
        if qsweep and not _WARMED_Q:
            _wmedian_k(np.array([1.0, 0.0]), np.array([1.0, 1.0]), 2,
                       z(2), z(2))
            _axis_pass_soft_k(
                P, n, G, 0, False, 0.0, 4.0,
                np.arange(2, dtype=i64), z(2, dtype=i64),
                np.array([0, 1, 2], dtype=i64), np.arange(2, dtype=i64),
                z(3, dtype=i64), z(1, dtype=i64), z(1, dtype=i64),
                z(3, dtype=i64), z(1, dtype=i64), z(1, dtype=i64),
                z(2, dtype=i64), z(2, dtype=i64),
                z(2), z(2), z(2), z(2), z(2), z(2, dtype=i64), z(2, dtype=i64),
                z(2), z(2), z(2), z(G * G, dtype=i64),
                z(G * G, dtype=i64), z(G * G, dtype=i64), z(G * G),
                z(G + 1, dtype=i64), z(G * G, dtype=i64),
                z(G + 1, dtype=i64), z(G * G, dtype=i64), z(G, dtype=i64),
                z(2), z(2), z(2, dtype=i64), z(1), z(1, dtype=i64),
                np.array([0, 1, 2], dtype=i64), np.arange(2, dtype=i64),
                np.arange(2, dtype=i64)[::-1].copy(), np.ones(2),
                np.array([0, 1, 2], dtype=i64), np.arange(2, dtype=i64),
                z(2), z(2), np.ones(2),
                z(2), z(2), z(2), z(2), 0.0, False)
            _WARMED_Q = True
        if disc and not _WARMED_D:
            ba_ptr = np.array([0, 1, 2], dtype=i64)
            ba_j = np.array([1, 0], dtype=i64)
            ba_w = np.ones(2)
            bp_ptr = np.array([0, 1, 2], dtype=i64)
            bp_X = np.ones(2)
            bp_Y = np.ones(2)
            bp_W = np.ones(2)
            ex = z(8, dtype=i64)
            _block_hp_k(P, 0, 0.0, 0.0, ba_ptr, ba_j, ba_w,
                        bp_ptr, bp_X, bp_Y, bp_W, ex, 0)
            _optimal_point_k(P, 0, ba_ptr, ba_j, ba_w,
                             bp_ptr, bp_X, bp_Y, bp_W,
                             z(4), z(4), z(4), z(4), z(4))
            _swap_delta_k(P, 0, 1, 0.0, 0.0, 1.0, 1.0,
                          ba_ptr, ba_j, ba_w, bp_ptr, bp_X, bp_Y, bp_W, ex)
            _discrete_gains_k(P, np.arange(2, dtype=i64),
                              ba_ptr, ba_j, ba_w, bp_ptr, bp_X, bp_Y, bp_W,
                              z(4), z(4), z(4), z(4), z(4),
                              z(2), z(2), z(2), z(2, dtype=i64))
            _legal_sweeps_k(
                P, n, G, 0.0, 4.0, 0.0, 4.0,
                np.arange(2, dtype=i64), z(2, dtype=i64),
                np.array([0, 1, 2], dtype=i64), np.arange(2, dtype=i64),
                z(3, dtype=i64), z(1, dtype=i64), z(1, dtype=i64),
                z(3, dtype=i64), z(1, dtype=i64), z(1, dtype=i64),
                z(2, dtype=i64), z(2, dtype=i64),
                z(2), z(2), z(2), z(2), z(2), z(2, dtype=i64), z(2, dtype=i64),
                z(2), z(2), z(2), z(G * G, dtype=i64),
                z(G * G, dtype=i64), z(G * G, dtype=i64), z(G * G),
                z(G + 1, dtype=i64), z(G * G, dtype=i64),
                z(G + 1, dtype=i64), z(G * G, dtype=i64), z(G, dtype=i64),
                z(2), z(2), z(2, dtype=i64), z(1), z(1, dtype=i64))
            _WARMED_D = True
    except Exception:  # pragma: no cover - warmup must never be fatal
        pass
    return _WARMED


def try_attach(ref) -> Optional[RefineKernel]:
    """Build a kernel for `ref`, or return None (caller keeps the Python path)."""
    if not NUMBA_AVAILABLE:
        return None
    try:
        if int(ref.n) <= 0 or not ref.groups:
            return None
        k = RefineKernel(ref, qsweep=qsweep_enabled(), disc=disc_enabled())
        if os.environ.get("PARTNER_REFINE_KERNEL_WARMUP", "1") in _TRUE:
            k.warmup()
        return k
    except Exception:
        return None


# =============================================================================
# Not ported (deliberate)
# =============================================================================
# (`_axis_pass(hold=False)` moved OUT of this list under
# `PARTNER_REFINE_KERNEL_QSWEEP`, and `_optimal_point` / `_block_hp` under
# `PARTNER_REFINE_KERNEL_DISC`; see the module docstring for the tie /
# float-re-association argument that made them portable.  `_optimal_point`
# still runs in Python inside `_evict`'s prologue whenever DISC is off --
# there it is worth < 1%, which is why it waited for a reason.)
#
# `_key` / `opt._hpwl` / `opt._violations`: not this module's -- `_violations`
# already has its own `PARTNER_SA_KERNEL` port, and `_hpwl` is two vectorized
# numpy reductions.  The discrete sub-profile reaches `_key` on a minority of
# `_legal_check` calls (most attempts are rejected on residual overlap first),
# so fusing it into `_legal_sweeps_k` would buy the wrong tail.
#
# `_reshape_chain` / `_relocate_to_free` / `_band_restack` and friends stay in
# Python: they are object-graph search, not numeric inner loops, and they do
# not appear in the rung profile (< 1 % each).
