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
Ported: `_axis_pass` **with `hold=True` only** (the legalization sweep --
`legalize`, `_legal_check`), the `_evict` anchor x variant scan,
`_has_overlap` and `_overlap_count`.

NOT ported: `_axis_pass(hold=False)` (the HPWL median sweep in `run`).  Its
per-group target is `_median_shift`, which ends in `np.argsort` on a
*non-stable* default quicksort over float keys that change as the sweep moves
blocks; reproducing numpy's introsort tie-break inside the loop is a separate,
evidence-gated increment.  Also not ported: `_optimal_point` (same reason --
it feeds `_wmedian`), so `_evict` keeps its Python prologue and the kernel
takes the anchor order as an argument.

Activation
----------
`PARTNER_REFINE_KERNEL=numba`.  Unset (the default) leaves the refiner on the
existing Python path with no behavioural change whatsoever: `_Refiner._nk` is
`None` and every dispatch site is a single `is None` test.
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


# =============================================================================
# the fused axis pass (hold=True)
# =============================================================================
@njit(cache=True)
def _axis_pass_hold_k(P, n, G, axis, invert, lim_lo, lim_hi,
                      group_of, kind, grp_ptr, grp_mem,
                      ch_ptr, ch_a, ch_b, cv_ptr, cv_a, cv_b,
                      pinx, piny,
                      c0, c1, f0, f1, cen, pin_this, pin_oth,
                      lo_f, hi_f, dmax, tight_idx, eg_i, eg_j, eg_c,
                      ein_ptr, ein_e, eout_ptr, eout_e, cur,
                      key_of, d, assigned, snap_key, snap_ord):
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

    # -- (E) forward assignment + rigid moves --------------------------------
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
            for t in range(grp_ptr[gi], grp_ptr[gi + 1]):
                P[grp_mem[t], axis] += dv
            # cluster contact re-snap: `sorted(g.cH, key=lambda ab: P[a, 0])`
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
                 "snap_ord", "calls")

    def __init__(self, ref):
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

    # -- hot path -------------------------------------------------------
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
        warm_process()


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


def kernel_enabled() -> bool:
    return os.environ.get("PARTNER_REFINE_KERNEL", "") == "numba"


_WARMED = False


def warm_process() -> bool:
    """Compile (or load from the on-disk cache) every njit entry point on a
    2-block dummy problem, without needing a `_Refiner`.

    Call this **in the parent, before the restart pool forks**: the children
    then inherit compiled code and pay nothing.  Without it the first
    `_Refiner` in each worker pays the cache load (~0.4 s measured), which on a
    1.7 s worker span is the whole first rung.  Idempotent and never raises.
    """
    global _WARMED
    if _WARMED or not NUMBA_AVAILABLE:
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
        k = RefineKernel(ref)
        if os.environ.get("PARTNER_REFINE_KERNEL_WARMUP", "1") in _TRUE:
            k.warmup()
        return k
    except Exception:
        return None


# =============================================================================
# Not ported (deliberate)
# =============================================================================
# `_axis_pass(hold=False)` -- the HPWL median sweep in `run` -- and
# `_optimal_point`, for the same reason: both end in `_wmedian`, whose
# `np.argsort(vals)` is numpy's *default* (introsort) argsort.  Its tie
# permutation is not reproducible from the algorithm alone, and `_wmedian`
# also `cumsum`s the reordered weights, so a different tie permutation is a
# different float sum.  Porting them would turn this toggle from a pure speed
# change into a search-trajectory change, which needs its own evaluator
# evidence.
#
# `_reshape_chain` / `_relocate_to_free` / `_band_restack` and friends stay in
# Python: they are object-graph search, not numeric inner loops, and they do
# not appear in the rung profile (< 1 % each).
