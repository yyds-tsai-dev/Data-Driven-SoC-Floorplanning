#!/usr/bin/env python3
"""Section-level split of `_axis_constraints` / `_axis_pass` / `_evict`.

Replaces the two methods with instrumented transcriptions (same arithmetic,
same order) that accumulate per-section wall time, then replays a ladder rung.

Usage:  uv run python scratchpad/refine_axis_split.py [test_id ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_profile import load_case, preds_for, make_rung  # noqa: E402

ACC: dict = {}


def _t(name, dt, k=1):
    e = ACC.setdefault(name, [0, 0.0])
    e[0] += k
    e[1] += dt


def install():
    import layout_refiner as rf
    from layout_refiner import _AxisCons, SEP_TOL
    from typing import List

    def _axis_constraints(self, axis: int, invert: bool = False):
        pc = time.perf_counter
        t0 = pc()
        P = self.P
        n = self.n
        o = 1 - axis
        c0 = P[:, axis]
        c1 = c0 + P[:, axis + 2]
        f0 = P[:, o]
        f1 = f0 + P[:, o + 2]
        ovf = (np.minimum(f1[:, None], f1[None, :])
               - np.maximum(f0[:, None], f0[None, :]))
        ovm = (np.minimum(c1[:, None], c1[None, :])
               - np.maximum(c0[:, None], c0[None, :]))
        both = (ovf > SEP_TOL) & (ovm > SEP_TOL)
        pin_this = self.kind == 2
        pin_oth = self.kind == 2
        pt = np.zeros(n, dtype=bool)
        po = np.zeros(n, dtype=bool)
        for g in self.groups:
            if (g.pin_x if axis == 0 else g.pin_y):
                pt[g.members] = True
            if (g.pin_y if axis == 0 else g.pin_x):
                po[g.members] = True
        pin_this = pin_this | pt
        pin_oth = pin_oth | po
        can_this = ~(pin_this[:, None] & pin_this[None, :])
        can_oth = ~(pin_oth[:, None] & pin_oth[None, :])
        pref = (ovm >= ovf) if invert else (ovm <= ovf)
        ov = ((ovf > SEP_TOL) & (ovm <= SEP_TOL)) \
            | (both & can_this & (~can_oth | pref))
        np.fill_diagonal(ov, False)
        cen = c0 + 0.5 * P[:, axis + 2]
        ridx = np.arange(n)
        left = (cen[:, None] < cen[None, :]) \
            | ((cen[:, None] == cen[None, :]) & (ridx[:, None] < ridx[None, :]))
        pairs = np.nonzero(ov & left)
        t1 = pc()
        _t("A_matrix", t1 - t0)
        _t("A_npairs", 0.0, len(pairs[0]))

        G = len(self.groups)
        gof = self.group_of
        NEG = -1e18
        POS = 1e18
        lo_fixed = np.full(G, NEG)
        hi_fixed = np.full(G, POS)
        edges_in: List[List] = [[] for _ in range(G)]
        edges_out: List[List] = [[] for _ in range(G)]
        tight: dict = {}
        for i, j in zip(*pairs):
            gi_, gj_ = gof[i], gof[j]
            if gi_ == gj_ and gi_ >= 0:
                continue
            c = float(c1[i] - c0[j])
            if gi_ >= 0 and gj_ >= 0:
                key = (gi_, gj_)
                if key not in tight or c > tight[key]:
                    tight[key] = c
            elif gj_ >= 0:
                if c > lo_fixed[gj_]:
                    lo_fixed[gj_] = c
            elif gi_ >= 0:
                if -c < hi_fixed[gi_]:
                    hi_fixed[gi_] = -c
        for (gi_, gj_), c in tight.items():
            edges_in[gj_].append((gi_, c))
            edges_out[gi_].append((gj_, c))
        t2 = pc()
        _t("B_pairloop", t2 - t1)
        _t("B_nedges", 0.0, len(tight))

        lim_lo = self.xmin if axis == 0 else self.ymin
        lim_hi = self.xmax if axis == 0 else self.ymax
        key_of = np.empty(G)
        for gi_, g in enumerate(self.groups):
            mem = g.members
            lo_fixed[gi_] = max(lo_fixed[gi_], lim_lo - float(c0[mem].min()))
            hi_fixed[gi_] = min(hi_fixed[gi_], lim_hi - float(c1[mem].max()))
            key_of[gi_] = float(c0[mem].min())
        pinned = np.array([(g.pin_x if axis == 0 else g.pin_y)
                           for g in self.groups])
        lo_fixed[pinned] = 0.0
        hi_fixed[pinned] = 0.0
        t3 = pc()
        _t("C_grouploop", t3 - t2)
        _t("C_ngroups", 0.0, G)

        order = np.argsort(key_of, kind="stable")
        dmax = hi_fixed.copy()
        for gi_ in order[::-1]:
            for gj_, c in edges_out[gi_]:
                v = dmax[gj_] - c
                if v < dmax[gi_]:
                    dmax[gi_] = v
        t4 = pc()
        _t("D_backward", t4 - t3)
        return _AxisCons(len(self.groups), lo_fixed, dmax, edges_in,
                         edges_out, order, pinned)

    def _axis_pass(self, axis: int, max_step=None, hold: bool = False,
                   invert: bool = False):
        pc = time.perf_counter
        cons = self._axis_constraints(axis, invert)
        t0 = pc()
        G = cons.G
        lo_fixed = cons.lo
        dmax = cons.dmax
        edges_in = cons.edges_in
        edges_out = cons.edges_out
        order = cons.order
        pinned = cons.pinned
        d = np.zeros(G)
        assigned = np.zeros(G, dtype=bool)
        for gi_ in order:
            g = self.groups[gi_]
            if pinned[gi_]:
                assigned[gi_] = True
                continue
            lo = lo_fixed[gi_]
            for gp_, c in edges_in[gi_]:
                base = d[gp_] if assigned[gp_] else 0.0
                if base + c > lo:
                    lo = base + c
            hi = dmax[gi_]
            for gj_, c in edges_out[gi_]:
                if assigned[gj_] and d[gj_] - c < hi:
                    hi = d[gj_] - c
            if lo > hi:
                assigned[gi_] = True
                continue
            if hold:
                t = 0.0
            else:
                t = self._median_shift(g, axis)
                if t is None:
                    t = 0.0
                if max_step is not None:
                    t = min(max(t, -max_step), max_step)
            d[gi_] = min(max(t, lo), hi)
            assigned[gi_] = True
            if abs(d[gi_]) > 1e-12:
                self._move(g, d[gi_], axis)
        _t("E_forward", pc() - t0)

    rf._Refiner._axis_constraints = _axis_constraints
    rf._Refiner._axis_pass = _axis_pass

    real_ev = rf._Refiner._evict

    def _ev(self, *a, **kw):
        t = time.perf_counter()
        try:
            return real_ev(self, *a, **kw)
        finally:
            _t("F_evict", time.perf_counter() - t)
    rf._Refiner._evict = _ev

    real_ho = rf._Refiner._has_overlap

    def _ho(self):
        t = time.perf_counter()
        try:
            return real_ho(self)
        finally:
            _t("G_hasoverlap", time.perf_counter() - t)
    rf._Refiner._has_overlap = _ho


def main():
    install()
    ids = [int(a) for a in sys.argv[1:]] or [95, 79, 64]
    for tid in ids:
        case = load_case(tid)
        preds = preds_for(tid)
        for label, expand in (("FIXED", 0.0), ("0.28", 0.28)):
            for pi in range(3):
                ACC.clear()
                opt, r = make_rung(case, preds[pi])
                if expand:
                    r.xmax += (r.xmax - r.xmin) * expand
                    r.ymax += (r.ymax - r.ymin) * expand
                t0 = time.time()
                ok = r.legalize_soft(14, deadline=t0 + 30.0, fine=False)
                el = time.time() - t0
                secs = {k: v for k, v in ACC.items()
                        if not k.endswith(("pairs", "edges", "groups"))}
                tot = sum(v[1] for v in secs.values())
                print(f"t{tid} n={case[0]} {label:5s} p{pi} ok={int(ok)} "
                      f"{el:6.3f}s  accounted {tot:6.3f}s "
                      f"({100 * tot / el:4.1f}%)")
                for k in sorted(secs, key=lambda k: -secs[k][1]):
                    c, s = secs[k]
                    print(f"     {k:14s} {c:6d} {s:7.3f}s {100 * s / el:5.1f}% "
                          f"{1e6 * s / max(c, 1):8.1f}us/call")
                for k in ("A_npairs", "B_nedges", "C_ngroups"):
                    if k in ACC:
                        base = max(ACC["A_matrix"][0], 1)
                        print(f"     {k:14s} avg {ACC[k][0] / base:8.1f}")


if __name__ == "__main__":
    main()
