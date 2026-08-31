#!/usr/bin/env python
"""Analytical (quadratic/least-squares) placement as a candidate seed.

Same consumption chain as scratchpad/rtaware/golden_probe.py (build/ev/
refine_prediction/post-pass/evaluator), but the PREDICTION handed to the
ladder is a closed-form LS placement built ONLY from legitimate inputs
(areas, b2b, p2b+pins, constraints, preplaced target_positions).

Usage (from FloorSet/iccad2026contest):
  uv run python .../ls_seed_probe.py --budget 5 --min-n 102 --mode ls --jsonl o.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "partner"),
           str(_REPO / "FloorSet" / "iccad2026contest"),
           str(_REPO / "FloorSet"), str(_REPO / "scripts" / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from golden_probe import build, ev, _Shim          # noqa: E402
from column_sa_legalizer import _ColumnOptimizer   # noqa: E402
from layout_refiner import refine_prediction       # noqa: E402

BITS = ((1, 0, 0), (2, 0, 1), (4, 1, 1), (8, 1, 0))   # bit, axis, side(0=lo)


def _edges(t, n, is_p2b, npin):
    a = np.asarray(t, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 3))
    a = a[a[:, 0] >= 0]
    if not len(a):
        return np.zeros((0, 3))
    lim0 = npin if is_p2b else n
    m = (a[:, 0] < lim0) & (a[:, 1] < n) & (a[:, 1] >= 0)
    return a[m]


def _spread(ctr, w, h, prep, x0, y0, W, H, kind):
    """Density spreading that PRESERVES the LS relative order (the analytic
    topology signal).  'affine': scale centers about the frame center so the
    center cloud fills the frame.  'quantile': per-axis area-quantile map ->
    marginally uniform density."""
    n = len(ctr)
    free = ~prep
    if kind == "none" or free.sum() < 2:
        return ctr
    out = ctr.copy()
    for ax, (lo, span, ext) in enumerate(((x0, W, w), (y0, H, h))):
        v = ctr[:, ax]
        if kind == "affine":
            vf = v[free]
            a, b = vf.min(), vf.max()
            if b - a < 1e-9:
                continue
            tlo, thi = lo + 0.5 * ext[free].mean(), lo + span - 0.5 * ext[free].mean()
            out[free, ax] = tlo + (vf - a) * (thi - tlo) / (b - a)
        else:                                    # area-quantile
            idx = np.argsort(v, kind="stable")
            tot = float(ext.sum())
            acc = 0.0
            pos = np.empty(n)
            for i in idx:
                pos[i] = lo + span * (acc + 0.5 * ext[i]) / max(tot, 1e-9)
                acc += ext[i]
            out[free, ax] = pos[free]
    return out


def analytic_seed(n, at, cons, tpos, b2b, p2b, pins, mode="ls",
                  util=0.95, tagw=2.0, spread="none"):
    """Closed-form LS placement -> (n,4) rects.  mode: 'ls' | 'lstag'."""
    area = np.asarray(at, dtype=np.float64)[:n]
    C = np.asarray(cons, dtype=np.float64)
    fixed = C[:, 0] != 0
    prep = C[:, 1] != 0
    mib = C[:, 2].astype(int) if C.shape[1] > 2 else np.zeros(n, int)
    clu = C[:, 3].astype(int) if C.shape[1] > 3 else np.zeros(n, int)
    bnd = C[:, 4].astype(int) if C.shape[1] > 4 else np.zeros(n, int)
    T = np.asarray(tpos, dtype=np.float64)
    P_pin = np.asarray(pins, dtype=np.float64)
    P_pin = P_pin[np.isfinite(P_pin).all(axis=1)] if P_pin.ndim == 2 else np.zeros((0, 2))
    npin = len(P_pin)

    # ---- shapes (golden-free): fixed/preplaced keep (w,h); soft -> square,
    #      MIB groups share the square of their mean area.
    w = np.empty(n); h = np.empty(n)
    for i in range(n):
        if (fixed[i] or prep[i]) and T[i, 2] > 0 and T[i, 3] > 0:
            w[i], h[i] = T[i, 2], T[i, 3]
        else:
            s = float(np.sqrt(max(area[i], 1e-9)))
            w[i] = h[i] = s
    for g in set(int(v) for v in mib if v > 0):
        idx = [i for i in range(n) if mib[i] == g and not (fixed[i] or prep[i])]
        if len(idx) > 1:
            s = float(np.sqrt(max(area[idx].mean(), 1e-9)))
            w[idx] = s; h[idx] = s

    # ---- frame: area/util, aspect + origin from the pin extent
    A = float(area.sum()) / util
    if npin:
        x0, y0 = P_pin[:, 0].min(), P_pin[:, 1].min()
        ex, ey = P_pin[:, 0].max() - x0, P_pin[:, 1].max() - y0
    else:
        x0 = y0 = 0.0; ex = ey = 1.0
    ar = (ex / ey) if (ex > 0 and ey > 0) else 1.0
    W = float(np.sqrt(A * ar)); H = A / W if W > 0 else float(np.sqrt(A))
    for i in range(n):                    # frame must contain preplaced boxes
        if prep[i] and T[i, 0] >= 0:
            W = max(W, T[i, 0] + T[i, 2] - x0); H = max(H, T[i, 1] + T[i, 3] - y0)
    fx1, fy1 = x0 + W, y0 + H
    cx0, cy0 = x0 + W / 2.0, y0 + H / 2.0

    E = _edges(b2b, n, False, npin)
    Q = _edges(p2b, n, True, npin)
    wmed = float(np.median(E[:, 2])) if len(E) else 1.0
    inc = np.zeros(n)
    for i, j, ww in E:
        inc[int(i)] += ww; inc[int(j)] += ww
    for pi, bi, ww in Q:
        inc[int(bi)] += ww
    inc_ref = float(np.mean(inc)) if n else 1.0
    if inc_ref <= 0:
        inc_ref = 1.0

    L = np.zeros((n, n)); rhs = np.zeros((n, 2))
    for i, j, ww in E:
        i, j = int(i), int(j)
        L[i, i] += ww; L[j, j] += ww; L[i, j] -= ww; L[j, i] -= ww
    for pi, bi, ww in Q:
        bi = int(bi)
        L[bi, bi] += ww; rhs[bi] += ww * P_pin[int(pi)]
    if mode == "lstag":
        for g in set(int(v) for v in clu if v > 0):     # cluster star springs
            idx = [i for i in range(n) if clu[i] == g]
            if len(idx) > 1:
                a0 = idx[0]
                for b in idx[1:]:
                    L[a0, a0] += wmed; L[b, b] += wmed
                    L[a0, b] -= wmed; L[b, a0] -= wmed
        for i in range(n):                               # boundary-tag pull
            if bnd[i] <= 0:
                continue
            bw = tagw * max(inc[i], inc_ref)
            for bit, ax, side in BITS:
                if not (int(bnd[i]) & bit):
                    continue
                lo, hi, ext = (x0, fx1, w[i]) if ax == 0 else (y0, fy1, h[i])
                tgt = lo + ext / 2.0 if side == 0 else hi - ext / 2.0
                L[i, i] += bw; rhs[i, ax] += bw * tgt
    eps = 1e-3 * max(inc_ref, 1e-9)
    for i in range(n):
        L[i, i] += eps; rhs[i] += eps * np.array([cx0, cy0])
    for i in range(n):                                   # preplaced Dirichlet
        if prep[i] and T[i, 0] >= 0:
            L[i, :] = 0.0; L[i, i] = 1.0
            rhs[i] = np.array([T[i, 0] + T[i, 2] / 2.0, T[i, 1] + T[i, 3] / 2.0])
    try:
        ctr = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        ctr = np.linalg.lstsq(L, rhs, rcond=None)[0]
    if not np.isfinite(ctr).all():
        ctr = np.tile([cx0, cy0], (n, 1)).astype(np.float64)
    ctr = _spread(ctr, w, h, prep, x0, y0, W, H, spread)

    out = np.empty((n, 4))
    for i in range(n):
        if prep[i] and T[i, 0] >= 0:
            out[i] = [T[i, 0], T[i, 1], T[i, 2], T[i, 3]]; continue
        x = min(max(ctr[i, 0] - w[i] / 2.0, x0), max(x0, fx1 - w[i]))
        y = min(max(ctr[i, 1] - h[i] / 2.0, y0), max(y0, fy1 - h[i]))
        out[i] = [x, y, w[i], h[i]]
    return out


def overlap_frac(P):
    n = len(P); tot = float((P[:, 2] * P[:, 3]).sum()); ov = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) - max(P[i, 0], P[j, 0])
            dy = min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) - max(P[i, 1], P[j, 1])
            if dx > 0 and dy > 0:
                ov += dx * dy
    return ov / tot if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=102)
    ap.add_argument("--ids", default="")
    ap.add_argument("--mode", default="ls")
    ap.add_argument("--spread", default="none")
    ap.add_argument("--jsonl", default="")
    a = ap.parse_args()
    from litetestLoader import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(a.data_path)
    ids = [int(t) for t in a.ids.split(",") if t.strip()] or list(range(len(ds)))
    from contest_optimizer import MyOptimizer
    shim = _Shim()
    out_f = open(a.jsonl, "w") if a.jsonl else None
    for tid in ids:
        n, at, cons, tpos, b2b, p2b, pins, golden, base = build(ds, tid)
        if n < a.min_n:
            continue
        t_ls = time.time()
        P = analytic_seed(n, at, cons, tpos, b2b, p2b, pins, mode=a.mode,
                          spread=a.spread)
        ls_s = time.time() - t_ls
        G = np.asarray([list(r) for r in golden], dtype=np.float64)
        sm = ev(P, base, cons, b2b, p2b, pins, at, tpos)
        row = {"tid": tid, "n": n, "mode": a.mode, "spread": a.spread,
               "budget": a.budget,
               "ls_ms": round(1000 * ls_s, 2),
               "s_hpwl": round(sm.hpwl_gap, 4), "s_area": round(sm.area_gap, 4),
               "s_ovf": round(overlap_frac(P), 4), "s_feas": bool(sm.is_feasible),
               "g_ovf": round(overlap_frac(G), 4)}
        t0 = time.time()
        opt = _ColumnOptimizer([tuple(map(float, r)) for r in P], at, cons,
                               tpos, b2b, p2b, pins, time.time() + a.budget, seed=0)
        try:
            ref = refine_prediction(opt, P.copy(), time.time() + a.budget, seed=41)
        except Exception as e:
            ref = None; row["ref_err"] = repr(e)[:100]
        row["ref_s"] = round(time.time() - t0, 2)
        if ref is None:
            row["refined"] = None
        else:
            rm = ev(ref, base, cons, b2b, p2b, pins, at, tpos)
            row.update(r_cost=round(rm.cost, 4), r_feas=bool(rm.is_feasible),
                       r_hpwl=round(rm.hpwl_gap, 4), r_area=round(rm.area_gap, 4),
                       r_V=round(rm.violations_relative, 4),
                       r_bnd=rm.boundary_violations, r_grp=rm.grouping_violations,
                       r_mib=rm.mib_violations)
            box = [(np.asarray([list(map(float, r)) for r in ref]), opt)]
            cur = [tuple(map(float, r)) for r in ref]
            try:
                for fn, kw in ((MyOptimizer._coord_polish, {"elapsed": 0.0}),
                               (MyOptimizer._final_seat, None),
                               (MyOptimizer._tag_compress, None),
                               (MyOptimizer._wall_repair_final, None)):
                    if kw is None:
                        cur = fn(shim, cur, at, cons, tpos, b2b, p2b, pins, box)
                    else:
                        cur = fn(shim, cur, at, cons, tpos, b2b, p2b, pins, **kw)
                pm = ev(cur, base, cons, b2b, p2b, pins, at, tpos)
                row.update(p_cost=round(pm.cost, 4), p_feas=bool(pm.is_feasible),
                           p_hpwl=round(pm.hpwl_gap, 4), p_area=round(pm.area_gap, 4),
                           p_V=round(pm.violations_relative, 4),
                           p_bnd=pm.boundary_violations, p_grp=pm.grouping_violations,
                           p_mib=pm.mib_violations)
            except Exception as e:
                row["post_err"] = repr(e)[:120]
        print(json.dumps(row), flush=True)
        if out_f:
            out_f.write(json.dumps(row) + "\n"); out_f.flush()
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
