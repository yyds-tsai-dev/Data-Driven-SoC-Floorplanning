"""Split the kernel path: Python wrapper overhead vs the njit call itself."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "partner"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import column_sa_legalizer as lg  # noqa: E402
import sa_numeric_kernel as snk  # noqa: E402
from synth_instances import build_instance  # noqa: E402


def run(case):
    inst = build_instance(**case)
    os.environ["PARTNER_SA_KERNEL"] = "numba"
    opt = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1e4, seed=case["seed"])
    os.environ.pop("PARTNER_SA_KERNEL", None)
    opt.prepare()
    cols = opt._cols
    c, _ = opt._evaluate(cols)
    snap, _ = opt._anneal(cols, time.time() + 1.0, c)
    cols = opt._restore(snap)

    k = opt._sa_kernel
    k.layout(cols)
    w = k._work
    s = k.stat
    C = len(cols)
    pos = np.zeros((w["n"], 4))

    def raw():
        snk._layout(w["col_u"], w["col_ptr"], C, opt.H, pos,
                    s.blkf, s.blki, s.uf, s.ui, s.bandi, s.chunkf, s.chunki,
                    s.entf, s.enti, s.ublk, s.anc, s.lock,
                    w["hcf"], w["plan_ent"], w["plan_ck"], w["plan_ckw"],
                    w["plani"], w["planf"], w["eckf"], w["tmp"],
                    s.static_ent, w["occ"], w["segs"], w["normal"],
                    w["pending"], w["pl_k"], w["pl_yb"], w["pl_yt"],
                    w["strip"], w["ucol"], w["col_x0"], w["col_w"],
                    w["col_valid"], w["col_pl"], w["outf"])

    reps = 5000
    raw()
    t0 = time.perf_counter()
    for _ in range(reps):
        raw()
    t_raw = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        k.layout(cols)
    t_wrap = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        np.zeros((w["n"], 4))
    t_alloc = (time.perf_counter() - t0) / reps

    col_u = w["col_u"]
    col_ptr = w["col_ptr"]

    def flatten():
        p = 0
        for ci in range(C):
            col_ptr[ci] = p
            for kk in cols[ci]:
                col_u[p] = kk
                p += 1
        col_ptr[C] = p

    t0 = time.perf_counter()
    for _ in range(reps):
        flatten()
    t_flat = (time.perf_counter() - t0) / reps

    x0 = w["col_x0"]
    cw = w["col_w"]
    valid = w["col_valid"]

    def spans():
        return [(x0[ci], x0[ci] + cw[ci]) if valid[ci] else None
                for ci in range(C)]

    t0 = time.perf_counter()
    for _ in range(reps):
        spans()
    t_span = (time.perf_counter() - t0) / reps

    print(f"n={case['n']:3d} units={len(opt.units):3d} cols={C:2d} | "
          f"njit {t_raw*1e6:7.1f}u  wrapper-total {t_wrap*1e6:7.1f}u  "
          f"(alloc {t_alloc*1e6:5.1f} flatten {t_flat*1e6:5.1f} "
          f"spans {t_span*1e6:5.1f} rest {(t_wrap-t_raw-t_alloc-t_flat-t_span)*1e6:5.1f})")


for case in [dict(n=40, seed=3), dict(n=64, seed=11, n_preplaced=6, n_clusters=10, n_mib=5),
             dict(n=100, seed=0), dict(n=120, seed=7)]:
    run(case)
