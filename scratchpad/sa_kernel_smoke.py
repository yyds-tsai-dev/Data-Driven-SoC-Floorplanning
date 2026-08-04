"""End-to-end smoke: a full sequential `run()` under both paths.

Exercises the post-anneal passes (`_greedy_polish`, `_repair_boundary`,
`_spread_tagged`, `_order_pass`, `_reduce_overflow`) that read `_unit_col`,
which the kernel replaces with an ndarray buffer. Single process, no pool,
no evaluator.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "partner"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import column_sa_legalizer as lg  # noqa: E402
from synth_instances import build_instance  # noqa: E402

CASES = [
    dict(n=40, seed=3),
    dict(n=100, seed=0),
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]


def overlaps(rects):
    bad = 0
    for i in range(len(rects)):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, wj, hj = rects[j]
            if (xi < xj + wj - 1e-6 and xj < xi + wi - 1e-6
                    and yi < yj + hj - 1e-6 and yj < yi + hi - 1e-6):
                bad += 1
    return bad


for case in CASES:
    for kernel in (False, True):
        inst = build_instance(**case)
        if kernel:
            os.environ["PARTNER_SA_KERNEL"] = "numba"
        else:
            os.environ.pop("PARTNER_SA_KERNEL", None)
        opt = lg._ColumnOptimizer(
            inst.rects, inst.area_targets, inst.constraints,
            inst.target_positions, inst.b2b, inst.p2b, inst.pins,
            time.time() + 4.0, seed=case["seed"])
        os.environ.pop("PARTNER_SA_KERNEL", None)
        t0 = time.perf_counter()
        rects = opt.run()
        dt = time.perf_counter() - t0
        hp, area, V = opt.final_metrics
        tag = "kernel" if kernel else "python"
        fb = opt._sa_kernel.fallbacks if opt._sa_kernel else 0
        print(f"n={case['n']:3d} {tag:6s} {dt:5.2f}s blocks={len(rects)} "
              f"overlaps={overlaps(rects)} hpwl={hp:12.1f} area={area:12.1f} "
              f"V={V} fallbacks={fb}")
