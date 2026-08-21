#!/usr/bin/env python3
"""Micro-benchmark for PARTNER_REFINE_KERNEL_DISC.

Converged spans (deadline far away), so both arms do the SAME work and the
ratio is a true speedup rather than "who spent the budget better".  Single
thread, synthetic instances, no evaluator, no pool.

Usage:  uv run python scratchpad/disc_bench.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "tests", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BANDS = [dict(n=100, seed=0),
         dict(n=116, seed=53, n_preplaced=5, n_clusters=8, anchor_clusters=4),
         dict(n=118, seed=7),
         dict(n=120, seed=59, n_preplaced=8, n_clusters=12,
              anchor_clusters=8, frac_boundary=0.30)]
ARMS = [("off", None), ("fuse", "fuse"), ("hp", "hp"), ("both", "fuse,hp")]


def _env(parts):
    os.environ["PARTNER_REFINE_KERNEL"] = "numba"
    os.environ["PARTNER_REFINE_KERNEL_QSWEEP"] = "1"
    os.environ["PARTNER_REFINE_FASTBUILD"] = "1"
    if parts is None:
        os.environ.pop("PARTNER_REFINE_KERNEL_DISC", None)
        os.environ.pop("PARTNER_REFINE_DISC_PARTS", None)
    else:
        os.environ["PARTNER_REFINE_KERNEL_DISC"] = "1"
        os.environ["PARTNER_REFINE_DISC_PARTS"] = parts


def one(case, parts, legal, inst):
    import layout_refiner as rf
    from synth_instances import make_optimizer
    _env(parts)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + 600.0)
    r = rf._Refiner(opt, legal, seed=11)
    t0 = time.time()
    out = r.run(time.time() + 600.0)
    return time.time() - t0, out


def main():
    import layout_refiner as rf
    from synth_instances import build_instance, make_optimizer
    reps = int(os.environ.get("DISC_BENCH_REPS", "3"))
    print(f"{'case':>10s} {'arm':>6s} {'run() s':>9s} {'vs off':>8s} "
          f"{'digest':>10s}")
    for case in BANDS:
        _env(None)
        inst = build_instance(**case)
        pred = np.asarray([list(x) for x in inst.rects], dtype=np.float64)
        opt0 = make_optimizer(inst, seed=3, deadline=time.time() + 600.0)
        r0 = rf._Refiner(opt0, pred, seed=11)
        r0.legalize(60, deadline=time.time() + 60.0)
        legal = r0.P.copy()
        base = None
        tag = "n%ds%d" % (case["n"], case["seed"])
        for name, parts in ARMS:
            best = 1e18
            dig = ""
            for _ in range(reps):
                el, out = one(case, parts, legal, inst)
                best = min(best, el)
                dig = hex(hash(np.asarray(out).tobytes()) & 0xFFFFFFFF)
            if base is None:
                base = best
            print(f"{tag:>10s} {name:>6s} {best:9.4f} "
                  f"{base / max(best, 1e-9):7.2f}x {dig:>10s}")


if __name__ == "__main__":
    main()
