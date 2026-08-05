#!/usr/bin/env python3
"""Micro-benchmark for PARTNER_REFINE_KERNEL: one rung (`legalize_soft`) and
one `_tighten`, old vs new, on the three real validation bands (n = 85 / 100 /
116) plus the JIT warmup cost.

Single thread, cached Direct predictions, no evaluator, no pool.

Usage:  uv run python scratchpad/refine_kernel_bench.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_profile import load_case, preds_for, build_opt  # noqa: E402

BANDS = [(64, 85), (79, 100), (95, 116)]


def fresh(case, pred, expand, kernel):
    import layout_refiner as rf
    if kernel:
        os.environ["PARTNER_REFINE_KERNEL"] = "numba"
    else:
        os.environ.pop("PARTNER_REFINE_KERNEL", None)
    opt = build_opt(case, pred, 600.0)
    r = rf._Refiner(opt, np.asarray(pred, dtype=np.float64), seed=327)
    if expand:
        r.xmax += (r.xmax - r.xmin) * expand
        r.ymax += (r.ymax - r.ymin) * expand
    return r


def bench():
    print(f"{'band':>12s} {'stage':>12s} {'py (s)':>9s} {'nb (s)':>9s} "
          f"{'speedup':>8s}")
    for tid, n in BANDS:
        case = load_case(tid)
        preds = preds_for(tid)
        for label, expand in (("rung FIXED", 0.0), ("rung 0.28", 0.28)):
            tp = tk = 0.0
            for pi in range(3):
                r = fresh(case, preds[pi], expand, False)
                t0 = time.time()
                r.legalize_soft(14, deadline=t0 + 600.0, fine=False)
                tp += time.time() - t0
                r = fresh(case, preds[pi], expand, True)
                t0 = time.time()
                r.legalize_soft(14, deadline=t0 + 600.0, fine=False)
                tk += time.time() - t0
            print(f"n={n:<10d} {label:>12s} {tp / 3:9.4f} {tk / 3:9.4f} "
                  f"{tp / max(tk, 1e-9):7.2f}x")
        # _tighten: legalize first (both paths, from the same start), then
        # time the frame anneal alone
        tp = tk = 0.0
        for pi in range(3):
            for kern, acc in ((False, 0), (True, 1)):
                r = fresh(case, preds[pi], 0.28, kern)
                r.legalize_soft(14, deadline=time.time() + 600.0, fine=False)
                t0 = time.time()
                r._tighten(t0 + 600.0)
                dt = time.time() - t0
                if acc:
                    tk += dt
                else:
                    tp += dt
        print(f"n={n:<10d} {'_tighten':>12s} {tp / 3:9.4f} {tk / 3:9.4f} "
              f"{tp / max(tk, 1e-9):7.2f}x")


_WARM_SRC = """
import os, sys, time
sys.path.insert(0, 'partner'); sys.path.insert(0, 'scratchpad')
os.environ['PARTNER_REFINE_KERNEL'] = 'numba'
import refine_numeric_kernel as rnk
t0 = time.time(); rnk.warm_process(); print('warm_process     %.3fs' % (time.time() - t0))
from refine_profile import load_case, preds_for, build_opt
import numpy as np, layout_refiner as rf
case = load_case(95); preds = preds_for(95)
opt = build_opt(case, preds[0], 600.0)
t0 = time.time(); r = rf._Refiner(opt, np.asarray(preds[0], np.float64), seed=327)
print('1st _Refiner     %.4fs  nk=%s' % (time.time() - t0, r._nk is not None))
opt2 = build_opt(case, preds[1], 600.0)
t0 = time.time(); r2 = rf._Refiner(opt2, np.asarray(preds[1], np.float64), seed=327)
print('2nd _Refiner     %.4fs' % (time.time() - t0))
"""


def warmup_cost():
    """JIT cost in a fresh process, cold (empty cache dir) then warm."""
    for tag, cold in (("COLD (empty numba cache)", True),
                      ("WARM (on-disk cache)", False)):
        e = dict(os.environ)
        if cold:
            e["NUMBA_CACHE_DIR"] = "/tmp/rk_cache_%d" % time.time_ns()
        else:
            e.pop("NUMBA_CACHE_DIR", None)
        out = subprocess.run([sys.executable, "-c", _WARM_SRC],
                             capture_output=True, text=True, env=e,
                             cwd=str(Path.cwd()))
        print(f"--- {tag} ---")
        print(out.stdout.strip() or out.stderr.strip()[-400:])


if __name__ == "__main__":
    bench()
    print()
    warmup_cost()
