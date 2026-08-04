"""Micro-benchmark: Python `_layout` vs the numba SoA kernel.

Single-threaded, synthetic instances, no dataset and no evaluator.
Reports isolated `_layout` time and the full SA inner-move throughput.

  uv run python scratchpad/sa_kernel_bench.py            # throughput
  uv run python scratchpad/sa_kernel_bench.py warmup     # JIT / cache-load cost
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
    dict(n=64, seed=11, n_preplaced=6, n_clusters=10, n_mib=5),
    dict(n=100, seed=0),
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
    dict(n=120, seed=7),
]


def make(case, kernel: bool):
    inst = build_instance(**case)
    if kernel:
        os.environ["PARTNER_SA_KERNEL"] = "numba"
    else:
        os.environ.pop("PARTNER_SA_KERNEL", None)
    opt = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1e4, seed=case["seed"])
    os.environ.pop("PARTNER_SA_KERNEL", None)
    return opt


def warm_state(opt, seconds=1.0):
    opt.prepare()
    cols = opt._cols
    cost, _ = opt._evaluate(cols)
    snap, _ = opt._anneal(cols, time.time() + seconds, cost)
    return opt._restore(snap)


def time_layout(opt, cols, reps):
    opt._layout(cols)
    t0 = time.perf_counter()
    for _ in range(reps):
        opt._layout(cols)
    return (time.perf_counter() - t0) / reps


def time_moves(opt, cols, reps):
    done = 0
    t0 = time.perf_counter()
    while done < reps:
        undo = opt._random_move(cols)
        if undo is None:
            continue
        opt._evaluate(cols)
        undo()
        done += 1
    return (time.perf_counter() - t0) / reps


def bench():
    print(f"{'case':>22} | {'layout py':>10} {'layout k':>10} {'x':>6} "
          f"| {'move/s py':>10} {'move/s k':>10} {'x':>6}")
    print("-" * 90)
    tot_py = tot_k = 0.0
    for case in CASES:
        label = f"n{case['n']}s{case['seed']}"
        rows = {}
        for kernel in (False, True):
            opt = make(case, kernel)
            assert (opt._sa_kernel is not None) == kernel
            cols = warm_state(opt)
            rows[kernel] = (time_layout(opt, cols, 3000),
                            time_moves(opt, cols, 3000))
        lp, mp = rows[False]
        lk, mk = rows[True]
        tot_py += mp
        tot_k += mk
        print(f"{label:>22} | {lp*1e6:9.1f}u {lk*1e6:9.1f}u {lp/lk:5.2f}x "
              f"| {1/mp:10.0f} {1/mk:10.0f} {mp/mk:5.2f}x")
    print("-" * 90)
    print(f"{'geometric-ish mean':>22} | {'':>10} {'':>10} {'':>6} "
          f"| {'':>10} {'':>10} {tot_py/tot_k:5.2f}x")


def warmup():
    """Cost of getting the kernel ready inside one process."""
    t_imp0 = time.perf_counter()
    import sa_numeric_kernel  # noqa: F401
    t_imp = time.perf_counter() - t_imp0

    inst = build_instance(n=100, seed=0)
    os.environ["PARTNER_SA_KERNEL"] = "numba"
    os.environ["PARTNER_SA_KERNEL_WARMUP"] = "0"
    t0 = time.perf_counter()
    opt = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1e4, seed=0)
    t_attach = time.perf_counter() - t0

    cols = opt._init_columns(10)
    t0 = time.perf_counter()
    opt._layout(cols)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(200):
        opt._layout(cols)
    t_steady = (time.perf_counter() - t0) / 200

    print(f"import sa_numeric_kernel   {t_imp*1e3:8.1f} ms")
    print(f"attach (build flat arrays) {t_attach*1e3:8.1f} ms")
    print(f"first _layout call         {t_first*1e3:8.1f} ms  <- JIT or cache load")
    print(f"steady-state _layout       {t_steady*1e6:8.1f} us")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "warmup":
        warmup()
    else:
        bench()
