"""Unprofiled Amdahl split of the partner SA per-move cost.

Measures, on a converged-ish `cols` state, the wall time of
  (a) _layout(cols)          -- geometry / stacking
  (b) _cost(pos, xr, yt)     -- hpwl + violations + area
  (c) _random_move + undo    -- proposal bookkeeping
so the per-move budget can be attributed without cProfile distortion.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "partner"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sa_kernel_synth import build_instance, make_optimizer  # noqa: E402


def bench(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main(n: int, seed: int, warm_s: float = 1.5):
    inst = build_instance(n=n, seed=seed)
    opt = make_optimizer(inst, seed=seed, deadline=time.time() + 1000.0)
    opt.prepare()
    cols = opt._cols
    cost, _ = opt._evaluate(cols)

    # warm the SA a little so `cols` looks like a mid-anneal state
    snap, _ = opt._anneal(cols, time.time() + warm_s, cost)
    cols = opt._restore(snap)

    # raw throughput of the true inner loop
    reps = 3000
    t0 = time.perf_counter()
    done = 0
    rng_moves = 0
    while done < reps:
        undo = opt._random_move(cols)
        rng_moves += 1
        if undo is None:
            continue
        opt._evaluate(cols)
        undo()
        done += 1
    loop_dt = (time.perf_counter() - t0) / reps

    pos, xr, yt = opt._layout(cols)
    t_layout = bench(lambda: opt._layout(cols), 2000)
    t_cost = bench(lambda: opt._cost(pos, xr, yt), 2000)
    t_hpwl = bench(lambda: opt._hpwl(pos), 2000)
    t_viol = bench(lambda: opt._violations(pos), 2000)

    def move_undo():
        u = opt._random_move(cols)
        if u is not None:
            u()
    t_move = bench(move_undo, 5000)

    ncols = len([c for c in cols if c])
    print(f"n={n} seed={seed} colcache={'on' if opt._dc_enabled else 'off'} "
          f"units={len(opt.units)} cols={ncols} "
          f"edges={len(opt.eI)} pins={len(opt.pB)} locked={len(opt.locked_rects)}")
    print(f"  full inner-move       {loop_dt*1e6:9.1f} us   ({1.0/loop_dt:8.0f} moves/s)")
    print(f"  _layout               {t_layout*1e6:9.1f} us   {100*t_layout/loop_dt:5.1f}%")
    print(f"  _cost   (total)       {t_cost*1e6:9.1f} us   {100*t_cost/loop_dt:5.1f}%")
    print(f"     _hpwl              {t_hpwl*1e6:9.1f} us   {100*t_hpwl/loop_dt:5.1f}%")
    print(f"     _violations        {t_viol*1e6:9.1f} us   {100*t_viol/loop_dt:5.1f}%")
    print(f"  _random_move + undo   {t_move*1e6:9.1f} us   {100*t_move/loop_dt:5.1f}%")
    resid = loop_dt - t_layout - t_cost - t_move
    print(f"  residual              {resid*1e6:9.1f} us   {100*resid/loop_dt:5.1f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100,
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
