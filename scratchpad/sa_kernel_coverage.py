"""Did the parity walk actually exercise the hard layout branches?

Counts, on the Python reference path, how often each structurally-risky branch
of `_layout_full` / `_stack_column` / `_band_solutions` fires across the same
instances + move walk the parity test uses.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "partner"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import column_sa_legalizer as lg  # noqa: E402
from synth_instances import build_instance  # noqa: E402

CASES = [
    dict(n=100, seed=0),
    dict(n=40, seed=3),
    dict(n=120, seed=7),
    dict(n=64, seed=11, n_preplaced=6, n_clusters=10, n_mib=5),
    dict(n=30, seed=13, n_preplaced=0, frac_fixed=0.0),
    dict(n=50, seed=17, frac_boundary=0.45),
    dict(n=21, seed=19, n_clusters=1, n_mib=1),
    dict(n=80, seed=53, n_preplaced=5, n_clusters=8, anchor_clusters=4),
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]

C = Counter()


def instrument(opt):
    o_stack = opt._stack_column
    o_bs = opt._band_solutions
    o_solve = opt._solve_band
    o_merge = opt._merge_band_once
    o_dyn = opt._dyn_split
    o_down = opt._place_unit_down
    o_strip = opt._band_strip

    def stack(ulist, x, w, pos):
        r = o_stack(ulist, x, w, pos)
        C["stack_column"] += 1
        occ = r[1]
        C["stack_with_obstacles" if occ else "stack_clean"] += 1
        if r[2] > opt.H * 1.0005:
            C["widen_retry_trigger"] += 1
        return r

    o_lay = opt._layout_full

    def lay(cols):
        r = o_lay(cols)
        C["layout"] += 1
        return r

    opt._layout_full = lay

    def bs(u, w):
        hit = u.hcache is not None and u.hcache[0] == w
        r = o_bs(u, w)
        C["bandsol_hit" if hit else "bandsol_miss"] += 1
        for eff, sol in r[2]:
            C["band_sol_ok" if sol is not None else "band_sol_none"] += 1
            if len(eff) > 1:
                C["band_multi_chunk"] += 1
        return r

    def solve(band, w):
        r = o_solve(band, w)
        C["solve_band_fail" if r is None else "solve_band_ok"] += 1
        return r

    def merge(band):
        C["merge_band_once"] += 1
        return o_merge(band)

    def dyn(ch, w):
        r = o_dyn(ch, w)
        C["dyn_split_none" if r is None else "dyn_split_ok"] += 1
        return r

    def down(u, x0, w, ytop, pos):
        C["place_unit_down"] += 1
        return o_down(u, x0, w, ytop, pos)

    def strip(x, w, lo, hi, placed):
        r = o_strip(x, w, lo, hi, placed)
        C["band_strip_none" if r is None else "band_strip_ok"] += 1
        return r

    opt._stack_column = stack
    opt._band_solutions = bs
    opt._solve_band = solve
    opt._merge_band_once = merge
    opt._dyn_split = dyn
    opt._place_unit_down = down
    opt._band_strip = strip


def main():
    for case in CASES:
        inst = build_instance(**case)
        opt = lg._ColumnOptimizer(
            inst.rects, inst.area_targets, inst.constraints,
            inst.target_positions, inst.b2b, inst.p2b, inst.pins,
            time.time() + 1000.0, seed=case["seed"])
        instrument(opt)
        cols = opt._init_columns(max(2, min(18, len(opt.units) // 6 + 2)))
        for step in range(400):
            opt.rng.seed(1000 + step)
            undo = opt._random_move(cols)
            if undo is None:
                continue
            opt._layout_full(cols)
            if step % 2:
                undo()
    for k in sorted(C):
        print(f"{k:24s} {C[k]}")


if __name__ == "__main__":
    main()
