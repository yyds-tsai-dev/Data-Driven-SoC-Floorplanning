"""Profile the partner column-slicing SA/refine hot loop on one validation case.

Runs a single-threaded `_ColumnOptimizer.run()` (bypassing the multiprocessing
pool) on a real FloorSet-Lite validation case under cProfile, so we can see the
leaf functions that dominate SA cumtime. This is the SA worker's true inner work
(_anneal -> _evaluate -> _layout + _cost(_hpwl,_violations)).

Usage (from repo root):
    cd FloorSet/iccad2026contest
    PYTHONPATH=$ROOT/partner:$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet \
      uv run python $ROOT/scripts/probes/profile_partner_sa.py --case 99 --budget 3.5
"""
from __future__ import annotations

import argparse
import cProfile
import math
import pstats
import sys
import time

import torch


def load_case(case: int):
    from litetestLoader import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest("../")
    sample = ds[case]
    area_target, b2b, p2b, pins, constraints = sample["input"]
    labels = sample["label"]
    n = int((area_target != -1).sum().item())
    # target_positions like the evaluator builds them
    # labels layout mirrors _extract_baseline: fp_sol etc. We only need fixed/
    # preplaced shape/pos, so reconstruct from labels[0] (positions w/h) if present.
    opt_target = torch.full((n, 4), -1.0)
    # labels[0] (fp_sol) is per-block polygon vertices; bbox gives (x,y,w,h).
    try:
        fp_sol = labels[0]
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(n):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_pre = nc > 1 and constraints[i, 1] != 0
            if not (is_fixed or is_pre):
                continue
            poly = fp_sol[i]
            xs = poly[:, 0]
            ys = poly[:, 1]
            gx, gy = float(xs.min()), float(ys.min())
            gw = float(xs.max()) - gx
            gh = float(ys.max()) - gy
            if is_pre:
                opt_target[i] = torch.tensor([gx, gy, gw, gh])
            elif is_fixed:
                opt_target[i, 2] = gw
                opt_target[i, 3] = gh
    except Exception as e:
        print(f"[warn] could not build target_positions from labels: {e}")
    return n, area_target[:n], b2b, p2b, pins, constraints[:n], opt_target


def build_opt(n, area_target, b2b, p2b, pins, constraints, target, budget, seed):
    import column_sa_legalizer as lg
    seeds = []
    for i in range(n):
        a = max(float(area_target[i]), 1e-9)
        s = math.sqrt(a)
        seeds.append((0.0, 0.0, s, s))
    deadline = time.time() + budget
    opt = lg._ColumnOptimizer(
        seeds, area_target, constraints, target, b2b, p2b, pins,
        deadline, seed=seed,
    )
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=99)
    ap.add_argument("--budget", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--top", type=int, default=35)
    ap.add_argument("--out", default="/tmp/sa_profile.pstats")
    args = ap.parse_args()

    n, at, b2b, p2b, pins, cons, tgt = load_case(args.case)
    print(f"case {args.case}: n={n} blocks, budget={args.budget}s, seed={args.seed}")

    opt = build_opt(n, at, b2b, p2b, pins, cons, tgt, args.budget, args.seed)
    if opt.locked_only():
        print("locked-only case, no SA to profile; pick another case")
        return

    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    out = opt.run()
    pr.disable()
    wall = time.time() - t0
    print(f"run() wall={wall:.3f}s, produced {len(out)} rects")

    pr.dump_stats(args.out)
    st = pstats.Stats(pr)
    print("\n===== cumtime (top) =====")
    st.sort_stats("cumulative").print_stats(args.top)
    print("\n===== tottime (leaf, top) =====")
    st.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
