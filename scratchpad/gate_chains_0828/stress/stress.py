"""Robustness stress sweep of the PACKAGED optimizer against adversarial
synthetic instances built with tests/synth_instances.build_instance.

Mirrors the evaluator's own loading/calling contract (iccad2026_evaluate.py
_load_optimizer + the solve() call site around lines 862-886), and reuses the
evaluator's own evaluate_solution()/compute_cost() for feasibility/cost
checks instead of reimplementing them.

Run with the clean venv python; PYTHONPATH must include
FloorSet/iccad2026contest and FloorSet.
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
import time
import traceback

REPO = "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning"
PKG_DIR = "/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/pack_test6/x/cadc1013"
OUT_CSV = "/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/stress/results.csv"

sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, PKG_DIR)  # package dir first, like the evaluator's _load_optimizer

import torch  # noqa: E402
from synth_instances import build_instance  # noqa: E402

from iccad2026_evaluate import evaluate_solution, compute_cost  # noqa: E402
from op_wrapper import MyOptimizer  # noqa: E402


def make_target_positions(n, constraints, tpos):
    """Reproduce the evaluator's opt_target_pos construction
    (iccad2026_evaluate.py lines ~868-880): -1 default, preplaced gets
    full xywh, fixed-shape gets only wh."""
    opt_target_pos = torch.full((n, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = tpos[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = tpos[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def build_grid():
    """Adversarial knob grid over build_instance's exposed parameters:
    n, seed, frac_fixed, n_preplaced, frac_boundary, n_clusters, n_mib,
    edge_mult, n_pins, anchor_clusters. No aspect-ratio knob is exposed by
    build_instance (areas come from a fixed lognormal; aspect ratio for
    fixed blocks is drawn from exp(N(0,0.45)) -- not adjustable), so we push
    the exposed knobs to their extremes instead."""
    cases = []

    def add(name, n, seed, **kw):
        cases.append((name, dict(n=n, seed=seed, **kw)))

    ns = [21, 60, 90, 105, 120]
    seeds = [1, 2, 3]

    # 1. Baseline-ish across n and seeds
    for n in ns:
        for seed in seeds:
            add(f"base_n{n}_s{seed}", n, seed)

    # 2. Many clusters (up to near n/2 groups of size>=2)
    for n in ns:
        for seed in seeds[:2]:
            add(f"manyclusters_n{n}_s{seed}", n, seed, n_clusters=max(2, n // 3))

    # 3. Many MIB groups
    for n in ns:
        for seed in seeds[:2]:
            add(f"manymib_n{n}_s{seed}", n, seed, n_mib=max(2, n // 4))

    # 4. Many fixed-shape blocks (extreme frac_fixed)
    for n in ns:
        for seed in seeds[:2]:
            add(f"fixedheavy_n{n}_s{seed}", n, seed, frac_fixed=0.6)

    # 5. Many preplaced blocks incl. interior obstacles.
    # build_instance places preplaced blocks uniformly in [0.05,0.6]*side in
    # both x and y -> already interior-biased (not on the true boundary at
    # side). We push n_preplaced high to stress this.
    for n in ns:
        for seed in seeds[:2]:
            add(f"preplacedheavy_n{n}_s{seed}", n, seed,
                n_preplaced=max(3, n // 5))

    # 6. Heavy boundary constraints
    for n in ns:
        for seed in seeds[:2]:
            add(f"boundaryheavy_n{n}_s{seed}", n, seed, frac_boundary=0.9)

    # 7. Zero pins
    for n in ns:
        add(f"zeropins_n{n}_s1", n, 1, n_pins=0)

    # 8. Very many pins
    for n in ns:
        add(f"manypins_n{n}_s1", n, 1, n_pins=max(200, n * 5))

    # 9. Anchored clusters (preplaced glued into cluster groups -> stresses
    # anchor-gluing branch in the legalizer)
    for n in ns:
        for seed in seeds[:2]:
            add(f"anchored_n{n}_s{seed}", n, seed,
                n_clusters=6, n_preplaced=5, anchor_clusters=4)

    # 10. Kitchen-sink combo: everything stressed at once
    for n in ns:
        for seed in seeds:
            add(f"kitchensink_n{n}_s{seed}", n, seed,
                frac_fixed=0.4, n_preplaced=max(3, n // 6),
                frac_boundary=0.5, n_clusters=max(3, n // 5),
                n_mib=max(2, n // 6), edge_mult=6.0,
                n_pins=max(100, n * 3), anchor_clusters=3)

    # 11. High edge_mult (dense b2b connectivity)
    for n in [60, 120]:
        add(f"denseedges_n{n}_s1", n, 1, edge_mult=15.0)

    # 12. n_preplaced == n (degenerate: everything preplaced)
    for n in [21, 60]:
        add(f"allpreplaced_n{n}_s1", n, 1, n_preplaced=n)

    # 13. n_fixed near-all
    for n in [21, 60]:
        add(f"almostallfixed_n{n}_s1", n, 1, frac_fixed=0.95)

    return cases


def dummy_baseline():
    # We only care about feasibility/violations, not gap-to-golden (no
    # golden layout exists for synthetic instances). Zero-gap baseline
    # keeps evaluate_solution's is_feasible / violations_relative /
    # cost machinery intact without fabricating a reference.
    return {}


def run():
    cases = build_grid()
    print(f"Total instances: {len(cases)}")

    optimizer = MyOptimizer()
    print(f"Optimizer instantiated: {type(optimizer)}")

    rows = []
    first_call_runtime = None

    for i, (name, kw) in enumerate(cases):
        n = kw["n"]
        seed = kw["seed"]
        row = dict(name=name, n=n, seed=seed, knobs=str(kw),
                   runtime=None, exception=None, is_feasible=None,
                   overlap=None, area_gap=None, hpwl_gap=None,
                   violations_relative=None, cost=None)
        try:
            inst = build_instance(**kw)
            constraints = inst.constraints
            tpos = inst.target_positions
            opt_tpos = make_target_positions(n, constraints, tpos)

            t0 = time.time()
            positions = optimizer.solve(
                n, inst.area_targets, inst.b2b, inst.p2b, inst.pins,
                constraints, opt_tpos,
            )
            dt = time.time() - t0
            row["runtime"] = dt
            if first_call_runtime is None:
                first_call_runtime = dt

            if not isinstance(positions, list) or len(positions) != n:
                raise ValueError(
                    f"solve() returned malformed result: type={type(positions)} "
                    f"len={len(positions) if hasattr(positions, '__len__') else 'NA'} expected {n}")

            metrics = evaluate_solution(
                {'positions': positions, 'runtime': dt},
                dummy_baseline(),
                constraints,
                inst.b2b,
                inst.p2b,
                inst.pins,
                inst.area_targets,
                tpos,
                median_runtime=1.0,
            )
            row["is_feasible"] = metrics.is_feasible
            row["area_gap"] = metrics.area_gap
            row["hpwl_gap"] = metrics.hpwl_gap
            row["violations_relative"] = metrics.violations_relative
            row["cost"] = compute_cost(metrics.hpwl_gap, metrics.area_gap,
                                        metrics.violations_relative, 1.0,
                                        metrics.is_feasible)
        except Exception as e:
            row["exception"] = f"{type(e).__name__}: {e}"
            row["cost"] = 10.0  # M_PENALTY-equivalent for ranking worst cases
            traceback.print_exc()

        rows.append(row)
        print(f"[{i+1}/{len(cases)}] {name}: runtime={row['runtime']} "
              f"feasible={row['is_feasible']} exc={row['exception']}")

    # Write CSV
    fieldnames = ["name", "n", "seed", "knobs", "runtime", "exception",
                  "is_feasible", "overlap", "area_gap", "hpwl_gap",
                  "violations_relative", "cost"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary
    n_total = len(rows)
    n_infeasible = sum(1 for r in rows if r["exception"] is None and not r["is_feasible"])
    n_exc = sum(1 for r in rows if r["exception"] is not None)
    print("\n===== SUMMARY =====")
    print(f"Total instances: {n_total}")
    print(f"Infeasible (no exception): {n_infeasible}")
    print(f"Exceptions: {n_exc}")
    print(f"First-call runtime: {first_call_runtime}")

    by_n = {}
    for r in rows:
        if r["runtime"] is None:
            continue
        by_n.setdefault(r["n"], []).append(r["runtime"])
    print("\nRuntime by n (max / p90 / mean):")
    for n in sorted(by_n):
        rts = sorted(by_n[n])
        mx = rts[-1]
        p90 = rts[int(0.9 * (len(rts) - 1))]
        mean = sum(rts) / len(rts)
        print(f"  n={n}: max={mx:.3f}s p90={p90:.3f}s mean={mean:.3f}s (count={len(rts)})")

    print("\nWorst 10 by cost:")
    ranked = sorted(rows, key=lambda r: (r["cost"] if r["cost"] is not None else -1), reverse=True)
    for r in ranked[:10]:
        print(f"  {r['name']} cost={r['cost']} feasible={r['is_feasible']} "
              f"exc={r['exception']} knobs={r['knobs']}")

    if n_exc:
        print("\nExceptions detail:")
        for r in rows:
            if r["exception"] is not None:
                print(f"  {r['name']}: {r['exception']}")


if __name__ == "__main__":
    run()
