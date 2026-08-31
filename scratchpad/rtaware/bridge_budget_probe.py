#!/usr/bin/env python
"""Offline headroom probe for the shipped GROUPING repair.

Question: is PARTNER_GROUP_BRIDGE_BUDGET (default 0.02 s, contest_optimizer.py
:1471-1478) the binding constraint on the residual grouping violations, or is
the MOVE SET in violation_killer._fix_grouping the limit?

Method: take a shipped 100-case layout set (evaluator --output JSON), rebuild
the instance-side `_ColumnOptimizer` scorer with exactly the shipping
convention (rects = the shipped layout, deadline = now + 1.0, seed = 0 -- the
same construction contest_optimizer.py:1428-1436 uses when direct_box is
empty), then call the SAME entry point the shipped pipeline calls,
`violation_killer.bridge_grouping_violations(scorer, layout, budget)`, at
budget in {0.02, 0.2, 1.0, 10.0}.

Reports per case: grouping / boundary / MIB counts before+after (the pass's own
exact counters), the EXACT evaluator no-runtime cost before+after (recomputed
with iccad2026_evaluate.evaluate_solution + compute_cost at runtime_factor 1),
wall time actually consumed, and whether _final_guards_ok + evaluator
feasibility still hold.

No evaluator run, no GPU, no solver edits: pure CPU replay on saved layouts.

Usage (run from FloorSet/iccad2026contest):
  uv run python ../../scratchpad/rtaware/bridge_budget_probe.py \
      ../../artifacts/shadow/limX10.json --data-path ../ --min-n 96
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "partner"),
           str(_REPO / "FloorSet" / "iccad2026contest"),
           str(_REPO / "FloorSet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from litetestLoader import FloorplanDatasetLiteTest            # noqa: E402
from column_sa_legalizer import _ColumnOptimizer               # noqa: E402
import violation_killer as VK                                  # noqa: E402
import iccad2026_evaluate as EV                                # noqa: E402

BUDGETS = (0.02, 0.2, 1.0, 10.0)


def build_inputs(dataset, idx):
    """Instance-side view of case `idx`, exactly as wall_seat_diag.build_opt."""
    sample = dataset[idx]
    area_target, b2b, p2b, pins, cons = sample["input"]
    polygons, _metrics = sample["label"]
    n = int((area_target != -1).sum().item())
    tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if not len(valid):
            continue
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        if nc > 1 and cons[i, 1] != 0:
            tpos[i] = torch.tensor([float(x_min), float(y_min),
                                    float(x_max - x_min), float(y_max - y_min)])
        elif nc > 0 and cons[i, 0] != 0:
            tpos[i, 2] = float(x_max - x_min)
            tpos[i, 3] = float(y_max - y_min)
    return (n, area_target[:n].detach().float().cpu(),
            cons[:n].detach().float().cpu(), tpos,
            b2b.detach().float().cpu(), p2b.detach().float().cpu(),
            pins.detach().float().cpu())


def make_scorer(layout, at, cons, tpos, b2b, p2b, pins):
    """contest_optimizer.py:1428-1436 convention."""
    return _ColumnOptimizer([tuple(map(float, r)) for r in layout],
                            at, cons, tpos, b2b, p2b, pins,
                            time.time() + 1.0, seed=0)


def exact_cost(layout, row, at, cons, tpos, b2b, p2b, pins):
    """Evaluator-exact no-runtime cost for `layout` (runtime_factor = 1)."""
    sol = {"positions": [tuple(map(float, r)) for r in layout], "runtime": 1.0}
    base = {"hpwl_baseline": row["hpwl_baseline"],
            "area_baseline": row["bbox_area_baseline"]}
    m = EV.evaluate_solution(sol, base, cons, b2b, p2b, pins, at,
                             target_positions=tpos, median_runtime=1.0)
    cost = EV.compute_cost(m.hpwl_gap, m.area_gap, m.violations_relative,
                           1.0, m.is_feasible)
    return cost, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--min-n", type=int, default=96)
    ap.add_argument("--ids", default="")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--jsonl", default="")
    a = ap.parse_args()
    budgets = [float(x) for x in a.budgets.split(",") if x.strip()]

    doc = json.load(open(a.result))
    rows = {int(r["test_id"]): r for r in doc["test_results"]}
    forced = {int(t) for t in a.ids.split(",") if t.strip()}
    ds = FloorplanDatasetLiteTest(a.data_path)

    want = sorted(t for t, r in rows.items()
                  if (t in forced)
                  or (int(r.get("grouping_violations") or 0) > 0
                      and int(r["block_count"]) >= a.min_n))
    print(f"# result={a.result} no_runtime={doc.get('total_score_no_runtime')}")
    print(f"# cases={want}")
    print(f"# budgets={budgets}")

    sink = open(a.jsonl, "w") if a.jsonl else None
    agg = {b: {"fixed": 0, "bits0": 0, "dcost": 0.0, "cases": 0, "sec": 0.0}
           for b in budgets}
    for tid in want:
        row = rows[tid]
        layout0 = row.get("raw_positions") or row.get("positions")
        if not layout0:
            print(f"tid={tid} SKIP no positions")
            continue
        layout0 = [tuple(map(float, r)) for r in layout0]
        n, at, cons, tpos, b2b, p2b, pins = build_inputs(ds, tid)
        if len(layout0) != n:
            print(f"tid={tid} SKIP len {len(layout0)} != n {n}")
            continue
        scorer0 = make_scorer(layout0, at, cons, tpos, b2b, p2b, pins)
        P0 = np.asarray(layout0, dtype=np.float64)
        g0 = VK._grouping_count(scorer0, P0)
        v0 = VK._violations_exact(scorer0, P0)
        b0 = len(VK._boundary_violators(scorer0, P0))
        m0 = VK._mib_count(scorer0, P0)
        c0, ev0 = exact_cost(layout0, row, at, cons, tpos, b2b, p2b, pins)
        print(f"tid={tid} n={n} eval_grp={row['grouping_violations']} "
              f"probe_grp={g0} bnd={b0} mib={m0} V={v0} "
              f"cost0={c0:.6f} (json {row['cost_no_runtime']:.6f}) "
              f"feas0={ev0.is_feasible}")
        for bud in budgets:
            # fresh scorer per budget: the pass mutates nothing, but its
            # deadline is baked in at construction.
            sc = make_scorer(layout0, at, cons, tpos, b2b, p2b, pins)
            t = time.perf_counter()
            out = VK.bridge_grouping_violations(sc, layout0, bud)
            dt = time.perf_counter() - t
            P1 = np.asarray([tuple(map(float, r)) for r in out],
                            dtype=np.float64)
            changed = (P1.shape != P0.shape) or bool(
                (np.abs(P1 - P0) > 1e-12).any())
            g1 = VK._grouping_count(sc, P1)
            b1 = len(VK._boundary_violators(sc, P1))
            mm1 = VK._mib_count(sc, P1)
            guards = VK._final_guards_ok(sc, P0, P1, list(sc.kind),
                                         list(sc.areas))
            c1, ev1 = exact_cost(out, row, at, cons, tpos, b2b, p2b, pins)
            agg[bud]["cases"] += 1
            agg[bud]["bits0"] += g0
            agg[bud]["fixed"] += max(0, g0 - g1)
            agg[bud]["dcost"] += (c1 - c0)
            agg[bud]["sec"] += dt
            print(f"   bud={bud:<6} t={dt:7.3f}s changed={int(changed)} "
                  f"grp {g0}->{g1} bnd {b0}->{b1} mib {m0}->{mm1} "
                  f"cost {c0:.6f}->{c1:.6f} d={c1 - c0:+.6f} "
                  f"guards={int(guards)} feas={int(ev1.is_feasible)} "
                  f"evgrp={ev1.grouping_violations}")
            if sink:
                sink.write(json.dumps({
                    "tid": tid, "n": n, "budget": bud, "wall_s": dt,
                    "changed": changed, "grp0": g0, "grp1": g1,
                    "bnd0": b0, "bnd1": b1, "mib0": m0, "mib1": mm1,
                    "cost0": c0, "cost1": c1, "guards_ok": bool(guards),
                    "feasible": bool(ev1.is_feasible),
                    "eval_grp1": int(ev1.grouping_violations)}) + "\n")
    if sink:
        sink.close()
    print("\n=== SUMMARY (probe grouping bits over the selected cases) ===")
    for bud in budgets:
        s = agg[bud]
        print(f"budget={bud:<6} cases={s['cases']} bits_before={s['bits0']} "
              f"bits_fixed={s['fixed']} "
              f"sum_cost_delta={s['dcost']:+.6f} "
              f"sum_wall={s['sec']:.2f}s "
              f"mean_wall={s['sec'] / max(s['cases'], 1):.3f}s")


main()
