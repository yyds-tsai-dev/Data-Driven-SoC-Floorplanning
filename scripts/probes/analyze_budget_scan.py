"""Summarize budget scan: no-runtime Q per MAX + alpha-referenced projection.

Deadline-bounded pipeline: official runtime = budget, so the projection
multiplies each case's quality by max(0.7, (runtime/alpha_median)^0.3).
Baseline (MAX=24) = cont_retrieval_direct_control.json.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MED = {int(r["test_id"]): float(r["median_runtime_s"]) for r in csv.DictReader(
    open(ROOT / "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"))}


def total(path):
    d = json.load(open(path))
    res = d["test_results"]
    lam = [math.exp(r["block_count"] / 12) for r in res]
    s = sum(lam)
    proj = 0.0
    tail_q = 0.0
    tail_lam = 0.0
    for r, l in zip(res, lam):
        q = 10.0 if not r["is_feasible"] else r["cost_no_runtime"]
        rtf = r["runtime_seconds"] / MED[r["test_id"]]
        proj += (l / s) * min(q * max(0.7, rtf ** 0.3), 10 - 1e-6)
        if r["block_count"] >= 100:
            tail_q += l * q
            tail_lam += l
    rt = sum(r["runtime_seconds"] for r in res)
    feas = sum(1 for r in res if r["is_feasible"])
    return (d["total_score_no_runtime"], proj, rt, feas,
            tail_q / max(tail_lam, 1e-9))


if __name__ == "__main__":
    print(f"{'config':30s} {'noRT_Q':>8s} {'alphaProj':>10s} {'sum_rt':>8s} "
          f"{'feas':>5s} {'tailQ(n>=100)':>14s}")
    for name in ["cont_retrieval_direct_control", "budget_scan_max8",
                 "budget_scan_max6", "budget_scan_max4"]:
        p = ROOT / f"artifacts/partner_eval/{name}.json"
        if p.exists():
            q, proj, rt, feas, tq = total(p)
            print(f"{name:30s} {q:8.4f} {proj:10.4f} {rt:7.0f}s {feas:5d} {tq:14.4f}")
        else:
            print(f"{name:30s} MISSING")
