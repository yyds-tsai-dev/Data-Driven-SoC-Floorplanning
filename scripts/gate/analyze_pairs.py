#!/usr/bin/env python
"""Paired analysis of full-100 eval JSONs: arm A (base) vs arm B (variant).

Usage: uv run python analyze_pairs.py A1.json A2.json ... -- B1.json B2.json ...
Weighted no-runtime score per official weighting; per-case paired deltas with
bootstrap CI over cases; runtime shift stats.
"""
import json
import math
import random
import sys


def load(path):
    d = json.load(open(path))
    cases = {}
    for r in d["test_results"]:
        tid = r["test_id"] if isinstance(r, dict) else r[0]
        cases[tid] = r
    return d, cases


def weighted(cases, field="cost_no_runtime"):
    ns = {t: c["block_count"] for t, c in cases.items()}
    mx = max(ns.values())
    ws = {t: math.exp((ns[t] - mx) / 12.0) for t in cases}
    tot = sum(ws.values())
    return sum(ws[t] * cases[t][field] for t in cases) / tot, ws


def arm_stats(paths):
    per_case = {}
    totals = []
    rts = []
    for p in paths:
        d, cases = load(p)
        totals.append(d["total_score_no_runtime"])
        rts.append(d["summary"]["avg_runtime"])
        for t, c in cases.items():
            per_case.setdefault(t, []).append(c)
    mean_case = {
        t: {
            "cost_no_runtime": sum(c["cost_no_runtime"] for c in v) / len(v),
            "v_rel": sum(c.get("violations_relative", 0.0) for c in v) / len(v),
            "hpwl_gap": sum(c.get("hpwl_gap", 0.0) for c in v) / len(v),
            "area_gap": sum(c.get("area_gap", 0.0) for c in v) / len(v),
            "runtime": sum(c.get("runtime_seconds", 0.0) for c in v) / len(v),
            "block_count": v[0]["block_count"],
        }
        for t, v in per_case.items()
    }
    return totals, rts, mean_case


def main():
    argv = sys.argv[1:]
    split = argv.index("--")
    a_paths, b_paths = argv[:split], argv[split + 1:]
    at, art, ac = arm_stats(a_paths)
    bt, brt, bc = arm_stats(b_paths)

    def msd(x):
        m = sum(x) / len(x)
        sd = (sum((v - m) ** 2 for v in x) / max(len(x) - 1, 1)) ** 0.5
        return m, sd

    am, asd = msd(at)
    bm, bsd = msd(bt)
    print(f"A: n={len(at)} noRT {am:.4f} ± {asd:.4f}   avg_rt {sum(art)/len(art):.3f}")
    print(f"B: n={len(bt)} noRT {bm:.4f} ± {bsd:.4f}   avg_rt {sum(brt)/len(brt):.3f}")

    tids = sorted(set(ac) & set(bc))
    mxn = max(ac[t]["block_count"] for t in tids)
    ws = {t: math.exp((ac[t]["block_count"] - mxn) / 12.0) for t in tids}
    W = sum(ws.values())
    deltas = {t: bc[t]["cost_no_runtime"] - ac[t]["cost_no_runtime"] for t in tids}
    wdelta = sum(ws[t] * deltas[t] for t in tids) / W
    print(f"paired weighted delta (B-A): {wdelta:+.4f}")

    rng = random.Random(0)
    boots = []
    for _ in range(4000):
        s = [rng.choice(tids) for _ in tids]
        boots.append(sum(ws[t] * deltas[t] for t in s) / sum(ws[t] for t in s))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    print(f"bootstrap 95% CI over cases: [{lo:+.4f}, {hi:+.4f}]")

    for f in ("v_rel", "hpwl_gap", "area_gap"):
        da = sum(ws[t] * ac[t][f] for t in tids) / W
        db = sum(ws[t] * bc[t][f] for t in tids) / W
        print(f"weighted {f}: A {da:.4f} -> B {db:.4f} ({db-da:+.4f})")

    band = [t for t in tids if ac[t]["block_count"] >= 100]
    wb = sum(ws[t] for t in band)
    print(f"n>=100 band paired delta: {sum(ws[t]*deltas[t] for t in band)/wb:+.4f}")
    ra = sum(ac[t]["runtime"] for t in tids) / len(tids)
    rb = sum(bc[t]["runtime"] for t in tids) / len(tids)
    mra = max(ac[t]["runtime"] for t in tids)
    mrb = max(bc[t]["runtime"] for t in tids)
    print(f"runtime/case: A avg {ra:.3f} max {mra:.2f} -> B avg {rb:.3f} max {mrb:.2f}")
    worst = sorted(tids, key=lambda t: ws[t] * deltas[t])[:5]
    best = sorted(tids, key=lambda t: ws[t] * deltas[t])[-5:]
    print("top weighted improvements:", [(t, round(deltas[t], 4)) for t in worst])
    print("top weighted regressions:", [(t, round(deltas[t], 4)) for t in best])


if __name__ == "__main__":
    main()
