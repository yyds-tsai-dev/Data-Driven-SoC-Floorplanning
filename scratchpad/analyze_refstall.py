"""refstall A/B: noRT / alphaProj / sum_rt / tailQ / feasible + runtime dist.

Same scoring as scripts/probes/analyze_budget_scan.py (deadline-bounded
pipeline => official runtime == budget, so the alpha projection multiplies
each case's quality by max(0.7, (runtime/alpha_median)**0.3)), plus the
per-case runtime distribution this experiment is judged on.
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
MED = {int(r["test_id"]): float(r["median_runtime_s"]) for r in csv.DictReader(
    open(ROOT / "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"))}


def summarize(path):
    d = json.load(open(path))
    res = d["test_results"]
    lam = [math.exp(r["block_count"] / 12) for r in res]
    s = sum(lam)
    proj = tail_q = tail_lam = 0.0
    for r, l in zip(res, lam):
        q = 10.0 if not r["is_feasible"] else r["cost_no_runtime"]
        rtf = r["runtime_seconds"] / MED[r["test_id"]]
        proj += (l / s) * min(q * max(0.7, rtf ** 0.3), 10 - 1e-6)
        if r["block_count"] >= 100:
            tail_q += l * q
            tail_lam += l
    rt = sorted(r["runtime_seconds"] for r in res)
    n = len(rt)
    return dict(noRT=d["total_score_no_runtime"], proj=proj, sum_rt=sum(rt),
                feas=sum(1 for r in res if r["is_feasible"]),
                tailQ=tail_q / max(tail_lam, 1e-9),
                avg=sum(rt) / n, p50=st.median(rt),
                p90=rt[int(0.9 * n) - 1], mx=rt[-1],
                per_case={r["test_id"]: (r["cost_no_runtime"],
                                        r["runtime_seconds"]) for r in res})


CFGS = ["control", "on", "both"]
ROUNDS = ["", "_rep2", "_rep3"]
hdr = (f"{'config':22s} {'noRT_Q':>8s} {'alphaProj':>10s} {'sum_rt':>8s} "
       f"{'feas':>5s} {'tailQ':>7s} {'avg':>6s} {'p50':>6s} {'p90':>6s} "
       f"{'max':>6s}")
print(hdr)
data = {}
for suf in ROUNDS:
    for c in CFGS:
        p = ROOT / f"artifacts/partner_eval/refstall_{c}{suf}.json"
        name = f"{c}{suf or '_rep1'}"
        if not p.exists():
            print(f"{name:22s} MISSING")
            continue
        v = summarize(p)
        data[(c, suf)] = v
        print(f"{name:22s} {v['noRT']:8.4f} {v['proj']:10.4f} "
              f"{v['sum_rt']:7.0f}s {v['feas']:5d} {v['tailQ']:7.4f} "
              f"{v['avg']:6.2f} {v['p50']:6.2f} {v['p90']:6.2f} {v['mx']:6.2f}")

print("\npaired same-round deltas vs control (negative = better)")
print(f"{'round':8s} {'arm':6s} {'dNoRT':>9s} {'dProj':>9s} {'dSumRt':>9s} "
      f"{'dSumRt%':>8s} {'dTailQ':>9s} {'cases_worse':>12s}")
for suf in ROUNDS:
    if ("control", suf) not in data:
        continue
    ctl = data[("control", suf)]
    for c in CFGS[1:]:
        if (c, suf) not in data:
            continue
        v = data[(c, suf)]
        worse = sum(1 for k, (q, _t) in v["per_case"].items()
                    if q > ctl["per_case"][k][0] + 1e-9)
        print(f"{suf or '_rep1':8s} {c:6s} {v['noRT'] - ctl['noRT']:9.4f} "
              f"{v['proj'] - ctl['proj']:9.4f} "
              f"{v['sum_rt'] - ctl['sum_rt']:8.1f}s "
              f"{100 * (v['sum_rt'] / ctl['sum_rt'] - 1):7.1f}% "
              f"{v['tailQ'] - ctl['tailQ']:9.4f} {worse:12d}")

print("\nmean over available rounds")
for c in CFGS:
    vs = [data[(c, s)] for s in ROUNDS if (c, s) in data]
    if not vs:
        continue
    print(f"{c:8s} noRT={sum(v['noRT'] for v in vs) / len(vs):.4f} "
          f"proj={sum(v['proj'] for v in vs) / len(vs):.4f} "
          f"sum_rt={sum(v['sum_rt'] for v in vs) / len(vs):.0f}s "
          f"avg={sum(v['avg'] for v in vs) / len(vs):.2f}s "
          f"p90={sum(v['p90'] for v in vs) / len(vs):.2f}s "
          f"(n={len(vs)})")
