"""csaref A/B: noRT / alphaProj / tailQ / per-band + hpwl_gap & area_gap.

Same scoring as scratchpad/analyze_refstall.py (deadline-bounded pipeline =>
official runtime == budget, so the alpha projection multiplies each case's
quality by max(0.7, (runtime/alpha_median)**0.3)), plus the hpwl/area factor
split -- the CSA bet is a pure hpwl_gap bet, so the split is what says whether
the capture happened and what it cost.
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
MED = {int(r["test_id"]): float(r["median_runtime_s"]) for r in csv.DictReader(
    open(ROOT / "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"))}
BANDS = [(0, 60), (60, 100), (100, 121)]


def summarize(path):
    d = json.load(open(path))
    res = d["test_results"]
    lam = [math.exp(r["block_count"] / 12) for r in res]
    s = sum(lam)
    proj = tail_q = tail_lam = 0.0
    bands = {b: [0.0, 0.0] for b in BANDS}
    for r, l in zip(res, lam):
        q = 10.0 if not r["is_feasible"] else r["cost_no_runtime"]
        rtf = r["runtime_seconds"] / MED[r["test_id"]]
        proj += (l / s) * min(q * max(0.7, rtf ** 0.3), 10 - 1e-6)
        n = r["block_count"]
        for b in BANDS:
            if b[0] <= n < b[1]:
                bands[b][0] += l * q
                bands[b][1] += l
        if n >= 100:
            tail_q += l * q
            tail_lam += l
    rt = sorted(r["runtime_seconds"] for r in res)
    tail = [r for r in res if r["block_count"] >= 100]
    return dict(
        noRT=d["total_score_no_runtime"], proj=proj, sum_rt=sum(rt),
        feas=sum(1 for r in res if r["is_feasible"]),
        tailQ=tail_q / max(tail_lam, 1e-9),
        bands={b: bands[b][0] / max(bands[b][1], 1e-9) for b in BANDS},
        hpwl=st.mean(r["hpwl_gap"] for r in res),
        area=st.mean(r["area_gap"] for r in res),
        vrel=st.mean(r["violations_relative"] for r in res),
        t_hpwl=st.mean(r["hpwl_gap"] for r in tail),
        t_area=st.mean(r["area_gap"] for r in tail),
        t_vrel=st.mean(r["violations_relative"] for r in tail),
        avg=sum(rt) / len(rt), p50=st.median(rt),
        p90=rt[int(0.9 * len(rt)) - 1], mx=rt[-1],
        per_case={r["test_id"]: (r["cost_no_runtime"], r["runtime_seconds"])
                  for r in res})


CFGS = ["control", "stall", "end"]
ROUNDS = ["", "_rep2", "_rep3"]
print(f"{'config':16s} {'noRT_Q':>8s} {'proj':>8s} {'sum_rt':>8s} {'feas':>5s} "
      f"{'tailQ':>7s} {'b<60':>7s} {'b60-99':>7s} {'b100+':>7s} "
      f"{'hpwlG':>7s} {'areaG':>7s} {'Vrel':>7s} {'p90':>6s} {'max':>6s}")
data = {}
for suf in ROUNDS:
    for c in CFGS:
        p = ROOT / f"artifacts/partner_eval/csaref_{c}{suf}.json"
        name = f"{c}{suf or '_rep1'}"
        if not p.exists():
            print(f"{name:16s} MISSING")
            continue
        v = summarize(p)
        data[(c, suf)] = v
        print(f"{name:16s} {v['noRT']:8.4f} {v['proj']:8.4f} {v['sum_rt']:7.0f}s "
              f"{v['feas']:5d} {v['tailQ']:7.4f} "
              f"{v['bands'][BANDS[0]]:7.4f} {v['bands'][BANDS[1]]:7.4f} "
              f"{v['bands'][BANDS[2]]:7.4f} {v['hpwl']:7.4f} {v['area']:7.4f} "
              f"{v['vrel']:7.4f} {v['p90']:6.2f} {v['mx']:6.2f}")

print("\npaired same-round deltas vs control (negative = better)")
print(f"{'round':8s} {'arm':6s} {'dNoRT':>9s} {'dProj':>9s} {'dTailQ':>9s} "
      f"{'dSumRt':>8s} {'d_t_hpwl':>9s} {'d_t_area':>9s} {'d_t_Vrel':>9s} "
      f"{'wins':>5s} {'loss':>5s}")
for suf in ROUNDS:
    if ("control", suf) not in data:
        continue
    ctl = data[("control", suf)]
    for c in CFGS[1:]:
        if (c, suf) not in data:
            continue
        v = data[(c, suf)]
        win = sum(1 for k, (q, _t) in v["per_case"].items()
                  if q < ctl["per_case"][k][0] - 1e-9)
        los = sum(1 for k, (q, _t) in v["per_case"].items()
                  if q > ctl["per_case"][k][0] + 1e-9)
        print(f"{suf or '_rep1':8s} {c:6s} {v['noRT'] - ctl['noRT']:9.4f} "
              f"{v['proj'] - ctl['proj']:9.4f} {v['tailQ'] - ctl['tailQ']:9.4f} "
              f"{v['sum_rt'] - ctl['sum_rt']:7.1f}s "
              f"{v['t_hpwl'] - ctl['t_hpwl']:9.4f} "
              f"{v['t_area'] - ctl['t_area']:9.4f} "
              f"{v['t_vrel'] - ctl['t_vrel']:9.4f} {win:5d} {los:5d}")

print("\nmean over available rounds")
for c in CFGS:
    vs = [data[(c, s)] for s in ROUNDS if (c, s) in data]
    if not vs:
        continue
    m = lambda k: sum(v[k] for v in vs) / len(vs)
    print(f"{c:8s} noRT={m('noRT'):.4f} proj={m('proj'):.4f} "
          f"tailQ={m('tailQ'):.4f} sum_rt={m('sum_rt'):.0f}s "
          f"t_hpwl={m('t_hpwl'):.4f} t_area={m('t_area'):.4f} "
          f"t_Vrel={m('t_vrel'):.4f} p90={m('p90'):.2f}s (n={len(vs)})")
if ("control", "") in data:
    ctl = [data[("control", s)] for s in ROUNDS if ("control", s) in data]
    for c in CFGS[1:]:
        pairs = [(data[(c, s)]["noRT"] - data[("control", s)]["noRT"])
                 for s in ROUNDS
                 if (c, s) in data and ("control", s) in data]
        if pairs:
            print(f"paired dNoRT {c}: {[round(x, 4) for x in pairs]} "
                  f"mean={sum(pairs) / len(pairs):+.4f}")
