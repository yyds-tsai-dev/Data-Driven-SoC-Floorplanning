"""Low-budget frontier analysis: Q-time elasticity down to the 0.2s/case regime.

Metrics match scratchpad/analyze_refstall.py (lambda ~ e^(n/12), alphaProj vs
alpha median with max(0.7, rtf^0.3), tailQ = lambda-weighted cost_no_runtime on
n>=100).  Adds per-band quality and runtime-vs-budget excess (fixed-overhead
floor diagnosis: excess = actual runtime - nominal budget curve).
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
MED = {int(r["test_id"]): float(r["median_runtime_s"]) for r in csv.DictReader(
    open(ROOT / "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"))}

# arm -> (BUDGET_MIN, BUDGET_MAX) actually exported in the chain scripts
ARMS = {
    "lbf_ctrl35_rep1": (0.8, 3.5),
    "lbf_max20": (0.8, 2.0),
    "lbf_max10": (0.8, 1.0),
    "lbf_max05": (0.3, 0.5),
    "lbf_max05_dm": (0.3, 0.5),
    "lbf_max025": (0.15, 0.25),
    "lbf_ctrl35_rep2": (0.8, 3.5),
    # cliff localization chain
    "lbf_max30": (0.8, 3.0),
    "lbf_max25": (0.8, 2.5),
    "lbf_max20_dmoff": (0.8, 2.0),
    "lbf_ctrl35_rep3": (0.8, 3.5),
    # pool-gate frontier chain (PARTNER_POOL_GATE=0 arms)
    "pg_ctrl35": (0.8, 3.5),
    "pg_max20": (0.8, 2.0),
    "pg_max10": (0.8, 1.0),
    "pg_max05": (0.3, 0.5),
    "pg_max025": (0.15, 0.25),
    "pg_max05_dm": (0.3, 0.5),
    "pg_max35_full": (0.8, 3.5),
    # ladder hypothesis probe
    "ctrl35_dmoff": (0.8, 3.5),
    "pg2_dmoff": (0.8, 2.0),
    "pg2_nref6": (0.8, 2.0),
    "pg2_rep2": (0.8, 2.0),
    # early-exit probe
    "ee_ctrl35": (0.8, 3.5),
    "ee_on35": (0.8, 3.5),
    "ee_on50": (0.8, 5.0),
    "ee_on35_pg0": (0.8, 3.5),
    # 0.25s operating-point refinement
    "pg025_rep2": (0.15, 0.25),
    "pg025_ee": (0.15, 0.25),
    "pg025_noflow": (0.15, 0.25),
    "pg05_ee": (0.3, 0.5),
}


def budget(n, lo, hi):
    return max(lo, min(hi, 0.06 * math.exp(n / 20.0)))


def summarize(path, lo, hi):
    d = json.load(open(path))
    res = d["test_results"]
    lam = [math.exp(r["block_count"] / 12) for r in res]
    s = sum(lam)
    proj = tail_q = tail_lam = 0.0
    bands = {"<60": [0.0, 0.0], "60-99": [0.0, 0.0], ">=100": [0.0, 0.0]}
    excess = []
    for r, l in zip(res, lam):
        q = 10.0 if not r["is_feasible"] else r["cost_no_runtime"]
        rtf = r["runtime_seconds"] / MED[r["test_id"]]
        proj += (l / s) * min(q * max(0.7, rtf ** 0.3), 10 - 1e-6)
        n = r["block_count"]
        key = "<60" if n < 60 else ("60-99" if n < 100 else ">=100")
        bands[key][0] += l * q
        bands[key][1] += l
        if n >= 100:
            tail_q += l * q
            tail_lam += l
        excess.append(r["runtime_seconds"] - budget(n, lo, hi))
    rt = sorted(r["runtime_seconds"] for r in res)
    n = len(rt)
    ex = sorted(excess)
    return dict(noRT=d["total_score_no_runtime"], proj=proj, sum_rt=sum(rt),
                feas=sum(1 for r in res if r["is_feasible"]),
                tailQ=tail_q / max(tail_lam, 1e-9),
                bands={k: v[0] / max(v[1], 1e-9) for k, v in bands.items()},
                avg=sum(rt) / n, p50=st.median(rt), p90=rt[int(0.9 * n) - 1],
                mx=rt[-1], ex_p50=ex[n // 2], ex_p90=ex[int(0.9 * n) - 1],
                ex_max=ex[-1],
                per_case={r["test_id"]: (r["cost_no_runtime"],
                                        r["runtime_seconds"]) for r in res})


hdr = (f"{'arm':16s} {'noRT_Q':>8s} {'proj':>7s} {'sum_rt':>7s} {'avg':>6s} "
       f"{'p90':>6s} {'max':>6s} {'feas':>5s} {'tailQ':>7s} {'b<60':>7s} "
       f"{'b60-99':>7s} {'b>=100':>7s} {'exP50':>6s} {'exP90':>6s} {'exMax':>6s}")
print(hdr)
data = {}
for arm, (lo, hi) in ARMS.items():
    p = ROOT / f"artifacts/partner_eval/{arm}.json"
    if not p.exists():
        print(f"{arm:16s} MISSING")
        continue
    v = summarize(p, lo, hi)
    data[arm] = v
    b = v["bands"]
    print(f"{arm:16s} {v['noRT']:8.4f} {v['proj']:7.4f} {v['sum_rt']:6.0f}s "
          f"{v['avg']:6.2f} {v['p90']:6.2f} {v['mx']:6.2f} {v['feas']:5d} "
          f"{v['tailQ']:7.4f} {b['<60']:7.4f} {b['60-99']:7.4f} "
          f"{b['>=100']:7.4f} {v['ex_p50']:6.2f} {v['ex_p90']:6.2f} "
          f"{v['ex_max']:6.2f}")

if "lbf_ctrl35_rep1" in data and "lbf_ctrl35_rep2" in data:
    a, b = data["lbf_ctrl35_rep1"], data["lbf_ctrl35_rep2"]
    print(f"\ncontrol drift rep2-rep1: dNoRT={b['noRT'] - a['noRT']:+.4f} "
          f"dProj={b['proj'] - a['proj']:+.4f}")
    ctrl_noRT = (a["noRT"] + b["noRT"]) / 2
    print("\nfrontier vs mean control (noRT cost paid per second saved):")
    mean_rt = (a["sum_rt"] + b["sum_rt"]) / 2
    for arm in ("lbf_max20", "lbf_max10", "lbf_max05", "lbf_max05_dm",
                "lbf_max025"):
        if arm not in data:
            continue
        v = data[arm]
        dq = v["noRT"] - ctrl_noRT
        dt = mean_rt - v["sum_rt"]
        print(f"  {arm:14s} dNoRT={dq:+.4f} saved={dt:6.0f}s "
              f"avg={v['avg']:.2f}s cost/100s={dq / max(dt, 1e-9) * 100:+.4f}")
