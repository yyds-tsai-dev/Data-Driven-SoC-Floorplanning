"""Compare noise-opt treatment vs the two no-noise-opt baselines using the
analyze_budget_scan alpha-projection math."""
import sys
from pathlib import Path
ROOT = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
sys.path.insert(0, str(ROOT / "scripts/probes"))
from analyze_budget_scan import total

PE = ROOT / "artifacts/partner_eval"
rows = [
    ("baseline budget35_ddim25_dm2", PE / "budget35_ddim25_dm2.json"),
    ("baseline budget35_dm2_rep2", PE / "budget35_dm2_rep2.json"),
    ("TREATMENT noiseopt", PE / "budget35_dm2_noiseopt.json"),
]
print(f"{'config':30s} {'noRT_Q':>8s} {'alphaProj':>10s} {'sum_rt':>8s} {'feas':>5s} {'tailQ':>8s}")
vals = {}
for name, p in rows:
    if not p.exists():
        print(f"{name:30s} MISSING")
        continue
    q, proj, rt, feas, tq = total(p)
    vals[name] = (q, proj, rt)
    print(f"{name:30s} {q:8.4f} {proj:10.4f} {rt:7.0f}s {feas:5d} {tq:8.4f}")

if "TREATMENT noiseopt" in vals:
    base = (vals["baseline budget35_ddim25_dm2"][0] + vals["baseline budget35_dm2_rep2"][0]) / 2
    tq, tproj, trt = vals["TREATMENT noiseopt"]
    brt = (vals["baseline budget35_ddim25_dm2"][2] + vals["baseline budget35_dm2_rep2"][2]) / 2
    print("\n--- verdict ---")
    print(f"baseline mean no_runtime = {base:.4f}  (1.1561/1.1518)")
    print(f"treatment no_runtime     = {tq:.4f}   delta = {base - tq:+.4f} (positive = improvement)")
    print(f"runtime: baseline mean {brt:.0f}s -> treatment {trt:.0f}s  ({trt/brt:.2f}x)")
    d = base - tq
    if d >= 0.005:
        print("=> DIRECTION CONFIRMED (>=0.005 beyond +-0.003 variance)")
    elif d >= 0.002:
        print("=> MARGINAL (0.002-0.005); scheduler decides on a re-test")
    else:
        print("=> WASH / REGRESSION -- honest kill")
