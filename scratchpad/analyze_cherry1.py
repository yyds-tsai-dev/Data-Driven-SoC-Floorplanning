"""Cherry-pick batch 1 A/B summary (noRT / alphaProj / sum_rt / feasible /
tailQ), reusing the metric definitions in scripts/probes/analyze_budget_scan.py.
Only same-round deltas are trusted (GPU shared with the flow v3 training)."""
import sys
from pathlib import Path

ROOT = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
sys.path.insert(0, str(ROOT / "scripts/probes"))
from analyze_budget_scan import total  # noqa: E402

ROUNDS = [("r1", ""), ("r2", "_rep2"), ("r3", "_rep3")]
CFGS = ["control", "fastsa", "stallstop"]

print(f"{'run':22s} {'noRT_Q':>8s} {'alphaProj':>10s} {'sum_rt':>8s} "
      f"{'feas':>5s} {'tailQ':>8s}   {'dNoRT':>8s} {'dProj':>8s} {'dRT':>7s}")
agg = {c: [] for c in CFGS}
for rname, suf in ROUNDS:
    base = None
    for cfg in CFGS:
        p = ROOT / f"artifacts/partner_eval/cherry1_{cfg}{suf}.json"
        if not p.exists():
            print(f"{rname+'/'+cfg:22s} MISSING")
            continue
        q, proj, rt, feas, tq = total(p)
        if cfg == "control":
            base = (q, proj, rt)
            d = "        -        -       -"
        else:
            d = f"{q-base[0]:+8.4f} {proj-base[1]:+8.4f} {rt-base[2]:+6.0f}s"
            agg[cfg].append((q - base[0], proj - base[1], rt - base[2]))
        print(f"{rname+'/'+cfg:22s} {q:8.4f} {proj:10.4f} {rt:7.0f}s "
              f"{feas:5d} {tq:8.4f}   {d}")
    print()

print("mean paired delta vs same-round control")
for cfg in CFGS[1:]:
    v = agg[cfg]
    if not v:
        continue
    n = len(v)
    print(f"  {cfg:10s} n={n}  dNoRT={sum(x[0] for x in v)/n:+.4f}  "
          f"dProj={sum(x[1] for x in v)/n:+.4f}  "
          f"dRT={sum(x[2] for x in v)/n:+.1f}s")
