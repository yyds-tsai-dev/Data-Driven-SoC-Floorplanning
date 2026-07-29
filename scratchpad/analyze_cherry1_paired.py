"""Per-case paired win/loss for cherry-pick batch 1 (pooled over rounds)."""
import json
from pathlib import Path

ROOT = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
SUFS = ["", "_rep2", "_rep3"]


def cases(p):
    d = json.load(open(p))
    return {r["test_id"]: r for r in d["test_results"]}


for cfg in ("fastsa", "stallstop"):
    w = l = t = 0
    dsum = 0.0
    tw = tl = 0
    for suf in SUFS:
        pc = ROOT / f"artifacts/partner_eval/cherry1_control{suf}.json"
        pt = ROOT / f"artifacts/partner_eval/cherry1_{cfg}{suf}.json"
        if not (pc.exists() and pt.exists()):
            continue
        C, T = cases(pc), cases(pt)
        for k in C:
            if k not in T:
                continue
            a = 10.0 if not C[k]["is_feasible"] else C[k]["cost_no_runtime"]
            b = 10.0 if not T[k]["is_feasible"] else T[k]["cost_no_runtime"]
            dsum += b - a
            if b < a - 1e-9:
                w += 1
                if C[k]["block_count"] >= 100:
                    tw += 1
            elif b > a + 1e-9:
                l += 1
                if C[k]["block_count"] >= 100:
                    tl += 1
            else:
                t += 1
    print(f"{cfg:10s} paired cases: win={w} loss={l} tie={t}  "
          f"mean per-case dCost={dsum/max(w+l+t,1):+.5f}   "
          f"n>=100: win={tw} loss={tl}")
