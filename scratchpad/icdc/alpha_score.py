"""alpha-curve probe, step 4: the fidelity -> score curve.

usage: alpha_score.py <arm>:<tag,tag> <arm>:<tag,tag> ...
   e.g. alpha_score.py c:c_r1,c_r2 a00:a00_r1,a00_r2 ...
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SS = Path(__file__).resolve().parent
BANDS = [(21, 59), (60, 89), (90, 99), (100, 109), (110, 120)]
ALPHA = {"c": None, "a00": 0.00, "a25": 0.25, "a50": 0.50, "a75": 0.75, "o": 1.00}


def load(tag):
    d = json.load(open(SS / f"eval_alpha_{tag}.json"))
    return d, {r["test_id"]: r for r in d["test_results"]}


def wtotal(rows, keys):
    if not keys:
        return float("nan")
    mx = max(rows[k]["block_count"] for k in keys)
    w = {k: math.exp((rows[k]["block_count"] - mx) / 12) for k in keys}
    return sum(rows[k]["cost_no_runtime"] * w[k] for k in keys) / sum(w.values())


def wshare(rows, keys):
    mx = max(rows[k]["block_count"] for k in rows)
    w = {k: math.exp((rows[k]["block_count"] - mx) / 12) for k in rows}
    return sum(w[k] for k in keys) / sum(w.values())


def merge(tags):
    per, rt, mrt, feas = {}, [], [], []
    for t in tags:
        d, rows = load(t)
        rt.append(d["summary"]["avg_runtime"])
        mrt.append(d["summary"]["max_runtime"])
        feas.append(sum(1 for r in rows.values()
                        if r.get("feasible", r.get("is_feasible", True))))
        for k, r in rows.items():
            per.setdefault(k, []).append(r["cost_no_runtime"])
    _d, rows0 = load(tags[0])
    m = {k: {"block_count": rows0[k]["block_count"],
             "cost_no_runtime": sum(v) / len(v)} for k, v in per.items()}
    return m, sum(rt) / len(rt), max(mrt), min(feas)


def main():
    arms = []
    for spec in sys.argv[1:]:
        name, tags = spec.split(":")
        arms.append((name, tags.split(",")))

    data = {}
    for name, tags in arms:
        data[name] = merge(tags)

    _m0, _, _, _ = data[arms[0][0]]
    allk = sorted(_m0)
    live = [k for k in allk if _m0[k]["block_count"] >= 103]
    ge95 = [k for k in allk if _m0[k]["block_count"] >= 95]

    print(f"live band (n>=103) = {len(live)} cases, weight share "
          f"{wshare(_m0, live):.3f};  n>=95 share {wshare(_m0, ge95):.3f}\n")
    hdr = (f"{'arm':>5} {'alpha':>6} {'full100':>8} {'n>=103':>8} {'n>=95':>8} "
           f"{'avg_rt':>7} {'max_rt':>7} {'feas':>5}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for name, tags in arms:
        m, art, mrt, feas = data[name]
        tot = wtotal(m, allk)
        a = ALPHA.get(name)
        astr = "ctl" if a is None else f"{a:.2f}"
        if name == "a00":
            base = tot
        print(f"{name:>5} {astr:>6} {tot:>8.4f} {wtotal(m, live):>8.4f} "
              f"{wtotal(m, ge95):>8.4f} {art:>7.3f} {mrt:>7.3f} {feas:>5d}")

    # curve shape: gain fraction realised by each alpha
    print("\n== curve shape (full-100 noRT, anchored on a00 -> o) ==")
    if "a00" in data and "o" in data:
        lo = wtotal(data["a00"][0], allk)
        hi = wtotal(data["o"][0], allk)
        span = lo - hi
        print(f"  a00={lo:.4f}  o={hi:.4f}  total span={span:+.4f}")
        prev_a, prev_v = 0.0, lo
        for name in ("a00", "a25", "a50", "a75", "o"):
            if name not in data:
                continue
            v = wtotal(data[name][0], allk)
            a = ALPHA[name]
            frac = (lo - v) / span if abs(span) > 1e-9 else float("nan")
            slope = ((prev_v - v) / (a - prev_a)) if a > prev_a else float("nan")
            print(f"  alpha={a:.2f}  noRT={v:.4f}  gain={lo - v:+.4f}  "
                  f"realised={frac * 100:5.1f}%  local slope d(noRT)/d(alpha)="
                  f"{-slope:+.4f}")
            prev_a, prev_v = a, v

    print("\n== per-case, live band (n>=103) ==")
    names = [n for n, _ in arms]
    print("  tid    n " + "".join(f"{n:>9}" for n in names))
    for k in sorted(live, key=lambda k: _m0[k]["block_count"]):
        row = "".join(f"{data[n][0][k]['cost_no_runtime']:>9.4f}" for n in names)
        print(f"{k:>5} {_m0[k]['block_count']:>4} {row}")


if __name__ == "__main__":
    main()
