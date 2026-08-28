#!/usr/bin/env python
"""Per-band paired analysis of full-100 eval JSONs, with channel attribution.

Usage:
    uv run python scripts/gate/band_pairs.py A1.json A2.json -- B1.json B2.json

Arm A is the reference (base), arm B the candidate; any number of reps per arm.

Why bands: a model change is not uniform in block count.  A tail-tilted flow
fine-tune measured on 2026-08-28 (docs/experiments/
2026-08-21-post-beta-p0-execution.md Sec.17t) RAISED the direct/flow channel's
refined-candidate supply at n>=90 and COLLAPSED it at n=76-89, where the case
then ships a column-slicing layout worth +0.15 to +2.0 cost.  A single
whole-suite delta averages those two effects into noise; the band split and
the per-band channel census show them separately.

Channel attribution (heuristic, from `positions` in the eval JSON): a
column-slicing layout places every block on one of C<=18 column x-coordinates
(plus the preplaced blocks), so its number of DISTINCT left-x values over n is
~0.2; a refined direct/flow layout keeps continuous coordinates and sits at
~0.75-0.85.  Measured on 1600 case-runs the two modes are separated by an
almost empty valley (6 samples in [0.40, 0.60)), so the 0.50 cut is not
delicate.  Override with BAND_PAIRS_NX_FRAC.

Weighting is the official exp((n - n_max)/12) normalised over ALL cases in the
file, so the per-band weighted deltas SUM to the whole-suite delta.
"""
import json
import math
import os
import sys
from collections import defaultdict

BANDS = ((0, 75), (76, 89), (90, 104), (105, 120))
NX_FRAC = float(os.environ.get("BAND_PAIRS_NX_FRAC", "0.50"))


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return {r["test_id"]: r for r in d["test_results"]}


def channel(rec):
    """'col' | 'dir' | '?'  -- see the module docstring."""
    pos = rec.get("positions")
    n = rec.get("block_count") or (len(pos) if pos else 0)
    if not pos or not n:
        return "?"
    xs = {round(float(r[0]), 3) for r in pos}
    return "col" if (len(xs) / n) <= NX_FRAC else "dir"


def arm(paths):
    """{tid: {'n':, 'cost': mean over reps, 'col': #reps shipping column}}"""
    acc = defaultdict(list)
    for p in paths:
        for tid, rec in load(p).items():
            acc[tid].append(rec)
    out = {}
    for tid, recs in acc.items():
        out[tid] = {
            "n": recs[0]["block_count"],
            "cost": sum(r["cost_no_runtime"] for r in recs) / len(recs),
            "col": sum(1 for r in recs if channel(r) == "col"),
            "reps": len(recs),
        }
    return out


def main(argv):
    if "--" not in argv:
        print(__doc__)
        return 2
    cut = argv.index("--")
    A, B = arm(argv[:cut]), arm(argv[cut + 1:])
    tids = sorted(set(A) & set(B))
    if not tids:
        print("no common test ids")
        return 2
    nmax = max(A[t]["n"] for t in tids)
    W = sum(math.exp((A[t]["n"] - nmax) / 12.0) for t in tids)
    w = {t: math.exp((A[t]["n"] - nmax) / 12.0) / W for t in tids}
    d = {t: w[t] * (B[t]["cost"] - A[t]["cost"]) for t in tids}

    print(f"cases={len(tids)}  reps A={A[tids[0]]['reps']} B={B[tids[0]]['reps']}"
          f"  weighting=exp((n-{nmax})/12) normalised over all {len(tids)} cases")
    print(f"{'band':>10} {'cases':>5} {'wDelta':>9} | "
          f"{'A col/dir':>11} {'B col/dir':>11} | {'A wcost':>8} {'B wcost':>8}")
    total = 0.0
    for lo, hi in BANDS:
        ts = [t for t in tids if lo <= A[t]["n"] <= hi]
        if not ts:
            continue
        bd = sum(d[t] for t in ts)
        total += bd
        acol = sum(A[t]["col"] for t in ts)
        bcol = sum(B[t]["col"] for t in ts)
        arep = sum(A[t]["reps"] for t in ts)
        brep = sum(B[t]["reps"] for t in ts)
        wa = sum(w[t] * A[t]["cost"] for t in ts)
        wb = sum(w[t] * B[t]["cost"] for t in ts)
        print(f"{lo:>4}-{hi:<5} {len(ts):>5} {bd:>+9.5f} | "
              f"{acol:>4}/{arep - acol:<6} {bcol:>4}/{brep - bcol:<6} | "
              f"{wa:>8.5f} {wb:>8.5f}")
    print(f"{'TOTAL':>10} {len(tids):>5} {total:>+9.5f} | "
          f"A={sum(w[t] * A[t]['cost'] for t in tids):.5f} "
          f"B={sum(w[t] * B[t]['cost'] for t in tids):.5f}")

    k = int(os.environ.get("BAND_PAIRS_TOPK", "3"))
    print("\nper-band movers (wDelta; ch = column-shipping reps A->B)")
    for lo, hi in BANDS:
        ts = [t for t in tids if lo <= A[t]["n"] <= hi]
        if not ts:
            continue
        srt = sorted(ts, key=lambda t: d[t])
        print(f"  n {lo}-{hi}:")
        for t in srt[:k]:
            if d[t] >= 0:
                break
            print(f"    improve tid={t:<3} n={A[t]['n']:<3} "
                  f"{A[t]['cost']:.4f} -> {B[t]['cost']:.4f} "
                  f"wD={d[t]:+.5f}  ch={A[t]['col']}->{B[t]['col']}")
        for t in reversed(srt[-k:]):
            if d[t] <= 0:
                break
            print(f"    regress tid={t:<3} n={A[t]['n']:<3} "
                  f"{A[t]['cost']:.4f} -> {B[t]['cost']:.4f} "
                  f"wD={d[t]:+.5f}  ch={A[t]['col']}->{B[t]['col']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
