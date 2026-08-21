"""ICDC Gate-0 pre-step: native (un-injected) weighted noRT of every candidate
layout file, i.e. "what would this layout score if it WERE the submission".

This is the quality axis of the transfer curve; the injected score (the other
axis) comes from the ORACLE_PRED_FILE chain.

usage: uv run python icdc_native.py <file.json> [<file.json> ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SS = Path(__file__).resolve().parent
sys.path.insert(0, str(SS))
import gr_lib as G  # noqa: E402

BANDS = [(21, 59), (60, 89), (90, 99), (100, 109), (110, 120)]


def wtotal(costs, ns, keys):
    if not keys:
        return float("nan")
    mx = max(ns[k] for k in keys)
    w = {k: math.exp((ns[k] - mx) / 12.0) for k in keys}
    return sum(costs[k] * w[k] for k in keys) / sum(w.values())


def main() -> None:
    ev = G.load_evaluator()
    cases = G.load_cases(ev, str(G.REPO / "FloorSet"))
    by_id = {c.test_id: c for c in cases}
    ns = {c.test_id: c.n for c in cases}
    print(f"loaded {len(cases)} cases")

    for path in sys.argv[1:]:
        p = Path(path)
        if not p.is_absolute():
            p = SS / p
        raw = json.load(open(p))
        costs, feas, miss = {}, 0, 0
        vrel, hg, ag = {}, {}, {}
        for k, v in raw.items():
            tid = int(k)
            c = by_id.get(tid)
            if c is None:
                continue
            rects = [tuple(float(x) for x in r) for r in v]
            if len(rects) != c.n:
                miss += 1
                continue
            m = G.evaluate(ev, c, rects)
            costs[tid] = float(m.cost_no_runtime)
            vrel[tid] = float(m.violations_relative)
            hg[tid] = max(0.0, float(m.hpwl_gap))
            ag[tid] = max(0.0, float(m.area_gap))
            feas += int(bool(m.is_feasible))
        keys = sorted(costs)
        line = [f"{p.name:24s} n_cases={len(keys):3d} feasible={feas:3d} "
                f"skipped={miss:3d}  noRT={wtotal(costs, ns, keys):.4f}"]
        for lo, hi in BANDS:
            bk = [k for k in keys if lo <= ns[k] <= hi]
            line.append(f"   band {lo:3d}-{hi:3d}: {wtotal(costs, ns, bk):.4f} "
                        f"(V_rel={wtotal(vrel, ns, bk):.4f} "
                        f"hg={wtotal(hg, ns, bk):.4f} ag={wtotal(ag, ns, bk):.4f})")
        print("\n".join(line))


if __name__ == "__main__":
    main()
