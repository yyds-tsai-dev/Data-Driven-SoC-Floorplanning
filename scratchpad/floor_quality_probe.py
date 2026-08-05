#!/usr/bin/env python3
"""Quality side of the PARTNER_FAST_SETUP anneal clamp.

The clamp does not remove overhead -- it removes SEARCH the case never asked
for.  This estimates what that search was worth: K independent restarts at
the OLD span (the flat 0.1 s floor) vs K at the NEW span (the planned share
of a 0.05 s budget), scored with the same true-cost rule `_parallel_solve`
uses to pick a winner, on a common normalizer.

Single thread, no pool, no evaluator.  This is a proxy, not evaluator
evidence: only a paired full run can promote or reject the flag.
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_anatomy import load_cases, med  # noqa: E402

K = int(os.environ.get("FLOOR_RESTARTS", "12"))
BUDGET = float(os.environ.get("FLOOR_BUDGET", "0.05"))


def chains(case, span, k, tag):
    import column_sa_legalizer as lg
    from contest_optimizer import _heuristic_init
    at = case["at"]
    cons = case["cons"]
    tpos = case["tpos"]
    b2b = case["b2b"]
    p2b = case["p2b"]
    pins = case["pins"]
    rects = _heuristic_init(at, cons, tpos, b2b, p2b, pins)
    out = []
    for s in range(k):
        opt = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                  time.time() + span, seed=11 * s + 7)
        opt.prepare()
        opt.finish(time.time() + span, max_runs=1)
        hp, area, V = opt.final_metrics
        out.append((hp, area, V, opt.area_ref, opt.n_soft_den))
    return out


def best(rows, hp_ref):
    def score(r):
        hp, area, V, area_ref, n_soft = r
        return (1.0 + 0.5 * ((hp - hp_ref) / hp_ref
                             + max(0.0, area / area_ref - 1.0))) \
            * math.exp(2.0 * V / n_soft)
    b = min(rows, key=score)
    return score(b), b


def main():
    targets = [int(a) for a in sys.argv[1:]] or [25, 50, 70]
    print("K=%d restarts per arm, budget=%.3f s" % (K, BUDGET))
    for case in load_cases(targets):
        # OLD: the flat floor.  NEW: the planned share of the case budget
        # (finish carves 28%% of the window for the polish).
        span_old = 0.1
        span_new = 0.72 * 0.85 * BUDGET
        rows_old = chains(case, span_old, K, "old")
        rows_new = chains(case, span_new, K, "new")
        hp_ref = max(min(r[0] for r in rows_old + rows_new), 1e-9)
        s_old, b_old = best(rows_old, hp_ref)
        s_new, b_new = best(rows_new, hp_ref)
        print("tid=%-4d n=%-4d span %5.1f -> %5.1f ms | best score %.5f -> "
              "%.5f (%+.2f%%) | V %d -> %d | hp %.0f -> %.0f"
              % (case["tid"], case["n"], 1e3 * span_old, 1e3 * span_new,
                 s_old, s_new, 100.0 * (s_new / s_old - 1.0),
                 b_old[2], b_new[2], b_old[0], b_new[0]))


if __name__ == "__main__":
    main()
