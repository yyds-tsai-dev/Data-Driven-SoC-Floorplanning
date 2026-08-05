#!/usr/bin/env python3
"""Per-case fixed cost, old vs new, at the goal operating point.

One arm per process (the pool is forked AFTER the arm env is set, exactly
like the submission wrapper does), a real restart pool, no evaluator, no
scoring: this measures WALL CLOCK per case only.

  FLOOR_ARM=off|on  FLOORSET_MAIN=<root> uv run python \
      scratchpad/floor_wall_bench.py [n ...]
"""

from __future__ import annotations

import os
import sys
import time

ARM = os.environ.get("FLOOR_ARM", "off")
if ARM == "on":
    os.environ["PARTNER_FAST_SETUP"] = "1"
else:
    os.environ.pop("PARTNER_FAST_SETUP", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_anatomy import load_cases, med  # noqa: E402

REPS = int(os.environ.get("FLOOR_WALL_REPS", "5"))
POOL = int(os.environ.get("PARTNER_POOL", "24"))


def budget_of(n):
    import math
    return max(0.05, min(0.75, 5e-5 * math.exp(n / 12.0)))


def main():
    import column_sa_legalizer as lg
    from contest_optimizer import _heuristic_init
    targets = [int(a) for a in sys.argv[1:]] or [25, 50, 70, 100, 120]
    cases = load_cases(targets)
    lg.init_worker_pool(POOL)
    print("arm=%s pool_ready=%s size=%d" % (ARM, lg._POOL_READY, lg._POOL_SIZE))
    for case in cases:
        at = case["at"]
        cons = case["cons"]
        tpos = case["tpos"]
        b2b = case["b2b"]
        p2b = case["p2b"]
        pins = case["pins"]
        n = case["n"]
        b = budget_of(n)
        seeds = []
        walls = []
        cases_t = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            rects = _heuristic_init(at, cons, tpos, b2b, p2b, pins)
            t1 = time.perf_counter()
            out = lg.legalize_rectangles(rects, at, cons, tpos,
                                         b2b_connectivity=b2b,
                                         p2b_connectivity=p2b, pins_pos=pins,
                                         deadline=time.time() + b, seed=17)
            t2 = time.perf_counter()
            assert len(out) == n
            seeds.append(t1 - t0)
            walls.append(t2 - t1)
            cases_t.append(t2 - t0)
        print("tid=%-4d n=%-4d budget=%5.3f  seed=%7.2f ms  legalize=%8.2f ms"
              "  case=%8.2f ms" % (case["tid"], n, b, 1e3 * med(seeds),
                                   1e3 * med(walls), 1e3 * med(cases_t)))
    lg._shutdown_pool()


if __name__ == "__main__":
    main()
