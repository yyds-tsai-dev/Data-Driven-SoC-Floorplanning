#!/usr/bin/env python3
"""Where the finish() deadline overshoot lives (goal tier, budget 0.05 s).

Wraps the stages `finish` runs -- _anneal, _greedy_polish, _layout,
_violations, _ensure_no_overlap -- and reports, per stage, the wall time and
how far past its own planned end it ran.  Single thread, no pool.

Usage:  FLOORSET_MAIN=<root> uv run python scratchpad/floor_finish_profile.py [n ...]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_anatomy import BUDGET, REPS, load_cases, med  # noqa: E402


def profile(case, span):
    import column_sa_legalizer as lg
    from contest_optimizer import _heuristic_init
    at = case["at"]
    cons = case["cons"]
    tpos = case["tpos"]
    b2b = case["b2b"]
    p2b = case["p2b"]
    pins = case["pins"]
    rects = _heuristic_init(at, cons, tpos, b2b, p2b, pins)

    ctr = {}
    names = ["_anneal", "_greedy_polish", "_layout", "_violations",
             "_hpwl", "_evaluate", "_random_move", "_restore", "_snapshot"]
    orig = {nm: getattr(lg._ColumnOptimizer, nm) for nm in names}

    def wrap(nm):
        real = orig[nm]

        def w(self, *a, **kw):
            t = time.perf_counter()
            try:
                return real(self, *a, **kw)
            finally:
                e = ctr.setdefault(nm, [0, 0.0])
                e[0] += 1
                e[1] += time.perf_counter() - t
        return w

    real_eno = lg._ensure_no_overlap

    def w_eno(*a, **kw):
        t = time.perf_counter()
        try:
            return real_eno(*a, **kw)
        finally:
            e = ctr.setdefault("_ensure_no_overlap", [0, 0.0])
            e[0] += 1
            e[1] += time.perf_counter() - t

    rows = []
    for rep in range(REPS):
        opt = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                  time.time() + span, seed=11 + rep)
        opt.prepare()
        ctr.clear()
        for nm in names:
            setattr(lg._ColumnOptimizer, nm, wrap(nm))
        lg._ensure_no_overlap = w_eno
        t0 = time.perf_counter()
        opt.finish(time.time() + span, max_runs=1)
        dt = time.perf_counter() - t0
        for nm in names:
            setattr(lg._ColumnOptimizer, nm, orig[nm])
        lg._ensure_no_overlap = real_eno
        rows.append((dt, {k: (v[0], v[1]) for k, v in ctr.items()}))
    return rows


def main():
    targets = [int(a) for a in sys.argv[1:]] or [25, 50, 70]
    for case in load_cases(targets):
        span = 0.85 * BUDGET
        rows = profile(case, span)
        tot = med([r[0] for r in rows])
        print("")
        print("=== tid=%d n=%d span=%.1f ms finish=%.1f ms (over %.1f ms) ==="
              % (case["tid"], case["n"], 1e3 * span, 1e3 * tot,
                 1e3 * (tot - span)))
        keys = set()
        for _dt, c in rows:
            keys |= set(c)
        for nm in sorted(keys):
            cs = med([c.get(nm, (0, 0.0))[0] for _d, c in rows])
            ts = med([c.get(nm, (0, 0.0))[1] for _d, c in rows])
            print("  %-20s n=%6d  %8.2f ms  (%5.1f%%)  %7.4f ms/call"
                  % (nm, cs, 1e3 * ts, 100.0 * ts / max(tot, 1e-9),
                     1e3 * ts / max(cs, 1)))


if __name__ == "__main__":
    main()
