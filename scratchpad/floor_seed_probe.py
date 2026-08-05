#!/usr/bin/env python3
"""How the `_heuristic_init` seed cost splits (edge counts + per-loop time)."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_anatomy import REPS, load_cases, med  # noqa: E402


def main():
    import torch
    from contest_optimizer import _heuristic_init
    targets = [int(a) for a in sys.argv[1:]] or [25, 50, 70]
    for case in load_cases(targets):
        at = case["at"]
        cons = case["cons"]
        tpos = case["tpos"]
        b2b = case["b2b"]
        p2b = case["p2b"]
        pins = case["pins"]
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            _heuristic_init(at, cons, tpos, b2b, p2b, pins)
            ts.append(time.perf_counter() - t0)
        nb = int((b2b[:, 0] != -1).sum().item())
        npb = int((p2b[:, 0] != -1).sum().item())
        print("tid=%-4d n=%-4d seed=%7.2f ms  b2b_edges=%-6d p2b_edges=%-6d "
              "pins=%d" % (case["tid"], case["n"], 1e3 * med(ts), nb, npb,
                           pins.shape[0]))


if __name__ == "__main__":
    main()
