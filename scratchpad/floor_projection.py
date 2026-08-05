#!/usr/bin/env python3
"""Projected full-validation runtime saving from PARTNER_FAST_SETUP.

Model (goal tier, b(n) = 5e-5 * exp(n/12) clamped to [0.05, 0.75]):
  anneal span  = 0.72 * 0.85 * b   (finish carves 28% for the polish, the
                                    pool carves 15% as the worker margin)
  old wall     = overhead + max(0.1, span)
  new wall     = overhead + span            (+ the measured seed saving)
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_anatomy import MAIN, med  # noqa: E402


def main():
    import torch
    from lite_dataset_test import FloorplanDatasetLiteTest
    import contest_optimizer as co
    ds = FloorplanDatasetLiteTest(str(MAIN / "FloorSet"))
    tot_old = tot_new = 0.0
    seed_old = seed_new = 0.0
    hit = 0
    for tid in range(100):
        s = ds[tid]
        area_target, b2b, p2b, pins, cons = s["input"]
        n = int((area_target != -1).sum().item())
        at = area_target[:n].float()
        cn = cons[:n].float()
        tp = torch.full((n, 4), -1.0)
        args = (at, cn, tp, b2b.float(), p2b.float(), pins.float())
        os.environ.pop("PARTNER_FAST_SETUP", None)
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            co._heuristic_init(*args)
            t.append(time.perf_counter() - t0)
        so = med(t)
        os.environ["PARTNER_FAST_SETUP"] = "1"
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            co._heuristic_init(*args)
            t.append(time.perf_counter() - t0)
        sn = med(t)
        os.environ.pop("PARTNER_FAST_SETUP", None)
        b = max(0.05, min(0.75, 5e-5 * math.exp(n / 12.0)))
        span = 0.72 * 0.85 * b
        old = max(0.1, span)
        if old > span + 1e-12:
            hit += 1
        tot_old += old
        tot_new += span
        seed_old += so
        seed_new += sn
    print("cases where the anneal floor binds: %d/100" % hit)
    print("anneal span total: %.3f s -> %.3f s  (saves %.3f s)"
          % (tot_old, tot_new, tot_old - tot_new))
    print("seed total:        %.3f s -> %.3f s  (saves %.3f s)"
          % (seed_old, seed_new, seed_old - seed_new))
    print("projected per-run saving: %.2f s"
          % ((tot_old - tot_new) + (seed_old - seed_new)))


if __name__ == "__main__":
    main()
