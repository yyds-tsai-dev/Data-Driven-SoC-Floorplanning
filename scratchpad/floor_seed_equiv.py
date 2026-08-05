#!/usr/bin/env python3
"""PARTNER_FAST_SETUP seed twin: bit-equality + speed over all 100 real
validation cases (function-level, no evaluator, no pool, single thread)."""

from __future__ import annotations

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
    bad = 0
    rows = []
    for tid in range(100):
        s = ds[tid]
        area_target, b2b, p2b, pins, cons = s["input"]
        n = int((area_target != -1).sum().item())
        polygons, _m = s["label"]
        tpos = torch.full((n, 4), -1.0)
        nc = cons.shape[1] if cons.dim() > 1 else 0
        for i in range(n):
            blk = polygons[i]
            valid = blk[blk[:, 0] != -1]
            if not len(valid):
                continue
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            if nc > 1 and cons[i, 1] != 0:
                tpos[i] = torch.tensor([float(x0), float(y0),
                                        float(x1 - x0), float(y1 - y0)])
            elif nc > 0 and cons[i, 0] != 0:
                tpos[i, 2] = float(x1 - x0)
                tpos[i, 3] = float(y1 - y0)
        at = area_target[:n].float()
        cn = cons[:n].float()
        bb = b2b.float()
        pb = p2b.float()
        pn = pins.float()
        args = (at, cn, tpos, bb, pb, pn)
        os.environ.pop("PARTNER_FAST_SETUP", None)
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            ref = co._heuristic_init(*args)
            t.append(time.perf_counter() - t0)
        t_old = med(t)
        os.environ["PARTNER_FAST_SETUP"] = "1"
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            got = co._heuristic_init(*args)
            t.append(time.perf_counter() - t0)
        t_new = med(t)
        os.environ.pop("PARTNER_FAST_SETUP", None)
        same = (len(ref) == len(got)
                and all(a == b for ra, rb in zip(ref, got)
                        for a, b in zip(ra, rb)))
        if not same:
            bad += 1
            print("MISMATCH tid=%d n=%d" % (tid, n))
        rows.append((tid, n, t_old, t_new))
    rows.sort(key=lambda r: -r[2])
    print("")
    print("worst-10 seed cost (old -> new):")
    for tid, n, a, b in rows[:10]:
        print("  tid=%-4d n=%-4d %8.2f ms -> %6.2f ms  (%.1fx)"
              % (tid, n, 1e3 * a, 1e3 * b, a / max(b, 1e-9)))
    tot_old = sum(r[2] for r in rows)
    tot_new = sum(r[3] for r in rows)
    print("  total over 100 cases: %.3f s -> %.3f s (saves %.3f s)"
          % (tot_old, tot_new, tot_old - tot_new))
    print("bit-exact: %d/100 cases" % (100 - bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
