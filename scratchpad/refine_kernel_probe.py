#!/usr/bin/env python3
"""Bit-exactness + speed probe for PARTNER_REFINE_KERNEL.

Runs a ladder rung (`legalize_soft`) twice from the same start state -- once
on the Python path, once on the kernel -- and compares the resulting `P`
arrays with `==`.  Then times both.

Usage:  uv run python scratchpad/refine_kernel_probe.py [test_id ...]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_profile import load_case, preds_for, build_opt  # noqa: E402


def fresh(case, pred, expand, kernel):
    import layout_refiner as rf
    if kernel:
        os.environ["PARTNER_REFINE_KERNEL"] = "numba"
    else:
        os.environ.pop("PARTNER_REFINE_KERNEL", None)
    opt = build_opt(case, pred, 120.0)
    r = rf._Refiner(opt, np.asarray(pred, dtype=np.float64), seed=327)
    if expand:
        r.xmax += (r.xmax - r.xmin) * expand
        r.ymax += (r.ymax - r.ymin) * expand
    return r


def one(case, pred, expand, label, tid):
    rp = fresh(case, pred, expand, False)
    t0 = time.time()
    ok_p = rp.legalize_soft(14, deadline=t0 + 300.0, fine=False)
    tp = time.time() - t0

    rk = fresh(case, pred, expand, True)
    assert rk._nk is not None, "kernel did not attach"
    t0 = time.time()
    ok_k = rk.legalize_soft(14, deadline=t0 + 300.0, fine=False)
    tk = time.time() - t0

    same = bool(np.array_equal(rp.P, rk.P))
    print(f"t{tid} {label:6s} ok {int(ok_p)}/{int(ok_k)}  "
          f"py {tp:6.3f}s  nb {tk:6.3f}s  x{tp / max(tk, 1e-9):5.2f}  "
          f"bit-exact={same}"
          + ("" if same else
             f"  maxdiff={np.abs(rp.P - rk.P).max():.3e}"
             f"  ndiff={int((rp.P != rk.P).sum())}"))
    return same, tp, tk


def main():
    ids = [int(a) for a in sys.argv[1:]] or [95, 79, 64]
    allsame = True
    sp, sk = 0.0, 0.0
    for tid in ids:
        case = load_case(tid)
        preds = preds_for(tid)
        print(f"--- t{tid} n={case[0]} ---")
        for label, expand in (("FIXED", 0.0), ("0.02", 0.02), ("0.28", 0.28),
                              ("0.50", 0.50)):
            for pi in range(3):
                s, tp, tk = one(case, preds[pi], expand, label, tid)
                allsame &= s
                sp += tp
                sk += tk
    print(f"\nTOTAL py {sp:.3f}s  nb {sk:.3f}s  x{sp / max(sk, 1e-9):.2f}  "
          f"all-bit-exact={allsame}")


if __name__ == "__main__":
    main()
