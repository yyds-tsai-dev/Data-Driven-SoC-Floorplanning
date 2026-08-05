#!/usr/bin/env python3
"""Hot-spot anatomy of the direct-refine rung (`legalize_soft` / `_tighten`).

Single-threaded, cached Direct predictions, no evaluator, no pool.  Replays
`_Refiner.legalize_soft` (a ladder rung) on real validation cases and reports
where the time goes: cProfile cumulative table + a hand-rolled counter on the
inner legalize loop.

Usage:  uv run python scratchpad/refine_profile.py [test_id ...]
"""

from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402

WT = Path(__file__).resolve().parents[1]
MAIN = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
for p in (WT / "partner", MAIN / "FloorSet", MAIN / "FloorSet/iccad2026contest"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CACHE = WT / "scratchpad" / "anytime_preds"


def load_case(test_id: int):
    import torch
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(MAIN / "FloorSet"))
    sample = ds[test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b, p2b, pins, cons = inputs
    n = int((area_target != -1).sum().item())
    polygons, _metrics = labels
    tpos_gt = []
    for i in range(n):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if len(valid):
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            tpos_gt.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            tpos_gt.append((0.0, 0.0, 1.0, 1.0))
    opt_tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and cons[i, 0] != 0
        is_pre = nc > 1 and cons[i, 1] != 0
        if is_pre:
            opt_tpos[i] = torch.tensor(list(tpos_gt[i]))
        elif is_fixed:
            opt_tpos[i, 2] = tpos_gt[i][2]
            opt_tpos[i, 3] = tpos_gt[i][3]
    return (n, area_target[:n].float(), cons[:n].float(), opt_tpos,
            b2b.float(), p2b.float(), pins.float(), tpos_gt)


def preds_for(test_id: int):
    d = np.load(CACHE / f"preds_{test_id}.npz")
    return [d[k] for k in sorted(d.files)]


def build_opt(case, pred, span):
    import column_sa_legalizer as lg
    n, at, cons, tpos, b2b, p2b, pins, _ = case
    rect_list = [tuple(map(float, pred[i])) for i in range(n)]
    deadline = time.time() + span
    opt = lg._ColumnOptimizer(rect_list, at, cons, tpos, b2b, p2b, pins,
                              deadline, seed=317, v_weight=1.0)
    opt._tag_anchor = False
    return opt


def make_rung(case, pred, span=60.0):
    """A ladder rung's starting state: the _Refiner right before the first
    `legalize_soft` in `refine_prediction`."""
    import layout_refiner as rf
    opt = build_opt(case, pred, span)
    r = rf._Refiner(opt, np.asarray(pred, dtype=np.float64), seed=327)
    return opt, r


_PATCHED = []


def _install(rf, ctr):
    names = ["_axis_constraints", "_axis_pass", "_has_overlap",
             "_overlap_count", "_move", "_median_shift", "_evict",
             "legalize", "_reimpose_seeded_edges", "_free_rects",
             "_reshape_chain", "_reshape_clear", "_relocate_to_free",
             "_overlap_pairs"]
    for nm in names:
        real = getattr(rf._Refiner, nm)

        def mk(real=real, nm=nm):
            def w(self_, *a, **kw):
                t = time.perf_counter()
                try:
                    return real(self_, *a, **kw)
                finally:
                    e = ctr.setdefault(nm, [0, 0.0])
                    e[0] += 1
                    e[1] += time.perf_counter() - t
            return w
        setattr(rf._Refiner, nm, mk())
        _PATCHED.append((rf._Refiner, nm, real))


def _remove(rf):
    while _PATCHED:
        cls, nm, real = _PATCHED.pop()
        setattr(cls, nm, real)


def time_stage(case, preds, expand, label, tid):
    import layout_refiner as rf
    for pi, pred in enumerate(preds[:3]):
        opt, r = make_rung(case, pred)
        if expand:
            r.xmax += (r.xmax - r.xmin) * expand
            r.ymax += (r.ymax - r.ymin) * expand
        ctr = {}
        _install(rf, ctr)
        t0 = time.time()
        ok = r.legalize_soft(14, deadline=t0 + 30.0, fine=False)
        el = time.time() - t0
        _remove(rf)
        tot = ctr.get("legalize", [0, 0.0])[1]
        print(f"  t{tid} {label:10s} pred{pi} ok={int(ok)} {el:6.3f}s "
              f"(legalize {tot:6.3f}s)")
        for k in sorted(ctr, key=lambda k: -ctr[k][1]):
            c, s = ctr[k]
            print(f"        {k:22s} {c:6d} calls {s:7.3f}s "
                  f"{100.0 * s / max(el, 1e-9):5.1f}%")


def main():
    ids = [int(a) for a in sys.argv[1:]] or [95, 79, 64]
    for tid in ids:
        case = load_case(tid)
        preds = preds_for(tid)
        print(f"\n=== test_id={tid} n={case[0]} ===", flush=True)
        time_stage(case, preds, 0.0, "rungFIXED", tid)
        time_stage(case, preds, 0.28, "rung0.28", tid)

    tid = ids[0]
    case = load_case(tid)
    preds = preds_for(tid)
    opt, r = make_rung(case, preds[0])
    r.xmax += (r.xmax - r.xmin) * 0.28
    r.ymax += (r.ymax - r.ymin) * 0.28
    pr = cProfile.Profile()
    pr.enable()
    r.legalize_soft(14, deadline=time.time() + 60.0, fine=False)
    pr.disable()
    print(f"\n=== cProfile: legalize_soft, t{tid} pred0 expand=0.28 ===")
    pstats.Stats(pr).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
