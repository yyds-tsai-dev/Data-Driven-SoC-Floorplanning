#!/usr/bin/env python
"""Feed the GOLDEN layout into our refine ladder + shipped post-pass chain.
Run from FloorSet/iccad2026contest.  Usage:
  uv run python .../golden_probe.py --budget 5 --min-n 102 --jsonl out.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "partner"), str(_REPO / "FloorSet" / "iccad2026contest"),
           str(_REPO / "FloorSet"), str(_REPO / "scripts" / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from litetestLoader import FloorplanDatasetLiteTest
import iccad2026_evaluate as EV
from column_sa_legalizer import _ColumnOptimizer
from layout_refiner import refine_prediction
from violation_killer import _violations_exact
from wall_seat_diag import classify


def build(ds, idx):
    s = ds[idx]
    at, b2b, p2b, pins, cons = s["input"]
    polys, metrics = s["label"]
    n = int((at != -1).sum().item())
    golden = []
    for i in range(n):
        blk = polys[i]; v = blk[blk[:, 0] != -1]
        if len(v):
            x0, y0 = v.min(dim=0).values; x1, y1 = v.max(dim=0).values
            golden.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            golden.append((0.0, 0.0, 1.0, 1.0))
    hb = EV.calculate_hpwl_b2b(golden, b2b); hp = EV.calculate_hpwl_p2b(golden, p2b, pins)
    area = EV.calculate_bbox_area(golden)
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0: area = float(metrics[0])
        if metrics[-2] > 0: hb = float(metrics[-2])
        if metrics[-1] >= 0: hp = float(metrics[-1])
    baseline = {"hpwl_baseline": hb + hp, "area_baseline": area}
    tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        pre = nc > 1 and cons[i, 1] != 0
        fix = nc > 0 and cons[i, 0] != 0
        if pre:
            tpos[i] = torch.tensor(list(golden[i]))
        elif fix:
            tpos[i, 2] = golden[i][2]; tpos[i, 3] = golden[i][3]
    return (n, at[:n].detach().float().cpu(), cons[:n].detach().float().cpu(),
            tpos, b2b.detach().float().cpu(), p2b.detach().float().cpu(),
            pins.detach().float().cpu(), golden, baseline)


def ev(pos, baseline, cons, b2b, p2b, pins, at, tpos):
    m = EV.evaluate_solution({"positions": [tuple(map(float, r)) for r in pos],
                              "runtime": 1.0}, baseline, cons, b2b, p2b, pins,
                             at, [tuple(map(float, r)) for r in tpos.tolist()],
                             median_runtime=1.0)
    return m


class _Shim:
    verbose = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=102)
    ap.add_argument("--ids", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--no-post", action="store_true")
    a = ap.parse_args()
    ds = FloorplanDatasetLiteTest(a.data_path)
    ids = [int(t) for t in a.ids.split(",") if t.strip()] or list(range(len(ds)))
    from contest_optimizer import MyOptimizer
    shim = _Shim()
    out_f = open(a.jsonl, "w") if a.jsonl else None
    for tid in ids:
        n, at, cons, tpos, b2b, p2b, pins, golden, base = build(ds, tid)
        if n < a.min_n:
            continue
        gm = ev(golden, base, cons, b2b, p2b, pins, at, tpos)
        opt0 = _ColumnOptimizer([tuple(map(float, r)) for r in golden], at, cons,
                                tpos, b2b, p2b, pins, time.time() + 1.0, seed=0)
        P = np.asarray([list(r) for r in golden], dtype=np.float64)
        cls = {}
        try:
            for r in classify(opt0, P):
                cls[r["cls"]] = cls.get(r["cls"], 0) + 1
        except Exception as e:
            cls = {"err": str(e)[:40]}
        row = {"tid": tid, "n": n, "g_cost": round(gm.cost, 4),
               "g_feas": bool(gm.is_feasible), "g_V": round(gm.violations_relative, 4),
               "g_bnd": gm.boundary_violations, "g_grp": gm.grouping_violations,
               "g_mib": gm.mib_violations, "g_nsoft": gm.max_possible_violations,
               "g_cls": cls, "budget": a.budget}
        t0 = time.time()
        opt = _ColumnOptimizer([tuple(map(float, r)) for r in golden], at, cons,
                               tpos, b2b, p2b, pins, time.time() + a.budget, seed=0)
        try:
            ref = refine_prediction(opt, P.copy(), time.time() + a.budget, seed=41)
        except Exception as e:
            ref = None; row["ref_err"] = str(e)[:80]
        row["ref_s"] = round(time.time() - t0, 2)
        if ref is None:
            row["refined"] = None
        else:
            rm = ev(ref, base, cons, b2b, p2b, pins, at, tpos)
            row.update(r_cost=round(rm.cost, 4), r_feas=bool(rm.is_feasible),
                       r_hpwl=round(rm.hpwl_gap, 4), r_area=round(rm.area_gap, 4),
                       r_V=round(rm.violations_relative, 4),
                       r_bnd=rm.boundary_violations, r_grp=rm.grouping_violations,
                       r_mib=rm.mib_violations)
            if not a.no_post:
                box = [(np.asarray([list(map(float, r)) for r in ref]), opt)]
                cur = [tuple(map(float, r)) for r in ref]
                try:
                    for fn, kw in ((MyOptimizer._coord_polish, {"elapsed": 0.0}),
                                   (MyOptimizer._final_seat, None),
                                   (MyOptimizer._tag_compress, None),
                                   (MyOptimizer._wall_repair_final, None)):
                        if kw is None:
                            cur = fn(shim, cur, at, cons, tpos, b2b, p2b, pins, box)
                        else:
                            cur = fn(shim, cur, at, cons, tpos, b2b, p2b, pins, **kw)
                    pm = ev(cur, base, cons, b2b, p2b, pins, at, tpos)
                    row.update(p_cost=round(pm.cost, 4), p_feas=bool(pm.is_feasible),
                               p_hpwl=round(pm.hpwl_gap, 4), p_area=round(pm.area_gap, 4),
                               p_V=round(pm.violations_relative, 4),
                               p_bnd=pm.boundary_violations, p_grp=pm.grouping_violations,
                               p_mib=pm.mib_violations)
                except Exception as e:
                    row["post_err"] = repr(e)[:120]
        print(json.dumps(row), flush=True)
        if out_f:
            out_f.write(json.dumps(row) + "\n"); out_f.flush()
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
