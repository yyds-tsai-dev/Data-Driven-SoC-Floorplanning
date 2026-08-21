"""Golden repair floor probe -- batch 2b (final).

Same topology-fixed, shape-frozen, preplaced-frozen repair as batch 1/2, but
boundary abutment now uses the evaluator's own definition (block attains the
layout min/max on that axis) instead of being pinned to the golden frame.
The frame may therefore shrink, which is score-free.

Stages, each accepted only on an exact-evaluator cost drop:
  A) extremal-boundary LP under golden relative order
  B) wall-reversal (reverse all of a block's in-/out-edges on one axis --
     provably keeps the constraint graph acyclic)
  C) grouping bridges (force contact between two components of a cluster)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_lib as G  # noqa: E402
from gr_repair2 import build_assign, split, reverse_edges, bridge_candidates  # noqa: E402

OUT = Path(__file__).resolve().parent


def run_pipeline(case, assign, contacts, enforce=None):
    """enforce=None -> pick greedily.  Returns (layout, blocked_set)."""
    n = case.n
    rects = case.rects
    x0, y0, x1, y1 = case.frame
    hor = G.transitive_reduce(n, split(assign)[0])
    ver = G.transitive_reduce(n, split(assign)[1])

    sw = np.array([r[2] for r in rects])
    sh = np.array([r[3] for r in rects])
    pre = set(np.nonzero(case.preplaced)[0].tolist())
    lox = np.full(n, x0); hix = np.full(n, x1)
    loy = np.full(n, y0); hiy = np.full(n, y1)
    eqx = {i: float(rects[i][0]) for i in pre}
    eqy = {i: float(rects[i][1]) for i in pre}

    status = G.boundary_status(rects, case.bound)
    want = {}   # block -> set of bits it needs
    for i, (code, miss) in status.items():
        bits = {bit for bit in (1, 2, 4, 8) if code & bit}
        want[i] = bits
    rowsx = G.contact_rows(contacts, rects, "x")
    rowsy = G.contact_rows(contacts, rects, "y")

    netsx, pinx = G.collect_nets(case, "x")
    netsy, piny = G.collect_nets(case, "y")

    def solve(sel, objective=True):
        minx = [i for i in sel if 1 in want[i]]
        maxx = [i for i in sel if 2 in want[i]]
        miny = [i for i in sel if 8 in want[i]]
        maxy = [i for i in sel if 4 in want[i]]
        sx = G.solve_axis_ext(n, sw, lox, hix, hor, netsx, pinx,
                              eqx, minx, maxx, rowsx, objective)
        if sx is None:
            return None
        sy = G.solve_axis_ext(n, sh, loy, hiy, ver, netsy, piny,
                              eqy, miny, maxy, rowsy, objective)
        if sy is None:
            return None
        return [(float(sx[i]), float(sy[i]), float(sw[i]), float(sh[i]))
                for i in range(n)]

    all_b = sorted(want)
    blocked = set()
    if enforce is None:
        sel = list(all_b)
        if solve(sel, objective=False) is None:
            # greedy: satisfied-in-golden first, then the rest one by one
            sat = [i for i in all_b if not status[i][1]]
            sel = [i for i in sat if solve([i], objective=False) is not None]
            base = list(sel)
            for i in sat:
                if i not in sel and solve(base + [i], objective=False) is not None:
                    base.append(i)
            sel = base
            rest = sorted((i for i in all_b if i not in sel),
                          key=lambda i: len(want[i]))
            for i in rest:
                if solve(sel + [i], objective=False) is not None:
                    sel.append(i)
                else:
                    blocked.add(i)
    else:
        sel = list(enforce)
    lay = solve(sel)
    if lay is None:
        return None, blocked, sel
    for (i, j, ax, _d) in contacts:      # exact abutment for shapely
        if ax == "x":
            lay[j] = (lay[i][0] + lay[i][2], lay[j][1], lay[j][2], lay[j][3])
        else:
            lay[j] = (lay[j][0], lay[i][1] + lay[i][3], lay[j][2], lay[j][3])
    return lay, blocked, sel


def process_case(ev, case, gold_m):
    rects = case.rects
    assign = build_assign(rects)
    contacts, _ = G.cluster_contacts(rects, case.clust)
    pre = set(np.nonzero(case.preplaced)[0].tolist())

    def score(a, c):
        lay, blk, sel = run_pipeline(case, a, c)
        if lay is None:
            return None, None, None
        m = G.evaluate(ev, case, lay)
        if not m.is_feasible:
            return None, None, None
        return m.cost_no_runtime, lay, blk

    best_cost, best_lay, blocked = score(assign, contacts)
    if best_lay is None:
        best_cost, best_lay, blocked = gold_m.cost_no_runtime, list(rects), set()
    stageA = best_cost
    edits = []

    # ---- B) wall reversal ----------------------------------------------
    for _round in range(2):
        cur = G.boundary_status(best_lay, case.bound)
        todo = [(i, miss) for i, (code, miss) in cur.items() if miss and i not in pre]
        progressed = False
        for i, miss in todo:
            for bit in miss:
                axis, direction = {1: ("h", "in"), 2: ("h", "out"),
                                   8: ("v", "in"), 4: ("v", "out")}[bit]
                ta = reverse_edges(assign, i, axis, direction)
                tc = [c for c in contacts if i not in (c[0], c[1])]
                c2, lay2, blk2 = score(ta, tc)
                if c2 is not None and c2 < best_cost - 1e-9:
                    assign, contacts = ta, tc
                    best_cost, best_lay, blocked = c2, lay2, blk2
                    edits.append(("reverse", int(i), int(bit)))
                    progressed = True
                    break
        if not progressed:
            break

    # ---- C) grouping bridges -------------------------------------------
    m = G.evaluate(ev, case, best_lay)
    if m.grouping_violations > 0:
        _, comps_now = G.cluster_contacts(best_lay, case.clust)
        for g, cs in comps_now.items():
            if len(cs) < 2:
                continue
            for cand_list in bridge_candidates(best_lay, cs):
                for (_d, i, j, gx, gy) in cand_list:
                    xi, yi, wi, hi_ = best_lay[i]
                    xj, yj, wj, hj = best_lay[j]
                    oy = min(yi + hi_, yj + hj) - max(yi, yj)
                    ox = min(xi + wi, xj + wj) - max(xi, xj)
                    if oy > 1e-6:
                        axes = ["x"]
                    elif ox > 1e-6:
                        axes = ["y"]
                    else:
                        axes = ["x", "y"] if gx <= gy else ["y", "x"]
                    hit = False
                    for ax in axes:
                        if ax == "x":
                            lo, hi = (i, j) if xi <= xj else (j, i)
                            delta = min(1.0, 0.5 * min(hi_, hj))
                        else:
                            lo, hi = (i, j) if yi <= yj else (j, i)
                            delta = min(1.0, 0.5 * min(wi, wj))
                        ta = dict(assign)
                        ta[(min(i, j), max(i, j))] = ("h" if ax == "x" else "v", lo, hi)
                        tc = list(contacts) + [(lo, hi, ax, delta)]
                        c2, lay2, blk2 = score(ta, tc)
                        if c2 is not None and c2 < best_cost - 1e-9:
                            assign, contacts = ta, tc
                            best_cost, best_lay, blocked = c2, lay2, blk2
                            edits.append(("bridge", int(lo), int(hi), ax))
                            hit = True
                            break
                    if hit:
                        break

    status = G.boundary_status(rects, case.bound)
    return best_lay, best_cost, {
        "stageA_cost": stageA,
        "edits": edits,
        "blocked": sorted(int(i) for i in (blocked or [])),
        "n_viol_preplaced": len([i for i, (c, mm) in status.items()
                                 if mm and i in pre]),
    }


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../"
    ev = G.load_evaluator()
    cases = G.load_cases(ev, data_path)
    rows, layouts = [], {}
    t0 = time.time()
    for case in cases:
        gold_m = G.evaluate(ev, case, case.rects)
        try:
            lay, cost, info = process_case(ev, case, gold_m)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            lay, cost, info = list(case.rects), gold_m.cost_no_runtime, {
                "error": repr(e), "edits": [], "stageA_cost": None,
                "blocked": [], "n_viol_preplaced": 0}
        if cost >= gold_m.cost_no_runtime - 1e-12:
            lay, cost = list(case.rects), gold_m.cost_no_runtime
        m = G.evaluate(ev, case, lay)
        layouts[case.test_id] = lay
        rows.append({
            "test_id": case.test_id, "n": case.n,
            "gold_cost": float(gold_m.cost_no_runtime),
            "gold_vbnd": int(gold_m.boundary_violations),
            "gold_vgrp": int(gold_m.grouping_violations),
            "n_soft": int(gold_m.max_possible_violations),
            "final_cost": float(m.cost_no_runtime),
            "feasible": bool(m.is_feasible),
            "overlap": int(m.overlap_violations),
            "area_viol": int(m.area_violations),
            "dim_viol": int(m.dimension_violations),
            "vbnd": int(m.boundary_violations), "vgrp": int(m.grouping_violations),
            "vmib": int(m.mib_violations), "vrel": float(m.violations_relative),
            "hpwl_gap": float(m.hpwl_gap), "area_gap": float(m.area_gap),
            **info,
        })
        sa = info.get("stageA_cost")
        print(f"[{case.test_id:3d}] n={case.n:3d} gold={gold_m.cost_no_runtime:.4f} "
              f"A={sa if sa else float('nan'):.4f} final={m.cost_no_runtime:.4f} "
              f"vbnd {gold_m.boundary_violations}->{m.boundary_violations} "
              f"vgrp {gold_m.grouping_violations}->{m.grouping_violations} "
              f"ed={len(info['edits'])} hgap={m.hpwl_gap:+.4f} "
              f"agap={m.area_gap:+.4f} feas={m.is_feasible}", flush=True)

    ns = [r["n"] for r in rows]

    def band(pred, key):
        sub = [(r[key], r["n"]) for r in rows if pred(r["n"])]
        return ev.compute_total_score([c for c, _ in sub], [n for _, n in sub])

    out = {
        "elapsed_s": time.time() - t0,
        "total_golden": ev.compute_total_score([r["gold_cost"] for r in rows], ns),
        "total_repaired": ev.compute_total_score([r["final_cost"] for r in rows], ns),
        "band_lt100_golden": band(lambda n: n < 100, "gold_cost"),
        "band_ge100_golden": band(lambda n: n >= 100, "gold_cost"),
        "band_lt100_repaired": band(lambda n: n < 100, "final_cost"),
        "band_ge100_repaired": band(lambda n: n >= 100, "final_cost"),
        "feasible_count": sum(1 for r in rows if r["feasible"]),
        "overlap_total": sum(r["overlap"] for r in rows),
        "area_viol_total": sum(r["area_viol"] for r in rows),
        "dim_viol_total": sum(r["dim_viol"] for r in rows),
        "gold_vbnd_total": sum(r["gold_vbnd"] for r in rows),
        "gold_vgrp_total": sum(r["gold_vgrp"] for r in rows),
        "final_vbnd_total": sum(r["vbnd"] for r in rows),
        "final_vgrp_total": sum(r["vgrp"] for r in rows),
        "final_vmib_total": sum(r["vmib"] for r in rows),
        "cases_at_1.0": sum(1 for r in rows if r["final_cost"] < 1 + 1e-9),
        "sum_positive_hpwl_gap": sum(max(0.0, r["hpwl_gap"]) for r in rows),
        "sum_positive_area_gap": sum(max(0.0, r["area_gap"]) for r in rows),
        "n_edits": sum(len(r["edits"]) for r in rows),
        "rows": rows,
    }
    (OUT / "gr_repair3.json").write_text(json.dumps(out, indent=2))
    (OUT / "gr_layouts3.json").write_text(json.dumps(
        {str(k): v for k, v in layouts.items()}))
    for k, v in out.items():
        if k != "rows":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
