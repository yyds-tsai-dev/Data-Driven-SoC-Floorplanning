"""R1 stage-A/B/C repair applied to OUR Production Solver Path output.

Reads the layouts the official evaluator recorded for the partner solver
(`positions` in the eval JSON), runs the same topology-fixed repair used on
the golden layouts, and re-scores with the official evaluator.

Difference vs the golden probe: our layouts already carry POSITIVE hpwl_gap
and area_gap, so there is no free-movement allowance and the bbox extent is
now itself a scoring lever.  The LP objective is therefore expressed directly
in score units,

    0.5 * hpwl/hpwl_baseline  +  0.5 * (W*H)/area_baseline

which is linear per axis once the other axis' extent is held fixed, so we run
an alternating x/y descent (each step is an exact LP, so the true quality
factor decreases monotonically).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_lib as G  # noqa: E402
from gr_repair2 import split, reverse_edges, bridge_candidates  # noqa: E402

OUT = Path(__file__).resolve().parent
HEADROOM = 0.20
ROUNDS = 3


def extent(rects, axis):
    if axis == "x":
        return (min(r[0] for r in rects), max(r[0] + r[2] for r in rects))
    return (min(r[1] for r in rects), max(r[1] + r[3] for r in rects))


def run_pipeline(case, base, assign, contacts, enforce=None):
    """base = the layout being repaired (our solver's output)."""
    n = case.n
    x0, x1 = extent(base, "x")
    y0, y1 = extent(base, "y")
    W, H = x1 - x0, y1 - y0
    hor = G.transitive_reduce(n, split(assign)[0])
    ver = G.transitive_reduce(n, split(assign)[1])

    sw = np.array([r[2] for r in base])
    sh = np.array([r[3] for r in base])
    pre = set(np.nonzero(case.preplaced)[0].tolist())
    eqx = {i: float(base[i][0]) for i in pre}
    eqy = {i: float(base[i][1]) for i in pre}
    lox = np.full(n, x0 - HEADROOM * W); hix = np.full(n, x1 + HEADROOM * W)
    loy = np.full(n, y0 - HEADROOM * H); hiy = np.full(n, y1 + HEADROOM * H)
    for i in pre:                      # preplaced must stay reachable
        lox[i] = min(lox[i], base[i][0]); hix[i] = max(hix[i], base[i][0] + sw[i])
        loy[i] = min(loy[i], base[i][1]); hiy[i] = max(hiy[i], base[i][1] + sh[i])

    status = G.boundary_status(base, case.bound)
    want = {i: {b for b in (1, 2, 4, 8) if code & b}
            for i, (code, miss) in status.items()}
    rowsx = G.contact_rows(contacts, base, "x")
    rowsy = G.contact_rows(contacts, base, "y")
    netsx, pinx = G.collect_nets(case, "x")
    netsy, piny = G.collect_nets(case, "y")

    hb = max(case.baseline["hpwl_baseline"], 1e-6)
    ab = max(case.baseline["area_baseline"], 1e-6)
    wh = 0.5 / hb

    def solve(sel, objective=True, rounds=ROUNDS):
        minx = [i for i in sel if 1 in want[i]]; maxx = [i for i in sel if 2 in want[i]]
        miny = [i for i in sel if 8 in want[i]]; maxy = [i for i in sel if 4 in want[i]]
        cur_h, cur_w = H, W
        sx = sy = None
        for _ in range(rounds if objective else 1):
            fx = 0.5 * cur_h / ab if objective else 0.0
            nx = G.solve_axis_ext(n, sw, lox, hix, hor, netsx, pinx, eqx,
                                  minx, maxx, rowsx, objective,
                                  hpwl_weight=wh, frame_weight=fx)
            if nx is None:
                return None
            sx = nx
            cur_w = max(sx + sw) - min(sx)
            fy = 0.5 * cur_w / ab if objective else 0.0
            ny = G.solve_axis_ext(n, sh, loy, hiy, ver, netsy, piny, eqy,
                                  miny, maxy, rowsy, objective,
                                  hpwl_weight=wh, frame_weight=fy)
            if ny is None:
                return None
            sy = ny
            cur_h = max(sy + sh) - min(sy)
        return [(float(sx[i]), float(sy[i]), float(sw[i]), float(sh[i]))
                for i in range(n)]

    all_b = sorted(want)
    blocked = set()
    if enforce is None:
        sel = list(all_b)
        if solve(sel, objective=False) is None:
            sat = [i for i in all_b if not status[i][1]]
            base_sel = []
            for i in sat:
                if solve(base_sel + [i], objective=False) is not None:
                    base_sel.append(i)
            sel = base_sel
            for i in sorted((i for i in all_b if i not in sel),
                            key=lambda i: len(want[i])):
                if solve(sel + [i], objective=False) is not None:
                    sel.append(i)
                else:
                    blocked.add(i)
    else:
        sel = list(enforce)
    lay = solve(sel)
    if lay is None:
        return None, blocked
    for (i, j, ax, _d) in contacts:
        if ax == "x":
            lay[j] = (lay[i][0] + lay[i][2], lay[j][1], lay[j][2], lay[j][3])
        else:
            lay[j] = (lay[j][0], lay[i][1] + lay[i][3], lay[j][2], lay[j][3])
    return lay, blocked


def process_case(ev, case, base, base_m):
    assign = G.build_assign_robust(base)
    contacts, _ = G.cluster_contacts(base, case.clust)
    pre = set(np.nonzero(case.preplaced)[0].tolist())

    def score(a, c):
        lay, blk = run_pipeline(case, base, a, c)
        if lay is None:
            return None, None
        m = G.evaluate(ev, case, lay)
        if not m.is_feasible:
            return None, None
        return m.cost_no_runtime, lay

    t0 = time.perf_counter()
    best_cost, best_lay = score(assign, contacts)
    if best_lay is None:
        best_cost, best_lay = base_m.cost_no_runtime, list(base)
    stageA, tA = best_cost, time.perf_counter() - t0
    edits = []

    for _round in range(2):                       # B) wall reversal
        cur = G.boundary_status(best_lay, case.bound)
        todo = [(i, miss) for i, (cd, miss) in cur.items() if miss and i not in pre]
        progressed = False
        for i, miss in todo:
            for bit in miss:
                axis, direction = {1: ("h", "in"), 2: ("h", "out"),
                                   8: ("v", "in"), 4: ("v", "out")}[bit]
                ta = reverse_edges(assign, i, axis, direction)
                tc = [c for c in contacts if i not in (c[0], c[1])]
                c2, lay2 = score(ta, tc)
                if c2 is not None and c2 < best_cost - 1e-9:
                    assign, contacts, best_cost, best_lay = ta, tc, c2, lay2
                    edits.append(("reverse", int(i), int(bit)))
                    progressed = True
                    break
        if not progressed:
            break

    m = G.evaluate(ev, case, best_lay)             # C) grouping bridges
    if m.grouping_violations > 0:
        _, comps = G.cluster_contacts(best_lay, case.clust)
        for g, cs in comps.items():
            if len(cs) < 2:
                continue
            for cand_list in bridge_candidates(best_lay, cs):
                for (_d, i, j, gx, gy) in cand_list:
                    xi, yi, wi, hi_ = best_lay[i]
                    xj, yj, wj, hj = best_lay[j]
                    oy = min(yi + hi_, yj + hj) - max(yi, yj)
                    ox = min(xi + wi, xj + wj) - max(xi, xj)
                    axes = (["x"] if oy > 1e-6 else ["y"] if ox > 1e-6
                            else (["x", "y"] if gx <= gy else ["y", "x"]))
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
                        c2, lay2 = score(ta, tc)
                        if c2 is not None and c2 < best_cost - 1e-9:
                            assign, contacts, best_cost, best_lay = ta, tc, c2, lay2
                            edits.append(("bridge", int(lo), int(hi), ax))
                            hit = True
                            break
                    if hit:
                        break
    return best_lay, best_cost, {"stageA_cost": stageA, "t_stageA": tA,
                                 "t_total": time.perf_counter() - t0,
                                 "edits": edits}


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../"
    src = OUT / (sys.argv[2] if len(sys.argv) > 2 else "eval_prod_r1base.json")
    ev = G.load_evaluator()
    cases = G.load_cases(ev, data_path)
    prod = {int(t["test_id"]): t for t in json.load(open(src))["test_results"]}

    rows, layouts = [], {}
    for case in cases:
        rec = prod[case.test_id]
        base = [tuple(p) for p in rec["positions"]]
        base_m = G.evaluate(ev, case, base)
        try:
            lay, cost, info = process_case(ev, case, base, base_m)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            lay, cost = list(base), base_m.cost_no_runtime
            info = {"stageA_cost": None, "t_stageA": 0, "t_total": 0,
                    "edits": [], "error": repr(e)}
        if cost >= base_m.cost_no_runtime - 1e-12:
            lay, cost = list(base), base_m.cost_no_runtime
        m = G.evaluate(ev, case, lay)
        layouts[case.test_id] = lay
        rows.append({
            "test_id": case.test_id, "n": case.n,
            "runtime_seconds": float(rec.get("runtime_seconds") or 0.0),
            "base_cost": float(base_m.cost_no_runtime),
            "base_vbnd": int(base_m.boundary_violations),
            "base_vgrp": int(base_m.grouping_violations),
            "base_vmib": int(base_m.mib_violations),
            "base_vrel": float(base_m.violations_relative),
            "base_hpwl_gap": float(base_m.hpwl_gap),
            "base_area_gap": float(base_m.area_gap),
            "base_feasible": bool(base_m.is_feasible),
            "n_soft": int(base_m.max_possible_violations),
            "final_cost": float(m.cost_no_runtime),
            "feasible": bool(m.is_feasible), "overlap": int(m.overlap_violations),
            "area_viol": int(m.area_violations), "dim_viol": int(m.dimension_violations),
            "vbnd": int(m.boundary_violations), "vgrp": int(m.grouping_violations),
            "vmib": int(m.mib_violations), "vrel": float(m.violations_relative),
            "hpwl_gap": float(m.hpwl_gap), "area_gap": float(m.area_gap),
            **info,
        })
        print(f"[{case.test_id:3d}] n={case.n:3d} base={base_m.cost_no_runtime:.4f} "
              f"final={m.cost_no_runtime:.4f} d={m.cost_no_runtime-base_m.cost_no_runtime:+.4f} "
              f"V {base_m.boundary_violations}/{base_m.grouping_violations}/"
              f"{base_m.mib_violations} -> {m.boundary_violations}/"
              f"{m.grouping_violations}/{m.mib_violations} "
              f"hg {base_m.hpwl_gap:+.3f}->{m.hpwl_gap:+.3f} "
              f"ag {base_m.area_gap:+.3f}->{m.area_gap:+.3f} "
              f"t={info['t_total']:.2f}s feas={m.is_feasible}", flush=True)

    ns = [r["n"] for r in rows]

    def band(pred, key):
        sub = [(r[key], r["n"]) for r in rows if pred(r["n"])]
        return ev.compute_total_score([c for c, _ in sub], [n for _, n in sub])

    ts = sorted(r["t_total"] for r in rows)
    tail = [r for r in rows if r["n"] >= 95]
    tail_only = [r["final_cost"] if r["n"] >= 95 else r["base_cost"] for r in rows]
    out = {
        "total_base": ev.compute_total_score([r["base_cost"] for r in rows], ns),
        "total_repaired": ev.compute_total_score([r["final_cost"] for r in rows], ns),
        "total_tail_only_n_ge_95": ev.compute_total_score(tail_only, ns),
        "band_lt100_base": band(lambda n: n < 100, "base_cost"),
        "band_ge100_base": band(lambda n: n >= 100, "base_cost"),
        "band_lt100_repaired": band(lambda n: n < 100, "final_cost"),
        "band_ge100_repaired": band(lambda n: n >= 100, "final_cost"),
        "base_feasible_count": sum(1 for r in rows if r["base_feasible"]),
        "feasible_count": sum(1 for r in rows if r["feasible"]),
        "overlap_total": sum(r["overlap"] for r in rows),
        "area_viol_total": sum(r["area_viol"] for r in rows),
        "dim_viol_total": sum(r["dim_viol"] for r in rows),
        "base_vbnd": sum(r["base_vbnd"] for r in rows),
        "base_vgrp": sum(r["base_vgrp"] for r in rows),
        "base_vmib": sum(r["base_vmib"] for r in rows),
        "final_vbnd": sum(r["vbnd"] for r in rows),
        "final_vgrp": sum(r["vgrp"] for r in rows),
        "final_vmib": sum(r["vmib"] for r in rows),
        "n_improved": sum(1 for r in rows if r["final_cost"] < r["base_cost"] - 1e-12),
        "mean_hpwl_gap_base": sum(r["base_hpwl_gap"] for r in rows) / len(rows),
        "mean_hpwl_gap_final": sum(r["hpwl_gap"] for r in rows) / len(rows),
        "mean_area_gap_base": sum(r["base_area_gap"] for r in rows) / len(rows),
        "mean_area_gap_final": sum(r["area_gap"] for r in rows) / len(rows),
        "t_median": ts[len(ts) // 2], "t_p90": ts[89], "t_max": ts[-1],
        "t_sum": sum(ts),
        "t_sum_tail_n_ge_95": sum(r["t_total"] for r in tail),
        "n_tail_cases": len(tail),
        "solver_runtime_mean": sum(r["runtime_seconds"] for r in rows) / len(rows),
        "n_edits": sum(len(r["edits"]) for r in rows),
        "rows": rows,
    }
    (OUT / "gr_prod.json").write_text(json.dumps(out, indent=2))
    (OUT / "gr_prod_layouts.json").write_text(json.dumps(
        {str(k): v for k, v in layouts.items()}))
    for k, v in out.items():
        if k != "rows":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
