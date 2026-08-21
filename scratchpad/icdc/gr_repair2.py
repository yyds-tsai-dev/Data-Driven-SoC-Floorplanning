"""Golden repair floor probe -- batch 2.

Batch 1 froze golden's relative order.  Batch 2 additionally allows two
*local, provably acyclic* topology edits, each accepted only if the exact
evaluator cost drops:

  A) wall-reversal: to bring a boundary-coded block b to a wall, reverse all
     of b's in-edges (or out-edges) on that axis.  b then has in-degree 0 so
     the constraint graph stays a DAG by construction.
  B) grouping bridge: force a contact between two disconnected components of
     the same cluster (abutment equality on one axis + a shared edge of
     positive length on the other).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_lib as G  # noqa: E402

OUT = Path(__file__).resolve().parent
TOL = 1e-7


def build_assign(rects, tol=1e-9):
    """dict (i,j) i<j -> (axis, lo, hi) with constraint c_lo+s_lo <= c_hi."""
    n = len(rects)
    assign = {}
    for i in range(n):
        xi, yi, wi, hi_ = rects[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = rects[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi_, yj + hj) - max(yi, yj)
            if oy > tol:
                ax = "h"
            elif ox > tol:
                ax = "v"
            else:
                gx = max(xj - (xi + wi), xi - (xj + wj))
                gy = max(yj - (yi + hi_), yi - (yj + hj))
                ax = "h" if gx >= gy else "v"
            if ax == "h":
                lo, hi = (i, j) if xi + wi <= xj + tol else (j, i)
            else:
                lo, hi = (i, j) if yi + hi_ <= yj + tol else (j, i)
            assign[(i, j)] = (ax, lo, hi)
    return assign


def split(assign):
    hor = [(lo, hi) for (ax, lo, hi) in assign.values() if ax == "h"]
    ver = [(lo, hi) for (ax, lo, hi) in assign.values() if ax == "v"]
    return hor, ver


def feasible_dc(n, edges, size, lo, hi, eq):
    lo2 = np.array(lo, float)
    hi2 = np.array(hi, float)
    for i, v in eq.items():
        lo2[i] = max(lo2[i], v)
        hi2[i] = min(hi2[i], v)
        if lo2[i] > hi2[i] + TOL:
            return False
    try:
        mn, mx = G.longest_path_bounds(n, edges, size, lo2, hi2)
    except RuntimeError:
        return False
    return bool(np.all(mn <= mx + 1e-7))


def run_pipeline(case, assign, contacts):
    """Selection + HPWL LP under a given topology and contact set."""
    n = case.n
    rects = case.rects
    x0, y0, x1, y1 = case.frame
    hor_full, ver_full = split(assign)
    hor = G.transitive_reduce(n, hor_full)
    ver = G.transitive_reduce(n, ver_full)

    cx = np.array([r[0] for r in rects]); sw = np.array([r[2] for r in rects])
    cy = np.array([r[1] for r in rects]); sh = np.array([r[3] for r in rects])

    pre = set(np.nonzero(case.preplaced)[0].tolist())
    lox = np.full(n, x0); hix = x1 - sw
    loy = np.full(n, y0); hiy = y1 - sh
    hardx = x1 * np.ones(n); hardy = y1 * np.ones(n)
    base_eqx = {i: float(cx[i]) for i in pre}
    base_eqy = {i: float(cy[i]) for i in pre}

    status = G.boundary_status(rects, case.bound)
    reqx, reqy, unsat = {}, {}, set()
    for i, (code, miss) in status.items():
        if code & 1 and code & 2:
            (reqx.__setitem__(i, x0) if abs(sw[i] - (x1 - x0)) < 1e-6
             else unsat.add(i))
        elif code & 1:
            reqx[i] = x0
        elif code & 2:
            reqx[i] = x1 - sw[i]
        if code & 8 and code & 4:
            (reqy.__setitem__(i, y0) if abs(sh[i] - (y1 - y0)) < 1e-6
             else unsat.add(i))
        elif code & 8:
            reqy[i] = y0
        elif code & 4:
            reqy[i] = y1 - sh[i]
    cand = [i for i in status if i not in pre and i not in unsat]

    rowsx = G.contact_rows(contacts, rects, "x")
    rowsy = G.contact_rows(contacts, rects, "y")

    def select(req, edges, size, lo, hi, base_eq, coord, rows, hard):
        def feas(eq):
            if not feasible_dc(n, edges, size, lo, hi, eq):
                return False
            if not rows:
                return True
            sol, _ = G.solve_axis(n, size, lo, hard, edges, [], [], eq,
                                  extra_le=rows)
            return sol is not None

        wanted = [i for i in cand if i in req]
        sat = [i for i in wanted if abs(coord[i] - req[i]) < 1e-6]
        bad = [i for i in wanted if abs(coord[i] - req[i]) >= 1e-6]
        eq = dict(base_eq)
        for i in sat:
            eq[i] = req[i]
        if not feas(eq):
            eq, sat = dict(base_eq), []
        trial = dict(eq)
        for i in bad:
            trial[i] = req[i]
        if feas(trial):
            return trial, set(sat) | set(bad), set()
        acc, blk = set(sat), set()
        bad.sort(key=lambda i: abs(coord[i] - req[i]))
        for i in bad:
            t = dict(eq)
            t[i] = req[i]
            if feas(t):
                eq, _ = t, acc.add(i)
            else:
                blk.add(i)
        return eq, acc, blk

    eqx, accx, blkx = select(reqx, hor, sw, lox, hix, base_eqx, cx, rowsx, hardx)
    eqy, accy, blky = select(reqy, ver, sh, loy, hiy, base_eqy, cy, rowsy, hardy)
    for _ in range(3):
        drop = {i for i in cand if i in reqx and i in reqy
                and not (i in accx and i in accy)}
        if not drop:
            break
        for i in drop:
            accx.discard(i); accy.discard(i)
            if i not in pre:
                eqx.pop(i, None); eqy.pop(i, None)

    netsx, pinx = G.collect_nets(case, "x")
    netsy, piny = G.collect_nets(case, "y")
    solx, _ = G.solve_axis(n, sw, lox, hardx, hor, netsx, pinx, eqx, extra_le=rowsx)
    soly, _ = G.solve_axis(n, sh, loy, hardy, ver, netsy, piny, eqy, extra_le=rowsy)
    if solx is None or soly is None:
        return None, blkx, blky
    new = [(float(solx[i]), float(soly[i]), float(sw[i]), float(sh[i]))
           for i in range(n)]
    # snap contact abutments so shapely sees an exact shared edge
    for (i, j, ax, _d) in contacts:
        if ax == "x":
            new[j] = (new[i][0] + new[i][2], new[j][1], new[j][2], new[j][3])
        else:
            new[j] = (new[j][0], new[i][1] + new[i][3], new[j][2], new[j][3])
    return new, blkx, blky


def reverse_edges(assign, b, axis, direction):
    """Reverse every in-/out-edge of b on `axis`.  b ends with in-degree 0
    (or out-degree 0), so the graph provably stays acyclic."""
    new = dict(assign)
    for key, (ax, lo, hi) in assign.items():
        if ax != axis or b not in key:
            continue
        if direction == "in" and hi == b:
            new[key] = (ax, b, lo)
        elif direction == "out" and lo == b:
            new[key] = (ax, hi, b)
    return new


def bridge_candidates(rects, comps_for_cluster, k=3):
    """Best cross-component pairs to force into contact."""
    out = []
    merged = list(comps_for_cluster[0])
    for c in comps_for_cluster[1:]:
        best = []
        for i in merged:
            xi, yi, wi, hi_ = rects[i]
            for j in c:
                xj, yj, wj, hj = rects[j]
                gx = max(0.0, max(xj - (xi + wi), xi - (xj + wj)))
                gy = max(0.0, max(yj - (yi + hi_), yi - (yj + hj)))
                best.append((gx + gy, i, j, gx, gy))
        best.sort()
        out.append(best[:k])
        merged = merged + list(c)
    return out


def process_case(ev, case, gold_m):
    n = case.n
    rects = case.rects
    assign = build_assign(rects)
    contacts, comps = G.cluster_contacts(rects, case.clust)
    status = G.boundary_status(rects, case.bound)
    pre = set(np.nonzero(case.preplaced)[0].tolist())

    def score(a, c):
        lay, bx, by = run_pipeline(case, a, c)
        if lay is None:
            return None, None, None
        m = G.evaluate(ev, case, lay)
        if not m.is_feasible:
            return None, None, None
        return m.cost_no_runtime, lay, (bx, by)

    best_cost, best_lay, blocked = score(assign, contacts)
    if best_lay is None:
        best_cost, best_lay = gold_m.cost_no_runtime, list(rects)
        blocked = (set(), set())
    stageA = best_cost
    edits = []

    # ---- A) wall reversal for boundary blocks still violating -----------
    for _round in range(2):
        m = G.evaluate(ev, case, best_lay)
        cur = G.boundary_status(best_lay, case.bound)
        todo = [(i, miss) for i, (code, miss) in cur.items()
                if miss and i not in pre]
        progressed = False
        for i, miss in todo:
            for bit in miss:
                axis, direction = {1: ("h", "in"), 2: ("h", "out"),
                                   8: ("v", "in"), 4: ("v", "out")}[bit]
                trial_assign = reverse_edges(assign, i, axis, direction)
                trial_contacts = [c for c in contacts if i not in (c[0], c[1])]
                c2, lay2, blk2 = score(trial_assign, trial_contacts)
                if c2 is not None and c2 < best_cost - 1e-9:
                    assign, contacts = trial_assign, trial_contacts
                    best_cost, best_lay, blocked = c2, lay2, blk2
                    edits.append(("reverse", int(i), bit))
                    progressed = True
                    break
        if not progressed:
            break

    # ---- B) grouping bridges -------------------------------------------
    m = G.evaluate(ev, case, best_lay)
    if m.grouping_violations > 0:
        _, comps_now = G.cluster_contacts(best_lay, case.clust)
        for g, cs in comps_now.items():
            if len(cs) < 2:
                continue
            for cand_list in bridge_candidates(best_lay, cs):
                done = False
                for (_d, i, j, gx, gy) in cand_list:
                    if done:
                        break
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
                    for ax in axes:
                        if ax == "x":
                            lo, hi = (i, j) if xi <= xj else (j, i)
                            delta = min(1.0, 0.5 * min(hi_, hj))
                        else:
                            lo, hi = (i, j) if yi <= yj else (j, i)
                            delta = min(1.0, 0.5 * min(wi, wj))
                        key = (min(i, j), max(i, j))
                        ta = dict(assign)
                        ta[key] = ("h" if ax == "x" else "v", lo, hi)
                        tc = list(contacts) + [(lo, hi, ax, delta)]
                        c2, lay2, blk2 = score(ta, tc)
                        if c2 is not None and c2 < best_cost - 1e-9:
                            assign, contacts = ta, tc
                            best_cost, best_lay, blocked = c2, lay2, blk2
                            edits.append(("bridge", int(lo), int(hi), ax))
                            done = True
                            break

    return best_lay, best_cost, {
        "stageA_cost": stageA,
        "edits": edits,
        "blocked_x": sorted(int(i) for i in (blocked[0] if blocked else [])),
        "blocked_y": sorted(int(i) for i in (blocked[1] if blocked else [])),
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
            lay, cost, info = list(case.rects), gold_m.cost_no_runtime, {
                "error": repr(e), "edits": [], "stageA_cost": None,
                "blocked_x": [], "blocked_y": [], "n_viol_preplaced": 0}
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
            "vbnd": int(m.boundary_violations),
            "vgrp": int(m.grouping_violations),
            "vmib": int(m.mib_violations),
            "vrel": float(m.violations_relative),
            "hpwl_gap": float(m.hpwl_gap),
            "area_gap": float(m.area_gap),
            **info,
        })
        print(f"[{case.test_id:3d}] n={case.n:3d} gold={gold_m.cost_no_runtime:.4f} "
              f"A={info['stageA_cost'] if info['stageA_cost'] else float('nan'):.4f} "
              f"final={m.cost_no_runtime:.4f} vbnd {gold_m.boundary_violations}->"
              f"{m.boundary_violations} vgrp {gold_m.grouping_violations}->"
              f"{m.grouping_violations} edits={len(info['edits'])} "
              f"hgap={m.hpwl_gap:+.4f} feas={m.is_feasible}", flush=True)

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
        "viol_preplaced_total": sum(r["n_viol_preplaced"] for r in rows),
        "cases_at_1.0": sum(1 for r in rows if r["final_cost"] < 1 + 1e-9),
        "sum_positive_hpwl_gap": sum(max(0.0, r["hpwl_gap"]) for r in rows),
        "n_edits": sum(len(r["edits"]) for r in rows),
        "rows": rows,
    }
    (OUT / "gr_repair2.json").write_text(json.dumps(out, indent=2))
    (OUT / "gr_layouts2.json").write_text(json.dumps(
        {str(k): v for k, v in layouts.items()}))
    for k, v in out.items():
        if k != "rows":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
