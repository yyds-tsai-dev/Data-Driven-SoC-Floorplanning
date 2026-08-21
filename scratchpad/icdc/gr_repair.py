"""Golden repair floor probe -- survey + topology-fixed boundary repair.

Batch 1 deliverable: how far does the golden no-runtime score fall once the
soft (boundary / grouping) violations are repaired under
  * frozen shapes, frozen preplaced positions, preserved relative order
  * HPWL-minimising placement inside the golden bounding frame
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


def axis_data(case, axis):
    r = case.rects
    if axis == "x":
        c = np.array([b[0] for b in r])
        s = np.array([b[2] for b in r])
    else:
        c = np.array([b[1] for b in r])
        s = np.array([b[3] for b in r])
    return c, s


def feasible_with(n, edges, size, lo, hi, eq):
    lo2 = np.array(lo, float)
    hi2 = np.array(hi, float)
    for i, v in eq.items():
        lo2[i] = max(lo2[i], v)
        hi2[i] = min(hi2[i], v)
        if lo2[i] > hi2[i] + TOL:
            return False, None, None
    mn, mx = G.longest_path_bounds(n, edges, size, lo2, hi2)
    ok = bool(np.all(mn <= mx + 1e-7))
    return ok, mn, mx


def process_case(ev, case, verbose=False):
    n = case.n
    rects = case.rects
    x0, y0, x1, y1 = case.frame
    hor, ver, gold_overlaps = G.build_topology(rects)
    hor = G.transitive_reduce(n, hor)
    ver = G.transitive_reduce(n, ver)

    cx, sw = axis_data(case, "x")
    cy, sh = axis_data(case, "y")

    pre = set(np.nonzero(case.preplaced)[0].tolist())
    fixed = set(np.nonzero(case.fixed)[0].tolist())

    lox = np.full(n, x0)
    hix = np.array([x1 - sw[i] for i in range(n)])
    loy = np.full(n, y0)
    hiy = np.array([y1 - sh[i] for i in range(n)])

    base_eqx = {i: float(cx[i]) for i in pre}
    base_eqy = {i: float(cy[i]) for i in pre}

    status = G.boundary_status(rects, case.bound)

    # ---- classify each boundary-coded block -----------------------------
    reqx, reqy, unsat_shape = {}, {}, set()
    for i, (code, miss) in status.items():
        if code & 1 and code & 2:
            if abs(sw[i] - (x1 - x0)) < 1e-6:
                reqx[i] = x0
            else:
                unsat_shape.add(i)
        elif code & 1:
            reqx[i] = x0
        elif code & 2:
            reqx[i] = x1 - sw[i]
        if code & 8 and code & 4:
            if abs(sh[i] - (y1 - y0)) < 1e-6:
                reqy[i] = y0
            else:
                unsat_shape.add(i)
        elif code & 8:
            reqy[i] = y0
        elif code & 4:
            reqy[i] = y1 - sh[i]

    viol = {i for i, (c, m) in status.items() if m}
    # blocks we may attempt to enforce
    cand = [i for i in status if i not in pre and i not in unsat_shape]
    pre_viol = sorted(i for i in viol if i in pre)

    # ---- grouping-contact preservation ---------------------------------
    contacts, comps = G.cluster_contacts(rects, case.clust)
    rowsx = G.contact_rows(contacts, rects, "x")
    rowsy = G.contact_rows(contacts, rects, "y")

    # ---- per-axis selection --------------------------------------------
    # Cheap exact difference-constraint test first; whenever grouping rows
    # are present we confirm with a zero-objective LP (the DC test ignores
    # the two-sided contact rows).
    def select(req, edges, size, lo, hi, base_eq, coord, rows, hard_hi):
        def feas(eq):
            ok, _, _ = feasible_with(n, edges, size, lo, hi, eq)
            if not ok:
                return False
            if not rows:
                return True
            sol, _ = G.solve_axis(n, size, lo, hard_hi, edges, [], [], eq,
                                  extra_le=rows)
            return sol is not None

        wanted = [i for i in cand if i in req]
        sat = [i for i in wanted if abs(coord[i] - req[i]) < 1e-6]
        bad = [i for i in wanted if abs(coord[i] - req[i]) >= 1e-6]
        eq = dict(base_eq)
        for i in sat:
            eq[i] = req[i]
        if not feas(eq):          # golden is a witness; should not trigger
            eq = dict(base_eq)
            sat = []
        trial = dict(eq)
        for i in bad:
            trial[i] = req[i]
        if feas(trial):
            return trial, set(sat) | set(bad), set()
        accepted, blocked = set(sat), set()
        bad.sort(key=lambda i: abs(coord[i] - req[i]))
        for i in bad:
            trial = dict(eq)
            trial[i] = req[i]
            if feas(trial):
                eq = trial
                accepted.add(i)
            else:
                blocked.add(i)
        return eq, accepted, blocked

    hardx = x1 * np.ones(n)
    hardy = y1 * np.ones(n)
    eqx, accx, blkx = select(reqx, hor, sw, lox, hix, base_eqx, cx, rowsx, hardx)
    eqy, accy, blky = select(reqy, ver, sh, loy, hiy, base_eqy, cy, rowsy, hardy)

    # ---- drop corner blocks that only got one axis ----------------------
    for _ in range(3):
        drop = set()
        for i in cand:
            nx, ny = i in reqx, i in reqy
            if nx and ny:
                if not (i in accx and i in accy):
                    drop.add(i)
        if not drop:
            break
        for i in drop:
            accx.discard(i)
            accy.discard(i)
            eqx.pop(i, None) if i not in pre else None
            eqy.pop(i, None) if i not in pre else None

    # ---- solve the two HPWL LPs ----------------------------------------
    netsx, pinx = G.collect_nets(case, "x")
    netsy, piny = G.collect_nets(case, "y")
    solx, objx = G.solve_axis(n, sw, lox, hardx, hor, netsx, pinx, eqx,
                              extra_le=rowsx)
    soly, objy = G.solve_axis(n, sh, loy, hardy, ver, netsy, piny, eqy,
                              extra_le=rowsy)

    info = {
        "test_id": case.test_id,
        "n": n,
        "n_boundary_blocks": len(status),
        "n_boundary_viol": len(viol),
        "n_viol_preplaced": len(pre_viol),
        "n_viol_unsat_shape": len([i for i in viol if i in unsat_shape]),
        "n_pre": len(pre),
        "n_fixed": len(fixed),
        "blocked_x": sorted(int(i) for i in blkx),
        "blocked_y": sorted(int(i) for i in blky),
        "golden_overlaps": gold_overlaps,
        "n_contacts": len(contacts),
        "n_clusters": len(comps),
        "gold_grp_components": sum(len(v) - 1 for v in comps.values()),
    }

    if solx is None or soly is None:
        info["lp_failed"] = True
        return list(rects), info

    new = [(float(solx[i]), float(soly[i]), float(sw[i]), float(sh[i]))
           for i in range(n)]
    return new, info


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../"
    ev = G.load_evaluator()
    cases = G.load_cases(ev, data_path)
    rows = []
    layouts = {}
    t0 = time.time()
    for case in cases:
        gold_m = G.evaluate(ev, case, case.rects)
        try:
            new, info = process_case(ev, case)
        except Exception as e:  # noqa: BLE001
            new, info = list(case.rects), {"test_id": case.test_id,
                                           "n": case.n, "error": repr(e)}
        new_m = G.evaluate(ev, case, new)
        accepted = bool(new_m.is_feasible
                        and new_m.cost_no_runtime < gold_m.cost_no_runtime - 1e-12)
        final = new if accepted else list(case.rects)
        fin_m = new_m if accepted else gold_m
        layouts[case.test_id] = final
        rows.append({
            **info,
            "gold_cost": float(gold_m.cost_no_runtime),
            "gold_vrel": float(gold_m.violations_relative),
            "gold_vbnd": int(gold_m.boundary_violations),
            "gold_vgrp": int(gold_m.grouping_violations),
            "n_soft": int(gold_m.max_possible_violations),
            "lp_cost": float(new_m.cost_no_runtime),
            "lp_feasible": bool(new_m.is_feasible),
            "lp_overlap": int(new_m.overlap_violations),
            "lp_area_viol": int(new_m.area_violations),
            "lp_dim_viol": int(new_m.dimension_violations),
            "lp_vbnd": int(new_m.boundary_violations),
            "lp_vgrp": int(new_m.grouping_violations),
            "lp_vmib": int(new_m.mib_violations),
            "lp_vrel": float(new_m.violations_relative),
            "lp_hpwl_gap": float(new_m.hpwl_gap),
            "lp_area_gap": float(new_m.area_gap),
            "accepted": accepted,
            "final_cost": float(fin_m.cost_no_runtime),
            "final_vbnd": int(fin_m.boundary_violations),
            "final_vgrp": int(fin_m.grouping_violations),
            "final_vrel": float(fin_m.violations_relative),
            "final_hpwl_gap": float(fin_m.hpwl_gap),
            "final_feasible": bool(fin_m.is_feasible),
        })
        print(f"[{case.test_id:3d}] n={case.n:3d} gold={gold_m.cost_no_runtime:.4f} "
              f"lp={new_m.cost_no_runtime:.4f} acc={accepted} "
              f"vbnd {gold_m.boundary_violations}->{new_m.boundary_violations} "
              f"vgrp {gold_m.grouping_violations}->{new_m.grouping_violations} "
              f"hgap={new_m.hpwl_gap:+.4f} feas={new_m.is_feasible}", flush=True)

    ns = [r["n"] for r in rows]
    total_gold = ev.compute_total_score([r["gold_cost"] for r in rows], ns)
    total_lp = ev.compute_total_score([r["lp_cost"] for r in rows], ns)
    total_fin = ev.compute_total_score([r["final_cost"] for r in rows], ns)

    def band(pred, key):
        sub = [(r[key], r["n"]) for r in rows if pred(r["n"])]
        if not sub:
            return None
        return ev.compute_total_score([c for c, _ in sub], [n for _, n in sub])

    out = {
        "elapsed_s": time.time() - t0,
        "total_golden": total_gold,
        "total_lp_all": total_lp,
        "total_final_accept_only": total_fin,
        "band_lt100_golden": band(lambda n: n < 100, "gold_cost"),
        "band_ge100_golden": band(lambda n: n >= 100, "gold_cost"),
        "band_lt100_final": band(lambda n: n < 100, "final_cost"),
        "band_ge100_final": band(lambda n: n >= 100, "final_cost"),
        "n_accepted": sum(1 for r in rows if r["accepted"]),
        "n_lp_infeasible": sum(1 for r in rows if not r["lp_feasible"]),
        "gold_vbnd_total": sum(r["gold_vbnd"] for r in rows),
        "gold_vgrp_total": sum(r["gold_vgrp"] for r in rows),
        "final_vbnd_total": sum(r["final_vbnd"] for r in rows),
        "final_vgrp_total": sum(r["final_vgrp"] for r in rows),
        "viol_preplaced_total": sum(r.get("n_viol_preplaced", 0) for r in rows),
        "blocked_total": sum(len(r.get("blocked_x", [])) + len(r.get("blocked_y", []))
                             for r in rows),
        "final_feasible_count": sum(1 for r in rows if r["final_feasible"]),
        "rows": rows,
    }
    (OUT / "gr_repair.json").write_text(json.dumps(out, indent=2))
    (OUT / "gr_layouts.json").write_text(json.dumps(
        {str(k): v for k, v in layouts.items()}))
    for k, v in out.items():
        if k != "rows":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
