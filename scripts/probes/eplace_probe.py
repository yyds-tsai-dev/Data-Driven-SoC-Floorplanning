#!/usr/bin/env python3
"""Probe: analytical global-placement (ePlace-style) arm vs the production
column-slicing backbone, on the 100 official validation cases.

Loads the 100-case validation set the same way ``src/icdc_engine/data.py``
(``load_test_cases``) does (backed by ``FloorplanDatasetLiteTest`` from
``FloorSet/iccad2026contest/iccad2026_evaluate.py``, the same class
``scripts/iccad2026_evaluate.py`` uses).  For each case, runs
``partner.eplace_arm.generate_candidates`` and separately runs the
production ``src/solver/contest_optimizer.py`` (``MyOptimizer``) under the
canonical G1 solver environment (``src/icdc_engine/g1_runtime.py``
``_SOLVER_ENV``, plus ``DIRECT_OFF=1 PARTNER_FLOW_SLOTS=0`` to keep the
probe CPU/light) as the production-champion baseline.

This is a probe-only script -- it does not modify solve() or any existing
file.  Standalone, new file.

Usage:
    uv run scripts/probes/eplace_probe.py --test-ids 90-99 --iters 300 \\
        --configs 6 --out artifacts/p0_newbox/eplace_probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "partner", REPO / "FloorSet" / "iccad2026contest",
           REPO / "FloorSet", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from icdc_engine.data import load_test_cases  # noqa: E402
import iccad2026_evaluate as ev  # noqa: E402

from eplace_arm import generate_candidates  # noqa: E402

# --legalize mode replicates the production direct-candidate consumption
# sequence (src/solver/contest_optimizer.py `_direct_worker`, around line
# 1694-1717: build a rect list from the raw prediction, wrap it in a fresh
# `_ColumnOptimizer` scorer, then call `layout_refiner.refine_prediction`
# with a deadline) so the ePlace candidates get the SAME legalize+refine
# treatment the model-driven direct arm gets before its output is judged.
from column_sa_legalizer import _ColumnOptimizer  # noqa: E402
from layout_refiner import refine_prediction  # noqa: E402


# ---------------------------------------------------------------------------
def parse_id_ranges(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def hpwl_of(rects: List[tuple], b2b: torch.Tensor, p2b: torch.Tensor,
            pins: torch.Tensor) -> float:
    return float(ev.calculate_hpwl_b2b(rects, b2b) + ev.calculate_hpwl_p2b(rects, p2b, pins))


def official_cost(rects, case) -> Dict[str, Any]:
    """Official cost_no_runtime (+ feasibility/violation breakdown) for a
    rect list, mirroring pseudo_hidden_eval.py's evaluate_case (same
    ``ev.evaluate_solution`` call, baselined on the case's own
    ``hpwl_ref``/``area_ref``)."""
    positions = [tuple(float(v) for v in r) for r in rects]
    metrics = ev.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        {"hpwl_baseline": case["hpwl_ref"], "area_baseline": case["area_ref"]},
        case["cons"], case["b2b"], case["p2b"], case["pins"], case["area"],
        target_positions=case["tp"].tolist(),
        median_runtime=1.0,
    )
    return {
        "cost_no_rt": metrics.cost_no_runtime,
        "feasible": metrics.is_feasible,
        "hpwl_gap": metrics.hpwl_gap,
        "area_gap": metrics.area_gap,
        "v_rel": metrics.violations_relative,
    }


def overlap_frac_of(rects: np.ndarray, a_tot: float) -> float:
    n = rects.shape[0]
    x = rects[:, 0]
    y = rects[:, 1]
    w = rects[:, 2]
    h = rects[:, 3]
    cx = x + w / 2.0
    cy = y + h / 2.0
    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        return 0.0
    dx = np.abs(cx[iu] - cx[ju])
    dy = np.abs(cy[iu] - cy[ju])
    ox = np.maximum(0.0, (w[iu] + w[ju]) / 2.0 - dx)
    oy = np.maximum(0.0, (h[iu] + h[ju]) / 2.0 - dy)
    return float(np.sum(ox * oy)) / max(a_tot, 1e-9)


def run_legalize_compare(candidates, eplace_all, case, tid, column_opt,
                          refine_deadline: float, max_candidates: int,
                          seed: int) -> Dict[str, Any]:
    """Push the best-by-raw-HPWL ePlace candidates through the SAME
    legalize+refine sequence the production direct arm uses, then compare
    official cost_no_runtime against the column-backbone champion.

    Mirrors src/solver/contest_optimizer.py `_direct_worker` (~line 1694-1717):
    for each raw prediction P, build ``rect_list`` from P, wrap it in a
    fresh ``_ColumnOptimizer`` scorer (same ctor args the production path
    passes: area targets, constraints, target_positions, b2b, p2b, pins,
    a deadline, a seed), then call ``layout_refiner.refine_prediction(opt,
    P, deadline, seed=...)``.  ``refine_prediction`` is pure geometry (numpy
    positions in, positions or None out) -- no GPU/model dependency was
    found; it consumes only ``opt`` (built from plain tensors/arrays) and
    the raw prediction array.
    """
    n = case["n"]
    area, cons, tp = case["area"], case["cons"], case["tp"]
    b2b, p2b, pins = case["b2b"], case["p2b"], case["pins"]

    order = sorted(range(len(eplace_all)), key=lambda i: eplace_all[i]["hpwl"])
    order = order[:max_candidates]

    tried = []
    for rank, idx in enumerate(order):
        P = np.asarray(candidates[idx], dtype=np.float64)
        rect_list = [tuple(float(v) for v in P[i]) for i in range(n)]
        deadline = time.time() + refine_deadline
        t0 = time.time()
        try:
            opt = _ColumnOptimizer(rect_list, area, cons, tp, b2b, p2b, pins,
                                    deadline, seed=seed * 1000 + rank)
            out = refine_prediction(opt, P, deadline, seed=seed * 1000 + rank)
        except Exception as exc:  # noqa: BLE001
            print(f"case {tid}: refine_prediction failed on candidate {idx}: {exc}")
            out = None
        dt = time.time() - t0

        if out is None:
            tried.append({"config": eplace_all[idx].get("config"),
                          "legal": False, "refine_time": dt, "reason": "no_output"})
            continue

        out_rects = np.asarray(out, dtype=np.float64)
        a_tot = float(area.sum().item())
        cost = official_cost(out_rects, case)
        hpwl_refined = hpwl_of([tuple(float(v) for v in r) for r in out_rects], b2b, p2b, pins)
        ov = overlap_frac_of(out_rects, a_tot)
        tried.append({
            "config": eplace_all[idx].get("config"),
            "legal": bool(cost["feasible"]),
            "refine_time": dt,
            "cost_no_rt": cost["cost_no_rt"],
            "hpwl": hpwl_refined,
            "overlap_frac": ov,
            "v_rel": cost["v_rel"],
        })

    legal = [t for t in tried if t.get("legal")]
    best = min(legal, key=lambda t: t["cost_no_rt"]) if legal else None

    column_cost = None
    column_hpwl = None
    if column_opt is not None:
        try:
            col_rects = column_opt.solve(n, area, b2b, p2b, pins, cons, tp)
            column_cost = official_cost(col_rects, case)
            column_hpwl = hpwl_of([tuple(float(v) for v in r) for r in col_rects], b2b, p2b, pins)
        except Exception as exc:  # noqa: BLE001
            print(f"case {tid}: column solve (legalize compare) failed: {exc}")
            column_cost = None

    eplace_cost = best["cost_no_rt"] if best is not None else None
    col_cost_v = column_cost["cost_no_rt"] if column_cost is not None else None
    delta = (eplace_cost - col_cost_v) if (eplace_cost is not None and col_cost_v is not None) else None

    return {
        "tried": tried,
        "n_tried": len(tried),
        "n_legal": len(legal),
        "best": best,
        "eplace_cost": eplace_cost,
        "column_cost": col_cost_v,
        "delta": delta,
        "refined_hpwl_ratio": (best["hpwl"] / column_hpwl
                               if (best is not None and column_hpwl) else None),
        "refined_overlap": best["overlap_frac"] if best is not None else None,
        "refine_time": sum(t["refine_time"] for t in tried),
    }


# ---------------------------------------------------------------------------
def load_column_optimizer():
    """Instantiate src/solver/contest_optimizer.py MyOptimizer under the
    canonical G1 solver env, the way pseudo_hidden_eval.py's load_optimizer
    does, but with DIRECT_OFF=1 PARTNER_FLOW_SLOTS=0 to keep this probe
    CPU-only and fast (per task spec)."""
    from icdc_engine.g1_runtime import _SOLVER_ENV  # noqa: WPS433 (local import, optional dep)

    env = dict(_SOLVER_ENV)
    env["DIRECT_OFF"] = "1"
    env["PARTNER_FLOW_SLOTS"] = "0"
    for k, v in env.items():
        os.environ[k] = str(v)

    optimizer_path = REPO / "src" / "solver" / "contest_optimizer.py"
    spec = importlib.util.spec_from_file_location("eplace_probe_optimizer", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["MyOptimizer", "ContestOptimizer", "Optimizer"]:
        if hasattr(module, name):
            return getattr(module, name)(verbose=False)
    raise ValueError(f"no optimizer class found in {optimizer_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ids", default="79-99")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--configs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/p0_newbox/eplace_probe.json")
    ap.add_argument("--legalize", action="store_true",
                     help="run ePlace candidates through the production "
                          "legalize+refine pipeline (refine_prediction) and "
                          "compare official cost_no_runtime vs the column "
                          "champion, instead of just raw HPWL")
    ap.add_argument("--refine-deadline", type=float, default=0.5,
                     help="seconds handed to refine_prediction per candidate "
                          "(--legalize mode)")
    ap.add_argument("--max-refine-candidates", type=int, default=4,
                     help="max ePlace candidates (best raw HPWL first) run "
                          "through refine_prediction per case")
    args = ap.parse_args()

    ids = parse_id_ranges(args.test_ids)

    cases = load_test_cases(ev)
    by_id = {c["test_id"]: c for c in cases}

    try:
        column_opt = load_column_optimizer()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load column-backbone optimizer: {exc}")
        column_opt = None

    results: List[Dict[str, Any]] = []
    for tid in ids:
        if tid not in by_id:
            print(f"test_id {tid} not found, skipping")
            continue
        case = by_id[tid]
        n = case["n"]
        area = case["area"]
        cons = case["cons"]
        b2b = case["b2b"]
        p2b = case["p2b"]
        pins = case["pins"]
        tp = case["tp"]
        a_tot = float(area.sum().item())

        t0 = time.time()
        candidates, diags = generate_candidates(
            area, b2b, p2b, pins, cons, tp,
            n_configs=args.configs, iters=args.iters, seed=args.seed)
        eplace_runtime = time.time() - t0

        eplace_all = []
        for rects, diag in zip(candidates, diags):
            rect_list = [tuple(float(v) for v in r) for r in rects]
            hpwl = hpwl_of(rect_list, b2b, p2b, pins)
            ov = overlap_frac_of(rects, a_tot)
            bbox = ev.calculate_bbox_area(rect_list)
            eplace_all.append({
                "config": diag.get("config_index"),
                "ws": diag.get("ws"), "gamma": diag.get("gamma"),
                "hpwl": hpwl, "overlap_frac": ov, "bbox_area": bbox,
            })

        if eplace_all:
            eplace_best = min(eplace_all, key=lambda d: d["hpwl"])
        else:
            eplace_best = None

        column_entry = None
        if column_opt is not None:
            try:
                t1 = time.time()
                col_rects = column_opt.solve(n, area, b2b, p2b, pins, cons, tp)
                col_runtime = time.time() - t1
                col_rect_list = [tuple(float(v) for v in r) for r in col_rects]
                col_hpwl = hpwl_of(col_rect_list, b2b, p2b, pins)
                col_bbox = ev.calculate_bbox_area(col_rect_list)
                column_entry = {"hpwl": col_hpwl, "bbox_area": col_bbox, "runtime": col_runtime}
            except Exception as exc:  # noqa: BLE001
                print(f"case {tid}: column solve failed: {exc}")
                column_entry = None

        hpwl_ratio_best = None
        if eplace_best is not None and column_entry is not None and column_entry["hpwl"] > 0:
            hpwl_ratio_best = eplace_best["hpwl"] / column_entry["hpwl"]

        legalize_entry = None
        if args.legalize:
            legalize_entry = run_legalize_compare(
                candidates, eplace_all, case, tid,
                column_opt=column_opt,
                refine_deadline=args.refine_deadline,
                max_candidates=args.max_refine_candidates,
                seed=args.seed,
            )

        entry = {
            "test_id": tid, "n": n,
            "eplace_best": eplace_best,
            "eplace_all": eplace_all,
            "column": column_entry,
            "eplace_runtime_total": eplace_runtime,
            "hpwl_ratio_best": hpwl_ratio_best,
            "legalize": legalize_entry,
        }
        results.append(entry)
        ov_str = f"{eplace_best['overlap_frac']:.3f}" if eplace_best else "NA"
        ratio_str = f"{hpwl_ratio_best:.4f}" if hpwl_ratio_best is not None else "NA"
        if legalize_entry is not None:
            le = legalize_entry
            print(f"case {tid:3d} n={n:3d} eplace_cost={le['eplace_cost']!s:>8} "
                  f"col_cost={le['column_cost']!s:>8} delta={le['delta']!s:>8} "
                  f"ov={le['refined_overlap']!s:>6} rt={le['refine_time']:.2f}s "
                  f"legal={le['n_legal']}/{le['n_tried']}")
        else:
            print(f"case {tid:3d} n={n:3d} ratio={ratio_str:>8} overlap={ov_str:>8} "
                  f"n_ok={len(eplace_all)}/{args.configs} t={eplace_runtime:.2f}s")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    ratios = [r["hpwl_ratio_best"] for r in results if r["hpwl_ratio_best"] is not None]
    overlaps = [r["eplace_best"]["overlap_frac"] for r in results if r["eplace_best"] is not None]
    print("\n--- summary ---")
    print(f"{'test_id':>8} {'n':>5} {'ratio':>8} {'overlap':>9}")
    for r in results:
        ratio_str = f"{r['hpwl_ratio_best']:.4f}" if r["hpwl_ratio_best"] is not None else "NA"
        ov_str = (f"{r['eplace_best']['overlap_frac']:.3f}"
                  if r["eplace_best"] is not None else "NA")
        print(f"{r['test_id']:>8} {r['n']:>5} {ratio_str:>8} {ov_str:>9}")
    if ratios:
        print(f"\nmean ratio={np.mean(ratios):.4f} median ratio={np.median(ratios):.4f}")
    if overlaps:
        print(f"mean overlap_frac={np.mean(overlaps):.4f} median overlap_frac={np.median(overlaps):.4f}")

    if args.legalize:
        print("\n--- legalize summary (official cost_no_runtime) ---")
        print(f"{'test_id':>8} {'n':>5} {'eplace':>9} {'column':>9} {'delta':>9} "
              f"{'hpwl_r':>8} {'ov':>6} {'rt':>6} {'legal':>7}")
        rows = []
        for r in results:
            le = r["legalize"]
            if le is None:
                continue
            rows.append((r["test_id"], r["n"], le))
            ec = f"{le['eplace_cost']:.4f}" if le["eplace_cost"] is not None else "NA"
            cc = f"{le['column_cost']:.4f}" if le["column_cost"] is not None else "NA"
            dd = f"{le['delta']:+.4f}" if le["delta"] is not None else "NA"
            hr = f"{le['refined_hpwl_ratio']:.4f}" if le["refined_hpwl_ratio"] is not None else "NA"
            ov = f"{le['refined_overlap']:.3f}" if le["refined_overlap"] is not None else "NA"
            print(f"{r['test_id']:>8} {r['n']:>5} {ec:>9} {cc:>9} {dd:>9} "
                  f"{hr:>8} {ov:>6} {le['refine_time']:>5.2f}s {le['n_legal']:>3}/{le['n_tried']:<3}")

        deltas = [(n, le["delta"]) for _, n, le in rows if le["delta"] is not None]
        if deltas:
            max_n = max(n for n, _ in deltas)
            weights = [np.exp((n - max_n) / 12.0) for n, _ in deltas]
            weighted_delta = sum(w * d for (n, d), w in zip(deltas, weights)) / sum(weights)
            wins = sum(1 for _, d in deltas if d < 0)
            losses = sum(1 for _, d in deltas if d > 0)
            ties = sum(1 for _, d in deltas if d == 0)
            print(f"\nweighted mean delta (eplace - column, exp((n-max_n)/12)) = {weighted_delta:+.4f}")
            print(f"wins(eplace<column)={wins} losses(eplace>column)={losses} ties={ties} "
                  f"of {len(deltas)} comparable cases")
        n_no_legal = sum(1 for r in results if r["legalize"] is not None
                         and r["legalize"]["n_legal"] == 0)
        if n_no_legal:
            print(f"\n{n_no_legal} case(s) produced NO legal refined ePlace candidate")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
