"""Dump a K-layout bank in the format rung (-1) / the oracle probe consumes.

Output: ``{"<test_id>": [[[x, y, w, h], ...] x K]}`` -- the MULTI form
`column_sa_legalizer.oracle_pred_override` auto-detects by rank.  The single
-layout form is deliberately *not* produced: broadcasting one layout across the
batch destroys the diversity a real K-sample engine has, and that homogenisation
alone was measured at ~+0.035 weighted noRT.

Only layouts that pass the rung-(-1) admission predicates are written.  A case
with no admissible layout is omitted, which `oracle_pred_override` degrades to
the control arm case-by-case rather than corrupting the run -- so the bank is
always safe to inject, and the coverage number is reported instead of hidden.

Alongside the bank it writes a `.report.json` with, per case: how many of the K
were admissible, the offline official `cost_no_runtime` of each (computed with
the real evaluator, no solver in the loop), and the engine's own energy.

usage:
  uv run python -m icdc.dump_bank --ckpt <ft.pt> --out bank.json [--min-n 100]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "partner", REPO / "FloorSet" / "iccad2026contest",
           REPO / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from icdc import data as D            # noqa: E402
from icdc import energy as EN         # noqa: E402
from icdc import engine as G          # noqa: E402
from icdc import tfdl as T            # noqa: E402


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "repo_iccad2026_evaluate", REPO / "scripts" / "iccad2026_evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["repo_iccad2026_evaluate"] = ev
    spec.loader.exec_module(ev)
    return ev


def official_cost(ev, case, rects) -> dict:
    m = ev.evaluate_solution(
        {"positions": list(rects), "runtime": 1.0},
        {"hpwl_baseline": case["hpwl_ref"], "area_baseline": case["area_ref"]},
        case["cons"].to(torch.float32), case["b2b"].to(torch.float32),
        case["p2b"].to(torch.float32), case["pins"].to(torch.float32),
        case["area"].to(torch.float32), case["golden"], median_runtime=1.0)
    return {"cost": float(m.cost_no_runtime), "feasible": bool(m.is_feasible),
            "v_rel": float(m.violations_relative),
            "hpwl_gap": float(m.hpwl_gap), "area_gap": float(m.area_gap)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-ema", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ev = load_evaluator()
    cases = [c for c in D.load_test_cases(ev) if c["n"] >= args.min_n]
    print(f"band n>={args.min_n}: {len(cases)} cases", flush=True)
    model, schedule, cfg, _ = G.load_model(args.ckpt, device=args.device,
                                           use_ema=bool(args.use_ema))
    model.eval()
    batch = D.collate(cases, device=args.device, dtype=torch.float32)
    t0 = time.time()
    bank, drift = G.sample_bank(model, schedule, batch, cfg, K=args.K,
                                steps=args.steps, seed=args.seed,
                                device=args.device)
    print(f"sampled {len(cases)}x{args.K} in {time.time()-t0:.1f}s", flush=True)

    bank = bank.cpu().numpy()
    dr = drift.cpu().numpy()
    b64 = D.collate(cases, dtype=torch.float64)
    out, report = {}, []
    n_ok = 0
    for k, c in enumerate(cases):
        n = c["n"]
        area, cons, tp = c["area"].numpy(), c["cons"].numpy(), c["tp"].numpy()
        lays, recs = [], []
        for j in range(args.K):
            P = np.ascontiguousarray(bank[k, j, :n], dtype=np.float64)
            chk = G.verify_hard_legal(P, area, cons, tp)
            row = {"k": j, "admissible": chk["ok"], "drift": float(dr[k, j])}
            if not chk["ok"]:
                row["failed"] = [p for p, v in chk.items() if not v and p != "ok"]
            row.update(official_cost(ev, c, [tuple(map(float, r)) for r in P]))
            if chk["ok"]:
                lays.append(P.tolist())
                n_ok += 1
            recs.append(row)
        if lays:
            out[str(c["test_id"])] = lays
        # engine's own energy on the admissible set, for the correlation record
        e_rows = [r["cost"] for r in recs if r["admissible"]]
        report.append({"test_id": c["test_id"], "n": n,
                       "admissible": len(lays), "K": args.K,
                       "cost_min": min(e_rows) if e_rows else None,
                       "cost_med": float(np.median(e_rows)) if e_rows else None,
                       "samples": recs})

    Path(args.out).write_text(json.dumps(out))
    Path(args.out + ".report.json").write_text(json.dumps(report, indent=1))
    costs = [r["cost"] for rr in report for r in rr["samples"] if r["admissible"]]
    best = [rr["cost_min"] for rr in report if rr["cost_min"] is not None]
    print(f"admissible {n_ok}/{len(cases)*args.K} "
          f"({100*n_ok/max(len(cases)*args.K,1):.0f}%); "
          f"cases with >=1 layout {len(out)}/{len(cases)}")
    if costs:
        print(f"offline official cost (admissible): median {np.median(costs):.4f} "
              f"mean {np.mean(costs):.4f} p10 {np.percentile(costs,10):.4f} "
              f"min {np.min(costs):.4f}")
        print(f"per-case best-of-K: median {np.median(best):.4f} "
              f"min {np.min(best):.4f} max {np.max(best):.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
