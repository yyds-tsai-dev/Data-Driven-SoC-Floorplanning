#!/usr/bin/env python3
"""Offline G0 gate for preplaced-frame outlier reinsertion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "partner", ROOT / "FloorSet",
              ROOT / "FloorSet" / "iccad2026contest"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from column_sa_legalizer import _ColumnOptimizer  # noqa: E402
from frame_reinsert import frame_reinsert, weighted_score  # noqa: E402
from icdc.data import load_test_cases  # noqa: E402
from icdc.dump_bank import official_cost  # noqa: E402


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "frame_reinsert_evaluator", ROOT / "scripts" / "iccad2026_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, payload) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2))
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", required=True,
                        help="full evaluator JSON containing test_results.positions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-path", default=str(ROOT / "FloorSet"))
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    evaluator = _load_evaluator()
    cases = load_test_cases(evaluator, args.data_path)
    source = json.loads(Path(args.layouts).read_text())
    source_rows = {int(row["test_id"]): row for row in source["test_results"]}
    rows = []
    processed = 0
    for case in cases:
        test_id = int(case["test_id"])
        source_row = source_rows[test_id]
        rects = [tuple(map(float, row)) for row in source_row["positions"]]
        before = official_cost(evaluator, case, rects)
        outcome = None
        after_rects = rects
        if case["n"] >= args.min_n and (not args.limit or processed < args.limit):
            scorer = _ColumnOptimizer(
                rects, case["area"], case["cons"], case["tp"],
                case["b2b"], case["p2b"], case["pins"],
                time.time() + 60.0, seed=0,
            )
            outcome = frame_reinsert(scorer, rects)
            after_rects = outcome.rects
            processed += 1
        after = official_cost(evaluator, case, after_rects)
        row = {
            "test_id": test_id,
            "n": int(case["n"]),
            "before": before,
            "after": after,
            "delta": float(after["cost"] - before["cost"]),
            "attempted": int(outcome.attempted) if outcome else 0,
            "accepted": int(outcome.accepted) if outcome else 0,
            "elapsed_s": float(outcome.elapsed_s) if outcome else 0.0,
            "reason": outcome.reason if outcome else "filtered",
        }
        rows.append(row)
        if outcome and (outcome.attempted or outcome.accepted):
            print(
                f"tid={test_id:02d} n={case['n']:3d} "
                f"try={outcome.attempted} accept={outcome.accepted} "
                f"delta={row['delta']:+.9f} dt={outcome.elapsed_s:.6f} "
                f"reason={outcome.reason}",
                flush=True,
            )

    counts = [row["n"] for row in rows]
    before_costs = [row["before"]["cost"] for row in rows]
    after_costs = [row["after"]["cost"] for row in rows]
    elapsed = [row["elapsed_s"] for row in rows if row["elapsed_s"] > 0.0]
    summary = {
        "processed": processed,
        "feasible_before": sum(bool(row["before"]["feasible"]) for row in rows),
        "feasible_after": sum(bool(row["after"]["feasible"]) for row in rows),
        "accepted_cases": sum(row["accepted"] > 0 for row in rows),
        "strict_wins": sum(row["delta"] < -1e-12 for row in rows),
        "weighted_before": weighted_score(before_costs, counts),
        "weighted_after": weighted_score(after_costs, counts),
        "weighted_gain": weighted_score(before_costs, counts)
                         - weighted_score(after_costs, counts),
        "elapsed_mean_s": statistics.mean(elapsed) if elapsed else 0.0,
        "elapsed_median_s": statistics.median(elapsed) if elapsed else 0.0,
        "elapsed_max_s": max(elapsed, default=0.0),
    }
    payload = {"layouts": str(Path(args.layouts).resolve()),
               "summary": summary, "rows": rows}
    _atomic_json(Path(args.out), payload)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
