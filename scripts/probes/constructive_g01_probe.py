#!/usr/bin/env python3
"""Run the isolated constructive G0/G1 gates with the official evaluator.

G0 feeds repaired-golden layouts through a lossy order/aspect/16x16-region
encoding before construction.  G1 supplies no label geometry and compares a
four-policy best-of-four portfolio with repaired production on the 21 n>=100
validation cases.  Nothing in this script is imported by the submission path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PARTNER = REPO / "partner"
if str(PARTNER) not in sys.path:
    sys.path.insert(0, str(PARTNER))

from constructive_g01 import (  # noqa: E402
    ConstructivePolicy,
    construct_candidate,
    construct_portfolio,
    default_policies,
    encode_oracle_codes,
    summarize_g1,
    weighted_score,
)
from icdc import data as icdc_data  # noqa: E402
from icdc.engine import verify_hard_legal  # noqa: E402


G0_POLICY = ConstructivePolicy(
    "oracle_region",
    "oracle",
    bbox_weight=0.65,
    hpwl_weight=1.00,
    anchor_weight=3.00,
    constraint_weight=1.50,
)


def _load_evaluator(data_path: Path):
    for path in (data_path, data_path / "iccad2026contest"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "constructive_g01_evaluator", REPO / "scripts" / "iccad2026_evaluate.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the official evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    return evaluator


def _load_layouts(path: Path) -> dict[int, np.ndarray]:
    raw = json.loads(path.read_text())
    layouts: dict[int, np.ndarray] = {}
    for key, value in raw.items():
        rects = np.asarray(value, dtype=np.float64)
        if rects.ndim == 3:
            if len(rects) != 1:
                raise ValueError(f"{path}: case {key} contains a multi-layout bank")
            rects = rects[0]
        if rects.ndim != 2 or rects.shape[1] != 4:
            raise ValueError(f"{path}: case {key} is not [N,4]")
        layouts[int(key)] = rects
    return layouts


def _official_metrics(evaluator, case: dict[str, Any], rects: np.ndarray):
    return evaluator.evaluate_solution(
        {"positions": [tuple(map(float, row)) for row in rects], "runtime": 1.0},
        {"hpwl_baseline": case["hpwl_ref"], "area_baseline": case["area_ref"]},
        case["cons"].to(torch.float32),
        case["b2b"].to(torch.float32),
        case["p2b"].to(torch.float32),
        case["pins"].to(torch.float32),
        case["area"].to(torch.float32),
        case["golden"],
        median_runtime=1.0,
    )


def _metric_record(metrics) -> dict[str, Any]:
    return {
        "cost": float(metrics.cost_no_runtime),
        "feasible": bool(metrics.is_feasible),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "v_rel": float(metrics.violations_relative),
        "boundary": int(metrics.boundary_violations),
        "grouping": int(metrics.grouping_violations),
        "mib": int(metrics.mib_violations),
    }


def run_g0(
    evaluator,
    cases: list[dict[str, Any]],
    repaired_golden: dict[int, np.ndarray],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        test_id = int(case["test_id"])
        oracle_rects = repaired_golden[test_id]
        codes = encode_oracle_codes(oracle_rects, region_bins=16)
        result = construct_candidate(
            case["area"].numpy(),
            case["cons"].numpy(),
            case["tp"].numpy(),
            case["b2b"].numpy(),
            case["p2b"].numpy(),
            case["pins"].numpy(),
            G0_POLICY,
            region_codes=codes.regions,
            aspect_codes=codes.log_aspect,
            oracle_order=codes.order,
            region_bins=codes.region_bins,
            frame_codes=(codes.frame_aspect_code, codes.utilization_code),
        )
        hard = verify_hard_legal(
            result.rects,
            case["area"].numpy(),
            case["cons"].numpy(),
            case["tp"].numpy(),
        )
        candidate_metrics = _official_metrics(evaluator, case, result.rects)
        calibration_metrics = _official_metrics(evaluator, case, oracle_rects)
        row = {
            "test_id": test_id,
            "n": int(case["n"]),
            "elapsed_s": result.elapsed_s,
            "hard_legal": bool(hard["ok"]),
            "hard_checks": hard,
            "candidate": _metric_record(candidate_metrics),
            "repaired_golden_calibration": _metric_record(calibration_metrics),
            "diagnostics": dict(result.diagnostics),
        }
        rows.append(row)
        print(
            f"G0 {test_id:02d} n={case['n']:3d} cost={row['candidate']['cost']:.4f} "
            f"legal={int(row['hard_legal'])} t={1e3 * result.elapsed_s:.1f}ms",
            flush=True,
        )
    counts = [row["n"] for row in rows]
    score = weighted_score([row["candidate"]["cost"] for row in rows], counts)
    calibration = weighted_score(
        [row["repaired_golden_calibration"]["cost"] for row in rows], counts
    )
    n120_times = [row["elapsed_s"] for row in rows if row["n"] == 120]
    max_n_time = max(
        row["elapsed_s"] for row in rows if row["n"] == max(counts)
    )
    all_legal = all(row["hard_legal"] and row["candidate"]["feasible"] for row in rows)
    decode_gate = max(n120_times) if n120_times else max_n_time
    return {
        "score": score,
        "repaired_golden_calibration_score": calibration,
        "hard_legal_cases": sum(row["hard_legal"] for row in rows),
        "case_count": len(rows),
        "n120_decode_s": decode_gate,
        "passed": all_legal and score <= 1.03 and decode_gate <= 0.020,
        "rows": rows,
    }


def run_g1(
    evaluator,
    all_cases: list[dict[str, Any]],
    production: dict[int, np.ndarray],
    case_limit: int | None = None,
) -> dict[str, Any]:
    cases = [case for case in all_cases if int(case["n"]) >= 100]
    if case_limit is not None:
        cases = cases[:case_limit]
    rows: list[dict[str, Any]] = []
    policies = default_policies()
    for case in cases:
        test_id = int(case["test_id"])
        baseline_metrics = _official_metrics(evaluator, case, production[test_id])
        results = construct_portfolio(
            case["area"].numpy(),
            case["cons"].numpy(),
            case["tp"].numpy(),
            case["b2b"].numpy(),
            case["p2b"].numpy(),
            case["pins"].numpy(),
            policies=policies,
        )
        candidates = []
        legal_count = 0
        for result in results:
            hard = verify_hard_legal(
                result.rects,
                case["area"].numpy(),
                case["cons"].numpy(),
                case["tp"].numpy(),
            )
            metrics = _official_metrics(evaluator, case, result.rects)
            admitted = bool(hard["ok"] and metrics.is_feasible)
            legal_count += int(admitted)
            candidates.append(
                {
                    "policy": result.policy,
                    "elapsed_s": result.elapsed_s,
                    "hard_legal": bool(hard["ok"]),
                    "admitted": admitted,
                    "metrics": _metric_record(metrics),
                    "diagnostics": dict(result.diagnostics),
                }
            )
        admissible_costs = [row["metrics"]["cost"] for row in candidates if row["admitted"]]
        best_raw = min(admissible_costs) if admissible_costs else 10.0
        baseline_cost = float(baseline_metrics.cost_no_runtime)
        portfolio_cost = min(baseline_cost, best_raw)
        row = {
            "test_id": test_id,
            "n": int(case["n"]),
            "baseline_cost": baseline_cost,
            "best_constructive_cost": best_raw,
            "candidate_cost": portfolio_cost,
            "legal_candidates": legal_count,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        rows.append(row)
        print(
            f"G1 {test_id:02d} n={case['n']:3d} base={baseline_cost:.4f} "
            f"best={best_raw:.4f} legal={legal_count}/{len(candidates)}",
            flush=True,
        )
    summary = summarize_g1(rows, [int(case["n"]) for case in all_cases])
    return {**summary, "rows": rows}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("g0", "g1", "both"), default="both")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--golden-layouts", type=Path)
    parser.add_argument("--production-layouts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="smoke only; take the first N loaded cases")
    args = parser.parse_args()

    if args.mode in ("g0", "both") and args.golden_layouts is None:
        parser.error("--golden-layouts is required for G0")
    if args.mode in ("g1", "both") and args.production_layouts is None:
        parser.error("--production-layouts is required for G1")

    evaluator = _load_evaluator(args.data_path)
    cases = icdc_data.load_test_cases(evaluator, str(args.data_path))
    payload: dict[str, Any] = {"mode": args.mode}
    if args.mode in ("g0", "both"):
        g0_cases = cases[: args.limit] if args.limit is not None else cases
        payload["g0"] = run_g0(evaluator, g0_cases, _load_layouts(args.golden_layouts))
    if args.mode in ("g1", "both"):
        payload["g1"] = run_g1(
            evaluator,
            cases,
            _load_layouts(args.production_layouts),
            case_limit=args.limit,
        )
    _atomic_write_json(args.out, payload)
    summary = {
        key: {field: value for field, value in result.items() if field != "rows"}
        for key, result in payload.items()
        if key != "mode"
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
