#!/usr/bin/env python3
"""Deterministic, fail-closed replay evidence for the changed-contact DAG."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

EXPECTED_BASELINE_SHA256 = "6e7089b1e0fa4510f5ffe11187232bee488b25da599435f7d369ba329a1349c5"
EXPECTED_BASELINE_SCORE = 1.1437448258795715
SCORE_LIMIT = EXPECTED_BASELINE_SCORE
SCHEMA = "group-dag-g0.v1"


def _stable_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(tmp, path)


def _weighted(costs: list[float], counts: list[int]) -> float:
    if not costs:
        return 0.0
    m = max(counts) if counts else 0
    ws = [math.exp((n - m) / 12) for n in counts]
    return sum(c * w for c, w in zip(costs, ws)) / sum(ws)


def _load_evaluator():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("g0_evaluator", root / "scripts" / "iccad2026_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def _default_loader(test_id: int, row: dict[str, Any]):
    # The probe intentionally loads inputs only for scoring; it never invokes an optimizer.
    ev = _load_evaluator()
    data = ev.get_validation_dataloader()
    return data.dataset[test_id] if hasattr(data, "dataset") else data[test_id]


def _default_evaluate(solution: dict[str, Any], case: Any):
    ev = _load_evaluator()
    if isinstance(case, dict):
        return ev.evaluate_solution(solution, **case)
    # Native validation samples are handled by the evaluator's public helper when available.
    if hasattr(case, "evaluate_solution"):
        return case.evaluate_solution(solution)
    raise TypeError("validation case cannot be scored by evaluate_solution")


def _metric(value: Any, name: str, default: float = 0.0) -> float:
    if isinstance(value, dict):
        value = value.get(name, default)
    else:
        value = getattr(value, name, default)
    return float(value)


def _call(fn: Callable, *args):
    """Permit compact fixture callbacks while keeping the production signatures explicit."""
    for n in range(len(args), 0, -1):
        try:
            return fn(*args[:n])
        except TypeError:
            if n == 1:
                raise
    return fn()


def _source_hygiene() -> bool:
    root = Path(__file__).resolve().parents[2] / "partner" / "violation_killer.py"
    text = root.read_text()
    start = text.find("def bridge_grouping_violations_dag")
    if start < 0:
        return False
    # Restrict policy scanning to the DAG implementation, avoiding unrelated
    # legacy evaluator helpers that legitimately mention generic block counts.
    end = text.find("\ndef ", start + 4)
    text = text[start:] if end < 0 else text[start:end]
    forbidden = ("coord_polish", "_fix_grouping", "test_id", "golden_positions", "raw_positions")
    return not any(token in text for token in forbidden)


def replay(args, *, case_loader=_default_loader, bridge_fn=None, evaluate_fn=_default_evaluate) -> int:
    source = Path(args.input)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        package = json.loads(raw)
        rows = package["test_results"]
        ids = [int(r["test_id"]) for r in rows]
        if len(rows) != 100 or ids != list(range(100)):
            raise ValueError("baseline must contain test IDs 0..99")
        baseline_score = _weighted([float(r["cost_no_runtime"]) for r in rows], [int(r["block_count"]) for r in rows])
        if digest != EXPECTED_BASELINE_SHA256 or not math.isclose(baseline_score, EXPECTED_BASELINE_SCORE, rel_tol=0, abs_tol=1e-12):
            return 2
    except Exception:
        return 2
    if bridge_fn is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "partner"))
        from violation_killer import bridge_grouping_violations_dag
        bridge_fn = bridge_grouping_violations_dag

    cases: list[dict[str, Any]] = []
    on_costs: list[float] = []
    off_costs: list[float] = []
    counts: list[int] = []
    timings: list[float] = []
    errors = 0
    feasible = 0
    for row in rows:
        tid = int(row["test_id"])
        before = row["positions"]
        try:
            scorer = _call(case_loader, tid, row)
            started = time.perf_counter()
            after = _call(bridge_fn, scorer, before, 0.003)
            elapsed = (time.perf_counter() - started) * 1000
            timings.append(elapsed)
            off = _call(evaluate_fn, {"positions": before}, scorer)
            on = _call(evaluate_fn, {"positions": after}, scorer)
            off_cost = _metric(off, "cost_no_runtime", _metric(off, "cost"))
            on_cost = _metric(on, "cost_no_runtime", _metric(on, "cost"))
            off_costs.append(off_cost); on_costs.append(on_cost); counts.append(int(row["block_count"]))
            ok = bool(getattr(on, "is_feasible", on.get("is_feasible", True) if isinstance(on, dict) else True))
            feasible += int(ok)
            grouping_before = _metric(off, "grouping_violations", row.get("grouping_violations", 0))
            grouping_after = _metric(on, "grouping_violations", grouping_before)
            v_before = _metric(off, "total_soft_violations", row.get("total_soft_violations", 0))
            v_after = _metric(on, "total_soft_violations", v_before)
            accepted = ok and grouping_after < grouping_before and v_after < v_before
            cases.append({"test_id": tid, "block_count": int(row["block_count"]), "positions": after,
                          "before_cost_no_runtime": off_cost, "after_cost_no_runtime": on_cost,
                          "grouping_before": grouping_before, "grouping_after": grouping_after,
                          "v_before": v_before, "v_after": v_after, "grouping_delta": grouping_after-grouping_before,
                          "v_delta": v_after-v_before, "runtime_ms": elapsed, "is_feasible": ok,
                          "error": None, "accepted": accepted})
        except Exception as exc:
            errors += 1
            cases.append({"test_id": tid, "block_count": int(row["block_count"]), "positions": before,
                          "runtime_ms": 0.0, "is_feasible": False, "error": type(exc).__name__, "accepted": False})
    source_ok = _source_hygiene()
    runtime = {"mean": statistics.mean(timings) if timings else 0.0, "median": statistics.median(timings) if timings else 0.0,
               "p95": sorted(timings)[max(0, math.ceil(.95*len(timings))-1)] if timings else 0.0, "max": max(timings) if timings else 0.0}
    score_off = _weighted(off_costs, counts) if off_costs else float("inf")
    score_on = _weighted(on_costs, counts) if on_costs else float("inf")
    manifest = {"schema": SCHEMA, "baseline": {"path": str(source), "sha256": digest, "score_off": baseline_score},
                "feasible": feasible, "errors": errors, "score_on": score_on, "score_off": score_off,
                "grouping_delta": sum(c.get("grouping_delta", 0) for c in cases), "v_delta": sum(c.get("v_delta", 0) for c in cases),
                "runtime_ms": runtime, "causal_mean_ms": runtime["mean"], "accepted": bool(feasible == 100 and errors == 0 and score_on <= SCORE_LIMIT and runtime["mean"] <= .75 and source_ok and all(c["accepted"] or c.get("grouping_before", 0) == 0 for c in cases)),
                "source_hygiene": source_ok}
    _stable_write(Path(args.output), cases)
    _stable_write(Path(args.manifest), manifest)
    return 0 if manifest["accepted"] else 1


def main(argv=None, **kwargs) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay"); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--manifest", required=True)
    ns = parser.parse_args(argv)
    return replay(ns, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
