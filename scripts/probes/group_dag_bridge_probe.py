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
import ast
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

EXPECTED_BASELINE_SHA256 = "6e7089b1e0fa4510f5ffe11187232bee488b25da599435f7d369ba329a1349c5"
EXPECTED_BASELINE_SCORE = 1.1437448258795715
SCORE_LIMIT = 1.1412448258795715
SCHEMA = "group-dag-g0.v1"
_EVALUATOR = None


def _stable_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _weighted(costs: list[float], counts: list[int]) -> float:
    if not costs:
        return 0.0
    m = max(counts) if counts else 0
    ws = [math.exp((n - m) / 12) for n in counts]
    return sum(c * w for c, w in zip(costs, ws)) / sum(ws)


def _load_evaluator():
    global _EVALUATOR
    if _EVALUATOR is not None:
        return _EVALUATOR
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "FloorSet"))
    spec = importlib.util.spec_from_file_location("g0_evaluator", root / "scripts" / "iccad2026_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    _EVALUATOR = mod
    return mod


def _default_loader(test_id: int, row: dict[str, Any]):
    # The probe intentionally loads inputs only for scoring; it never invokes an optimizer.
    root = Path(__file__).resolve().parents[2]
    ev = _load_evaluator()
    from icdc_engine.data import load_test_cases
    return load_test_cases(ev, data_path=str(root / "FloorSet"))[test_id]


def _default_cases():
    root = Path(__file__).resolve().parents[2]
    ev = _load_evaluator()
    from icdc_engine.data import load_test_cases
    cases = load_test_cases(ev, data_path=str(root / "FloorSet"))
    out = []
    for case in cases:
        out.append(dict(case))
    return out


def _default_evaluate(solution: dict[str, Any], case: Any):
    ev = _load_evaluator()
    if isinstance(case, dict) and "cons" in case:
        return ev.evaluate_solution(solution, {"hpwl_baseline": case["hpwl_ref"], "area_baseline": case["area_ref"]}, case["cons"], case["b2b"], case["p2b"], case["pins"], case["area"], target_positions=case["tp"].tolist(), median_runtime=1.0)
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


def _source_hygiene_text(text: str) -> bool:
    """Audit the DAG function and its module-local helper call closure."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    root = functions.get("bridge_grouping_violations_dag")
    if root is None:
        return False
    forbidden = ("coord_polish", "_fix_grouping", "test_id", "golden_positions", "saved_positions", "raw_positions", "coordinates")
    queue = [root]; seen: set[str] = set()
    while queue:
        node = queue.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
        attrs = [n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)]
        constants = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if any(any(token in value for token in forbidden) for value in names + attrs + constants):
            return False
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) and any("coord_polish" in alias.name for alias in n.names) for n in ast.walk(node)):
            return False
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id in functions:
                queue.append(functions[call.func.id])
    # A forbidden selector in a local declaration is suspicious even when a
    # mutation makes the declaration unreachable from the root call.
    if any(token in name for name in functions for token in ("test_id", "golden_positions", "raw_positions")):
        return False
    return True


def _source_hygiene() -> bool:
    root = Path(__file__).resolve().parents[2] / "src" / "solver" / "violation_killer.py"
    return _source_hygiene_text(root.read_text())


def replay(args, *, case_loader=_default_loader, bridge_fn=None, evaluate_fn=_default_evaluate, clock=time.perf_counter) -> int:
    source = Path(args.input)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        package = json.loads(raw)
        if package.get("schema") not in (None, SCHEMA):
            raise ValueError("unsupported baseline schema")
        rows = package["test_results"]
        ids = [int(r["test_id"]) for r in rows]
        injected = bridge_fn is not None or case_loader is not _default_loader or evaluate_fn is not _default_evaluate
        if (not injected and len(rows) != 100) or ids != list(range(len(rows))):
            raise ValueError("baseline must contain test IDs 0..99")
        baseline_score = _weighted([float(r["cost_no_runtime"]) for r in rows], [int(r["block_count"]) for r in rows])
        if digest != EXPECTED_BASELINE_SHA256 or not math.isclose(baseline_score, EXPECTED_BASELINE_SCORE, rel_tol=0, abs_tol=1e-12):
            return 2
    except Exception:
        return 2
    production_mode = bridge_fn is None and case_loader is _default_loader and evaluate_fn is _default_evaluate
    cached_cases = None
    if production_mode:
        cached_cases = _default_cases()
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "solver"))
        from violation_killer import bridge_grouping_violations_dag
        bridge_fn = lambda context, positions, budget: bridge_grouping_violations_dag(context["optimizer"], positions, budget)

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
            scorer = (cached_cases[tid] if production_mode else case_loader(tid, row))
            if production_mode:
                from partner.column_sa_legalizer import _ColumnOptimizer
                scorer = dict(scorer)
                scorer["optimizer"] = _ColumnOptimizer(row["positions"], torch.as_tensor(scorer["area"]), torch.as_tensor(scorer["cons"]), torch.as_tensor(scorer["tp"]), torch.as_tensor(scorer["b2b"]), torch.as_tensor(scorer["p2b"]), torch.as_tensor(scorer["pins"]), time.time() + 60., seed=0)
            started = clock()
            after = bridge_fn(scorer, before, 0.003)
            elapsed = (clock() - started) * 1000
            timings.append(elapsed)
            before_arr = np.asarray(before, dtype=float); after_arr = np.asarray(after, dtype=float)
            if before_arr.shape != after_arr.shape or not np.isfinite(after_arr).all(): raise ValueError("invalid candidate")
            off = evaluate_fn({"positions": before_arr.tolist(), "runtime": 1.0}, scorer)
            off_cost = _metric(off, "cost_no_runtime", _metric(off, "cost"))
            expected_off = float(row["cost_no_runtime"])
            if not math.isclose(off_cost, expected_off, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"baseline_replay_mismatch:{off_cost!r}!={expected_off!r}")
            on = evaluate_fn({"positions": after_arr.tolist(), "runtime": 1.0}, scorer)
            on_cost = _metric(on, "cost_no_runtime", _metric(on, "cost"))
            off_costs.append(off_cost); on_costs.append(on_cost); counts.append(int(row["block_count"]))
            ok = bool(getattr(on, "is_feasible", on.get("is_feasible", True) if isinstance(on, dict) else True))
            feasible += int(ok)
            grouping_before = _metric(off, "grouping_violations", row.get("grouping_violations", 0))
            grouping_after = _metric(on, "grouping_violations", grouping_before)
            v_before = _metric(off, "total_soft_violations", row.get("total_soft_violations", 0))
            v_after = _metric(on, "total_soft_violations", v_before)
            after = after_arr.tolist(); changed = not np.array_equal(after_arr, before_arr)
            strict = (not changed) or (ok and grouping_after < grouping_before and v_after < v_before)
            accepted = changed and strict
            cases.append({"test_id": tid, "block_count": int(row["block_count"]), "positions": after,
                          "before_cost_no_runtime": off_cost, "after_cost_no_runtime": on_cost,
                          "grouping_before": grouping_before, "grouping_after": grouping_after,
                          "v_before": v_before, "v_after": v_after, "grouping_delta": grouping_after-grouping_before,
                          "v_delta": v_after-v_before, "runtime_ms": elapsed, "is_feasible": ok,
                          "error": None, "changed": changed, "candidate_strict": strict, "accepted": accepted})
        except Exception as exc:
            errors += 1
            cases.append({"test_id": tid, "block_count": int(row["block_count"]), "positions": before,
                          "runtime_ms": 0.0, "is_feasible": False, "error": type(exc).__name__, "accepted": False})
    source_ok = _source_hygiene()
    runtime = {"mean": statistics.mean(timings) if timings else 0.0, "median": statistics.median(timings) if timings else 0.0,
               "p95": sorted(timings)[max(0, math.ceil(.95*len(timings))-1)] if timings else 0.0, "max": max(timings) if timings else 0.0}
    score_off = _weighted(off_costs, counts) if errors == 0 and off_costs else None
    score_on = _weighted(on_costs, counts) if errors == 0 and on_costs else None
    case_path = Path(args.output)
    _stable_write(case_path, cases)
    case_digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    manifest = {"schema": SCHEMA, "baseline": {"path": str(source), "sha256": digest, "score_off": baseline_score,
                                                    "cases_path": str(case_path), "cases_sha256": case_digest},
                "feasible": feasible, "errors": errors, "score_on": score_on, "score_off": score_off,
                "grouping_delta": sum(c.get("grouping_delta", 0) for c in cases), "v_delta": sum(c.get("v_delta", 0) for c in cases),
                "runtime_ms": runtime, "causal_mean_ms": runtime["mean"], "accepted": bool(feasible == len(rows) and errors == 0 and score_on is not None and score_on <= SCORE_LIMIT and runtime["mean"] <= .75 and source_ok and all(c.get("candidate_strict", False) for c in cases)),
                "source_hygiene": source_ok}
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
