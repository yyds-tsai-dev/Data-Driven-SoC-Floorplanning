#!/usr/bin/env python3
"""Pseudo-hidden evaluation harness.

Evaluates the production partner solver (``src/solver/contest_optimizer.py``,
``MyOptimizer``) on a deterministic sample of held-out rows drawn from
``FloorSet/floorset_lite`` (the 1M-sample training set), producing an
evaluator-style JSON comparable to the official 100-case runs.

This is a *new* probe file only -- it reuses (does not reimplement) the
row -> solve()-input conversion in ``src/icdc_engine/data.py``
(``BandFileSampler._instance`` / ``target_positions_from_rects``), the
heldout/train split in ``src/icdc_engine/topology_data.py``
(``split_for_id``), and the official cost/violation math in
``scripts/iccad2026_evaluate.py`` (``evaluate_solution``,
``compute_total_score``).

Usage:
    uv run scripts/probes/pseudo_hidden_eval.py \\
        --sample-per-n 5 --n-min 21 --n-max 120 --seed 20260821 \\
        --split heldout --out artifacts/p0_newbox/pseudo_hidden_<tag>.json \\
        --tag <tag> [--limit N]

The caller is responsible for exporting whatever ``PARTNER_*`` /
``FLOORSET_*`` runtime env the solve() call should observe -- this script
does not set or hardcode any of them; it only reads ``os.environ`` via the
optimizer module it imports.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
LITE_ROOT = REPO / "FloorSet" / "floorset_lite"

for _p in (REPO / "partner", REPO / "FloorSet" / "iccad2026contest",
           REPO / "FloorSet", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reused conversion machinery -- do NOT reimplement.
from icdc_engine.data import BandFileSampler, target_positions_from_rects  # noqa: E402
from icdc_engine.topology_data import split_for_id  # noqa: E402

# Reused official cost/violation machinery -- do NOT reimplement.
import iccad2026_evaluate as ev  # noqa: E402


# ---------------------------------------------------------------------------
# File index: n -> list of relative paths ("worker_W/layouts_K.th")
# ---------------------------------------------------------------------------
def _relpath(path: str) -> str:
    return str(Path(path).relative_to(LITE_ROOT))


def build_or_load_index(cache_path: Path, n_min: int, n_max: int,
                         needed_per_n: int, seed: int) -> Dict[int, List[str]]:
    """Scan floorset_lite files to find which n each file holds.

    Caches progress in ``cache_path`` (JSON: {n: [relpaths]}) so repeat
    runs (and the full 500-case run after a smoke test) don't rescan files
    already known to hold an n outside the requested range, or that are
    already accounted for. Scanning stops early once every n in
    [n_min, n_max] has at least ``needed_per_n`` files recorded, or every
    file has been visited.
    """
    index: Dict[int, List[str]] = {}
    scanned: set = set()
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        index = {int(k): v for k, v in raw.get("index", {}).items()}
        scanned = set(raw.get("scanned", []))

    all_files = sorted(glob.glob(str(LITE_ROOT / "worker_*" / "layouts*.th")))
    rng = random.Random(seed ^ 0x1CDC)
    rng.shuffle(all_files)

    def satisfied() -> bool:
        return all(len(index.get(n, [])) >= needed_per_n
                   for n in range(n_min, n_max + 1))

    for path in all_files:
        rel = _relpath(path)
        if rel in scanned:
            continue
        if satisfied():
            break
        try:
            d = torch.load(path, map_location="cpu")
            n = int((d[0][0, :, 0] != -1).sum().item())
        except Exception:
            scanned.add(rel)
            continue
        scanned.add(rel)
        if n_min <= n <= n_max:
            index.setdefault(n, []).append(rel)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"index": index, "scanned": sorted(scanned)}))
    return index


# ---------------------------------------------------------------------------
# Deterministic heldout sampling
# ---------------------------------------------------------------------------
def sample_plan(index: Dict[int, List[str]], n_min: int, n_max: int,
                 sample_per_n: int, seed: int, split: str,
                 heldout_mod: int,
                 row_ok=None) -> List[Dict[str, Any]]:
    """[{n, file, index, instance_id}, ...] deterministic given seed.

    row_ok(rel, row) -> bool optionally rejects rows (e.g. MIB-heterogeneous
    training rows that cannot occur on the validation-style hidden set)."""
    plan = []
    for n in range(n_min, n_max + 1):
        files = list(index.get(n, []))
        if not files:
            continue
        rng = random.Random(f"{seed}:{n}")
        rng.shuffle(files)
        found = 0
        for rel in files:
            if found >= sample_per_n:
                break
            row_indices = list(range(112))
            rng.shuffle(row_indices)
            for row in row_indices:
                if found >= sample_per_n:
                    break
                iid = f"{rel}#{row}"
                if split_for_id(iid, heldout_mod) != split:
                    continue
                if row_ok is not None and not row_ok(rel, row):
                    continue
                plan.append({"n": n, "file": rel, "index": row, "instance_id": iid})
                found += 1
    return plan


def make_mib_spread_filter(max_spread: float):
    """Reject rows containing an MIB group whose member target areas differ by
    more than max_spread (relative).  Such groups make 'identical dims + hard
    1% area tolerance' mathematically unsatisfiable; the hidden/validation
    sets do not contain them (validation MIB groups are area-homogeneous and
    the beta rank-1 raw score of 1.0027 rules out unavoidable violations), so
    keeping them in the pseudo-hidden pool would inject a fake gap."""
    cache: Dict[str, Any] = {}

    def row_ok(rel: str, row: int) -> bool:
        if rel not in cache:
            cache.clear()
            cache[rel] = torch.load(str(LITE_ROOT / rel), map_location="cpu")
        case = BandFileSampler._instance(cache[rel], row)
        cons, area = case["cons"], case["area"]
        groups: Dict[int, List[float]] = {}
        for i in range(len(cons)):
            g = int(cons[i][2])
            if g > 0:
                groups.setdefault(g, []).append(float(area[i]))
        for members in groups.values():
            if len(members) >= 2:
                mx = max(members)
                if mx > 0 and (mx - min(members)) / mx > max_spread:
                    return False
        return True

    return row_ok


# ---------------------------------------------------------------------------
# Optimizer loading (same pattern as iccad2026_evaluate.EvaluationHarness._load_optimizer)
# ---------------------------------------------------------------------------
def load_optimizer(optimizer_path: Path):
    spec = importlib.util.spec_from_file_location("pseudo_hidden_optimizer_module", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and issubclass(obj, ev.FloorplanOptimizer)
                and obj.__name__ != "FloorplanOptimizer"):
            return obj(verbose=False)
    for name in ["MyOptimizer", "Optimizer", "ContestOptimizer"]:
        if hasattr(module, name):
            return getattr(module, name)(verbose=False)
    raise ValueError(f"no optimizer class found in {optimizer_path}")


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------
def evaluate_case(optimizer, case: Dict[str, Any]) -> Dict[str, Any]:
    n = case["n"]
    area = case["area"]
    cons = case["cons"]
    b2b = case["b2b"]
    p2b = case["p2b"]
    pins = case["pins"]
    tp = case["tp"]

    t0 = time.time()
    rects = optimizer.solve(n, area, b2b, p2b, pins, cons, tp)
    runtime = time.time() - t0

    positions = [tuple(float(v) for v in r) for r in rects]
    metrics = ev.evaluate_solution(
        {"positions": positions, "runtime": runtime},
        {"hpwl_baseline": case["hpwl_ref"], "area_baseline": case["area_ref"]},
        cons, b2b, p2b, pins, area,
        target_positions=tp.tolist(),
        median_runtime=1.0,
    )
    return {
        "hpwl_gap": metrics.hpwl_gap,
        "area_gap": metrics.area_gap,
        "v_rel": metrics.violations_relative,
        "v_boundary": metrics.boundary_violations,
        "v_grouping": metrics.grouping_violations,
        "v_mib": metrics.mib_violations,
        "feasible": metrics.is_feasible,
        "cost_no_rt": metrics.cost_no_runtime,
        "runtime": runtime,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-per-n", type=int, default=5)
    ap.add_argument("--n-min", type=int, default=21)
    ap.add_argument("--n-max", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--split", choices=["heldout", "train"], default="heldout")
    ap.add_argument("--heldout-mod", type=int, default=10)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap total cases (smoke testing)")
    ap.add_argument("--optimizer-path", type=str,
                     default=str(REPO / "src" / "solver" / "contest_optimizer.py"))
    ap.add_argument("--index-cache", type=str,
                     default=str(REPO / "artifacts" / "p0_newbox" / "lite_index_cache.json"))
    ap.add_argument("--mib-max-spread", type=float, default=0.02,
                     help="reject rows with MIB-group area spread above this "
                          "(hidden/validation-style homogeneity); 0 disables")
    args = ap.parse_args()

    load_avg_before = os.getloadavg()

    index = build_or_load_index(Path(args.index_cache), args.n_min, args.n_max,
                                 args.sample_per_n, args.seed)
    row_ok = (make_mib_spread_filter(args.mib_max_spread)
              if args.mib_max_spread and args.mib_max_spread > 0 else None)
    plan = sample_plan(index, args.n_min, args.n_max, args.sample_per_n,
                        args.seed, args.split, args.heldout_mod, row_ok=row_ok)
    if args.limit is not None:
        plan = plan[: args.limit]

    optimizer = load_optimizer(Path(args.optimizer_path))

    per_case = []
    costs_no_rt = []
    ns = []
    runtimes = []
    num_feasible = 0

    cache_path = None
    cache_data = None
    for item in plan:
        rel = item["file"]
        row = item["index"]
        n = item["n"]
        entry = {"id": item["instance_id"], "n": n, "file": rel, "index": row}
        try:
            path = str(LITE_ROOT / rel)
            if path != cache_path:
                cache_data = torch.load(path, map_location="cpu")
                cache_path = path
            case = BandFileSampler._instance(cache_data, row)
            result = evaluate_case(optimizer, case)
            entry.update(result)
        except Exception as exc:  # crash containment: never kill the run
            entry.update({
                "hpwl_gap": None, "area_gap": None, "v_rel": None,
                "v_boundary": None, "v_grouping": None, "v_mib": None,
                "feasible": False, "cost_no_rt": 10.0, "runtime": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            })
        per_case.append(entry)
        costs_no_rt.append(entry["cost_no_rt"])
        ns.append(n)
        if entry.get("runtime") is not None:
            runtimes.append(entry["runtime"])
        if entry.get("feasible"):
            num_feasible += 1

    total_score_no_runtime = ev.compute_total_score(costs_no_rt, ns) if costs_no_rt else None
    load_avg_after = os.getloadavg()

    out = {
        "tag": args.tag,
        "seed": args.seed,
        "sample_plan": {
            "sample_per_n": args.sample_per_n,
            "n_min": args.n_min,
            "n_max": args.n_max,
            "split": args.split,
            "heldout_mod": args.heldout_mod,
            "limit": args.limit,
            "cases": [{"instance_id": p["instance_id"], "file": p["file"],
                       "index": p["index"], "n": p["n"]} for p in plan],
        },
        "per_case": per_case,
        "total_score_no_runtime": total_score_no_runtime,
        "num_feasible": num_feasible,
        "num_cases": len(per_case),
        "avg_runtime": (sum(runtimes) / len(runtimes)) if runtimes else None,
        "load_avg_before": list(load_avg_before),
        "load_avg_after": list(load_avg_after),
        "optimizer_path": args.optimizer_path,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print(f"wrote {out_path}")
    print(f"cases={len(per_case)} feasible={num_feasible} "
          f"total_score_no_runtime={total_score_no_runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
