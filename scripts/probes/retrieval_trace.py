#!/usr/bin/env python3
"""Read-only candidate-provenance trace for the bounded R4 retrieval portfolio.

This is a diagnostic, not a full solver replay.  It compares the production
forced source quota with a source-neutral prefix of the one shared rank.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (
    REPO_ROOT / "FloorSet" / "iccad2026contest",
    REPO_ROOT / "FloorSet",
    REPO_ROOT / "partner",
    REPO_ROOT / "scripts" / "probes",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from candidate_supply_claude import CandidateBatch
from iccad2026_evaluate import evaluate_solution
from legalizer_claude import _POOL_SIZE, _worker_refine
from lite_dataset_test import FloorplanDatasetLiteTest
from my_opt_claude import FIRST_R4_RETRIEVAL_SLOTS, MyOptimizer, _select_ranked_source_quota


def parse_case_ids(value: str) -> list[int]:
    """Parse explicitly requested, ordered evaluation case IDs."""
    if not value.strip():
        raise ValueError("--case-ids must not be empty")
    try:
        case_ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--case-ids must be comma-separated integers") from error
    if any(case_id < 0 for case_id in case_ids):
        raise ValueError("--case-ids must be non-negative")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("--case-ids must not contain duplicates")
    return case_ids


def _selection_k(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--selection-k must be auto or a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--selection-k must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-ids", required=True, type=parse_case_ids)
    parser.add_argument("--selection-k", default="auto", type=_selection_k)
    parser.add_argument("--refine-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "FloorSet")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if not math.isfinite(args.refine_seconds) or args.refine_seconds <= 0.0:
        parser.error("--refine-seconds must be finite and positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_contract(index_path: Path) -> dict[str, object]:
    """Fingerprint only the existing index inputs; never open them for write."""
    manifest = index_path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing retrieval index manifest: {manifest}")
    shard_paths = sorted(index_path.glob("n_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"retrieval index has no shards: {index_path}")
    return {
        "path": str(index_path.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "shards": {path.name: sha256_file(path) for path in shard_paths},
    }


def json_safe(value: Any) -> Any:
    """Convert probe values to deterministic JSON primitives without NaNs."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def resolve_selection_k(block_count: int, selection_k: str | int) -> int:
    """Mirror legalizer_claude's active ``n_ref`` capacity rule."""
    if selection_k != "auto":
        return int(selection_k)
    resolved = min(8, max(3, _POOL_SIZE // 3))
    try:
        requested = int(float(os.environ.get("PARTNER_NREF", "0") or 0))
        min_n = int(float(os.environ.get("PARTNER_NREF_MIN_N", "95")))
    except ValueError:
        requested, min_n = 0, 95
    if requested > 0 and block_count >= min_n:
        resolved = max(3, min(requested, _POOL_SIZE - 6))
    return resolved


def stable_candidate_id(
    case_id: int, source: str, source_position: int, retrieval_source_id: int | None,
) -> str:
    source_token = "none" if retrieval_source_id is None else str(int(retrieval_source_id))
    return f"case-{case_id}:{source}:{source_position}:source-{source_token}"


def refinement_seed(seed: int, candidate_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{candidate_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def select_policy_candidate_indexes(
    predictions: list[np.ndarray], sources: list[str], order: list[int],
    *, selection_k: int, retrieval_quota: int,
) -> dict[str, list[int]]:
    """Apply both policies to exactly one shared source-neutral rank."""
    forced_predictions = _select_ranked_source_quota(
        predictions, sources, order, selection_k, retrieval_quota,
    )
    indexes_by_identity = {id(prediction): index for index, prediction in enumerate(predictions)}
    try:
        forced_indexes = [indexes_by_identity[id(prediction)] for prediction in forced_predictions]
    except KeyError as error:
        raise RuntimeError("production quota selector returned a non-union candidate") from error
    return {
        "forced_quota": forced_indexes,
        "maximum_quota": list(order[:selection_k]),
    }


def _score_layout(sample: dict[str, object], positions: np.ndarray, n: int) -> dict[str, object]:
    from gen_decoder_probe import _baseline, _golden_rects

    at, b2b, p2b, pins, constraints = sample["input"]
    baseline = _baseline(sample, n, b2b, p2b, pins)
    golden = _golden_rects(sample, n)
    metrics = evaluate_solution(
        {"positions": [tuple(map(float, row)) for row in positions], "runtime": 1.0},
        baseline, constraints, b2b, p2b, pins, at[:n], golden, median_runtime=1.0,
    )
    return {
        "cost_no_runtime": float(metrics.cost_no_runtime),
        "is_feasible": bool(metrics.is_feasible),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "violations_relative": float(metrics.violations_relative),
    }


def refine_union_candidates(
    candidates: list[dict[str, object]], areas: np.ndarray, constraints: np.ndarray,
    target_positions: np.ndarray, b2b: np.ndarray, p2b: np.ndarray, pins: np.ndarray,
    *, refine_seconds: float, seed: int,
) -> None:
    """Give every union member equal independent worker allowance and trace failures."""
    for candidate in candidates:
        candidate_seed = refinement_seed(seed, str(candidate["candidate_id"]))
        deadline = time.time() + refine_seconds
        payload = (
            np.asarray(candidate["prediction"], dtype=np.float64), areas, constraints,
            target_positions, b2b, p2b, pins, deadline, candidate_seed, 1.0, False,
        )
        candidate["refinement_seed"] = candidate_seed
        candidate["refine_seconds"] = refine_seconds
        try:
            result = _worker_refine(payload)
        except Exception as error:  # defensive: worker failures must remain visible
            candidate["refinement_failure"] = json_safe(error)
            candidate["refined_prediction"] = None
            continue
        if result is None:
            candidate["refinement_failure"] = "worker_returned_none"
            candidate["refined_prediction"] = None
            continue
        candidate["refined_prediction"] = np.asarray(result[0], dtype=np.float64)
        candidate["worker_metrics"] = {
            "hpwl": float(result[1]), "bbox_area": float(result[2]), "violations": int(result[3]),
        }


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, prior in old.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _retrieval_metadata(batch: CandidateBatch, position: int) -> dict[str, object]:
    metadata = batch.metadata
    return {
        "metadata_scope": "chosen_only",
        "source_id": metadata.get("source_ids", [])[position],
        "distance": metadata.get("retrieval_distances", [])[position],
        "transform": metadata.get("transforms", [])[position],
        "match_cost": metadata.get("match_costs", [])[position],
        "match_confidence": metadata.get("match_confidences", [])[position],
    }


def _policy_summary(policy_indexes: list[int], candidates: list[dict[str, object]]) -> dict[str, object]:
    selected = [candidates[index] for index in policy_indexes]
    ranked = min(selected, key=lambda item: int(item["rank_position"])) if selected else None
    raw = min(selected, key=lambda item: float(item["raw_score"]["cost_no_runtime"])) if selected else None
    refined = [item for item in selected if item.get("refined_score") is not None]
    refined_winner = min(
        refined, key=lambda item: float(item["refined_score"]["cost_no_runtime"]),
    ) if refined else None
    return {
        "candidate_ids": [str(item["candidate_id"]) for item in selected],
        "candidate_path_proxy_winner_id": None if ranked is None else ranked["candidate_id"],
        "official_no_runtime_raw_winner": None if raw is None else {
            "candidate_id": raw["candidate_id"], "score": raw["raw_score"],
        },
        "official_no_runtime_refined_winner": None if refined_winner is None else {
            "candidate_id": refined_winner["candidate_id"], "score": refined_winner["refined_score"],
        },
    }


def _case_trace(
    optimizer: MyOptimizer, sample: dict[str, object], case_id: int, *, selection_k: str | int,
    refine_seconds: float, seed: int,
) -> dict[str, object]:
    from gen_decoder_probe import _opt_target_positions

    at, b2b, p2b, pins, constraints = sample["input"]
    n = int((at != -1).sum().item())
    if n <= 0:
        raise ValueError("evaluation sample has no active blocks")
    target_positions = _opt_target_positions(sample, n)
    resolved_k = resolve_selection_k(n, selection_k)
    quota = min(resolved_k, optimizer.retrieval_slots, FIRST_R4_RETRIEVAL_SLOTS)
    direct = optimizer._sample_direct_raw_preds(
        n, at, constraints, target_positions, b2b, p2b, pins, resolved_k, oversample=True,
    )
    retrieval = optimizer._sample_retrieval_preds(
        n, at, constraints, target_positions, b2b, p2b, pins, quota,
    )
    predictions = direct + retrieval.predictions
    sources = ["direct"] * len(direct) + ["retrieval"] * len(retrieval.predictions)
    order = optimizer._rank_portfolio(predictions, n, at, constraints, b2b)
    policies = select_policy_candidate_indexes(
        predictions, sources, order, selection_k=resolved_k, retrieval_quota=quota,
    )
    rank_position = {candidate_index: position for position, candidate_index in enumerate(order)}
    candidates: list[dict[str, object]] = []
    for index, (prediction, source) in enumerate(zip(predictions, sources)):
        source_position = index - len(direct) if source == "retrieval" else index
        retrieval_info = _retrieval_metadata(retrieval, source_position) if source == "retrieval" else None
        source_id = None if retrieval_info is None else int(retrieval_info["source_id"])
        candidate = {
            "candidate_id": stable_candidate_id(case_id, source, source_position, source_id),
            "source": source,
            "source_position": source_position,
            "rank_position": rank_position[index],
            "prediction": np.asarray(prediction, dtype=np.float64),
            "retrieval": retrieval_info,
        }
        candidate["raw_score"] = _score_layout(sample, candidate["prediction"], n)
        candidates.append(candidate)
    union_indexes = sorted({index for indexes in policies.values() for index in indexes})
    union_candidates = [candidates[index] for index in union_indexes]
    arrays = tuple(t[:n].detach().cpu().numpy() for t in (at, constraints, target_positions))
    refine_union_candidates(
        union_candidates, arrays[0], arrays[1], arrays[2],
        b2b.detach().cpu().numpy(), p2b.detach().cpu().numpy(), pins.detach().cpu().numpy(),
        refine_seconds=refine_seconds, seed=seed,
    )
    for candidate in union_candidates:
        refined = candidate.get("refined_prediction")
        if refined is not None:
            candidate["refined_score"] = _score_layout(sample, np.asarray(refined), n)
    return {
        "case_id": case_id,
        "block_count": n,
        "resolved_selection_k": resolved_k,
        "retrieval_quota": quota,
        "generation": {
            "direct_count": len(direct),
            "retrieval_count": len(retrieval.predictions),
            "retrieval_generation_s": retrieval.generation_s,
            "retrieval_metadata": retrieval.metadata,
            "retrieval_metadata_scope": "chosen_only",
        },
        "shared_rank": order,
        "policies": {name: _policy_summary(indexes, candidates) for name, indexes in policies.items()},
        "candidates": candidates,
    }


def run_trace(args: argparse.Namespace) -> dict[str, object]:
    index_path = args.index.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    before_contract = index_contract(index_path)
    started = time.perf_counter()
    env = {
        "PARTNER_RETRIEVAL_INDEX": str(index_path),
        "PARTNER_RETRIEVAL_SLOTS": str(FIRST_R4_RETRIEVAL_SLOTS),
        "DIRECT_CKPT": str(checkpoint_path),
    }
    with _temporary_environment(env):
        optimizer = MyOptimizer(checkpoint_path=str(checkpoint_path), device=args.device)
        if optimizer.retrieval_index is None:
            raise RuntimeError("production optimizer did not load the requested retrieval index")
        runtime_environment = {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(("PARTNER_", "DIRECT_", "CUDA_", "FLOORSET_"))
        }
        dataset = FloorplanDatasetLiteTest(str(args.data_path))
        cases = []
        for case_id in args.case_ids:
            if case_id >= len(dataset):
                raise IndexError(f"case ID {case_id} is outside evaluation dataset length {len(dataset)}")
            cases.append(_case_trace(
                optimizer, dataset[case_id], case_id, selection_k=args.selection_k,
                refine_seconds=args.refine_seconds, seed=args.seed,
            ))
    after_contract = index_contract(index_path)
    if after_contract != before_contract:
        raise RuntimeError("read-only index contract changed during retrieval trace")
    return {
        "probe": "retrieval_r4_candidate_provenance",
        "interpretation_boundary": (
            "candidate-path diagnostic only; excludes concurrent column restarts, shared absolute "
            "deadline effects, fusion, phase-B, and vkill"
        ),
        "run_contract": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "index": before_contract,
            "case_ids": args.case_ids,
            "selection_k": args.selection_k,
            "seed": args.seed,
            "refine_seconds": args.refine_seconds,
            "device": str(optimizer.device),
            "environment": runtime_environment,
            "python": sys.version,
            "platform": platform.platform(),
            "wall_time_s": time.perf_counter() - started,
        },
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = json_safe(run_trace(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
