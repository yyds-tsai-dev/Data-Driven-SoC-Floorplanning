#!/usr/bin/env python3
"""Read-only retrieval diagnostics on evaluation layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT / "FloorSet", REPO_ROOT / "partner"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from lite_dataset_test import FloorplanDatasetLiteTest
from retrieval_features_claude import extract_retrieval_features
from retrieval_index_claude import RetrievalIndex
from retrieval_matching_claude import MatchResult, match_blocks
from retrieval_transfer_claude import remap_boundary_node_features, transfer_layout


MATCH_MAX_COST = 2.0
TRANSFORMS = ("identity", "mirror_x", "mirror_y", "transpose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("FloorSet"))
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _target_arrays(sample: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    area, b2b, p2b, pins, constraints = [tensor.cpu().numpy() for tensor in sample["input"]]
    n = int((area != -1).sum())
    if n <= 0:
        raise ValueError("evaluation sample has no active blocks")
    polygons, _metrics = sample["label"]
    polygon_array = polygons.cpu().numpy()
    target_positions = np.full((n, 4), -1.0, dtype=np.float32)
    for block_index in range(n):
        block = np.asarray(polygon_array[block_index], dtype=np.float32)
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            x, y, width, height = 0.0, 0.0, 1.0, 1.0
        else:
            minimum = valid.min(axis=0)
            maximum = valid.max(axis=0)
            x, y = float(minimum[0]), float(minimum[1])
            width, height = float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])
        fixed = constraints[block_index, 0] != 0 if constraints.shape[1] > 0 else False
        preplaced = constraints[block_index, 1] != 0 if constraints.shape[1] > 1 else False
        if preplaced:
            target_positions[block_index] = (x, y, width, height)
        elif fixed:
            target_positions[block_index, 2:4] = (width, height)
    return area[:n], b2b, p2b, pins, constraints[:n], target_positions


def _overlap_proxy(rectangles: np.ndarray) -> float:
    x0, y0 = rectangles[:, 0], rectangles[:, 1]
    x1, y1 = x0 + rectangles[:, 2], y0 + rectangles[:, 3]
    overlap_x = np.maximum(0.0, np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0))
    overlap_y = np.maximum(0.0, np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0))
    return float(np.triu(overlap_x * overlap_y, k=1).sum())


def _hpwl_proxy(rectangles: np.ndarray, b2b: np.ndarray, p2b: np.ndarray, pins: np.ndarray) -> float:
    centers = rectangles[:, :2] + 0.5 * rectangles[:, 2:4]
    total = 0.0
    for edge in np.asarray(b2b):
        if edge[0] < 0:
            continue
        left, right, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if 0 <= left < len(centers) and 0 <= right < len(centers):
            total += weight * float(np.abs(centers[left] - centers[right]).sum())
    for edge in np.asarray(p2b):
        if edge[0] < 0:
            continue
        pin_index, block_index, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if 0 <= pin_index < len(pins) and 0 <= block_index < len(centers):
            total += weight * float(np.abs(np.asarray(pins[pin_index, :2]) - centers[block_index]).sum())
    return total


def _best_match(retrieved, target_nodes: np.ndarray) -> tuple[int | None, str | None, MatchResult | None]:
    best: tuple[float, int, str, MatchResult] | None = None
    for source_position in range(len(retrieved.source_ids)):
        for transform in TRANSFORMS:
            source_nodes = remap_boundary_node_features(retrieved.node_features[source_position], transform)
            result = match_blocks(source_nodes, target_nodes, max_cost=MATCH_MAX_COST)
            if result.accepted:
                candidate = (result.total_cost, source_position, transform, result)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
    if best is None:
        return None, None, None
    _cost, source_position, transform, result = best
    return source_position, transform, result


def _base_row(case_id: int, block_count: int, load_s: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "n": block_count,
        "retrieved_source_ids": [],
        "retrieved_distances": [],
        "match_cost": None,
        "match_confidence": None,
        "raw_overlap_proxy": None,
        "raw_hpwl_proxy": None,
        "transform": None,
        "cold_index_load_s": load_s,
        "warm_query_s": 0.0,
        "matching_s": 0.0,
        "transfer_s": 0.0,
    }


def run_probe(index_path: Path, data_path: Path, *, cases: int, top_k: int) -> list[dict[str, object]]:
    if cases < 0 or top_k < 0:
        raise ValueError("--cases and --top-k must be non-negative")
    load_started = time.perf_counter()
    index = RetrievalIndex.load(index_path)
    cold_index_load_s = time.perf_counter() - load_started
    dataset = FloorplanDatasetLiteTest(str(data_path))

    rows: list[dict[str, object]] = []
    for case_id in range(min(cases, len(dataset))):
        block_count = 0
        try:
            area, b2b, p2b, pins, constraints, target_positions = _target_arrays(dataset[case_id])
            block_count = len(area)
            row = _base_row(case_id, block_count, cold_index_load_s)
            target_features = extract_retrieval_features(
                area, b2b, p2b, pins, constraints, target_positions
            )
            query_started = time.perf_counter()
            retrieved = index.query(block_count, target_features.global_vector, top_k=top_k)
            row["warm_query_s"] = time.perf_counter() - query_started
            row["retrieved_source_ids"] = retrieved.source_ids.astype(int).tolist()
            row["retrieved_distances"] = retrieved.distances.astype(float).tolist()
            if len(retrieved.source_ids) == 0:
                row["status"] = "no_same_n_source"
                rows.append(row)
                continue

            matching_started = time.perf_counter()
            source_position, transform, match = _best_match(retrieved, target_features.node_matrix)
            row["matching_s"] = time.perf_counter() - matching_started
            if match is None or source_position is None or transform is None:
                row["status"] = "no_compatible_match"
                rows.append(row)
                continue

            row["match_cost"] = float(match.total_cost)
            row["match_confidence"] = float(match.confidence)
            row["transform"] = transform
            transfer_started = time.perf_counter()
            transferred = transfer_layout(
                retrieved.fp_xywh[source_position],
                match.target_to_source,
                area,
                constraints,
                target_positions,
                transform,
            )
            row["transfer_s"] = time.perf_counter() - transfer_started
            row["raw_overlap_proxy"] = _overlap_proxy(transferred)
            row["raw_hpwl_proxy"] = _hpwl_proxy(transferred, b2b, p2b, pins)
            row["status"] = "matched"
            rows.append(row)
        except (IndexError, ValueError, OSError) as error:
            row = _base_row(case_id, block_count, cold_index_load_s)
            row["status"] = "unsupported"
            row["error"] = str(error)
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = run_probe(args.index, args.data_path, cases=args.cases, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
