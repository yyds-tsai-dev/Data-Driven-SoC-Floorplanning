#!/usr/bin/env python3
"""Build a bounded, train-only retrieval index for offline experiments."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT / "FloorSet", REPO_ROOT / "partner"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from lite_dataset import FloorplanDatasetLite
from retrieval_features_claude import extract_retrieval_features
from retrieval_index_claude import RetrievalIndex, RetrievalShard, save_index


FEATURE_VERSION = 1
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("FloorSet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def reservoir_indices_by_block_count(
    block_counts: Iterable[int], *, max_per_n: int, seed: int
) -> dict[int, list[int]]:
    """Choose at most ``max_per_n`` deterministic uniform source ids per N."""
    if max_per_n <= 0:
        raise ValueError("--max-per-n must be positive")

    samples: dict[int, list[int]] = {}
    seen: dict[int, int] = {}
    rngs: dict[int, np.random.Generator] = {}
    for source_id, raw_n in enumerate(block_counts):
        block_count = int(raw_n)
        if block_count <= 0:
            raise ValueError(f"training source {source_id} has invalid block count {block_count}")
        bucket = samples.setdefault(block_count, [])
        count = seen.get(block_count, 0) + 1
        seen[block_count] = count
        if len(bucket) < max_per_n:
            bucket.append(source_id)
            continue
        rng = rngs.setdefault(block_count, np.random.default_rng([int(seed), block_count]))
        replacement = int(rng.integers(0, count))
        if replacement < max_per_n:
            bucket[replacement] = source_id
    return samples


def _publish_directory_no_replace(temporary: Path, output: Path) -> None:
    """Atomically publish a sibling directory only when its destination is absent.

    Linux's ``renameat2(RENAME_NOREPLACE)`` is required here because ordinary
    ``rename`` can replace an empty directory after a check-then-rename race.
    Failing closed is preferable to risking an existing index on platforms that
    do not expose the no-replace primitive.
    """
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(temporary),
        _AT_FDCWD,
        os.fsencode(output),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, f"refusing to overwrite existing retrieval index: {output}")
    raise OSError(error_number, f"atomic no-replace publication failed for {output}")


def _sample_arrays(
    sample: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    area, b2b, p2b, pins, constraints = [tensor.cpu().numpy() for tensor in sample["input"]]
    _tree, fp_sol, _metrics = sample["label"]
    n = int((area != -1).sum())
    if n <= 0:
        raise ValueError("training sample has no active blocks")
    fp = fp_sol[:n].cpu().numpy()
    fp_xywh = fp[:, [2, 3, 0, 1]].astype(np.float32)
    target_positions = np.full((n, 4), -1.0, dtype=np.float32)
    active_constraints = constraints[:n]
    fixed = active_constraints[:, 0] != 0
    preplaced = active_constraints[:, 1] != 0
    target_positions[fixed | preplaced, 2:4] = fp_xywh[fixed | preplaced, 2:4]
    target_positions[preplaced, 0:2] = fp_xywh[preplaced, 0:2]
    return area[:n], b2b, p2b, pins, active_constraints, target_positions, fp_xywh


def _block_counts(dataset: FloorplanDatasetLite) -> Iterable[int]:
    for source_id in range(len(dataset)):
        area = dataset[source_id]["input"][0].cpu().numpy()
        yield int((area != -1).sum())


def _build_shards(dataset: FloorplanDatasetLite, selected: dict[int, list[int]]) -> list[RetrievalShard]:
    shards: list[RetrievalShard] = []
    for block_count in sorted(selected):
        source_ids = np.asarray(sorted(selected[block_count]), dtype=np.int64)
        global_features: list[np.ndarray] = []
        node_features: list[np.ndarray] = []
        floorplans: list[np.ndarray] = []
        for source_id in source_ids:
            area, b2b, p2b, pins, constraints, target_positions, fp_xywh = _sample_arrays(
                dataset[int(source_id)]
            )
            if len(area) != block_count:
                raise ValueError(f"training source {source_id} changed block count between passes")
            features = extract_retrieval_features(
                area, b2b, p2b, pins, constraints, target_positions
            )
            global_features.append(features.global_vector)
            node_features.append(features.node_matrix)
            floorplans.append(fp_xywh)
        globals_array = np.stack(global_features).astype(np.float32, copy=False)
        feature_mean = globals_array.mean(axis=0, dtype=np.float64).astype(np.float32)
        feature_scale = globals_array.std(axis=0, dtype=np.float64).astype(np.float32)
        shards.append(
            RetrievalShard(
                block_count=block_count,
                source_ids=source_ids,
                global_features=globals_array,
                node_features=np.stack(node_features).astype(np.float32, copy=False),
                fp_xywh=np.stack(floorplans).astype(np.float32, copy=False),
                feature_mean=feature_mean,
                feature_scale=feature_scale,
            )
        )
    return shards


def build_index(data_path: Path, output: Path, *, max_per_n: int, seed: int) -> dict[str, object]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing retrieval index: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = FloorplanDatasetLite(str(data_path))
    selected = reservoir_indices_by_block_count(_block_counts(dataset), max_per_n=max_per_n, seed=seed)
    shards = _build_shards(dataset, selected)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        save_index(temporary, shards, source_split="train", feature_version=FEATURE_VERSION)
        RetrievalIndex.load(temporary)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing retrieval index: {output}")
        _publish_directory_no_replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "output": str(output),
        "source_split": "train",
        "feature_version": FEATURE_VERSION,
        "shard_counts": {str(shard.block_count): int(len(shard.source_ids)) for shard in shards},
    }


def main() -> None:
    args = parse_args()
    summary = build_index(args.data_path, args.output, max_per_n=args.max_per_n, seed=args.seed)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
