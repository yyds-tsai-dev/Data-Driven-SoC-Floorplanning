"""Train-only, integrity-checked retrieval shards.

The index deliberately stores only training examples.  It keeps one shard per
block count so retrieval never mixes layouts with incompatible node counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


_ARRAY_KEYS = (
    "source_ids",
    "global_features",
    "node_features",
    "fp_xywh",
    "feature_mean",
    "feature_scale",
)
_FEATURE_DIM = 24
_NODE_FEATURE_DIM = 16
_FP_DIM = 4


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_array(
    array: np.ndarray,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class RetrievalShard:
    block_count: int
    source_ids: np.ndarray
    global_features: np.ndarray
    node_features: np.ndarray
    fp_xywh: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def __post_init__(self) -> None:
        block_count = _require_int(self.block_count, "block_count", minimum=1)
        count = int(self.source_ids.shape[0]) if isinstance(self.source_ids, np.ndarray) else -1
        if count < 0:
            raise ValueError("source_ids must be a numpy array")
        _require_array(self.source_ids, "source_ids", (count,), np.dtype(np.int64))
        _require_array(
            self.global_features,
            "global_features",
            (count, _FEATURE_DIM),
            np.dtype(np.float32),
        )
        _require_array(
            self.node_features,
            "node_features",
            (count, block_count, _NODE_FEATURE_DIM),
            np.dtype(np.float32),
        )
        _require_array(
            self.fp_xywh,
            "fp_xywh",
            (count, block_count, _FP_DIM),
            np.dtype(np.float32),
        )
        _require_array(
            self.feature_mean,
            "feature_mean",
            (_FEATURE_DIM,),
            np.dtype(np.float32),
        )
        _require_array(
            self.feature_scale,
            "feature_scale",
            (_FEATURE_DIM,),
            np.dtype(np.float32),
        )
        if np.any(self.feature_scale < 0):
            raise ValueError("feature_scale must be non-negative")


@dataclass(frozen=True)
class RetrievalResult:
    source_ids: np.ndarray
    distances: np.ndarray
    node_features: np.ndarray
    fp_xywh: np.ndarray


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_path(root: Path, block_count: int) -> Path:
    return root / f"n_{block_count:03d}.npz"


def _normalized(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    features64 = np.asarray(features, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64)
    scale64 = np.asarray(scale, dtype=np.float64)
    z_score = (features64 - mean64) / np.maximum(scale64, 1e-6)
    normalized = z_score / np.maximum(np.linalg.norm(z_score, axis=-1, keepdims=True), 1e-9)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized retrieval features must be finite")
    return normalized


def _manifest_entry(shard: RetrievalShard, root: Path) -> dict[str, object]:
    path = _shard_path(root, shard.block_count)
    return {
        "n": shard.block_count,
        "count": int(shard.source_ids.shape[0]),
        "file_sha256": _sha256_file(path),
        "fp_xywh_sha256": _sha256_array(shard.fp_xywh),
    }


def save_index(
    root: Path,
    shards: Iterable[RetrievalShard],
    *,
    source_split: str,
    feature_version: int,
) -> None:
    """Save validated training shards without pickle-backed data."""
    if source_split != "train":
        raise ValueError("retrieval index is train-only")
    _require_int(feature_version, "feature_version", minimum=0)

    shard_list = list(shards)
    seen_block_counts: set[int] = set()
    for shard in shard_list:
        if not isinstance(shard, RetrievalShard):
            raise ValueError("shards must contain RetrievalShard values")
        shard.__post_init__()
        if shard.block_count in seen_block_counts:
            raise ValueError("only one shard is allowed for each block_count")
        seen_block_counts.add(shard.block_count)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for shard in shard_list:
        np.savez_compressed(
            _shard_path(root, shard.block_count),
            source_ids=shard.source_ids,
            global_features=shard.global_features,
            node_features=shard.node_features,
            fp_xywh=shard.fp_xywh,
            feature_mean=shard.feature_mean,
            feature_scale=shard.feature_scale,
        )

    manifest = {
        "source_split": "train",
        "feature_version": int(feature_version),
        "shards": [_manifest_entry(shard, root) for shard in shard_list],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _read_manifest(path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid retrieval index manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError("retrieval index manifest must be an object")
    return manifest


def _verified_shard(root: Path, entry: object) -> RetrievalShard:
    if not isinstance(entry, dict):
        raise ValueError("retrieval index shard entry must be an object")
    expected_fields = {"n", "count", "file_sha256", "fp_xywh_sha256"}
    if set(entry) != expected_fields:
        raise ValueError("retrieval index shard entry has invalid fields")
    block_count = _require_int(entry["n"], "manifest shard n", minimum=1)
    count = _require_int(entry["count"], "manifest shard count", minimum=0)
    file_hash = entry["file_sha256"]
    array_hash = entry["fp_xywh_sha256"]
    if not isinstance(file_hash, str) or not isinstance(array_hash, str):
        raise ValueError("retrieval index shard SHA-256 values must be strings")
    path = _shard_path(root, block_count)
    try:
        actual_file_hash = _sha256_file(path)
    except OSError as error:
        raise ValueError(f"missing retrieval index shard for n={block_count}") from error
    if actual_file_hash != file_hash:
        raise ValueError(f"retrieval index file SHA-256 mismatch for n={block_count}")

    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(_ARRAY_KEYS):
                raise ValueError(f"retrieval index shard n={block_count} has invalid array keys")
            arrays = {key: stored[key] for key in _ARRAY_KEYS}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "invalid array keys" in str(error):
            raise
        raise ValueError(f"invalid retrieval index shard for n={block_count}") from error

    if _sha256_array(arrays["fp_xywh"]) != array_hash:
        raise ValueError(f"retrieval index fp_xywh SHA-256 mismatch for n={block_count}")
    if arrays["source_ids"].shape != (count,):
        raise ValueError(f"retrieval index shard count mismatch for n={block_count}")
    return RetrievalShard(block_count=block_count, **arrays)


class RetrievalIndex:
    def __init__(self, shards: Mapping[int, RetrievalShard]) -> None:
        self.shards = dict(shards)
        for block_count, shard in self.shards.items():
            if block_count != shard.block_count:
                raise ValueError("retrieval shard key must match block_count")

    @classmethod
    def load(cls, root: Path) -> "RetrievalIndex":
        root = Path(root)
        manifest = _read_manifest(root / "manifest.json")
        if manifest.get("source_split") != "train":
            raise ValueError("retrieval index is train-only")
        _require_int(manifest.get("feature_version"), "manifest feature_version", minimum=0)
        entries = manifest.get("shards")
        if not isinstance(entries, list):
            raise ValueError("retrieval index manifest shards must be a list")

        shards: dict[int, RetrievalShard] = {}
        for entry in entries:
            shard = _verified_shard(root, entry)
            if shard.block_count in shards:
                raise ValueError("retrieval index manifest has duplicate block counts")
            shards[shard.block_count] = shard
        return cls(shards)

    def query(
        self,
        block_count: int,
        global_vector: np.ndarray,
        top_k: int,
        exclude_source_id: int | None = None,
    ) -> RetrievalResult:
        block_count = _require_int(block_count, "block_count", minimum=1)
        top_k = _require_int(top_k, "top_k", minimum=0)
        shard = self.shards.get(block_count)
        if shard is None:
            return RetrievalResult(
                source_ids=np.empty(0, dtype=np.int64),
                distances=np.empty(0, dtype=np.float64),
                node_features=np.empty((0, block_count, _NODE_FEATURE_DIM), dtype=np.float32),
                fp_xywh=np.empty((0, block_count, _FP_DIM), dtype=np.float32),
            )

        query_vector = np.asarray(global_vector, dtype=np.float64)
        if query_vector.shape != (_FEATURE_DIM,):
            raise ValueError(f"global_vector must have shape ({_FEATURE_DIM},)")
        if not np.isfinite(query_vector).all():
            raise ValueError("global_vector must be finite")
        query = _normalized(query_vector[None], shard.feature_mean, shard.feature_scale)[0]
        keys = _normalized(shard.global_features, shard.feature_mean, shard.feature_scale)
        distances = 1.0 - keys @ query
        if not np.isfinite(distances).all():
            raise ValueError("retrieval query distances must be finite")
        order = np.argsort(distances, kind="stable")
        if exclude_source_id is not None:
            order = order[shard.source_ids[order] != exclude_source_id]
        order = order[:top_k]
        return RetrievalResult(
            source_ids=shard.source_ids[order],
            distances=distances[order],
            node_features=shard.node_features[order],
            fp_xywh=shard.fp_xywh[order],
        )
