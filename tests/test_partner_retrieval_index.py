import json
from pathlib import Path

import numpy as np
import pytest

from retrieval_index_claude import RetrievalIndex, RetrievalShard, save_index


def _shard() -> RetrievalShard:
    return RetrievalShard(
        block_count=2,
        source_ids=np.array([10, 11], dtype=np.int64),
        global_features=np.array([[0.0] * 24, [1.0] * 24], dtype=np.float32),
        node_features=np.zeros((2, 2, 16), dtype=np.float32),
        fp_xywh=np.ones((2, 2, 4), dtype=np.float32),
        feature_mean=np.zeros(24, dtype=np.float32),
        feature_scale=np.ones(24, dtype=np.float32),
    )


def test_index_queries_same_n_and_excludes_self(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)

    index = RetrievalIndex.load(tmp_path)

    result = index.query(2, np.zeros(24, dtype=np.float32), top_k=1, exclude_source_id=10)

    assert result.source_ids.tolist() == [11]


def test_index_rejects_evaluation_manifest(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["source_split"] = "evaluation"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="train-only"):
        RetrievalIndex.load(tmp_path)


def test_save_rejects_non_train_source_split(tmp_path: Path):
    with pytest.raises(ValueError, match="train-only"):
        save_index(tmp_path, [_shard()], source_split="evaluation", feature_version=1)


def test_load_rejects_tampered_shard_file_hash(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)
    shard_path = tmp_path / "n_002.npz"
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="file SHA-256"):
        RetrievalIndex.load(tmp_path)


def test_load_rejects_mismatched_fp_xywh_array_hash(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["fp_xywh_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="fp_xywh SHA-256"):
        RetrievalIndex.load(tmp_path)


def test_query_keeps_distances_finite_for_extreme_feature_with_zero_scale(tmp_path: Path):
    shard = _shard()
    shard.global_features[0, 0] = np.finfo(np.float32).max
    shard.feature_scale[0] = 0.0
    save_index(tmp_path, [shard], source_split="train", feature_version=1)

    index = RetrievalIndex.load(tmp_path)
    query = np.zeros(24, dtype=np.float32)
    query[0] = np.finfo(np.float32).max
    result = index.query(2, query, top_k=2)

    assert np.isfinite(result.distances).all()


def test_query_missing_block_count_returns_correctly_shaped_empty_arrays(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)

    result = RetrievalIndex.load(tmp_path).query(3, np.zeros(24, dtype=np.float32), top_k=2)

    assert result.source_ids.shape == (0,)
    assert result.distances.shape == (0,)
    assert result.node_features.shape == (0, 3, 16)
    assert result.fp_xywh.shape == (0, 3, 4)


def test_query_preserves_source_order_for_all_zero_distance_ties(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)

    result = RetrievalIndex.load(tmp_path).query(2, np.zeros(24, dtype=np.float32), top_k=2)

    assert result.source_ids.tolist() == [10, 11]
