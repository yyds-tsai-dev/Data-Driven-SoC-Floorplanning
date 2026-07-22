import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_builder_module():
    path = REPO_ROOT / "scripts" / "build_partner_retrieval_index.py"
    spec = importlib.util.spec_from_file_location("build_partner_retrieval_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_probe_module():
    path = REPO_ROOT / "scripts" / "probes" / "retrieval_probe.py"
    spec = importlib.util.spec_from_file_location("retrieval_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])


def _sample(area, floorplan=None):
    block_count = len(area)
    constraints = np.zeros((block_count, 5), dtype=np.float32)
    inputs = (
        _FakeTensor(area),
        _FakeTensor([[-1, -1, -1]]),
        _FakeTensor([[-1, -1, -1]]),
        _FakeTensor([[0, 0]]),
        _FakeTensor(constraints),
    )
    if floorplan is None:
        floorplan = np.zeros((block_count, 4), dtype=np.float32)
    return {"input": inputs, "label": (_FakeTensor([]), _FakeTensor(floorplan), _FakeTensor([]))}


def test_retrieval_builder_help_is_noninteractive_and_exposes_required_flags():
    result = subprocess.run(
        [sys.executable, "scripts/build_partner_retrieval_index.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in ("--data-path", "--output", "--max-per-n", "--seed"):
        assert flag in result.stdout


def test_probe_help_is_noninteractive_and_exposes_required_flags():
    result = subprocess.run(
        [sys.executable, "scripts/probes/retrieval_probe.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in ("--index", "--data-path", "--cases", "--top-k", "--output"):
        assert flag in result.stdout


def test_builder_is_hard_coded_to_the_training_split_only():
    text = (REPO_ROOT / "scripts" / "build_partner_retrieval_index.py").read_text()

    assert "--source-split" not in text
    assert 'source_split="train"' in text
    assert "FloorplanDatasetLiteTest" not in text


def test_reservoir_sampling_is_deterministic_and_capped_per_block_count():
    builder = _load_builder_module()
    block_counts = [2] * 20 + [3] * 10 + [4]

    first = builder.reservoir_indices_by_block_count(block_counts, max_per_n=3, seed=17)
    second = builder.reservoir_indices_by_block_count(block_counts, max_per_n=3, seed=17)

    assert first == second
    assert {n: len(indices) for n, indices in first.items()} == {2: 3, 3: 3, 4: 1}
    assert all(0 <= index < 20 for index in first[2])
    assert all(20 <= index < 30 for index in first[3])
    assert first[4] == [30]


def test_atomic_publish_refuses_an_existing_empty_directory_without_touching_either_side(tmp_path):
    builder = _load_builder_module()
    source = tmp_path / "temporary-index"
    destination = tmp_path / "published-index"
    source.mkdir()
    destination.mkdir()
    (source / "manifest.json").write_text("source")

    with pytest.raises(FileExistsError):
        builder._publish_directory_no_replace(source, destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert (source / "manifest.json").read_text() == "source"


def test_builder_creates_train_only_artifact_with_converted_floorplans(monkeypatch, tmp_path):
    builder = _load_builder_module()
    dataset = [_sample([1.0, 4.0], [[10, 20, 3, 4], [30, 40, 5, 6]])]
    monkeypatch.setattr(builder, "FloorplanDatasetLite", lambda _path: dataset)
    output = tmp_path / "pilot"

    summary = builder.build_index(tmp_path, output, max_per_n=1, seed=17)

    assert summary["source_split"] == "train"
    assert json.loads((output / "manifest.json").read_text())["source_split"] == "train"
    shard = builder.RetrievalIndex.load(output).shards[2]
    assert shard.source_ids.tolist() == [0]
    np.testing.assert_array_equal(shard.fp_xywh[0], [[3, 4, 10, 20], [5, 6, 30, 40]])


def test_builder_keeps_existing_output_and_cleans_no_temporary_publish_directory(monkeypatch, tmp_path):
    builder = _load_builder_module()
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    monkeypatch.setattr(builder, "FloorplanDatasetLite", lambda _path: [_sample([1.0])])

    with pytest.raises(FileExistsError):
        builder.build_index(tmp_path, output, max_per_n=1, seed=17)

    assert (output / "sentinel").read_text() == "keep"
    assert list(tmp_path.glob(".pilot.tmp-*")) == []


def test_probe_returns_complete_matched_and_no_same_n_rows_without_mutating_index(monkeypatch, tmp_path):
    probe = _load_probe_module()
    index_path = tmp_path / "index"
    index_path.mkdir()
    manifest = index_path / "manifest.json"
    manifest.write_text("read-only")
    dataset = [_sample([1.0, 1.0]), _sample([1.0, 1.0, 1.0])]

    class FakeIndex:
        def query(self, block_count, _vector, top_k):
            if block_count == 3:
                return SimpleNamespace(
                    source_ids=np.empty(0, dtype=np.int64),
                    distances=np.empty(0, dtype=np.float64),
                    node_features=np.empty((0, 3, 16), dtype=np.float32),
                    fp_xywh=np.empty((0, 3, 4), dtype=np.float32),
                )
            return SimpleNamespace(
                source_ids=np.array([7], dtype=np.int64),
                distances=np.array([0.25]),
                node_features=np.zeros((1, 2, 16), dtype=np.float32),
                fp_xywh=np.array([[[0, 0, 1, 1], [2, 0, 1, 1]]], dtype=np.float32),
            )

    monkeypatch.setattr(probe.RetrievalIndex, "load", lambda _path: FakeIndex())
    monkeypatch.setattr(probe, "FloorplanDatasetLiteTest", lambda _path: dataset)
    monkeypatch.setattr(
        probe,
        "extract_retrieval_features",
        lambda area, *_args: SimpleNamespace(
            global_vector=np.zeros(24, dtype=np.float32),
            node_matrix=np.zeros((len(area), 16), dtype=np.float32),
        ),
    )

    rows = probe.run_probe(index_path, tmp_path, cases=2, top_k=1)

    expected_fields = {
        "case_id", "n", "retrieved_source_ids", "retrieved_distances",
        "match_cost", "match_confidence", "raw_overlap_proxy", "raw_hpwl_proxy",
        "transform", "cold_index_load_s", "warm_query_s", "matching_s", "transfer_s", "status",
    }
    assert rows[0]["status"] == "matched"
    assert rows[1]["status"] == "no_same_n_source"
    assert all(expected_fields <= row.keys() for row in rows)
    assert manifest.read_text() == "read-only"
