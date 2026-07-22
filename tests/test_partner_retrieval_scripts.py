import importlib.util
import errno
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


def _evaluation_sample(area, constraints=None, polygons=None):
    active_count = int(np.count_nonzero(np.asarray(area) != -1))
    if constraints is None:
        constraints = np.zeros((len(area), 5), dtype=np.float32)
    if polygons is None:
        polygons = np.zeros((active_count, 1, 2), dtype=np.float32)
    inputs = (
        _FakeTensor(area),
        _FakeTensor([[-1, -1, -1]]),
        _FakeTensor([[-1, -1, -1]]),
        _FakeTensor([[0, 0]]),
        _FakeTensor(constraints),
    )
    return {"input": inputs, "label": (_FakeTensor(polygons), _FakeTensor([]))}


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


def _unsupported_renameat2(*_args):
    raise OSError(errno.EINVAL, "renameat2 unsupported by this filesystem")


def test_atomic_publish_falls_back_to_a_sibling_symlink_when_renameat2_is_unsupported(
    monkeypatch, tmp_path
):
    builder = _load_builder_module()
    source = tmp_path / ".temporary-index"
    destination = tmp_path / "published-index"
    source.mkdir()
    (source / "manifest.json").write_text("source")
    monkeypatch.setattr(builder, "_renameat2_no_replace", _unsupported_renameat2)

    retained = builder._publish_directory_no_replace(source, destination)

    assert retained
    assert destination.is_symlink()
    assert destination.readlink() == Path(source.name)
    assert (destination / "manifest.json").read_text() == "source"


def test_symlink_fallback_preserves_an_existing_dangling_destination(monkeypatch, tmp_path):
    builder = _load_builder_module()
    source = tmp_path / ".temporary-index"
    destination = tmp_path / "published-index"
    source.mkdir()
    destination.symlink_to("missing-index", target_is_directory=True)
    monkeypatch.setattr(builder, "_renameat2_no_replace", _unsupported_renameat2)

    with pytest.raises(FileExistsError):
        builder._publish_directory_no_replace(source, destination)

    assert destination.is_symlink()
    assert destination.readlink() == Path("missing-index")
    assert source.exists()


def test_builder_creates_train_only_artifact_with_converted_floorplans(monkeypatch, tmp_path):
    builder = _load_builder_module()
    dataset = [_sample([1.0, 4.0], [[10, 20, 3, 4], [30, 40, 5, 6]])]
    monkeypatch.setattr(builder, "FloorplanDatasetLite", lambda _path: dataset)
    monkeypatch.setattr(builder, "_renameat2_no_replace", _unsupported_renameat2)
    output = tmp_path / "pilot"

    summary = builder.build_index(tmp_path, output, max_per_n=1, seed=17)

    assert summary["source_split"] == "train"
    assert output.is_symlink()
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


def test_builder_cleans_temporary_directory_when_symlink_fallback_fails(monkeypatch, tmp_path):
    builder = _load_builder_module()
    output = tmp_path / "pilot"
    monkeypatch.setattr(builder, "FloorplanDatasetLite", lambda _path: [_sample([1.0])])
    monkeypatch.setattr(builder, "_renameat2_no_replace", _unsupported_renameat2)

    def fail_symlink(*_args, **_kwargs):
        raise OSError(errno.EIO, "simulated symlink failure")

    monkeypatch.setattr(builder.os, "symlink", fail_symlink)

    with pytest.raises(OSError, match="simulated symlink failure"):
        builder.build_index(tmp_path, output, max_per_n=1, seed=17)

    assert not output.exists()
    assert not output.is_symlink()
    assert list(tmp_path.glob(".pilot.tmp-*")) == []


def test_probe_uses_evaluator_aligned_polygon_anchors_for_transfer_and_proxies(monkeypatch, tmp_path):
    probe = _load_probe_module()
    constraints = np.array(
        [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    polygons = np.array(
        [
            [[0, 0], [4, 0], [4, 2], [-1, -1]],
            [[2, 3], [5, 3], [5, 9], [-1, -1]],
            [[7, 10], [11, 10], [11, 13], [-1, -1]],
        ],
        dtype=np.float32,
    )
    sample = _evaluation_sample([1.0, 1.0, 1.0, -1.0], constraints, polygons)
    expected_targets = np.array(
        [[-1, -1, -1, -1], [-1, -1, 3, 6], [7, 10, 4, 3]], dtype=np.float32
    )

    *_, target_positions = probe._target_arrays(sample)
    np.testing.assert_array_equal(target_positions, expected_targets)

    class FakeIndex:
        def query(self, _block_count, _vector, top_k):
            return SimpleNamespace(
                source_ids=np.array([7], dtype=np.int64),
                distances=np.array([0.25]),
                node_features=np.zeros((1, 3, 16), dtype=np.float32),
                fp_xywh=np.array([[[0, 0, 1, 1], [2, 0, 1, 1], [4, 0, 1, 1]]], dtype=np.float32),
            )

    observed = {}

    def fake_features(area, _b2b, _p2b, _pins, _constraints, targets):
        observed["feature_target_positions"] = targets.copy()
        return SimpleNamespace(
            global_vector=np.zeros(24, dtype=np.float32),
            node_matrix=np.zeros((len(area), 16), dtype=np.float32),
        )

    def capture_overlap(rectangles):
        observed["overlap_rectangles"] = rectangles.copy()
        return 0.0

    def capture_hpwl(rectangles, *_args):
        observed["hpwl_rectangles"] = rectangles.copy()
        return 0.0

    monkeypatch.setattr(probe.RetrievalIndex, "load", lambda _path: FakeIndex())
    monkeypatch.setattr(probe, "FloorplanDatasetLiteTest", lambda _path: [sample])
    monkeypatch.setattr(probe, "extract_retrieval_features", fake_features)
    monkeypatch.setattr(probe, "_overlap_proxy", capture_overlap)
    monkeypatch.setattr(probe, "_hpwl_proxy", capture_hpwl)

    rows = probe.run_probe(tmp_path / "index", tmp_path, cases=1, top_k=1)

    assert rows[0]["status"] == "matched"
    np.testing.assert_array_equal(observed["feature_target_positions"], expected_targets)
    np.testing.assert_array_equal(observed["overlap_rectangles"][1, 2:4], [3, 6])
    np.testing.assert_array_equal(observed["overlap_rectangles"][2], [7, 10, 4, 3])
    np.testing.assert_array_equal(observed["hpwl_rectangles"], observed["overlap_rectangles"])


def test_probe_returns_complete_matched_and_no_same_n_rows_without_mutating_index(monkeypatch, tmp_path):
    probe = _load_probe_module()
    index_path = tmp_path / "index"
    index_path.mkdir()
    manifest = index_path / "manifest.json"
    manifest.write_text("read-only")
    dataset = [_evaluation_sample([1.0, 1.0]), _evaluation_sample([1.0, 1.0, 1.0])]

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
