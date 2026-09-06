from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import torch
import numpy as np

from icdc_engine.engine import verify_hard_legal
from icdc_engine.qa_contract import (
    QA_RELATIVE_PATH,
    SCORER_RELATIVE_PATH,
    preflight_qa_contract,
    qa_manifest_fields,
    score_provided_local_no_runtime,
)
from icdc_engine.topology_artifact_guard import (
    assert_no_dense_fp_artifact,
    canonical_json_bytes,
    write_checked_json,
)
from icdc_engine.topology_data import (
    CorpusSourceReceipt,
    validate_raw_source,
    verified_training_fp_row,
)


REPO = Path(__file__).resolve().parents[1]


def _raw_source(tree_value: float = 0.0) -> list[torch.Tensor]:
    inp = torch.tensor(
        [[[4.0, 0.0, 0.0, 0.0, 0.0, 0.0], [9.0, 1.0, 1.0, 0.0, 0.0, 0.0]]]
    )
    b2b = torch.tensor([[[0.0, 1.0, 1.0]]])
    p2b = torch.empty((1, 0, 3))
    pins = torch.empty((1, 0, 2))
    tree = torch.tensor([[[tree_value, 0.0, 1.0]]])
    fp_sol = torch.tensor([[[2.0, 2.0, 11.0, 13.0], [3.0, 3.0, 17.0, 19.0]]])
    metrics = torch.tensor([[25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 5.0]])
    return [inp, b2b, p2b, pins, tree, fp_sol, metrics]


def _receipt(file_digest: str = "a" * 64) -> CorpusSourceReceipt:
    return CorpusSourceReceipt("worker_0/layouts_0.th", file_digest, 0, "b" * 64)


def _teacher_module():
    name = "_icdc_topology_teacher_p0_contract"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "scripts" / "probes" / "icdc_topology_teacher.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_verified_training_row_uses_fp_not_tree_and_keeps_fp_transient():
    first = verified_training_fp_row(_raw_source(0.0), _receipt("a" * 64))
    second = verified_training_fp_row(_raw_source(9.0), _receipt("c" * 64))

    assert validate_raw_source(_raw_source()) == (1, 2)
    assert first.fp_xywh.dtype is torch.float64
    assert first.fp_xywh.device.type == "cpu"
    assert first.fp_xywh.tolist() == [
        [11.0, 13.0, 2.0, 2.0],
        [17.0, 19.0, 3.0, 3.0],
    ]
    assert first.case == second.case
    assert first.input_fingerprint == second.input_fingerprint
    assert torch.equal(first.fp_xywh, second.fp_xywh)
    assert first.receipt.file_sha256 != second.receipt.file_sha256
    assert "fp_xywh" not in first.case
    assert "fp_sol" not in first.case
    assert "tree_sol" not in first.case


def test_tree_sol_change_does_not_change_teacher_input_or_output():
    teacher = _teacher_module()
    source_a = _raw_source(0.0)
    source_b = _raw_source(123.5)

    assert teacher._validate_source_shard(source_a) == (1, 2)
    assert teacher._validate_source_shard(source_b) == (1, 2)
    first = teacher._source_case_from_shard(source_a, 0, "worker_0/layouts_0.th#0")
    second = teacher._source_case_from_shard(source_b, 0, "worker_0/layouts_0.th#0")
    assert first == second
    assert teacher._sample_seed(20260813, first["instance_id"], 0) == teacher._sample_seed(
        20260813, second["instance_id"], 0
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda source: source[:6],
        lambda source: [*source[:4], torch.zeros((1, 2, 3)), *source[5:]],
        lambda source: [*source[:5], torch.zeros((1, 2, 3)), source[6]],
        lambda source: [*source[:6], torch.zeros((1, 7))],
    ),
)
def test_raw_source_rejects_wrong_seven_tensor_roles(mutator):
    with pytest.raises(ValueError, match="source"):
        validate_raw_source(mutator(_raw_source()))


def test_checked_artifacts_reject_dense_fp_but_keep_sparse_topology(tmp_path):
    sparse = {
        "schema": "fp_topology_v1",
        "version": 1,
        "axis_edges": [{"src": 0, "dst": 1, "axis": 0}],
        "contacts": [],
    }
    assert_no_dense_fp_artifact(sparse)
    digest = write_checked_json(tmp_path / "sparse.json", sparse)
    assert digest == hashlib.sha256((tmp_path / "sparse.json").read_bytes()).hexdigest()
    assert (tmp_path / "sparse.json").read_bytes() == canonical_json_bytes(sparse) + b"\n"

    bad_values = (
        {"fp_sol": [[1.0, 2.0, 3.0, 4.0]]},
        {"rects": [[0.0, 0.0, 1.0, 1.0]]},
        {"axis_edges": [], "origin": 2.0},
        {"contacts": [], "overlap_magnitude": 1.0},
        {"value": float("nan")},
        {"tensor": torch.zeros(1)},
    )
    for index, bad in enumerate(bad_values):
        path = tmp_path / f"bad-{index}.json"
        with pytest.raises(ValueError, match="dense fp artifact|canonical JSON"):
            write_checked_json(path, bad)
        assert not path.exists()


def _qa_case() -> dict[str, object]:
    return {
        "instance_id": "qa-soft",
        "n": 3,
        "area": [1.0, 1.0, 1.0],
        "cons": [
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        "tp": [[5.0, 0.0, 1.0, 1.0], [-1.0] * 4, [-1.0] * 4],
        "b2b": [],
        "p2b": [],
        "pins": [],
        "hpwl_ref": 0.0,
        "area_ref": 11.0,
    }


def _qa_area_case() -> dict[str, object]:
    return {
        "instance_id": "qa-area",
        "n": 1,
        "area": [100.0],
        "cons": [[0, 0, 0, 0, 0]],
        "tp": [[-1.0] * 4],
        "b2b": [],
        "p2b": [],
        "pins": [],
        "hpwl_ref": 0.0,
        "area_ref": 100.0,
    }


def test_qa_preflight_and_official_a4_a5_a6_semantics(tmp_path):
    evidence = preflight_qa_contract(REPO)
    fields = qa_manifest_fields(evidence)
    assert fields["qa_sha256"] == (
        "60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db"
    )

    case = _qa_case()
    golden_with_soft_v = torch.tensor(
        [[5.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [10.0, 0.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    audit = score_provided_local_no_runtime(case, golden_with_soft_v, evidence)
    assert audit.feasible
    assert audit.boundary_violations == 1
    assert audit.grouping_violations == 1
    assert audit.cost_no_runtime < 10.0

    moved_preplaced = golden_with_soft_v.clone()
    moved_preplaced[0, 0] = 4.0
    assert not score_provided_local_no_runtime(case, moved_preplaced, evidence).feasible

    for area, feasible in ((99.0, True), (101.0, True), (98.99, False), (101.01, False)):
        rect = torch.tensor([[0.0, 0.0, area, 1.0]], dtype=torch.float64)
        assert score_provided_local_no_runtime(
            _qa_area_case(), rect, evidence
        ).feasible is feasible
        hard = verify_hard_legal(
            rect.numpy(), np.asarray([100.0]),
            np.asarray([[0, 0, 0, 0, 0]]),
            np.asarray([[-1.0, -1.0, -1.0, -1.0]]),
        )
        assert hard["area"] is feasible
        assert hard["ok"] is feasible

    copied_qa = tmp_path / QA_RELATIVE_PATH
    copied_scorer = tmp_path / SCORER_RELATIVE_PATH
    copied_qa.parent.mkdir(parents=True)
    copied_scorer.parent.mkdir(parents=True)
    shutil.copyfile(REPO / QA_RELATIVE_PATH, copied_qa)
    shutil.copyfile(REPO / SCORER_RELATIVE_PATH, copied_scorer)
    assert preflight_qa_contract(tmp_path).qa_sha256 == evidence.qa_sha256
    copied_qa.write_bytes(b"changed")
    with pytest.raises(ValueError, match="QA PDF SHA256"):
        preflight_qa_contract(tmp_path)
