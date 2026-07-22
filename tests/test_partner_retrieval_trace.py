from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "scripts" / "probes" / "retrieval_trace.py"


def _load_trace_module():
    spec = importlib.util.spec_from_file_location("retrieval_trace", TRACE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prediction(value: float) -> np.ndarray:
    return np.full((1, 4), value, dtype=np.float64)


def test_trace_help_exposes_required_diagnostic_flags():
    result = subprocess.run(
        [sys.executable, str(TRACE_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in (
        "--index", "--checkpoint", "--case-ids", "--selection-k",
        "--refine-seconds", "--seed", "--output",
    ):
        assert flag in result.stdout


def test_case_ids_are_arbitrary_non_contiguous_and_ordered():
    trace = _load_trace_module()

    assert trace.parse_case_ids("99,0,49,10") == [99, 0, 49, 10]
    with pytest.raises(ValueError, match="duplicate"):
        trace.parse_case_ids("1,1")


def test_shared_rank_drives_forced_and_maximum_policies(monkeypatch):
    trace = _load_trace_module()
    predictions = [_prediction(10), _prediction(11), _prediction(20), _prediction(21)]
    sources = ["direct", "direct", "retrieval", "retrieval"]
    calls = []

    def forced_selector(preds, got_sources, order, total, quota):
        calls.append((preds, got_sources, order, total, quota))
        return [preds[2], preds[0]]

    monkeypatch.setattr(trace, "_select_ranked_source_quota", forced_selector)
    policies = trace.select_policy_candidate_indexes(
        predictions, sources, [2, 0, 3, 1], selection_k=2, retrieval_quota=1,
    )

    assert calls == [(predictions, sources, [2, 0, 3, 1], 2, 1)]
    assert policies["forced_quota"] == [2, 0]
    assert policies["maximum_quota"] == [2, 0]


def test_forced_selection_uses_direct_backfill_when_retrieval_is_short():
    trace = _load_trace_module()
    predictions = [_prediction(10), _prediction(11), _prediction(12), _prediction(20)]
    selected = trace.select_policy_candidate_indexes(
        predictions,
        ["direct", "direct", "direct", "retrieval"],
        [3, 0, 1, 2],
        selection_k=4,
        retrieval_quota=2,
    )

    assert selected["forced_quota"] == [3, 0, 1, 2]
    assert selected["maximum_quota"] == [3, 0, 1, 2]


def test_candidate_ids_and_refinement_seeds_are_stable():
    trace = _load_trace_module()

    candidate_id = trace.stable_candidate_id(49, "retrieval", 1, 71)
    assert candidate_id == trace.stable_candidate_id(49, "retrieval", 1, 71)
    assert candidate_id != trace.stable_candidate_id(49, "direct", 1, None)
    assert trace.refinement_seed(9001, candidate_id) == trace.refinement_seed(9001, candidate_id)


def test_union_candidates_receive_identical_refinement_allowance(monkeypatch):
    trace = _load_trace_module()
    calls = []
    monkeypatch.setattr(trace.time, "time", lambda: 100.0)

    def worker(payload):
        calls.append(payload)
        return None

    monkeypatch.setattr(trace, "_worker_refine", worker)
    candidates = [
        {"candidate_id": "direct-0", "prediction": _prediction(1), "source": "direct"},
        {"candidate_id": "retrieval-0", "prediction": _prediction(2), "source": "retrieval"},
    ]
    trace.refine_union_candidates(
        candidates,
        np.array([1.0]), np.zeros((1, 5)), np.full((1, 4), -1.0),
        np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)),
        refine_seconds=1.25, seed=9001,
    )

    assert len(calls) == 2
    assert all(payload[7] == 101.25 for payload in calls)
    assert [row["refine_seconds"] for row in candidates] == [1.25, 1.25]


def test_retrieval_metadata_and_failures_are_json_safe_and_explicit():
    trace = _load_trace_module()
    payload = {
        "source_ids": np.array([71], dtype=np.int64),
        "distance": np.float32(0.25),
        "rejected": np.int64(1),
        "failure": RuntimeError("worker failed"),
        "nonfinite": float("nan"),
    }

    safe = trace.json_safe(payload)

    assert safe == {
        "source_ids": [71], "distance": pytest.approx(0.25), "rejected": 1,
        "failure": "RuntimeError: worker failed", "nonfinite": None,
    }
    assert json.loads(json.dumps(safe)) == safe


def test_index_contract_is_read_only_and_hashes_manifest_and_shards(tmp_path):
    trace = _load_trace_module()
    shard = tmp_path / "n_021.npz"
    shard.write_bytes(b"immutable shard")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shards": [{"n": 21}]}), encoding="utf-8")
    before = trace.index_contract(tmp_path)

    after = trace.index_contract(tmp_path)

    assert before == after
    assert before["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert before["shards"] == {"n_021.npz": hashlib.sha256(shard.read_bytes()).hexdigest()}


def test_trace_module_does_not_contain_index_build_or_production_writes():
    text = TRACE_PATH.read_text(encoding="utf-8")

    assert "save_index(" not in text
    assert "build_partner_retrieval_index" not in text
    assert "partner/my_opt_claude.py" not in text
