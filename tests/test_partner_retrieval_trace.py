from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


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


def test_case_trace_builds_evaluator_faithful_anchors_from_golden_layout(monkeypatch):
    trace = _load_trace_module()
    seen_targets = []

    class FakeOptimizer:
        retrieval_slots = 0

        def _sample_direct_raw_preds(self, _n, _at, _cons, targets, *_args, **_kwargs):
            seen_targets.append(targets.clone())
            return [_prediction(3)]

        def _sample_retrieval_preds(self, *_args, **_kwargs):
            return trace.CandidateBatch("retrieval", [], 0.0, {})

        def _rank_portfolio(self, *_args):
            return [0]

    sample = {
        "input": (
            torch.tensor([1.0]), torch.empty((0, 3)), torch.empty((0, 3)),
            torch.empty((0, 2)), torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]),
        ),
        "label": (
            torch.tensor([[[4.0, 5.0], [6.0, 5.0], [6.0, 8.0], [4.0, 8.0]]]),
            torch.empty(0),
        ),
    }
    monkeypatch.setattr(trace, "_score_layout", lambda *_args: {"cost_no_runtime": 1.0})
    monkeypatch.setattr(trace, "refine_union_candidates", lambda *_args, **_kwargs: None)

    trace._case_trace(FakeOptimizer(), sample, 49, selection_k=1, refine_seconds=1.0, seed=9001)

    assert len(seen_targets) == 1
    assert seen_targets[0].tolist() == [[4.0, 5.0, 2.0, 3.0]]


def test_auto_selection_k_reads_live_legalizer_pool_capacity(monkeypatch):
    trace = _load_trace_module()
    monkeypatch.setattr(trace.column_sa_legalizer, "_POOL_SIZE", 24)
    monkeypatch.setenv("PARTNER_NREF", "15")
    monkeypatch.setenv("PARTNER_NREF_MIN_N", "95")

    assert trace.resolve_selection_k(94, "auto") == 8
    assert trace.resolve_selection_k(95, "auto") == 15


def test_auto_selection_k_matches_production_minimum_for_small_live_pool(monkeypatch):
    trace = _load_trace_module()
    monkeypatch.setattr(trace.column_sa_legalizer, "_POOL_SIZE", 2)
    monkeypatch.setenv("PARTNER_NREF", "15")
    monkeypatch.setenv("PARTNER_NREF_MIN_N", "95")

    assert trace.resolve_selection_k(94, "auto") == 3
    assert trace.resolve_selection_k(95, "auto") == 3


def test_run_trace_rejects_silently_unloaded_direct_checkpoint(monkeypatch, tmp_path):
    trace = _load_trace_module()
    checkpoint = tmp_path / "direct.pt"
    checkpoint.write_bytes(b"checkpoint")

    class FakeOptimizer:
        retrieval_index = object()
        direct_model = None

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(trace, "MyOptimizer", FakeOptimizer)
    monkeypatch.setattr(trace, "index_contract", lambda _path: {"manifest_sha256": "x", "shards": {}})
    args = SimpleNamespace(
        index=tmp_path, checkpoint=checkpoint, device="cpu", data_path=tmp_path,
        case_ids=[0], selection_k="auto", refine_seconds=1.0, seed=9001,
    )

    with pytest.raises(RuntimeError, match="Direct checkpoint.*did not load"):
        trace.run_trace(args)


def test_score_layout_uses_real_evaluator_helper_contract_on_golden_sample():
    trace = _load_trace_module()
    dataset = trace.FloorplanDatasetLiteTest(str(REPO_ROOT / "FloorSet"))
    sample = dataset[0]
    n = int((sample["input"][0] != -1).sum().item())
    from gen_decoder_probe import _golden_rects

    score = trace._score_layout(sample, np.asarray(_golden_rects(sample, n)), n)

    assert np.isfinite(score["cost_no_runtime"])
    assert isinstance(score["is_feasible"], bool)
    for field in ("hpwl_gap", "area_gap", "violations_relative"):
        assert np.isfinite(score[field])


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
    assert "partner/contest_optimizer.py" not in text
