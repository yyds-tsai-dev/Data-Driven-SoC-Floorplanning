import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.probes.flow_candidate_probe as probe
from scripts.probes.flow_candidate_probe import (
    REQUIRED_ROW_FIELDS,
    best_of_k,
    build_matrix,
    parse_args,
    timed_call,
    validate_flow_checkpoint,
    validate_row,
)


def _candidate(index: int = 0) -> dict[str, object]:
    return {
        "candidate_index": index,
        "anchor_exact": True,
        "anchor_max_error": 0.0,
        "raw_overlap": 0,
        "raw_hpwl_proxy": 11.0,
        "raw_boundary_violations": 0,
        "raw_group_violations": 0,
        "raw_mib_violations": 0,
    }


def _row(**updates) -> dict[str, object]:
    row = {
        "case_id": 21,
        "model": "flow",
        "solver": "euler",
        "steps": 4,
        "nfe": 4,
        "samples": 1,
        "seed": 20260723,
        "cold_load_s": 0.1,
        "warm_latency_s": 0.2,
        "peak_memory_bytes": 0,
        "finite": True,
        "anchor_exact": True,
        "anchor_max_error": 0.0,
        "raw_overlap": 0,
        "raw_hpwl_proxy": 11.0,
        "raw_boundary_violations": 0,
        "raw_group_violations": 0,
        "raw_mib_violations": 0,
        "best_of_k": _candidate(),
        "candidates": [_candidate()],
        "repair": {"available": False, "reason": "not run"},
    }
    row.update(updates)
    return row


def test_probe_matrix_reports_nfe_not_only_steps():
    rows = build_matrix(flow_steps=[4, 8], solvers=["euler", "heun"])

    assert ("flow", "euler", 4, 4) in rows
    assert ("flow", "heun", 4, 8) in rows
    assert ("direct", "ddim", 50, 50) in rows


def test_parser_requires_comparable_inputs_and_keeps_case_ids():
    args = parse_args(
        [
            "--direct-checkpoint",
            "direct.pt",
            "--flow-checkpoint",
            "flow.pt",
            "--cases",
            "21",
            "68",
            "--samples",
            "8",
            "--flow-steps",
            "4",
            "8",
            "--solvers",
            "euler",
            "heun",
            "--output",
            "probe.json",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.cases == [21, 68]
    assert args.samples == 8
    assert args.flow_steps == [4, 8]
    assert args.solvers == ["euler", "heun"]


def test_schema_requires_candidate_metrics_and_deterministic_best_of_k():
    row = _row()
    validate_row(row)

    winner = best_of_k([_candidate(8), _candidate(3)])

    assert winner["candidate_index"] == 3
    with pytest.raises(ValueError, match="missing required row fields"):
        validate_row({"case_id": 21})
    malformed = dict(row)
    malformed["candidates"] = [{"candidate_index": 0}]
    with pytest.raises(ValueError, match="candidate missing required fields"):
        validate_row(malformed)


@pytest.mark.parametrize(
    "field",
    [
        "anchor_max_error",
        "raw_overlap",
        "raw_hpwl_proxy",
        "raw_boundary_violations",
        "raw_group_violations",
        "raw_mib_violations",
    ],
)
def test_candidate_schema_rejects_negative_raw_metrics(field):
    candidate = _candidate()
    candidate[field] = -0.1

    with pytest.raises(ValueError, match=field):
        probe.validate_candidate(candidate)


@pytest.mark.parametrize("field", ["anchor_max_error", "raw_overlap", "raw_hpwl_proxy"])
def test_best_of_k_rejects_negative_candidate_metrics(field):
    candidate = _candidate()
    candidate[field] = -0.1

    with pytest.raises(ValueError, match=field):
        best_of_k([candidate])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warm_latency_s", float("nan")),
        ("warm_latency_s", float("inf")),
        ("cold_load_s", -0.1),
        ("samples", -1),
        ("nfe", True),
        ("peak_memory_bytes", True),
        ("finite", "true"),
        ("raw_hpwl_proxy", "eleven"),
    ],
)
def test_row_schema_rejects_nonfinite_or_malformed_measurements(field, value):
    with pytest.raises(ValueError, match=field):
        validate_row(_row(**{field: value}))


def test_row_schema_requires_one_candidate_per_sample_and_a_matching_best_record():
    with pytest.raises(ValueError, match="candidate count"):
        validate_row(_row(samples=2))
    bad_best = _candidate()
    bad_best["raw_hpwl_proxy"] = 12.0
    with pytest.raises(ValueError, match="best_of_k must match"):
        validate_row(_row(best_of_k=bad_best))


def test_run_probe_measures_one_model_lifecycle_at_a_time(monkeypatch):
    events = []
    live_models = set()
    args = SimpleNamespace(
        samples=1,
        flow_steps=[4],
        solvers=["euler"],
        cases=[21, 68],
        direct_checkpoint=Path("direct.pt"),
        flow_checkpoint=Path("flow.pt"),
        data_path=Path("data"),
    )

    monkeypatch.setattr(probe, "_require_checkpoint", lambda path, _label: path)
    monkeypatch.setattr(probe, "_load_official_case", lambda case_id, _data: ("case", case_id))
    monkeypatch.setattr(probe, "_initialize_cuda_context", lambda _device: events.append("context"), raising=False)

    def load_model(_path, *, method, device):
        assert not live_models
        live_models.add(method)
        events.append(f"load:{method}")
        return method, SimpleNamespace(), 0.1

    def warm_model(model, _config, _case, *, model_name, samples, device):
        assert model == model_name and live_models == {model_name} and samples == 1
        events.append(f"warm:{model_name}")

    def sample_row(*, case_id, model, matrix_row, **_kwargs):
        model_name, solver, steps, nfe = matrix_row
        assert model == model_name and live_models == {model_name}
        events.append(f"sample:{model_name}:{case_id}")
        return _row(
            case_id=case_id,
            model=model_name,
            solver=solver,
            steps=steps,
            nfe=nfe,
        )

    def release_model(_device):
        assert len(live_models) == 1
        model = next(iter(live_models))
        events.append(f"release:{model}")
        live_models.clear()

    monkeypatch.setattr(probe, "_load_model", load_model)
    monkeypatch.setattr(probe, "_warm_model", warm_model)
    monkeypatch.setattr(probe, "_sample_row", sample_row)
    monkeypatch.setattr(probe, "_release_model", release_model, raising=False)

    result = probe.run_probe(args)

    assert events == [
        "context",
        "load:direct",
        "warm:direct",
        "sample:direct:21",
        "sample:direct:68",
        "release:direct",
        "load:flow",
        "warm:flow",
        "sample:flow:21",
        "sample:flow:68",
        "release:flow",
    ]
    assert not live_models
    assert result["metadata"]["cold_load_order"] == ["direct", "flow"]


def test_evaluator_visible_hard_anchors_mask_non_visible_label_coordinates():
    constraints = torch.tensor(
        [
            [0, 1, 0, 0, 0],  # preplaced: evaluator exposes xywh
            [1, 0, 0, 0, 0],  # fixed: evaluator exposes only wh
            [0, 0, 0, 0, 0],  # movable: evaluator exposes nothing
        ]
    )
    label_positions = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]]
    )

    anchors = probe.evaluator_visible_target_positions(label_positions, constraints, block_count=3)
    assert torch.equal(
        anchors,
        torch.tensor([[10.0, 11.0, 12.0, 13.0], [-1.0, -1.0, 22.0, 23.0], [-1.0, -1.0, -1.0, -1.0]]),
    )

    altered = label_positions.clone()
    altered[1, :2] = torch.tensor([999.0, 998.0])
    altered[2] = torch.tensor([997.0, 996.0, 995.0, 994.0])
    assert torch.equal(
        anchors,
        probe.evaluator_visible_target_positions(altered, constraints, block_count=3),
    )


@pytest.mark.parametrize("checkpoint", [{"args": {}}, {"args": {"training_method": "diffusion"}}])
def test_flow_checkpoint_rejects_non_flow_methods_before_loading_weights(checkpoint):
    with pytest.raises(ValueError, match="flow_matching_v1"):
        validate_flow_checkpoint(checkpoint)


def test_timed_call_synchronizes_cuda_immediately_around_sampling():
    events = []
    clock = iter([1.0, 1.25])

    result, latency_s = timed_call(
        lambda: events.append("sample") or "result",
        device_type="cuda",
        synchronize=lambda: events.append("sync"),
        clock=lambda: next(clock),
    )

    assert result == "result"
    assert latency_s == 0.25
    assert events == ["sync", "sample", "sync"]
