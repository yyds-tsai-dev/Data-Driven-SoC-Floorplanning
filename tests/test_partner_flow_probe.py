import argparse
import sys
from pathlib import Path

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
    row = {field: 0 for field in REQUIRED_ROW_FIELDS}
    row.update(
        model="flow",
        solver="euler",
        finite=True,
        repair={"available": False},
        candidates=[_candidate()],
        best_of_k=_candidate(),
    )
    validate_row(row)

    winner = best_of_k(
        [
            {
                "candidate_index": 8,
                "raw_overlap": 0.0,
                "raw_boundary_violations": 0,
                "raw_group_violations": 0,
                "raw_mib_violations": 0,
                "raw_hpwl_proxy": 11.0,
            },
            {
                "candidate_index": 3,
                "raw_overlap": 0.0,
                "raw_boundary_violations": 0,
                "raw_group_violations": 0,
                "raw_mib_violations": 0,
                "raw_hpwl_proxy": 11.0,
            },
        ]
    )

    assert winner["candidate_index"] == 3
    with pytest.raises(ValueError, match="missing required row fields"):
        validate_row({"case_id": 21})
    malformed = dict(row)
    malformed["candidates"] = [{"candidate_index": 0}]
    with pytest.raises(ValueError, match="candidate missing required fields"):
        validate_row(malformed)


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
