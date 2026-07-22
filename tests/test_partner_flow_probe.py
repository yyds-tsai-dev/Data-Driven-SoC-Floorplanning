import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probes.flow_candidate_probe import (
    REQUIRED_ROW_FIELDS,
    best_of_k,
    build_matrix,
    parse_args,
    timed_call,
    validate_flow_checkpoint,
    validate_row,
)


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
        candidates=[{"candidate_index": 0}],
        best_of_k={"candidate_index": 0},
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
