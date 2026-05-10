import json

import torch

from floorset_arch.diagnostics import placement_metrics, repair_delta
from floorset_arch.models import Placement, Rect
from floorset_arch.models import SolverConfig
from floorset_arch.optimizer import ArchitectureV3Optimizer
from floorset_arch.parser import parse_instance
from floorset_arch.repair import repair_placement


def _diagnostic_instance(block_count=2):
    return parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(block_count, 5),
        torch.full((block_count, 4), -1.0),
    )


def test_placement_metrics_counts_overlap_and_bbox():
    inst = _diagnostic_instance()
    placement = Placement({0: Rect(0, 0, 2, 2), 1: Rect(1, 0, 2, 2)})

    metrics = placement_metrics(inst, placement)

    assert metrics["overlap_count"] == 1
    assert metrics["bbox_area"] == 6.0


def test_repair_delta_reports_average_motion_and_quality_delta():
    before = Placement({0: Rect(0, 0, 2, 2), 1: Rect(2, 0, 2, 2)})
    after = Placement({0: Rect(1, 0, 2, 2), 1: Rect(2, 3, 2, 2)})
    before_metrics = {"bbox_area": 8.0, "hpwl_proxy": 10.0}
    after_metrics = {"bbox_area": 20.0, "hpwl_proxy": 7.0}

    delta = repair_delta(before, after, before_metrics, after_metrics)

    assert delta["avg_moved_manhattan"] == 2.0
    assert delta["bbox_area_delta"] == 12.0
    assert delta["hpwl_proxy_delta"] == -3.0


def test_optimizer_writes_repair_trace_jsonl(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("FLOORSET_REPAIR_TRACE_JSONL", str(trace_path))

    problem = {
        "block_count": 2,
        "area_targets": torch.tensor([4.0, 4.0]),
        "b2b_connectivity": torch.empty(0, 3),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(2, 5),
        "target_positions": torch.full((2, 4), -1.0),
    }
    optimizer = ArchitectureV3Optimizer(
        config=SolverConfig(checkpoint_repo_relative=False, default_checkpoint="missing.pt")
    )

    optimizer.solve(**problem)

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows
    assert rows[0]["candidate"]["profile"] == "soft"
    assert rows[0]["candidate"]["kind"] == "relative_order"
    assert "before_repair" in rows[0]
    assert "after_repair" in rows[0]
    assert "delta" in rows[0]


def test_overlap_repair_uses_quality_proxy_not_first_frontier():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 20.0]),
        torch.empty(0, 3),
        torch.tensor([[0.0, 1.0, 100.0]]),
        torch.tensor([[5.0, 1.0]]),
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(1.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 10.0),
        }
    )

    repaired = repair_placement(inst, placement, SolverConfig(max_repair_passes=1))

    assert repaired.rects[1].x == 4.0
    assert repaired.rects[1].y == 0.0
