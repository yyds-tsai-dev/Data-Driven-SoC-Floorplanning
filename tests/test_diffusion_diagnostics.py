from __future__ import annotations

import pytest
import torch

from floorset_arch.diffusion.diagnostics import (
    build_diffusion_diagnostic_report,
    constraint_annotation_summary,
    placement_from_fp_sol,
    placement_panel_title,
    plot_diffusion_diagnostic,
)
from floorset_arch.models import Placement, Rect
from floorset_arch.parser import parse_instance


def _instance():
    return parse_instance(
        2,
        torch.tensor([4.0, 4.0]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 2.0]]),
        None,
    )


def _constraint_instance():
    return parse_instance(
        4,
        torch.tensor([4.0, 4.0, 9.0, 9.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor(
            [
                [1.0, 0.0, 2.0, 7.0, 1.0],
                [0.0, 1.0, 2.0, 7.0, 2.0],
                [0.0, 0.0, 3.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 8.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
    )


def test_placement_panel_title_includes_cost_and_diagnostic_metrics():
    title = placement_panel_title(
        "Diffusion raw",
        {
            "overlap_count": 3,
            "boundary_violations": 1,
            "group_violations": 2,
            "mib_violations": 4,
            "bbox_area": 123.4,
        },
        cost=5.678,
    )

    assert "Diffusion raw" in title
    assert "overlaps=3" in title
    assert "soft=7" in title
    assert "bbox=123" in title
    assert "cost=5.6780" in title


def test_constraint_annotation_summary_marks_cluster_mib_fixed_preplaced_and_boundary():
    annotations = constraint_annotation_summary(_constraint_instance())

    assert annotations[0]["cluster"] == 7
    assert annotations[0]["mib"] == 2
    assert annotations[0]["fixed"] is True
    assert annotations[0]["boundary"] == "L"
    assert annotations[1]["preplaced"] is True
    assert annotations[1]["boundary"] == "R"
    assert annotations[2]["mib"] == 3
    assert annotations[2]["boundary"] == "T"
    assert annotations[3]["boundary"] == "B"


def test_placement_from_fp_sol_converts_official_width_height_xy_order():
    placement = placement_from_fp_sol(
        torch.tensor([[2.0, 3.0, 5.0, 7.0], [4.0, 6.0, 11.0, 13.0]]),
        block_count=2,
    )

    assert placement.to_position_list(2) == [(5.0, 7.0, 2.0, 3.0), (11.0, 13.0, 4.0, 6.0)]


def test_placement_from_fp_sol_accepts_validation_polygon_labels():
    placement = placement_from_fp_sol(
        torch.tensor(
            [
                [[5.0, 7.0], [5.0, 10.0], [7.0, 10.0], [7.0, 7.0], [5.0, 7.0]],
                [[11.0, 13.0], [11.0, 19.0], [15.0, 19.0], [15.0, 13.0], [11.0, 13.0]],
            ]
        ),
        block_count=2,
    )

    assert placement.to_position_list(2) == [(5.0, 7.0, 2.0, 3.0), (11.0, 13.0, 4.0, 6.0)]


def test_build_diffusion_diagnostic_report_compares_raw_repaired_and_golden():
    inst = _instance()
    raw = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(1.0, 0.0, 2.0, 2.0)})
    repaired = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(2.0, 0.0, 2.0, 2.0)})
    golden = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 2.0, 2.0, 2.0)})

    report = build_diffusion_diagnostic_report(inst, raw, repaired, golden, case_id=7)

    assert report["case_id"] == 7
    assert report["placements"]["raw"]["metrics"]["overlap_count"] == 1
    assert report["placements"]["repaired"]["metrics"]["overlap_count"] == 0
    assert report["placements"]["golden"]["metrics"]["overlap_count"] == 0
    assert report["repair_delta"]["avg_moved_manhattan"] == pytest.approx(0.5)


def test_plot_diffusion_diagnostic_writes_three_panel_png(tmp_path):
    pytest.importorskip("matplotlib")
    inst = _instance()
    raw = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(1.0, 0.0, 2.0, 2.0)})
    repaired = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(2.0, 0.0, 2.0, 2.0)})
    golden = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 2.0, 2.0, 2.0)})

    output = plot_diffusion_diagnostic(
        inst,
        raw,
        repaired,
        golden,
        tmp_path / "case7.png",
        case_id=7,
    )

    assert output.exists()
    assert output.name == "case7.png"
