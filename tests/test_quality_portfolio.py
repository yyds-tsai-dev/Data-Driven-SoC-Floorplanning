import torch

from floorset_arch.geometry import Rect, bbox, has_overlaps
from floorset_arch.models import Placement, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.quality_portfolio import refine_quality_candidate
from floorset_arch.repair import soft_violation_counts
from floorset_arch.scoring import hpwl_proxy


def test_hpwl_refine_moves_connected_block_closer_without_soft_regression(monkeypatch):
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.tensor([[0.0, 1.0, 20.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        torch.full((3, 4), -1.0),
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(80.0, 0.0, 2.0, 2.0),
            2: Rect(40.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "8")
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "32")

    refined = refine_quality_candidate(inst, placement, SolverConfig(), "hpwl_refine")

    assert not has_overlaps(list(refined.rects.values()))
    assert soft_violation_counts(inst, refined) == soft_violation_counts(inst, placement)
    assert hpwl_proxy(inst, refined.rects) < hpwl_proxy(inst, placement.rects)


def test_area_refine_reduces_bbox_without_soft_regression(monkeypatch):
    inst = parse_instance(
        4,
        torch.full((4,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(4, 5),
        torch.full((4, 4), -1.0),
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(100.0, 100.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "8")
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "32")

    refined = refine_quality_candidate(inst, placement, SolverConfig(), "area_refine")

    assert not has_overlaps(list(refined.rects.values()))
    assert soft_violation_counts(inst, refined) == soft_violation_counts(inst, placement)
    assert bbox(list(refined.rects.values())).area < bbox(list(placement.rects.values())).area
