import torch

from floorset_arch.geometry import edge_touch_length
from floorset_arch.models import AnchorGuidance, Rect, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.relative_order import (
    _cluster_grouping_pressure,
    _narrow_grouping_axis_bonus,
    construct_relative_order_placement,
)


def test_relative_order_can_clamp_guidance_to_preplaced_frame(monkeypatch):
    monkeypatch.setenv("FLOORSET_PREPLACED_FRAME_CLAMP", "1")
    inst = parse_instance(
        2,
        torch.tensor([4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor(
            [
                    [0.0, 1.0, 0.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [8.0, 8.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
    )
    inst.anchor_guidance = AnchorGuidance(rect_priors={1: Rect(48.0, 48.0, 2.0, 2.0)})

    placement = construct_relative_order_placement(
        inst,
        SolverConfig(anchor_translation_strength=1.0),
    )

    assert placement.rects[1].right <= 10.0
    assert placement.rects[1].top <= 10.0


def test_relative_order_supports_wide_and_tall_shape_profiles():
    inst = parse_instance(
        1,
        torch.tensor([16.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(1, 5),
        torch.full((1, 4), -1.0),
    )

    wide = construct_relative_order_placement(inst, profile="wide").rects[0]
    tall = construct_relative_order_placement(inst, profile="tall").rects[0]

    assert wide.width > wide.height
    assert tall.height > tall.width
    assert abs(wide.area - 16.0) <= 1e-6
    assert abs(tall.area - 16.0) <= 1e-6


def test_grouping_adjacency_bias_can_chain_compact_profile(monkeypatch):
    constraints = torch.zeros(4, 5)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
    inst = parse_instance(
        4,
        torch.full((4,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((4, 4), -1.0),
    )
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(20.0, 20.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
            3: Rect(0.0, 2.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "1")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert edge_touch_length(placement.rects[0], placement.rects[1]) > 0.0


def test_narrow_grouping_pair_bias_does_not_enable_global_key_blend(monkeypatch):
    constraints = torch.zeros(3, 5)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
    constraints[2, 3] = 1.0
    inst = parse_instance(
        3,
        torch.full((3,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((3, 4), -1.0),
    )
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(60.0, 0.0, 2.0, 2.0),
            1: Rect(30.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "1")
    monkeypatch.setenv("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "1")
    monkeypatch.setenv("FLOORSET_GROUPING_ADJACENCY_KEY_BLEND", "1.0")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert placement.rects[0].x >= placement.rects[1].right


def test_narrow_grouping_pair_bias_only_applies_to_ambiguous_same_cluster_pairs(monkeypatch):
    constraints = torch.zeros(4, 5)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
    constraints[2, 3] = 2.0
    constraints[3, 3] = 2.0
    inst = parse_instance(
        4,
        torch.full((4,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((4, 4), -1.0),
    )
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(3.0, 3.0, 2.0, 2.0),
            2: Rect(20.0, 0.0, 2.0, 2.0),
            3: Rect(20.0, 80.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "1")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_MIN_MEMBERS", "2")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_AMBIGUITY_MARGIN", "0.30")
    widths = [2.0, 2.0, 2.0, 2.0]
    heights = [2.0, 2.0, 2.0, 2.0]
    raw_x = [1.0, 4.0, 21.0, 21.0]
    raw_y = [1.0, 4.0, 1.0, 81.0]

    assert _cluster_grouping_pressure([0, 1], raw_x, raw_y, widths, heights)
    assert _cluster_grouping_pressure([2, 3], raw_x, raw_y, widths, heights)
    assert _narrow_grouping_axis_bonus(
        1.5,
        1.5,
        enabled=True,
        pressure=True,
        orientation="H",
        bonus=0.12,
        ambiguity_margin=0.30,
    ) == (1.62, 1.5)
    assert _narrow_grouping_axis_bonus(
        0.0,
        40.0,
        enabled=True,
        pressure=True,
        orientation="V",
        bonus=0.12,
        ambiguity_margin=0.30,
    ) == (0.0, 40.0)

    placement = construct_relative_order_placement(inst, profile="compact")

    assert edge_touch_length(placement.rects[0], placement.rects[1]) > 0.0
    assert edge_touch_length(placement.rects[2], placement.rects[3]) == 0.0
