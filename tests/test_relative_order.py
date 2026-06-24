import torch

from floorset_arch.geometry import edge_touch_length
from floorset_arch.models import AnchorGuidance, Rect, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.relative_order import (
    _cluster_grouping_pressure,
    _narrow_grouping_pair_bias_enabled,
    _narrow_grouping_axis_bonus,
    construct_relative_order_placement,
)


def test_narrow_grouping_pair_bias_defaults_on_but_can_be_disabled(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", raising=False)
    assert _narrow_grouping_pair_bias_enabled()

    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "0")
    assert not _narrow_grouping_pair_bias_enabled()


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


def test_removed_broad_grouping_flag_does_not_enable_global_key_blend(monkeypatch):
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
    monkeypatch.setenv("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "1")
    monkeypatch.setenv("FLOORSET_GROUPING_ADJACENCY_KEY_BLEND", "1.0")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert placement.rects[0].x >= placement.rects[1].right


def test_narrow_grouping_pair_bias_flips_ambiguous_same_cluster_axis(monkeypatch):
    constraints = torch.zeros(3, 5)
    constraints[:, 3] = 1.0
    inst = parse_instance(
        3,
        torch.full((3,), 16.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((3, 4), -1.0),
    )
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(0.0, 0.0, 8.0, 2.0),
            1: Rect(4.0, 2.2, 8.0, 2.0),
            2: Rect(80.0, 0.0, 8.0, 2.0),
        }
    )
    config = SolverConfig(anchor_translation_strength=0.0)
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_MIN_MEMBERS", "2")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_AMBIGUITY_MARGIN", "1.0")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_AXIS_BONUS", "1.0")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_PRESSURE", "1.0")

    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "0")
    disabled = construct_relative_order_placement(inst, config, profile="soft")
    assert disabled.rects[0].top <= disabled.rects[1].y

    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "1")
    enabled = construct_relative_order_placement(inst, config, profile="soft")
    assert enabled.rects[0].right <= enabled.rects[1].x


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
