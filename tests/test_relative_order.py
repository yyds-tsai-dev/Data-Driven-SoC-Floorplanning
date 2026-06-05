import torch

from floorset_arch.geometry import edge_touch_length
from floorset_arch.models import AnchorGuidance, Rect, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.relative_order import construct_relative_order_placement


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
