import torch

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
