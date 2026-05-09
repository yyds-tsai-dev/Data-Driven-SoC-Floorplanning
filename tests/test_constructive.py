import torch

from floorset_arch.constructive import construct_beam_placement
from floorset_arch.geometry import has_overlaps
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.models import AnchorGuidance, Rect, SolverConfig
from floorset_arch.parser import parse_instance


def test_beam_decoder_keeps_preplaced_obstacle_and_avoids_overlap():
    areas = torch.tensor([4.0, 4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    inst = parse_instance(
        3,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        targets,
    )

    placement = construct_beam_placement(inst, SolverConfig(beam_width=2), build_hetero_floorplan_graph(inst))

    assert placement.rects[0].as_tuple() == (0.0, 0.0, 2.0, 2.0)
    assert not has_overlaps(list(placement.rects.values()))


def test_beam_decoder_uses_anchor_guidance_as_relative_prior():
    areas = torch.tensor([4.0, 4.0, 4.0])
    inst = parse_instance(
        3,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        None,
    )
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(20.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(10.0, 0.0, 2.0, 2.0),
        },
        priority={0: 0.1, 1: 0.9, 2: 0.5},
    )

    placement = construct_beam_placement(inst, SolverConfig(beam_width=2), build_hetero_floorplan_graph(inst))

    assert not has_overlaps(list(placement.rects.values()))
    assert placement.rects[1].x <= placement.rects[2].x <= placement.rects[0].x
