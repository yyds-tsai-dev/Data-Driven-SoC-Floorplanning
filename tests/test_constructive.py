import torch

from floorset_arch.constructive import construct_initial_placement
from floorset_arch.geometry import has_overlaps
from floorset_arch.parser import parse_instance


def test_constructive_placement_keeps_preplaced_obstacle_and_avoids_overlap():
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

    placement = construct_initial_placement(inst)

    assert placement.rects[0].as_tuple() == (0.0, 0.0, 2.0, 2.0)
    assert not has_overlaps(list(placement.rects.values()))

