import torch

from floorset_arch.geometry import Rect, has_overlaps
from floorset_arch.models import Placement
from floorset_arch.parser import parse_instance
from floorset_arch.repair import repair_placement


def test_repair_does_not_move_preplaced_or_resize_fixed_blocks():
    areas = torch.tensor([4.0, 9.0, 16.0])
    constraints = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [-1.0, -1.0, 3.0, 3.0],
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
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 1.0, 9.0),
            2: Rect(0.0, 0.0, 4.0, 4.0),
        }
    )

    repaired = repair_placement(inst, placement)

    assert repaired.rects[0].as_tuple() == (0.0, 0.0, 2.0, 2.0)
    assert repaired.rects[1].width == 3.0
    assert repaired.rects[1].height == 3.0
    assert not has_overlaps(list(repaired.rects.values()))


def test_repair_skips_incompatible_mib_shape_unification():
    areas = torch.tensor([4.0, 9.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        2,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    placement = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(2.0, 0.0, 3.0, 3.0)})

    repaired = repair_placement(inst, placement)

    assert repaired.rects[0].area == 4.0
    assert repaired.rects[1].area == 9.0

