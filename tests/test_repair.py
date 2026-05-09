import torch

from floorset_arch.geometry import Rect, bbox, edge_touch_length, has_overlaps
from floorset_arch.models import Placement
from floorset_arch.parser import parse_instance
from floorset_arch.repair import repair_placement, soft_violation_counts


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


def test_repair_connects_cluster_members_when_space_is_available():
    areas = torch.tensor([4.0, 4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        3,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(8.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
        }
    )

    repaired = repair_placement(inst, placement)

    assert not has_overlaps(list(repaired.rects.values()))
    assert edge_touch_length(repaired.rects[0], repaired.rects[1]) > 0.0


def test_boundary_repair_tries_alternate_edge_slots_when_current_slot_overlaps():
    areas = torch.tensor([4.0, 4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        3,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(8.0, 0.0, 2.0, 2.0),
            2: Rect(8.0, 3.0, 2.0, 2.0),
        }
    )

    repaired = repair_placement(inst, placement)
    bounds = bbox(list(repaired.rects.values()))

    assert not has_overlaps(list(repaired.rects.values()))
    assert abs(repaired.rects[0].right - bounds.right) <= 1e-6


def test_guarded_repair_reduces_soft_violation_counts():
    areas = torch.tensor([4.0, 4.0, 4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        4,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )

    before = sum(soft_violation_counts(inst, placement))
    repaired = repair_placement(inst, placement)
    after = sum(soft_violation_counts(inst, repaired))

    assert not has_overlaps(list(repaired.rects.values()))
    assert after < before
