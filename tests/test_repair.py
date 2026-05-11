import torch

from floorset_arch.geometry import Rect, bbox, edge_touch_length, has_overlaps
from floorset_arch.models import Placement
from floorset_arch.parser import parse_instance
from floorset_arch.repair import _overlap_repair_candidate_limit, _repair_boundary, repair_placement, soft_violation_counts


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


def test_boundary_repair_keeps_already_satisfied_block_in_place():
    areas = torch.tensor([4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
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
    placement = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(4.0, 0.0, 2.0, 2.0)})

    _repair_boundary(inst, placement)

    assert placement.rects[0].as_tuple() == (0.0, 0.0, 2.0, 2.0)


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


def test_large_case_boundary_repair_searches_wider_axis_candidates(monkeypatch):
    block_count = 118
    areas = torch.full((block_count,), 1.0)
    constraints = torch.zeros(block_count, 5)
    constraints[0, 4] = 2.0
    inst = parse_instance(
        block_count,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    rects = {0: Rect(0.0, 0.0, 1.0, 1.0)}
    for block in range(1, block_count):
        if block <= 40:
            rects[block] = Rect(10.0, float(block - 1), 1.0, 1.0)
        else:
            x = float((block - 41) % 10)
            y = 100.0 + float((block - 41) // 10)
            rects[block] = Rect(x, y, 1.0, 1.0)
    placement = Placement(rects)
    monkeypatch.setenv("FLOORSET_BOUNDARY_AXIS_CAP", "24")
    monkeypatch.setenv("FLOORSET_BOUNDARY_CROSS_AXIS_CAP", "160")

    repaired = repair_placement(inst, placement)
    bounds = bbox(list(repaired.rects.values()))

    assert not has_overlaps(list(repaired.rects.values()))
    assert abs(repaired.rects[0].right - bounds.right) <= 1e-6


def test_boundary_repair_searches_far_cross_axis_slot_for_large_edge_case(monkeypatch):
    block_count = 100
    areas = torch.full((block_count,), 1.0)
    constraints = torch.zeros(block_count, 5)
    constraints[0, 4] = 2.0
    inst = parse_instance(
        block_count,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    rects = {0: Rect(0.0, 0.0, 1.0, 1.0)}
    for block in range(1, 31):
        rects[block] = Rect(10.0, float(block - 1), 1.0, 1.0)
    for block in range(31, block_count):
        rects[block] = Rect(0.0, 100.0 + float(block), 1.0, 1.0)
    placement = Placement(rects)
    monkeypatch.setenv("FLOORSET_BOUNDARY_AXIS_CAP", "24")

    _repair_boundary(inst, placement)
    bounds = bbox(list(placement.rects.values()))

    assert abs(placement.rects[0].right - bounds.right) <= 1e-6
    assert placement.rects[0].y >= 30.0
    assert not has_overlaps(list(placement.rects.values()))


def test_large_case_boundary_pass_can_override_axis_cap(monkeypatch):
    block_count = 118
    areas = torch.full((block_count,), 1.0)
    constraints = torch.zeros(block_count, 5)
    constraints[0, 4] = 2.0
    inst = parse_instance(
        block_count,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    rects = {0: Rect(0.0, 0.0, 1.0, 1.0)}
    for block in range(1, block_count):
        if block <= 40:
            rects[block] = Rect(10.0, float(block - 1), 1.0, 1.0)
        else:
            x = float((block - 41) % 10)
            y = 100.0 + float((block - 41) // 10)
            rects[block] = Rect(x, y, 1.0, 1.0)
    placement = Placement(rects)
    monkeypatch.setenv("FLOORSET_BOUNDARY_AXIS_CAP", "24")
    monkeypatch.setenv("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP", "160")

    _repair_boundary(inst, placement)
    bounds = bbox(list(placement.rects.values()))

    assert not has_overlaps(list(placement.rects.values()))
    assert abs(placement.rects[0].right - bounds.right) <= 1e-6


def test_overlap_repair_candidate_limit_is_env_controlled(monkeypatch):
    monkeypatch.delenv("FLOORSET_OVERLAP_REPAIR_CANDIDATES", raising=False)
    inst = parse_instance(
        119,
        torch.ones(119),
        torch.zeros(1699, 3),
        torch.zeros(908, 3),
        torch.empty(0, 2),
        torch.zeros(119, 5),
        None,
    )

    assert _overlap_repair_candidate_limit(inst) == 24

    monkeypatch.setenv("FLOORSET_OVERLAP_REPAIR_CANDIDATES", "64")

    assert _overlap_repair_candidate_limit(inst) == 64
