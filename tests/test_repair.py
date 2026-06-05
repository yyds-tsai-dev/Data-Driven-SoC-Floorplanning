from types import SimpleNamespace

import torch

import floorset_arch.repair as repair_module
from floorset_arch.geometry import Rect, bbox, edge_touch_length, has_overlaps
from floorset_arch.models import Placement, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.repair import (
    _geometry_preserving_refine,
    _overlap_repair_candidate_limit,
    _repair_boundary,
    _score_better_v10_soft,
    _shrink_satisfied_boundary_edges,
    _v10_soft_repair_eligible,
    repair_placement,
    soft_violation_counts,
)


def _soft_test_instance():
    areas = torch.full((4,), 4.0)
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return parse_instance(
        4,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )


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


def test_v10_soft_repair_is_disabled_by_default(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.delenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", raising=False)
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.HEAVY),
    )

    assert not _v10_soft_repair_eligible(inst, placement)


def test_v10_soft_repair_allows_medium_and_heavy_when_enabled(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")

    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.MEDIUM),
    )
    assert _v10_soft_repair_eligible(inst, placement)

    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.HEAVY),
    )
    assert _v10_soft_repair_eligible(inst, placement)


def test_v10_soft_repair_light_tier_requires_soft_pressure(monkeypatch):
    inst = _soft_test_instance()
    low_soft = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    high_soft = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")
    monkeypatch.setenv("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_SOFT", "3")
    monkeypatch.setenv("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_RELATIVE", "1.0")
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.LIGHT),
    )

    assert not _v10_soft_repair_eligible(inst, low_soft)
    assert _v10_soft_repair_eligible(inst, high_soft)


def test_v10_soft_repair_skips_none_risk_tier(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.NONE),
    )

    assert not _v10_soft_repair_eligible(inst, placement)


def test_v10_soft_accepts_grouping_improvement_with_grouping_slack(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(7.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    scores = {id(current): 100.0, id(candidate): 111.0}
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, placement: scores[id(placement)],
    )

    assert _score_better_v10_soft(inst, SolverConfig(), candidate, current)


def test_v10_soft_rejects_boundary_only_improvement_beyond_boundary_slack(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(7.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    scores = {id(current): 100.0, id(candidate): 107.0}
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, placement: scores[id(placement)],
    )

    assert not _score_better_v10_soft(inst, SolverConfig(), candidate, current)


def test_v10_soft_rejects_overlap_regression_even_when_soft_improves(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, _placement: 1.0,
    )

    assert not _score_better_v10_soft(inst, SolverConfig(), candidate, current)


def test_v10_soft_requires_geometry_improvement_when_soft_does_not_improve(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(8.0, 0.0, 2.0, 2.0),
        }
    )
    scores = {id(current): 100.0, id(candidate): 99.98}
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, placement: scores[id(placement)],
    )

    assert _score_better_v10_soft(inst, SolverConfig(), candidate, current)

    scores[id(candidate)] = 99.995

    assert not _score_better_v10_soft(inst, SolverConfig(), candidate, current)


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


def test_no_guidance_geometry_refine_moves_far_block_without_soft_regression(monkeypatch):
    areas = torch.full((4,), 4.0)
    inst = parse_instance(
        4,
        areas,
        torch.tensor([[0.0, 3.0, 10.0], [1.0, 3.0, 10.0], [2.0, 3.0, 10.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(4, 5),
        None,
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(100.0, 100.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_GEOMETRY_REFINE_MAX_BLOCKS", "16")
    before_bounds = bbox(list(placement.rects.values()))

    refined = _geometry_preserving_refine(inst, placement, SolverConfig(local_search_moves=16))
    after_bounds = bbox(list(refined.rects.values()))

    assert not has_overlaps(list(refined.rects.values()))
    assert soft_violation_counts(inst, refined) == soft_violation_counts(inst, placement)
    assert after_bounds.area < before_bounds.area
    assert refined.rects[3].x < placement.rects[3].x


def test_boundary_edge_shrink_pulls_satisfied_right_edge_inward():
    areas = torch.full((4,), 4.0)
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
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
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(98.0, 4.0, 2.0, 2.0),
            3: Rect(98.0, 6.0, 2.0, 2.0),
        }
    )

    refined = _shrink_satisfied_boundary_edges(inst, placement)
    bounds = bbox(list(refined.rects.values()))

    assert not has_overlaps(list(refined.rects.values()))
    assert soft_violation_counts(inst, refined) == (0, 0, 0)
    assert bounds.right < 100.0
    assert abs(refined.rects[2].right - bounds.right) <= 1e-6
    assert abs(refined.rects[3].right - bounds.right) <= 1e-6
