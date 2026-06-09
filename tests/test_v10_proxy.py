from __future__ import annotations

import torch

from floorset_arch.models import Placement, Rect
from floorset_arch.parser import parse_instance
from floorset_arch.v10_proxy import (
    hard_legality_rank,
    v10_proxy_better,
    v10_proxy_cost,
    v10_proxy_rank,
)


def _inst(
    block_count: int = 3,
    *,
    boundary: int = 0,
    fixed: bool = False,
    preplaced: bool = False,
):
    constraints = torch.zeros(block_count, 5)
    if boundary:
        constraints[:boundary, 4] = 1.0
    if fixed:
        constraints[0, 0] = 1.0
    if preplaced:
        constraints[0, 1] = 1.0
    targets = torch.full((block_count, 4), -1.0)
    if fixed:
        targets[0] = torch.tensor([-1.0, -1.0, 2.0, 2.0])
    if preplaced:
        targets[0] = torch.tensor([0.0, 0.0, 2.0, 2.0])
    return parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        targets,
    )


def test_hard_legality_rank_penalizes_missing_overlap_fixed_and_preplaced():
    inst = _inst(3, fixed=True, preplaced=False)
    legal = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(3.0, 0.0, 2.0, 2.0),
            2: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    missing = Placement(
        {0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(3.0, 0.0, 2.0, 2.0)}
    )
    overlap = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(1.0, 0.0, 2.0, 2.0),
            2: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    fixed_bad = Placement(
        {
            0: Rect(0.0, 0.0, 1.0, 4.0),
            1: Rect(3.0, 0.0, 2.0, 2.0),
            2: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )

    assert hard_legality_rank(inst, legal) < hard_legality_rank(inst, missing)
    assert hard_legality_rank(inst, legal) < hard_legality_rank(inst, overlap)
    assert hard_legality_rank(inst, legal) < hard_legality_rank(inst, fixed_bad)

    preplaced_inst = _inst(3, preplaced=True)
    preplaced_bad = Placement(
        {
            0: Rect(1.0, 0.0, 2.0, 2.0),
            1: Rect(3.0, 0.0, 2.0, 2.0),
            2: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    assert hard_legality_rank(preplaced_inst, legal) < hard_legality_rank(
        preplaced_inst, preplaced_bad
    )


def test_v10_proxy_cost_balances_quality_and_soft_penalty():
    inst = _inst(3, boundary=1)
    compact_dirty = Placement(
        {
            0: Rect(4.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
        }
    )
    huge_clean = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(200.0, 0.0, 2.0, 2.0),
            2: Rect(202.0, 0.0, 2.0, 2.0),
        }
    )

    assert v10_proxy_cost(inst, compact_dirty) < v10_proxy_cost(inst, huge_clean)
    assert v10_proxy_rank(inst, compact_dirty) < v10_proxy_rank(inst, huge_clean)


def test_v10_proxy_better_rejects_proxy_regression_even_when_soft_improves():
    inst = _inst(3, boundary=1)
    current = Placement(
        {
            0: Rect(4.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
        }
    )
    huge_clean = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(200.0, 0.0, 2.0, 2.0),
            2: Rect(202.0, 0.0, 2.0, 2.0),
        }
    )

    assert not v10_proxy_better(inst, huge_clean, current)


def test_v10_proxy_better_allows_soft_tie_within_tolerance(monkeypatch):
    inst = _inst(3, boundary=1)
    current = Placement(
        {
            0: Rect(4.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
        }
    )
    soft_better = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 3.0, 2.0, 2.0),
            2: Rect(2.0, 3.0, 2.0, 2.0),
        }
    )

    monkeypatch.setattr(
        "floorset_arch.v10_proxy.v10_proxy_cost",
        lambda _inst, placement, metrics=None: 1.0,
    )

    assert v10_proxy_better(inst, soft_better, current)
