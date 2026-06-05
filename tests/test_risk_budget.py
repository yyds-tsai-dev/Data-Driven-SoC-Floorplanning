import math
from types import SimpleNamespace

import torch

from floorset_arch.risk_budget import (
    BudgetTier,
    instance_risk_budget,
    v10_score_share,
)


def _inst(
    block_count,
    *,
    b2b=0,
    p2b=0,
    boundary=0,
    cluster_sizes=(),
    mib_sizes=(),
    fixed=0,
    preplaced=0,
):
    return SimpleNamespace(
        block_count=block_count,
        valid_b2b=torch.zeros(b2b, 3),
        valid_p2b=torch.zeros(p2b, 3),
        boundary={i: 1 for i in range(boundary)},
        cluster_groups={
            idx + 1: list(range(size))
            for idx, size in enumerate(cluster_sizes)
        },
        mib_groups={
            idx + 1: list(range(size))
            for idx, size in enumerate(mib_sizes)
        },
        fixed=set(range(fixed)),
        preplaced=set(range(fixed, fixed + preplaced)),
    )


def test_v10_score_share_matches_exp_n_over_12():
    share_120 = v10_score_share(120)
    expected = math.exp((120 - 120) / 12) / sum(
        math.exp((n - 120) / 12) for n in range(21, 121)
    )

    assert share_120 == expected
    assert 0.079 < share_120 < 0.081


def test_medium_large_dense_case_gets_budget_without_118_block_gate():
    inst = _inst(
        104,
        b2b=4300,
        p2b=2100,
        boundary=30,
        cluster_sizes=(18, 12, 10),
        mib_sizes=(8, 7),
        fixed=6,
        preplaced=4,
    )

    budget = instance_risk_budget(inst)

    assert budget.block_count == 104
    assert budget.score_share > 0.0
    assert budget.constraint_density > 0.5
    assert budget.net_density > 60.0
    assert budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}


def test_120_block_low_density_case_is_not_automatically_heavy():
    inst = _inst(
        120,
        b2b=120,
        p2b=120,
        boundary=2,
        cluster_sizes=(),
        mib_sizes=(),
        fixed=0,
        preplaced=0,
    )

    budget = instance_risk_budget(inst)

    assert 0.079 < budget.score_share < 0.081
    assert budget.constraint_density < 0.05
    assert budget.net_density == 2.0
    assert budget.tier in {BudgetTier.NONE, BudgetTier.LIGHT}


def test_budget_tier_can_drive_high_risk_without_id_specific_logic():
    dense_medium = _inst(
        104,
        b2b=4300,
        p2b=2100,
        boundary=30,
        cluster_sizes=(18, 12, 10),
        mib_sizes=(8, 7),
        fixed=6,
        preplaced=4,
    )
    sparse_tail = _inst(120, b2b=120, p2b=120, boundary=2)

    assert instance_risk_budget(dense_medium).tier in {
        BudgetTier.MEDIUM,
        BudgetTier.HEAVY,
    }
    assert instance_risk_budget(sparse_tail).tier in {
        BudgetTier.NONE,
        BudgetTier.LIGHT,
    }
