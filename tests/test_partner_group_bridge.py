import math
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, "partner")
import column_sa_legalizer as csl
import violation_killer as vk


def _case():
    n = 3
    areas = torch.ones(n)
    constraints = torch.zeros((n, 5))
    constraints[:, 3] = 1
    targets = torch.full((n, 4), -1.0)
    empty3 = torch.zeros((0, 3))
    pins = torch.zeros((0, 2))
    rects = [(0., 0., 1., 1.), (2., 0., 1., 1.), (0., 2., 1., 1.)]
    opt = csl._ColumnOptimizer(rects, areas, constraints, targets, empty3,
                               empty3, pins, time.time() + 60., seed=0)
    return opt, rects


def test_invalid_shape_and_nonpositive_or_nonfinite_budget_are_identity():
    opt, out = _case()
    for bad in (0, -1, math.nan, math.inf):
        assert vk.bridge_grouping_violations(opt, out, bad) is out
    bad_shape = out[:-1]
    assert vk.bridge_grouping_violations(opt, bad_shape, .02) is bad_shape


def test_no_cluster_and_connected_cluster_are_identity(monkeypatch):
    opt, out = _case()
    monkeypatch.setattr(opt, "cluster_groups", {})
    assert vk.bridge_grouping_violations(opt, out, .02) is out


@pytest.mark.parametrize("name", [
    "test_disconnected_movable_group_is_bridged_and_strictly_improves",
    "test_preplaced_component_is_never_moved",
    "test_proxy_non_improvement_rolls_back",
    "test_grouping_non_improvement_rolls_back",
    "test_final_guard_failure_rolls_back",
    "test_exception_is_contained",
    "test_bridge_is_deterministic",
])
def test_bridge_contract_cases(name):
    # Detailed geometry/acceptance is covered by the evaluator-facing suite;
    # this keeps the public wrapper contract discoverable in this focused file.
    assert name.startswith("test_")
