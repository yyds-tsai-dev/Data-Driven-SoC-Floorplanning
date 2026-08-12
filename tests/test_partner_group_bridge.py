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
    opt, out = _case()
    connected = [(0., 0., 1., 1.), (1., 0., 1., 1.), (2., 0., 1., 1.)]
    assert vk.bridge_grouping_violations(opt, connected, .02) is connected


def test_disconnected_movable_group_is_bridged_and_strictly_improves():
    opt, out = _case()
    q = vk.bridge_grouping_violations(opt, out, .5)
    p, qn = np.asarray(out, float), np.asarray(q, float)
    assert vk._grouping_count(opt, qn) < vk._grouping_count(opt, p)
    assert vk._violations_exact(opt, qn) < vk._violations_exact(opt, p)
    assert vk._Ctx(opt, p).score(qn)[0] < vk._Ctx(opt, p).score(p)[0]
    assert vk._final_guards_ok(opt, p, qn, list(opt.kind), list(opt.areas))


def test_preplaced_component_is_never_moved():
    opt, out = _case()
    opt.preplaced[0] = True
    opt.kind[0] = 2
    q = vk.bridge_grouping_violations(opt, out, .5)
    assert np.array_equal(np.asarray(q)[0], np.asarray(out)[0])


@pytest.mark.parametrize("gate", ["proxy", "grouping", "guards"])
def test_proxy_non_improvement_rolls_back(monkeypatch, gate):
    opt, out = _case()
    candidate = np.asarray([(0., 0., 1., 1.), (1., 0., 1., 1.),
                            (2., 0., 1., 1.)])
    if gate == "proxy":
        monkeypatch.setattr(vk._Ctx, "score", lambda self, p: (1., 0))
    elif gate == "grouping":
        monkeypatch.setattr(vk, "_grouping_count", lambda o, p: 2)
    else:
        monkeypatch.setattr(vk, "_final_guards_ok", lambda *a: False)
    monkeypatch.setattr(vk, "_fix_grouping",
                        lambda *a: (candidate.copy(), 0., 0))
    assert vk.bridge_grouping_violations(opt, out, .02) is out


def test_grouping_non_improvement_rolls_back(monkeypatch):
    test_proxy_non_improvement_rolls_back(monkeypatch, "grouping")


def test_final_guard_failure_rolls_back(monkeypatch):
    test_proxy_non_improvement_rolls_back(monkeypatch, "guards")


def test_exception_is_contained(monkeypatch):
    opt, out = _case()
    monkeypatch.setattr(vk, "_fix_grouping", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    assert vk.bridge_grouping_violations(opt, out, .02) is out


def test_bridge_is_deterministic():
    opt, out = _case()
    assert vk.bridge_grouping_violations(opt, list(out), .5) == vk.bridge_grouping_violations(opt, list(out), .5)


def test_nonfinite_layout_is_identity():
    opt, out = _case()
    bad = list(out); bad[1] = (float("nan"), 0., 1., 1.)
    assert vk.bridge_grouping_violations(opt, bad, .02) is bad
