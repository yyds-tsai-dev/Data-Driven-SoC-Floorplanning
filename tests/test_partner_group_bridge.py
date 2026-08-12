import math
import sys
import time

import numpy as np
import pytest
import torch
from dataclasses import replace

sys.path.insert(0, "partner")
import column_sa_legalizer as csl
import violation_killer as vk


def _case(preplaced=False):
    n = 3
    areas = torch.ones(n)
    constraints = torch.zeros((n, 5))
    constraints[:, 3] = 1
    targets = torch.full((n, 4), -1.0)
    if preplaced:
        constraints[0, 1] = 1
        targets[0] = torch.tensor([0., 0., 1., 1.])
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
    opt, out = _case(preplaced=True)
    q = vk.bridge_grouping_violations(opt, out, .5)
    p, qn = np.asarray(out, float), np.asarray(q, float)
    assert q != out
    assert vk._grouping_count(opt, qn) < vk._grouping_count(opt, p)
    assert vk._violations_exact(opt, qn) < vk._violations_exact(opt, p)
    assert vk._Ctx(opt, p).score(qn)[0] < vk._Ctx(opt, p).score(p)[0]
    assert vk._final_guards_ok(opt, p, qn, list(opt.kind), list(opt.areas))
    assert np.array_equal(np.asarray(q)[0], np.asarray(out)[0])


def _rollback_case(monkeypatch, gate):
    opt, out = _case()
    candidate = np.asarray([(0., 0., 1., 1.), (1., 0., 1., 1.),
                            (2., 0., 1., 1.)])
    if gate == "proxy":
        monkeypatch.setattr(vk._Ctx, "score", lambda self, p: (2., 0) if np.array_equal(p, candidate) else (2., 2))
    elif gate == "grouping":
        monkeypatch.setattr(vk, "_grouping_count", lambda o, p: 2)
    else:
        monkeypatch.setattr(vk, "_final_guards_ok", lambda *a: False)
    monkeypatch.setattr(vk, "_fix_grouping",
                        lambda *a: (candidate.copy(), 0., 0))
    assert vk.bridge_grouping_violations(opt, out, .02) is out


def test_proxy_non_improvement_rolls_back(monkeypatch):
    _rollback_case(monkeypatch, "proxy")


def test_grouping_non_improvement_rolls_back(monkeypatch):
    _rollback_case(monkeypatch, "grouping")


def test_final_guard_failure_rolls_back(monkeypatch):
    _rollback_case(monkeypatch, "guards")


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


def test_axis_solver_rejects_bad_equality_cycle_pin_and_bounds():
    base = vk._AxisProblem(np.array([0., 2.]), np.ones(2),
                           np.array([-10., -10.]), np.array([10., 10.]),
                           np.array([False, False]), (), ())
    assert vk._solve_axis_dag(replace(base, equalities=(
        vk._AxisEquality(0, 1, 0.), vk._AxisEquality(1, 0, 1.)))) is None
    assert vk._solve_axis_dag(replace(base, edges=(vk._AxisEdge(0, 0, 1.),))) is None
    pinned = replace(base, pinned=np.array([True, True]),
                     equalities=(vk._AxisEquality(0, 1, 3.),))
    assert vk._solve_axis_dag(pinned) is None
    cycle = replace(base, edges=(vk._AxisEdge(0, 1, 1.), vk._AxisEdge(1, 0, 1.)))
    assert vk._solve_axis_dag(cycle) is None
    bounded = replace(base, lower=np.array([0., 0.]), upper=np.array([0., 1.]),
                      edges=(vk._AxisEdge(0, 1, 2.),))
    assert vk._solve_axis_dag(bounded) is None


def test_axis_solver_preserves_pin_satisfies_edges_and_is_repeatable():
    problem = vk._AxisProblem(
        coords=np.array([0., 7., 11.]), sizes=np.array([2., 2., 1.]),
        lower=np.array([0., 0., 0.]), upper=np.array([0., 20., 20.]),
        pinned=np.array([True, False, False]),
        equalities=(vk._AxisEquality(1, 2, 4.),),
        edges=(vk._AxisEdge(0, 1, 3.),),)
    first = vk._solve_axis_dag(problem)
    second = vk._solve_axis_dag(problem)
    assert first is not None and np.array_equal(first, second)
    assert first[0] == 0.
    assert first[1] >= first[0] + 3.
    assert first[2] == pytest.approx(first[1] + 4., abs=1e-9)


@pytest.mark.parametrize("bad_index", [0.0, True, -1, 2])
def test_axis_solver_rejects_malformed_indices(bad_index):
    base = vk._AxisProblem(np.zeros(2), np.ones(2), np.full(2, -10.),
                           np.full(2, 10.), np.zeros(2, dtype=bool), (), ())
    assert vk._solve_axis_dag(replace(base, equalities=(vk._AxisEquality(bad_index, 1, 0.),))) is None
    assert vk._solve_axis_dag(replace(base, edges=(vk._AxisEdge(0, bad_index, 0.),))) is None


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), -float("inf")])
def test_axis_solver_rejects_nonfinite_deltas(delta):
    base = vk._AxisProblem(np.zeros(2), np.ones(2), np.full(2, -10.),
                           np.full(2, 10.), np.zeros(2, dtype=bool), (), ())
    assert vk._solve_axis_dag(replace(base, equalities=(vk._AxisEquality(0, 1, delta),))) is None
    assert vk._solve_axis_dag(replace(base, edges=(vk._AxisEdge(0, 1, delta),))) is None


@pytest.mark.parametrize("field", ["coords", "sizes", "lower", "upper"])
@pytest.mark.parametrize("bad", [np.array(["x", "y"]), np.array([1 + 0j, 2 + 0j]), np.array([True, False]), np.array([object(), object()])])
def test_axis_solver_rejects_malformed_numeric_arrays(field, bad):
    base = vk._AxisProblem(np.zeros(2), np.ones(2), np.full(2, -10.),
                           np.full(2, 10.), np.zeros(2, dtype=bool), (), ())
    assert vk._solve_axis_dag(replace(base, **{field: bad})) is None


def test_axis_solver_rejects_non_bool_pinned_and_nonfinite_fields():
    base = vk._AxisProblem(np.zeros(2), np.ones(2), np.full(2, -10.),
                           np.full(2, 10.), np.zeros(2, dtype=bool), (), ())
    assert vk._solve_axis_dag(replace(base, pinned=np.array([0, 1]))) is None
    for field in ("coords", "sizes", "lower", "upper"):
        bad = np.zeros(2); bad[0] = np.nan
        assert vk._solve_axis_dag(replace(base, **{field: bad})) is None


def test_axis_solver_rejects_malformed_top_level_and_containers():
    base = vk._AxisProblem(np.zeros(2), np.ones(2), np.full(2, -10.),
                           np.full(2, 10.), np.zeros(2, dtype=bool), (), ())
    for bad in (None, 1, "x"):
        assert vk._solve_axis_dag(bad) is None
    for field in ("coords", "sizes", "lower", "upper", "pinned"):
        for bad in (0., [0., 0.], (0., 0.)):
            assert vk._solve_axis_dag(replace(base, **{field: bad})) is None
    for field in ("equalities", "edges"):
        for bad in (None, [], "x", [base.equalities]):
            assert vk._solve_axis_dag(replace(base, **{field: bad})) is None
    assert vk._solve_axis_dag(replace(base, equalities=(object(),))) is None
    assert vk._solve_axis_dag(replace(base, edges=(object(),))) is None


@pytest.mark.parametrize("value", ["x", object(), 1 + 0j, True, float("nan"), float("inf"), 10 ** 400, -(10 ** 400)])
def test_axis_solver_rejects_malformed_scalar_gaps(value):
    base = vk._AxisProblem(np.zeros(2), np.ones(2), np.full(2, -10.),
                           np.full(2, 10.), np.zeros(2, dtype=bool), (), ())
    assert vk._solve_axis_dag(replace(base, equalities=(vk._AxisEquality(0, 1, value),))) is None
    assert vk._solve_axis_dag(replace(base, edges=(vk._AxisEdge(0, 1, value),))) is None
