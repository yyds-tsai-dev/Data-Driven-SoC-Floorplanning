import math
import sys
import time
import types

import numpy as np
import pytest
import torch
from dataclasses import replace

sys.path.insert(0, "partner")
import column_sa_legalizer as csl
import violation_killer as vk


def _coordinated_chain_case():
    out = [(0., 0., 1., 1.), (1., 0., 1., 1.), (1., 2., 1., 1.), (2., 2., 1., 1.)]
    n = 4
    areas = torch.ones(n)
    constraints = torch.zeros((n, 5)); constraints[:, 0] = 1.; constraints[:, 3] = 1.
    targets = torch.full((n, 4), -1.); targets[:, 2:] = 1.
    b2b = torch.zeros((0, 3))
    pins = torch.tensor([(0.5, 0.5), (4.5, -1.5), (2.0, 2.5), (2.0, 2.5)])
    p2b = torch.tensor([(0., 0., 16.), (1., 1., 0.25), (2., 2., 16.), (3., 3., 1.)])
    opt = csl._ColumnOptimizer(out, areas, constraints, targets, b2b, p2b, pins,
                               time.time() + 60., seed=0)
    return opt, out


def test_dag_bridge_creates_positive_shared_edge_and_reduces_grouping():
    opt, out = _coordinated_chain_case(); P = np.asarray(out, float)
    assert vk.bridge_grouping_violations(opt, out, 0.2) is out
    assert len(vk._components(P, np.asarray(opt.cluster_groups[1]))) == 2
    before = vk._grouping_count(opt, P)
    got = vk.bridge_grouping_violations_dag(opt, out, 0.2); Q = np.asarray(got, float)
    assert vk._grouping_count(opt, Q) < before
    assert vk._violations_exact(opt, Q) < vk._violations_exact(opt, P)
    assert vk._Ctx(opt, P).score(Q)[0] < vk._Ctx(opt, P).score(P)[0]
    assert np.array_equal(Q[:, 2:], P[:, 2:])


def test_positive_shared_edge_rejects_corner_only_touch():
    choice = vk._ContactChoice(1, 0, 1, 0, True, 0.)
    assert vk._positive_shared_edge(np.asarray([(0.,0.,2.,2.), (2.,1.,2.,2.)]), choice)
    assert not vk._positive_shared_edge(np.asarray([(0.,0.,2.,2.), (2.,2.,2.,2.)]), choice)


def test_impossible_dag_bridge_is_exact_identity(monkeypatch):
    opt, out = _coordinated_chain_case(); monkeypatch.setattr(vk, '_solve_axis_dag', lambda p: None)
    assert vk.bridge_grouping_violations_dag(opt, out, .2) is out


def test_dag_bridge_is_repeatable_and_preserves_hard_geometry():
    opt, out = _coordinated_chain_case(); one = vk.bridge_grouping_violations_dag(opt, list(out), .2)
    two = vk.bridge_grouping_violations_dag(opt, list(out), .2)
    assert one == two and np.array_equal(np.asarray(one)[:,2:], np.asarray(out)[:,2:])


def test_dag_mechanism_never_calls_legacy_grouping_or_coord_polish(monkeypatch):
    opt, out = _coordinated_chain_case()
    monkeypatch.setattr(vk, '_fix_grouping', lambda *a, **k: pytest.fail('legacy grouping called'))
    got = vk.bridge_grouping_violations_dag(opt, out, .2)
    assert vk._grouping_count(opt, np.asarray(got)) < vk._grouping_count(opt, np.asarray(out))


def test_contact_forest_spans_each_component_without_closing_a_cycle():
    opt, out = _coordinated_chain_case(); P = np.asarray(out)
    forest = vk._contact_forest(P, opt.cluster_groups[1])
    assert forest == ((0, 0, 1), (1, 2, 3))


def test_contact_choice_clips_signed_perpendicular_delta_inside_join_bounds():
    opt, out = _coordinated_chain_case()
    chosen = next(c for c in vk._enumerate_contact_choices(opt, np.asarray(out), 4) if (c.a,c.b,c.axis,c.a_before_b)==(0,2,0,True))
    assert chosen.perp_delta == pytest.approx(1. - vk.JOIN, abs=0.)


def test_separation_edges_assign_once_with_coordinate_deltas():
    opt, out = _coordinated_chain_case(); P = np.asarray(out)
    assert vk._separation_edges(P, 0) == (vk._AxisEdge(0,1,1.), vk._AxisEdge(0,3,2.), vk._AxisEdge(2,3,1.))

def test_dag_invalid_budget_is_exact_identity():
    opt, out = _coordinated_chain_case()
    for budget in (0, -1, float("nan"), float("inf")):
        assert vk.bridge_grouping_violations_dag(opt, out, budget) is out

def test_dag_reversed_id_forest_and_equalities():
    opt, out = _coordinated_chain_case()
    P = np.asarray([(1., 0., 1., 1.), (0., 0., 1., 1.),
                    (2., 2., 1., 1.), (1., 2., 1., 1.)])
    forest = vk._contact_forest(P, opt.cluster_groups[1])
    assert len(forest) == 2
    xe, ye = vk._forest_equalities(P, forest)
    assert len(xe) == len(ye) == 2

def test_dag_pinned_conflict_is_identity():
    opt, out = _coordinated_chain_case(); P = np.asarray(out, float)
    problem = vk._make_axis_problem(P, 0, (vk._AxisEquality(0, 2, 0.),), (), [2, 1, 1, 1])
    problem = replace(problem, pinned=np.array([True, False, True, False]))
    assert vk._solve_axis_dag(problem) is None

def test_soft_profile_regression_rejects_lost_relations():
    opt, out = _coordinated_chain_case(); P = np.asarray(out, float)
    before = vk._soft_profile(opt, P)
    changed = P.copy(); changed[1, 0] += .25
    assert not vk._profile_nonregressing(before, vk._soft_profile(opt, changed))

def test_dag_component_cap_skips_solver(monkeypatch):
    opt, out = _coordinated_chain_case()
    opt.cluster_groups[1] = list(range(14))
    called = False
    monkeypatch.setattr(vk, "_solve_axis_dag", lambda p: pytest.fail("solver called"))
    assert vk.bridge_grouping_violations_dag(opt, out, .2) is out


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
