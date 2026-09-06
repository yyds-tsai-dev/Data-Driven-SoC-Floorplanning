import math
import sys
import time
import types
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from dataclasses import replace

sys.path.insert(0, "partner")
import column_sa_legalizer as csl
import violation_killer as vk


def _load_probe_module(name="group_dag_bridge_probe"):
    path = Path("scripts/probes/group_dag_bridge_probe.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _two_case_g0_baseline():
    return {"test_results": [
        {"test_id": i, "block_count": 2, "positions": [[0., 0., 1., 1.]],
         "cost_no_runtime": 1.0, "grouping_violations": 1,
         "total_soft_violations": 1} for i in range(2)]}


def _fixture_loader(test_id, row):
    return {"test_id": test_id}


def _fixture_bridge(ctx, positions, budget_s):
    return [[float(positions[0][0]) + 1., 0., 1., 1.]]


def _fixture_evaluate(solution, ctx):
    changed = solution["positions"][0][0] != 0
    return {"cost_no_runtime": .5 if changed else 1., "is_feasible": True,
            "grouping_violations": 0 if changed else 1,
            "total_soft_violations": 0 if changed else 1}


def test_g0_probe_writes_separate_deterministic_case_and_manifest_files(tmp_path, monkeypatch):
    probe = _load_probe_module(); source = tmp_path / "baseline.json"
    source.write_text(json.dumps(_two_case_g0_baseline(), sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE", 1.0)
    out, manifest = tmp_path / "cases.json", tmp_path / "manifest.json"
    args = ["replay", "--input", str(source), "--output", str(out), "--manifest", str(manifest)]
    tick = iter([0., .0001] * 4)
    assert probe.main(args, case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=_fixture_evaluate, clock=lambda: next(tick)) == 0
    first = (out.read_bytes(), manifest.read_bytes())
    tick = iter([0., .0001] * 4)
    assert probe.main(args, case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=_fixture_evaluate, clock=lambda: next(tick)) == 0
    assert first == (out.read_bytes(), manifest.read_bytes())
    m = json.loads(manifest.read_text())
    assert set(m) == {"schema", "baseline", "feasible", "errors", "score_on", "score_off", "grouping_delta", "v_delta", "runtime_ms", "causal_mean_ms", "accepted", "source_hygiene"}
    assert m["schema"] == "group-dag-g0.v1"
    assert m["baseline"]["cases_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_g0_probe_fails_closed_on_baseline_hash_or_score_mismatch(tmp_path, monkeypatch):
    probe = _load_probe_module(); source = tmp_path / "baseline.json"; source.write_text(json.dumps(_two_case_g0_baseline()))
    out, manifest = tmp_path / "o.json", tmp_path / "m.json"; out.write_text("old"); manifest.write_text("old")
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", "0" * 64)
    assert probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(manifest)]) != 0
    assert out.read_text() == manifest.read_text() == "old"


def test_g0_probe_changed_candidate_requires_strict_deltas_and_gate_threshold(tmp_path, monkeypatch):
    probe = _load_probe_module(); assert probe.SCORE_LIMIT == 1.1412448258795715
    source = tmp_path / "b.json"; source.write_text(json.dumps(_two_case_g0_baseline(), sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE", 1.0)
    def non_strict(solution, ctx):
        changed = solution["positions"][0][0] != 0
        return {"cost_no_runtime": .5 if changed else 1.0, "is_feasible": True,
                "grouping_violations": 1, "total_soft_violations": 1}
    out, man = tmp_path / "o.json", tmp_path / "m.json"
    rc = probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)],
                    case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=non_strict)
    assert rc == 1
    report = json.loads(man.read_text()); rows = json.loads(out.read_text())
    assert report["errors"] == 0 and report["feasible"] == 2 and report["accepted"] is False
    assert all(row["changed"] and not row["candidate_strict"] and not row["accepted"] for row in rows)


def test_g0_probe_off_replay_binds_each_case_not_only_weighted_aggregate(tmp_path, monkeypatch):
    probe = _load_probe_module(); source = tmp_path / "b.json"; source.write_text(json.dumps(_two_case_g0_baseline(), sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE", 1.0)
    def offset(solution, ctx):
        changed = solution["positions"][0][0] != 0
        if not changed:
            cost = .9 if ctx["test_id"] == 0 else 1.1
        else:
            cost = .5
        return {"cost_no_runtime": cost, "is_feasible": True, "grouping_violations": 0 if changed else 1, "total_soft_violations": 0 if changed else 1}
    out, man = tmp_path / "o.json", tmp_path / "m.json"
    rc = probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)], case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=offset)
    assert rc == 1
    report = json.loads(man.read_text()); assert report["errors"] == 2


def test_g0_probe_source_hygiene_reachable_ast_only(tmp_path):
    probe = _load_probe_module()
    source = Path("src/solver/violation_killer.py").read_text()
    assert probe._source_hygiene_text(source)
    mutated = source.replace(
        '    if os.environ.get("PARTNER_GROUP_DAG_BRIDGE_DEBUG") != "1":',
        '    coord_polish.forbidden()\n    if os.environ.get("PARTNER_GROUP_DAG_BRIDGE_DEBUG") != "1":',
        1,
    )
    assert probe._source_hygiene_text(mutated) is False


def test_g0_probe_writes_case_hash_link_and_rejects_duplicate_ids(tmp_path, monkeypatch):
    probe = _load_probe_module(); source = tmp_path / "b.json"; data = _two_case_g0_baseline(); data["test_results"][1]["test_id"] = 0; source.write_text(json.dumps(data, sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()); monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE", 1.0)
    out, man = tmp_path / "o.json", tmp_path / "m.json"; assert probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)], case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=_fixture_evaluate) != 0
    assert not out.exists() and not man.exists()


def test_g0_probe_runtime_and_infeasibility_fail_the_gate(tmp_path, monkeypatch):
    probe = _load_probe_module(); source = tmp_path / "b.json"; source.write_text(json.dumps(_two_case_g0_baseline(), sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()); monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE", 1.0)
    out, man = tmp_path / "o.json", tmp_path / "m.json"
    ticks = iter([0., .001, .002, .003])
    assert probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)], case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=_fixture_evaluate, clock=lambda: next(ticks)) == 1
    assert json.loads(man.read_text())["causal_mean_ms"] > .75
    def infeasible(solution, ctx):
        changed = solution["positions"][0][0] != 0
        return {"cost_no_runtime": .5 if changed else 1.0,
                "is_feasible": False if changed else True,
                "grouping_violations": 0 if changed else 1,
                "total_soft_violations": 0 if changed else 1}
    assert probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)], case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=infeasible) == 1
    report = json.loads(man.read_text()); rows = json.loads(out.read_text())
    assert report["errors"] == 0 and report["feasible"] == 0 and report["accepted"] is False
    assert all(row["changed"] and not row["candidate_strict"] for row in rows)


def test_g0_probe_rejects_error_and_explicit_bad_schema(tmp_path, monkeypatch):
    probe = _load_probe_module(); source = tmp_path / "b.json"; data = _two_case_g0_baseline(); data["schema"] = "wrong"; source.write_text(json.dumps(data, sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()); monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE", 1.0)
    out, man = tmp_path / "o.json", tmp_path / "m.json"
    assert probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)], case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=_fixture_evaluate) == 2
    data.pop("schema"); source.write_text(json.dumps(data, sort_keys=True)); monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    def fail(*args):
        raise TypeError("scorer failure")
    assert probe.main(["replay", "--input", str(source), "--output", str(out), "--manifest", str(man)], case_loader=_fixture_loader, bridge_fn=_fixture_bridge, evaluate_fn=fail) == 1
    assert json.loads(man.read_text())["errors"] == 2


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
    fake_coord_polish = types.SimpleNamespace(
        polish_layout=lambda *a, **k: pytest.fail('coord polish called')
    )
    monkeypatch.setitem(sys.modules, 'coord_polish', fake_coord_polish)
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


def _soft_profile_regression_cases():
    before = np.asarray([
        (0., 0., 1., 1.),
        (0., 2., 1., 1.),
        (1., 2., 1., 1.),
        (4., 0., 1., 1.),
        (0., 4., 1., 1.),
    ])
    n = len(before)
    areas = torch.ones(n)
    constraints = torch.zeros((n, 5))
    constraints[:, 0] = 1.
    constraints[0, 4] = 1 | 2 | 4 | 8
    constraints[1:3, 2] = 1.
    constraints[1:3, 3] = 1.
    targets = torch.full((n, 4), -1.)
    targets[:, 2:] = 1.
    empty3 = torch.zeros((0, 3))
    pins = torch.zeros((0, 2))
    opt = csl._ColumnOptimizer(before, areas, constraints, targets, empty3,
                               empty3, pins, time.time() + 60., seed=0)
    assert 1 in opt.mib_groups
    bit_swap = before.copy()
    bit_swap[0, :2] = (4., 4.)
    pair_loss = before.copy()
    pair_loss[2, :2] = (2., 2.)
    return opt, before, bit_swap, pair_loss


def test_soft_profile_preserves_exact_boundary_group_and_mib_relations():
    opt, before, bit_swap, pair_loss = _soft_profile_regression_cases()
    p0 = vk._soft_profile(opt, before)
    assert p0.boundary_satisfied == frozenset({(0, 1), (0, 8)})
    assert p0.grouping_connected == frozenset({(1, 1, 2)})
    assert p0.mib_equal == frozenset({(1, 1, 2)})
    assert not vk._profile_nonregressing(p0, vk._soft_profile(opt, bit_swap))
    assert not vk._profile_nonregressing(p0, vk._soft_profile(opt, pair_loss))


def test_contact_forest_equalities_keep_exact_offsets_for_normal_and_reversed_ids():
    opt, out = _coordinated_chain_case()
    normal = np.asarray(out, float)
    forest = vk._contact_forest(normal, opt.cluster_groups[1])
    assert forest == ((0, 0, 1), (1, 2, 3))
    x_eq, y_eq = vk._forest_equalities(normal, forest)
    assert [(e.left, e.right, e.delta) for e in x_eq] == [(0, 1, 1.), (2, 3, 1.)]
    assert [(e.left, e.right, e.delta) for e in y_eq] == [(0, 1, 0.), (2, 3, 0.)]
    reversed_ids = np.asarray([
        (1., 0., 1., 1.), (0., 0., 1., 1.),
        (2., 2., 1., 1.), (1., 2., 1., 1.),
    ])
    reversed_forest = vk._contact_forest(reversed_ids, opt.cluster_groups[1])
    assert reversed_forest == ((0, 0, 1), (1, 2, 3))
    rx_eq, ry_eq = vk._forest_equalities(reversed_ids, reversed_forest)
    assert [(e.left, e.right, e.delta) for e in rx_eq] == [(0, 1, -1.), (2, 3, -1.)]
    assert [(e.left, e.right, e.delta) for e in ry_eq] == [(0, 1, 0.), (2, 3, 0.)]


def test_separation_edges_include_exact_y_axis_delta_tuple():
    _, out = _coordinated_chain_case()
    P = np.asarray(out, float)
    assert vk._separation_edges(P, 1) == (
        vk._AxisEdge(0, 2, 2.),
        vk._AxisEdge(1, 2, 2.),
        vk._AxisEdge(1, 3, 2.),
    )


def test_valid_more_than_twelve_components_skip_solver_before_edge_build(monkeypatch):
    n = 26
    P = np.asarray([(float(3 * i), 0., 1., 1.) for i in range(n)])
    opt = types.SimpleNamespace(n=n, cluster_groups={1: list(range(n))})
    original = [tuple(row) for row in P]
    monkeypatch.setattr(vk, "_enumerate_separation_edges",
                        lambda *args, **kwargs: pytest.fail("edge enumeration called"))
    monkeypatch.setattr(vk, "_solve_axis_dag",
                        lambda *args: pytest.fail("solver called"))
    assert vk.bridge_grouping_violations_dag(opt, original, .2) is original


def _two_component_raw_cap_case():
    n = 182
    P = np.zeros((n, 4), dtype=float)
    P[:, 2:] = 1.
    P[:91, 1] = np.arange(91, dtype=float)
    P[91:, 0] = 1000.
    P[91:, 1] = np.arange(91, dtype=float)
    opt = types.SimpleNamespace(
        n=n,
        cluster_groups={1: list(range(n))},
        kind=np.ones(n, dtype=int),
    )
    choice = vk._ContactChoice(1, 0, 91, 0, True, 0.)
    return opt, P, choice


def test_raw_separation_cap_stops_at_cap_plus_one_before_solver(monkeypatch):
    opt, P, choice = _two_component_raw_cap_case()
    result = vk._enumerate_separation_edges(
        P, 0, time.perf_counter() + 10., max_edges=8192
    )
    assert result.over_cap is True
    assert result.timed_out is False
    assert len(result.edges) == 8193
    monkeypatch.setattr(vk, "_solve_axis_dag",
                        lambda *args: pytest.fail("solver called"))
    assert vk._project_changed_contact(opt, P, choice, 10.) is None


def test_expired_first_raw_axis_never_enters_second_builder(monkeypatch):
    opt, P, choice = _two_component_raw_cap_case()
    calls = []

    def fake_edges(layout, axis, deadline, max_edges=8192):
        calls.append(axis)
        return vk._EdgeEnumeration((), over_cap=False, timed_out=(axis == 0))

    monkeypatch.setattr(vk, "_enumerate_separation_edges", fake_edges)
    assert vk._project_changed_contact(opt, P, choice, 10.) is None
    assert calls == [0]


def test_fake_clock_expiry_after_first_raw_axis_short_circuits_second(monkeypatch):
    opt, P, choice = _two_component_raw_cap_case()
    calls = []
    clock = [0.0]
    monkeypatch.setattr(vk.time, "perf_counter", lambda: clock[0])

    def fake_edges(layout, axis, deadline, max_edges=8192):
        calls.append(axis)
        if axis == 0:
            clock[0] = deadline + 1.0
        return vk._EdgeEnumeration((), over_cap=False, timed_out=False)

    monkeypatch.setattr(vk, "_enumerate_separation_edges", fake_edges)
    assert vk._project_changed_contact(
        opt, P, choice, 10., deadline=10.0
    ) is None
    assert calls == [0]


def test_dag_fake_clock_expiry_after_second_solver_keeps_exact_input_identity(monkeypatch):
    opt, out = _coordinated_chain_case()
    clock = [100.0]
    monkeypatch.setattr(vk.time, "perf_counter", lambda: clock[0])
    original_solver = vk._solve_axis_dag
    solver_calls = []

    def advancing_solver(problem):
        solver_calls.append(problem)
        result = original_solver(problem)
        if len(solver_calls) == 2:
            clock[0] = 100.25
        return result

    monkeypatch.setattr(vk, "_solve_axis_dag", advancing_solver)
    got = vk.bridge_grouping_violations_dag(opt, out, .2)
    assert len(solver_calls) == 2
    assert got is out


def test_project_changed_contact_rejects_malformed_inputs_directly():
    opt, out = _coordinated_chain_case()
    P = np.asarray(out, float)
    choice = vk._ContactChoice(1, 0, 2, 0, True, 1. - vk.JOIN)
    for bad in (None, np.zeros((3, 4)), np.full((4, 4), np.nan),
                np.asarray([(0., 0., 0., 1.), *out[1:]])):
        assert vk._project_changed_contact(opt, bad, choice, 1.) is None
    for bad_choice in (
        replace(choice, axis=2),
        replace(choice, a=-1),
        replace(choice, b=99),
        replace(choice, group_id=99),
        replace(choice, perp_delta=float("nan")),
    ):
        assert vk._project_changed_contact(opt, P, bad_choice, 1.) is None
    malformed = types.SimpleNamespace(n=4, cluster_groups={1: [0, 2, 99, 3]})
    assert vk._project_changed_contact(malformed, P, choice, 1.) is None


def test_dag_debug_output_is_stable_and_default_off(monkeypatch, capsys):
    opt, out = _coordinated_chain_case()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", "1")
    assert vk.bridge_grouping_violations_dag(opt, out, 0.) is out
    debug = capsys.readouterr().out
    assert "dag_bridge" in debug
    assert "reason=invalid_budget" in debug
    assert "elapsed_ms=" in debug
    monkeypatch.delenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG")
    assert vk.bridge_grouping_violations_dag(opt, out, 0.) is out
    assert capsys.readouterr().out == ""
