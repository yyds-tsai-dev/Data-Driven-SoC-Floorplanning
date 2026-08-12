"""`PARTNER_TAG_COMPRESS` -- compact the final layout onto preplaced tag lines.

A preplaced block cannot move, so a boundary tag it carries is satisfiable only
if the final bbox edge on that side IS the block's own edge.  `_edge_seat` can
translate, dilate, or rigidly pull at most 8 outliers; the blocks that overshoot
such a line sit in packed chains, so the residual survives to the evaluator.
This pass pushes the whole chain back onto the line, paying the residual out of
the soft blocks' unused 1% area tolerance.

What must hold:
  * flag off -> a single env lookup, the SAME list object, no import;
  * flag on -> a reachable single-block line is seated;
  * flag on -> a two-block CHAIN is seated (the propagation, not just a
    single translate, is what makes the pass different from `_edge_seat`);
  * flag on -> a line whose chain terminates on a second preplaced block is
    ABORTED, not forced (hard legality precedes soft repair);
  * flag on -> randomized layouts stay legal (no overlap, preplaced origin and
    dimensions untouched, fixed shapes untouched, soft areas inside the 1%
    hard tolerance) and the evaluator-faithful boundary+grouping+MIB total
    never rises;
  * any failure is contained.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl          # noqa: E402
import contest_optimizer as co             # noqa: E402
import layout_refiner as lr                # noqa: E402
import tag_compress as tc                  # noqa: E402

_ENV = ("PARTNER_TAG_COMPRESS", "PARTNER_TAG_COMPRESS_DEBUG",
        "PARTNER_GROUP_BRIDGE", "PARTNER_GROUP_BRIDGE_BUDGET",
        "PARTNER_GROUP_BRIDGE_DEBUG", "PARTNER_GROUP_DAG_BRIDGE",
        "PARTNER_GROUP_DAG_BRIDGE_BUDGET", "PARTNER_GROUP_DAG_BRIDGE_DEBUG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)


def _opt():
    return co.ContestOptimizer() if hasattr(co, "ContestOptimizer") \
        else co.MyOptimizer()


def _inst(areas, rects, pre_idx, tag_bit, tag_idx=None):
    """A tiny instance: `pre_idx` blocks are preplaced (locked where they
    are), `tag_idx` (default the first preplaced) carries `tag_bit`."""
    n = len(rects)
    at = torch.tensor([float(a) for a in areas])
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    for i in pre_idx:
        cons[i, 1] = 1.0
        tpos[i] = torch.tensor(list(rects[i]))
    cons[pre_idx[0] if tag_idx is None else tag_idx, 4] = float(tag_bit)
    return (n, at, cons, tpos, torch.zeros((0, 3)), torch.zeros((0, 3)),
            torch.zeros((0, 2)), rects)


def _scorer(at, cons, tpos, b2b, p2b, pins, rects):
    return csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                time.time() + 60.0, seed=0)


def _single():
    """Block 0 preplaced at x in [0, 10] and tagged RIGHT; block 1 pokes
    0.05 past the line and can pay for it out of its area tolerance."""
    rects = [(0.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 10.05, 10.0),
             (0.0, 20.0, 10.0, 10.0)]
    return _inst([100.0, 100.5, 100.0], rects, [0], 2)


def _chain():
    """Same line, but two blocks abut across it -- seating the outer one
    requires pushing the inner one too."""
    rects = [(0.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 5.0, 10.0),
             (5.0, 10.0, 5.05, 10.0)]
    return _inst([100.0, 50.0, 50.5], rects, [0], 2)


def _blocked():
    """Same line, but a SECOND preplaced block sits past it: the push would
    have to move a hard-locked block, so it must abort."""
    rects = [(0.0, 0.0, 10.0, 10.0),
             (0.0, 10.0, 10.05, 10.0),
             (0.0, 20.0, 10.0, 10.0)]
    inst = _inst([100.0, 100.5, 100.0], rects, [0, 1], 2, tag_idx=0)
    return inst


# ------------------------------------------------------------------ off path
def test_off_is_identity(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    boom = []
    monkeypatch.setattr(tc, "tag_compress",
                        lambda *a, **k: boom.append(1) or a[1])
    out = list(rects)
    got = _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    assert got is out                      # same object, no copy
    assert not boom                        # never called


def test_constructor_warms_dependencies_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(tc, "warm_dependencies",
                        lambda: calls.append("warm"))

    _opt()
    assert calls == []

    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    _opt()
    assert calls == ["warm"]


def test_group_bridge_flag_warms_dependencies_in_constructor(monkeypatch):
    calls = []
    monkeypatch.setattr(tc, "warm_dependencies",
                        lambda: calls.append("warm"))
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    _opt()
    assert calls == ["warm"]


def test_both_final_flags_off_are_identity_without_scorer(monkeypatch):
    out = [(0.0, 0.0, 1.0, 1.0)]
    monkeypatch.setattr(co, "_ColumnOptimizer",
                        lambda *a, **k: pytest.fail("scorer constructed"))
    assert _opt()._tag_compress(out, torch.ones(1), torch.zeros((1, 5)),
                                torch.full((1, 4), -1.0), torch.zeros((0, 3)),
                                torch.zeros((0, 3)), torch.zeros((0, 2)), None) is out


def test_group_bridge_works_with_tag_compress_off(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setattr(tc, "tag_compress",
                        lambda *a, **k: pytest.fail("tag pass called"))
    seen = []
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda scorer, value, budget: seen.append((budget, value)) or value + [(9., 9., 1., 1.)])
    out = list(rects)
    got = _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    assert seen and seen[0][0] == pytest.approx(0.02)
    assert got != out


def test_tag_then_bridge_share_one_scorer(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    sentinel = object()
    made = []
    events = []
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: made.append(1) or sentinel)
    monkeypatch.setattr(tc, "tag_compress", lambda scorer, value: events.append(("tag", scorer)) or value)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda scorer, value, budget: events.append(("bridge", scorer)) or value)
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert made == [1]
    assert events == [("tag", sentinel), ("bridge", sentinel)]


def test_bridge_failure_preserves_tag_result(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    accepted = [(1.0, 2.0, 3.0, 4.0)] * len(rects)
    monkeypatch.setattr(tc, "tag_compress", lambda scorer, value: accepted)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins,
                                None) is accepted


def test_group_bridge_debug_reports_self_paired_fields(monkeypatch, capsys):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE_DEBUG", "1")
    out = list(rects)
    _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    err = capsys.readouterr().err
    for field in ("ms=", "grouping=", "V=", "hpwl=", "bbox=", "committed="):
        assert field in err


def test_bridge_debug_failure_preserves_bridge_result(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE_DEBUG", "1")
    accepted = [(8.0, 8.0, 1.0, 1.0)] * len(rects)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda *a: accepted)
    monkeypatch.setattr("violation_killer._grouping_count",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("diagnostic")))
    out = list(rects)
    assert _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None) is accepted


def test_bridge_debug_ms_covers_bridge_call(monkeypatch, capsys):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE_DEBUG", "1")
    clock = [10.0]
    monkeypatch.setattr(co.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda *a: (clock.__setitem__(0, clock[0] + 0.25) or a[1]))
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert "ms=250.000" in capsys.readouterr().err


def test_dag_default_off_is_identity_and_does_not_construct_scorer(monkeypatch):
    out = [(0.0, 0.0, 1.0, 1.0)]
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: pytest.fail("scorer constructed"))
    assert _opt()._tag_compress(out, torch.ones(1), torch.zeros((1, 5)), torch.full((1, 4), -1.0), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 2)), None) is out


def test_dag_only_runs_local_then_dag_with_one_scorer(monkeypatch):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    made, events, sentinel = [], [], object()
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: made.append(1) or sentinel)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda s, v, b: events.append(("local", s)) or v)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda s, v, b: events.append(("dag", s)) or v)
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert made == [1] and events == [("local", sentinel), ("dag", sentinel)]


def test_dag_literal_zero_is_disabled(monkeypatch):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "0")
    events = []
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: events.append("local") or a[1])
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: events.append("dag") or a[1])
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert events == ["local"]


def test_dag_only_literal_zero_is_exact_identity_without_scorer(monkeypatch):
    out = [(0.0, 0.0, 1.0, 1.0)]
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "0")
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: pytest.fail("scorer constructed"))
    assert _opt()._tag_compress(out, torch.ones(1), torch.zeros((1, 5)), torch.full((1, 4), -1.0), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 2)), None) is out


def test_dag_only_warms_tag_dependencies_once(monkeypatch):
    calls = []
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setattr(tc, "warm_dependencies", lambda: calls.append("warm"))
    _opt()
    assert calls == ["warm"]


def test_all_flags_are_ordered_tag_local_dag_and_share_scorer(monkeypatch):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    monkeypatch.setenv("PARTNER_GROUP_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    made, events, sentinel = [], [], object()
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: made.append(1) or sentinel)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    tagged = [(1.0, 1.0, 1.0, 1.0)] * len(rects)
    local = [(2.0, 2.0, 1.0, 1.0)] * len(rects)
    dag = np.asarray([(3.0, 3.0, 1.0, 1.0)] * len(rects))
    monkeypatch.setattr(tc, "tag_compress", lambda s, v: events.append(("tag", s, v)) or tagged)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda s, v, b: events.append(("local", s, v)) or local)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda s, v, b: events.append(("dag", s, v)) or dag)
    got = _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert made == [1]
    assert [event[0] for event in events] == ["tag", "local", "dag"]
    assert events[1][2] is tagged and events[2][2] is local
    assert got is dag


@pytest.mark.parametrize("debug", [None, "1"])
def test_zero_residual_grouping_skips_dag_and_preserves_local_identity(monkeypatch, debug):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    if debug is not None:
        monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", debug)
    local = [(4.0, 5.0, 1.0, 1.0)] * len(rects)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 0)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: pytest.fail("DAG called"))
    assert _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None) is local


def test_dag_debug_ndarray_candidate_is_preserved_and_reports_commit(monkeypatch, capsys):
    from types import SimpleNamespace
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", "1")
    clock = [10.0]
    scorer = SimpleNamespace(_hpwl=lambda p: 2.0)
    local = [(4.0, 5.0, 1.0, 1.0)] * len(rects)
    candidate = np.asarray([(6.0, 7.0, 1.0, 1.0)] * len(rects))
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: scorer)
    monkeypatch.setattr(co.time, "perf_counter", lambda: clock.__setitem__(0, clock[0] + 0.25) or clock[0])
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer._violations_exact", lambda *a: 0)
    monkeypatch.setattr("violation_killer._bbox_area", lambda *a: 3.0)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: candidate)
    got = _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    err = capsys.readouterr().err
    assert got is candidate
    assert "ms=250.000" in err
    assert "grouping=" in err and "V=" in err and "hpwl=" in err and "bbox=" in err and "committed=1" in err


def test_dag_debug_equal_candidate_reports_zero_commit(monkeypatch, capsys):
    from types import SimpleNamespace
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", "1")
    scorer = SimpleNamespace(_hpwl=lambda p: 2.0)
    local = [(4.0, 5.0, 1.0, 1.0)] * len(rects)
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: scorer)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer._violations_exact", lambda *a: 0)
    monkeypatch.setattr("violation_killer._bbox_area", lambda *a: 3.0)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: list(local))
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert "committed=0" in capsys.readouterr().err


def test_dag_debug_diagnostic_failure_does_not_discard_candidate(monkeypatch):
    from types import SimpleNamespace
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", "1")
    local = [(4.0, 5.0, 1.0, 1.0)] * len(rects)
    candidate = np.asarray([(6.0, 7.0, 1.0, 1.0)] * len(rects))
    scorer = SimpleNamespace(_hpwl=lambda p: (_ for _ in ()).throw(RuntimeError("diag")))
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: scorer)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: candidate)
    assert _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None) is candidate


@pytest.mark.parametrize("bad", [np.zeros((1, 3)), np.full((1, 4), np.nan), object()])
def test_invalid_dag_output_preserves_exact_local_identity(monkeypatch, bad):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    local = [(4.0, 5.0, 1.0, 1.0)] * len(rects)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: bad)
    assert _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None) is local


def test_dag_env_cleanup_names_are_all_registered():
    assert {"PARTNER_GROUP_DAG_BRIDGE", "PARTNER_GROUP_DAG_BRIDGE_BUDGET", "PARTNER_GROUP_DAG_BRIDGE_DEBUG"} <= set(_ENV)


def test_dag_exception_preserves_local_identity(monkeypatch):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    local = [(3.0, 4.0, 1.0, 1.0)] * len(rects)
    called = []
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda *a: (called.append(True), (_ for _ in ()).throw(RuntimeError("dag")))[1])
    assert _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None) is local
    assert called == [True]


def test_dag_budget_fallback_and_debug_literal_one(monkeypatch, capsys):
    _, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_BUDGET", "bad")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", "true")
    seen = []
    monkeypatch.setattr("violation_killer.bridge_grouping_violations", lambda *a: a[1])
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag", lambda s, v, b: seen.append(b) or v)
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert seen == [pytest.approx(0.003)]
    assert "ms=" not in capsys.readouterr().err


def test_warm_dependencies_is_idempotent():
    previous = tc._EXACT_VIOL_FN
    try:
        tc._EXACT_VIOL_FN = None
        first = tc.warm_dependencies()
        second = tc.warm_dependencies()
        assert callable(first)
        assert second is first
    finally:
        tc._EXACT_VIOL_FN = previous


def test_failure_is_contained(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")

    def _boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(tc, "tag_compress", _boom)
    out = list(rects)
    assert _opt()._tag_compress(
        out, at, cons, tpos, b2b, p2b, pins, None) == out


def test_pure_function_swallows_a_broken_opt():
    class _Bad:
        n = 3

        def __getattr__(self, k):
            raise RuntimeError("nope")

    out = [(0.0, 0.0, 1.0, 1.0)]
    assert tc.tag_compress(_Bad(), out) is out


# ------------------------------------------------------------------- on path
@pytest.mark.parametrize("case", ["single", "chain"])
def test_on_seats_a_reachable_line(monkeypatch, case):
    n, at, cons, tpos, b2b, p2b, pins, rects = \
        _single() if case == "single" else _chain()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    sc = _scorer(at, cons, tpos, b2b, p2b, pins, rects)
    P = np.asarray(rects, dtype=np.float64)
    v0 = lr.full_violations(sc, P)
    assert v0 >= 1                         # the fixture must actually violate

    got = _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins,
                               None)
    Q = np.asarray([tuple(map(float, r)) for r in got], dtype=np.float64)
    assert lr.full_violations(sc, Q) < v0
    # the tagged block's own right edge IS the bbox right edge now
    assert abs((Q[:, 0] + Q[:, 2]).max() - 10.0) < 1e-6
    # preplaced untouched, soft areas inside the hard tolerance
    assert np.allclose(Q[0], P[0])
    for i in range(1, n):
        assert abs(Q[i, 2] * Q[i, 3] - float(at[i])) / float(at[i]) <= 0.01


def test_on_aborts_when_the_chain_hits_a_locked_block(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _blocked()
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    out = list(rects)
    got = _opt()._tag_compress(out, at, cons, tpos, b2b, p2b, pins, None)
    assert [tuple(map(float, r)) for r in got] == \
        [tuple(map(float, r)) for r in out]


def test_on_keeps_legality_and_never_raises_violations(monkeypatch):
    monkeypatch.setenv("PARTNER_TAG_COMPRESS", "1")
    rng = np.random.default_rng(11)
    o = _opt()
    for _t in range(24):
        rows, cols = 3, 3
        n = rows * cols
        rects = [None] * n
        areas = [0.0] * n
        y = 0.0
        for r in range(rows):
            x = 0.0
            rh = 0.0
            for c in range(cols):
                k = r * cols + c
                w = 10.0 + float(rng.integers(0, 30)) / 10.0
                h = 10.0 + float(rng.integers(0, 30)) / 10.0
                rects[k] = (x, y, w, h)
                areas[k] = w * h
                x += w
                rh = max(rh, h)
            y += rh
        pre = int(rng.integers(0, n))
        bit = int(rng.choice([1, 2, 4, 8]))
        n_, at, cons, tpos, b2b, p2b, pins, rects = \
            _inst(areas, rects, [pre], bit)
        sc = _scorer(at, cons, tpos, b2b, p2b, pins, rects)
        P = np.asarray(rects, dtype=np.float64)
        v0 = lr.full_violations(sc, P)
        got = o._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins,
                              None)
        Q = np.asarray([tuple(map(float, r)) for r in got], dtype=np.float64)
        assert Q.shape == P.shape
        assert lr.full_violations(sc, Q) <= v0
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(Q[i, 0] + Q[i, 2], Q[j, 0] + Q[j, 2]) \
                    - max(Q[i, 0], Q[j, 0])
                oy = min(Q[i, 1] + Q[i, 3], Q[j, 1] + Q[j, 3]) \
                    - max(Q[i, 1], Q[j, 1])
                assert not (ox > 1e-6 and oy > 1e-6), (i, j)
        assert np.allclose(Q[pre], P[pre])
        for i in range(n):
            assert abs(Q[i, 2] * Q[i, 3] - float(at[i])) \
                / float(at[i]) <= 0.01 + 1e-9
        # bbox never grows on either axis
        assert (Q[:, 0] + Q[:, 2]).max() - Q[:, 0].min() \
            <= (P[:, 0] + P[:, 2]).max() - P[:, 0].min() + 1e-9
        assert (Q[:, 1] + Q[:, 3]).max() - Q[:, 1].min() \
            <= (P[:, 1] + P[:, 3]).max() - P[:, 1].min() + 1e-9
