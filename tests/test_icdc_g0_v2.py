from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from icdc.g0_v2 import (
    G0CaseResult,
    PopulationAccumulator,
    canonical_case_record,
    evaluate_case,
    transient_fp_xywh,
)


def _source(tree_value: float = 0.0) -> list[torch.Tensor]:
    inp = torch.tensor(
        [[[9.0, 0, 0, 0, 0, 0], [25.0, 0, 0, 0, 0, 0]]],
        dtype=torch.float32,
    )
    empty3 = torch.empty((1, 0, 3), dtype=torch.float32)
    empty2 = torch.empty((1, 0, 2), dtype=torch.float32)
    tree = torch.full((1, 1, 3), tree_value, dtype=torch.float32)
    fp = torch.tensor(
        [[[3.0, 4.0, 10.0, 20.0], [5.0, 6.0, 30.0, 40.0]]],
        dtype=torch.float32,
    )
    metrics = torch.ones((1, 8), dtype=torch.float32)
    return [inp, empty3, empty3.clone(), empty2, tree, fp, metrics]


def _case(*, soft_group: bool = False, preplaced: bool = False) -> dict:
    cons = [
        [0, int(preplaced), 0, 7 if soft_group else 0, 0],
        [0, 0, 0, 7 if soft_group else 0, 0],
    ]
    return {
        "instance_id": "worker_0/layouts_0.th#0",
        "n": 2,
        "area": [4.0, 4.0],
        "cons": cons,
        "tp": [[0.0, 0.0, 2.0, 2.0], [-1.0, -1.0, -1.0, -1.0]]
        if preplaced
        else [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
        "b2b": [],
        "p2b": [],
        "pins": [],
        "hpwl_ref": 1.0,
        "area_ref": 8.0,
    }


class _Scorer:
    def __init__(self, costs: list[float], feasible: list[bool] | None = None):
        self.costs = iter(costs)
        self.feasible = iter(feasible or [True] * len(costs))
        self.calls: list[dict] = []

    def evaluate_solution(self, solution, *args, **kwargs):
        self.calls.append({"solution": solution, "args": args, "kwargs": kwargs})
        return SimpleNamespace(
            is_feasible=next(self.feasible), cost_no_runtime=next(self.costs)
        )


def _admit_identity(proposal: torch.Tensor, _case: dict):
    return proposal.clone(), torch.zeros((proposal.shape[0], 2), dtype=torch.float64)


def _rects(offset: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [2.0 + offset, 0.0, 2.0, 2.0]],
        dtype=torch.float64,
    )


def test_transient_fp_converts_whxy_to_xywh_and_tree_is_invariant():
    first = transient_fp_xywh(_source(0.0), 0, 2)
    second = transient_fp_xywh(_source(9.0), 0, 2)
    assert first.dtype is torch.float64 and first.device.type == "cpu"
    assert first.tolist() == [
        [10.0, 20.0, 3.0, 4.0],
        [30.0, 40.0, 5.0, 6.0],
    ]
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("row", "n", "message"),
    [(-1, 2, "row"), (1, 2, "row"), (0, 3, "block count")],
)
def test_transient_fp_rejects_bad_row_or_count(row, n, message):
    with pytest.raises(ValueError, match=message):
        transient_fp_xywh(_source(), row, n)


def test_soft_v_fp_is_admitted_and_official_cost_selects_winner():
    scorer = _Scorer([1.4, 1.1])
    out = evaluate_case(
        _rects(1.0),
        _rects(0.0),
        _case(soft_group=True),
        scorer,
        sample_seed=17,
        admit=_admit_identity,
    )
    assert out.winner == "transient-fp-exact-tfdl"
    assert out.base_cost == 1.4
    assert out.teacher_cost == 1.1
    assert out.teacher_candidate_cost == 1.1
    assert out.teacher_status == "winner"
    assert len(scorer.calls) == 2
    assert all(call["kwargs"]["median_runtime"] == 1.0 for call in scorer.calls)
    assert all(call["solution"]["runtime"] == 1.0 for call in scorer.calls)


@pytest.mark.parametrize(
    ("teacher_cost", "admitted", "status"),
    [(2.0, True, "not_improved"), (None, False, "admission_failed")],
)
def test_nonimproving_or_rejected_teacher_retains_base(
    teacher_cost, admitted, status
):
    scorer = _Scorer([1.25] + ([] if teacher_cost is None else [teacher_cost]))
    admit = _admit_identity if admitted else lambda proposal, case: None
    out = evaluate_case(
        _rects(1.0), _rects(), _case(), scorer, sample_seed=17, admit=admit
    )
    assert out.winner == "production-base"
    assert out.base_cost == 1.25
    assert out.teacher_cost == 1.25
    assert out.teacher_candidate_cost == teacher_cost
    assert out.teacher_status == status
    assert len(scorer.calls) == 1 + int(admitted)


def test_shifted_preplaced_transient_fp_is_rejected_before_scoring():
    scorer = _Scorer([1.2])
    candidate = _rects()
    candidate[0, 0] = 0.25
    out = evaluate_case(
        _rects(), candidate, _case(preplaced=True), scorer, sample_seed=17
    )
    assert out.winner == "production-base"
    assert out.teacher_status == "admission_failed"
    assert len(scorer.calls) == 1


def test_case_result_and_canonical_json_contain_no_dense_coordinates():
    out = evaluate_case(
        _rects(1.0),
        _rects(),
        _case(),
        _Scorer([1.4, 1.1]),
        sample_seed=17,
        admit=_admit_identity,
    )
    encoded = canonical_case_record(out)
    decoded = json.loads(encoded)
    assert encoded.endswith(b"\n")
    assert set(decoded) == {
        "base_cost",
        "base_hard_audit",
        "instance_id",
        "n",
        "sample_seed",
        "teacher_candidate_cost",
        "teacher_cost",
        "teacher_hard_audit",
        "teacher_status",
        "winner",
    }
    for forbidden in (b"fp_sol", b"fp_xywh", b"positions", b"rects", b"golden"):
        assert forbidden not in encoded
    with pytest.raises(FrozenInstanceError):
        out.winner = "production-base"


def _result(instance_id: str, n: int, base: float, teacher: float) -> G0CaseResult:
    return G0CaseResult(
        instance_id=instance_id,
        n=n,
        sample_seed=17,
        base_cost=base,
        teacher_cost=teacher,
        teacher_candidate_cost=teacher,
        winner="transient-fp-exact-tfdl" if teacher < base else "production-base",
        teacher_status="winner" if teacher < base else "not_improved",
        sparse_label=None,
        base_hard_audit={"ok": True},
        teacher_hard_audit={"ok": True},
    )


def _weighted(costs: list[float], counts: list[int]) -> float:
    weights = [math.exp(n / 12.0) for n in counts]
    return sum(c * w for c, w in zip(costs, weights)) / sum(weights)


def test_population_uses_exp_n_over_12_and_requires_complete_registration():
    pop = PopulationAccumulator()
    pop.register("a", 100)
    pop.register("b", 112)
    pop.add(_result("a", 100, 1.3, 1.1))
    incomplete = pop.finish(complete=True)
    assert incomplete.terminal_state == "KILLED_LEGALITY_OR_COVERAGE"
    pop.add(_result("b", 112, 1.2, 1.0))
    summary = pop.finish(complete=True)
    expected_base = _weighted([1.3, 1.2], [100, 112])
    expected_teacher = _weighted([1.1, 1.0], [100, 112])
    assert summary.base_h == pytest.approx(expected_base)
    assert summary.teacher_h == pytest.approx(expected_teacher)
    assert summary.delta_h == pytest.approx(expected_base - expected_teacher)
    assert summary.terminal_state == "TARGET_GAIN_MET"


def test_population_rejects_duplicates_and_nonfinite_values():
    pop = PopulationAccumulator()
    pop.register("a", 100)
    with pytest.raises(ValueError, match="duplicate"):
        pop.register("a", 100)
    with pytest.raises(ValueError, match="registered"):
        pop.add(_result("b", 100, 1.0, 1.0))
    with pytest.raises(ValueError, match="cost"):
        pop.add(_result("a", 100, float("inf"), 1.0))


@pytest.mark.parametrize(
    ("base", "teacher", "terminal"),
    [
        (1.60, 1.51, "KILLED_TEACHER_GT_1_5"),
        (1.20, 1.19, "STOP_HARD_GAIN_MISSED"),
        (1.20, 1.18, "TARGET_GAIN_MISSED_NO_TRAINING_AUTHORITY"),
        (1.20, 1.17, "TARGET_GAIN_MET"),
    ],
)
def test_population_terminal_state_precedence(base, teacher, terminal):
    pop = PopulationAccumulator()
    pop.register("a", 100)
    pop.add(_result("a", 100, base, teacher))
    assert pop.finish(complete=True).terminal_state == terminal


def test_population_tracer_never_authorizes_training():
    pop = PopulationAccumulator(authorizing=False)
    pop.register("a", 100)
    pop.add(_result("a", 100, 1.2, 1.0))
    summary = pop.finish(complete=True)
    assert summary.authorizing is False
    assert summary.terminal_state == "NON_AUTHORIZING_TRACER"
