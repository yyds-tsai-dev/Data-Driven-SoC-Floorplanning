from __future__ import annotations

import json
import math
import importlib.util
import sys
import random
import time
from pathlib import Path
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
from icdc.topology_data import CorpusSourceReceipt, TopologyLabel, fingerprint_case
from icdc.topology_data import SparseEdge
from icdc.topology_prior import _separation_edges


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
    assert all(
        isinstance(call["args"][1], torch.Tensor)
        and tuple(call["args"][1].shape) == (2, 5)
        for call in scorer.calls
    )


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


def test_base_with_soft_group_v_retains_sparse_label():
    out = evaluate_case(
        _rects(1.0),
        _rects(),
        _case(soft_group=True),
        _Scorer([1.2]),
        sample_seed=17,
        admit=lambda proposal, case: None,
    )
    assert out.winner == "production-base"
    assert out.teacher_status == "admission_failed"
    assert out.sparse_label is not None
    assert out.sparse_label.instance_id == _case()["instance_id"]


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


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/probes/icdc_fp_teacher_g0.py"
    spec = importlib.util.spec_from_file_location("icdc_fp_teacher_g0_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_production_environment_is_exact_3d3f():
    runner = _runner_module()
    assert runner.production_environment() == {
        "PARTNER_NREF": "6",
        "PARTNER_FLOW_SLOTS": "3",
        "PARTNER_OVERSAMPLE": "1",
        "PARTNER_DIRECT_SOLVER": "dpmpp",
        "PARTNER_DDIM_STEPS": "2",
        "PARTNER_FLOW_SOLVER": "euler",
        "PARTNER_FLOW_STEPS": "8",
    }


def test_runner_portfolio_contract_is_fail_closed():
    runner = _runner_module()
    valid = {
        "pool_ready": True,
        "requested_k": 6,
        "raw_candidate_count": 6,
        "direct_count": 3,
        "flow_count": 3,
        "oversample": False,
        "flow_exception": None,
        "fallback": False,
    }
    assert runner.validate_portfolio_receipt(valid) == valid
    for field, bad in (
        ("pool_ready", False),
        ("requested_k", 5),
        ("raw_candidate_count", 7),
        ("direct_count", 4),
        ("flow_count", 2),
        ("oversample", True),
        ("flow_exception", "boom"),
        ("fallback", True),
    ):
        with pytest.raises(ValueError, match="portfolio"):
            runner.validate_portfolio_receipt({**valid, field: bad})


def test_runner_canonical_sparse_label_contains_no_dense_geometry():
    runner = _runner_module()
    label = TopologyLabel(
        "worker_0/layouts_0.th#0", 2, 17, 1.1, 1.4, 1.4 / 1.1, (), (), ()
    )
    encoded = runner.canonical_sparse_label(label)
    decoded = json.loads(encoded)
    assert decoded["instance_id"] == "worker_0/layouts_0.th#0"
    for forbidden in (b"fp_sol", b"fp_xywh", b"positions", b"rects", b"golden"):
        assert forbidden not in encoded


def test_runner_streams_numeric_shards_once_and_publishes_only_sparse_evidence(
    tmp_path,
):
    runner = _runner_module()
    events = []

    def iter_shards(_root):
        return [
            (1, 10, tmp_path / "worker_1/layouts_10.th"),
            (0, 2, tmp_path / "worker_0/layouts_2.th"),
        ]

    def read_shard(_root, worker, layout):
        events.append(("read", worker, layout))
        return f"raw-{worker}-{layout}".encode(), [torch.ones((1, 1, 6))]

    def validate_source(_source):
        return 1, 120

    def make_row(_source, relative_path, digest, index):
        iid = f"{relative_path}#{index}"
        case = {
            "instance_id": iid,
            "n": 1,
            "area": [4.0],
            "cons": [[0, 0, 0, 0, 0]],
            "tp": [[-1.0, -1.0, -1.0, -1.0]],
            "b2b": [],
            "p2b": [],
            "pins": [],
            "hpwl_ref": 1.0,
            "area_ref": 4.0,
        }
        return runner.TrainingRow(
            CorpusSourceReceipt(relative_path, digest, index, fingerprint_case(case)),
            case,
            torch.tensor([[9.0, 8.0, 2.0, 2.0]], dtype=torch.float64),
        )

    def solve_base(_optimizer, row):
        events.append(("solve", row.case["instance_id"]))
        return runner.BaseSolveResult(
            torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float64),
            {
                "pool_ready": True,
                "requested_k": 6,
                "raw_candidate_count": 6,
                "direct_count": 3,
                "flow_count": 3,
                "oversample": False,
                "flow_exception": None,
                "fallback": False,
            },
        )

    def evaluate(base, fp_seed, case, scorer, *, sample_seed):
        events.append(("evaluate", case["instance_id"]))
        assert fp_seed.tolist() == [[9.0, 8.0, 2.0, 2.0]]
        label = TopologyLabel(
            case["instance_id"], 1, sample_seed, 1.0, 1.2, 1.2, (), (), ()
        )
        return G0CaseResult(
            case["instance_id"], 1, sample_seed, 1.2, 1.0, 1.0,
            "transient-fp-exact-tfdl", "winner", label,
            {"ok": True}, {"ok": True},
        )

    deps = runner.RuntimeDependencies(
        iter_shards=iter_shards,
        read_shard=read_shard,
        validate_source=validate_source,
        make_row=make_row,
        build_optimizer=lambda: object(),
        solve_base=solve_base,
        scorer=object(),
        evaluate=evaluate,
        bindings=lambda: {"test_binding": "a" * 64},
    )
    out_dir = tmp_path / "out"
    summary = runner.run_g0(
        tmp_path,
        out_dir,
        n_min=1,
        heldout_mod=1,
        max_files=2,
        deps=deps,
    )
    assert summary.terminal_state == "NON_AUTHORIZING_TRACER"
    assert events == [
        ("read", 0, 2),
        ("solve", "worker_0/layouts_2.th#0"),
        ("evaluate", "worker_0/layouts_2.th#0"),
        ("read", 1, 10),
        ("solve", "worker_1/layouts_10.th#0"),
        ("evaluate", "worker_1/layouts_10.th#0"),
    ]
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "cases.jsonl",
        "labels.jsonl",
        "manifest.json",
        "population.json",
    ]
    for path in out_dir.iterdir():
        payload = path.read_bytes()
        for forbidden in (
            b"fp_sol", b"fp_xywh", b"positions", b"rects", b"golden",
            b"9.0", b"8.0",
        ):
            assert forbidden not in payload


def _legacy_separation_edges(rects, n):
    edges = []
    for axis in (0, 1):
        candidates = []
        for first in range(n):
            for second in range(first + 1, n):
                gx = max(
                    rects[first][0] - rects[second][0] - rects[second][2],
                    rects[second][0] - rects[first][0] - rects[first][2],
                )
                gy = max(
                    rects[first][1] - rects[second][1] - rects[second][3],
                    rects[second][1] - rects[first][1] - rects[first][3],
                )
                if (axis == 0 and gx < gy) or (axis == 1 and gx >= gy):
                    continue
                ci = rects[first][axis] + rects[first][axis + 2] / 2
                cj = rects[second][axis] + rects[second][axis + 2] / 2
                src, dst = (
                    (first, second)
                    if (ci, first) <= (cj, second)
                    else (second, first)
                )
                margin = max(
                    0.0,
                    rects[dst][axis]
                    - rects[src][axis]
                    - rects[src][axis + 2],
                )
                candidates.append((src, dst, margin))
        for src, dst, margin in candidates:
            reachable = set()
            frontier = [
                value
                for start, value, _ in candidates
                if start == src and value != dst
            ]
            while frontier:
                node = frontier.pop()
                if node in reachable:
                    continue
                reachable.add(node)
                frontier.extend(
                    value for start, value, _ in candidates if start == node
                )
            if dst not in reachable:
                edges.append(SparseEdge(src, dst, axis, margin, "sep", 1.0))
    return edges


def test_bitset_separation_edges_are_exactly_legacy_equivalent():
    rng = random.Random(20260814)
    for n in range(2, 15):
        rects = [
            [
                rng.uniform(-10, 10),
                rng.uniform(-10, 10),
                rng.uniform(0.1, 4),
                rng.uniform(0.1, 4),
            ]
            for _ in range(n)
        ]
        assert _separation_edges(rects, n) == _legacy_separation_edges(rects, n)


def test_separation_edges_n112_compiles_under_two_seconds():
    rng = random.Random(17)
    rects = [
        [
            rng.uniform(-100, 100),
            rng.uniform(-100, 100),
            rng.uniform(0.1, 10),
            rng.uniform(0.1, 10),
        ]
        for _ in range(112)
    ]
    started = time.perf_counter()
    result = _separation_edges(rects, 112)
    elapsed = time.perf_counter() - started
    assert result
    assert elapsed < 2.0
