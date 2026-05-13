import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

CONTEST_DIR = Path(__file__).resolve().parents[1] / "FloorSet" / "iccad2026contest"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

import iccad2026_evaluate as evaluator  # noqa: E402


def test_compute_cost_can_disable_runtime_adjustment():
    with_runtime = evaluator.compute_cost(0.1, 0.2, 0.25, 4.0, True)
    without_runtime = evaluator.compute_cost(0.1, 0.2, 0.25, 4.0, True, use_runtime=False)

    expected_quality = 1 + evaluator.ALPHA * (0.1 + 0.2)
    expected_violation = math.exp(evaluator.BETA * 0.25)

    assert with_runtime > without_runtime
    assert without_runtime == expected_quality * expected_violation


def test_compute_cost_breakdown_exposes_formula_factors():
    breakdown = evaluator.compute_cost_breakdown(0.1, -0.2, 0.25, 4.0, True)

    assert breakdown["positive_hpwl_gap"] == 0.1
    assert breakdown["positive_area_gap"] == 0
    assert breakdown["quality_factor"] == 1 + evaluator.ALPHA * 0.1
    assert breakdown["violation_factor"] == math.exp(evaluator.BETA * 0.25)
    assert breakdown["runtime_factor"] == 4.0
    assert breakdown["runtime_adjustment"] == math.pow(4.0, evaluator.GAMMA)
    assert breakdown["cost"] == (
        breakdown["quality_factor"]
        * breakdown["violation_factor"]
        * breakdown["runtime_adjustment"]
    )


def test_no_runtime_total_uses_no_runtime_costs():
    results = [
        evaluator.TestResult(0, 21, True, 0.0, 0.0, 0.0, 1.0, cost=2.0, cost_no_runtime=1.0),
        evaluator.TestResult(1, 120, True, 0.0, 0.0, 0.0, 8.0, cost=8.0, cost_no_runtime=3.0),
    ]

    total = evaluator.compute_total_score(
        [r.cost_no_runtime for r in results],
        [r.block_count for r in results],
    )

    assert total > 2.99


def test_score_contributors_are_weighted_by_block_count():
    results = [
        evaluator.TestResult(0, 21, True, 0.0, 0.0, 0.0, 1.0, cost=10.0, cost_no_runtime=10.0),
        evaluator.TestResult(1, 120, True, 0.0, 0.0, 0.0, 1.0, cost=2.0, cost_no_runtime=2.0),
    ]

    contributors = evaluator.summarize_score_contributors(results)

    assert contributors[0]["test_id"] == 1
    assert contributors[0]["cost"] == 2.0
    assert contributors[0]["block_count"] == 120
    assert contributors[0]["score_contribution"] > contributors[1]["score_contribution"]
    assert contributors[0]["score_contribution_percent"] > 99.0


def test_eval_env_loader_prefers_dotenv_checkpoint(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FLOORSET_GNN_CHECKPOINT=checkpoints/from-dotenv.pt\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLOORSET_GNN_CHECKPOINT", "checkpoints/stale-shell.pt")

    evaluator.load_env_defaults(env_file, override_keys={"FLOORSET_GNN_CHECKPOINT"})

    assert os.environ["FLOORSET_GNN_CHECKPOINT"] == "checkpoints/from-dotenv.pt"


def test_evaluate_uses_monotonic_timer_for_runtime(monkeypatch):
    evaluator_obj = evaluator.ContestEvaluator(data_path="../", verbose=False)
    evaluator_obj.dataset = [
        {
            "input": (
                torch.tensor([4.0]),
                torch.empty(0, 3),
                torch.empty(0, 3),
                torch.empty(0, 2),
                torch.zeros(1, 5),
            ),
            "label": None,
        }
    ]
    monkeypatch.setattr(evaluator_obj, "_load_dataset", lambda: None)
    monkeypatch.setattr(evaluator_obj, "_extract_baseline", lambda *_args: ({"hpwl_baseline": 1.0, "area_baseline": 1.0}, None))
    monkeypatch.setattr(evaluator_obj, "_load_optimizer", lambda _path: SimpleNamespace(solve=lambda *_args: [(0.0, 0.0, 2.0, 2.0)]))

    def fake_evaluate_solution(solution, *_args, **_kwargs):
        return evaluator.SolutionMetrics(
            is_feasible=True,
            overlap_violations=0,
            area_violations=0,
            dimension_violations=0,
            hpwl_b2b=0.0,
            hpwl_p2b=0.0,
            hpwl_total=0.0,
            hpwl_baseline=1.0,
            hpwl_gap=0.0,
            bbox_area=4.0,
            bbox_area_baseline=1.0,
            area_gap=0.0,
            fixed_violations=0,
            preplaced_violations=0,
            boundary_violations=0,
            grouping_violations=0,
            mib_violations=0,
            total_soft_violations=0,
            max_possible_violations=1,
            violations_relative=0.0,
            runtime_seconds=solution["runtime"],
            cost=1.0,
            cost_no_runtime=1.0,
        )

    monkeypatch.setattr(evaluator, "evaluate_solution", fake_evaluate_solution)
    wall_clock = iter([100.0, 90.0])
    monotonic = iter([10.0, 11.5])
    monkeypatch.setattr(evaluator.time, "time", lambda: next(wall_clock))
    monkeypatch.setattr(evaluator.time, "perf_counter", lambda: next(monotonic))

    result = evaluator_obj.evaluate("fake_optimizer.py", test_ids=[0])

    assert result.test_results[0].runtime_seconds == 1.5
