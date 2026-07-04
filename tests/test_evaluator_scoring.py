import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOORSET_ROOT = ROOT / "FloorSet"
SCRIPTS_EVALUATOR = ROOT / "scripts" / "iccad2026_evaluate.py"


def _load_scripts_evaluator():
    if str(FLOORSET_ROOT) not in sys.path:
        sys.path.insert(0, str(FLOORSET_ROOT))
    spec = importlib.util.spec_from_file_location(
        "scripts_iccad2026_evaluate",
        SCRIPTS_EVALUATOR,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_scripts_evaluator()


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


def test_feasible_cost_is_capped_below_infeasible_penalty():
    cost = evaluator.compute_cost(100.0, 100.0, 1.0, 100.0, True)

    assert cost == evaluator.M_PENALTY - 1e-6
    assert cost < evaluator.compute_cost(0.0, 0.0, 0.0, 1.0, False)


def test_no_runtime_total_uses_v10_exp_n_over_12_weighting():
    results = [
        evaluator.TestResult(0, 21, True, 0.0, 0.0, 0.0, 1.0, cost=2.0, cost_no_runtime=1.0),
        evaluator.TestResult(1, 120, True, 0.0, 0.0, 0.0, 8.0, cost=8.0, cost_no_runtime=3.0),
    ]

    total = evaluator.compute_total_score(
        [r.cost_no_runtime for r in results],
        [r.block_count for r in results],
    )
    weights = [math.exp((21 - 120) / 12), math.exp((120 - 120) / 12)]
    expected = (1.0 * weights[0] + 3.0 * weights[1]) / sum(weights)

    assert total == expected


def test_score_contributors_are_weighted_by_block_count():
    results = [
        evaluator.TestResult(i - 21, i, True, 0.0, 0.0, 0.0, 1.0, cost=1.0, cost_no_runtime=1.0)
        for i in range(21, 121)
    ]

    contributors = evaluator.summarize_score_contributors(results, limit=100)
    weights = [math.exp((i - 120) / 12) for i in range(21, 121)]
    expected_weight_120 = weights[-1] / sum(weights)
    expected_weight_116_120 = sum(weights[-5:]) / sum(weights)

    assert contributors[0]["test_id"] == 99
    assert contributors[0]["block_count"] == 120
    assert contributors[0]["score_weight"] == expected_weight_120
    assert 0.079 < contributors[0]["score_weight"] < 0.081
    assert sum(row["score_weight"] for row in contributors[:5]) == pytest.approx(expected_weight_116_120)
    assert 0.34 < expected_weight_116_120 < 0.342


def test_eval_env_loader_prefers_dotenv_checkpoint(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FLOORSET_GNN_CHECKPOINT=checkpoints/from-dotenv.pt\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLOORSET_GNN_CHECKPOINT", "checkpoints/stale-shell.pt")

    evaluator.load_env_defaults(env_file, override_keys={"FLOORSET_GNN_CHECKPOINT"})

    assert os.environ["FLOORSET_GNN_CHECKPOINT"] == "checkpoints/from-dotenv.pt"


def test_verbose_eval_env_labels_gnn_checkpoint_as_fallback(
    monkeypatch, tmp_path, capsys
):
    (tmp_path / "src").mkdir()
    (tmp_path / ".env").write_text(
        "FLOORSET_GNN_CHECKPOINT=checkpoints/from-dotenv.pt\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "find_repo_root", lambda start=None: tmp_path)
    monkeypatch.delenv("FLOORSET_GNN_CHECKPOINT", raising=False)
    monkeypatch.delenv("FLOORSET_GNN_CHECKPOINT_SOURCE", raising=False)

    evaluator.load_repo_env_defaults(verbose=True)

    captured = capsys.readouterr()
    assert "Using GNN fallback checkpoint:" in captured.out
    assert "Using checkpoint:" not in captured.out


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
