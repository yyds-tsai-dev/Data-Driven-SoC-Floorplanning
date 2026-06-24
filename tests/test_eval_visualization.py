import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_evaluator():
    root = Path.cwd()
    for path in (root / "FloorSet", root / "FloorSet/iccad2026contest"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    path = Path("scripts/iccad2026_evaluate.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator()
EvalTestResult = evaluator.TestResult
save_predicted_floorplan_pngs = evaluator.save_predicted_floorplan_pngs


def test_save_predicted_floorplan_pngs_writes_single_case(tmp_path):
    pytest.importorskip("matplotlib")
    result = SimpleNamespace(
        submission_name="unit",
        test_results=[
            EvalTestResult(
                test_id=7,
                block_count=2,
                is_feasible=True,
                hpwl_gap=0.0,
                area_gap=0.0,
                violations_relative=0.0,
                runtime_seconds=0.1,
                cost=3.25,
                positions=[(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 3.0, 3.0)],
            )
        ],
    )

    written = save_predicted_floorplan_pngs(result, tmp_path, top_k=10)

    assert len(written) == 1
    assert written[0].name == "case_7_cost_3.2500.png"
    assert written[0].exists()


def test_save_predicted_floorplan_pngs_writes_top_10_by_cost(tmp_path):
    pytest.importorskip("matplotlib")
    results = []
    for idx in range(12):
        results.append(
            EvalTestResult(
                test_id=idx,
                block_count=1,
                is_feasible=True,
                hpwl_gap=0.0,
                area_gap=0.0,
                violations_relative=0.0,
                runtime_seconds=0.1,
                cost=float(idx),
                positions=[(float(idx), 0.0, 1.0, 1.0)],
            )
        )
    result = SimpleNamespace(submission_name="unit", test_results=results)

    written = save_predicted_floorplan_pngs(result, tmp_path, top_k=10)

    assert len(written) == 10
    assert written[0].name.startswith("top_cost_rank_01_case_11_cost_11.0000")
    assert written[-1].name.startswith("top_cost_rank_10_case_2_cost_2.0000")
