from __future__ import annotations

import pytest
import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.targets import build_diffusion_targets
from floorset_arch.parser import parse_instance
from floorset_arch.training.eval_probe_dataset import (
    EvalProbeDataset,
    adapt_eval_probe_sample,
    polygon_fp_sol_to_training_rects,
)


def _polygon_sample():
    area = torch.tensor([4.0, 9.0])
    constraints = torch.zeros(2, 5)
    b2b = torch.tensor([[0.0, 1.0, 2.0]])
    p2b = torch.empty(0, 3)
    pins = torch.empty(0, 2)
    polygons = torch.tensor(
        [
            [[10.0, 20.0], [12.0, 20.0], [12.0, 22.0], [10.0, 22.0], [-1.0, -1.0]],
            [[3.0, 4.0], [6.0, 4.0], [6.0, 7.0], [3.0, 7.0], [-1.0, -1.0]],
        ]
    )
    metrics = torch.tensor([100.0, 0.0, 1.0, 1.0, 0.0, 0.0, 11.0, 13.0])
    return {
        "input": (area, b2b, p2b, pins, constraints),
        "label": (polygons, metrics),
    }


class _FakeEvalDataset:
    def __init__(self, count: int = 100):
        self._count = count

    def __len__(self):
        return self._count

    def __getitem__(self, index: int):
        return _polygon_sample()


def test_polygon_fp_sol_to_training_rects_returns_training_order_whxy():
    sample = _polygon_sample()
    polygons = sample["label"][0]

    fp_sol = polygon_fp_sol_to_training_rects(polygons, block_count=2, case_index=7)

    assert fp_sol.shape == (2, 4)
    assert torch.allclose(fp_sol[0], torch.tensor([2.0, 2.0, 10.0, 20.0]))
    assert torch.allclose(fp_sol[1], torch.tensor([3.0, 3.0, 3.0, 4.0]))


def test_adapt_eval_probe_sample_returns_training_lite_contract():
    adapted = adapt_eval_probe_sample(_polygon_sample(), case_index=3)
    inputs = adapted["input"]
    labels = adapted["label"]

    assert len(inputs) == 5
    assert len(labels) == 3
    tree_sol, fp_sol, metrics_sol = labels
    assert tree_sol.shape == (0, 3)
    assert tree_sol.dtype == torch.float32
    assert torch.allclose(fp_sol[0], torch.tensor([2.0, 2.0, 10.0, 20.0]))
    assert torch.allclose(
        metrics_sol,
        torch.tensor([100.0, 0.0, 1.0, 1.0, 0.0, 0.0, 11.0, 13.0]),
    )


def test_eval_probe_dataset_requires_exactly_100_cases_by_default():
    with pytest.raises(ValueError, match="expected 100 evaluation cases, found 99"):
        EvalProbeDataset("FloorSet", dataset=_FakeEvalDataset(99))


def test_eval_probe_dataset_allows_custom_count_for_tests():
    dataset = EvalProbeDataset("FloorSet", dataset=_FakeEvalDataset(2), expected_count=2)

    sample = dataset[1]

    assert len(dataset) == 2
    assert sample["label"][0].shape == (0, 3)
    assert sample["label"][1].shape == (2, 4)


def test_polygon_adapter_reports_case_index_for_bad_polygon():
    bad = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])

    with pytest.raises(ValueError, match="case 12 block 0 has non-positive rectangle"):
        polygon_fp_sol_to_training_rects(bad, block_count=1, case_index=12)


def test_adapted_sample_builds_diffusion_targets_without_tree_edges():
    adapted = adapt_eval_probe_sample(_polygon_sample(), case_index=5)
    area, b2b, p2b, pins, constraints = adapted["input"]
    tree_sol, fp_sol, metrics_sol = adapted["label"]
    inst = parse_instance(2, area, b2b, p2b, pins, constraints, None)
    graph = build_diffusion_graph_inputs(inst)

    targets = build_diffusion_targets(inst, graph, fp_sol, tree_sol, metrics_sol)

    assert targets.center.shape == (2, 2)
    assert targets.log_aspect.shape == (2,)
    assert targets.tree_pair_mask.any().item() is False
    assert targets.quality_labels.numel() == 3
