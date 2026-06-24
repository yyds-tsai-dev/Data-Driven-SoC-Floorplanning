from __future__ import annotations

import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.targets import (
    build_diffusion_targets,
    parse_tree_sol_edges,
    split_metrics_sol,
)
from floorset_arch.diffusion.training import diffusion_training_loss
from floorset_arch.parser import parse_instance


def _inst():
    return parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        None,
    )


def test_parse_tree_sol_edges_filters_padding_and_invalid_rows():
    tree_sol = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [-1.0, -1.0, -1.0],
            [3.0, 0.0, 0.0],
        ]
    )

    edges = parse_tree_sol_edges(tree_sol, block_count=3)

    assert edges.parent.tolist() == [0, 1]
    assert edges.child.tolist() == [1, 2]
    assert edges.side.tolist() == [0, 1]


def test_metrics_sol_split_keeps_solution_quality_out_of_inference_features():
    metrics = torch.tensor([100.0, 4.0, 7.0, 3.0, 4.0, 2.0, 55.0, 66.0])

    split = split_metrics_sol(metrics)

    assert split.instance_stats.tolist() == [4.0, 7.0, 3.0, 4.0, 2.0]
    assert split.quality_labels.tolist() == [100.0, 55.0, 66.0]


def test_build_diffusion_targets_aligns_pair_index_with_tree_labels():
    inst = _inst()
    graph_inputs = build_diffusion_graph_inputs(inst)
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0, 4.0],
        ]
    )
    tree_sol = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics_sol = torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0])

    targets = build_diffusion_targets(inst, graph_inputs, fp_sol, tree_sol, metrics_sol)

    assert targets.center.shape == (3, 2)
    assert targets.log_aspect.shape == (3,)
    assert targets.pair_axis_label.shape[0] == graph_inputs.pair_index.shape[0]
    assert targets.tree_pair_mask.any()
    assert targets.quality_labels.tolist() == [25.0, 10.0, 12.0]


def test_diffusion_training_loss_uses_tree_and_quality_targets():
    inst = _inst()
    graph_inputs = build_diffusion_graph_inputs(inst)
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0, 4.0],
        ]
    )
    tree_sol = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics_sol = torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0])
    targets = build_diffusion_targets(inst, graph_inputs, fp_sol, tree_sol, metrics_sol)
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph_inputs, variant="raw", hidden_dim=16, layers=1
    )

    loss, parts = diffusion_training_loss(model, graph_inputs, targets, seed=3)

    assert loss.requires_grad
    assert parts["tree"] >= 0.0
    assert parts["quality"] >= 0.0
