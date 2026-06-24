from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.sampling import load_diffusion_checkpoint
from floorset_arch.diffusion.targets import build_diffusion_targets
from floorset_arch.diffusion.training import DiffusionLossConfig, diffusion_training_loss
from floorset_arch.parser import parse_instance


def _tiny_instance():
    return parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        None,
    )


def _tiny_targets():
    inst = _tiny_instance()
    graph = build_diffusion_graph_inputs(inst)
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0, 4.0],
        ]
    )
    tree_sol = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics_sol = torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0])
    return graph, build_diffusion_targets(inst, graph, fp_sol, tree_sol, metrics_sol)


def _clustered_instance(cluster_ids):
    return parse_instance(
        4,
        torch.tensor([4.0, 9.0, 16.0, 25.0]),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 3.0, 3.0]]),
        torch.tensor([[10.0, 20.0], [30.0, 5.0]]),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, float(cluster_ids[0]), 1.0],
                [0.0, 0.0, 1.0, float(cluster_ids[1]), 0.0],
                [0.0, 0.0, 0.0, float(cluster_ids[2]), 2.0],
                [0.0, 0.0, 0.0, float(cluster_ids[3]), 4.0],
            ]
        ),
        None,
    )


def _clustered_targets(cluster_ids):
    inst = _clustered_instance(cluster_ids)
    graph = build_diffusion_graph_inputs(inst)
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0, 4.0],
            [5.0, 5.0, 6.0, 4.0],
        ]
    )
    tree_sol = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 3.0, 0.0]]
    )
    metrics_sol = torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0])
    return graph, build_diffusion_targets(inst, graph, fp_sol, tree_sol, metrics_sol)


def test_diffusion_training_loss_uses_random_ddpm_timestep_schedule():
    graph, targets = _tiny_targets()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=16, layers=1
    )
    config = DiffusionLossConfig(max_steps=1000, noise_schedule="cosine")

    loss, parts = diffusion_training_loss(model, graph, targets, seed=5, config=config)

    assert loss.requires_grad
    assert 0.0 <= parts["timestep_mean"] < 1000.0
    assert parts["noise_schedule"] == "cosine"
    assert parts["tree"] >= 0.0
    assert parts["quality"] >= 0.0


def test_diffusion_quality_loss_uses_stable_scale_for_large_metrics():
    graph, targets = _clustered_targets([1, 2, 3, 4])
    targets = replace(
        targets,
        quality_labels=torch.tensor([12_000_000.0, 3_000_000.0, 2_000_000.0]),
    )
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=16, layers=1
    )

    _loss, parts = diffusion_training_loss(
        model,
        graph,
        targets,
        seed=7,
        config=DiffusionLossConfig(max_steps=32),
    )

    assert parts["quality"] < 1000.0


def test_hgt_lite_diffusion_loss_accepts_different_factor_counts_across_samples():
    first_graph, first_targets = _clustered_targets([1, 2, 3, 4])
    second_graph, second_targets = _clustered_targets([1, 2, 3, 0])
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        first_graph, variant="hgt_lite", hidden_dim=16, layers=1
    )

    first_loss, _first_parts = diffusion_training_loss(
        model,
        first_graph,
        first_targets,
        seed=1,
        config=DiffusionLossConfig(max_steps=32),
    )
    second_loss, _second_parts = diffusion_training_loss(
        model,
        second_graph,
        second_targets,
        seed=2,
        config=DiffusionLossConfig(max_steps=32),
    )

    assert first_loss.requires_grad
    assert second_loss.requires_grad


def test_diffusion_training_smoke_writes_loadable_checkpoint(tmp_path):
    from floorset_arch.training.train_diffusion import main, parse_args

    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--checkpoint-prefix",
            "diffusion",
            "--checkpoint-tag",
            "smoke",
            "--synthetic-smoke-samples",
            "2",
            "--num-samples",
            "2",
            "--val-samples",
            "1",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--variant",
            "hgt_lite",
            "--hidden-dim",
            "16",
            "--layers",
            "1",
            "--batch-size",
            "1",
            "--accumulation-steps",
            "1",
            "--print-every",
            "0",
            "--wandb-mode",
            "disabled",
        ]
    )

    main(args)

    checkpoint = Path(tmp_path) / "diffusion_latest_smoke.pt"
    assert checkpoint.exists()
    graph = build_diffusion_graph_inputs(_tiny_instance())
    model = load_diffusion_checkpoint(checkpoint, graph)
    assert model.variant == "hgt_lite"


def test_diffusion_run_tag_names_variant_not_anchor_encoder():
    from floorset_arch.training.train_diffusion import build_diffusion_run_tag, parse_args

    args = parse_args(["--variant", "hgt_lite", "--num-samples", "800000"])

    tag = build_diffusion_run_tag(args, stamp="0624")

    assert "diffhgt_lite" in tag
    assert "encmpnn" not in tag
