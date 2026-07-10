from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from floorset_arch.diffusion.contracts import DIFFUSION_FEATURE_VERSION
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


def test_diffusion_training_loss_reports_aspect_and_layout_terms():
    graph, targets = _tiny_targets()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=16, layers=1
    )
    base_config = DiffusionLossConfig(
        max_steps=32,
        pair_weight=0.0,
        tree_weight=0.0,
        quality_weight=0.0,
        aspect_weight=0.0,
        overlap_weight=0.0,
        bbox_weight=0.0,
    )
    layout_config = replace(
        base_config,
        aspect_weight=0.5,
        overlap_weight=0.5,
        bbox_weight=0.1,
        boundary_weight=0.1,
        mib_weight=0.1,
    )

    base_loss, _base_parts = diffusion_training_loss(
        model, graph, targets, seed=11, config=base_config
    )
    layout_loss, layout_parts = diffusion_training_loss(
        model, graph, targets, seed=11, config=layout_config
    )

    assert layout_parts["aspect"] >= 0.0
    assert layout_parts["layout_overlap"] >= 0.0
    assert layout_parts["layout_bbox"] >= 0.0
    assert layout_parts["layout_boundary"] >= 0.0
    assert layout_parts["layout_mib"] >= 0.0
    assert layout_loss.item() >= base_loss.item()


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
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["ema_decay"] == 0.9999
    assert payload["ema_model_state_dict"]
    graph = build_diffusion_graph_inputs(_tiny_instance())
    model = load_diffusion_checkpoint(checkpoint, graph)
    assert model.variant == "hgt_lite"


def test_load_diffusion_checkpoint_prefers_ema_weights_for_inference(tmp_path):
    graph = build_diffusion_graph_inputs(_tiny_instance())
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=16, layers=1
    )
    raw_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema_state = {key: value.detach().clone() for key, value in raw_state.items()}
    floating_key = next(key for key, value in ema_state.items() if value.is_floating_point())
    ema_state[floating_key] = ema_state[floating_key] + 1.0
    checkpoint = tmp_path / "diffusion.pt"
    torch.save(
        {
            "model_state_dict": raw_state,
            "ema_model_state_dict": ema_state,
            "ema_decay": 0.9999,
            "variant": "raw",
            "hidden_dim": 16,
            "layers": 1,
            "loss_config": {"max_steps": 32, "noise_schedule": "cosine"},
            "feature_version": DIFFUSION_FEATURE_VERSION,
        },
        checkpoint,
    )

    inference_model = load_diffusion_checkpoint(checkpoint, graph)
    raw_model = load_diffusion_checkpoint(checkpoint, graph, use_ema=False)

    assert torch.allclose(inference_model.state_dict()[floating_key], ema_state[floating_key])
    assert torch.allclose(raw_model.state_dict()[floating_key], raw_state[floating_key])


def test_load_diffusion_checkpoint_rejects_feature_version_mismatch(tmp_path):
    import pytest

    graph = build_diffusion_graph_inputs(_tiny_instance())
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=16, layers=1
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": "raw",
            "hidden_dim": 16,
            "layers": 1,
            "loss_config": {"max_steps": 32, "noise_schedule": "cosine"},
            # No feature_version key -> treated as legacy v1.
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="feature_version mismatch"):
        load_diffusion_checkpoint(checkpoint, graph)


def test_diffusion_training_eval_env_uses_diffusion_checkpoint(tmp_path, monkeypatch):
    from floorset_arch.training.train_diffusion import _diffusion_training_eval_env

    checkpoint = tmp_path / "diffusion.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("FLOORSET_GNN_CHECKPOINT", "old_gnn.pt")

    env = _diffusion_training_eval_env(checkpoint, use_ema=False)

    assert env["FLOORSET_DIFFUSION_CHECKPOINT"] == str(checkpoint.resolve())
    assert env["FLOORSET_DIFFUSION_CHECKPOINT_SOURCE"] == "training_eval"
    assert env["FLOORSET_DIFFUSION_USE_EMA"] == "0"
    assert env["FLOORSET_GNN_CHECKPOINT"] == ""


def test_diffusion_parse_args_exposes_evaluator_promotion_flags():
    from floorset_arch.training.train_diffusion import parse_args

    args = parse_args(
        [
            "--checkpoint-metrics-manifest",
            "checkpoints/diffusion_metrics.jsonl",
            "--evaluator-best-checkpoint",
            "checkpoints/diffusion_best_evaluator.pt",
            "--train-evaluate-each-epoch",
            "--train-eval-output-dir",
            "checkpoints/training_eval",
            "--train-eval-tail-ids",
            "95,96",
            "--train-eval-use-raw",
        ]
    )

    assert args.checkpoint_metrics_manifest == "checkpoints/diffusion_metrics.jsonl"
    assert args.evaluator_best_checkpoint == "checkpoints/diffusion_best_evaluator.pt"
    assert args.train_evaluate_each_epoch is True
    assert args.train_eval_output_dir == "checkpoints/training_eval"
    assert args.train_eval_tail_ids == "95,96"
    assert args.train_eval_use_ema is False


def test_diffusion_run_tag_names_variant_not_anchor_encoder():
    from floorset_arch.training.train_diffusion import build_diffusion_run_tag, parse_args

    args = parse_args(["--variant", "hgt_lite", "--num-samples", "800000"])

    tag = build_diffusion_run_tag(args, stamp="0624")

    assert "diffhgt_lite" in tag
    assert "encmpnn" not in tag
