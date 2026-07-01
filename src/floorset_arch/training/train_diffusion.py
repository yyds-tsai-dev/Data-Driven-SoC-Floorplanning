from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.sampling import load_diffusion_checkpoint
from floorset_arch.diffusion.targets import build_diffusion_targets
from floorset_arch.diffusion.training import DiffusionLossConfig, diffusion_training_loss
from floorset_arch.parser import parse_instance
from floorset_arch.training.eval_probe_dataset import EvalProbeDataset
from floorset_arch.training.train import (
    FloorplanDatasetLite,
    ROOT,
    TRAINING_EVAL_ENV_OVERRIDES,
    _parse_train_eval_tail_ids,
    choose_window_start,
    iter_batch_samples,
    make_loader,
    maybe_init_wandb,
    refresh_evaluator_best_checkpoint,
    valid_block_count,
)
from floorset_arch.training.promote_checkpoint import checkpoint_metric_from_eval_json
from floorset_arch.training.selection import append_metric_record


def _is_eval_probe_mode(args) -> bool:
    return getattr(args, "dataset_mode", "lite") == "eval-probe"


def _validate_diffusion_training_args(args) -> None:
    if not _is_eval_probe_mode(args):
        return
    if float(getattr(args, "tree_weight", 0.0)) != 0.0:
        raise ValueError("eval-probe requires --tree-weight 0")
    if bool(getattr(args, "train_evaluate_each_epoch", False)):
        raise ValueError("eval-probe uses final-only evaluator")


def build_diffusion_run_tag(args, stamp: str | None = None) -> str:
    when = stamp or datetime.now().strftime("%m%d")
    variant = str(getattr(args, "variant", "hgt_lite")).replace("-", "_")
    parts = [
        when,
        f"ns{int(args.num_samples)}",
        f"ep{int(args.epochs)}",
        f"diff{variant}",
        f"h{int(args.hidden_dim)}",
        f"l{int(args.layers)}",
        f"steps{int(args.max_diffusion_steps)}",
        f"acc{int(args.accumulation_steps)}",
        f"bs{int(args.batch_size)}",
    ]
    return "_".join(parts)


def _checkpoint_payload(
    model: GraphConditionedPlacementDiffusion,
    args,
    loss_config: DiffusionLossConfig,
    epoch: int,
    train_stats: dict[str, float],
    val_stats: dict[str, float],
    optimizer=None,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "variant": model.variant,
        "hidden_dim": model.hidden_dim,
        "layers": int(args.layers),
        "epoch": int(epoch),
        "train_stats": dict(train_stats),
        "val_stats": dict(val_stats),
        "args": vars(args),
        "loss_config": asdict(loss_config),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if ema_decay is not None:
        payload["ema_decay"] = float(ema_decay)
    if ema_state is not None:
        payload["ema_model_state_dict"] = {
            key: value.detach().clone() for key, value in ema_state.items()
        }
    return payload


def _save_diffusion_checkpoint(
    path: str | Path,
    model: GraphConditionedPlacementDiffusion,
    args,
    loss_config: DiffusionLossConfig,
    epoch: int,
    train_stats: dict[str, float],
    val_stats: dict[str, float],
    optimizer=None,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model,
            args,
            loss_config,
            epoch,
            train_stats,
            val_stats,
            optimizer=optimizer,
            ema_state=ema_state,
            ema_decay=ema_decay,
        ),
        output,
    )
    return output


def _prepare_diffusion_sample(sample, device: torch.device):
    area_targets, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics_sol = sample
    block_count = valid_block_count(area_targets)
    if block_count <= 0:
        raise ValueError("diffusion training sample has no valid blocks")
    inst = parse_instance(
        block_count,
        area_targets,
        b2b,
        p2b,
        pins,
        constraints,
        None,
    )
    graph = build_diffusion_graph_inputs(inst, device=device)
    targets = build_diffusion_targets(inst, graph, fp_sol, tree_sol, metrics_sol)
    return graph, targets


def _clone_ema_state(model: GraphConditionedPlacementDiffusion) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: GraphConditionedPlacementDiffusion,
    decay: float,
) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if not value.is_floating_point():
                ema_state[key] = value.detach().clone()
                continue
            if key not in ema_state:
                ema_state[key] = value.detach().clone()
                continue
            ema_state[key].mul_(float(decay)).add_(value.detach(), alpha=1.0 - float(decay))


def _optimizer_step(
    model,
    optimizer,
    args,
    pending_samples: int,
    force: bool = False,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
) -> int:
    if pending_samples <= 0:
        return pending_samples
    accumulation_steps = max(1, int(args.accumulation_steps))
    if not force and pending_samples < accumulation_steps:
        return pending_samples
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
    optimizer.step()
    if ema_state is not None and ema_decay > 0.0:
        _update_ema_state(ema_state, model, ema_decay)
    optimizer.zero_grad(set_to_none=True)
    return 0


def run_epoch(
    model: GraphConditionedPlacementDiffusion,
    optimizer,
    loader,
    device: torch.device,
    args,
    loss_config: DiffusionLossConfig,
    epoch: int,
    train: bool,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
) -> dict[str, float]:
    model.train(train)
    sums = {
        "loss": 0.0,
        "denoise": 0.0,
        "pair": 0.0,
        "tree": 0.0,
        "quality": 0.0,
        "aspect": 0.0,
        "layout_overlap": 0.0,
        "layout_bbox": 0.0,
        "layout_net": 0.0,
        "layout_cluster": 0.0,
        "layout_boundary": 0.0,
        "layout_mib": 0.0,
        "timestep_mean": 0.0,
    }
    count = 0
    skipped = 0
    pending_samples = 0
    accumulation_steps = max(1, int(args.accumulation_steps))
    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        for sample in iter_batch_samples(batch):
            try:
                graph, targets = _prepare_diffusion_sample(sample, device)
            except ValueError:
                skipped += 1
                continue

            seed = int(args.seed) + int(epoch) * 1_000_003 + count
            with torch.set_grad_enabled(train):
                loss, parts = diffusion_training_loss(
                    model,
                    graph,
                    targets,
                    seed=seed,
                    config=loss_config,
                )
                if train:
                    (loss / accumulation_steps).backward()
                    pending_samples += 1
                    pending_samples = _optimizer_step(
                        model,
                        optimizer,
                        args,
                        pending_samples,
                        force=False,
                        ema_state=ema_state,
                        ema_decay=ema_decay,
                    )

            sums["loss"] += float(loss.detach().cpu().item())
            sums["denoise"] += float(parts["denoise"])
            sums["pair"] += float(parts["pair"])
            sums["tree"] += float(parts["tree"])
            sums["quality"] += float(parts["quality"])
            sums["aspect"] += float(parts["aspect"])
            sums["layout_overlap"] += float(parts["layout_overlap"])
            sums["layout_bbox"] += float(parts["layout_bbox"])
            sums["layout_net"] += float(parts["layout_net"])
            sums["layout_cluster"] += float(parts["layout_cluster"])
            sums["layout_boundary"] += float(parts["layout_boundary"])
            sums["layout_mib"] += float(parts["layout_mib"])
            sums["timestep_mean"] += float(parts["timestep_mean"])
            count += 1

            if train and args.print_every > 0 and count % int(args.print_every) == 0:
                print(
                    f"[epoch {epoch:03d} step {count:05d}] "
                    f"loss={loss.item():.5f} denoise={parts['denoise']:.5f} "
                    f"pair={parts['pair']:.5f} tree={parts['tree']:.5f} "
                    f"quality={parts['quality']:.5f} aspect={parts['aspect']:.5f} "
                    f"overlap={parts['layout_overlap']:.5f} t={parts['timestep_mean']:.1f}",
                    flush=True,
                )

    if train:
        _optimizer_step(
            model,
            optimizer,
            args,
            pending_samples,
            force=True,
            ema_state=ema_state,
            ema_decay=ema_decay,
        )

    stats = {key: value / max(count, 1) for key, value in sums.items()}
    stats["used"] = float(count)
    stats["skipped"] = float(skipped)
    return stats


def _tiny_sample(offset: float = 0.0):
    return (
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]),
        torch.tensor(
            [
                [2.0, 2.0, 0.0 + offset, 0.0],
                [3.0, 3.0, 3.0 + offset, 0.0],
                [4.0, 4.0, 0.0 + offset, 4.0],
            ]
        ),
        torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0]),
    )


def _synthetic_smoke_loader(sample_count: int):
    samples = []
    for idx in range(max(1, int(sample_count))):
        sample = _tiny_sample(offset=0.1 * idx)
        samples.append(tuple(value.unsqueeze(0) for value in sample))
    return samples


def _make_loaders(args):
    if int(args.synthetic_smoke_samples) > 0:
        train_loader = _synthetic_smoke_loader(args.synthetic_smoke_samples)
        val_loader = _synthetic_smoke_loader(
            max(1, min(args.val_samples, args.synthetic_smoke_samples))
        )
        return (
            train_loader,
            val_loader,
            0,
            len(train_loader) - 1,
            0,
            len(val_loader) - 1,
            len(train_loader),
        )

    if _is_eval_probe_mode(args):
        dataset = EvalProbeDataset(args.data_path)
        total = len(dataset)
        train_loader, ts, te = make_loader(
            dataset,
            0,
            total,
            True,
            args.seed,
            args.num_workers,
            args.batch_size,
        )
        return train_loader, None, ts, te, -1, -1, total

    dataset = FloorplanDatasetLite(args.data_path)
    total = len(dataset)
    val_start = choose_window_start(
        total, args.val_samples, args.seed + 777, 0, args.val_start
    )
    val_loader, vs, ve = make_loader(
        dataset,
        val_start,
        args.val_samples,
        False,
        args.seed,
        args.num_workers,
        args.batch_size,
    )
    train_start = choose_window_start(
        total, args.num_samples, args.seed, 1, args.window_start
    )
    train_loader, ts, te = make_loader(
        dataset,
        train_start,
        args.num_samples,
        True,
        args.seed + 1,
        args.num_workers,
        args.batch_size,
    )
    return train_loader, val_loader, ts, te, vs, ve, total


def _first_graph(loader, device: torch.device):
    for batch in loader:
        for sample in iter_batch_samples(batch):
            return _prepare_diffusion_sample(sample, device)[0]
    raise RuntimeError("diffusion training loader is empty")


def _load_resume_optimizer_state(
    optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
) -> bool:
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("optimizer_state_dict")
    if state is None:
        return False
    try:
        optimizer.load_state_dict(state)
    except ValueError:
        return False
    return True


def _load_resume_ema_state(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("ema_model_state_dict")
    if state is None:
        return None
    return {key: value.detach().clone().to(device) for key, value in state.items()}


def _diffusion_training_eval_env(checkpoint: Path, use_ema: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    env.update(TRAINING_EVAL_ENV_OVERRIDES)
    env["FLOORSET_GNN_CHECKPOINT"] = ""
    env["FLOORSET_GNN_CHECKPOINT_SOURCE"] = "training_eval_disabled"
    env["FLOORSET_DIFFUSION_CHECKPOINT"] = str(checkpoint.resolve())
    env["FLOORSET_DIFFUSION_CHECKPOINT_SOURCE"] = "training_eval"
    env["FLOORSET_DIFFUSION_USE_EMA"] = "1" if use_ema else "0"
    pythonpath_parts = [
        str(ROOT / "FloorSet" / "iccad2026contest"),
        str(ROOT / "FloorSet"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def run_diffusion_training_evaluator(
    checkpoint: str | Path,
    epoch: int,
    args,
    metrics_manifest: str | Path,
    evaluator_best_path: str | Path,
):
    checkpoint_path = Path(checkpoint)
    output_dir = (
        Path(args.train_eval_output_dir)
        if getattr(args, "train_eval_output_dir", "")
        else Path(args.output_dir) / "training_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_json = output_dir / f"{checkpoint_path.stem}_epoch{int(epoch):03d}_full_eval.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "iccad2026_evaluate.py"),
        "--data-path",
        "../",
        "--evaluate",
        str(ROOT / "src" / "architecture_v11_optimizer.py"),
        "--verbose",
        "--output",
        str(eval_json.resolve()),
    ]
    subprocess.run(
        command,
        cwd=ROOT / "FloorSet" / "iccad2026contest",
        env=_diffusion_training_eval_env(
            checkpoint_path,
            use_ema=bool(getattr(args, "train_eval_use_ema", True)),
        ),
        check=True,
    )
    record = checkpoint_metric_from_eval_json(
        checkpoint=str(checkpoint_path),
        eval_json=eval_json,
        epoch=epoch,
        metric_source="diffusion_training_full_eval",
        val_loss=None,
        tail_ids=_parse_train_eval_tail_ids(
            getattr(args, "train_eval_tail_ids", "95,96,97,98,99")
        ),
    )
    append_metric_record(metrics_manifest, record)
    return refresh_evaluator_best_checkpoint(metrics_manifest, evaluator_best_path)


def main(args) -> None:
    _validate_diffusion_training_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    train_loader, val_loader, ts, te, vs, ve, total = _make_loaders(args)
    first_graph = _first_graph(train_loader, device)
    if args.resume_checkpoint:
        model = load_diffusion_checkpoint(
            Path(args.resume_checkpoint),
            first_graph,
            map_location=device,
            use_ema=False,
        )
        model = model.to(device)
        args.variant = model.variant
        args.hidden_dim = model.hidden_dim
    else:
        model = GraphConditionedPlacementDiffusion.from_graph_inputs(
            first_graph,
            variant=args.variant,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if args.resume_checkpoint:
        loaded = _load_resume_optimizer_state(optimizer, args.resume_checkpoint, device)
        print(f"Resume optimizer state loaded: {loaded}", flush=True)

    ema_decay = float(getattr(args, "ema_decay", 0.9999))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("--ema-decay must be in the range [0, 1)")
    ema_state = None
    if ema_decay > 0.0:
        if args.resume_checkpoint:
            ema_state = _load_resume_ema_state(args.resume_checkpoint, device)
        if ema_state is None:
            ema_state = _clone_ema_state(model)

    loss_config = DiffusionLossConfig(
        max_steps=args.max_diffusion_steps,
        noise_schedule=args.noise_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        pair_weight=args.pair_weight,
        tree_weight=args.tree_weight,
        quality_weight=args.quality_weight,
        aspect_weight=args.aspect_weight,
        overlap_weight=args.overlap_weight,
        bbox_weight=args.bbox_weight,
        net_weight=args.net_weight,
        cluster_weight=args.cluster_weight,
        boundary_weight=args.boundary_weight,
        mib_weight=args.mib_weight,
        noise_samples=args.noise_samples,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.checkpoint_tag or build_diffusion_run_tag(args)
    latest_path = out_dir / f"{args.checkpoint_prefix}_latest_{run_tag}.pt"
    best_loss_name = "best_probe_loss" if _is_eval_probe_mode(args) else "best_val_loss"
    best_loss_path = out_dir / f"{args.checkpoint_prefix}_{best_loss_name}_{run_tag}.pt"
    metrics_manifest = (
        Path(args.checkpoint_metrics_manifest)
        if getattr(args, "checkpoint_metrics_manifest", "")
        else out_dir / f"{args.checkpoint_prefix}_checkpoint_metrics_{run_tag}.jsonl"
    )
    evaluator_best_path = (
        Path(args.evaluator_best_checkpoint)
        if getattr(args, "evaluator_best_checkpoint", "")
        else out_dir / f"{args.checkpoint_prefix}_best_evaluator_{run_tag}.pt"
    )
    best_val = float("inf")
    wandb_run = maybe_init_wandb(args)

    print("=" * 72)
    print("Architecture v11 graph-conditioned diffusion training")
    print(f"  dataset mode     = {args.dataset_mode}")
    print(f"  dataset samples  = {total}")
    print(f"  train window     = {ts}..{te}")
    if _is_eval_probe_mode(args):
        print("  val window       = disabled")
    else:
        print(f"  val window       = {vs}..{ve}")
    print(f"  device           = {device}")
    print(f"  variant          = {args.variant}")
    print(f"  hidden/layers    = {args.hidden_dim}/{args.layers}")
    print(f"  diffusion steps  = {args.max_diffusion_steps}")
    print(f"  noise schedule   = {args.noise_schedule}")
    print(f"  ema decay        = {ema_decay}")
    print(f"  accumulation     = {max(1, args.accumulation_steps)}")
    print(f"  batch size       = {max(1, args.batch_size)}")
    print(f"  checkpoint tag   = {run_tag}")
    print("=" * 72, flush=True)

    empty_stats = {
        "loss": 0.0,
        "denoise": 0.0,
        "pair": 0.0,
        "tree": 0.0,
        "quality": 0.0,
        "aspect": 0.0,
        "layout_overlap": 0.0,
        "layout_bbox": 0.0,
        "layout_net": 0.0,
        "layout_cluster": 0.0,
        "layout_boundary": 0.0,
        "layout_mib": 0.0,
        "timestep_mean": 0.0,
        "used": 0.0,
        "skipped": 0.0,
    }
    for epoch in range(1, int(args.epochs) + 1):
        if int(args.synthetic_smoke_samples) <= 0 and not _is_eval_probe_mode(args):
            dataset = FloorplanDatasetLite(args.data_path)
            train_start = choose_window_start(
                len(dataset), args.num_samples, args.seed, epoch, args.window_start
            )
            train_loader, ts, te = make_loader(
                dataset,
                train_start,
                args.num_samples,
                True,
                args.seed + epoch,
                args.num_workers,
                args.batch_size,
            )
            print(f"Epoch {epoch:03d}: train window {ts}..{te}", flush=True)

        train_stats = run_epoch(
            model,
            optimizer,
            train_loader,
            device,
            args,
            loss_config,
            epoch,
            train=True,
            ema_state=ema_state,
            ema_decay=ema_decay,
        )
        if _is_eval_probe_mode(args):
            val_stats = dict(train_stats)
        else:
            val_stats = run_epoch(
                model, optimizer, val_loader, device, args, loss_config, epoch, train=False
            )
        print(
            f"Epoch {epoch:03d} train loss={train_stats['loss']:.5f} "
            f"denoise={train_stats['denoise']:.5f} pair={train_stats['pair']:.5f} "
            f"tree={train_stats['tree']:.5f} quality={train_stats['quality']:.5f} "
            f"aspect={train_stats['aspect']:.5f} overlap={train_stats['layout_overlap']:.5f} "
            f"used={train_stats['used']:.0f} skipped={train_stats['skipped']:.0f}",
            flush=True,
        )
        if _is_eval_probe_mode(args):
            print(
                f"Epoch {epoch:03d} probe loss={val_stats['loss']:.5f} "
                f"denoise={val_stats['denoise']:.5f} pair={val_stats['pair']:.5f} "
                f"tree={val_stats['tree']:.5f} quality={val_stats['quality']:.5f} "
                f"aspect={val_stats['aspect']:.5f} overlap={val_stats['layout_overlap']:.5f} "
                f"used={val_stats['used']:.0f} skipped={val_stats['skipped']:.0f}",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:03d} val   loss={val_stats['loss']:.5f} "
                f"denoise={val_stats['denoise']:.5f} pair={val_stats['pair']:.5f} "
                f"tree={val_stats['tree']:.5f} quality={val_stats['quality']:.5f} "
                f"aspect={val_stats['aspect']:.5f} overlap={val_stats['layout_overlap']:.5f} "
                f"used={val_stats['used']:.0f} skipped={val_stats['skipped']:.0f}",
                flush=True,
            )
        if wandb_run is not None:
            heldout_prefix = "probe" if _is_eval_probe_mode(args) else "val"
            wandb_run.log(
                {
                    "epoch": epoch,
                    **{f"train/{key}": value for key, value in train_stats.items()},
                    **{f"{heldout_prefix}/{key}": value for key, value in val_stats.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        _save_diffusion_checkpoint(
            latest_path,
            model,
            args,
            loss_config,
            epoch,
            train_stats,
            val_stats,
            optimizer=optimizer,
            ema_state=ema_state,
            ema_decay=ema_decay,
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            _save_diffusion_checkpoint(
                best_loss_path,
                model,
                args,
                loss_config,
                epoch,
                train_stats,
                val_stats,
                optimizer=optimizer,
                ema_state=ema_state,
                ema_decay=ema_decay,
            )
            print(f"Saved best checkpoint to {best_loss_path}", flush=True)
        selected_evaluator = None
        if args.train_evaluate_each_epoch and int(args.synthetic_smoke_samples) <= 0:
            selected_evaluator = run_diffusion_training_evaluator(
                latest_path,
                epoch,
                args,
                metrics_manifest,
                evaluator_best_path,
            )
        if selected_evaluator is not None:
            print(
                "Updated diffusion evaluator-best checkpoint "
                f"{evaluator_best_path} from {selected_evaluator.checkpoint} "
                f"(no-runtime={selected_evaluator.total_score_no_runtime}, "
                f"tail={selected_evaluator.tail_weighted_no_runtime}, "
                f"soft={selected_evaluator.soft_violations}, "
                f"runtime={selected_evaluator.avg_runtime})",
                flush=True,
            )
        empty_stats = val_stats

    best_label = "probe" if _is_eval_probe_mode(args) else "val"
    print(f"Best {best_label} loss: {best_val:.5f}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best {best_label}-loss checkpoint: {best_loss_path}")
    if args.train_evaluate_each_epoch:
        print(f"Evaluator metric manifest: {metrics_manifest}")
        print(f"Evaluator-best checkpoint: {evaluator_best_path}")
    if wandb_run is not None:
        summary_prefix = "probe" if _is_eval_probe_mode(args) else "val"
        wandb_run.summary[f"best_{summary_prefix}_loss"] = best_val
        wandb_run.summary[f"last_{summary_prefix}_loss"] = empty_stats["loss"]
        wandb_run.finish()


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-mode", choices=("lite", "eval-probe"), default="lite")
    parser.add_argument("--data-path", default="FloorSet")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--num-samples", type=int, default=800000)
    parser.add_argument("--val-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--window-start", type=int, default=-1)
    parser.add_argument("--val-start", type=int, default=-1)
    parser.add_argument("--variant", choices=("raw", "hgt_lite"), default="hgt_lite")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--tree-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.01)
    parser.add_argument("--aspect-weight", type=float, default=0.05)
    parser.add_argument("--overlap-weight", type=float, default=0.05)
    parser.add_argument("--bbox-weight", type=float, default=0.01)
    parser.add_argument("--net-weight", type=float, default=0.01)
    parser.add_argument("--cluster-weight", type=float, default=0.02)
    parser.add_argument("--boundary-weight", type=float, default=0.02)
    parser.add_argument("--mib-weight", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--checkpoint-prefix", default="diffusion")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--checkpoint-metrics-manifest",
        default="",
        help=(
            "Evaluator metric JSONL used to refresh evaluator-best diffusion "
            "checkpoints. When omitted, training uses a run-tagged manifest in "
            "the output dir."
        ),
    )
    parser.add_argument(
        "--evaluator-best-checkpoint",
        default="",
        help=(
            "Destination for the best diffusion checkpoint selected from "
            "evaluator evidence. When omitted, training writes a run-tagged "
            "best_evaluator checkpoint."
        ),
    )
    parser.add_argument(
        "--train-evaluate-each-epoch",
        action="store_true",
        help=(
            "After each epoch, run the full evaluator on the latest diffusion "
            "checkpoint, append evaluator evidence, and refresh evaluator-best."
        ),
    )
    parser.add_argument(
        "--train-eval-output-dir",
        default="",
        help="Directory for per-epoch diffusion full-evaluation JSON outputs.",
    )
    parser.add_argument(
        "--train-eval-tail-ids",
        default="95,96,97,98,99",
        help="Comma-separated validation ids used for tail_weighted_no_runtime records.",
    )
    parser.add_argument(
        "--train-eval-use-raw",
        dest="train_eval_use_ema",
        action="store_false",
        help="Evaluate raw diffusion weights instead of EMA weights.",
    )
    parser.add_argument(
        "--train-eval-use-ema",
        dest="train_eval_use_ema",
        action="store_true",
        help="Evaluate EMA diffusion weights when available.",
    )
    parser.set_defaults(train_eval_use_ema=True)
    parser.add_argument(
        "--synthetic-smoke-samples",
        type=int,
        default=0,
        help="Use small in-memory samples for training-path smoke tests.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="floorset-v11-diffusion")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--wandb-mode", default="online", choices=("online", "offline", "disabled")
    )
    args = parser.parse_args(argv)
    try:
        _validate_diffusion_training_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    main(parse_args())
