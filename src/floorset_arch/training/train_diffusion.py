from __future__ import annotations

import argparse
import random
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
from floorset_arch.training.train import (
    FloorplanDatasetLite,
    choose_window_start,
    iter_batch_samples,
    make_loader,
    maybe_init_wandb,
    valid_block_count,
)


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


def _optimizer_step(model, optimizer, args, pending_samples: int, force: bool = False) -> int:
    if pending_samples <= 0:
        return pending_samples
    accumulation_steps = max(1, int(args.accumulation_steps))
    if not force and pending_samples < accumulation_steps:
        return pending_samples
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
    optimizer.step()
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
) -> dict[str, float]:
    model.train(train)
    sums = {
        "loss": 0.0,
        "denoise": 0.0,
        "pair": 0.0,
        "tree": 0.0,
        "quality": 0.0,
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
                    )

            sums["loss"] += float(loss.detach().cpu().item())
            sums["denoise"] += float(parts["denoise"])
            sums["pair"] += float(parts["pair"])
            sums["tree"] += float(parts["tree"])
            sums["quality"] += float(parts["quality"])
            sums["timestep_mean"] += float(parts["timestep_mean"])
            count += 1

            if train and args.print_every > 0 and count % int(args.print_every) == 0:
                print(
                    f"[epoch {epoch:03d} step {count:05d}] "
                    f"loss={loss.item():.5f} denoise={parts['denoise']:.5f} "
                    f"pair={parts['pair']:.5f} tree={parts['tree']:.5f} "
                    f"quality={parts['quality']:.5f} t={parts['timestep_mean']:.1f}",
                    flush=True,
                )

    if train:
        _optimizer_step(model, optimizer, args, pending_samples, force=True)

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


def main(args) -> None:
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

    loss_config = DiffusionLossConfig(
        max_steps=args.max_diffusion_steps,
        noise_schedule=args.noise_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        pair_weight=args.pair_weight,
        tree_weight=args.tree_weight,
        quality_weight=args.quality_weight,
        noise_samples=args.noise_samples,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.checkpoint_tag or build_diffusion_run_tag(args)
    latest_path = out_dir / f"{args.checkpoint_prefix}_latest_{run_tag}.pt"
    best_val_loss_path = out_dir / f"{args.checkpoint_prefix}_best_val_loss_{run_tag}.pt"
    best_val = float("inf")
    wandb_run = maybe_init_wandb(args)

    print("=" * 72)
    print("Architecture v11 graph-conditioned diffusion training")
    print(f"  dataset samples  = {total}")
    print(f"  train window     = {ts}..{te}")
    print(f"  val window       = {vs}..{ve}")
    print(f"  device           = {device}")
    print(f"  variant          = {args.variant}")
    print(f"  hidden/layers    = {args.hidden_dim}/{args.layers}")
    print(f"  diffusion steps  = {args.max_diffusion_steps}")
    print(f"  noise schedule   = {args.noise_schedule}")
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
        "timestep_mean": 0.0,
        "used": 0.0,
        "skipped": 0.0,
    }
    for epoch in range(1, int(args.epochs) + 1):
        if int(args.synthetic_smoke_samples) <= 0:
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
            model, optimizer, train_loader, device, args, loss_config, epoch, train=True
        )
        val_stats = run_epoch(
            model, optimizer, val_loader, device, args, loss_config, epoch, train=False
        )
        print(
            f"Epoch {epoch:03d} train loss={train_stats['loss']:.5f} "
            f"denoise={train_stats['denoise']:.5f} pair={train_stats['pair']:.5f} "
            f"tree={train_stats['tree']:.5f} quality={train_stats['quality']:.5f} "
            f"used={train_stats['used']:.0f} skipped={train_stats['skipped']:.0f}",
            flush=True,
        )
        print(
            f"Epoch {epoch:03d} val   loss={val_stats['loss']:.5f} "
            f"denoise={val_stats['denoise']:.5f} pair={val_stats['pair']:.5f} "
            f"tree={val_stats['tree']:.5f} quality={val_stats['quality']:.5f} "
            f"used={val_stats['used']:.0f} skipped={val_stats['skipped']:.0f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    **{f"train/{key}": value for key, value in train_stats.items()},
                    **{f"val/{key}": value for key, value in val_stats.items()},
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
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            _save_diffusion_checkpoint(
                best_val_loss_path,
                model,
                args,
                loss_config,
                epoch,
                train_stats,
                val_stats,
                optimizer=optimizer,
            )
            print(f"Saved best checkpoint to {best_val_loss_path}", flush=True)
        empty_stats = val_stats

    print(f"Best val loss: {best_val:.5f}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best val-loss checkpoint: {best_val_loss_path}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["last_val_loss"] = empty_stats["loss"]
        wandb_run.finish()


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--tree-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--checkpoint-prefix", default="diffusion")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--resume-checkpoint", default="")
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
