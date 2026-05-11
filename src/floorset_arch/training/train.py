from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from floorset_arch.features import build_anchor_edge_tensors, build_anchor_node_features
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.parser import parse_instance
from floorset_arch.training.checkpoint import (
    build_run_tag,
    load_checkpoint,
    save_anchor_checkpoint,
)
from floorset_arch.training.losses import (
    build_anchor_targets,
    build_pairwise_relation_targets,
    compute_anchor_losses,
    constraint_weights,
    is_constraint_clean_training_sample,
    pairwise_relation_loss,
    sample_pairs,
)

ROOT = Path(__file__).resolve().parents[3]
FLOORSET_DIR = ROOT / "FloorSet"
if FLOORSET_DIR.exists() and str(FLOORSET_DIR) not in sys.path:
    sys.path.insert(0, str(FLOORSET_DIR))

from lite_dataset import FloorplanDatasetLite, floorplan_collate  # noqa: E402


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requested; install dependencies with `uv sync` or `pip install wandb`."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        config=vars(args),
    )
    return run


def valid_block_count(area_targets: torch.Tensor) -> int:
    return int((area_targets.detach().flatten() != -1).sum().item())


def choose_window_start(
    total: int, count: int, seed: int, epoch: int, fixed_start: int
) -> int:
    if total <= count:
        return 0
    if fixed_start >= 0:
        return min(fixed_start, total - count)
    return random.Random(seed + 1009 * epoch).randint(0, total - count)


def make_loader(
    dataset, start: int, count: int, shuffle: bool, seed: int, num_workers: int
):
    total = len(dataset)
    n = min(max(1, count), total)
    start = max(0, min(start, total - n)) if total > n else 0
    indices = list(range(start, start + n))
    if shuffle:
        random.Random(seed).shuffle(indices)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        collate_fn=floorplan_collate,
        num_workers=num_workers,
    )
    return loader, start, start + n - 1


def unpack_batch(batch):
    area_targets, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
    return (
        area_targets.squeeze(0),
        b2b.squeeze(0),
        p2b.squeeze(0),
        pins.squeeze(0),
        constraints.squeeze(0),
        fp_sol.squeeze(0),
        metrics.squeeze(0),
    )


def run_epoch(
    model, optimizer, loader, device: torch.device, args, epoch: int, train: bool
) -> dict[str, float]:
    model.train(train)
    sums = {
        key: 0.0
        for key in (
            "loss",
            "anchor",
            "aspect",
            "priority",
            "order",
            "edge",
            "ord_frac",
            "ord_acc",
            "pairwise",
            "pair_acc",
        )
    }
    count = 0
    skipped = 0
    accumulation_steps = max(1, int(args.accumulation_steps))
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        area_targets, b2b, p2b, pins, constraints, fp_sol, _metrics = unpack_batch(
            batch
        )
        if not is_constraint_clean_training_sample(fp_sol, area_targets, b2b, p2b, pins, constraints):
            skipped += 1
            continue
        block_count = valid_block_count(area_targets)
        inst = parse_instance(
            block_count, area_targets, b2b, p2b, pins, constraints, fp_sol
        )
        node_feat, scale = build_anchor_node_features(inst, device=device)
        edge_index, edge_attr = build_anchor_edge_tensors(inst, device=device)
        targets = build_anchor_targets(fp_sol, block_count, scale, device)
        weights = constraint_weights(constraints, block_count, device, args)
        pairs = sample_pairs(block_count, args.pairwise_pairs, device)

        with torch.set_grad_enabled(train):
            pred = model(node_feat, edge_index, edge_attr, pairs=pairs)
            loss, parts = compute_anchor_losses(
                pred, targets, weights, inst.valid_b2b, args
            )
            pair_targets = build_pairwise_relation_targets(
                fp_sol[:block_count].to(device),
                pairs,
                min_gap=args.min_order_gap * max(float(scale), 1.0),
                clear_ratio=args.clear_ratio,
            )
            pair_loss, pair_acc = pairwise_relation_loss(pred["pair_logits"], pair_targets)
            loss = loss + args.pairwise_weight * pair_loss
            if train:
                (loss / accumulation_steps).backward()
                if (count + 1) % accumulation_steps == 0:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        sums["loss"] += float(loss.item())
        sums["anchor"] += float(parts["anchor"].item())
        sums["aspect"] += float(parts["aspect"].item())
        sums["priority"] += float(parts["priority"].item())
        sums["order"] += float(parts["order"].item())
        sums["edge"] += float(parts["edge"].item())
        sums["ord_frac"] += float(parts["ord_frac"])
        sums["ord_acc"] += float(parts["ord_acc"])
        sums["pairwise"] += float(pair_loss.item())
        sums["pair_acc"] += float(pair_acc)
        count += 1
        if train and args.print_every > 0 and count % args.print_every == 0:
            print(
                f"[epoch {epoch:03d} step {count:05d}] "
                f"loss={loss.item():.5f} anchor={parts['anchor'].item():.5f} "
                f"aspect={parts['aspect'].item():.5f} priority={parts['priority'].item():.5f} "
                f"order={parts['order'].item():.5f} edge={parts['edge'].item():.5f} "
                f"ord_acc={float(parts['ord_acc']):.3f} pair_acc={pair_acc:.3f}",
                flush=True,
            )
    if train and count % accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    stats = {key: value / max(count, 1) for key, value in sums.items()}
    stats["skipped"] = float(skipped)
    stats["used"] = float(count)
    return stats


def load_resume_model(
    path: str, device: torch.device, args
) -> tuple[FloorplanGNN, int, dict | None]:
    checkpoint_path = Path(path)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    state = payload.get("model_state_dict") or payload.get("model_state")
    if state is None:
        raise RuntimeError(f"Resume checkpoint has no model state: {checkpoint_path}")

    model_config = payload.get("model_config", {})
    node_feat_dim = int(
        payload.get("node_feat_dim", model_config.get("node_feat_dim", 0))
    )
    hidden_dim = int(
        payload.get("hidden_dim", model_config.get("hidden_dim", args.hidden_dim))
    )
    layers = int(payload.get("layers", model_config.get("layers", args.layers)))
    dropout = float(payload.get("dropout", model_config.get("dropout", args.dropout)))
    if node_feat_dim <= 0:
        raise RuntimeError(
            f"Resume checkpoint has no node feature dimension: {checkpoint_path}"
        )

    if hidden_dim != args.hidden_dim or layers != args.layers:
        print(
            "Resume checkpoint model config overrides CLI: "
            f"hidden/layers {args.hidden_dim}/{args.layers} -> {hidden_dim}/{layers}",
            flush=True,
        )
        args.hidden_dim = hidden_dim
        args.layers = layers
    args.dropout = dropout

    model = FloorplanGNN(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state, strict=False)
    resume_epoch = int(payload.get("epoch", 0))
    print(f"Loaded resume checkpoint from {checkpoint_path}", flush=True)
    print(f"Resume checkpoint epoch = {resume_epoch}", flush=True)
    if payload.get("val_stats"):
        print(
            f"Resume checkpoint val loss = {payload['val_stats'].get('loss')}",
            flush=True,
        )
    return model, resume_epoch, payload.get("optimizer_state_dict")


def main(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dataset = FloorplanDatasetLite(args.data_path)
    total = len(dataset)
    val_start = choose_window_start(
        total, args.val_samples, args.seed + 777, 0, args.val_start
    )
    val_loader, vs, ve = make_loader(
        dataset, val_start, args.val_samples, False, args.seed, args.num_workers
    )

    model = None
    optimizer = None
    resume_epoch_offset = 0
    optimizer_state = None
    if args.resume_checkpoint:
        model, resume_epoch_offset, optimizer_state = load_resume_model(
            args.resume_checkpoint, device, args
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        if optimizer_state is not None and not args.ignore_optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            print("Loaded optimizer state from resume checkpoint", flush=True)
        elif optimizer_state is None:
            print(
                "Resume checkpoint has no optimizer state; using fresh AdamW",
                flush=True,
            )

    best_val = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.checkpoint_tag or build_run_tag(args)
    latest_path = out_dir / f"{args.checkpoint_prefix}_latest_{run_tag}.pt"
    best_path = out_dir / f"{args.checkpoint_prefix}_best_{run_tag}.pt"
    wandb_run = maybe_init_wandb(args)

    print("=" * 72)
    print("Architecture v3 Anchor-GNN training")
    print(f"  dataset samples  = {total}")
    print(f"  train per epoch  = {args.num_samples}")
    print(f"  val window       = {vs}..{ve}")
    print(f"  device           = {device}")
    print(f"  hidden/layers    = {args.hidden_dim}/{args.layers}")
    print(f"  dropout          = {args.dropout}")
    print(f"  accumulation     = {max(1, args.accumulation_steps)}")
    print(f"  checkpoint tag   = {run_tag}")
    if args.resume_checkpoint:
        print(f"  resume checkpoint= {args.resume_checkpoint}")
    print("=" * 72, flush=True)

    for local_epoch in range(1, args.epochs + 1):
        epoch = resume_epoch_offset + local_epoch
        train_start = choose_window_start(
            total, args.num_samples, args.seed, epoch, args.window_start
        )
        train_loader, ts, te = make_loader(
            dataset,
            train_start,
            args.num_samples,
            True,
            args.seed + epoch,
            args.num_workers,
        )
        print(f"Epoch {epoch:03d}: train window {ts}..{te}", flush=True)

        if model is None:
            first = next(iter(train_loader))
            area_targets, b2b, p2b, pins, constraints, _fp_sol, _metrics = unpack_batch(
                first
            )
            inst = parse_instance(
                valid_block_count(area_targets),
                area_targets,
                b2b,
                p2b,
                pins,
                constraints,
                None,
            )
            node_feat, _scale = build_anchor_node_features(inst, device=device)
            model = FloorplanGNN(
                node_feat_dim=node_feat.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.layers,
                dropout=args.dropout,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            train_loader, _, _ = make_loader(
                dataset,
                train_start,
                args.num_samples,
                True,
                args.seed + epoch,
                args.num_workers,
            )
            print(f"Created model with node_feat_dim={node_feat.shape[1]}", flush=True)

        train_stats = run_epoch(
            model, optimizer, train_loader, device, args, epoch, train=True
        )
        val_stats = run_epoch(
            model, optimizer, val_loader, device, args, epoch, train=False
        )
        print(
            f"Epoch {epoch:03d} train loss={train_stats['loss']:.5f} "
            f"anchor={train_stats['anchor']:.5f} aspect={train_stats['aspect']:.5f} "
            f"priority={train_stats['priority']:.5f} order={train_stats['order']:.5f} "
            f"edge={train_stats['edge']:.5f} pair={train_stats['pairwise']:.5f} "
            f"ord_acc={train_stats['ord_acc']:.3f} pair_acc={train_stats['pair_acc']:.3f} "
            f"used={train_stats['used']:.0f} skipped={train_stats['skipped']:.0f}",
            flush=True,
        )
        print(
            f"Epoch {epoch:03d} val   loss={val_stats['loss']:.5f} "
            f"anchor={val_stats['anchor']:.5f} aspect={val_stats['aspect']:.5f} "
            f"priority={val_stats['priority']:.5f} order={val_stats['order']:.5f} "
            f"edge={val_stats['edge']:.5f} pair={val_stats['pairwise']:.5f} "
            f"ord_acc={val_stats['ord_acc']:.3f} pair_acc={val_stats['pair_acc']:.3f} "
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

        save_anchor_checkpoint(
            latest_path, model, args, epoch, train_stats, val_stats, optimizer=optimizer
        )
        if args.write_stable_checkpoints:
            save_anchor_checkpoint(
                out_dir / f"{args.checkpoint_prefix}_latest.pt",
                model,
                args,
                epoch,
                train_stats,
                val_stats,
                optimizer=optimizer,
            )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_anchor_checkpoint(
                best_path,
                model,
                args,
                epoch,
                train_stats,
                val_stats,
                optimizer=optimizer,
            )
            if args.write_stable_checkpoints:
                save_anchor_checkpoint(
                    out_dir / f"{args.checkpoint_prefix}_best.pt",
                    model,
                    args,
                    epoch,
                    train_stats,
                    val_stats,
                    optimizer=optimizer,
                )
            print(f"Saved best checkpoint to {best_path}", flush=True)
            if args.write_stable_checkpoints:
                print(
                    f"Updated stable checkpoint at {out_dir / f'{args.checkpoint_prefix}_best.pt'}",
                    flush=True,
                )

    print(f"Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {best_path}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="FloorSet")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window-start", type=int, default=-1)
    parser.add_argument("--val-start", type=int, default=-1)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--aspect-weight", type=float, default=0.12)
    parser.add_argument("--priority-weight", type=float, default=0.08)
    parser.add_argument("--order-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.08)
    parser.add_argument("--pairwise-weight", type=float, default=0.20)
    parser.add_argument("--pairwise-pairs", type=int, default=4096)
    parser.add_argument("--boundary-weight-boost", type=float, default=0.45)
    parser.add_argument("--cluster-weight-boost", type=float, default=0.20)
    parser.add_argument("--mib-weight-boost", type=float, default=0.25)
    parser.add_argument("--order-pairs", type=int, default=4096)
    parser.add_argument("--clear-ratio", type=float, default=1.8)
    parser.add_argument("--min-order-gap", type=float, default=0.025)
    parser.add_argument("--order-temp", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--checkpoint-prefix", default="gnn")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--write-stable-checkpoints", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--ignore-optimizer-state", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="floorset-arch-v3")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--wandb-mode", default="online", choices=("online", "offline", "disabled")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
