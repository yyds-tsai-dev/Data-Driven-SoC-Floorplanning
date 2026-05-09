from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from floorset_arch.features import build_anchor_edge_tensors, build_anchor_node_features
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.parser import parse_instance


ROOT = Path(__file__).resolve().parents[3]
FLOORSET_DIR = ROOT / "FloorSet"
if FLOORSET_DIR.exists() and str(FLOORSET_DIR) not in sys.path:
    sys.path.insert(0, str(FLOORSET_DIR))

from lite_dataset import FloorplanDatasetLite, floorplan_collate  # noqa: E402


def valid_block_count(area_targets: torch.Tensor) -> int:
    return int((area_targets.detach().flatten() != -1).sum().item())


def choose_window_start(total: int, count: int, seed: int, epoch: int, fixed_start: int) -> int:
    if total <= count:
        return 0
    if fixed_start >= 0:
        return min(fixed_start, total - count)
    return random.Random(seed + 1009 * epoch).randint(0, total - count)


def make_loader(dataset, start: int, count: int, shuffle: bool, seed: int, num_workers: int):
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


def build_targets(fp_sol: torch.Tensor, block_count: int, scale: float, device: torch.device) -> dict[str, torch.Tensor]:
    gt = fp_sol[:block_count].float().to(device)
    width = gt[:, 0].clamp_min(1e-6)
    height = gt[:, 1].clamp_min(1e-6)
    x = gt[:, 2]
    y = gt[:, 3]
    anchor = torch.stack([x + width / 2.0, y + height / 2.0], dim=1) / max(float(scale), 1.0)
    log_aspect = torch.log(width / height).clamp(-2.5, 2.5)
    center_sum = anchor.sum(dim=1)
    span = (center_sum.max() - center_sum.min()).clamp_min(1e-6)
    priority = 1.0 - (center_sum - center_sum.min()) / span
    return {"anchor": anchor, "log_aspect": log_aspect, "priority": priority}


def constraint_weights(constraints: torch.Tensor, block_count: int, device: torch.device, args) -> torch.Tensor:
    weights = torch.ones(block_count, device=device)
    if constraints is None or constraints.dim() <= 1:
        return weights
    c = constraints[:block_count].to(device)
    if c.shape[1] > 0:
        weights = weights + 0.10 * (c[:, 0] != 0).float()
    if c.shape[1] > 1:
        weights = weights + 0.10 * (c[:, 1] != 0).float()
    if c.shape[1] > 2:
        weights = weights + args.mib_weight_boost * (c[:, 2] != 0).float()
    if c.shape[1] > 3:
        weights = weights + args.cluster_weight_boost * (c[:, 3] != 0).float()
    if c.shape[1] > 4:
        weights = weights + args.boundary_weight_boost * (c[:, 4] != 0).float()
    return weights


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if loss.dim() > 1:
        loss = loss.sum(dim=1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def sample_pairs(block_count: int, max_pairs: int, device: torch.device) -> torch.Tensor:
    pairs = [(i, j) for i in range(block_count) for j in range(i + 1, block_count)]
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
    if max_pairs > 0 and pair_tensor.shape[0] > max_pairs:
        pair_tensor = pair_tensor[torch.randperm(pair_tensor.shape[0], device=device)[:max_pairs]]
    return pair_tensor


def order_aux_loss(pred_anchor: torch.Tensor, target_anchor: torch.Tensor, args) -> tuple[torch.Tensor, float, float]:
    device = pred_anchor.device
    pairs = sample_pairs(pred_anchor.shape[0], args.order_pairs, device)
    if pairs.numel() == 0:
        return pred_anchor.sum() * 0.0, 0.0, 1.0
    i = pairs[:, 0]
    j = pairs[:, 1]
    gt_delta = target_anchor[j] - target_anchor[i]
    abs_dx = gt_delta[:, 0].abs()
    abs_dy = gt_delta[:, 1].abs()
    clear_x = (abs_dx > args.clear_ratio * abs_dy) & (abs_dx > args.min_order_gap)
    clear_y = (abs_dy > args.clear_ratio * abs_dx) & (abs_dy > args.min_order_gap)

    loss = pred_anchor.sum() * 0.0
    used = 0
    acc_sum = 0.0
    acc_count = 0
    if clear_x.any():
        sign = torch.sign(gt_delta[clear_x, 0])
        pred_delta = pred_anchor[j[clear_x], 0] - pred_anchor[i[clear_x], 0]
        loss = loss + F.softplus(-sign * pred_delta / args.order_temp).mean()
        used += int(clear_x.sum().item())
        acc_sum += float(((pred_delta * sign) > 0).float().mean().item())
        acc_count += 1
    if clear_y.any():
        sign = torch.sign(gt_delta[clear_y, 1])
        pred_delta = pred_anchor[j[clear_y], 1] - pred_anchor[i[clear_y], 1]
        loss = loss + F.softplus(-sign * pred_delta / args.order_temp).mean()
        used += int(clear_y.sum().item())
        acc_sum += float(((pred_delta * sign) > 0).float().mean().item())
        acc_count += 1
    return loss, used / max(float(pairs.shape[0]), 1.0), acc_sum / max(acc_count, 1)


def edge_delta_loss(pred_anchor: torch.Tensor, target_anchor: torch.Tensor, valid_b2b: torch.Tensor) -> torch.Tensor:
    edges = []
    weights = []
    for i_f, j_f, weight_f in valid_b2b.tolist():
        i = int(i_f)
        j = int(j_f)
        weight = max(float(weight_f), 0.0)
        if 0 <= i < pred_anchor.shape[0] and 0 <= j < pred_anchor.shape[0] and i != j:
            edges.append((i, j))
            weights.append(weight)
    if not edges:
        return pred_anchor.sum() * 0.0
    edge_t = torch.tensor(edges, dtype=torch.long, device=pred_anchor.device)
    weight_t = torch.log1p(torch.tensor(weights, dtype=torch.float32, device=pred_anchor.device))
    weight_t = weight_t / weight_t.mean().clamp_min(1.0)
    pred_delta = pred_anchor[edge_t[:, 1]] - pred_anchor[edge_t[:, 0]]
    target_delta = target_anchor[edge_t[:, 1]] - target_anchor[edge_t[:, 0]]
    per = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").sum(dim=1)
    return (per * weight_t).mean()


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


def run_epoch(model, optimizer, loader, device: torch.device, args, epoch: int, train: bool) -> dict[str, float]:
    model.train(train)
    sums = {key: 0.0 for key in ("loss", "anchor", "aspect", "priority", "order", "edge", "ord_frac", "ord_acc")}
    count = 0
    for batch in loader:
        area_targets, b2b, p2b, pins, constraints, fp_sol, _metrics = unpack_batch(batch)
        block_count = valid_block_count(area_targets)
        inst = parse_instance(block_count, area_targets, b2b, p2b, pins, constraints, fp_sol)
        node_feat, scale = build_anchor_node_features(inst, device=device)
        edge_index, edge_attr = build_anchor_edge_tensors(inst, device=device)
        targets = build_targets(fp_sol, block_count, scale, device)
        weights = constraint_weights(constraints, block_count, device, args)

        with torch.set_grad_enabled(train):
            pred = model(node_feat, edge_index, edge_attr)
            anchor = weighted_smooth_l1(pred["anchor"], targets["anchor"], weights)
            aspect = weighted_smooth_l1(pred["log_aspect"], targets["log_aspect"], weights)
            priority = weighted_smooth_l1(pred["priority"], targets["priority"], weights)
            order, ord_frac, ord_acc = order_aux_loss(pred["anchor"], targets["anchor"], args)
            edge = edge_delta_loss(pred["anchor"], targets["anchor"], inst.valid_b2b)
            loss = (
                args.anchor_weight * anchor
                + args.aspect_weight * aspect
                + args.priority_weight * priority
                + args.order_weight * order
                + args.edge_weight * edge
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        sums["loss"] += float(loss.item())
        sums["anchor"] += float(anchor.item())
        sums["aspect"] += float(aspect.item())
        sums["priority"] += float(priority.item())
        sums["order"] += float(order.item())
        sums["edge"] += float(edge.item())
        sums["ord_frac"] += float(ord_frac)
        sums["ord_acc"] += float(ord_acc)
        count += 1
        if train and args.print_every > 0 and count % args.print_every == 0:
            print(
                f"[epoch {epoch:03d} step {count:05d}] "
                f"loss={loss.item():.5f} anchor={anchor.item():.5f} aspect={aspect.item():.5f} "
                f"priority={priority.item():.5f} order={order.item():.5f} edge={edge.item():.5f} "
                f"ord_acc={ord_acc:.3f}",
                flush=True,
            )
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_checkpoint(path: Path, model: FloorplanGNN, args, epoch: int, train_stats, val_stats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_feat_dim": model.node_feat_dim,
            "hidden_dim": model.hidden_dim,
            "layers": model.num_layers,
            "epoch": epoch,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "args": vars(args),
        },
        path,
    )


def main(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dataset = FloorplanDatasetLite(args.data_path)
    total = len(dataset)
    val_start = choose_window_start(total, args.val_samples, args.seed + 777, 0, args.val_start)
    val_loader, vs, ve = make_loader(dataset, val_start, args.val_samples, False, args.seed, args.num_workers)

    model = None
    optimizer = None
    best_val = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Architecture v2 Anchor-GNN training")
    print(f"  dataset samples  = {total}")
    print(f"  train per epoch  = {args.num_samples}")
    print(f"  val window       = {vs}..{ve}")
    print(f"  device           = {device}")
    print(f"  hidden/layers    = {args.hidden_dim}/{args.layers}")
    print("=" * 72, flush=True)

    for epoch in range(1, args.epochs + 1):
        train_start = choose_window_start(total, args.num_samples, args.seed, epoch, args.window_start)
        train_loader, ts, te = make_loader(dataset, train_start, args.num_samples, True, args.seed + epoch, args.num_workers)
        print(f"Epoch {epoch:03d}: train window {ts}..{te}", flush=True)

        if model is None:
            first = next(iter(train_loader))
            area_targets, b2b, p2b, pins, constraints, _fp_sol, _metrics = unpack_batch(first)
            inst = parse_instance(valid_block_count(area_targets), area_targets, b2b, p2b, pins, constraints, None)
            node_feat, _scale = build_anchor_node_features(inst, device=device)
            model = FloorplanGNN(node_feat_dim=node_feat.shape[1], hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            train_loader, _, _ = make_loader(dataset, train_start, args.num_samples, True, args.seed + epoch, args.num_workers)
            print(f"Created model with node_feat_dim={node_feat.shape[1]}", flush=True)

        train_stats = run_epoch(model, optimizer, train_loader, device, args, epoch, train=True)
        val_stats = run_epoch(model, optimizer, val_loader, device, args, epoch, train=False)
        print(
            f"Epoch {epoch:03d} train loss={train_stats['loss']:.5f} "
            f"anchor={train_stats['anchor']:.5f} aspect={train_stats['aspect']:.5f} "
            f"priority={train_stats['priority']:.5f} order={train_stats['order']:.5f} "
            f"edge={train_stats['edge']:.5f} ord_acc={train_stats['ord_acc']:.3f}",
            flush=True,
        )
        print(
            f"Epoch {epoch:03d} val   loss={val_stats['loss']:.5f} "
            f"anchor={val_stats['anchor']:.5f} aspect={val_stats['aspect']:.5f} "
            f"priority={val_stats['priority']:.5f} order={val_stats['order']:.5f} "
            f"edge={val_stats['edge']:.5f} ord_acc={val_stats['ord_acc']:.3f}",
            flush=True,
        )

        save_checkpoint(out_dir / "gnn_latest.pt", model, args, epoch, train_stats, val_stats)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(out_dir / "gnn_best.pt", model, args, epoch, train_stats, val_stats)
            print(f"Saved best checkpoint to {out_dir / 'gnn_best.pt'}", flush=True)

    print(f"Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {out_dir / 'gnn_best.pt'}")


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
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--aspect-weight", type=float, default=0.12)
    parser.add_argument("--priority-weight", type=float, default=0.08)
    parser.add_argument("--order-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.08)
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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
