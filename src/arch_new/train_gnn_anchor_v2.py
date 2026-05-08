#!/usr/bin/env python3
"""
Train Anchor-GNN v2 for the guarded relative-order solver.

Why this version:
  - The best current solver uses GNN anchors as soft guidance.
  - Naive pairwise GNN was too brittle because pairwise decisions become hard
    topology constraints.
  - This trainer keeps anchor prediction as the main objective, but adds
    order-aware auxiliary losses on the predicted anchors themselves.

Recommended command:
  python -u train_gnn_anchor_v2.py --num-samples 4000 --epochs 10 --device cuda --print-every 200 | tee train_anchor_v2.log

It saves checkpoints/gnn_best.pt, compatible with my_optimizer_v_guarded_best.py.
"""

import argparse
import random
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from liteLoader import FloorplanDatasetLite
from lite_dataset import floorplan_collate as train_floorplan_collate

from gnn_model import (
    FloorplanGNN,
    valid_block_count,
    build_node_features,
    build_edge_tensors,
    build_targets,
)


def make_window_loader(dataset, start, num_samples, batch_size, shuffle, seed):
    total = len(dataset)
    n = min(num_samples, total)
    start = max(0, min(start, total - n)) if total > n else 0
    indices = list(range(start, start + n))
    if shuffle:
        random.Random(seed).shuffle(indices)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_floorplan_collate,
        num_workers=0,
    )
    return loader, start, start + n - 1


def choose_window_start(total, n, seed, epoch, fixed_start=-1):
    if total <= n:
        return 0
    if fixed_start >= 0:
        return min(fixed_start, total - n)
    rng = random.Random(seed + 1009 * epoch)
    return rng.randint(0, total - n)


def constraint_weights(constraints, block_count, device, args):
    n = block_count
    w = torch.ones(n, device=device)
    if constraints is None or constraints.dim() <= 1:
        return w
    c = constraints[:n].to(device)
    if c.shape[1] > 3:
        w = w + args.cluster_anchor_boost * (c[:, 3] != 0).float()
    if c.shape[1] > 4:
        w = w + args.boundary_anchor_boost * (c[:, 4] != 0).float()
    # Fixed/preplaced are deterministic in solver; do not overweight too much.
    if c.shape[1] > 0:
        w = w + 0.10 * (c[:, 0] != 0).float()
    if c.shape[1] > 1:
        w = w + 0.10 * (c[:, 1] != 0).float()
    return w


def weighted_anchor_loss(pred_anchor, target_anchor, weights):
    per = F.smooth_l1_loss(pred_anchor, target_anchor, reduction="none").sum(dim=1)
    return (per * weights).sum() / weights.sum().clamp_min(1.0)


def sample_all_pairs(n, max_pairs, device):
    if n <= 1:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    total = n * (n - 1) // 2
    # For N <= 120 all pairs are only 7140, which is manageable.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pairs = torch.tensor(pairs, dtype=torch.long, device=device)
    if max_pairs > 0 and total > max_pairs:
        idx = torch.randperm(total, device=device)[:max_pairs]
        pairs = pairs[idx]
    return pairs


def order_aux_loss(pred_anchor, target_anchor, n, args, device):
    """
    Make predicted anchors preserve clear x/y ordering from ground truth.

    This is intentionally softer than pairwise relation classification:
    it only updates the anchor coordinates, so wrong pair labels cannot directly
    become hard graph constraints.
    """
    pairs = sample_all_pairs(n, args.order_pairs, device)
    if pairs.numel() == 0:
        return pred_anchor.sum() * 0.0, 0.0, 0.0

    i = pairs[:, 0]
    j = pairs[:, 1]
    gt_dx = target_anchor[j, 0] - target_anchor[i, 0]
    gt_dy = target_anchor[j, 1] - target_anchor[i, 1]
    abs_dx = gt_dx.abs()
    abs_dy = gt_dy.abs()

    # Clear horizontal / vertical pairs. Ambiguous diagonal pairs are ignored.
    clear_x = (abs_dx > args.clear_ratio * abs_dy) & (abs_dx > args.min_order_gap)
    clear_y = (abs_dy > args.clear_ratio * abs_dx) & (abs_dy > args.min_order_gap)

    temp = args.order_temp
    loss = pred_anchor.sum() * 0.0
    used = 0
    acc_sum = 0.0
    acc_count = 0

    if clear_x.any():
        ii = i[clear_x]
        jj = j[clear_x]
        sign = torch.sign(gt_dx[clear_x]).clamp(min=-1.0, max=1.0)
        pred_diff = pred_anchor[jj, 0] - pred_anchor[ii, 0]
        loss_x = F.softplus(-sign * pred_diff / temp).mean()
        loss = loss + loss_x
        used += int(clear_x.sum().item())
        acc_sum += float(((pred_diff * sign) > 0).float().mean().item())
        acc_count += 1

    if clear_y.any():
        ii = i[clear_y]
        jj = j[clear_y]
        sign = torch.sign(gt_dy[clear_y]).clamp(min=-1.0, max=1.0)
        pred_diff = pred_anchor[jj, 1] - pred_anchor[ii, 1]
        loss_y = F.softplus(-sign * pred_diff / temp).mean()
        loss = loss + loss_y
        used += int(clear_y.sum().item())
        acc_sum += float(((pred_diff * sign) > 0).float().mean().item())
        acc_count += 1

    if acc_count == 0:
        acc = 1.0
    else:
        acc = acc_sum / acc_count

    return loss, used / max(float(pairs.shape[0]), 1.0), acc


def edge_delta_loss(pred_anchor, target_anchor, b2b_conn, n, device, max_edges=2048):
    edges = []
    weights = []
    if b2b_conn is not None:
        for e in b2b_conn:
            if float(e[0]) < 0.0:
                continue
            i = int(e[0]); j = int(e[1]); w = float(e[2])
            if 0 <= i < n and 0 <= j < n and i != j:
                edges.append((i, j)); weights.append(max(w, 0.0))
    if not edges:
        return pred_anchor.sum() * 0.0
    if len(edges) > max_edges:
        idx = torch.randperm(len(edges), device=device)[:max_edges].cpu().tolist()
        edges = [edges[k] for k in idx]
        weights = [weights[k] for k in idx]
    edge_t = torch.tensor(edges, dtype=torch.long, device=device)
    wt = torch.tensor(weights, dtype=torch.float32, device=device)
    wt = torch.log1p(wt)
    wt = wt / wt.mean().clamp_min(1.0)
    i = edge_t[:, 0]
    j = edge_t[:, 1]
    pred_delta = pred_anchor[j] - pred_anchor[i]
    gt_delta = target_anchor[j] - target_anchor[i]
    per = F.smooth_l1_loss(pred_delta, gt_delta, reduction="none").sum(dim=1)
    return (per * wt).mean()


def train_epoch(model, optimizer, loader, device, args, epoch, train=True):
    if train:
        model.train()
    else:
        model.eval()

    sums = {"loss": 0.0, "anchor": 0.0, "order": 0.0, "edge": 0.0, "ord_frac": 0.0, "ord_acc": 0.0}
    count = 0

    for batch in loader:
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
        area_target = area_target.squeeze(0)
        b2b_conn = b2b_conn.squeeze(0)
        p2b_conn = p2b_conn.squeeze(0)
        pins_pos = pins_pos.squeeze(0)
        constraints = constraints.squeeze(0)
        fp_sol = fp_sol.squeeze(0)

        n = valid_block_count(area_target)
        node_feat, scale = build_node_features(
            area_targets=area_target,
            b2b_connectivity=b2b_conn,
            p2b_connectivity=p2b_conn,
            pins_pos=pins_pos,
            constraints=constraints,
            block_count=n,
            device=device,
        )
        edge_index, edge_attr = build_edge_tensors(n, b2b_conn, device)
        targets = build_targets(fp_sol, n, scale, device)
        target_anchor = targets["anchor"]
        weights = constraint_weights(constraints, n, device, args)

        with torch.set_grad_enabled(train):
            pred = model(node_feat, edge_index, edge_attr)
            anchor = weighted_anchor_loss(pred["anchor"], target_anchor, weights)
            order, ord_frac, ord_acc = order_aux_loss(pred["anchor"], target_anchor, n, args, device)
            edge_l = edge_delta_loss(pred["anchor"], target_anchor, b2b_conn, n, device)
            loss = args.anchor_weight * anchor + args.order_weight * order + args.edge_weight * edge_l

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        sums["loss"] += float(loss.item())
        sums["anchor"] += float(anchor.item())
        sums["order"] += float(order.item())
        sums["edge"] += float(edge_l.item())
        sums["ord_frac"] += float(ord_frac)
        sums["ord_acc"] += float(ord_acc)
        count += 1

        if train and args.print_every > 0 and count % args.print_every == 0:
            print(
                f"[epoch {epoch:03d} step {count:05d}] "
                f"loss={loss.item():.6f} anchor={anchor.item():.6f} "
                f"order={order.item():.6f} edge={edge_l.item():.6f} "
                f"ord_frac={ord_frac:.3f} ord_acc={ord_acc:.3f} blocks={n}",
                flush=True,
            )

    for k in sums:
        sums[k] /= max(count, 1)
    return sums


def save_checkpoint(path, model, args, epoch, train_stats, val_stats=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "node_feat_dim": model.node_feat_dim,
        "hidden_dim": model.hidden_dim,
        "layers": model.num_layers,
        "epoch": epoch,
        "avg_loss": train_stats["loss"],
        "train_stats": train_stats,
        "val_stats": val_stats,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dataset = FloorplanDatasetLite(args.data_path)
    total = len(dataset)

    # Fixed validation window for checkpoint selection.
    val_n = min(args.val_samples, total)
    val_start = choose_window_start(total, val_n, args.seed + 777, 0, args.val_start)
    val_loader, vs, ve = make_window_loader(dataset, val_start, val_n, 1, False, args.seed)

    model = None
    optimizer = None
    best_val = float("inf")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Anchor-GNN v2 training")
    print(f"  total samples      = {total}")
    print(f"  train per epoch    = {args.num_samples}")
    print(f"  val window         = {vs}..{ve}")
    print(f"  device             = {device}")
    print(f"  hidden/layers      = {args.hidden_dim}/{args.layers}")
    print("=" * 70, flush=True)

    for epoch in range(1, args.epochs + 1):
        train_start = choose_window_start(total, args.num_samples, args.seed, epoch, args.window_start)
        train_loader, ts, te = make_window_loader(dataset, train_start, args.num_samples, 1, True, args.seed + epoch)
        print(f"Epoch {epoch:03d}: train window {ts}..{te}", flush=True)

        # Lazily create model after seeing feature dimension.
        if model is None:
            first_batch = next(iter(train_loader))
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = first_batch
            area_target = area_target.squeeze(0)
            b2b_conn = b2b_conn.squeeze(0)
            p2b_conn = p2b_conn.squeeze(0)
            pins_pos = pins_pos.squeeze(0)
            constraints = constraints.squeeze(0)
            n = valid_block_count(area_target)
            node_feat, _ = build_node_features(area_target, b2b_conn, p2b_conn, pins_pos, constraints, n, device)

            model = FloorplanGNN(node_feat_dim=node_feat.shape[1], hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print("Created model")
            print(f"  node_feat_dim = {node_feat.shape[1]}")
            print(f"  hidden_dim    = {args.hidden_dim}")
            print(f"  layers        = {args.layers}")
            print("=" * 70, flush=True)

            # Re-create train loader because we consumed one item.
            train_loader, _, _ = make_window_loader(dataset, train_start, args.num_samples, 1, True, args.seed + epoch)

        train_stats = train_epoch(model, optimizer, train_loader, device, args, epoch, train=True)
        val_stats = train_epoch(model, optimizer, val_loader, device, args, epoch, train=False)

        print("-" * 70)
        print(
            f"Epoch {epoch:03d} train: "
            f"loss={train_stats['loss']:.6f} anchor={train_stats['anchor']:.6f} "
            f"order={train_stats['order']:.6f} edge={train_stats['edge']:.6f} "
            f"ord_acc={train_stats['ord_acc']:.4f}"
        )
        print(
            f"Epoch {epoch:03d} val  : "
            f"loss={val_stats['loss']:.6f} anchor={val_stats['anchor']:.6f} "
            f"order={val_stats['order']:.6f} edge={val_stats['edge']:.6f} "
            f"ord_acc={val_stats['ord_acc']:.4f}"
        )
        print("-" * 70, flush=True)

        save_checkpoint(out_dir / "gnn_latest.pt", model, args, epoch, train_stats, val_stats)

        score_for_best = val_stats["loss"]
        if score_for_best < best_val:
            best_val = score_for_best
            save_checkpoint(out_dir / "gnn_best.pt", model, args, epoch, train_stats, val_stats)
            print(f"Saved best checkpoint to {out_dir / 'gnn_best.pt'}", flush=True)

    print("=" * 70)
    print("Training done")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Best checkpoint: {out_dir / 'gnn_best.pt'}")
    print("=" * 70, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="../")
    p.add_argument("--num-samples", type=int, default=4000)
    p.add_argument("--val-samples", type=int, default=300)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window-start", type=int, default=-1)
    p.add_argument("--val-start", type=int, default=-1)
    p.add_argument("--hidden-dim", type=int, default=160)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1.0)
    p.add_argument("--order-weight", type=float, default=0.25)
    p.add_argument("--edge-weight", type=float, default=0.08)
    p.add_argument("--boundary-anchor-boost", type=float, default=0.45)
    p.add_argument("--cluster-anchor-boost", type=float, default=0.20)
    p.add_argument("--order-pairs", type=int, default=4096)
    p.add_argument("--clear-ratio", type=float, default=1.8)
    p.add_argument("--min-order-gap", type=float, default=0.025)
    p.add_argument("--order-temp", type=float, default=0.08)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--print-every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
