#!/usr/bin/env python3
"""Train a graph-conditioned v-prediction diffusion model for FloorSet."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader
from diffusion_data import batch_to_device, build_condition, fp_sol_to_z0, z_to_rectangles
from diffusion_model import DiffusionSchedule, GraphDiffusionDenoiser, ModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="../")
    p.add_argument("--checkpoint-dir", default="checkpoints/diffusion")
    p.add_argument("--resume", default=None, help="checkpoint path or 'latest'")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--z-repr", choices=["xylogwh", "xyaspect"], default="xyaspect")
    p.add_argument("--relation-feat-dim", type=int, default=8)
    p.add_argument("--node-feat-dim", type=int, default=26)
    p.add_argument("--disable-z0-head", action="store_true")
    p.add_argument("--z0-loss-weight", type=float, default=0.50)
    p.add_argument("--overlap-loss-weight", type=float, default=0.12)
    p.add_argument("--boundary-loss-weight", type=float, default=0.05)
    p.add_argument("--corner-boundary-loss-weight", type=float, default=0.03)
    p.add_argument("--cluster-loss-weight", type=float, default=0.05)
    p.add_argument("--cluster-overlap-loss-weight", type=float, default=0.20)
    p.add_argument("--mib-loss-weight", type=float, default=0.02)
    p.add_argument("--shape-loss-weight", type=float, default=1.20)
    p.add_argument("--aspect-loss-weight", type=float, default=1.20)
    p.add_argument("--wh-loss-weight", type=float, default=1.00)
    p.add_argument("--separation-loss-weight", type=float, default=0.50)
    p.add_argument("--order-loss-weight", type=float, default=0.12)
    p.add_argument("--order-loss-pairs", type=int, default=4096)
    p.add_argument("--gt-expand-scale", type=float, default=1.0, help="expand GT x/y distances from bbox origin; W/H unchanged")
    p.add_argument("--gt-expand-jitter", type=float, default=0.0, help="uniform +/- jitter added to gt expand scale during training")
    p.add_argument("--require-cuda", action="store_true", help="fail fast unless the selected device is CUDA")
    p.add_argument("--amp", action="store_true", help="use CUDA mixed precision to reduce activation memory")
    p.add_argument("--gradient-checkpointing", action="store_true", help="recompute GNN activations during backward to reduce VRAM")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_checkpoint(
    model: GraphDiffusionDenoiser,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    epoch: int,
    best_loss: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "best_val_loss": best_loss,
        "config": vars(args),
        "model_config": model.config.__dict__,
        "rng_state": rng_state(),
    }


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path, name: str) -> Path:
    path = checkpoint_dir / name
    atomic_torch_save(state, path)
    if name != "latest.pt":
        atomic_torch_save(state, checkpoint_dir / "latest.pt")
    print(f"saved checkpoint: {path}")
    return path


def resolve_resume(checkpoint_dir: Path, resume: Optional[str]) -> Optional[Path]:
    if not resume:
        return None
    return checkpoint_dir / "latest.pt" if resume == "latest" else Path(resume)


def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0) / loss.shape[-1]


def expanded_fp_target(
    fp_sol: torch.Tensor,
    area_target: torch.Tensor,
    args: argparse.Namespace,
    train: bool = True,
    constraints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return fp target [w,h,x,y] with optional GT clearance expansion.

    This keeps W/H exactly equal to the official GT and only scales x/y offsets
    from the GT bbox lower-left.  The model therefore learns aspect accurately
    while seeing a less-overlapped placement target.
    """
    scale_base = float(getattr(args, "gt_expand_scale", 1.0))
    jitter = float(getattr(args, "gt_expand_jitter", 0.0)) if train else 0.0
    if abs(scale_base - 1.0) < 1e-9 and jitter <= 0.0:
        return fp_sol
    mask = valid_mask = area_target != -1
    x = fp_sol[..., 2]
    y = fp_sol[..., 3]
    inf = torch.full_like(x, float("inf"))
    min_x = torch.where(valid_mask, x, inf).min(dim=1).values.unsqueeze(-1)
    min_y = torch.where(valid_mask, y, inf).min(dim=1).values.unsqueeze(-1)
    if jitter > 0.0:
        scale = scale_base + (torch.rand((fp_sol.shape[0], 1), dtype=fp_sol.dtype, device=fp_sol.device) * 2.0 - 1.0) * jitter
        scale = scale.clamp_min(1.0)
    else:
        scale = fp_sol.new_full((fp_sol.shape[0], 1), scale_base)
    out = fp_sol.clone()
    out[..., 2] = torch.where(mask, min_x + (x - min_x) * scale, x)
    out[..., 3] = torch.where(mask, min_y + (y - min_y) * scale, y)
    if constraints is not None and constraints.shape[-1] > 1:
        preplaced = (constraints[..., 1] != 0) & mask
        out[..., 2] = torch.where(preplaced, fp_sol[..., 2], out[..., 2])
        out[..., 3] = torch.where(preplaced, fp_sol[..., 3], out[..., 3])
    return out


def known_target_positions_from_fp(fp_sol: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Build the fixed/preplaced geometry that is also available at inference.

    Soft-block GT is deliberately not copied here.  Fixed/preplaced W/H and
    preplaced X/Y are constraint inputs, so using them keeps train/test inputs
    aligned while giving the graph encoder the anchor geometry it needs.
    """
    out = torch.full_like(fp_sol, -1.0)
    if constraints.shape[-1] <= 1:
        return out
    fixed = constraints[..., 0] != 0 if constraints.shape[-1] > 0 else torch.zeros_like(fp_sol[..., 0], dtype=torch.bool)
    preplaced = constraints[..., 1] != 0
    wh_known = fixed | preplaced
    out[..., 2] = torch.where(wh_known, fp_sol[..., 0], out[..., 2])
    out[..., 3] = torch.where(wh_known, fp_sol[..., 1], out[..., 3])
    out[..., 0] = torch.where(preplaced, fp_sol[..., 2], out[..., 0])
    out[..., 1] = torch.where(preplaced, fp_sol[..., 3], out[..., 1])
    return out


def soft_overlap_loss(rects: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    overlap = pair_overlap_area(rects)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    area_scale = (rects[..., 2] * rects[..., 3]).clamp_min(1e-6).mean(dim=1).view(-1, 1, 1).detach()
    normalized = (overlap / area_scale) * pair_mask
    active = normalized > 1e-8
    if torch.any(active):
        active_loss = normalized.masked_select(active).mean()
        density = active.to(normalized.dtype).sum() / pair_mask.to(normalized.dtype).sum().clamp_min(1.0)
        return active_loss + density
    return normalized.sum() * 0.0


def pair_overlap_area(rects: torch.Tensor) -> torch.Tensor:
    x1, y1, w, h = rects.unbind(dim=-1)
    x2 = x1 + w
    y2 = y1 + h
    ox = (torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1)) - torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))).clamp_min(0)
    oy = (torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1)) - torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))).clamp_min(0)
    return ox * oy


def detached_layout_metrics(rects: torch.Tensor, gt_rects: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    overlap = pair_overlap_area(rects)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    overlap_pair_frac = (((overlap > 1e-7) & pair_mask).to(rects.dtype).sum() / pair_mask.to(rects.dtype).sum().clamp_min(1.0)).detach()
    pred_aspect = torch.log(rects[..., 2].clamp_min(1e-6) / rects[..., 3].clamp_min(1e-6))
    gt_aspect = torch.log(gt_rects[..., 2].clamp_min(1e-6) / gt_rects[..., 3].clamp_min(1e-6))
    aspect_mae = ((pred_aspect - gt_aspect).abs() * mask.to(rects.dtype)).sum() / mask.to(rects.dtype).sum().clamp_min(1.0)
    pred_log_wh = torch.log(rects[..., 2:4].clamp_min(1e-6))
    gt_log_wh = torch.log(gt_rects[..., 2:4].clamp_min(1e-6))
    wh_log_mae = ((pred_log_wh - gt_log_wh).abs() * mask.unsqueeze(-1).to(rects.dtype)).sum()
    wh_log_mae = wh_log_mae / (2.0 * mask.to(rects.dtype).sum().clamp_min(1.0))
    return {
        "overlap_pair_frac": overlap_pair_frac,
        "aspect_mae": aspect_mae.detach(),
        "wh_log_mae": wh_log_mae.detach(),
    }


def boundary_loss(rects: torch.Tensor, constraints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if constraints.shape[-1] <= 4:
        return rects.new_tensor(0.0)
    x, y, w, h = rects.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    inf = torch.full_like(x, float("inf"))
    ninf = torch.full_like(x, -float("inf"))
    min_x = torch.where(mask, x, inf).min(dim=1).values.unsqueeze(-1)
    min_y = torch.where(mask, y, inf).min(dim=1).values.unsqueeze(-1)
    max_x = torch.where(mask, x2, ninf).max(dim=1).values.unsqueeze(-1)
    max_y = torch.where(mask, y2, ninf).max(dim=1).values.unsqueeze(-1)
    code = constraints[..., 4].long()
    active = (code != 0) & mask
    scale = torch.sqrt(((max_x - min_x).clamp_min(1.0) * (max_y - min_y).clamp_min(1.0))).detach()
    losses = []
    if torch.any(code & 1):
        losses.append(((x - min_x).abs() / scale) * ((code & 1) != 0))
    if torch.any(code & 2):
        losses.append(((max_x - x2).abs() / scale) * ((code & 2) != 0))
    if torch.any(code & 4):
        losses.append(((max_y - y2).abs() / scale) * ((code & 4) != 0))
    if torch.any(code & 8):
        losses.append(((y - min_y).abs() / scale) * ((code & 8) != 0))
    if not losses:
        return rects.new_tensor(0.0)
    loss = torch.stack(losses, dim=0).sum(dim=0)
    return (loss * active).sum() / active.sum().clamp_min(1)


def corner_boundary_loss(rects: torch.Tensor, constraints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Extra loss for blocks constrained to two or more boundary sides.

    General boundary loss treats an RB corner block and a plain R block with the
    same weight.  The legalizer failure we observed is exactly that a plain R
    block can occupy the corner while RB stays only on bottom.  This term makes
    corner contacts more expensive to miss during training.
    """
    if constraints.shape[-1] <= 4:
        return rects.new_tensor(0.0)
    x, y, w, h = rects.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    inf = torch.full_like(x, float("inf"))
    ninf = torch.full_like(x, -float("inf"))
    min_x = torch.where(mask, x, inf).min(dim=1).values.unsqueeze(-1)
    min_y = torch.where(mask, y, inf).min(dim=1).values.unsqueeze(-1)
    max_x = torch.where(mask, x2, ninf).max(dim=1).values.unsqueeze(-1)
    max_y = torch.where(mask, y2, ninf).max(dim=1).values.unsqueeze(-1)
    code = constraints[..., 4].long()
    bit_count = (
        ((code & 1) != 0).long()
        + ((code & 2) != 0).long()
        + ((code & 4) != 0).long()
        + ((code & 8) != 0).long()
    )
    active = (bit_count >= 2) & mask
    if not torch.any(active):
        return rects.new_tensor(0.0)
    scale = torch.sqrt(((max_x - min_x).clamp_min(1.0) * (max_y - min_y).clamp_min(1.0))).detach()
    loss = torch.zeros_like(x)
    loss = loss + (((x - min_x).abs() / scale) ** 2) * ((code & 1) != 0)
    loss = loss + (((max_x - x2).abs() / scale) ** 2) * ((code & 2) != 0)
    loss = loss + (((max_y - y2).abs() / scale) ** 2) * ((code & 4) != 0)
    loss = loss + (((y - min_y).abs() / scale) ** 2) * ((code & 8) != 0)
    return (loss * active).sum() / active.sum().clamp_min(1)


def shape_loss(z_pred: torch.Tensor, z_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Explicitly supervise W/H channels beyond the generic z0 loss."""
    loss = F.smooth_l1_loss(z_pred[..., 2:4], z_target[..., 2:4], reduction="none")
    weight = mask.unsqueeze(-1).to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0) / 2.0


def aspect_loss(z_pred: torch.Tensor, z_target: torch.Tensor, mask: torch.Tensor, z_repr: str = "xylogwh") -> torch.Tensor:
    """Area-preserving legalizers are sensitive to wrong aspect orientation."""
    if z_repr == "xyaspect":
        pred = z_pred[..., 2]
        target = z_target[..., 2]
    else:
        pred = z_pred[..., 2] - z_pred[..., 3]
        target = z_target[..., 2] - z_target[..., 3]
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask.to(loss.dtype)).sum() / mask.sum().clamp_min(1.0)


def wh_relative_loss(pred_rects: torch.Tensor, gt_rects: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Direct W/H supervision in rectangle space, not only normalized z space."""
    pred_log = torch.log(pred_rects[..., 2:4].clamp_min(1e-6))
    gt_log = torch.log(gt_rects[..., 2:4].clamp_min(1e-6))
    loss = F.smooth_l1_loss(pred_log, gt_log, reduction="none")
    weight = mask.unsqueeze(-1).to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0) / 2.0


def pairwise_separation_loss(
    pred_rects: torch.Tensor,
    gt_rects: torch.Tensor,
    mask: torch.Tensor,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Harder no-overlap supervision from GT topology.

    For pairs separated in expanded GT, require the same separating axis in the
    prediction with a small positive margin.  This directly teaches a legalizer-
    friendly topology instead of only penalizing overlap area after it happens.
    """
    px, py, pw, ph = pred_rects.unbind(dim=-1)
    gx, gy, gw, gh = gt_rects.unbind(dim=-1)
    px2 = px + pw
    py2 = py + ph
    gx2 = gx + gw
    gy2 = gy + gh
    mean_size = torch.sqrt((gt_rects[..., 2] * gt_rects[..., 3]).clamp_min(1e-6)).mean(dim=1)
    losses = []
    per_batch_pairs = max(1, int(max_pairs) // max(1, pred_rects.shape[0]))
    for b in range(pred_rects.shape[0]):
        valid = torch.where(mask[b])[0]
        if valid.numel() <= 1:
            continue
        i_all, j_all = torch.meshgrid(valid, valid, indexing="ij")
        i_all = i_all.reshape(-1)
        j_all = j_all.reshape(-1)
        nonself = i_all != j_all
        i_all = i_all[nonself]
        j_all = j_all[nonself]
        left = gx2[b, i_all] <= gx[b, j_all]
        below = gy2[b, i_all] <= gy[b, j_all]
        relation = left | below
        i_all = i_all[relation]
        j_all = j_all[relation]
        left = left[relation]
        below = below[relation]
        if i_all.numel() == 0:
            continue
        inf = torch.full_like(gx[b, i_all], float("inf"))
        sep_gap = torch.minimum(
            torch.where(left, (gx[b, j_all] - gx2[b, i_all]).clamp_min(0.0), inf),
            torch.where(below, (gy[b, j_all] - gy2[b, i_all]).clamp_min(0.0), inf),
        )
        if i_all.numel() > per_batch_pairs:
            keep = torch.topk(-sep_gap, per_batch_pairs).indices
            i_all = i_all[keep]
            j_all = j_all[keep]
            left = left[keep]
            below = below[keep]
        margin = (0.015 * mean_size[b]).detach()
        vals = []
        if torch.any(left):
            v = F.relu(px2[b, i_all[left]] + margin - px[b, j_all[left]]) / mean_size[b].detach().clamp_min(1.0)
            vals.append(F.smooth_l1_loss(v, torch.zeros_like(v), reduction="none", beta=0.5))
        if torch.any(below):
            v = F.relu(py2[b, i_all[below]] + margin - py[b, j_all[below]]) / mean_size[b].detach().clamp_min(1.0)
            vals.append(F.smooth_l1_loss(v, torch.zeros_like(v), reduction="none", beta=0.5))
        if vals:
            losses.append(torch.cat(vals).mean())
    if not losses:
        return pred_rects.new_tensor(0.0)
    return torch.stack(losses).mean()


def pairwise_order_loss(
    pred_rects: torch.Tensor,
    gt_rects: torch.Tensor,
    mask: torch.Tensor,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Supervise GT left/right and below/above relations without a discrete tree.

    The legalizer preserves relative order.  This loss teaches the denoiser the
    same relation graph directly from GT rectangles, which is safer than asking
    the model to emit one arbitrary B*-tree representation.
    """
    px, py, pw, ph = pred_rects.unbind(dim=-1)
    gx, gy, gw, gh = gt_rects.unbind(dim=-1)
    px2 = px + pw
    py2 = py + ph
    gx2 = gx + gw
    gy2 = gy + gh

    mean_size = torch.sqrt((pred_rects[..., 2] * pred_rects[..., 3]).clamp_min(1e-6)).mean(dim=1)
    losses = []
    per_batch_pairs = max(1, int(max_pairs) // max(1, pred_rects.shape[0]))
    for b in range(pred_rects.shape[0]):
        valid = torch.where(mask[b])[0]
        if valid.numel() <= 1:
            continue
        i_all, j_all = torch.meshgrid(valid, valid, indexing="ij")
        i_all = i_all.reshape(-1)
        j_all = j_all.reshape(-1)
        nonself = i_all != j_all
        i_all = i_all[nonself]
        j_all = j_all[nonself]
        relation = (gx2[b, i_all] <= gx[b, j_all]) | (gy2[b, i_all] <= gy[b, j_all])
        i_all = i_all[relation]
        j_all = j_all[relation]
        if i_all.numel() == 0:
            continue
        if i_all.numel() > per_batch_pairs:
            perm = torch.randperm(i_all.numel(), device=pred_rects.device)[:per_batch_pairs]
            i_all = i_all[perm]
            j_all = j_all[perm]
        margin = (0.01 * mean_size[b]).detach()
        vals = []
        left = gx2[b, i_all] <= gx[b, j_all]
        if torch.any(left):
            vals.append(F.relu(px2[b, i_all[left]] + margin - px[b, j_all[left]]))
        below = gy2[b, i_all] <= gy[b, j_all]
        if torch.any(below):
            vals.append(F.relu(py2[b, i_all[below]] + margin - py[b, j_all[below]]))
        if vals:
            losses.append(torch.cat(vals).mean() / mean_size[b].detach().clamp_min(1.0))
    if not losses:
        return pred_rects.new_tensor(0.0)
    return torch.stack(losses).mean()


def edge_distance_matrix(rects: torch.Tensor) -> torch.Tensor:
    x, y, w, h = rects.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    dx = torch.maximum(torch.maximum(x.unsqueeze(2) - x2.unsqueeze(1), x.unsqueeze(1) - x2.unsqueeze(2)), torch.zeros_like(x.unsqueeze(2)))
    dy = torch.maximum(torch.maximum(y.unsqueeze(2) - y2.unsqueeze(1), y.unsqueeze(1) - y2.unsqueeze(2)), torch.zeros_like(y.unsqueeze(2)))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def cluster_loss(
    rects: torch.Tensor,
    constraints: torch.Tensor,
    mask: torch.Tensor,
    target_rects: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if constraints.shape[-1] <= 3:
        return rects.new_tensor(0.0)
    cluster = constraints[..., 3].long()
    dist = edge_distance_matrix(rects)
    target_dist = edge_distance_matrix(target_rects) if target_rects is not None else None
    losses = []
    for b in range(rects.shape[0]):
        max_group = int(cluster[b].max().detach().cpu().item()) if cluster[b].numel() else 0
        for g in range(1, max_group + 1):
            idx = torch.where((cluster[b] == g) & mask[b])[0]
            if idx.numel() <= 1:
                continue
            d = dist[b][idx][:, idx]
            eye = torch.eye(idx.numel(), dtype=torch.bool, device=rects.device)
            if target_dist is None:
                nearest = d.masked_fill(eye, float("inf")).min(dim=1).values
                losses.append(nearest.mean())
            else:
                td = target_dist[b][idx][:, idx]
                nearest_j = td.masked_fill(eye, float("inf")).argmin(dim=1)
                row = torch.arange(idx.numel(), device=rects.device)
                pred_gap = d[row, nearest_j]
                target_gap = td[row, nearest_j].detach()
                scale = torch.sqrt((target_rects[b, idx, 2] * target_rects[b, idx, 3]).clamp_min(1e-6)).mean().detach().clamp_min(1.0)
                losses.append(F.smooth_l1_loss(pred_gap / scale, target_gap / scale, reduction="mean", beta=0.2))
    if not losses:
        return rects.new_tensor(0.0)
    if target_dist is not None:
        return torch.stack(losses).mean()
    scale = torch.sqrt((rects[..., 2] * rects[..., 3]).clamp_min(1e-6)).mean().detach().clamp_min(1.0)
    return torch.stack(losses).mean() / scale


def cluster_overlap_loss(rects: torch.Tensor, constraints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if constraints.shape[-1] <= 3:
        return rects.new_tensor(0.0)
    cluster = constraints[..., 3].long()
    overlap = pair_overlap_area(rects)
    area_scale = (rects[..., 2] * rects[..., 3]).clamp_min(1e-6).mean(dim=1).view(-1, 1, 1).detach()
    losses = []
    for b in range(rects.shape[0]):
        max_group = int(cluster[b].max().detach().cpu().item()) if cluster[b].numel() else 0
        for g in range(1, max_group + 1):
            idx = torch.where((cluster[b] == g) & mask[b])[0]
            if idx.numel() <= 1:
                continue
            pair = overlap[b][idx][:, idx]
            eye = torch.eye(idx.numel(), dtype=torch.bool, device=rects.device)
            pair_mask = ~eye
            losses.append((pair / area_scale[b]).masked_select(pair_mask).mean())
    if not losses:
        return rects.new_tensor(0.0)
    return torch.stack(losses).mean()


def mib_shape_loss(rects: torch.Tensor, constraints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if constraints.shape[-1] <= 2:
        return rects.new_tensor(0.0)
    mib = constraints[..., 2].long()
    log_aspect = torch.log(rects[..., 2].clamp_min(1e-6) / rects[..., 3].clamp_min(1e-6))
    losses = []
    for b in range(rects.shape[0]):
        max_group = int(mib[b].max().detach().cpu().item()) if mib[b].numel() else 0
        for g in range(1, max_group + 1):
            idx = torch.where((mib[b] == g) & mask[b])[0]
            if idx.numel() <= 1:
                continue
            vals = log_aspect[b, idx]
            losses.append(((vals - vals.mean()) ** 2).mean())
    if not losses:
        return rects.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_loss(
    model: GraphDiffusionDenoiser,
    schedule: DiffusionSchedule,
    batch,
    args: argparse.Namespace,
    train: bool = True,
) -> Dict[str, torch.Tensor]:
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, _tree_sol, fp_sol, _metrics = batch
    fp_target = expanded_fp_target(fp_sol, area_target, args, train=train, constraints=constraints)
    target_positions = known_target_positions_from_fp(fp_sol, constraints)
    z_repr = getattr(model.config, "z_repr", getattr(args, "z_repr", "xylogwh"))
    z0, mask, _scale = fp_sol_to_z0(fp_target, area_target, z_repr=z_repr)
    cond = build_condition(
        area_target,
        b2b_conn,
        p2b_conn,
        pins_pos,
        constraints,
        target_positions=target_positions,
        relation_feat_dim=getattr(model.config, "relation_feat_dim", 0),
        node_feat_dim=getattr(model.config, "node_feat_dim", 13),
    )
    bsz = area_target.shape[0]
    t = torch.randint(0, schedule.timesteps, (bsz,), device=area_target.device)
    noise = torch.randn_like(z0)
    alpha, sigma = schedule.alpha_sigma(t)
    z_t = (alpha * z0 + sigma * noise) * mask.unsqueeze(-1)
    v_target = (alpha * noise - sigma * z0) * mask.unsqueeze(-1)
    v_pred, z0_pred = model(
        z_t,
        t,
        cond["node_feat"],
        cond["adj"],
        cond["mask"],
        rel_feat=cond.get("rel_feat"),
        return_z0=True,
    )
    v_loss = masked_mean((v_pred - v_target) ** 2, mask)
    z0_hat = (alpha * z_t - sigma * v_pred) * mask.unsqueeze(-1)
    recon_loss = masked_mean(F.smooth_l1_loss(z0_hat, z0, reduction="none"), mask)
    z0_loss = masked_mean(F.smooth_l1_loss(z0_hat, z0, reduction="none"), mask)
    if z0_pred is not None:
        z0_loss = z0_loss + 0.50 * masked_mean(F.smooth_l1_loss(z0_pred, z0, reduction="none"), mask)
    rects = z_to_rectangles(
        z0_hat,
        area_target,
        target_positions=target_positions,
        constraints=constraints,
        z_repr=z_repr,
    )
    gt_rects = torch.stack([fp_target[..., 2], fp_target[..., 3], fp_target[..., 0], fp_target[..., 1]], dim=-1)
    overlap = z0.new_tensor(0.0)
    boundary = z0.new_tensor(0.0)
    corner_boundary = z0.new_tensor(0.0)
    cluster = z0.new_tensor(0.0)
    cluster_overlap = z0.new_tensor(0.0)
    mib = z0.new_tensor(0.0)
    shape = z0.new_tensor(0.0)
    aspect = z0.new_tensor(0.0)
    wh = z0.new_tensor(0.0)
    separation = z0.new_tensor(0.0)
    order = z0.new_tensor(0.0)
    if args.overlap_loss_weight > 0:
        overlap = soft_overlap_loss(rects, mask)
    if args.boundary_loss_weight > 0:
        boundary = boundary_loss(rects, constraints, mask)
    if args.corner_boundary_loss_weight > 0:
        corner_boundary = corner_boundary_loss(rects, constraints, mask)
    if args.cluster_loss_weight > 0:
        cluster = cluster_loss(rects, constraints, mask, target_rects=gt_rects)
    if args.cluster_overlap_loss_weight > 0:
        cluster_overlap = cluster_overlap_loss(rects, constraints, mask)
    if args.mib_loss_weight > 0:
        mib = mib_shape_loss(rects, constraints, mask)
    if args.shape_loss_weight > 0:
        shape = shape_loss(z0_hat, z0, mask)
    if args.aspect_loss_weight > 0:
        aspect = aspect_loss(z0_hat, z0, mask, z_repr=z_repr)
    if args.wh_loss_weight > 0:
        wh = wh_relative_loss(rects, gt_rects, mask)
    if args.separation_loss_weight > 0:
        separation = pairwise_separation_loss(rects, gt_rects, mask, max_pairs=args.order_loss_pairs)
    if args.order_loss_weight > 0:
        order = pairwise_order_loss(rects, gt_rects, mask, max_pairs=args.order_loss_pairs)
    metrics = detached_layout_metrics(rects, gt_rects, mask)
    loss = (
        v_loss
        + 0.20 * recon_loss
        + args.z0_loss_weight * z0_loss
        + args.overlap_loss_weight * overlap
        + args.boundary_loss_weight * boundary
        + args.corner_boundary_loss_weight * corner_boundary
        + args.cluster_loss_weight * cluster
        + args.cluster_overlap_loss_weight * cluster_overlap
        + args.mib_loss_weight * mib
        + args.shape_loss_weight * shape
        + args.aspect_loss_weight * aspect
        + args.wh_loss_weight * wh
        + args.separation_loss_weight * separation
        + args.order_loss_weight * order
    )
    return {
        "loss": loss,
        "v_loss": v_loss.detach(),
        "recon_loss": recon_loss.detach(),
        "z0_loss": z0_loss.detach(),
        "overlap_loss": overlap.detach(),
        "boundary_loss": boundary.detach(),
        "corner_boundary_loss": corner_boundary.detach(),
        "cluster_loss": cluster.detach(),
        "cluster_overlap_loss": cluster_overlap.detach(),
        "mib_loss": mib.detach(),
        "shape_loss": shape.detach(),
        "aspect_loss": aspect.detach(),
        "wh_loss": wh.detach(),
        "separation_loss": separation.detach(),
        "order_loss": order.detach(),
        "overlap_pair_frac": metrics["overlap_pair_frac"],
        "aspect_mae": metrics["aspect_mae"],
        "wh_log_mae": metrics["wh_log_mae"],
    }


@torch.no_grad()
def validate(model: GraphDiffusionDenoiser, schedule: DiffusionSchedule, loader, device: torch.device, args: argparse.Namespace) -> float:
    model.eval()
    losses = []
    for batch in loader:
        if isinstance(batch, list) and len(batch) == 8:
            batch = batch_to_device(batch, device)
            losses.append(float(train_loss(model, schedule, batch, args, train=False)["loss"].item()))
        if len(losses) >= 4:
            break
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def append_log(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise SystemExit(f"--require-cuda requested, but selected device is {device}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda requested, but torch.cuda.is_available() is false")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = GraphDiffusionDenoiser(
        ModelConfig(
            d_model=args.d_model,
            layers=args.layers,
            dropout=args.dropout,
            timesteps=args.timesteps,
            z_repr=args.z_repr,
            node_feat_dim=args.node_feat_dim,
            relation_feat_dim=max(0, args.relation_feat_dim),
            predict_z0_head=not args.disable_z0_head,
        )
    ).to(device)
    model.gradient_checkpointing = bool(args.gradient_checkpointing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.max_steps, 1), eta_min=args.lr * 0.05)
    diffusion = DiffusionSchedule(args.timesteps, device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    global_step = 0
    epoch = 0
    best_loss = float("inf")

    resume_path = resolve_resume(checkpoint_dir, args.resume)
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = int(ckpt.get("global_step", 0))
        epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_val_loss", float("inf")))
        restore_rng_state(ckpt.get("rng_state", {}))
        print(f"resumed from {resume_path} at step {global_step}, epoch {epoch}")
    elif resume_path is not None:
        print(f"resume checkpoint not found: {resume_path}; starting fresh")

    train_loader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = train_loader

    log_path = checkpoint_dir / "train_log.jsonl"
    model.train()
    stop_requested = False
    accum_steps = max(1, int(args.grad_accum_steps))

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; will save checkpoint after this step")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while global_step < args.max_steps:
        epoch += 1
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                losses = train_loss(model, diffusion, batch, args)
                scaled_loss = losses["loss"] / accum_steps
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % accum_steps != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            lr = scheduler.get_last_lr()[0]
            row = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(losses["loss"].detach().item()),
                "v_loss": float(losses["v_loss"].item()),
                "recon_loss": float(losses["recon_loss"].item()),
                "z0_loss": float(losses["z0_loss"].item()),
                "overlap_loss": float(losses["overlap_loss"].item()),
                "boundary_loss": float(losses["boundary_loss"].item()),
                "corner_boundary_loss": float(losses["corner_boundary_loss"].item()),
                "cluster_loss": float(losses["cluster_loss"].item()),
                "cluster_overlap_loss": float(losses["cluster_overlap_loss"].item()),
                "mib_loss": float(losses["mib_loss"].item()),
                "shape_loss": float(losses["shape_loss"].item()),
                "aspect_loss": float(losses["aspect_loss"].item()),
                "wh_loss": float(losses["wh_loss"].item()),
                "separation_loss": float(losses["separation_loss"].item()),
                "order_loss": float(losses["order_loss"].item()),
                "overlap_pair_frac": float(losses["overlap_pair_frac"].item()),
                "aspect_mae": float(losses["aspect_mae"].item()),
                "wh_log_mae": float(losses["wh_log_mae"].item()),
                "lr": float(lr),
            }
            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"step {global_step} epoch {epoch} loss {row['loss']:.5f} "
                    f"v {row['v_loss']:.5f} recon {row['recon_loss']:.5f} "
                    f"z0 {row['z0_loss']:.5f} "
                    f"ov {row['overlap_loss']:.5f} bd {row['boundary_loss']:.5f} "
                    f"cbd {row['corner_boundary_loss']:.5f} "
                    f"cl {row['cluster_loss']:.5f} cov {row['cluster_overlap_loss']:.5f} "
                    f"mib {row['mib_loss']:.5f} shape {row['shape_loss']:.5f} "
                    f"asp {row['aspect_loss']:.5f} wh {row['wh_loss']:.5f} "
                    f"sep {row['separation_loss']:.5f} ord {row['order_loss']:.5f} "
                    f"ovp {row['overlap_pair_frac']:.4f} amae {row['aspect_mae']:.4f} "
                    f"whmae {row['wh_log_mae']:.4f} lr {lr:.2e}"
                )
            append_log(log_path, row)
            if global_step % args.save_every == 0:
                state = make_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, args)
                save_checkpoint(state, checkpoint_dir, f"step_{global_step:08d}.pt")
                val_loss = validate(model, diffusion, val_loader, device, args)
                if val_loss < best_loss:
                    best_loss = val_loss
                    state = make_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, args)
                    save_checkpoint(state, checkpoint_dir, "best.pt")
            if stop_requested:
                state = make_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, args)
                save_checkpoint(state, checkpoint_dir, "interrupt.pt")
                save_checkpoint(state, checkpoint_dir, "latest.pt")
                print(f"stopped by signal; checkpoint saved at step {global_step}")
                raise SystemExit(130)
            if global_step >= args.max_steps:
                break
        state = make_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, args)
        save_checkpoint(state, checkpoint_dir, "latest.pt")

    val_loss = validate(model, diffusion, val_loader, device, args)
    if val_loss < best_loss:
        best_loss = val_loss
        state = make_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, args)
        save_checkpoint(state, checkpoint_dir, "best.pt")
    state = make_checkpoint(model, optimizer, scheduler, global_step, epoch, best_loss, args)
    save_checkpoint(state, checkpoint_dir, "latest.pt")
    print(f"done: step {global_step}, best_loss {best_loss:.6f}")


if __name__ == "__main__":
    main()
