from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from floorset_arch.models import Instance


ROOT = Path(__file__).resolve().parents[3]
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
if CONTEST_DIR.exists() and str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

try:
    from iccad2026_evaluate import compute_training_loss_differentiable
except Exception:
    compute_training_loss_differentiable = None


def _overlap_proxy(positions: torch.Tensor) -> torch.Tensor:
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    penalty = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            ox = torch.relu(torch.minimum(x[i] + w[i], x[j] + w[j]) - torch.maximum(x[i], x[j]))
            oy = torch.relu(torch.minimum(y[i] + h[i], y[j] + h[j]) - torch.maximum(y[i], y[j]))
            penalty = penalty + ox * oy
    return penalty / torch.clamp((w * h).sum(), min=1.0)


def _hard_snap_penalty(positions: torch.Tensor, inst: Instance) -> torch.Tensor:
    penalty = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    for block in inst.fixed | inst.preplaced:
        target = inst.target_rects.get(block)
        if target is None:
            continue
        target_dims = torch.tensor([target.width, target.height], dtype=positions.dtype, device=positions.device)
        penalty = penalty + F.smooth_l1_loss(positions[block, 2:4], target_dims)
        if block in inst.preplaced:
            target_xy = torch.tensor([target.x, target.y], dtype=positions.dtype, device=positions.device)
            penalty = penalty + F.smooth_l1_loss(positions[block, 0:2], target_xy)
    return penalty


def compute_v1_loss(
    positions: torch.Tensor,
    target_positions_xywh: torch.Tensor,
    inst: Instance,
    metrics: torch.Tensor,
    return_parts: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target_positions_xywh.to(device=positions.device, dtype=positions.dtype)
    supervised = F.smooth_l1_loss(positions, target)
    overlap = _overlap_proxy(positions)
    snap = _hard_snap_penalty(positions, inst)

    if compute_training_loss_differentiable is not None:
        proxy = compute_training_loss_differentiable(
            positions,
            inst.valid_b2b.to(device=positions.device, dtype=positions.dtype),
            inst.valid_p2b.to(device=positions.device, dtype=positions.dtype),
            inst.pins_pos.to(device=positions.device, dtype=positions.dtype),
            inst.area_targets[: inst.block_count].to(device=positions.device, dtype=positions.dtype),
            metrics.to(device=positions.device, dtype=positions.dtype),
        )
    else:
        proxy = supervised

    total = supervised + 0.1 * proxy + 10.0 * overlap + snap
    parts = {
        "supervised": supervised.detach(),
        "proxy": proxy.detach(),
        "overlap": overlap.detach(),
        "snap": snap.detach(),
    }
    if return_parts:
        return total, parts
    return total

