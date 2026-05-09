from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Tuple

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


def _bbox_area(positions: torch.Tensor) -> torch.Tensor:
    x_min = positions[:, 0].min()
    y_min = positions[:, 1].min()
    x_max = (positions[:, 0] + positions[:, 2]).max()
    y_max = (positions[:, 1] + positions[:, 3]).max()
    return torch.relu(x_max - x_min) * torch.relu(y_max - y_min)


def _boundary_hinge(positions: torch.Tensor, inst: Instance) -> torch.Tensor:
    if not inst.boundary:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    x_min = positions[:, 0].min()
    y_min = positions[:, 1].min()
    x_max = (positions[:, 0] + positions[:, 2]).max()
    y_max = (positions[:, 1] + positions[:, 3]).max()
    penalty = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    scale = torch.clamp(torch.sqrt(torch.clamp(_bbox_area(positions), min=1.0)), min=1.0)
    for block, code in inst.boundary.items():
        rect = positions[block]
        if code & 1:
            penalty = penalty + torch.abs(rect[0] - x_min) / scale
        if code & 2:
            penalty = penalty + torch.abs(rect[0] + rect[2] - x_max) / scale
        if code & 4:
            penalty = penalty + torch.abs(rect[1] + rect[3] - y_max) / scale
        if code & 8:
            penalty = penalty + torch.abs(rect[1] - y_min) / scale
    return penalty / max(len(inst.boundary), 1)


def _group_adjacency_surrogate(positions: torch.Tensor, inst: Instance) -> torch.Tensor:
    penalty = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    count = 0
    for members in inst.cluster_groups.values():
        if len(members) <= 1:
            continue
        for offset, i in enumerate(members):
            a = positions[i]
            best = None
            for j in members[offset + 1 :]:
                b = positions[j]
                horizontal_gap = torch.minimum(torch.abs(a[0] + a[2] - b[0]), torch.abs(b[0] + b[2] - a[0]))
                vertical_overlap = torch.relu(torch.minimum(a[1] + a[3], b[1] + b[3]) - torch.maximum(a[1], b[1]))
                vertical_gap = torch.minimum(torch.abs(a[1] + a[3] - b[1]), torch.abs(b[1] + b[3] - a[1]))
                horizontal_overlap = torch.relu(torch.minimum(a[0] + a[2], b[0] + b[2]) - torch.maximum(a[0], b[0]))
                edge_gap = torch.minimum(horizontal_gap + 0.01 / torch.clamp(vertical_overlap, min=0.01),
                                         vertical_gap + 0.01 / torch.clamp(horizontal_overlap, min=0.01))
                best = edge_gap if best is None else torch.minimum(best, edge_gap)
            if best is not None:
                penalty = penalty + best
                count += 1
    return penalty / max(count, 1)


def _mib_shape_loss(positions: torch.Tensor, inst: Instance) -> torch.Tensor:
    penalty = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    count = 0
    for members in inst.mib_groups.values():
        if len(members) <= 1:
            continue
        ref = positions[members[0], 2:4].detach()
        for block in members[1:]:
            penalty = penalty + F.smooth_l1_loss(positions[block, 2:4], ref)
            count += 1
    return penalty / max(count, 1)


def compute_v1_loss(
    positions: torch.Tensor,
    target_positions_xywh: torch.Tensor,
    inst: Instance,
    metrics: torch.Tensor,
    weights: Mapping[str, float] | None = None,
    return_parts: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    target = target_positions_xywh.to(device=positions.device, dtype=positions.dtype)
    supervised = F.smooth_l1_loss(positions, target)
    overlap = _overlap_proxy(positions)
    snap = _hard_snap_penalty(positions, inst)
    boundary = _boundary_hinge(positions, inst)
    group = _group_adjacency_surrogate(positions, inst)
    mib = _mib_shape_loss(positions, inst)
    area = _bbox_area(positions) / torch.clamp(inst.area_targets[: inst.block_count].to(device=positions.device, dtype=positions.dtype).sum(), min=1.0)

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

    total = (
        weights.get("supervised", 1.0) * supervised
        + weights.get("proxy", 0.1) * proxy
        + weights.get("overlap", 10.0) * overlap
        + weights.get("snap", 1.0) * snap
        + weights.get("boundary", 2.0) * boundary
        + weights.get("group", 1.0) * group
        + weights.get("mib", 1.0) * mib
        + weights.get("area", 0.02) * area
    )
    parts = {
        "supervised": supervised.detach(),
        "proxy": proxy.detach(),
        "overlap": overlap.detach(),
        "snap": snap.detach(),
        "boundary": boundary.detach(),
        "group": group.detach(),
        "mib": mib.detach(),
        "area": area.detach(),
    }
    if return_parts:
        return total, parts
    return total
