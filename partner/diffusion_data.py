#!/usr/bin/env python3
"""Data adapters for graph-conditioned layout diffusion.

The official training label ``fp_sol`` is ``[w, h, x, y]``.  The default
diffusion target is a normalized rectangle representation:
``z = [x / S, y / S, log(w / S), log(h / S)]``, where ``S`` is the square root
of the valid target-area sum.  Older 3D checkpoints using
``[cx / S, cy / S, log(w / h)]`` are still supported at inference time.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


LOG_ASPECT_MIN = -3.0
LOG_ASPECT_MAX = 3.0
LOG_SIZE_MIN = -8.0
LOG_SIZE_MAX = 2.0


def valid_block_mask(area_target: torch.Tensor) -> torch.Tensor:
    """Return ``[B, N]`` or ``[N]`` mask for non-padding blocks."""
    return area_target != -1


def layout_scale(area_target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute S = sqrt(sum valid target areas), preserving batch dims."""
    if mask is None:
        mask = valid_block_mask(area_target)
    areas = torch.where(mask, area_target.clamp_min(0.0), torch.zeros_like(area_target))
    return torch.sqrt(areas.sum(dim=-1).clamp_min(1.0))


def fp_sol_to_z0(
    fp_sol: torch.Tensor,
    area_target: torch.Tensor,
    z_repr: str = "xylogwh",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert padded ``fp_sol`` rectangles ``[..., N, 4]`` from ``[w,h,x,y]`` to z0."""
    mask = valid_block_mask(area_target)
    scale = layout_scale(area_target, mask)
    while scale.dim() < fp_sol.dim() - 1:
        scale = scale.unsqueeze(-1)

    w = fp_sol[..., 0].clamp_min(1e-6)
    h = fp_sol[..., 1].clamp_min(1e-6)
    x = fp_sol[..., 2]
    y = fp_sol[..., 3]
    if z_repr == "xyaspect":
        z0 = torch.stack(
            [
                x / scale,
                y / scale,
                torch.log(w / h).clamp(LOG_ASPECT_MIN, LOG_ASPECT_MAX),
                torch.zeros_like(w),
            ],
            dim=-1,
        )
    else:
        z0 = torch.stack(
            [
                x / scale,
                y / scale,
                torch.log(w / scale).clamp(LOG_SIZE_MIN, LOG_SIZE_MAX),
                torch.log(h / scale).clamp(LOG_SIZE_MIN, LOG_SIZE_MAX),
            ],
            dim=-1,
        )
    z0 = torch.where(mask.unsqueeze(-1), z0, torch.zeros_like(z0))
    return z0, mask, scale.squeeze(-1) if scale.dim() > area_target.dim() else scale


def z_to_rectangles(
    z: torch.Tensor,
    area_target: torch.Tensor,
    target_positions: Optional[torch.Tensor] = None,
    constraints: Optional[torch.Tensor] = None,
    clamp_log_aspect: Tuple[float, float] = (LOG_ASPECT_MIN, LOG_ASPECT_MAX),
    z_repr: str = "xylogwh",
) -> torch.Tensor:
    """Convert z to rectangles ``[..., N, 4]`` in ``[x,y,w,h]``.

    4D z is interpreted as ``[x/S, y/S, log(w/S), log(h/S)]``.  3D z is kept
    for compatibility with older checkpoints and is interpreted as
    ``[cx/S, cy/S, log(w/h)]`` with exact target area.
    """
    mask = valid_block_mask(area_target)
    scale = layout_scale(area_target, mask)
    while scale.dim() < z.dim() - 1:
        scale = scale.unsqueeze(-1)
    if z.shape[-1] >= 4:
        x = z[..., 0] * scale
        y = z[..., 1] * scale
        areas = torch.where(mask, area_target.clamp_min(1e-6), torch.ones_like(area_target))
        if z_repr == "xyaspect":
            log_aspect = z[..., 2].clamp(*clamp_log_aspect)
        else:
            log_aspect = (z[..., 2] - z[..., 3]).clamp(*clamp_log_aspect)
        aspect = torch.exp(log_aspect)
        w = torch.sqrt(areas * aspect).clamp_min(1e-6)
        h = torch.sqrt(areas / aspect).clamp_min(1e-6)
        rects = torch.stack([x, y, w.clamp_min(1e-6), h.clamp_min(1e-6)], dim=-1)
    else:
        areas = torch.where(mask, area_target.clamp_min(1e-6), torch.ones_like(area_target))
        log_aspect = z[..., 2].clamp(*clamp_log_aspect)
        aspect = torch.exp(log_aspect)
        w = torch.sqrt(areas * aspect).clamp_min(1e-6)
        h = torch.sqrt(areas / aspect).clamp_min(1e-6)
        cx = z[..., 0] * scale
        cy = z[..., 1] * scale
        rects = torch.stack([cx - 0.5 * w, cy - 0.5 * h, w, h], dim=-1)

    if target_positions is not None and constraints is not None:
        c = constraints
        fixed = c[..., 0] != 0 if c.shape[-1] > 0 else torch.zeros_like(mask)
        preplaced = c[..., 1] != 0 if c.shape[-1] > 1 else torch.zeros_like(mask)
        tp_valid_wh = (target_positions[..., 2] > 0) & (target_positions[..., 3] > 0)
        wh_mask = (fixed | preplaced) & tp_valid_wh
        rects = torch.where(
            wh_mask.unsqueeze(-1),
            torch.stack([rects[..., 0], rects[..., 1], target_positions[..., 2], target_positions[..., 3]], dim=-1),
            rects,
        )
        xy_mask = preplaced & (target_positions[..., 0] >= 0) & (target_positions[..., 1] >= 0)
        rects = torch.where(
            xy_mask.unsqueeze(-1),
            torch.stack([target_positions[..., 0], target_positions[..., 1], rects[..., 2], rects[..., 3]], dim=-1),
            rects,
        )
    return torch.where(mask.unsqueeze(-1), rects, torch.zeros_like(rects))


def _dense_adj(area_target: torch.Tensor, b2b_connectivity: torch.Tensor) -> torch.Tensor:
    bsz, n = area_target.shape
    adj = area_target.new_zeros((bsz, n, n))
    for b in range(bsz):
        for edge in b2b_connectivity[b]:
            if edge[0] == -1:
                continue
            i, j = int(edge[0].item()), int(edge[1].item())
            if 0 <= i < n and 0 <= j < n:
                w = float(edge[2].item())
                if math.isfinite(w) and w > 0:
                    adj[b, i, j] += w
                    adj[b, j, i] += w
    return adj


def normalized_adjacency(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Symmetric dense adjacency normalization with self loops."""
    eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device).unsqueeze(0)
    adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2) + eye * mask.unsqueeze(1)
    deg = adj.sum(dim=-1).clamp_min(1e-6)
    deg_inv_sqrt = deg.rsqrt()
    return adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)


def build_relation_features(
    area_target: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    constraints: torch.Tensor,
    relation_feat_dim: int = 8,
    adj_raw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build pairwise relation features ``[B, N, N, relation_feat_dim]``.

    These features keep constraints as pair information instead of compressing
    every group id into one node scalar.  The model can then distinguish
    "connected by net" from "same cluster" or "same MIB".
    """
    if relation_feat_dim <= 0:
        bsz, n = area_target.shape
        return area_target.new_zeros((bsz, n, n, 0))

    bsz, n = area_target.shape
    dtype = area_target.dtype
    device = area_target.device
    mask = valid_block_mask(area_target)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    eye = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
    nonself = pair_mask & ~eye

    if adj_raw is None:
        adj_raw = _dense_adj(area_target, b2b_connectivity)
    max_w = adj_raw.amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    b2b_weight = torch.log1p(adj_raw) / torch.log1p(max_w)
    b2b_binary = (adj_raw > 0).to(dtype)

    c = constraints
    zeros_i = torch.zeros((bsz, n), dtype=torch.long, device=device)
    mib = c[..., 2].long().clamp_min(0) if c.shape[-1] > 2 else zeros_i
    cluster = c[..., 3].long().clamp_min(0) if c.shape[-1] > 3 else zeros_i
    boundary = c[..., 4].long().clamp_min(0) if c.shape[-1] > 4 else zeros_i
    fixed = c[..., 0].to(dtype) if c.shape[-1] > 0 else torch.zeros((bsz, n), dtype=dtype, device=device)
    preplaced = c[..., 1].to(dtype) if c.shape[-1] > 1 else torch.zeros((bsz, n), dtype=dtype, device=device)

    same_mib = ((mib.unsqueeze(2) == mib.unsqueeze(1)) & (mib.unsqueeze(2) > 0) & nonself).to(dtype)
    same_cluster = (
        (cluster.unsqueeze(2) == cluster.unsqueeze(1)) & (cluster.unsqueeze(2) > 0) & nonself
    ).to(dtype)
    shared_boundary = (
        ((boundary.unsqueeze(2) & boundary.unsqueeze(1)) != 0)
        & (boundary.unsqueeze(2) > 0)
        & (boundary.unsqueeze(1) > 0)
        & nonself
    ).to(dtype)
    any_boundary = (((boundary.unsqueeze(2) > 0) | (boundary.unsqueeze(1) > 0)) & nonself).to(dtype)
    locked_pair = (((fixed + preplaced).unsqueeze(2) > 0) | ((fixed + preplaced).unsqueeze(1) > 0)).to(dtype)

    safe_area = torch.where(mask, area_target.clamp_min(1e-6), torch.ones_like(area_target))
    log_area = torch.log(safe_area)
    area_similarity = torch.exp(-(log_area.unsqueeze(2) - log_area.unsqueeze(1)).abs()).to(dtype)

    feats = [
        nonself.to(dtype),
        b2b_binary,
        b2b_weight,
        same_cluster,
        same_mib,
        shared_boundary,
        any_boundary,
        locked_pair * nonself.to(dtype),
        area_similarity * nonself.to(dtype),
    ]
    rel = torch.stack(feats[:relation_feat_dim], dim=-1)
    if relation_feat_dim > len(feats):
        pad = area_target.new_zeros((bsz, n, n, relation_feat_dim - len(feats)))
        rel = torch.cat([rel, pad], dim=-1)
    return rel * nonself.unsqueeze(-1)


def build_condition(
    area_target: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor] = None,
    relation_feat_dim: int = 0,
    node_feat_dim: int = 13,
) -> Dict[str, torch.Tensor]:
    """Build node features, normalized dense adjacency, and mask from padded tensors."""
    if area_target.dim() == 1:
        area_target = area_target.unsqueeze(0)
        b2b_connectivity = b2b_connectivity.unsqueeze(0)
        p2b_connectivity = p2b_connectivity.unsqueeze(0)
        pins_pos = pins_pos.unsqueeze(0)
        constraints = constraints.unsqueeze(0)
        if target_positions is not None:
            target_positions = target_positions.unsqueeze(0)

    device = area_target.device
    dtype = area_target.dtype
    bsz, n = area_target.shape
    mask = valid_block_mask(area_target)
    scale = layout_scale(area_target, mask).clamp_min(1.0)
    safe_area = torch.where(mask, area_target.clamp_min(1e-6), torch.ones_like(area_target))

    adj_raw = _dense_adj(area_target, b2b_connectivity)
    adj = normalized_adjacency(adj_raw, mask)
    b2b_degree = (adj_raw > 0).sum(dim=-1).to(dtype)
    b2b_weight = adj_raw.sum(dim=-1)

    p2b_degree = torch.zeros((bsz, n), dtype=dtype, device=device)
    p2b_weight = torch.zeros((bsz, n), dtype=dtype, device=device)
    pin_x_sum = torch.zeros((bsz, n), dtype=dtype, device=device)
    pin_y_sum = torch.zeros((bsz, n), dtype=dtype, device=device)
    for b in range(bsz):
        for edge in p2b_connectivity[b]:
            if edge[0] == -1:
                continue
            p, block = int(edge[0].item()), int(edge[1].item())
            if 0 <= block < n and 0 <= p < pins_pos.shape[1] and pins_pos[b, p, 0] != -1:
                w = max(float(edge[2].item()), 0.0)
                p2b_degree[b, block] += 1.0
                p2b_weight[b, block] += w
                pin_x_sum[b, block] += w * pins_pos[b, p, 0]
                pin_y_sum[b, block] += w * pins_pos[b, p, 1]
    pin_denom = p2b_weight.clamp_min(1e-6)
    pin_x = pin_x_sum / pin_denom / scale.unsqueeze(-1)
    pin_y = pin_y_sum / pin_denom / scale.unsqueeze(-1)

    c = constraints
    zeros = torch.zeros((bsz, n), dtype=dtype, device=device)
    fixed = c[..., 0].to(dtype) if c.shape[-1] > 0 else zeros
    preplaced = c[..., 1].to(dtype) if c.shape[-1] > 1 else zeros
    mib = (c[..., 2].clamp_min(0) / 32.0).to(dtype) if c.shape[-1] > 2 else zeros
    cluster = (c[..., 3].clamp_min(0) / 32.0).to(dtype) if c.shape[-1] > 3 else zeros
    boundary_code = c[..., 4].long().clamp_min(0) if c.shape[-1] > 4 else torch.zeros((bsz, n), dtype=torch.long, device=device)
    boundary = (boundary_code.to(dtype) / 15.0).to(dtype)
    boundary_l = ((boundary_code & 1) != 0).to(dtype)
    boundary_r = ((boundary_code & 2) != 0).to(dtype)
    boundary_t = ((boundary_code & 4) != 0).to(dtype)
    boundary_b = ((boundary_code & 8) != 0).to(dtype)
    boundary_count = (boundary_l + boundary_r + boundary_t + boundary_b) / 4.0
    boundary_corner = (boundary_count > 0.25).to(dtype)
    if target_positions is None:
        target_positions = torch.full((bsz, n, 4), -1.0, dtype=dtype, device=device)
    else:
        target_positions = target_positions.to(device=device, dtype=dtype)
    has_target_wh = ((target_positions[..., 2] > 0) & (target_positions[..., 3] > 0) & mask).to(dtype)
    has_target_xy = ((target_positions[..., 0] >= 0) & (target_positions[..., 1] >= 0) & mask).to(dtype)
    target_w = torch.where(has_target_wh.bool(), target_positions[..., 2] / scale.unsqueeze(-1), zeros)
    target_h = torch.where(has_target_wh.bool(), target_positions[..., 3] / scale.unsqueeze(-1), zeros)
    target_x = torch.where(has_target_xy.bool(), target_positions[..., 0] / scale.unsqueeze(-1), zeros)
    target_y = torch.where(has_target_xy.bool(), target_positions[..., 1] / scale.unsqueeze(-1), zeros)
    target_log_aspect = torch.where(
        has_target_wh.bool(),
        torch.log((target_positions[..., 2] / target_positions[..., 3]).clamp_min(1e-6)).clamp(-3.0, 3.0) / 3.0,
        zeros,
    )

    features = [
        torch.log(safe_area) / 10.0,
        torch.sqrt(safe_area) / scale.unsqueeze(-1),
        fixed,
        preplaced,
        mib,
        cluster,
        boundary,
        torch.log1p(b2b_degree) / 5.0,
        torch.log1p(b2b_weight) / 10.0,
        torch.log1p(p2b_degree) / 5.0,
        torch.log1p(p2b_weight) / 10.0,
        pin_x,
        pin_y,
        boundary_l,
        boundary_r,
        boundary_t,
        boundary_b,
        boundary_count,
        boundary_corner,
        has_target_wh,
        has_target_xy,
        target_w,
        target_h,
        target_x,
        target_y,
        target_log_aspect,
    ]
    if node_feat_dim < len(features):
        features = features[:node_feat_dim]
    feat = torch.stack(features, dim=-1)
    if node_feat_dim > feat.shape[-1]:
        pad = torch.zeros((*feat.shape[:-1], node_feat_dim - feat.shape[-1]), dtype=dtype, device=device)
        feat = torch.cat([feat, pad], dim=-1)
    feat = torch.where(mask.unsqueeze(-1), feat, torch.zeros_like(feat))
    rel_feat = build_relation_features(area_target, b2b_connectivity, constraints, relation_feat_dim, adj_raw=adj_raw)
    return {"node_feat": feat, "adj": adj, "mask": mask, "scale": scale, "rel_feat": rel_feat}


def batch_to_device(batch, device: torch.device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]
