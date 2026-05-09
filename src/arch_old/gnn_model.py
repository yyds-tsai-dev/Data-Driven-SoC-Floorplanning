#!/usr/bin/env python3
"""
Anchor-GNN v2 for ICCAD 2026 FloorSet.

This file is intentionally compatible with the existing guarded relative-order
optimizer:
  from gnn_model import FloorplanGNN, build_node_features, build_edge_tensors

The optimizer expects model(node_feat, edge_index, edge_attr) to return:
  {"anchor": [N,2], "priority": [N], "log_aspect": [N]}

Main idea:
  - keep GNN as a soft anchor predictor, not a hard topology controller
  - train anchors strongly
  - add order-aware auxiliary loss in train_gnn_anchor_v2.py
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def valid_block_count(area_targets: torch.Tensor) -> int:
    flat = area_targets.detach().flatten()
    return int((flat != -1).sum().item())


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def build_node_features(
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    block_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Build per-block features used by both training and inference."""

    n = block_count
    area = area_targets[:n].float().to(device)
    area = torch.clamp(area, min=1.0)
    sqrt_area = torch.sqrt(area)
    total_area = torch.clamp(area.sum(), min=1.0)
    scale = float(torch.sqrt(total_area).item())

    degree = torch.zeros(n, device=device)
    pin_degree = torch.zeros(n, device=device)
    pin_cx = torch.zeros(n, device=device)
    pin_cy = torch.zeros(n, device=device)
    pin_wsum = torch.zeros(n, device=device)

    if b2b_connectivity is not None:
        for edge in b2b_connectivity:
            if _safe_float(edge[0]) < 0.0:
                continue
            i = int(edge[0])
            j = int(edge[1])
            w = _safe_float(edge[2])
            if 0 <= i < n and 0 <= j < n:
                degree[i] += w
                degree[j] += w

    if p2b_connectivity is not None:
        for edge in p2b_connectivity:
            if _safe_float(edge[0]) < 0.0:
                continue
            pin_idx = int(edge[0])
            block_idx = int(edge[1])
            w = _safe_float(edge[2])
            if 0 <= block_idx < n and 0 <= pin_idx < pins_pos.shape[0]:
                px = _safe_float(pins_pos[pin_idx, 0])
                py = _safe_float(pins_pos[pin_idx, 1])
                if px != -1.0 and py != -1.0:
                    pin_degree[block_idx] += w
                    pin_cx[block_idx] += w * px
                    pin_cy[block_idx] += w * py
                    pin_wsum[block_idx] += w

    has_pin = pin_wsum > 0.0
    pin_cx = torch.where(has_pin, pin_cx / pin_wsum.clamp_min(1e-6), torch.zeros_like(pin_cx))
    pin_cy = torch.where(has_pin, pin_cy / pin_wsum.clamp_min(1e-6), torch.zeros_like(pin_cy))

    max_degree = torch.clamp(degree.max(), min=1.0)
    max_pin_degree = torch.clamp(pin_degree.max(), min=1.0)

    fixed = torch.zeros(n, device=device)
    preplaced = torch.zeros(n, device=device)
    mib_flag = torch.zeros(n, device=device)
    cluster_flag = torch.zeros(n, device=device)
    left_b = torch.zeros(n, device=device)
    right_b = torch.zeros(n, device=device)
    top_b = torch.zeros(n, device=device)
    bottom_b = torch.zeros(n, device=device)

    mib_norm = torch.zeros(n, device=device)
    cluster_norm = torch.zeros(n, device=device)

    if constraints is not None and constraints.dim() > 1:
        c = constraints[:n].to(device)
        if c.shape[1] > 0:
            fixed = (c[:, 0] != 0).float()
        if c.shape[1] > 1:
            preplaced = (c[:, 1] != 0).float()
        if c.shape[1] > 2:
            mib_id = c[:, 2].float()
            mib_flag = (mib_id != 0).float()
            max_mib = torch.clamp(mib_id.abs().max(), min=1.0)
            mib_norm = mib_id / max_mib
        if c.shape[1] > 3:
            cluster_id = c[:, 3].float()
            cluster_flag = (cluster_id != 0).float()
            max_cluster = torch.clamp(cluster_id.abs().max(), min=1.0)
            cluster_norm = cluster_id / max_cluster
        if c.shape[1] > 4:
            bc = c[:, 4].long()
            left_b = ((bc & 1) != 0).float()
            right_b = ((bc & 2) != 0).float()
            top_b = ((bc & 4) != 0).float()
            bottom_b = ((bc & 8) != 0).float()

    # All features are roughly normalized. Avoid raw large coordinates except
    # normalized pin centroid.
    feat = torch.stack(
        [
            sqrt_area / max(scale, 1.0),
            torch.log1p(area) / 10.0,
            degree / max_degree,
            torch.log1p(degree) / 10.0,
            pin_degree / max_pin_degree,
            has_pin.float(),
            pin_cx / max(scale, 1.0),
            pin_cy / max(scale, 1.0),
            fixed,
            preplaced,
            mib_flag,
            cluster_flag,
            left_b,
            right_b,
            top_b,
            bottom_b,
            mib_norm,
            cluster_norm,
        ],
        dim=1,
    )

    return feat.float(), scale


def build_edge_tensors(
    block_count: int,
    b2b_connectivity: torch.Tensor,
    device: torch.device,
):
    """Return directed edge_index [2,E] and edge_attr [E,1]."""

    n = block_count
    src = []
    dst = []
    weights = []

    if b2b_connectivity is not None:
        for edge in b2b_connectivity:
            if _safe_float(edge[0]) < 0.0:
                continue
            i = int(edge[0])
            j = int(edge[1])
            w = max(_safe_float(edge[2]), 0.0)
            if 0 <= i < n and 0 <= j < n and i != j:
                ww = math.log1p(w)
                src.extend([i, j])
                dst.extend([j, i])
                weights.extend([ww, ww])

    if not src:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        return edge_index, edge_attr

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
    if edge_attr.numel() > 0:
        edge_attr = edge_attr / edge_attr.max().clamp_min(1.0)
    return edge_index, edge_attr


def build_targets(fp_sol: torch.Tensor, block_count: int, scale: float, device: torch.device):
    """Compatibility helper for older train scripts."""
    gt = fp_sol[:block_count].float().to(device)
    w = gt[:, 0]
    h = gt[:, 1]
    x = gt[:, 2]
    y = gt[:, 3]
    center = torch.stack([x + w / 2.0, y + h / 2.0], dim=1) / max(float(scale), 1.0)
    log_aspect = torch.log(torch.clamp(w, min=1e-6) / torch.clamp(h, min=1e-6)).clamp(-2.0, 2.0)
    return {"anchor": center, "log_aspect": log_aspect}


class FloorplanGNN(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int = 160, num_layers: int = 5, dropout: float = 0.05):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.node_in = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.upd_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Global context lets every node know rough design-level statistics.
        head_in = hidden_dim * 2 + node_feat_dim
        self.anchor_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.priority_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.aspect_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_in(node_feat)
        n = h.shape[0]

        if edge_index.numel() == 0:
            return h

        src = edge_index[0]
        dst = edge_index[1]

        for msg_mlp, upd_mlp, norm in zip(self.msg_mlps, self.upd_mlps, self.norms):
            msg_in = torch.cat([h[src], edge_attr], dim=1)
            msg = msg_mlp(msg_in)

            aggr = torch.zeros(n, self.hidden_dim, dtype=h.dtype, device=h.device)
            aggr.index_add_(0, dst, msg)

            deg = torch.zeros(n, 1, dtype=h.dtype, device=h.device)
            deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=h.dtype, device=h.device))
            aggr = aggr / deg.clamp_min(1.0)

            upd = upd_mlp(torch.cat([h, aggr], dim=1))
            h = norm(h + upd)

        return h

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        h = self.encode(node_feat, edge_index, edge_attr)
        g = h.mean(dim=0, keepdim=True).expand_as(h)
        z = torch.cat([h, g, node_feat], dim=1)

        anchor = self.anchor_head(z)
        priority = self.priority_head(z).squeeze(-1)
        log_aspect = self.aspect_head(z).squeeze(-1).clamp(-2.5, 2.5)

        return {"anchor": anchor, "priority": priority, "log_aspect": log_aspect}
