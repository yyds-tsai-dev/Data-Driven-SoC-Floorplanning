from __future__ import annotations

import torch
from torch import nn

from floorset_arch.features import ModelInputs
from floorset_arch.models import ModelPrediction


class MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(h)
        if edge_index.numel() > 0:
            src, dst = edge_index
            msg = self.msg_proj(h[src]) * edge_weight.unsqueeze(1)
            agg.index_add_(0, dst, msg)
        return self.norm(torch.relu(self.self_proj(h) + agg))


class SimpleGraphFloorplanner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 3):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(MessageLayer(hidden_dim) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: ModelInputs) -> ModelPrediction:
        h = self.input(inputs.block_features)
        for layer in self.layers:
            h = layer(h, inputs.edge_index, inputs.edge_weight)
        raw = self.head(h)
        centers = torch.sigmoid(raw[:, :2])
        log_aspect = torch.clamp(raw[:, 2], min=-3.0, max=3.0)
        priority = self.priority_head(h).squeeze(-1)
        return ModelPrediction(centers=centers, log_aspect=log_aspect, priority=priority, order_logits=priority)
