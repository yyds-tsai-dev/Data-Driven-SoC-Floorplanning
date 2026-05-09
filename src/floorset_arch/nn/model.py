from __future__ import annotations

import torch
from torch import nn



class FloorplanGNN(nn.Module):
    """Anchor-GNN v2 compatible with the arch_new trained checkpoints."""

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

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(node_feat, edge_index, edge_attr)
        g = h.mean(dim=0, keepdim=True).expand_as(h)
        z = torch.cat([h, g, node_feat], dim=1)
        return {
            "anchor": self.anchor_head(z),
            "priority": self.priority_head(z).squeeze(-1),
            "log_aspect": self.aspect_head(z).squeeze(-1).clamp(-2.5, 2.5),
        }
