from __future__ import annotations

import torch
from torch import nn


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, edge_type_count: int = 1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.edge_bias = nn.Linear(1, num_heads, bias=False)
        self.edge_type_bias = nn.Embedding(max(1, edge_type_count), num_heads)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = None
        if edge_index.numel() > 0:
            n = h.shape[0]
            src = edge_index[0]
            dst = edge_index[1]
            edge_bias = self.edge_bias(edge_attr).transpose(0, 1)
            if edge_type is not None and edge_type.numel() == edge_attr.shape[0]:
                type_bias = self.edge_type_bias(
                    edge_type.to(device=h.device).clamp(0, self.edge_type_bias.num_embeddings - 1)
                ).transpose(0, 1)
                edge_bias = edge_bias + type_bias
            attn_mask = h.new_zeros((self.num_heads, n, n))
            attn_mask[:, dst, src] = edge_bias
        attn_out, _weights = self.attn(
            h.unsqueeze(0),
            h.unsqueeze(0),
            h.unsqueeze(0),
            attn_mask=attn_mask,
            need_weights=False,
        )
        h = self.attn_norm(h + self.dropout(attn_out.squeeze(0)))
        h = self.ff_norm(h + self.dropout(self.ff(h)))
        return h


class FloorplanGNN(nn.Module):
    """Anchor-GNN v2 compatible with the arch_new trained checkpoints."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 160,
        num_layers: int = 5,
        dropout: float = 0.05,
        encoder_type: str = "mpnn",
        num_heads: int = 4,
        structural_feat_dim: int = 0,
        edge_type_count: int = 1,
    ):
        super().__init__()
        if encoder_type not in {"mpnn", "graph-transformer"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.encoder_type = encoder_type
        self.num_heads = num_heads
        self.structural_feat_dim = structural_feat_dim
        self.edge_type_count = max(1, edge_type_count)

        self.node_in = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        if encoder_type == "mpnn":
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
        else:
            self.structural_in = (
                nn.Sequential(
                    nn.Linear(structural_feat_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
                if structural_feat_dim > 0
                else None
            )
            self.transformer_layers = nn.ModuleList(
                GraphTransformerLayer(hidden_dim, num_heads, dropout, self.edge_type_count)
                for _ in range(num_layers)
            )

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
        pair_in = hidden_dim * 4
        self.pair_head = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor | None = None,
        structural_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.node_in(node_feat)
        if self.encoder_type == "graph-transformer":
            if self.structural_in is not None:
                if structural_feat is None:
                    structural_feat = h.new_zeros((h.shape[0], self.structural_feat_dim))
                h = h + self.structural_in(structural_feat.to(device=h.device, dtype=h.dtype))
            for layer in self.transformer_layers:
                h = layer(h, edge_index, edge_attr, edge_type=edge_type)
            return h
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

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor | None = None,
        structural_feat: torch.Tensor | None = None,
        pairs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encode(
            node_feat,
            edge_index,
            edge_attr,
            edge_type=edge_type,
            structural_feat=structural_feat,
        )
        g = h.mean(dim=0, keepdim=True).expand_as(h)
        z = torch.cat([h, g, node_feat], dim=1)
        output = {
            "anchor": self.anchor_head(z),
            "priority": self.priority_head(z).squeeze(-1),
            "log_aspect": self.aspect_head(z).squeeze(-1).clamp(-2.5, 2.5),
        }
        if pairs is not None:
            if pairs.numel() == 0:
                output["pair_logits"] = h.new_empty((0, 2))
            else:
                src = h[pairs[:, 0]]
                dst = h[pairs[:, 1]]
                output["pair_logits"] = self.pair_head(torch.cat([src, dst, dst - src, (dst - src).abs()], dim=1))
        return output
