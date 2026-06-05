from __future__ import annotations

import math

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


def _relation_key(relation: tuple[str, str, str]) -> str:
    return "__".join(relation)


def _edge_softmax_by_dst(scores: torch.Tensor, dst: torch.Tensor, dst_count: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    if dst_count <= 0:
        return torch.zeros_like(scores)
    valid = (dst >= 0) & (dst < dst_count)
    if not valid.any():
        return torch.zeros_like(scores)
    alpha = torch.zeros_like(scores)
    safe_dst = dst[valid].to(device=scores.device)
    valid_scores = scores[valid]
    expanded_dst = safe_dst[:, None].expand(-1, scores.shape[1])
    max_per_dst = scores.new_full((dst_count, scores.shape[1]), -torch.inf)
    max_per_dst.scatter_reduce_(
        0, expanded_dst, valid_scores, reduce="amax", include_self=True
    )
    exp_scores = torch.exp(valid_scores - max_per_dst[safe_dst])
    denom = scores.new_zeros((dst_count, scores.shape[1]))
    denom.scatter_add_(0, expanded_dst, exp_scores)
    alpha[valid] = exp_scores / denom[safe_dst].clamp_min(1e-12)
    return alpha


class HGTLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        node_types: tuple[str, ...],
        relation_specs: tuple[tuple[str, str, str], ...],
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.node_types = tuple(node_types)
        self.relation_specs = tuple(relation_specs)
        self.query = nn.ModuleDict(
            {node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in self.node_types}
        )
        self.rel_key = nn.ModuleDict(
            {_relation_key(relation): nn.Linear(hidden_dim, hidden_dim) for relation in self.relation_specs}
        )
        self.rel_value = nn.ModuleDict(
            {_relation_key(relation): nn.Linear(hidden_dim, hidden_dim) for relation in self.relation_specs}
        )
        self.rel_edge_bias = nn.ModuleDict(
            {_relation_key(relation): nn.Linear(1, num_heads, bias=False) for relation in self.relation_specs}
        )
        self.rel_gate = nn.ParameterDict(
            {
                _relation_key(relation): nn.Parameter(torch.ones(1))
                for relation in self.relation_specs
            }
        )
        self.attn_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.ff = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for node_type in self.node_types
            }
        )
        self.ff_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: dict[str, torch.Tensor],
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        edge_attr: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        messages = {
            node_type: states[node_type].new_zeros(states[node_type].shape)
            for node_type in self.node_types
        }
        for relation in self.relation_specs:
            if relation not in edge_index:
                continue
            src_type, _edge_type, dst_type = relation
            edges = edge_index[relation]
            if edges.numel() == 0:
                continue
            src = edges[0].to(device=states[src_type].device)
            dst = edges[1].to(device=states[dst_type].device)
            key = _relation_key(relation)
            src_h = states[src_type][src]
            dst_h = states[dst_type][dst]
            q = self.query[dst_type](dst_h).view(-1, self.num_heads, self.head_dim)
            k = self.rel_key[key](src_h).view(-1, self.num_heads, self.head_dim)
            v = self.rel_value[key](src_h).view(-1, self.num_heads, self.head_dim)
            scores = (q * k).sum(dim=-1) / math.sqrt(max(float(self.head_dim), 1.0))
            attr = edge_attr.get(relation)
            if attr is None:
                attr = states[src_type].new_ones((src.numel(), 1))
            attr = attr.to(device=states[src_type].device, dtype=states[src_type].dtype)
            scores = scores + self.rel_edge_bias[key](attr)
            alpha = _edge_softmax_by_dst(scores, dst, states[dst_type].shape[0])
            msg = (alpha.unsqueeze(-1) * v).reshape(-1, self.hidden_dim)
            msg = msg * self.rel_gate[key].to(device=msg.device, dtype=msg.dtype)
            messages[dst_type].index_add_(0, dst, msg)

        updated: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            h = self.attn_norm[node_type](states[node_type] + self.dropout(messages[node_type]))
            h = self.ff_norm[node_type](h + self.dropout(self.ff[node_type](h)))
            updated[node_type] = h
        return updated


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
        hgt_node_feat_dims: dict[str, int] | None = None,
        hgt_relation_specs: tuple[tuple[str, str, str], ...] | list[tuple[str, str, str]] | None = None,
    ):
        super().__init__()
        if encoder_type not in {"mpnn", "graph-transformer", "hgt"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.encoder_type = encoder_type
        self.num_heads = num_heads
        self.structural_feat_dim = structural_feat_dim
        self.edge_type_count = max(1, edge_type_count)
        self.hgt_node_feat_dims = dict(hgt_node_feat_dims or {"block": node_feat_dim})
        self.hgt_relation_specs = tuple(tuple(relation) for relation in (hgt_relation_specs or ()))

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
        elif encoder_type == "graph-transformer":
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
        else:
            self.hgt_node_in = nn.ModuleDict(
                {
                    node_type: nn.Sequential(
                        nn.Linear(feat_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                    )
                    for node_type, feat_dim in sorted(self.hgt_node_feat_dims.items())
                }
            )
            self.hgt_layers = nn.ModuleList(
                HGTLayer(
                    hidden_dim,
                    num_heads,
                    dropout,
                    tuple(sorted(self.hgt_node_feat_dims)),
                    self.hgt_relation_specs,
                )
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
        hgt_node_features: dict[str, torch.Tensor] | None = None,
        hgt_edge_index: dict[tuple[str, str, str], torch.Tensor] | None = None,
        hgt_edge_attr: dict[tuple[str, str, str], torch.Tensor] | None = None,
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
        if self.encoder_type == "hgt":
            if hgt_node_features is None:
                hgt_node_features = {"block": node_feat}
            states: dict[str, torch.Tensor] = {}
            for node_type, projector in self.hgt_node_in.items():
                features = hgt_node_features.get(node_type)
                if features is None:
                    features = node_feat.new_empty((0, self.hgt_node_feat_dims[node_type]))
                states[node_type] = projector(features.to(device=node_feat.device, dtype=node_feat.dtype))
            for layer in self.hgt_layers:
                states = layer(states, hgt_edge_index or {}, hgt_edge_attr or {})
            return states["block"]
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

    def hgt_relation_gate_values(self) -> list[dict[str, float]]:
        if self.encoder_type != "hgt":
            return []
        values = []
        for layer in self.hgt_layers:
            values.append(
                {
                    key: float(param.detach().cpu().item())
                    for key, param in layer.rel_gate.items()
                }
            )
        return values

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor | None = None,
        structural_feat: torch.Tensor | None = None,
        hgt_node_features: dict[str, torch.Tensor] | None = None,
        hgt_edge_index: dict[tuple[str, str, str], torch.Tensor] | None = None,
        hgt_edge_attr: dict[tuple[str, str, str], torch.Tensor] | None = None,
        pairs: torch.Tensor | None = None,
        block_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encode(
            node_feat,
            edge_index,
            edge_attr,
            edge_type=edge_type,
            structural_feat=structural_feat,
            hgt_node_features=hgt_node_features,
            hgt_edge_index=hgt_edge_index,
            hgt_edge_attr=hgt_edge_attr,
        )
        if block_batch is None:
            g = h.mean(dim=0, keepdim=True).expand_as(h)
        else:
            batch = block_batch.to(device=h.device, dtype=torch.long)
            graph_count = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            graph_sum = h.new_zeros((graph_count, h.shape[1]))
            graph_sum.index_add_(0, batch, h)
            graph_count_t = h.new_zeros((graph_count, 1))
            graph_count_t.index_add_(
                0,
                batch,
                torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device),
            )
            g = graph_sum[batch] / graph_count_t[batch].clamp_min(1.0)
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
