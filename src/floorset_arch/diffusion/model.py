from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from floorset_arch.diffusion.contracts import DiffusionGraphInputs, Relation


def _time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = timesteps.float().view(-1, 1) * freq.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


def _relation_key(relation: Relation) -> str:
    return "__".join(relation)


class HGTLiteConditioner(nn.Module):
    def __init__(
        self,
        node_feat_dims: dict[str, int],
        relation_specs: Sequence[Relation],
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.node_types = tuple(node_feat_dims)
        self.relation_specs = tuple(relation_specs)
        self.input = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
                for node_type, dim in node_feat_dims.items()
            }
        )
        self.relation_msg = nn.ModuleDict(
            {
                _relation_key(relation): nn.Linear(hidden_dim + 1, hidden_dim)
                for relation in relation_specs
            }
        )
        self.updates = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU(),
                        )
                        for node_type in self.node_types
                    }
                )
                for _ in range(max(1, layers))
            ]
        )

    def forward(self, graph: DiffusionGraphInputs) -> torch.Tensor:
        states = {
            node_type: self.input[node_type](features.float())
            for node_type, features in graph.node_features.items()
            if node_type in self.input
        }
        for update in self.updates:
            messages = {
                node_type: torch.zeros_like(state) for node_type, state in states.items()
            }
            for relation in self.relation_specs:
                src_type, _edge_type, dst_type = relation
                if src_type not in states or dst_type not in states:
                    continue
                edges = graph.edge_index[relation]
                if edges.numel() == 0:
                    continue
                src = edges[0].to(states[src_type].device)
                dst = edges[1].to(states[dst_type].device)
                attr = graph.edge_attr[relation].to(states[src_type].device)
                msg = self.relation_msg[_relation_key(relation)](
                    torch.cat([states[src_type][src], attr], dim=1)
                )
                messages[dst_type].index_add_(0, dst, msg)
            states = {
                node_type: states[node_type] + update[node_type](messages[node_type])
                for node_type in states
            }
        return states["block"]


class GraphConditionedPlacementDiffusion(nn.Module):
    def __init__(
        self,
        raw_block_dim: int,
        raw_pair_dim: int,
        global_dim: int,
        node_feat_dims: dict[str, int],
        relation_specs: Sequence[Relation],
        variant: str = "raw",
        hidden_dim: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if variant not in {"raw", "hgt_lite"}:
            raise ValueError("variant must be raw or hgt_lite")
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.block_in = nn.Linear(raw_block_dim + 3 + hidden_dim + hidden_dim, hidden_dim)
        self.global_in = nn.Linear(global_dim, hidden_dim)
        self.time_dim = hidden_dim
        self.hgt: HGTLiteConditioner | None = None
        if variant == "hgt_lite":
            self.hgt = HGTLiteConditioner(
                node_feat_dims, relation_specs, hidden_dim, layers
            )
        self.block_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + raw_pair_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_graph_inputs(
        cls,
        graph: DiffusionGraphInputs,
        variant: str,
        hidden_dim: int = 128,
        layers: int = 2,
    ) -> GraphConditionedPlacementDiffusion:
        return cls(
            raw_block_dim=int(graph.raw_block_features.shape[1]),
            raw_pair_dim=int(graph.raw_pair_features.shape[1]),
            global_dim=int(graph.global_features.numel()),
            node_feat_dims={
                key: int(value.shape[1]) for key, value in graph.node_features.items()
            },
            relation_specs=graph.relation_specs,
            variant=variant,
            hidden_dim=hidden_dim,
            layers=layers,
        )

    def _block_context(self, graph: DiffusionGraphInputs) -> torch.Tensor:
        if self.hgt is None:
            return graph.raw_block_features.new_zeros(
                (graph.raw_block_features.shape[0], self.hidden_dim)
            )
        return self.hgt(graph)

    def forward(
        self,
        graph: DiffusionGraphInputs,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        samples, blocks, _dims = x_t.shape
        block_context = self._block_context(graph)
        global_context = self.global_in(graph.global_features.float()).view(1, 1, -1)
        global_context = global_context.expand(samples, blocks, -1)
        time_context = _time_embedding(timesteps.to(x_t.device), self.time_dim)
        time_context = time_context.view(samples, 1, -1).expand(samples, blocks, -1)
        raw_block = graph.raw_block_features.float().to(x_t.device)
        raw_block = raw_block.view(1, blocks, -1).expand(samples, blocks, -1)
        hgt_block = block_context.to(x_t.device).view(1, blocks, -1)
        hgt_block = hgt_block.expand(samples, blocks, -1)
        h = self.block_in(
            torch.cat(
                [raw_block, x_t.float(), hgt_block, time_context + global_context],
                dim=2,
            )
        )
        eps_pred = self.block_mlp(h)

        pair_index = graph.pair_index.to(x_t.device)
        hi = h[:, pair_index[:, 0], :]
        hj = h[:, pair_index[:, 1], :]
        xi = x_t[:, pair_index[:, 0], :2]
        xj = x_t[:, pair_index[:, 1], :2]
        delta = xj - xi
        distance = torch.linalg.norm(delta, dim=2, keepdim=True)
        raw_pair = graph.raw_pair_features.float().to(x_t.device)
        raw_pair = raw_pair.view(1, pair_index.shape[0], -1).expand(samples, -1, -1)
        pair_features = torch.cat(
            [
                hi,
                hj,
                raw_pair,
                delta,
                distance,
                distance.clamp_min(1e-6).reciprocal(),
            ],
            dim=2,
        )
        pairwise_axis_logits = self.pair_mlp(pair_features)
        quality_pred = self.quality_head(h.mean(dim=1)).flatten()
        return {
            "eps_pred": eps_pred,
            "pairwise_axis_logits": pairwise_axis_logits,
            "quality_pred": quality_pred,
        }
