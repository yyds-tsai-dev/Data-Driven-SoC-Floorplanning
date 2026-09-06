#!/usr/bin/env python3
"""Dense graph denoiser and diffusion schedule for FloorSet layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    node_feat_dim: int = 13
    z_dim: int = 4
    z_repr: str = "xylogwh"
    relation_feat_dim: int = 0
    d_model: int = 192
    layers: int = 6
    dropout: float = 0.1
    timesteps: int = 1000
    predict_z0_head: bool = False


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DenseGNNLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float, relation_feat_dim: int = 0):
        super().__init__()
        self.relation_feat_dim = relation_feat_dim
        in_dim = 3 * d_model if relation_feat_dim > 0 else 2 * d_model
        self.norm = nn.LayerNorm(in_dim)
        if relation_feat_dim > 0:
            self.rel_gate = nn.Sequential(nn.Linear(relation_feat_dim, d_model), nn.SiLU(), nn.Linear(d_model, 1))
            self.rel_proj = nn.Sequential(nn.Linear(relation_feat_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        rel_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.relation_feat_dim > 0 and rel_feat is not None and rel_feat.shape[-1] > 0:
            pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(h.dtype)
            rel_gate = torch.sigmoid(self.rel_gate(rel_feat)).squeeze(-1) * pair_mask
            rel_adj = adj + rel_gate
            rel_adj = rel_adj / rel_adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            msg = torch.bmm(rel_adj, h)
            rel_msg = (self.rel_proj(rel_feat) * rel_gate.unsqueeze(-1)).sum(dim=2)
            rel_msg = rel_msg / rel_gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
            layer_input = torch.cat([h, msg, rel_msg], dim=-1)
        else:
            msg = torch.bmm(adj, h)
            layer_input = torch.cat([h, msg], dim=-1)
        update = self.mlp(self.norm(layer_input))
        h = h + update
        return h * mask.unsqueeze(-1)


class GraphDiffusionDenoiser(nn.Module):
    """Graph-conditioned v-prediction denoiser.

    Args:
        z_t: ``[B, N, z_dim]`` noisy layout variables.
        t: ``[B]`` integer diffusion timestep.
        node_feat: ``[B, N, F]`` condition features.
        adj: ``[B, N, N]`` normalized dense adjacency.
        mask: ``[B, N]`` valid block mask.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.d_model
        self.input = nn.Linear(self.config.z_dim + self.config.node_feat_dim, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))
        self.layers = nn.ModuleList(
            [DenseGNNLayer(d, self.config.dropout, self.config.relation_feat_dim) for _ in range(self.config.layers)]
        )
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.config.z_dim)
        self.z0_out = nn.Linear(d, self.config.z_dim) if self.config.predict_z0_head else None
        self.gradient_checkpointing = False

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        node_feat: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        rel_feat: Optional[torch.Tensor] = None,
        return_z0: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.input(torch.cat([z_t, node_feat], dim=-1))
        h = h + self.time_mlp(sinusoidal_embedding(t, self.config.d_model)).unsqueeze(1)
        h = h * mask.unsqueeze(-1)
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                h = checkpoint(lambda hidden, layer=layer: layer(hidden, adj, mask, rel_feat), h, use_reentrant=False)
            else:
                h = layer(h, adj, mask, rel_feat)
        h = self.out_norm(h)
        v = self.out(h) * mask.unsqueeze(-1)
        if return_z0:
            z0 = self.z0_out(h) * mask.unsqueeze(-1) if self.z0_out is not None else None
            return v, z0
        return v


def make_beta_schedule(timesteps: int = 1000, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if schedule != "cosine":
        raise ValueError(f"unknown beta schedule: {schedule}")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    s = 0.008
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999)


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device: torch.device | str = "cpu"):
        betas = make_beta_schedule(timesteps, schedule).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(ab), torch.sqrt((1.0 - ab).clamp_min(0.0))


@torch.no_grad()
def ddim_sample(
    model: GraphDiffusionDenoiser,
    node_feat: torch.Tensor,
    adj: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 40,
    eta: float = 0.0,
    schedule: DiffusionSchedule | None = None,
    generator: torch.Generator | None = None,
    rel_feat: torch.Tensor | None = None,
    z0_blend: float = 0.0,
) -> torch.Tensor:
    """DDIM-like sampler returning z0 estimate ``[B, N, z_dim]``."""
    device = node_feat.device
    sched = schedule or DiffusionSchedule(model.config.timesteps, device=device)
    sched.to(device)
    bsz, n, _ = node_feat.shape
    z = torch.randn((bsz, n, model.config.z_dim), device=device, generator=generator) * mask.unsqueeze(-1)
    times = torch.linspace(sched.timesteps - 1, 0, steps, device=device).long()
    for idx, t_val in enumerate(times):
        t = torch.full((bsz,), int(t_val.item()), device=device, dtype=torch.long)
        alpha, sigma = sched.alpha_sigma(t)
        if z0_blend > 0.0 and getattr(model.config, "predict_z0_head", False):
            v, z0_direct = model(z, t, node_feat, adj, mask, rel_feat=rel_feat, return_z0=True)
        else:
            v = model(z, t, node_feat, adj, mask, rel_feat=rel_feat)
            z0_direct = None
        z0 = alpha * z - sigma * v
        if z0_direct is not None:
            blend = max(0.0, min(float(z0_blend), 1.0))
            z0 = (1.0 - blend) * z0 + blend * z0_direct
        eps = sigma * z + alpha * v
        if idx == len(times) - 1:
            z = z0
            break
        t_next = torch.full((bsz,), int(times[idx + 1].item()), device=device, dtype=torch.long)
        alpha_next, sigma_next = sched.alpha_sigma(t_next)
        if eta > 0:
            noise = torch.randn(z.shape, device=device, generator=generator) * mask.unsqueeze(-1)
            z = alpha_next * z0 + sigma_next * ((1.0 - eta) * eps + eta * noise)
        else:
            z = alpha_next * z0 + sigma_next * eps
        z = z * mask.unsqueeze(-1)
    if z.shape[-1] >= 4 and getattr(model.config, "z_repr", "xylogwh") == "xyaspect":
        z[..., 2] = z[..., 2].clamp(-3.0, 3.0)
        z[..., 3] = z[..., 3].clamp(-2.0, 2.0)
    elif z.shape[-1] >= 4:
        z[..., 2:] = z[..., 2:].clamp(-8.0, 2.0)
    else:
        z[..., 2] = z[..., 2].clamp(-3.0, 3.0)
    return z * mask.unsqueeze(-1)


@torch.no_grad()
def ddim_refine(
    model: GraphDiffusionDenoiser,
    z_init: torch.Tensor,
    node_feat: torch.Tensor,
    adj: torch.Tensor,
    mask: torch.Tensor,
    start_t: int = 250,
    steps: int = 20,
    eta: float = 0.0,
    schedule: DiffusionSchedule | None = None,
    generator: torch.Generator | None = None,
    rel_feat: torch.Tensor | None = None,
    z0_blend: float = 0.0,
) -> torch.Tensor:
    """DDIM refine from a provided initial layout instead of pure noise."""
    device = node_feat.device
    sched = schedule or DiffusionSchedule(model.config.timesteps, device=device)
    sched.to(device)
    start_t = max(0, min(int(start_t), sched.timesteps - 1))
    t0 = torch.full((z_init.shape[0],), start_t, device=device, dtype=torch.long)
    alpha, sigma = sched.alpha_sigma(t0)
    noise = torch.randn(z_init.shape, device=device, generator=generator) * mask.unsqueeze(-1)
    z = (alpha * z_init + sigma * noise) * mask.unsqueeze(-1)
    times = torch.linspace(start_t, 0, max(1, steps), device=device).long()
    for idx, t_val in enumerate(times):
        t = torch.full((z_init.shape[0],), int(t_val.item()), device=device, dtype=torch.long)
        alpha, sigma = sched.alpha_sigma(t)
        if z0_blend > 0.0 and getattr(model.config, "predict_z0_head", False):
            v, z0_direct = model(z, t, node_feat, adj, mask, rel_feat=rel_feat, return_z0=True)
        else:
            v = model(z, t, node_feat, adj, mask, rel_feat=rel_feat)
            z0_direct = None
        z0 = alpha * z - sigma * v
        if z0_direct is not None:
            blend = max(0.0, min(float(z0_blend), 1.0))
            z0 = (1.0 - blend) * z0 + blend * z0_direct
        eps = sigma * z + alpha * v
        if idx == len(times) - 1:
            z = z0
            break
        t_next = torch.full((z_init.shape[0],), int(times[idx + 1].item()), device=device, dtype=torch.long)
        alpha_next, sigma_next = sched.alpha_sigma(t_next)
        if eta > 0:
            noise = torch.randn(z.shape, device=device, generator=generator) * mask.unsqueeze(-1)
            z = alpha_next * z0 + sigma_next * ((1.0 - eta) * eps + eta * noise)
        else:
            z = alpha_next * z0 + sigma_next * eps
        z = z * mask.unsqueeze(-1)
    if z.shape[-1] >= 4 and getattr(model.config, "z_repr", "xylogwh") == "xyaspect":
        z[..., 2] = z[..., 2].clamp(-3.0, 3.0)
        z[..., 3] = z[..., 3].clamp(-2.0, 2.0)
    elif z.shape[-1] >= 4:
        z[..., 2:] = z[..., 2:].clamp(-8.0, 2.0)
    else:
        z[..., 2] = z[..., 2].clamp(-3.0, 3.0)
    return z * mask.unsqueeze(-1)
