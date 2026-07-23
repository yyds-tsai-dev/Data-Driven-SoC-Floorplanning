#!/usr/bin/env python3
"""High-fidelity coordinate predictor for the direct-prediction pipeline.

Design (each choice has a measured/literature-backed reason):
  * Relation-biased multi-head attention (Graphormer-style) instead of the
    mean-aggregation DenseGNN: precise packing needs sharp, per-pair routing
    of information; attention with additive per-head biases from the pair
    features (net weight, same-cluster, same-MIB, ...) provides it. n<=130
    so dense attention is cheap.
  * AdaLN-Zero time conditioning (DiT recipe): stronger than additive time
    embeddings for conditional diffusion.
  * Self-conditioning (Analog Bits): the denoiser receives its previous x0
    estimate as extra input; consistently improves coordinate diffusion.
  * v-prediction + min-SNR-weighted x0 loss (computed in the trainer).
  * EMA weights for sampling.
  * Anchor clamping during sampling: preplaced x/y and fixed shapes are
    known inputs, so the corresponding z channels are re-imposed (noised to
    the current level) at every step — the trajectory can never drift away
    from the hard constraints that anchor the global frame.

The z representation is the repo-standard "xyaspect":
    z = [x/S, y/S, log(w/h), 0],  S = sqrt(sum of target areas)
so all existing helpers (fp_sol_to_z0 / z_to_rectangles) apply unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model import DiffusionSchedule, sinusoidal_embedding


@dataclass
class DirectModelConfig:
    node_feat_dim: int = 26
    relation_feat_dim: int = 9
    z_dim: int = 4
    z_repr: str = "xyaspect"
    d_model: int = 512
    layers: int = 12
    heads: int = 8
    dropout: float = 0.0
    timesteps: int = 1000
    self_conditioning: bool = True


class RelBiasAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dk = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, bias: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.heads, self.dk).transpose(1, 2)
        k = k.view(B, N, self.heads, self.dk).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk) + bias
        neg = torch.finfo(att.dtype).min
        att = att.masked_fill(~mask.view(B, 1, 1, N), neg)
        att = att.softmax(dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


class DiTBlock(nn.Module):
    """Pre-LN transformer block with AdaLN-Zero time modulation."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = RelBiasAttention(d_model, heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, h: torch.Tensor, temb: torch.Tensor,
                bias: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(temb).unsqueeze(1).chunk(6, dim=-1)
        x = self.norm1(h) * (1 + s1) + b1
        h = h + g1 * self.attn(x, bias, mask)
        x = self.norm2(h) * (1 + s2) + b2
        h = h + g2 * self.mlp(x)
        return h * mask.unsqueeze(-1)


class DirectDenoiser(nn.Module):
    def __init__(self, config: DirectModelConfig | None = None):
        super().__init__()
        self.config = config or DirectModelConfig()
        c = self.config
        in_dim = c.z_dim + c.node_feat_dim + (c.z_dim if c.self_conditioning else 0)
        self.input = nn.Linear(in_dim, c.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(c.d_model, 4 * c.d_model), nn.SiLU(),
            nn.Linear(4 * c.d_model, c.d_model))
        self.rel_bias = nn.Sequential(
            nn.Linear(c.relation_feat_dim, 64), nn.SiLU(),
            nn.Linear(64, c.heads))
        self.blocks = nn.ModuleList(
            [DiTBlock(c.d_model, c.heads, c.dropout) for _ in range(c.layers)])
        self.out_norm = nn.LayerNorm(c.d_model, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(c.d_model, 2 * c.d_model))
        nn.init.zeros_(self.out_ada[1].weight)
        nn.init.zeros_(self.out_ada[1].bias)
        self.out = nn.Linear(c.d_model, c.z_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        node_feat: torch.Tensor,
        adj: torch.Tensor,          # unused; kept for API parity
        mask: torch.Tensor,
        rel_feat: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.config
        parts = [z_t, node_feat]
        if c.self_conditioning:
            parts.append(torch.zeros_like(z_t) if self_cond is None else self_cond)
        h = self.input(torch.cat(parts, dim=-1))
        temb = self.time_mlp(sinusoidal_embedding(t, c.d_model))
        if rel_feat is not None and rel_feat.shape[-1] > 0:
            bias = self.rel_bias(rel_feat).permute(0, 3, 1, 2)
        else:
            B, N, _ = z_t.shape
            bias = z_t.new_zeros((B, c.heads, N, N))
        h = h * mask.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, temb, bias, mask)
        s, b = self.out_ada(temb).unsqueeze(1).chunk(2, dim=-1)
        h = self.out_norm(h) * (1 + s) + b
        return self.out(h) * mask.unsqueeze(-1)   # v-prediction


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        sd = model.state_dict()
        for k in sd:
            sd[k].copy_(self.shadow[k].to(sd[k].dtype))

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone().float() for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Anchor helpers: preplaced x/y and fixed/preplaced aspect are known inputs
# ---------------------------------------------------------------------------
def known_z_channels(
    area_target: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (z_known [B,N,4], chan_mask [B,N,4]) for hard-constraint dims."""
    B, N = area_target.shape
    z_known = area_target.new_zeros((B, N, 4))
    known = torch.zeros((B, N, 4), dtype=torch.bool, device=area_target.device)
    if constraints.shape[-1] <= 1:
        return z_known, known
    s = scale.view(B, 1).clamp_min(1.0)
    preplaced = (constraints[..., 1] != 0) & (area_target != -1)
    fixed = (constraints[..., 0] != 0) & (area_target != -1)
    tp = target_positions
    has_xy = preplaced & (tp[..., 0] >= 0) & (tp[..., 1] >= 0)
    has_wh = (fixed | preplaced) & (tp[..., 2] > 0) & (tp[..., 3] > 0)
    z_known[..., 0] = torch.where(has_xy, tp[..., 0] / s, z_known[..., 0])
    z_known[..., 1] = torch.where(has_xy, tp[..., 1] / s, z_known[..., 1])
    la = torch.log((tp[..., 2] / tp[..., 3].clamp_min(1e-6)).clamp_min(1e-6)).clamp(-3, 3)
    z_known[..., 2] = torch.where(has_wh, la, z_known[..., 2])
    known[..., 0] = has_xy
    known[..., 1] = has_xy
    known[..., 2] = has_wh
    return z_known, known


@torch.no_grad()
def sample_direct(
    model: DirectDenoiser,
    cond: Dict[str, torch.Tensor],
    schedule: DiffusionSchedule,
    steps: int = 50,
    generator: Optional[torch.Generator] = None,
    z_known: Optional[torch.Tensor] = None,
    known_mask: Optional[torch.Tensor] = None,
    guidance=None,
) -> torch.Tensor:
    """DDIM sampling with self-conditioning and hard-anchor clamping."""
    node_feat = cond["node_feat"]
    mask = cond["mask"]
    rel_feat = cond.get("rel_feat")
    device = node_feat.device
    B, N, _ = node_feat.shape
    zdim = model.config.z_dim
    z = torch.randn((B, N, zdim), device=device, generator=generator)
    z = z * mask.unsqueeze(-1)
    times = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()
    sc = None
    for idx, t_val in enumerate(times):
        t = torch.full((B,), int(t_val.item()), device=device, dtype=torch.long)
        alpha, sigma = schedule.alpha_sigma(t)
        if z_known is not None and known_mask is not None:
            noise = torch.randn(z.shape, device=device, generator=generator)
            z_imposed = alpha * z_known + sigma * noise
            z = torch.where(known_mask, z_imposed, z)
        v = model(z, t, node_feat, cond.get("adj"), mask,
                  rel_feat=rel_feat, self_cond=sc)
        z0 = alpha * z - sigma * v
        z0[..., 2] = z0[..., 2].clamp(-3.0, 3.0)
        if z_known is not None and known_mask is not None:
            z0 = torch.where(known_mask, z_known, z0)
        if guidance is not None:
            data_time = 1.0 - float(t_val.item()) / max(schedule.timesteps - 1, 1)
            with torch.enable_grad():
                z0 = guidance(z0, data_time)
            z0[..., 2] = z0[..., 2].clamp(-3.0, 3.0)
            if z_known is not None and known_mask is not None:
                z0 = torch.where(known_mask, z_known, z0)
            sc = z0
            eps = (z - alpha * z0) / sigma.clamp_min(1e-6)
        else:
            sc = z0
            eps = sigma * z + alpha * v
        if idx == len(times) - 1:
            z = z0
            break
        t_next = torch.full((B,), int(times[idx + 1].item()),
                            device=device, dtype=torch.long)
        alpha_n, sigma_n = schedule.alpha_sigma(t_next)
        z = (alpha_n * z0 + sigma_n * eps) * mask.unsqueeze(-1)
    return z * mask.unsqueeze(-1)
