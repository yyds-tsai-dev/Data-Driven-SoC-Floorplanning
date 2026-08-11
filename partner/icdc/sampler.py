"""Differentiable few-step sampling from the shipped `DirectDenoiser`.

`direct_diffusion_model.sample_direct_dpmpp` is the production sampler and is
wrapped in `@torch.no_grad`.  This module re-implements the *same* update rule
without the no-grad decorator so the energy can be back-propagated straight
through the sampling trajectory (DRaFT-style direct gradient, not RL), with two
additions the trainer needs:

  * K samples per instance in one batch (the group soft-min objective needs a
    group, and the batch is the group);
  * `grad_steps`: only the last few denoising steps keep their graph, the
    earlier ones run under no_grad.  With S=2 the default keeps both, but the
    knob is what makes S=4 affordable on a shared 23 GB L4.

Everything else is bit-for-bit the shipped contract: hard-anchor imposition
before each model call, aspect clamp plus anchor overwrite on the x0 estimate,
x0 self-conditioning, `lower_order_final`.  Matching the production sampler
matters -- the fine-tune is only worth anything if the trajectory it optimises
is the trajectory inference will run.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def expand_cond(cond: Dict[str, torch.Tensor], K: int) -> Dict[str, torch.Tensor]:
    """Repeat every conditioning tensor K times along the batch axis.

    `repeat_interleave` (not `repeat`) so sample k of instance b lands at
    b*K + k and the group reshape in the trainer is a plain `view(B, K)`.
    """
    out = {}
    for key, val in cond.items():
        out[key] = (val.repeat_interleave(K, dim=0)
                    if torch.is_tensor(val) else val)
    return out


def sample_differentiable(
    model,
    cond: Dict[str, torch.Tensor],
    schedule,
    steps: int = 2,
    generator: Optional[torch.Generator] = None,
    z_known: Optional[torch.Tensor] = None,
    known_mask: Optional[torch.Tensor] = None,
    grad_steps: Optional[int] = None,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DPM-Solver++(2M) with gradients.  Returns the final z0 estimate."""
    node_feat = cond["node_feat"]
    mask = cond["mask"]
    rel_feat = cond.get("rel_feat")
    device = node_feat.device
    B, N, _ = node_feat.shape
    zdim = model.config.z_dim
    if noise is None:
        noise = torch.randn((B, N, zdim), device=device, generator=generator)
    z = noise * mask.unsqueeze(-1)
    times = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()
    if grad_steps is None or grad_steps > steps:
        grad_steps = steps
    sc = None
    prev_z0 = None
    prev_lam = None
    for idx, t_val in enumerate(times):
        want_grad = idx >= steps - grad_steps
        with torch.set_grad_enabled(want_grad and torch.is_grad_enabled()):
            t = torch.full((B,), int(t_val.item()), device=device, dtype=torch.long)
            alpha, sigma = schedule.alpha_sigma(t)
            if z_known is not None and known_mask is not None:
                anc = torch.randn(z.shape, device=device, generator=generator)
                z = torch.where(known_mask, alpha * z_known + sigma * anc, z)
            v = model(z, t, node_feat, cond.get("adj"), mask,
                      rel_feat=rel_feat, self_cond=sc)
            z0 = alpha * z - sigma * v
            z0 = torch.cat([z0[..., :2], z0[..., 2:3].clamp(-3.0, 3.0),
                            z0[..., 3:]], dim=-1)
            if z_known is not None and known_mask is not None:
                z0 = torch.where(known_mask, z_known, z0)
            sc = z0.detach()
            if idx == len(times) - 1:
                return z0 * mask.unsqueeze(-1)
            lam = torch.log(alpha.clamp_min(1e-12)) - torch.log(sigma.clamp_min(1e-12))
            t_next = torch.full((B,), int(times[idx + 1].item()),
                                device=device, dtype=torch.long)
            alpha_n, sigma_n = schedule.alpha_sigma(t_next)
            lam_n = torch.log(alpha_n.clamp_min(1e-12)) - torch.log(sigma_n.clamp_min(1e-12))
            h = lam_n - lam
            d = z0
            if prev_z0 is not None and idx + 1 < len(times) - 1:
                r = (lam - prev_lam) / h.clamp_min(1e-12)
                d = z0 + (z0 - prev_z0) / (2.0 * r.clamp_min(1e-12))
            z = (sigma_n / sigma.clamp_min(1e-12)) * z - alpha_n * torch.expm1(-h) * d
            z = z * mask.unsqueeze(-1)
            prev_z0 = z0
            prev_lam = lam
    return z * mask.unsqueeze(-1)
