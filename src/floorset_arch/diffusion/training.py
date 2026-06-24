from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from floorset_arch.diffusion.contracts import DiffusionGraphInputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.targets import DiffusionTargets


@dataclass(frozen=True)
class DiffusionLossConfig:
    max_steps: int = 1000
    noise_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    pair_weight: float = 0.25
    tree_weight: float = 0.25
    quality_weight: float = 0.01
    noise_samples: int = 1


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _alpha_bar(
    timesteps: torch.Tensor,
    config: DiffusionLossConfig,
) -> torch.Tensor:
    max_steps = max(1, int(config.max_steps))
    t = timesteps.clamp(0, max_steps - 1).to(torch.float32)
    if config.noise_schedule == "linear":
        betas = torch.linspace(
            float(config.beta_start),
            float(config.beta_end),
            max_steps,
            device=timesteps.device,
            dtype=torch.float32,
        )
        return torch.cumprod(1.0 - betas, dim=0)[timesteps.clamp(0, max_steps - 1)]
    if config.noise_schedule != "cosine":
        raise ValueError("noise_schedule must be cosine or linear")

    s = 0.008
    u = (t + 1.0) / float(max_steps)
    base = math.cos((s / (1.0 + s)) * math.pi * 0.5) ** 2
    values = torch.cos(((u + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    return (values / base).clamp(1e-5, 0.9999)


def diffusion_training_loss(
    model: GraphConditionedPlacementDiffusion,
    graph: DiffusionGraphInputs,
    targets: DiffusionTargets,
    seed: int | None = 0,
    config: DiffusionLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    config = config or DiffusionLossConfig()
    generator = torch.Generator(device=graph.area.device)
    if seed is not None:
        generator.manual_seed(int(seed))
    clean = torch.cat([targets.center, targets.log_aspect.view(-1, 1)], dim=1)
    clean = clean.view(1, graph.area.shape[0], 3)
    noise_samples = max(1, int(config.noise_samples))
    clean = clean.expand(noise_samples, -1, -1)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    timestep = torch.randint(
        low=0,
        high=max(1, int(config.max_steps)),
        size=(noise_samples,),
        generator=generator,
        device=clean.device,
        dtype=torch.long,
    )
    alpha = _alpha_bar(timestep, config).view(noise_samples, 1, 1)
    noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
    pred = model(graph, noisy, timestep)
    denoise = F.mse_loss(pred["eps_pred"], noise)

    pair_mask = targets.pair_axis_label >= 0
    if pair_mask.any():
        pair_loss = F.cross_entropy(
            pred["pairwise_axis_logits"][:, pair_mask, :].reshape(-1, 3),
            targets.pair_axis_label[pair_mask]
            .to(pred["pairwise_axis_logits"].device)
            .view(1, -1)
            .expand(noise_samples, -1)
            .reshape(-1),
        )
    else:
        pair_loss = _zero_like_loss(denoise)

    if targets.tree_pair_mask.any():
        tree_logits = pred["pairwise_axis_logits"][
            :, targets.tree_pair_mask, :2
        ].reshape(-1, 2)
        tree_loss = F.cross_entropy(
            tree_logits,
            targets.tree_side_label[targets.tree_pair_mask]
            .to(tree_logits.device)
            .view(1, -1)
            .expand(noise_samples, -1)
            .reshape(-1),
        )
    else:
        tree_loss = _zero_like_loss(denoise)

    if targets.quality_labels.numel() > 0:
        quality_target = torch.log1p(
            targets.quality_labels.float().clamp_min(0.0)
        ).mean().view(1)
        quality_target = quality_target.to(pred["quality_pred"].device)
        quality_loss = F.mse_loss(
            pred["quality_pred"].view(noise_samples),
            quality_target.expand(noise_samples),
        )
    else:
        quality_loss = _zero_like_loss(denoise)

    loss = (
        denoise
        + float(config.pair_weight) * pair_loss
        + float(config.tree_weight) * tree_loss
        + float(config.quality_weight) * quality_loss
    )
    return loss, {
        "total": float(loss.detach().cpu().item()),
        "denoise": float(denoise.detach().cpu().item()),
        "pair": float(pair_loss.detach().cpu().item()),
        "tree": float(tree_loss.detach().cpu().item()),
        "quality": float(quality_loss.detach().cpu().item()),
        "quality_target": float(quality_target.detach().cpu().item())
        if targets.quality_labels.numel() > 0
        else 0.0,
        "timestep_mean": float(timestep.float().detach().cpu().mean().item()),
        "noise_schedule": config.noise_schedule,
    }
