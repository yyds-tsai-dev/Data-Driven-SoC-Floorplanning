from __future__ import annotations

import torch
import torch.nn.functional as F

from floorset_arch.diffusion.contracts import DiffusionGraphInputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.targets import DiffusionTargets


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def diffusion_training_loss(
    model: GraphConditionedPlacementDiffusion,
    graph: DiffusionGraphInputs,
    targets: DiffusionTargets,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    generator = torch.Generator(device=graph.area.device)
    generator.manual_seed(int(seed))
    clean = torch.cat([targets.center, targets.log_aspect.view(-1, 1)], dim=1)
    clean = clean.view(1, graph.area.shape[0], 3)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    timestep = torch.tensor([1], dtype=torch.long, device=clean.device)
    noisy = clean + 0.1 * noise
    pred = model(graph, noisy, timestep)
    denoise = F.mse_loss(pred["eps_pred"], noise)

    pair_mask = targets.pair_axis_label >= 0
    if pair_mask.any():
        pair_loss = F.cross_entropy(
            pred["pairwise_axis_logits"][0][pair_mask],
            targets.pair_axis_label[pair_mask].to(pred["pairwise_axis_logits"].device),
        )
    else:
        pair_loss = _zero_like_loss(denoise)

    if targets.tree_pair_mask.any():
        tree_logits = pred["pairwise_axis_logits"][0][targets.tree_pair_mask, :2]
        tree_loss = F.cross_entropy(
            tree_logits,
            targets.tree_side_label[targets.tree_pair_mask].to(tree_logits.device),
        )
    else:
        tree_loss = _zero_like_loss(denoise)

    if targets.quality_labels.numel() > 0:
        quality_target = targets.quality_labels.float().mean().view(1)
        quality_target = quality_target.to(pred["quality_pred"].device)
        quality_loss = F.mse_loss(pred["quality_pred"].view(1), quality_target)
    else:
        quality_loss = _zero_like_loss(denoise)

    loss = denoise + 0.25 * pair_loss + 0.25 * tree_loss + 0.01 * quality_loss
    return loss, {
        "denoise": float(denoise.detach().cpu().item()),
        "pair": float(pair_loss.detach().cpu().item()),
        "tree": float(tree_loss.detach().cpu().item()),
        "quality": float(quality_loss.detach().cpu().item()),
    }
