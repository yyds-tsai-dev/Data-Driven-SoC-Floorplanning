from __future__ import annotations

import math

import torch

from floorset_arch.diffusion.contracts import (
    DiffusionPlacementPrior,
    PlacementTensorBatch,
)
from floorset_arch.models import Instance, Placement, Rect


def concretize_diffusion_prior(
    inst: Instance, prior: DiffusionPlacementPrior
) -> PlacementTensorBatch:
    centers = torch.nan_to_num(
        prior.centers.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    log_aspect = torch.nan_to_num(prior.log_aspect.float(), nan=0.0).clamp(
        -2.5, 2.5
    )
    area = inst.area_targets[: inst.block_count].float().to(centers.device).clamp_min(1.0)
    aspect = torch.exp(log_aspect)
    width = torch.sqrt(area.view(1, -1) * aspect).clamp_min(1.0)
    height = torch.sqrt(area.view(1, -1) / aspect.clamp_min(1e-6)).clamp_min(1.0)
    x = (centers[:, :, 0] - width * 0.5).clamp_min(0.0)
    y = (centers[:, :, 1] - height * 0.5).clamp_min(0.0)
    rects = torch.stack([x, y, width, height], dim=2)

    for block, target in inst.target_rects.items():
        if block in inst.fixed or block in inst.preplaced:
            rects[:, block, 2] = float(target.width)
            rects[:, block, 3] = float(target.height)
        if block in inst.preplaced:
            rects[:, block, 0] = float(target.x)
            rects[:, block, 1] = float(target.y)

    quality = prior.quality_pred
    if quality is None:
        score_features = torch.zeros(
            (prior.sample_count, 1), dtype=rects.dtype, device=rects.device
        )
    else:
        score_features = quality.view(-1, 1).to(rects.device)
    return PlacementTensorBatch(
        rect_xywh=rects,
        pairwise_axis_logits=prior.pairwise_axis_logits,
        pair_index=prior.pair_index,
        score_features=score_features,
        source=f"diffusion:{prior.variant}",
    )


def placement_from_tensor_candidate(
    inst: Instance, batch: PlacementTensorBatch, index: int
) -> Placement:
    rects: dict[int, Rect] = {}
    sample = batch.rect_xywh[index].detach().cpu()
    for block in range(inst.block_count):
        x, y, width, height = [float(value) for value in sample[block].tolist()]
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            x, y, width, height = 0.0, 0.0, 1.0, 1.0
        rects[block] = Rect(
            max(0.0, x),
            max(0.0, y),
            max(1.0, width),
            max(1.0, height),
        )
    return Placement(rects)
