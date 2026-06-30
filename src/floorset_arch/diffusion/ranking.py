from __future__ import annotations

import torch
import torch.nn.functional as F

from floorset_arch.diagnostics import placement_metrics
from floorset_arch.diffusion.contracts import PlacementTensorBatch
from floorset_arch.models import Instance, Placement
from floorset_arch.v10_proxy import v10_proxy_rank


def _overlap_proxy(rect_xywh: torch.Tensor) -> torch.Tensor:
    x1 = rect_xywh[:, :, 0]
    y1 = rect_xywh[:, :, 1]
    x2 = x1 + rect_xywh[:, :, 2].clamp_min(1.0)
    y2 = y1 + rect_xywh[:, :, 3].clamp_min(1.0)
    score = torch.zeros(
        rect_xywh.shape[0],
        dtype=rect_xywh.dtype,
        device=rect_xywh.device,
    )
    for i in range(rect_xywh.shape[1]):
        for j in range(i + 1, rect_xywh.shape[1]):
            ix = (
                torch.minimum(x2[:, i], x2[:, j])
                - torch.maximum(x1[:, i], x1[:, j])
            ).clamp_min(0.0)
            iy = (
                torch.minimum(y2[:, i], y2[:, j])
                - torch.maximum(y1[:, i], y1[:, j])
            ).clamp_min(0.0)
            score = score + ix * iy
    return score


def _pairwise_consistency_penalty(batch: PlacementTensorBatch) -> torch.Tensor:
    logits = batch.pairwise_axis_logits
    if batch.pair_index.numel() == 0 or logits.numel() == 0 or logits.shape[2] < 2:
        return torch.zeros(
            batch.rect_xywh.shape[0],
            dtype=batch.rect_xywh.dtype,
            device=batch.rect_xywh.device,
        )
    pair_index = batch.pair_index.to(batch.rect_xywh.device)
    centers = batch.rect_xywh[:, :, :2] + 0.5 * batch.rect_xywh[:, :, 2:4]
    delta = centers[:, pair_index[:, 1], :] - centers[:, pair_index[:, 0], :]
    labels = torch.where(delta[:, :, 0].abs() >= delta[:, :, 1].abs(), 0, 1)
    class_logits = logits[:, :, :3]
    per_pair = F.cross_entropy(
        class_logits.reshape(-1, class_logits.shape[2]),
        labels.reshape(-1).to(logits.device),
        reduction="none",
    ).view(logits.shape[0], -1)
    return per_pair.mean(dim=1).to(batch.rect_xywh.device)


def tensor_prefilter_score(batch: PlacementTensorBatch) -> torch.Tensor:
    overlap = _overlap_proxy(batch.rect_xywh)
    bbox_right = batch.rect_xywh[:, :, 0] + batch.rect_xywh[:, :, 2]
    bbox_top = batch.rect_xywh[:, :, 1] + batch.rect_xywh[:, :, 3]
    bbox = bbox_right.max(dim=1).values * bbox_top.max(dim=1).values
    model_score = torch.zeros_like(overlap)
    if batch.score_features is not None and batch.score_features.numel() > 0:
        model_score = batch.score_features[:, 0].to(overlap.device)
    pair_consistency = _pairwise_consistency_penalty(batch)
    return overlap * 1000.0 + bbox * 0.001 + pair_consistency * 0.25 + model_score


def select_tensor_shortlist(batch: PlacementTensorBatch, top_k: int) -> torch.Tensor:
    score = tensor_prefilter_score(batch)
    k = max(1, min(int(top_k), int(score.numel())))
    return torch.argsort(score)[:k]


def rank_repaired_placement(inst: Instance, placement: Placement) -> tuple:
    metrics = placement_metrics(inst, placement)
    return v10_proxy_rank(inst, placement, metrics)
