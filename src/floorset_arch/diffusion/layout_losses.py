from __future__ import annotations

import torch

from floorset_arch.diffusion.contracts import DiffusionGraphInputs


def rect_tensors_from_x0(
    graph: DiffusionGraphInputs,
    x0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = max(float(graph.scale), 1.0)
    centers = x0[:, :, :2] * scale
    log_aspect = x0[:, :, 2].clamp(-2.5, 2.5)
    area = graph.area.float().to(x0.device).view(1, -1).clamp_min(1.0)
    aspect = torch.exp(log_aspect)
    width = torch.sqrt(area * aspect).clamp_min(1.0)
    height = torch.sqrt(area / aspect.clamp_min(1e-6)).clamp_min(1.0)
    x1 = centers[:, :, 0] - 0.5 * width
    y1 = centers[:, :, 1] - 0.5 * height
    x2 = centers[:, :, 0] + 0.5 * width
    y2 = centers[:, :, 1] + 0.5 * height
    return x1, y1, x2, y2


def overlap_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    block_count = int(x0.shape[1])
    if block_count < 2:
        return x0.new_zeros(x0.shape[0])
    x1, y1, x2, y2 = rect_tensors_from_x0(graph, x0)
    left, right = torch.triu_indices(block_count, block_count, offset=1, device=x0.device)
    overlap_x = (
        torch.minimum(x2[:, left], x2[:, right])
        - torch.maximum(x1[:, left], x1[:, right])
    ).clamp_min(0.0)
    overlap_y = (
        torch.minimum(y2[:, left], y2[:, right])
        - torch.maximum(y1[:, left], y1[:, right])
    ).clamp_min(0.0)
    scale_area = max(float(graph.scale), 1.0) ** 2
    return (overlap_x * overlap_y).sum(dim=1) / scale_area


def bbox_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = rect_tensors_from_x0(graph, x0)
    width = (x2.max(dim=1).values - x1.min(dim=1).values).clamp_min(0.0)
    height = (y2.max(dim=1).values - y1.min(dim=1).values).clamp_min(0.0)
    scale_area = max(float(graph.scale), 1.0) ** 2
    return width * height / scale_area


def pair_distance_loss(
    graph: DiffusionGraphInputs,
    x0: torch.Tensor,
    feature_index: int,
) -> torch.Tensor:
    if graph.pair_index.numel() == 0 or graph.raw_pair_features.shape[1] <= feature_index:
        return x0.new_zeros(x0.shape[0])
    pair_index = graph.pair_index.to(x0.device)
    weights = graph.raw_pair_features[:, feature_index].float().to(x0.device).clamp_min(0.0)
    if bool((weights <= 0).all().item()):
        return x0.new_zeros(x0.shape[0])
    centers = x0[:, :, :2]
    delta = centers[:, pair_index[:, 0], :] - centers[:, pair_index[:, 1], :]
    distance = torch.linalg.norm(delta, dim=2)
    return (distance * weights.view(1, -1)).sum(dim=1) / weights.sum().clamp_min(1e-6)


def boundary_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    codes = graph.boundary_codes.to(x0.device)
    if not bool((codes != 0).any().item()):
        return x0.new_zeros(x0.shape[0])
    x1, y1, x2, y2 = rect_tensors_from_x0(graph, x0)
    left = x1.min(dim=1).values.view(-1, 1)
    bottom = y1.min(dim=1).values.view(-1, 1)
    right = x2.max(dim=1).values.view(-1, 1)
    top = y2.max(dim=1).values.view(-1, 1)
    loss = x0.new_zeros(x0.shape[0])
    count = 0
    scale = max(float(graph.scale), 1.0)
    for bit, values, target in (
        (1, x1, left),
        (2, x2, right),
        (4, y2, top),
        (8, y1, bottom),
    ):
        mask = (codes & bit) != 0
        if bool(mask.any().item()):
            loss = loss + (values[:, mask] - target).abs().sum(dim=1) / scale
            count += int(mask.sum().item())
    return loss / max(count, 1)


def mib_aspect_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    if graph.pair_index.numel() == 0 or graph.raw_pair_features.shape[1] <= 3:
        return x0.new_zeros(x0.shape[0])
    pair_index = graph.pair_index.to(x0.device)
    weights = graph.raw_pair_features[:, 3].float().to(x0.device).clamp_min(0.0)
    if bool((weights <= 0).all().item()):
        return x0.new_zeros(x0.shape[0])
    log_aspect = x0[:, :, 2]
    diff = (log_aspect[:, pair_index[:, 0]] - log_aspect[:, pair_index[:, 1]]).abs()
    return (diff * weights.view(1, -1)).sum(dim=1) / weights.sum().clamp_min(1e-6)
