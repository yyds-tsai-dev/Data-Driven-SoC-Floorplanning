from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from floorset_arch.diffusion.contracts import (
    DIFFUSION_FEATURE_VERSION,
    DiffusionGraphInputs,
    DiffusionPlacementPrior,
)
from floorset_arch.diffusion.layout_losses import denormalize_center
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.training import DiffusionLossConfig, _alpha_bar


@dataclass(frozen=True)
class DiffusionGuidanceConfig:
    enabled: bool = True
    step_size: float = 0.04
    max_grad_norm: float = 1.0
    overlap_weight: float = 1.0
    bbox_weight: float = 0.02
    net_weight: float = 0.02
    cluster_weight: float = 0.20
    boundary_weight: float = 0.15
    mib_weight: float = 0.15
    center_clip: float = 2.0
    log_aspect_clip: float = 2.5


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def diffusion_guidance_config_from_env() -> DiffusionGuidanceConfig:
    return DiffusionGuidanceConfig(
        enabled=_env_flag("FLOORSET_DIFFUSION_GUIDANCE", True),
        step_size=_env_float("FLOORSET_DIFFUSION_GUIDANCE_STEP", 0.04),
        max_grad_norm=_env_float("FLOORSET_DIFFUSION_GUIDANCE_MAX_GRAD", 1.0),
        overlap_weight=_env_float("FLOORSET_DIFFUSION_GUIDANCE_OVERLAP", 1.0),
        bbox_weight=_env_float("FLOORSET_DIFFUSION_GUIDANCE_BBOX", 0.02),
        net_weight=_env_float("FLOORSET_DIFFUSION_GUIDANCE_NET", 0.02),
        cluster_weight=_env_float("FLOORSET_DIFFUSION_GUIDANCE_CLUSTER", 0.20),
        boundary_weight=_env_float("FLOORSET_DIFFUSION_GUIDANCE_BOUNDARY", 0.15),
        mib_weight=_env_float("FLOORSET_DIFFUSION_GUIDANCE_MIB", 0.15),
        center_clip=_env_float("FLOORSET_DIFFUSION_CENTER_CLIP", 2.0),
        log_aspect_clip=_env_float("FLOORSET_DIFFUSION_LOG_ASPECT_CLIP", 2.5),
    )


def _sampling_timestep_schedule(
    max_steps: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    max_steps = max(1, int(max_steps))
    count = max(1, min(int(steps), max_steps))
    return torch.linspace(
        max_steps - 1,
        0,
        count,
        device=device,
        dtype=torch.float32,
    ).round().long()


def _loss_config_from_payload(payload: dict) -> DiffusionLossConfig:
    raw = payload.get("loss_config") or {}
    allowed = {field.name for field in fields(DiffusionLossConfig)}
    values = {key: raw[key] for key in allowed if key in raw}
    return DiffusionLossConfig(**values)


def _clamp_x0(x0: torch.Tensor, config: DiffusionGuidanceConfig) -> torch.Tensor:
    centers = x0[:, :, :2].clamp(-float(config.center_clip), float(config.center_clip))
    log_aspect = x0[:, :, 2:].clamp(
        -float(config.log_aspect_clip), float(config.log_aspect_clip)
    )
    return torch.cat([centers, log_aspect], dim=2)


def _rect_tensors(
    graph: DiffusionGraphInputs,
    x0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = denormalize_center(graph, x0[:, :, :2])
    log_aspect = x0[:, :, 2].clamp(-2.5, 2.5)
    area = graph.area.float().to(x0.device).view(1, -1).clamp_min(1.0)
    aspect = torch.exp(log_aspect)
    width = torch.sqrt(area * aspect).clamp_min(1.0)
    height = torch.sqrt(area / aspect.clamp_min(1e-6)).clamp_min(1.0)
    x1 = centers[:, :, 0] - 0.5 * width
    y1 = centers[:, :, 1] - 0.5 * height
    x2 = centers[:, :, 0] + 0.5 * width
    y2 = centers[:, :, 1] + 0.5 * height
    return x1, y1, x2, y2, width, height


def _overlap_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    block_count = int(x0.shape[1])
    if block_count < 2:
        return x0.new_zeros(x0.shape[0])
    x1, y1, x2, y2, _width, _height = _rect_tensors(graph, x0)
    left_index, right_index = torch.triu_indices(
        block_count,
        block_count,
        offset=1,
        device=x0.device,
    )
    ix = (
        torch.minimum(x2[:, left_index], x2[:, right_index])
        - torch.maximum(x1[:, left_index], x1[:, right_index])
    ).clamp_min(0.0)
    iy = (
        torch.minimum(y2[:, left_index], y2[:, right_index])
        - torch.maximum(y1[:, left_index], y1[:, right_index])
    ).clamp_min(0.0)
    scale_area = max(float(graph.scale), 1.0) ** 2
    return (ix * iy).sum(dim=1) / scale_area


def _bbox_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2, _width, _height = _rect_tensors(graph, x0)
    width = (x2.max(dim=1).values - x1.min(dim=1).values).clamp_min(0.0)
    height = (y2.max(dim=1).values - y1.min(dim=1).values).clamp_min(0.0)
    scale_area = max(float(graph.scale), 1.0) ** 2
    return width * height / scale_area


def _pair_distance_loss(
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


def _boundary_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    codes = graph.boundary_codes.to(x0.device)
    mask = codes != 0
    if not bool(mask.any().item()):
        return x0.new_zeros(x0.shape[0])
    x1, y1, x2, y2, _width, _height = _rect_tensors(graph, x0)
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
        bit_mask = (codes & bit) != 0
        if bool(bit_mask.any().item()):
            loss = loss + (values[:, bit_mask] - target).abs().sum(dim=1) / scale
            count += int(bit_mask.sum().item())
    return loss / max(count, 1)


def _mib_aspect_loss(graph: DiffusionGraphInputs, x0: torch.Tensor) -> torch.Tensor:
    if graph.pair_index.numel() == 0 or graph.raw_pair_features.shape[1] <= 3:
        return x0.new_zeros(x0.shape[0])
    pair_index = graph.pair_index.to(x0.device)
    weights = graph.raw_pair_features[:, 3].float().to(x0.device).clamp_min(0.0)
    if bool((weights <= 0).all().item()):
        return x0.new_zeros(x0.shape[0])
    log_aspect = x0[:, :, 2]
    diff = (log_aspect[:, pair_index[:, 0]] - log_aspect[:, pair_index[:, 1]]).abs()
    return (diff * weights.view(1, -1)).sum(dim=1) / weights.sum().clamp_min(1e-6)


def _layout_guidance_loss(
    graph: DiffusionGraphInputs,
    x0: torch.Tensor,
    quality_pred: torch.Tensor | None,
    config: DiffusionGuidanceConfig,
) -> torch.Tensor:
    loss = x0.new_zeros(x0.shape[0])
    if config.overlap_weight:
        loss = loss + float(config.overlap_weight) * _overlap_loss(graph, x0)
    if config.bbox_weight:
        loss = loss + float(config.bbox_weight) * _bbox_loss(graph, x0)
    if config.net_weight:
        loss = loss + float(config.net_weight) * _pair_distance_loss(graph, x0, 1)
    if config.cluster_weight:
        loss = loss + float(config.cluster_weight) * _pair_distance_loss(graph, x0, 2)
    if config.boundary_weight:
        loss = loss + float(config.boundary_weight) * _boundary_loss(graph, x0)
    if config.mib_weight:
        loss = loss + float(config.mib_weight) * _mib_aspect_loss(graph, x0)
    return loss


def apply_layout_guidance(
    graph: DiffusionGraphInputs,
    x0: torch.Tensor,
    quality_pred: torch.Tensor | None,
    config: DiffusionGuidanceConfig,
) -> torch.Tensor:
    if not config.enabled:
        return x0
    with torch.enable_grad():
        guided = x0.detach().clone().requires_grad_(True)
        loss = _layout_guidance_loss(graph, guided, quality_pred, config).sum()
        grad = torch.autograd.grad(loss, guided, allow_unused=True)[0]
        if grad is None:
            return x0
        movable = graph.movable_mask.to(grad.device).view(1, -1, 1).float()
        grad = grad * movable
        norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        grad_scale = (float(config.max_grad_norm) / norm).clamp(max=1.0)
        updated = guided - float(config.step_size) * grad * grad_scale
    return _clamp_x0(updated.detach(), config)


def sample_diffusion_prior(
    model: GraphConditionedPlacementDiffusion,
    graph: DiffusionGraphInputs,
    samples: int = 8,
    steps: int = 16,
    seed: int | None = None,
    loss_config: DiffusionLossConfig | None = None,
    guidance_config: DiffusionGuidanceConfig | None = None,
) -> DiffusionPlacementPrior:
    loss_config = loss_config or getattr(model, "diffusion_loss_config", None)
    loss_config = loss_config or DiffusionLossConfig()
    guidance_config = guidance_config or diffusion_guidance_config_from_env()
    generator = torch.Generator(device=graph.area.device)
    if seed is not None:
        generator.manual_seed(int(seed))
    block_count = graph.area.shape[0]
    x_t = torch.randn(
        (samples, block_count, 3),
        generator=generator,
        device=graph.area.device,
    )
    pair_logits = torch.zeros(
        (samples, graph.pair_index.shape[0], 3),
        device=graph.area.device,
    )
    quality = torch.zeros(samples, device=graph.area.device)
    schedule = _sampling_timestep_schedule(
        max_steps=loss_config.max_steps,
        steps=steps,
        device=graph.area.device,
    )
    for index, step in enumerate(schedule.tolist()):
        t = torch.full((samples,), int(step), dtype=torch.long, device=graph.area.device)
        with torch.no_grad():
            out = model(graph, x_t, t)
            eps = out["eps_pred"]
            pair_logits = out["pairwise_axis_logits"]
            quality = out["quality_pred"]
            alpha = _alpha_bar(t, loss_config).view(samples, 1, 1).to(x_t.device)
            alpha = alpha.clamp(1e-5, 0.999999)
            x0 = (x_t - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
        x0 = apply_layout_guidance(graph, x0, quality, guidance_config)
        if index + 1 < int(schedule.numel()):
            prev_t = torch.full(
                (samples,),
                int(schedule[index + 1].item()),
                dtype=torch.long,
                device=graph.area.device,
            )
            prev_alpha = _alpha_bar(prev_t, loss_config).view(samples, 1, 1).to(x_t.device)
            prev_alpha = prev_alpha.clamp(1e-5, 0.999999)
        else:
            prev_alpha = torch.ones_like(alpha)
        x_t = prev_alpha.sqrt() * x0 + (1.0 - prev_alpha).sqrt() * eps
        pair_logits = out["pairwise_axis_logits"]
        quality = out["quality_pred"]
    centers = denormalize_center(graph, x_t[:, :, :2])
    log_aspect = x_t[:, :, 2].clamp(-2.5, 2.5)
    return DiffusionPlacementPrior(
        centers=centers,
        log_aspect=log_aspect,
        pairwise_axis_logits=pair_logits,
        pair_index=graph.pair_index,
        quality_pred=quality,
        uncertainty=x_t.detach().std(dim=2),
        scale=graph.scale,
        variant=model.variant,
    )


def load_diffusion_checkpoint(
    path: Path,
    graph: DiffusionGraphInputs,
    map_location: str | torch.device = "cpu",
    use_ema: bool = True,
) -> GraphConditionedPlacementDiffusion:
    payload = torch.load(path, map_location=map_location)
    ckpt_version = int(payload.get("feature_version", 1))
    graph_version = int(getattr(graph, "feature_version", DIFFUSION_FEATURE_VERSION))
    if ckpt_version != graph_version:
        raise ValueError(
            "diffusion feature_version mismatch: checkpoint was trained with "
            f"feature_version={ckpt_version} but the graph inputs are "
            f"feature_version={graph_version}. Rebuild the graph inputs with the "
            "matching version (legacy checkpoints without the field are v1), or "
            "retrain under the current schema."
        )
    variant = str(payload.get("variant", "raw"))
    hidden_dim = int(payload.get("hidden_dim", 128))
    layers = int(payload.get("layers", 2))
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph,
        variant=variant,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    state_key = "model_state_dict"
    if use_ema and payload.get("ema_model_state_dict") is not None:
        state_key = "ema_model_state_dict"
    model.load_state_dict(payload[state_key], strict=True)
    model.diffusion_loss_config = _loss_config_from_payload(payload)
    model.diffusion_checkpoint_state = "ema" if state_key == "ema_model_state_dict" else "raw"
    model.eval()
    return model
