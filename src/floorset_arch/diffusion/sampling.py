from __future__ import annotations

from pathlib import Path

import torch

from floorset_arch.diffusion.contracts import (
    DiffusionGraphInputs,
    DiffusionPlacementPrior,
)
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion


def sample_diffusion_prior(
    model: GraphConditionedPlacementDiffusion,
    graph: DiffusionGraphInputs,
    samples: int = 8,
    steps: int = 16,
    seed: int | None = None,
) -> DiffusionPlacementPrior:
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
    for step in reversed(range(max(1, steps))):
        t = torch.full(
            (samples,),
            step,
            dtype=torch.long,
            device=graph.area.device,
        )
        out = model(graph, x_t, t)
        eps = out["eps_pred"]
        x_t = x_t - eps / float(max(steps, 1))
        pair_logits = out["pairwise_axis_logits"]
        quality = out["quality_pred"]
    centers = x_t[:, :, :2] * max(float(graph.scale), 1.0)
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
) -> GraphConditionedPlacementDiffusion:
    payload = torch.load(path, map_location=map_location)
    variant = str(payload.get("variant", "raw"))
    hidden_dim = int(payload.get("hidden_dim", 128))
    layers = int(payload.get("layers", 2))
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph,
        variant=variant,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model
