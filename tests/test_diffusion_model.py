from __future__ import annotations

import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.sampling import sample_diffusion_prior
from floorset_arch.parser import parse_instance


def _graph_inputs():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        torch.tensor([[0.0, 2.0, 1.0]]),
        torch.tensor([[5.0, 7.0]]),
        torch.zeros(3, 5),
        None,
    )
    return build_diffusion_graph_inputs(inst)


def test_raw_diffusion_forward_shapes():
    graph = _graph_inputs()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=24, layers=1
    )
    noisy = torch.zeros(2, 3, 3)
    timesteps = torch.tensor([4, 4], dtype=torch.long)

    out = model(graph, noisy, timesteps)

    assert out["eps_pred"].shape == (2, 3, 3)
    assert out["pairwise_axis_logits"].shape == (2, graph.pair_index.shape[0], 3)
    assert out["quality_pred"].shape == (2,)


def test_hgt_lite_diffusion_forward_shapes():
    graph = _graph_inputs()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="hgt_lite", hidden_dim=24, layers=1
    )
    noisy = torch.zeros(1, 3, 3)
    timesteps = torch.tensor([2], dtype=torch.long)

    out = model(graph, noisy, timesteps)

    assert out["eps_pred"].shape == (1, 3, 3)
    assert out["pairwise_axis_logits"].shape[1] == graph.pair_index.shape[0]


def test_sampling_returns_prior_without_anchor_guidance():
    graph = _graph_inputs()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(
        graph, variant="raw", hidden_dim=16, layers=1
    )

    prior = sample_diffusion_prior(model, graph, samples=2, steps=3, seed=11)

    assert prior.centers.shape == (2, 3, 2)
    assert prior.log_aspect.shape == (2, 3)
    assert prior.pair_index.shape == graph.pair_index.shape
    assert not hasattr(prior, "rect_priors")
