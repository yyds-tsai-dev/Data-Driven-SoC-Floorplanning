from __future__ import annotations

from pathlib import Path

import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.sampling import (
    DiffusionGuidanceConfig,
    _layout_guidance_loss,
    _sampling_timestep_schedule,
    apply_layout_guidance,
    sample_diffusion_prior,
)
from floorset_arch.diffusion.training import DiffusionLossConfig
from floorset_arch.parser import parse_instance


def _graph_inputs():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.tensor([[0.0, 1.0, 3.0], [1.0, 2.0, 2.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        None,
    )
    return build_diffusion_graph_inputs(inst)


class _ZeroEpsModel(torch.nn.Module):
    variant = "raw"

    def __init__(self):
        super().__init__()
        self.timesteps: list[int] = []

    def forward(self, graph, x_t, timesteps):
        self.timesteps.extend(int(value) for value in timesteps.detach().cpu().tolist())
        return {
            "eps_pred": torch.zeros_like(x_t),
            "pairwise_axis_logits": torch.zeros(
                x_t.shape[0],
                graph.pair_index.shape[0],
                3,
                dtype=x_t.dtype,
                device=x_t.device,
            ),
            "quality_pred": torch.zeros(x_t.shape[0], dtype=x_t.dtype, device=x_t.device),
        }


def test_sampling_timestep_schedule_spans_training_noise_steps():
    steps = _sampling_timestep_schedule(max_steps=1000, steps=4, device=torch.device("cpu"))

    assert steps.tolist() == [999, 666, 333, 0]


def test_sample_diffusion_prior_uses_training_schedule_timesteps():
    graph = _graph_inputs()
    model = _ZeroEpsModel()

    sample_diffusion_prior(
        model,
        graph,
        samples=1,
        steps=4,
        seed=3,
        loss_config=DiffusionLossConfig(max_steps=1000, noise_schedule="cosine"),
        guidance_config=DiffusionGuidanceConfig(enabled=False),
    )

    assert model.timesteps == [999, 666, 333, 0]


def test_layout_guidance_step_reduces_overlap_and_constraint_proxy():
    graph = _graph_inputs()
    x0 = torch.tensor(
        [
            [
                [0.50, 0.50, 0.0],
                [0.55, 0.50, 0.0],
                [1.20, 0.50, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    config = DiffusionGuidanceConfig(
        enabled=True,
        step_size=0.15,
        overlap_weight=2.0,
        bbox_weight=0.01,
        net_weight=0.02,
        cluster_weight=0.5,
        boundary_weight=0.2,
        mib_weight=0.5,
    )

    before = _layout_guidance_loss(graph, x0, quality_pred=None, config=config)
    guided = apply_layout_guidance(graph, x0, quality_pred=None, config=config)
    after = _layout_guidance_loss(graph, guided, quality_pred=None, config=config)

    assert after.item() < before.item()


def test_optimizer_defaults_to_schedule_aligned_sampling_budget():
    text = Path("src/floorset_arch/optimizer.py").read_text(encoding="utf-8")

    assert 'FLOORSET_DIFFUSION_STEPS", "64"' in text
