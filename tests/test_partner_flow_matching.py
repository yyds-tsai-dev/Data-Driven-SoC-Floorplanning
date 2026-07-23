from types import SimpleNamespace

import pytest
import torch

from flow_matching_claude import endpoint_from_velocity, flow_path, sample_flow


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.register_buffer("velocity", velocity)
        self.config = SimpleNamespace(z_dim=velocity.shape[-1], timesteps=1000)

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        return self.velocity.expand_as(z) * mask.unsqueeze(-1)


class RecordingVelocity(ConstantVelocity):
    def __init__(self, velocity):
        super().__init__(velocity)
        self.states = []
        self.times = []

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        self.states.append(z.detach().clone())
        self.times.append(t.detach().clone())
        return super().forward(z, t, node_feat, adj, mask, rel_feat, self_cond)


def _condition(batch=1, blocks=2, dtype=torch.float32):
    return {
        "node_feat": torch.zeros(batch, blocks, 1, dtype=dtype),
        "adj": torch.eye(blocks, dtype=dtype).expand(batch, -1, -1),
        "mask": torch.tensor([[True, False]]).expand(batch, -1),
        "rel_feat": torch.zeros(batch, blocks, blocks, 0, dtype=dtype),
    }


def test_flow_path_endpoint_reconstruction_is_exact():
    z0 = torch.tensor([[[2.0, -1.0]]])
    noise = torch.tensor([[[0.5, 3.0]]])
    t = torch.tensor([0.25])
    z_t, target = flow_path(z0, noise, t)
    torch.testing.assert_close(endpoint_from_velocity(z_t, target, t), z0)


def test_euler_reports_one_nfe_per_step_and_masks_padding():
    model = ConstantVelocity(torch.ones(1, 2, 4))
    sample = sample_flow(
        model,
        _condition(),
        steps=4,
        solver="euler",
        generator=torch.Generator().manual_seed(1),
    )
    assert sample.nfe == 4
    assert sample.solver == "euler"
    assert torch.equal(sample.z[:, 1], torch.zeros_like(sample.z[:, 1]))


def test_heun_reports_two_nfe_per_step():
    model = ConstantVelocity(torch.ones(1, 2, 4))
    sample = sample_flow(
        model,
        _condition(),
        steps=4,
        solver="heun",
        generator=torch.Generator().manual_seed(1),
    )
    assert sample.nfe == 8
    assert sample.solver == "heun"


def test_known_channels_are_exact_at_final_time():
    model = ConstantVelocity(torch.zeros(1, 2, 4))
    known = torch.tensor([[[7.0, 5.0, 1.0, 0.0], [0.0] * 4]])
    known_mask = torch.tensor([[[True, True, True, False], [False] * 4]])
    sample = sample_flow(
        model,
        _condition(),
        steps=4,
        solver="euler",
        generator=torch.Generator().manual_seed(1),
        z_known=known,
        known_mask=known_mask,
    )
    torch.testing.assert_close(sample.z[known_mask], known[known_mask])


@pytest.mark.parametrize(("steps", "solver"), [(0, "euler"), (-1, "euler"), (1, "rk4")])
def test_invalid_sampler_arguments_raise_value_error(steps, solver):
    model = ConstantVelocity(torch.ones(1, 2, 4))
    with pytest.raises(ValueError, match="steps must be positive"):
        sample_flow(model, _condition(), steps=steps, solver=solver)


def test_seeded_sampling_is_deterministic():
    model = ConstantVelocity(torch.full((1, 2, 4), 0.25))
    first = sample_flow(
        model,
        _condition(),
        steps=3,
        solver="heun",
        generator=torch.Generator().manual_seed(17),
    )
    second = sample_flow(
        model,
        _condition(),
        steps=3,
        solver="heun",
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(first.z, second.z)


def test_sampler_preserves_condition_floating_dtype():
    model = ConstantVelocity(torch.ones(1, 2, 4, dtype=torch.float64))
    sample = sample_flow(
        model,
        _condition(dtype=torch.float64),
        steps=2,
        solver="euler",
        generator=torch.Generator().manual_seed(3),
    )
    assert sample.z.device.type == "cpu"
    assert sample.z.dtype is torch.float64


def test_known_channels_follow_one_noise_path_through_heun_stages():
    model = RecordingVelocity(torch.zeros(1, 2, 4))
    known = torch.tensor([[[7.0, 5.0, 1.0, 0.0], [0.0] * 4]])
    known_mask = torch.tensor([[[True, True, True, False], [False] * 4]])
    seed = 29
    generator = torch.Generator().manual_seed(seed)
    sample_flow(
        model,
        _condition(),
        steps=2,
        solver="heun",
        generator=generator,
        z_known=known,
        known_mask=known_mask,
    )

    expected_generator = torch.Generator().manual_seed(seed)
    torch.randn((1, 2, 4), generator=expected_generator)
    known_noise = torch.randn((1, 2, 4), generator=expected_generator)
    for state, time in zip(model.states, model.times, strict=True):
        scalar_time = time[0] / (model.config.timesteps - 1)
        expected = (1.0 - scalar_time) * known_noise + scalar_time * known
        torch.testing.assert_close(state[known_mask], expected[known_mask])


def test_self_condition_aspect_channel_is_clamped():
    """Sampler must clamp the aspect channel of self-conditioning to match
    the training-side clamp (flow_train_claude.py), else train/sample skew."""
    captured = []

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(z_dim=4, timesteps=1000)

        def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
            if self_cond is not None:
                captured.append(self_cond.detach().clone())
            # huge velocity so the endpoint estimate's aspect channel exceeds 3
            v = torch.zeros_like(z)
            v[..., 2] = 100.0
            return v * mask.unsqueeze(-1)

    model = RecordingModel()
    sample_flow(model, _condition(), steps=3, solver="euler",
                generator=torch.Generator().manual_seed(0))
    assert captured, "self_cond was never passed back to the model"
    for sc in captured:
        assert sc[..., 2].abs().max() <= 3.0 + 1e-6
