"""Contract tests for the DPM-Solver++(2M) direct sampler
(src/solver/direct_diffusion_model.py::sample_direct_dpmpp).

Covers: (1) steps=1 is bitwise-identical to DDIM (both collapse to a single
x0 readout), (2) both samplers track the same probability-flow ODE — the
few-step dpmpp endpoint is at least as close to a fine-step DDIM reference
as the same-step DDIM endpoint (order 2 vs order 1), (3) hard anchors are
exact in the endpoint, (4) the aspect channel respects the clamp, (5) NFE ==
steps, (6) masked-out nodes stay zero, (7) same generator seed is
deterministic.
"""
from types import SimpleNamespace

import torch

from diffusion_model import DiffusionSchedule
from direct_diffusion_model import sample_direct, sample_direct_dpmpp


class ZScaledV(torch.nn.Module):
    """A z-dependent v-prediction field so trajectories are non-trivial."""

    def __init__(self, z_dim=4, gain=0.1):
        super().__init__()
        self.config = SimpleNamespace(z_dim=z_dim, timesteps=1000)
        self.gain = gain
        self.calls = 0

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        self.calls += 1
        return self.gain * z * mask.unsqueeze(-1)


def _condition(batch=1, blocks=4, dtype=torch.float32):
    return {
        "node_feat": torch.zeros(batch, blocks, 1, dtype=dtype),
        "adj": torch.eye(blocks, dtype=dtype).expand(batch, -1, -1),
        "mask": torch.ones(batch, blocks, dtype=torch.bool),
        "rel_feat": torch.zeros(batch, blocks, blocks, 0, dtype=dtype),
    }


def _schedule():
    return DiffusionSchedule(1000, device="cpu")


def test_steps1_bitwise_matches_ddim():
    model = ZScaledV()
    cond = _condition()
    sched = _schedule()
    ref = sample_direct(model, cond, sched, steps=1,
                        generator=torch.Generator().manual_seed(7))
    got = sample_direct_dpmpp(model, cond, sched, steps=1,
                              generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


def test_few_step_error_not_worse_than_ddim():
    model = ZScaledV()
    cond = _condition()
    sched = _schedule()
    ref = sample_direct(model, cond, sched, steps=800,
                        generator=torch.Generator().manual_seed(3))
    ddim8 = sample_direct(model, cond, sched, steps=8,
                          generator=torch.Generator().manual_seed(3))
    dpmpp8 = sample_direct_dpmpp(model, cond, sched, steps=8,
                                 generator=torch.Generator().manual_seed(3))
    err_ddim = (ddim8 - ref).norm()
    err_dpmpp = (dpmpp8 - ref).norm()
    assert err_dpmpp <= err_ddim * 1.05


def test_anchors_exact_in_endpoint():
    model = ZScaledV()
    cond = _condition()
    sched = _schedule()
    z_known = torch.zeros(1, 4, 4)
    z_known[0, 0, 0] = 0.25
    z_known[0, 0, 1] = -0.5
    z_known[0, 1, 2] = 1.5
    known = torch.zeros(1, 4, 4, dtype=torch.bool)
    known[0, 0, 0] = True
    known[0, 0, 1] = True
    known[0, 1, 2] = True
    out = sample_direct_dpmpp(model, cond, sched, steps=6,
                              generator=torch.Generator().manual_seed(5),
                              z_known=z_known, known_mask=known)
    torch.testing.assert_close(out[known], z_known[known])


def test_aspect_channel_clamped():
    model = ZScaledV(gain=5.0)
    cond = _condition()
    sched = _schedule()
    out = sample_direct_dpmpp(model, cond, sched, steps=6,
                              generator=torch.Generator().manual_seed(9))
    assert out[..., 2].abs().max() <= 3.0 + 1e-6


def test_nfe_equals_steps():
    model = ZScaledV()
    cond = _condition()
    sched = _schedule()
    sample_direct_dpmpp(model, cond, sched, steps=10,
                        generator=torch.Generator().manual_seed(1))
    assert model.calls == 10


def test_masked_nodes_stay_zero():
    model = ZScaledV()
    cond = _condition(blocks=5)
    cond["mask"][0, 3:] = False
    sched = _schedule()
    out = sample_direct_dpmpp(model, cond, sched, steps=6,
                              generator=torch.Generator().manual_seed(13))
    assert out[0, 3:].abs().max() == 0.0


def test_same_seed_deterministic():
    model = ZScaledV()
    cond = _condition()
    sched = _schedule()
    a = sample_direct_dpmpp(model, cond, sched, steps=8,
                            generator=torch.Generator().manual_seed(21))
    b = sample_direct_dpmpp(model, cond, sched, steps=8,
                            generator=torch.Generator().manual_seed(21))
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
