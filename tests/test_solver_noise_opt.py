"""Mechanism tests for initial-noise optimization (src/solver/noise_optimization.py).

Covers: (1) sample_flow_diff is numerically identical to the frozen no_grad
sample_flow on the same draws, (2) optimize_noise drives a toy energy down while
holding z_T on the typical-set sphere and freezing known channels, (3) anchors
are exact in the sampled endpoint.
"""
from types import SimpleNamespace

import torch

from flow_matching_model import sample_flow
from noise_optimization import (
    NoiseOptConfig,
    make_direct_sampler,
    optimize_noise,
    sample_direct_diff,
    sample_flow_diff,
)


class ZScaledVelocity(torch.nn.Module):
    """A z-dependent velocity field so trajectories (and gradients) are real."""

    def __init__(self, z_dim=4, gain=0.1):
        super().__init__()
        self.config = SimpleNamespace(z_dim=z_dim, timesteps=1000)
        self.gain = gain

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        return self.gain * z * mask.unsqueeze(-1)


class ZeroVelocity(torch.nn.Module):
    """v=0 => the Euler endpoint equals the free part of the seed exactly."""

    def __init__(self, z_dim=4):
        super().__init__()
        self.config = SimpleNamespace(z_dim=z_dim, timesteps=1000)

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        return torch.zeros_like(z) * mask.unsqueeze(-1)


def _condition(batch=1, blocks=4, z_dim=4, dtype=torch.float32):
    return {
        "node_feat": torch.zeros(batch, blocks, 1, dtype=dtype),
        "adj": torch.eye(blocks, dtype=dtype).expand(batch, -1, -1),
        "mask": torch.ones(batch, blocks, dtype=torch.bool),
        "rel_feat": torch.zeros(batch, blocks, blocks, 0, dtype=dtype),
    }


def _draw_like_sample_flow(shape, has_known, seed):
    """Reproduce sample_flow's draw order: z_raw first, then known_noise."""
    gen = torch.Generator().manual_seed(seed)
    z_raw = torch.randn(shape, generator=gen)
    known_noise = torch.randn(shape, generator=gen) if has_known else None
    return z_raw, known_noise


# --------------------------------------------------------------------------- #
# 1. differentiable twin == frozen sampler
# --------------------------------------------------------------------------- #
def test_sample_flow_diff_matches_sample_flow_no_known():
    model = ZScaledVelocity()
    cond = _condition()
    shape = (1, 4, 4)
    z_raw, _ = _draw_like_sample_flow(shape, has_known=False, seed=11)

    ref = sample_flow(model, cond, steps=6, solver="euler",
                      generator=torch.Generator().manual_seed(11))
    got = sample_flow_diff(model, cond, steps=6, solver="euler", z_init=z_raw)
    torch.testing.assert_close(got, ref.z)


def test_sample_flow_diff_matches_sample_flow_with_known():
    model = ZScaledVelocity()
    cond = _condition()
    shape = (1, 4, 4)
    z_known = torch.zeros(shape)
    z_known[:, 0, :3] = torch.tensor([0.7, 0.5, 1.0])
    known_mask = torch.zeros(shape, dtype=torch.bool)
    known_mask[:, 0, :3] = True

    z_raw, known_noise = _draw_like_sample_flow(shape, has_known=True, seed=29)
    ref = sample_flow(model, cond, steps=5, solver="euler",
                      generator=torch.Generator().manual_seed(29),
                      z_known=z_known, known_mask=known_mask)
    got = sample_flow_diff(model, cond, steps=5, solver="euler", z_init=z_raw,
                           z_known=z_known, known_mask=known_mask,
                           known_noise=known_noise)
    torch.testing.assert_close(got, ref.z)


def test_sample_flow_diff_carries_gradient_to_seed():
    model = ZScaledVelocity()
    cond = _condition()
    z_init = torch.randn(1, 4, 4, requires_grad=True)
    z0 = sample_flow_diff(model, cond, steps=4, solver="euler", z_init=z_init)
    z0.pow(2).sum().backward()
    assert z_init.grad is not None
    assert torch.isfinite(z_init.grad).all()
    assert z_init.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# 1b. differentiable DDIM twin == frozen sample_direct
# --------------------------------------------------------------------------- #
def _direct_draws(shape, steps, has_known, seed):
    """Reproduce sample_direct's draw order: z first, then one per step."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(shape, generator=gen)
    kn = None
    if has_known:
        kn = torch.stack([torch.randn(shape, generator=gen) for _ in range(steps)])
    return z, kn


def test_sample_direct_diff_matches_sample_direct_no_known():
    from direct_diffusion_model import sample_direct
    from diffusion_model import DiffusionSchedule
    model = ZScaledVelocity()
    cond = _condition()
    sched = DiffusionSchedule(1000)
    steps = 6
    z, _ = _direct_draws((1, 4, 4), steps, has_known=False, seed=41)
    ref = sample_direct(model, cond, sched, steps=steps,
                        generator=torch.Generator().manual_seed(41))
    got = sample_direct_diff(model, cond, sched, steps=steps, z_init=z)
    torch.testing.assert_close(got, ref)


def test_sample_direct_diff_matches_sample_direct_with_known():
    from direct_diffusion_model import sample_direct
    from diffusion_model import DiffusionSchedule
    model = ZScaledVelocity()
    cond = _condition()
    sched = DiffusionSchedule(1000)
    steps = 5
    z_known = torch.zeros(1, 4, 4)
    z_known[:, 0, :3] = torch.tensor([0.7, 0.5, 1.0])
    known_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    known_mask[:, 0, :3] = True
    z, kn = _direct_draws((1, 4, 4), steps, has_known=True, seed=53)
    ref = sample_direct(model, cond, sched, steps=steps,
                        generator=torch.Generator().manual_seed(53),
                        z_known=z_known, known_mask=known_mask)
    got = sample_direct_diff(model, cond, sched, steps=steps, z_init=z,
                             known_noise=kn, z_known=z_known, known_mask=known_mask)
    torch.testing.assert_close(got, ref)


def test_sample_direct_diff_gradient_and_checkpoint_agree():
    from diffusion_model import DiffusionSchedule
    model = ZScaledVelocity()
    cond = _condition()
    sched = DiffusionSchedule(1000)
    z, _ = _direct_draws((1, 4, 4), 4, has_known=False, seed=7)
    z1 = z.clone().requires_grad_(True)
    z2 = z.clone().requires_grad_(True)
    out1 = sample_direct_diff(model, cond, sched, 4, z1, grad_checkpoint=False)
    out2 = sample_direct_diff(model, cond, sched, 4, z2, grad_checkpoint=True)
    torch.testing.assert_close(out1, out2)
    out1.pow(2).sum().backward()
    out2.pow(2).sum().backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()
    torch.testing.assert_close(z1.grad, z2.grad)


def test_optimize_noise_direct_sampler_runs_and_anchors():
    from diffusion_model import DiffusionSchedule
    model = ZeroVelocity()
    cond = _condition(blocks=4)
    sched = DiffusionSchedule(1000)
    steps = 6
    z_T0 = torch.randn(1, 4, 4)
    z_known = torch.zeros(1, 4, 4)
    z_known[:, 0, :3] = torch.tensor([0.3, 0.2, 0.4])
    known_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    known_mask[:, 0, :3] = True
    kn = torch.randn(steps, 1, 4, 4)
    sampler = make_direct_sampler(model, cond, sched, steps, z_known, known_mask, kn)

    def energy_fn(z0):
        return (z0 ** 2).reshape(z0.shape[0], -1).sum(dim=1)

    cfg = NoiseOptConfig(rounds=6, lr=0.1, steps=steps)
    _z, res = optimize_noise(None, cond, ctx=None, z_T0=z_T0, z_known=z_known,
                             known_mask=known_mask, cfg=cfg, energy_fn=energy_fn,
                             sample_fn=sampler, return_result=True)
    # per-step anchoring makes the endpoint known channels exactly z_known
    torch.testing.assert_close(res.z0[known_mask], z_known[known_mask])
    assert torch.isfinite(res.energy).all()


# --------------------------------------------------------------------------- #
# 2. optimize_noise mechanism on a toy energy
# --------------------------------------------------------------------------- #
def test_optimize_noise_drives_toy_energy_down_on_sphere():
    torch.manual_seed(0)
    model = ZeroVelocity()          # z0_free == z_T_free exactly
    cond = _condition(blocks=4)
    z_dim = 4
    z_T0 = torch.randn(1, 4, z_dim)
    # toy energy: squared distance to a fixed on-sphere target. With the seed
    # held on the typical-set sphere this is minimized by rotating z toward the
    # target, so the trajectory energy must fall.
    u = torch.randn(1, 4, z_dim)
    u = u / u.reshape(1, -1).norm() * (4 * z_dim) ** 0.5

    def energy_fn(z0):
        return 0.5 * (z0 - u).pow(2).reshape(z0.shape[0], -1).sum(dim=1)

    cfg = NoiseOptConfig(rounds=40, lr=0.05, steps=4, solver="euler",
                         warmup=5, patience=20)
    z_T_opt, res = optimize_noise(
        model, cond, ctx=None, z_T0=z_T0, z_known=None, known_mask=None,
        cfg=cfg, energy_fn=energy_fn, return_result=True,
    )
    trace = res.energy[:, 0]        # [iters+1]
    assert torch.isfinite(trace).all()
    assert torch.isfinite(z_T_opt).all()
    # mechanism-health gate: the trajectory argmin is >= 30% below the start
    drop = (trace[0] - trace.min()) / trace[0].abs().clamp_min(1e-6)
    assert drop >= 0.30, f"energy dropped only {float(drop):.3f}"
    # returned seed is the argmin point, not the last iterate
    assert res.energy[res.best_iter[0], 0] <= trace[0]
    # per-iter logs recorded (coordinator item 5)
    assert res.history and "grad_norm_mean" in res.history[0]
    # z_T stayed on the typical-set sphere: ||z_free|| ~ sqrt(#free)
    d_free = 4 * z_dim
    norm = z_T_opt.reshape(1, -1).norm()
    assert abs(float(norm) - d_free ** 0.5) <= 0.05 * d_free ** 0.5


def test_optimize_noise_freezes_known_channels_and_anchors_endpoint():
    torch.manual_seed(1)
    model = ZeroVelocity()
    cond = _condition(blocks=4)
    z_dim = 4
    z_T0 = torch.randn(1, 4, z_dim)

    z_known = torch.zeros(1, 4, z_dim)
    z_known[:, 0, :3] = torch.tensor([0.6, 0.4, 0.5])
    known_mask = torch.zeros(1, 4, z_dim, dtype=torch.bool)
    known_mask[:, 0, :3] = True

    def energy_fn(z0):
        # push everything toward origin; free part fights sphere, known frozen
        return (z0 ** 2).reshape(z0.shape[0], -1).sum(dim=1)

    cfg = NoiseOptConfig(rounds=8, lr=0.1, steps=6, solver="euler")
    z_T_opt, res = optimize_noise(
        model, cond, ctx=None, z_T0=z_T0, z_known=z_known,
        known_mask=known_mask, cfg=cfg, energy_fn=energy_fn, return_result=True,
    )
    # known channels of the SEED are frozen at their initial value
    torch.testing.assert_close(z_T_opt[known_mask], z_T0[known_mask])
    # known channels of the sampled ENDPOINT are exactly z_known
    torch.testing.assert_close(res.z0[known_mask], z_known[known_mask])


def test_optimize_noise_default_signature_returns_seed_only():
    model = ZeroVelocity()
    cond = _condition(blocks=3)
    z_T0 = torch.randn(1, 3, 4)

    def energy_fn(z0):
        return (z0 ** 2).reshape(z0.shape[0], -1).sum(dim=1)

    out = optimize_noise(model, cond, ctx=None, z_T0=z_T0, z_known=None,
                         known_mask=None, cfg=NoiseOptConfig(rounds=3),
                         energy_fn=energy_fn)
    assert isinstance(out, torch.Tensor)
    assert out.shape == z_T0.shape
