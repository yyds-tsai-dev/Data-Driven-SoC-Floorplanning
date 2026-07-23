import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partner"))

from physics_guidance_claude import (GuidanceContext, guidance_energy,
                                     rects_from_z)


def _ctx(n=2, boundary=(0, 0), groups=(0, 0), pairs=None, areas=(100.0, 100.0)):
    B = 1
    N = n
    area = torch.tensor([list(areas)], dtype=torch.float32)
    mask = torch.ones(B, N, dtype=torch.bool)
    known = torch.zeros(B, N, 4, dtype=torch.bool)
    scale = torch.sqrt(area.sum(dim=1)).clamp_min(1.0)
    p = pairs or []
    return GuidanceContext(
        area=area, mask=mask, known_mask=known, scale=scale,
        boundary_code=torch.tensor([list(boundary)], dtype=torch.long),
        group_id=torch.tensor([list(groups)], dtype=torch.long),
        pair_i=torch.tensor([a for a, _, _ in p], dtype=torch.long),
        pair_j=torch.tensor([b for _, b, _ in p], dtype=torch.long),
        pair_w=torch.tensor([w for _, _, w in p], dtype=torch.float32),
    )


def _z(xy_pairs, aspect=0.0):
    # xy in *normalized* units (z0 = x/S); aspect = log(w/h)
    z = torch.zeros(1, len(xy_pairs), 4)
    for i, (x, y) in enumerate(xy_pairs):
        z[0, i, 0] = x
        z[0, i, 1] = y
        z[0, i, 2] = aspect
    return z


def test_rects_match_z_to_rectangles_semantics():
    ctx = _ctx(areas=(100.0, 25.0))
    z = _z([(0.5, 0.25), (0.0, 0.0)], aspect=0.0)
    x, y, w, h = rects_from_z(z, ctx)
    s = float(ctx.scale[0])
    assert abs(float(x[0, 0]) - 0.5 * s) < 1e-5
    assert abs(float(w[0, 0]) - 10.0) < 1e-4   # sqrt(100*1)
    assert abs(float(h[0, 1]) - 5.0) < 1e-4    # sqrt(25/1)


def test_overlap_energy_pushes_blocks_apart():
    ctx = _ctx()
    # two 10x10 blocks at identical corner -> full overlap
    z = _z([(0.1, 0.1), (0.1, 0.1)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 1.0, 0.0, 0.0, 0.0)
    assert e.item() > 0.0
    g, = torch.autograd.grad(e, z)
    # symmetric stack: gradients on x must be opposite in sign or zero-sum
    assert abs(float(g[0, 0, 0] + g[0, 1, 0])) < 1e-5
    # separated blocks -> zero energy
    z2 = _z([(0.0, 0.0), (0.9, 0.9)])
    e2 = guidance_energy(z2, ctx, 1.0, 0.0, 0.0, 0.0)
    assert e2.item() < 1e-8


def test_boundary_energy_pulls_coded_block_to_wall():
    # block 0 coded left-wall (bit 1), block 1 free and further left
    ctx = _ctx(boundary=(1, 0))
    z = _z([(0.5, 0.3), (0.0, 0.0)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 0.0, 1.0, 0.0, 0.0)
    assert e.item() > 0.0
    g, = torch.autograd.grad(e, z)
    assert float(g[0, 0, 0]) > 0.0   # descent moves block 0 left (toward X0)
    # already at the left edge of the arrangement -> ~zero energy
    z2 = _z([(0.0, 0.3), (0.5, 0.0)])
    e2 = guidance_energy(z2, ctx, 0.0, 1.0, 0.0, 0.0)
    assert e2.item() < 1e-6


def test_hpwl_energy_pulls_connected_pair_together():
    ctx = _ctx(pairs=[(0, 1, 2.0)])
    z = _z([(0.0, 0.0), (0.8, 0.0)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 0.0, 0.0, 1.0, 0.0)
    g, = torch.autograd.grad(e, z)
    assert float(g[0, 0, 0]) < 0.0   # descent moves block 0 toward block 1
    assert float(g[0, 1, 0]) > 0.0


def test_padding_blocks_contribute_nothing():
    ctx = _ctx()
    ctx.mask[0, 1] = False
    z = _z([(0.1, 0.1), (0.1, 0.1)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 1.0, 1.0, 0.0, 0.0)
    assert e.item() < 1e-8            # only one real block: no overlap, boundary un-coded


from physics_guidance_claude import GuidanceConfig, guide_x0, make_guidance


def test_guide_x0_reduces_overlap_energy():
    ctx = _ctx()
    cfg = GuidanceConfig(k_steps=8, eta=0.05, t_gate=0.5)
    z = _z([(0.1, 0.1), (0.1, 0.1)])
    e0 = guidance_energy(z, ctx, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, cfg.w_group)
    z1 = guide_x0(z, ctx, cfg, data_time=0.9)
    e1 = guidance_energy(z1, ctx, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, cfg.w_group)
    assert e1.item() < e0.item()


def test_guide_x0_identity_outside_t_gate():
    ctx = _ctx()
    cfg = GuidanceConfig(t_gate=0.35)
    z = _z([(0.1, 0.1), (0.1, 0.1)])
    out = guide_x0(z, ctx, cfg, data_time=0.2)   # 0.2 < 1-0.35
    assert torch.equal(out, z)


def test_guide_x0_freezes_known_and_aspect_channels():
    ctx = _ctx()
    ctx.known_mask[0, 0, :] = True               # block 0 fully anchored
    cfg = GuidanceConfig(k_steps=4, guide_aspect=False, t_gate=1.0)
    z = _z([(0.1, 0.1), (0.1, 0.1)], aspect=0.3)
    out = guide_x0(z, ctx, cfg, data_time=1.0)
    assert torch.equal(out[0, 0], z[0, 0])                  # anchored block untouched
    torch.testing.assert_close(out[..., 2], z[..., 2])      # aspect frozen
    assert not torch.equal(out[0, 1, :2], z[0, 1, :2])      # free block moved


def test_guide_x0_respects_trust_radius():
    ctx = _ctx()
    cfg = GuidanceConfig(k_steps=1, eta=100.0, trust_radius=0.03, t_gate=1.0)
    z = _z([(0.1, 0.1), (0.1, 0.1)])
    out = guide_x0(z, ctx, cfg, data_time=1.0)
    assert float((out - z).abs().max()) <= 0.03 + 1e-6
