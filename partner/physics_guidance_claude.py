"""Training-free physics-guided sampling: analytic energies over predicted
layouts (z-space), used by the Direct DDIM and Flow samplers.

Standalone by design: mirrors the xyaspect decode of
``diffusion_data.z_to_rectangles`` and the boundary-code semantics of
``my_opt_claude._viol_est`` (self-referential arrangement bbox), but does
not import either (z_to_rectangles is non-differentiable w.r.t. our use:
it hard-overwrites anchor channels).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class GuidanceConfig:
    k_steps: int = 5
    eta: float = 0.05
    t_gate: float = 0.35            # active when data_time >= 1 - t_gate
    w_overlap: float = 1.0
    w_boundary: float = 0.5
    w_hpwl: float = 0.1
    w_group: float = 0.0
    guide_aspect: bool = False
    trust_radius: float = 0.05      # max |Δz| per inner step (normalized units)

    @classmethod
    def from_env(cls) -> "GuidanceConfig":
        def f(name, default):
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default
        return cls(
            k_steps=int(f("PGUIDE_K", 5)),
            eta=f("PGUIDE_ETA", 0.05),
            t_gate=f("PGUIDE_TGATE", 0.35),
            w_overlap=f("PGUIDE_W_OVERLAP", 1.0),
            w_boundary=f("PGUIDE_W_BOUNDARY", 0.5),
            w_hpwl=f("PGUIDE_W_HPWL", 0.1),
            w_group=f("PGUIDE_W_GROUP", 0.0),
            guide_aspect=os.environ.get("PGUIDE_ASPECT") == "1",
            trust_radius=f("PGUIDE_TRUST", 0.05),
        )


@dataclass
class GuidanceContext:
    area: torch.Tensor           # [B,N] target areas (pad -> any; masked out)
    mask: torch.Tensor           # [B,N] bool
    known_mask: torch.Tensor     # [B,N,4] bool hard-anchor channels
    scale: torch.Tensor          # [B]
    boundary_code: torch.Tensor  # [B,N] long bitmask 1=L 2=R 4=T 8=B
    group_id: torch.Tensor       # [B,N] long (0 = none)
    pair_i: torch.Tensor         # [P] long
    pair_j: torch.Tensor         # [P] long
    pair_w: torch.Tensor         # [P] float

    def expand(self, batch: int) -> "GuidanceContext":
        e = lambda t: t.expand(batch, *t.shape[1:])
        return GuidanceContext(
            area=e(self.area), mask=e(self.mask), known_mask=e(self.known_mask),
            scale=self.scale.expand(batch), boundary_code=e(self.boundary_code),
            group_id=e(self.group_id),
            pair_i=self.pair_i, pair_j=self.pair_j, pair_w=self.pair_w,
        )


def rects_from_z(z: torch.Tensor, ctx: GuidanceContext):
    """Differentiable mirror of z_to_rectangles' xyaspect branch."""
    s = ctx.scale.view(-1, 1)
    x = z[..., 0] * s
    y = z[..., 1] * s
    area = torch.where(ctx.mask, ctx.area.clamp_min(1e-6),
                       torch.ones_like(ctx.area))
    aspect = torch.exp(z[..., 2].clamp(-3.0, 3.0))
    w = torch.sqrt(area * aspect).clamp_min(1e-6)
    h = torch.sqrt(area / aspect).clamp_min(1e-6)
    return x, y, w, h


def guidance_energy(z: torch.Tensor, ctx: GuidanceContext,
                    w_overlap: float, w_boundary: float,
                    w_hpwl: float, w_group: float) -> torch.Tensor:
    x, y, w, h = rects_from_z(z, ctx)
    m = ctx.mask.to(z.dtype)
    x1, y1 = x + w, y + h
    diag = ctx.scale.view(-1, 1).clamp_min(1.0)
    e = z.new_zeros(())

    if w_overlap:
        # signed-distance pairwise overlap area (soft-block differentiable)
        ox = (torch.minimum(x1.unsqueeze(2), x1.unsqueeze(1))
              - torch.maximum(x.unsqueeze(2), x.unsqueeze(1))).clamp_min(0.0)
        oy = (torch.minimum(y1.unsqueeze(2), y1.unsqueeze(1))
              - torch.maximum(y.unsqueeze(2), y.unsqueeze(1))).clamp_min(0.0)
        pm = m.unsqueeze(2) * m.unsqueeze(1)
        ov = ox * oy * pm
        ov = ov - torch.diag_embed(torch.diagonal(ov, dim1=1, dim2=2))
        e = e + w_overlap * 0.5 * ov.sum() / (diag.squeeze(1) ** 2).sum()

    if w_boundary and int(ctx.boundary_code.max()) > 0:
        # frame = arrangement bbox, detached: guidance moves blocks toward
        # the frame instead of collapsing the frame onto the block
        big = 10.0 * float(diag.max())
        X0 = torch.where(ctx.mask, x, torch.full_like(x, big)).min(dim=1, keepdim=True).values.detach()
        X1 = torch.where(ctx.mask, x1, torch.full_like(x1, -big)).max(dim=1, keepdim=True).values.detach()
        Y0 = torch.where(ctx.mask, y, torch.full_like(y, big)).min(dim=1, keepdim=True).values.detach()
        Y1 = torch.where(ctx.mask, y1, torch.full_like(y1, -big)).max(dim=1, keepdim=True).values.detach()
        code = ctx.boundary_code
        gb = z.new_zeros(x.shape)
        gb = gb + torch.where(code & 1 > 0, (x - X0).clamp_min(0.0), torch.zeros_like(x))
        gb = gb + torch.where(code & 2 > 0, (X1 - x1).clamp_min(0.0), torch.zeros_like(x))
        gb = gb + torch.where(code & 4 > 0, (Y1 - y1).clamp_min(0.0), torch.zeros_like(x))
        gb = gb + torch.where(code & 8 > 0, (y - Y0).clamp_min(0.0), torch.zeros_like(x))
        e = e + w_boundary * (gb * m / diag).sum() / m.sum().clamp_min(1.0)

    if w_hpwl and ctx.pair_i.numel():
        cx, cy = x + 0.5 * w, y + 0.5 * h
        dx = (cx[:, ctx.pair_i] - cx[:, ctx.pair_j]).abs()
        dy = (cy[:, ctx.pair_i] - cy[:, ctx.pair_j]).abs()
        e = e + w_hpwl * (ctx.pair_w * (dx + dy) / diag).sum() \
            / ctx.pair_w.sum().clamp_min(1e-6)

    if w_group and int(ctx.group_id.max()) > 0:
        cx, cy = x + 0.5 * w, y + 0.5 * h
        for gid in torch.unique(ctx.group_id):
            if int(gid) <= 0:
                continue
            gm = (ctx.group_id == gid) & ctx.mask
            if int(gm.sum()) < 2:
                continue
            gx = (cx * gm).sum(dim=1, keepdim=True) / gm.sum(dim=1, keepdim=True).clamp_min(1)
            gy = (cy * gm).sum(dim=1, keepdim=True) / gm.sum(dim=1, keepdim=True).clamp_min(1)
            e = e + w_group * (((cx - gx) ** 2 + (cy - gy) ** 2) * gm / diag ** 2).sum() \
                / gm.sum().clamp_min(1)
    return e


def guide_x0(z0: torch.Tensor, ctx: GuidanceContext, cfg: GuidanceConfig,
             data_time: float) -> torch.Tensor:
    if data_time < 1.0 - cfg.t_gate or cfg.k_steps <= 0:
        return z0
    strength = data_time * data_time      # endpoint estimates reliable near clean end
    z = z0.detach()
    # Tiny deterministic per-node offset to break min/max ties in the overlap
    # energy (exactly-coincident blocks otherwise yield an exact-zero
    # subgradient and never separate). Does not alter the stored z, only the
    # point at which the gradient is evaluated each inner step.
    n_nodes = z0.shape[1]
    tie_break = (1e-6 * torch.arange(n_nodes, device=z0.device,
                                     dtype=z0.dtype)).view(1, n_nodes, 1)
    for _ in range(cfg.k_steps):
        z = z.detach().requires_grad_(True)
        zt = z + tie_break
        e = guidance_energy(zt, ctx, cfg.w_overlap, cfg.w_boundary,
                            cfg.w_hpwl, cfg.w_group)
        (g,) = torch.autograd.grad(e, z)
        g = g.masked_fill(ctx.known_mask, 0.0)
        if not cfg.guide_aspect:
            g = g.clone()
            g[..., 2:] = 0.0
        step = (cfg.eta * strength * g).clamp(-cfg.trust_radius, cfg.trust_radius)
        z = (z - step).detach()
        z = torch.cat([z[..., :2],
                       z[..., 2:3].clamp(-3.0, 3.0), z[..., 3:]], dim=-1)
        z = torch.where(ctx.known_mask, z0, z)
        z = z * ctx.mask.unsqueeze(-1)
    return z


def make_guidance(ctx: GuidanceContext, cfg: GuidanceConfig):
    def _fn(z0: torch.Tensor, data_time: float) -> torch.Tensor:
        return guide_x0(z0, ctx, cfg, data_time)
    return _fn


def build_context(area_target: torch.Tensor, constraints: torch.Tensor,
                  b2b: torch.Tensor, scale: torch.Tensor,
                  known_mask: torch.Tensor) -> GuidanceContext:
    mask = area_target > 0
    c = constraints
    ncol = c.shape[-1]
    boundary = c[..., 4].round().long() if ncol > 4 else torch.zeros_like(mask, dtype=torch.long)
    group = c[..., 3].round().long() if ncol > 3 else torch.zeros_like(mask, dtype=torch.long)
    n = area_target.shape[1]
    w = b2b[:n, :n]
    iu = torch.triu_indices(n, n, offset=1)
    vals = w[iu[0], iu[1]] + w[iu[1], iu[0]]
    nz = vals > 0
    return GuidanceContext(
        area=area_target, mask=mask, known_mask=known_mask,
        scale=scale.reshape(-1), boundary_code=boundary, group_id=group,
        pair_i=iu[0][nz], pair_j=iu[1][nz], pair_w=vals[nz].float(),
    )
