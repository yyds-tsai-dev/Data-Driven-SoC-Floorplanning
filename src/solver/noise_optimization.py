"""Initial-noise optimization for the frozen flow-matching prior.

Freeze the flow model weights and, per sample, run projected gradient descent
on the *initial noise* z_T that seeds a short differentiable Euler flow. The
loss is a per-sample analytic layout energy (overlap / boundary / hpwl) so that
"what the frozen model naturally generates from the optimized noise" is a better
candidate than what it generates from an unstructured random draw.

Design notes
------------
* Convention. This straight-path flow integrates t=0 (noise) -> t=1 (data).
  The task's "z_T" is therefore the ``z`` at the *start* of the Euler loop.
* Differentiability. ``flow_matching_model.sample_flow`` is ``@torch.no_grad``
  and draws its own noise, so it cannot carry a gradient back to the seed. This
  module adds ``sample_flow_diff``: the identical numerics with autograd left on
  and the initial noise + known-channel noise passed in as fixed inputs. The
  known-channel imposition uses ``torch.where`` (differentiable) exactly like
  ``sample_flow``, so the two agree to floating point on the same draws.
* Anchors. At the final step ``t1 = steps/steps = 1.0`` the imposed schedule
  ``(1-t1)*known_noise + t1*z_known`` collapses to ``z_known`` exactly, so the
  output known channels equal ``z_known`` by construction regardless of the
  optimized noise. Known channels of z_T are additionally *frozen* (never
  stepped): their only downstream effect is via the imposed schedule, so a live
  gradient there would just drift a dead parameter.

Optimizer recipe (lineage: DNO 2312.11994 / ReNO / D-Flow)
----------------------------------------------------------
* Adam on the seed with a *per-sample unit-normalized* gradient, linear warmup
  then cosine decay of the learning rate, a hard iteration cap, and early
  stopping when the layout energy stops improving.
* Typical-set regularization is dual: (i) a soft chi_d radial log-density
  penalty ``-lambda[(d-1)log||z|| - ||z||^2/2]`` every iter (min at
  ||z||=sqrt(d-1)), and (ii) a hard per-(sample,channel) mean-0/std-1
  standardization of the free entries every ``proj_every`` iters and once at the
  end -- so the returned seed sits on the training manifold.
* Free-lunch selection: the *argmin-energy* point along the trajectory is
  returned per sample, not the last iterate.
* Per-sample energy uses ``guidance_energy_per_sample`` (each sample normalized
  by its own scale), never the batch-summed ``guidance_energy`` -- the batch
  total normalization was the guidance bug that let hpwl dominate. Because
  samples are independent, ``loss = e.sum()`` gives correct per-sample gradients.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional

import torch
import torch.utils.checkpoint

from flow_matching_model import endpoint_from_velocity
from physics_guidance import GuidanceContext, guidance_energy_per_sample


# --------------------------------------------------------------------------- #
# Differentiable flow sampler (numerically identical to sample_flow)
# --------------------------------------------------------------------------- #
def sample_flow_diff(
    model,
    condition: Mapping[str, torch.Tensor],
    steps: int,
    solver: str,
    z_init: torch.Tensor,
    z_known: torch.Tensor | None = None,
    known_mask: torch.Tensor | None = None,
    known_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-enabled twin of ``flow_matching_model.sample_flow``.

    Same Euler/Heun update, self-conditioning, and known-channel imposition as
    ``sample_flow`` -- but the initial noise ``z_init`` and (when anchoring) the
    fixed ``known_noise`` are supplied by the caller instead of drawn, and the
    ``@torch.no_grad`` guard is removed so gradients flow back to ``z_init``.
    Returns the clean endpoint z0 (``[B, N, z_dim]``).
    """
    if steps <= 0 or solver not in {"euler", "heun"}:
        raise ValueError("steps must be positive and solver must be euler or heun")

    node_feat = condition["node_feat"]
    mask = condition["mask"]
    batch, _blocks = mask.shape
    device = node_feat.device
    dtype = node_feat.dtype
    valid = mask.unsqueeze(-1)

    z = z_init * valid
    has_known = z_known is not None and known_mask is not None
    if has_known:
        if known_noise is None:
            raise ValueError("known_noise must be supplied when anchoring")
        z = torch.where(known_mask, known_noise, z) * valid

    def imposed_known(scalar_t: float) -> torch.Tensor:
        assert known_noise is not None and z_known is not None
        return (1.0 - scalar_t) * known_noise + scalar_t * z_known

    def velocity(state: torch.Tensor, scalar_t: float, self_cond=None) -> torch.Tensor:
        t_model = torch.full(
            (batch,),
            scalar_t * (model.config.timesteps - 1),
            device=device,
            dtype=dtype,
        )
        return model(
            state,
            t_model,
            node_feat,
            condition["adj"],
            mask,
            rel_feat=condition.get("rel_feat"),
            self_cond=self_cond,
        )

    self_condition = None
    for step in range(steps):
        t0 = step / steps
        t1 = (step + 1) / steps
        dt = t1 - t0
        v0 = velocity(z, t0, self_condition)
        # self-conditioning is an auxiliary input; detaching matches sample_flow
        # (and the training-time detach) and keeps the two numerically identical
        # while the primary z-path stays differentiable.
        self_condition = endpoint_from_velocity(
            z, v0, torch.full((batch,), t0, device=device, dtype=dtype)
        ).detach()
        self_condition[..., 2] = self_condition[..., 2].clamp(-3.0, 3.0)
        if has_known:
            self_condition = torch.where(known_mask, z_known, self_condition)

        if solver == "euler":
            z_next = z + dt * v0
        else:
            predicted = z + dt * v0
            if has_known:
                predicted = torch.where(known_mask, imposed_known(t1), predicted) * valid
            v1 = velocity(predicted, t1, self_condition)
            z_next = z + 0.5 * dt * (v0 + v1)

        if has_known:
            z_next = torch.where(known_mask, imposed_known(t1), z_next)
        z = z_next * valid

    return z


# --------------------------------------------------------------------------- #
# Differentiable DDIM sampler (numerically identical to sample_direct)
# --------------------------------------------------------------------------- #
def sample_direct_diff(
    model,
    cond: Mapping[str, torch.Tensor],
    schedule,
    steps: int,
    z_init: torch.Tensor,
    known_noise: torch.Tensor | None = None,
    z_known: torch.Tensor | None = None,
    known_mask: torch.Tensor | None = None,
    grad_checkpoint: bool = False,
) -> torch.Tensor:
    """Autograd-enabled twin of ``direct_diffusion_model.sample_direct`` (no guidance).

    Same v-prediction DDIM update, self-conditioning, per-step hard-anchor
    imposition, and aspect clamp -- but ``z_init`` and the per-step
    ``known_noise`` (``[steps, B, N, z_dim]``) are supplied by the caller instead
    of drawn, and the ``@torch.no_grad`` guard is removed so gradients flow back
    to ``z_init``. The in-place aspect clamp of ``sample_direct`` is expressed
    with ``torch.cat`` here (identical values, autograd-safe). Self-conditioning
    is detached (matches sample_direct's no_grad state, halves the unroll graph).
    Returns the clean endpoint z0 (``[B, N, z_dim]``).
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    node_feat = cond["node_feat"]
    mask = cond["mask"]
    rel_feat = cond.get("rel_feat")
    adj = cond.get("adj")
    device = node_feat.device
    B, N, _ = node_feat.shape
    valid = mask.unsqueeze(-1)
    has_known = z_known is not None and known_mask is not None
    if has_known and known_noise is None:
        raise ValueError("known_noise ([steps,B,N,z]) required when anchoring")

    def clamp_aspect(z0: torch.Tensor) -> torch.Tensor:
        return torch.cat([z0[..., :2], z0[..., 2:3].clamp(-3.0, 3.0), z0[..., 3:]], dim=-1)

    z = z_init * valid
    times = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()
    sc = None
    for idx in range(len(times)):
        t = torch.full((B,), int(times[idx].item()), device=device, dtype=torch.long)
        alpha, sigma = schedule.alpha_sigma(t)
        if has_known:
            z_imposed = alpha * z_known + sigma * known_noise[idx]
            z = torch.where(known_mask, z_imposed, z)

        def vel(z_in, _sc=sc, _t=t):
            return model(z_in, _t, node_feat, adj, mask, rel_feat=rel_feat, self_cond=_sc)

        if grad_checkpoint:
            v = torch.utils.checkpoint.checkpoint(vel, z, use_reentrant=False)
        else:
            v = vel(z)

        z0 = alpha * z - sigma * v
        z0 = clamp_aspect(z0)
        if has_known:
            z0 = torch.where(known_mask, z_known, z0)
        sc = z0.detach()
        eps = sigma * z + alpha * v
        if idx == len(times) - 1:
            z = z0
            break
        t_next = torch.full((B,), int(times[idx + 1].item()), device=device, dtype=torch.long)
        alpha_n, sigma_n = schedule.alpha_sigma(t_next)
        z = (alpha_n * z0 + sigma_n * eps) * valid
    return z * valid


def make_direct_sampler(model, cond, schedule, steps, z_known, known_mask,
                        known_noise, grad_checkpoint=False):
    """Bind a ``seed -> z0`` DDIM sampler for ``optimize_noise(sample_fn=...)``."""
    def _fn(seed: torch.Tensor) -> torch.Tensor:
        return sample_direct_diff(
            model, cond, schedule, steps, seed, known_noise=known_noise,
            z_known=z_known, known_mask=known_mask, grad_checkpoint=grad_checkpoint)
    return _fn


# --------------------------------------------------------------------------- #
# Config / result
# --------------------------------------------------------------------------- #
@dataclass
class NoiseOptConfig:
    rounds: int = 50            # hard iteration cap (DNO)
    lr: float = 0.05
    steps: int = 8              # flow Euler steps unrolled per iter
    solver: str = "euler"
    warmup: int = 10            # linear lr warmup iters, then cosine decay
    patience: int = 10          # early-stop: no layout-energy improvement
    chi_lambda: float = 0.01    # chi_d radial log-density penalty weight
    proj_every: int = 10        # channel-wise standardization cadence
    unit_grad: bool = True      # per-sample unit-normalized gradient
    w_overlap: float = 1.0
    w_boundary: float = 0.5
    w_hpwl: float = 0.1         # per-sample normalization fixed -> hpwl is safe
    w_group: float = 0.0

    @classmethod
    def from_env(cls) -> "NoiseOptConfig":
        def f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            rounds=int(f("PARTNER_NOISE_OPT_ROUNDS", 50)),
            lr=f("PARTNER_NOISE_OPT_LR", 0.05),
            steps=int(f("PARTNER_NOISE_OPT_STEPS", 8)),
            solver=os.environ.get("PARTNER_NOISE_OPT_SOLVER", "euler"),
            warmup=int(f("PARTNER_NOISE_OPT_WARMUP", 10)),
            patience=int(f("PARTNER_NOISE_OPT_PATIENCE", 10)),
            chi_lambda=f("PARTNER_NOISE_OPT_CHI_LAMBDA", 0.01),
            proj_every=int(f("PARTNER_NOISE_OPT_PROJ_EVERY", 10)),
            unit_grad=os.environ.get("PARTNER_NOISE_OPT_UNIT_GRAD", "1") != "0",
            w_overlap=f("PARTNER_NOISE_OPT_W_OVERLAP", 1.0),
            w_boundary=f("PARTNER_NOISE_OPT_W_BOUNDARY", 0.5),
            w_hpwl=f("PARTNER_NOISE_OPT_W_HPWL", 0.1),
            w_group=f("PARTNER_NOISE_OPT_W_GROUP", 0.0),
        )


@dataclass
class NoiseOptResult:
    z_T: torch.Tensor              # best-along-trajectory seed [B, N, z_dim]
    z0: torch.Tensor               # endpoint sampled from z_T   [B, N, z_dim]
    energy: torch.Tensor           # [iters+1, B] per-sample layout-energy trace
    best_iter: torch.Tensor        # [B] argmin iter per sample
    history: List[dict] = field(default_factory=list)  # per-iter scalar logs


# --------------------------------------------------------------------------- #
# Typical-set helpers
# --------------------------------------------------------------------------- #
def _free_mask(mask: torch.Tensor, known_mask: Optional[torch.Tensor],
               z_dim: int) -> torch.Tensor:
    """[B, N, z_dim] bool: valid block channels that are not hard-anchored."""
    valid = mask.unsqueeze(-1).expand(-1, -1, z_dim)
    if known_mask is None:
        return valid.clone()
    return valid & (~known_mask)


def _free_norm(z: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm of the free part -> [B]."""
    B = z.shape[0]
    return (z * free).reshape(B, -1).pow(2).sum(dim=1).clamp_min(1e-12).sqrt()


def _standardize_channels(z: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """Per-(sample, channel) mean-0/std-1 over free entries (D-Flow style).

    Standardizes only the free entries of each channel; known/pad entries are
    left untouched (the caller restores/zeros them). Puts each free channel back
    onto the iid-N(0,1) training distribution, so the total free norm returns to
    ~sqrt(#free) (typical set)."""
    cnt = free.sum(dim=1).clamp_min(1.0)                 # [B, z_dim]
    mean = (z * free).sum(dim=1) / cnt                   # [B, z_dim]
    centered = (z - mean.unsqueeze(1)) * free
    var = centered.pow(2).sum(dim=1) / cnt
    std = var.sqrt().clamp_min(1e-6)
    z_std = (z - mean.unsqueeze(1)) / std.unsqueeze(1)
    return torch.where(free, z_std, z)


def _chi_reg(free_norm: torch.Tensor, d: torch.Tensor, lam: float) -> torch.Tensor:
    """chi_d radial log-density penalty (per sample), minimized at sqrt(d-1)."""
    r = free_norm.clamp_min(1e-6)
    return -lam * ((d - 1.0) * torch.log(r) - 0.5 * r * r)


# --------------------------------------------------------------------------- #
# Core: optimize the initial noise
# --------------------------------------------------------------------------- #
def optimize_noise(
    flow_model,
    cond: Mapping[str, torch.Tensor],
    ctx: Optional[GuidanceContext],
    z_T0: torch.Tensor,
    z_known: torch.Tensor | None,
    known_mask: torch.Tensor | None,
    cfg: NoiseOptConfig,
    energy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    sample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_result: bool = False,
):
    """Projected-gradient descent on the initial noise against a layout energy.

    ``sample_fn(seed) -> z0`` maps a seed to a clean endpoint; when omitted a
    flow sampler is built from ``flow_model``/``cond``. Pass
    :func:`make_direct_sampler` for the DDIM path. Returns the best-along-
    trajectory seed ``z_T_opt`` (and optionally a :class:`NoiseOptResult` with
    the endpoint, per-sample energy trace, argmin iter, and per-iter logs).
    """
    mask = cond["mask"]
    z_dim = z_T0.shape[-1]
    valid = mask.unsqueeze(-1)
    device = z_T0.device
    B = z_T0.shape[0]

    default_energy = energy_fn is None
    if default_energy:
        if ctx is None:
            raise ValueError("ctx is required when energy_fn is not supplied")

        def energy_fn(z0: torch.Tensor) -> torch.Tensor:  # noqa: E306
            return guidance_energy_per_sample(
                z0, ctx, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, cfg.w_group)

    def components(z0: torch.Tensor):
        """(overlap, boundary, hpwl) per-sample terms for logging (default)."""
        if not default_energy:
            return None
        ov = guidance_energy_per_sample(z0, ctx, cfg.w_overlap, 0.0, 0.0, 0.0)
        bd = guidance_energy_per_sample(z0, ctx, 0.0, cfg.w_boundary, 0.0, 0.0)
        hp = guidance_energy_per_sample(z0, ctx, 0.0, 0.0, cfg.w_hpwl, 0.0)
        return ov, bd, hp

    free = _free_mask(mask, known_mask, z_dim)
    d_free = free.reshape(B, -1).sum(dim=1).clamp_min(1.0)     # [B]
    sqrt_d = d_free.sqrt()
    has_known = z_known is not None and known_mask is not None

    z_T = (z_T0.detach().clone() * valid)
    z_init_free = (z_T * free).detach().clone()               # for cos-similarity
    known_noise = (z_T * known_mask).detach().clone() if has_known else None
    with torch.no_grad():
        z_T.data = _standardize_channels(z_T, free)           # start on manifold
        if has_known:
            z_T.data = torch.where(known_mask, known_noise, z_T.data)
        z_T.data = z_T.data * valid
    z_T.requires_grad_(True)

    optim = torch.optim.Adam([z_T], lr=cfg.lr)

    if sample_fn is None:
        def sample(seed: torch.Tensor) -> torch.Tensor:
            return sample_flow_diff(
                flow_model, cond, cfg.steps, cfg.solver, seed,
                z_known=z_known, known_mask=known_mask, known_noise=known_noise)
    else:
        sample = sample_fn

    def lr_at(it: int) -> float:
        if it < cfg.warmup:
            return cfg.lr * (it + 1) / max(1, cfg.warmup)
        prog = (it - cfg.warmup) / max(1, cfg.rounds - cfg.warmup)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    best_e = torch.full((B,), float("inf"), device=device)
    best_zT = z_T.detach().clone()
    best_z0 = torch.zeros_like(z_T)
    best_iter = torch.zeros(B, dtype=torch.long, device=device)
    trace: List[torch.Tensor] = []
    history: List[dict] = []
    since_improve = 0
    prev_mean_best = float("inf")

    for it in range(cfg.rounds):
        optim.zero_grad(set_to_none=True)
        z0 = sample(z_T)
        e = energy_fn(z0)                                     # [B] layout energy
        trace.append(e.detach())

        # free-lunch: keep the per-sample argmin point along the trajectory
        with torch.no_grad():
            improved = e < best_e
            if improved.any():
                best_e = torch.where(improved, e, best_e)
                best_zT[improved] = z_T.detach()[improved]
                best_z0[improved] = z0.detach()[improved]
                best_iter[improved] = it

        fnorm = _free_norm(z_T, free)                        # differentiable
        reg = _chi_reg(fnorm, d_free, cfg.chi_lambda)
        loss = e.sum() + reg.sum()
        loss.backward()

        with torch.no_grad():
            g = z_T.grad
            g = g * free if g is not None else torch.zeros_like(z_T)
            grad_norm = g.reshape(B, -1).norm(dim=1)         # pre-normalize
            if cfg.unit_grad:
                g = g / grad_norm.clamp_min(1e-12).view(B, *([1] * (z_T.ndim - 1)))
            z_T.grad = g

        for grp in optim.param_groups:
            grp["lr"] = lr_at(it)
        optim.step()

        with torch.no_grad():
            z_T.data = z_T.data * valid
            if has_known:
                z_T.data = torch.where(known_mask, known_noise, z_T.data)
            if cfg.proj_every > 0 and (it + 1) % cfg.proj_every == 0:
                z_T.data = _standardize_channels(z_T.data, free)
                if has_known:
                    z_T.data = torch.where(known_mask, known_noise, z_T.data)
                z_T.data = z_T.data * valid

        # per-iter scalar log (coordinator item 5)
        with torch.no_grad():
            ratio = (_free_norm(z_T, free) / sqrt_d)
            cos = torch.nn.functional.cosine_similarity(
                (z_T * free).reshape(B, -1), z_init_free.reshape(B, -1), dim=1)
            log = dict(
                iter=it, lr=round(lr_at(it), 5),
                energy_mean=float(e.mean()),
                grad_norm_mean=float(grad_norm.mean()),
                z_ratio_mean=float(ratio.mean()),
                z_ratio_min=float(ratio.min()), z_ratio_max=float(ratio.max()),
                cos_z_init_mean=float(cos.mean()),
                sphere_alarm=bool((ratio < 0.9).any() or (ratio > 1.1).any()),
            )
            comps = components(z0)
            if comps is not None:
                ov, bd, hp = comps
                log.update(overlap_mean=float(ov.mean()),
                           boundary_mean=float(bd.mean()),
                           hpwl_mean=float(hp.mean()))
            history.append(log)

        # early stopping on layout-energy plateau (DNO)
        mean_best = float(best_e.mean())
        if mean_best < prev_mean_best - 1e-9:
            prev_mean_best = mean_best
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    # return the argmin seed, standardized onto the manifold, with its endpoint
    with torch.no_grad():
        best_zT = _standardize_channels(best_zT, free)
        if has_known:
            best_zT = torch.where(known_mask, known_noise, best_zT)
        best_zT = best_zT * valid
        best_z0 = sample(best_zT)
        trace.append(energy_fn(best_z0).detach())

    if not return_result:
        return best_zT
    energy = torch.stack(trace, dim=0)                        # [iters+1, B]
    return best_zT, NoiseOptResult(z_T=best_zT, z0=best_z0, energy=energy,
                                   best_iter=best_iter, history=history)
