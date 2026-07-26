"""Straight-path conditional flow-matching helpers and samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class FlowSample:
    z: torch.Tensor
    nfe: int
    solver: str


def _broadcast_time(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], *([1] * (z.ndim - 1))).to(z)


def flow_path(
    z0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the linear noise-to-data path and its constant velocity."""
    tau = _broadcast_time(t, z0)
    return (1.0 - tau) * noise + tau * z0, z0 - noise


def endpoint_from_velocity(
    z_t: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the data endpoint of a straight path from its velocity."""
    tau = _broadcast_time(t, z_t)
    return z_t + (1.0 - tau) * velocity


def endpoint_snr_weight(t: torch.Tensor, gamma: float) -> torch.Tensor:
    """Straight-path SNR analogue for weighting the endpoint (x0) loss.

    On the linear path ``z_t = (1-t)*noise + t*z0`` the implied endpoint error
    equals ``(1-t) * (velocity error)``, so an unweighted endpoint MSE
    implicitly scales as ``(1-t)**2`` and is dominated by the high-noise
    (low-t) regime where the endpoint is least reliable.  Multiplying by
    ``t**2 / (1-t)**2`` -- the data/noise coefficient-variance ratio, the flow
    analogue of the diffusion SNR ``alpha**2 / sigma**2`` -- cancels that
    ``(1-t)**2`` amplification and leaves a net ``t**2`` emphasis on the
    underlying velocity error, matching the geometry losses.  The clamp mirrors
    ``direct_train_v2``'s min-SNR-gamma and keeps the weight finite as t -> 1.

    MEASURED NET-NEGATIVE (v2, 2026-07-27).  The diffusion analogy is a category
    error: in diffusion ``x0_hat = (z_t - sigma*eps_hat)/alpha`` *amplifies* the
    residual by ``1/alpha`` as the noise level rises, which is the pathology
    min-SNR repairs.  On the straight path ``x0_hat = z_t + (1-t)*v_hat`` has
    residual ``(1-t)*(v_hat - v)`` -- attenuated, never amplified -- so there is
    no pathology to fix.  Applying this weight moved 84% of the endpoint
    gradient mass out of ``t < 0.5`` and cost ~5% coordinate accuracy and ~21%
    overlap quality.  Kept for ablation only; not a default.
    """
    flat = t.reshape(-1).to(torch.float32)
    ratio = (flat * flat) / ((1.0 - flat) * (1.0 - flat)).clamp_min(1e-8)
    return ratio.clamp(max=gamma)


def endpoint_time_weight(t: torch.Tensor, mode: str, gamma: float) -> torch.Tensor:
    """Return the per-sample time weight for the endpoint (x0) loss.

    ``none`` (v1 and the v2.1 default) leaves the endpoint loss unweighted, so
    it acts as a dense ``(1-t)``-profiled coordinate-space auxiliary spanning
    the whole path.  ``snr`` selects the disproved min-SNR analogue above.
    """
    if mode == "none":
        return torch.ones_like(t.reshape(-1), dtype=torch.float32)
    if mode == "snr":
        return endpoint_snr_weight(t, gamma)
    raise ValueError("x0 time weighting must be 'none' or 'snr'")


def sample_flow_t(
    n: int,
    device,
    generator=None,
    term_prob: float = 0.0,
    term_band: float = 0.02,
) -> torch.Tensor:
    """Sample flow times in ``[0, 1)``, optionally with mass near the endpoint.

    With probability ``term_prob`` a sample is drawn from ``[1 - term_band, 1)``
    instead of uniformly, covering the terminal neighbourhood that the Heun
    corrector (and Euler's final substep) evaluate at but uniform ``t`` never
    reaches.  ``term_prob = 0`` (the v2.1 default) is exactly uniform ``t``.

    Measured in v2: the band does improve every metric at ``t >= 0.97``, but it
    spends 10% of the sample budget where the endpoint is already ~10x more
    accurate than at low t, and removes 10% of the training mass from the
    ``t**2``-gated geometry losses whose useful range is ``t < 0.95``.  It also
    inflates the logged ``v`` and ``mib`` averages, because terminal samples sit
    at the velocity ceiling and at ``mib``'s non-zero ground-truth floor.  Off
    by default until an ablation shows it pays for itself.
    """
    t = torch.rand((n,), device=device, generator=generator)
    if term_prob > 0.0 and term_band > 0.0:
        term = torch.rand((n,), device=device, generator=generator) < term_prob
        t_term = 1.0 - torch.rand((n,), device=device, generator=generator) * term_band
        t = torch.where(term, t_term, t)
    return t


@torch.no_grad()
def sample_flow(
    model,
    condition: Mapping[str, torch.Tensor],
    steps: int,
    solver: str,
    generator=None,
    z_known: torch.Tensor | None = None,
    known_mask: torch.Tensor | None = None,
) -> FlowSample:
    """Integrate a conditional velocity field with Euler or Heun updates."""
    if steps <= 0 or solver not in {"euler", "heun"}:
        raise ValueError("steps must be positive and solver must be euler or heun")

    node_feat = condition["node_feat"]
    mask = condition["mask"]
    batch, blocks = mask.shape
    device = node_feat.device
    dtype = node_feat.dtype
    valid = mask.unsqueeze(-1)
    z = torch.randn(
        (batch, blocks, model.config.z_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ) * valid
    has_known = z_known is not None and known_mask is not None
    known_noise = None
    if has_known:
        known_noise = torch.randn(
            z.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        z = torch.where(known_mask, known_noise, z) * valid

    nfe = 0
    self_condition = None

    def imposed_known(scalar_t: float) -> torch.Tensor:
        assert known_noise is not None
        assert z_known is not None
        return (1.0 - scalar_t) * known_noise + scalar_t * z_known

    def velocity(state: torch.Tensor, scalar_t: float, self_cond=None) -> torch.Tensor:
        nonlocal nfe
        t_model = torch.full(
            (batch,),
            scalar_t * (model.config.timesteps - 1),
            device=device,
            dtype=dtype,
        )
        nfe += 1
        return model(
            state,
            t_model,
            node_feat,
            condition["adj"],
            mask,
            rel_feat=condition.get("rel_feat"),
            self_cond=self_cond,
        )

    for step in range(steps):
        t0 = step / steps
        t1 = (step + 1) / steps
        dt = t1 - t0
        v0 = velocity(z, t0, self_condition)
        self_condition = endpoint_from_velocity(
            z,
            v0,
            torch.full((batch,), t0, device=device, dtype=dtype),
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

    return FlowSample(z=z, nfe=nfe, solver=solver)
