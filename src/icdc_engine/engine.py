"""The IC/DC engine: checkpoint -> K legal layouts per instance.

One object so the trainer, the dump script and the tests all drive the *same*
path: conditioning -> differentiable few-step sampling -> exact-area decode ->
TFDL -> hard-legal rectangles.  Nothing here is allowed to produce a layout the
downstream rung-(-1) admission gate would reject; `verify_hard_legal` is the
contract check, deliberately written against the same predicates
`src/solver/layout_refiner._guard_hard_ok` uses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PARTNER = Path(__file__).resolve().parents[1] / "solver"
if str(PARTNER) not in sys.path:
    sys.path.insert(0, str(PARTNER))

from . import energy as EN
from . import tfdl as T
from .sampler import expand_cond, sample_differentiable


def load_model(ckpt_path: str, device="cuda", use_ema: bool = True):
    from direct_diffusion_model import DirectDenoiser, DirectModelConfig
    from diffusion_model import DiffusionSchedule
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DirectModelConfig(**ck["model_config"])
    model = DirectDenoiser(cfg).to(device)
    state = ck["ema"] if (use_ema and "ema" in ck) else ck["model"]
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()})
    schedule = DiffusionSchedule(cfg.timesteps, device=device)
    return model, schedule, cfg, ck


def build_cond(batch: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    from direct_diffusion_train import fast_condition
    return fast_condition(
        batch["area"], batch["b2b"], batch["p2b"], batch["pins"],
        batch["cons"], target_positions=batch["tp"],
        relation_feat_dim=cfg.relation_feat_dim,
        node_feat_dim=cfg.node_feat_dim)


def known_channels(batch: Dict[str, torch.Tensor]):
    from direct_diffusion_model import known_z_channels
    return known_z_channels(batch["area"], batch["cons"], batch["tp"],
                            batch["scale"])


def z_to_legal(z: torch.Tensor, batch: Dict[str, torch.Tensor],
               exact: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z -> (legal rects, raw decoded rects, per-block pin drift)."""
    rects = EN.decode_rects(z, batch["area"], batch["cons"], batch["tp"],
                            batch["scale"])
    mask = batch["area"] > 0
    pin = EN.preplaced_mask(batch["cons"], batch["tp"], batch["area"])
    code = (batch["cons"][..., 4].long() if batch["cons"].shape[-1] > 4
            else torch.zeros_like(batch["area"]).long())
    legal, drift = T.tfdl(rects, mask, pin, pin_xy=batch["tp"][..., :2],
                          boundary_code=code, exact=exact)
    return legal, rects, drift


@torch.no_grad()
def sample_bank(model, schedule, batch: Dict[str, torch.Tensor], cfg,
                K: int = 6, steps: int = 2, seed: int = 0,
                device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """K legal layouts per instance, in float64 for the dump.

    float64 is not cosmetic: `_guard_hard_ok` rejects a pair overlapping by
    more than 1e-7 on both axes, and coordinates run to ~600, so float32's
    ~7e-5 rounding on an abutment equality would fail the gate the whole bank
    exists to pass.  The model still runs in float32; only the decode and the
    compaction are promoted.
    """
    cond = build_cond(batch, cfg)
    z_known, known = known_channels(batch)
    condK = expand_cond(cond, K)
    zkK = z_known.repeat_interleave(K, dim=0)
    knK = known.repeat_interleave(K, dim=0)
    gen = torch.Generator(device=device).manual_seed(seed)
    z = sample_differentiable(model, condK, schedule, steps=steps,
                              generator=gen, z_known=zkK, known_mask=knK)
    batch64 = {k: (v.to(torch.float64) if torch.is_floating_point(v) else v)
               for k, v in batch.items()}
    bK = {k: (v.repeat_interleave(K, dim=0) if torch.is_tensor(v) else v)
          for k, v in batch64.items()}
    legal, _raw, drift = z_to_legal(z.to(torch.float64), bK, exact=True)
    B = batch["area"].shape[0]
    N = batch["area"].shape[1]
    return legal.view(B, K, N, 4), drift.amax(dim=(1, 2)).view(B, K)


# ---------------------------------------------------------------------------
# contract check against rung-(-1)
# ---------------------------------------------------------------------------
def verify_hard_legal(P: np.ndarray, area: np.ndarray, cons: np.ndarray,
                      tp: np.ndarray, area_tol: float = 0.01,
                      dim_tol: float = 1e-5) -> Dict[str, bool]:
    """Same predicates as `layout_refiner._guard_hard_ok`, on raw arrays.

    Returns the per-predicate verdict so a failure says *which* invariant
    broke instead of just "rejected".
    """
    n = P.shape[0]
    out = {}
    out["shape"] = P.shape == (n, 4) and bool(np.isfinite(P).all())
    out["positive"] = bool((P[:, 2] > 0).all() and (P[:, 3] > 0).all())
    fixed = cons[:, 0] != 0
    pre = cons[:, 1] != 0
    soft = ~(fixed | pre)
    out["area"] = bool(
        soft.sum() == 0 or
        (np.abs(P[soft, 2] * P[soft, 3] - area[soft])
         / np.maximum(area[soft], 1e-9) <= area_tol).all())
    hard = fixed | pre
    out["dims"] = bool(
        hard.sum() == 0 or
        ((np.abs(P[hard, 2] - tp[hard, 2]) <= dim_tol).all()
         and (np.abs(P[hard, 3] - tp[hard, 3]) <= dim_tol).all()))
    out["preplaced"] = bool(
        pre.sum() == 0 or
        ((np.abs(P[pre, 0] - tp[pre, 0]) <= dim_tol).all()
         and (np.abs(P[pre, 1] - tp[pre, 1]) <= dim_tol).all()))
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > 1e-7) & (oy > 1e-7)
    np.fill_diagonal(bad, False)
    out["overlap"] = bool(not bad.any())
    out["ok"] = all(out.values())
    return out
