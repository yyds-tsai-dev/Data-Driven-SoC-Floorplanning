#!/usr/bin/env python3
"""v2 trainer for the direct-prediction model — run on a second machine.

Deltas vs direct_train_claude.py (v1), each targeting a measured v1 gap:

  1. Block-count loss weighting  exp((n-60)/60), clipped [0.5, 3].
     v1's error is tail-heavy (p90 = 2.7x median) exactly on the large
     cases, and the contest weights cases by exp(n/12).  Batches are
     same-n (file-shuffle keeps a batch inside one 112-layout file), so
     this weights whole batches — mild enough for Adam.
  2. Low-noise oversampling: 50% of timesteps drawn from [0, T/4).
     Sampling precision lives at low t; uniform t spends half the compute
     on coarse structure the model already knows.
  3. Direct HPWL loss (b2b + p2b, exact contest wirelength on the decoded
     rects, hinged at the GT value): relu(hpwl - hpwl_gt)/hpwl_gt, gated
     by alpha_bar^2 like the other geometry losses.  v1 only imitates GT
     coordinates; this also tells the model WHICH deviations matter.
  4. Capacity: d_model 640, 14 layers, 10 heads (~115M params).  v1's
     shallow late-stage slope (-0.002 pos_l1 / 100k steps) suggests a
     capacity floor, not a data limit (1M samples, 16 epochs).
     EMA 0.9998 (slower, matching the bigger model), lr 8e-5, warmup 4k.

Feature set / z-repr / sampler are IDENTICAL to v1 on purpose: a v2
checkpoint drops into the same inference path (direct_eval_claude,
my_opt_claude via DIRECT_CKPT env) with zero code changes and zero
train/test-skew risk.

Second-machine setup:
  git clone <repo> && cd FloorSet/iccad2026contest
  # dataset (~24 GB) auto-downloads on first run via lite_dataset
  python3 direct_train_v2_claude.py --checkpoint-dir checkpoints/direct_v2 \
      --amp --batch-size 12 --max-steps 800000 --num-workers 4
  # resume: append  --resume latest
  # bigger GPU (>=24GB, A100/4090): --batch-size 24 --lr 1.1e-4

Bring the result back:  copy checkpoints/direct_v2/latest.pt to this
machine and either point DIRECT_CKPT=/path/to/latest.pt when running
iccad2026_evaluate, or drop it into checkpoints/direct_v1/.
Evaluate a checkpoint:  python3 direct_eval_claude.py <ckpt> [--e2e 88 92 99]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import direct_train_claude as V1
from diffusion_data import fp_sol_to_z0, z_to_rectangles
from direct_model_claude import known_z_channels
from diffusion_train import known_target_positions_from_fp


def hpwl_pair(rects: torch.Tensor, b2b: torch.Tensor, p2b: torch.Tensor,
              pins: torch.Tensor) -> torch.Tensor:
    """Exact contest HPWL (b2b + p2b) per sample, fully vectorized."""
    B, N, _ = rects.shape
    cx = rects[..., 0] + 0.5 * rects[..., 2]
    cy = rects[..., 1] + 0.5 * rects[..., 3]

    i = b2b[..., 0]
    j = b2b[..., 1]
    w = b2b[..., 2]
    ok = ((i >= 0) & (i < N) & (j >= 0) & (j < N)).to(rects.dtype)
    w = w.clamp_min(0.0) * ok
    ii = i.long().clamp(0, N - 1)
    jj = j.long().clamp(0, N - 1)
    hx = (torch.gather(cx, 1, ii) - torch.gather(cx, 1, jj)).abs()
    hy = (torch.gather(cy, 1, ii) - torch.gather(cy, 1, jj)).abs()
    hp = (w * (hx + hy)).sum(dim=1)

    Np = pins.shape[1]
    p = p2b[..., 0]
    blk = p2b[..., 1]
    w2 = p2b[..., 2]
    pi = p.long().clamp(0, max(Np - 1, 0))
    px = torch.gather(pins[..., 0], 1, pi)
    py = torch.gather(pins[..., 1], 1, pi)
    ok2 = ((p >= 0) & (p < Np) & (blk >= 0) & (blk < N) & (px != -1)).to(rects.dtype)
    w2 = w2.clamp_min(0.0) * ok2
    bi = blk.long().clamp(0, N - 1)
    hx2 = (px - torch.gather(cx, 1, bi)).abs()
    hy2 = (py - torch.gather(cy, 1, bi)).abs()
    hp = hp + (w2 * (hx2 + hy2)).sum(dim=1)
    return hp


def train_step_v2(model, ema, schedule, batch, args, rng, amp_dtype, amp_on):
    area, b2b, p2b, pins, cons, _tree, fp, _metrics = batch
    if args.augment:
        pins, cons, fp = V1.augment_batch(area, pins, cons, fp, rng)
    target_positions = known_target_positions_from_fp(fp, cons)
    z0, mask, scale = fp_sol_to_z0(fp, area, z_repr="xyaspect")
    cond = V1.fast_condition(
        area, b2b, p2b, pins, cons,
        target_positions=target_positions,
        relation_feat_dim=model.config.relation_feat_dim,
        node_feat_dim=model.config.node_feat_dim,
    )
    B = area.shape[0]
    device = area.device
    T = schedule.timesteps

    # low-noise oversampling
    t_hi = torch.randint(0, T, (B,), device=device, generator=rng)
    t_lo = torch.randint(0, max(T // 4, 1), (B,), device=device, generator=rng)
    pick = torch.rand(B, device=device, generator=rng) < args.low_t_frac
    t = torch.where(pick, t_lo, t_hi)

    noise = torch.randn(z0.shape, device=device, generator=rng)
    alpha, sigma = schedule.alpha_sigma(t)
    z_t = (alpha * z0 + sigma * noise) * mask.unsqueeze(-1)
    v_target = (alpha * noise - sigma * z0) * mask.unsqueeze(-1)
    z_known, known = known_z_channels(area, cons, target_positions, scale)

    sc = None
    if model.config.self_conditioning and \
            float(torch.rand((), generator=rng, device=device)) < args.self_cond_prob:
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=amp_dtype, enabled=amp_on):
            v1_ = model(z_t, t, cond["node_feat"], cond["adj"], mask,
                        rel_feat=cond["rel_feat"], self_cond=None)
        v1_ = v1_.float()
        sc = (alpha * z_t - sigma * v1_)
        sc[..., 2] = sc[..., 2].clamp(-3, 3)
        sc = torch.where(known, z_known, sc).detach()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
        v = model(z_t, t, cond["node_feat"], cond["adj"], mask,
                  rel_feat=cond["rel_feat"], self_cond=sc)
    v = v.float()

    mw = mask.unsqueeze(-1).to(v.dtype)
    v_loss = ((v - v_target).pow(2) * mw).sum() / mw.sum().clamp_min(1) / v.shape[-1]

    z0_hat = alpha * z_t - sigma * v
    snr = (alpha.pow(2) / sigma.pow(2).clamp_min(1e-8)).view(B)
    w_snr = snr.clamp(max=args.min_snr_gamma).view(B, 1, 1)
    code = cons[..., 4].long() if cons.shape[-1] > 4 else torch.zeros_like(area).long()
    node_w = torch.where((code != 0) & mask,
                         torch.full_like(area, args.bnd_node_weight),
                         torch.ones_like(area))
    chan_w = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device).view(1, 1, 4)
    l = F.smooth_l1_loss(z0_hat, z0, reduction="none", beta=0.02)
    l = l * chan_w * node_w.unsqueeze(-1) * mw * w_snr
    x0_loss = l.sum() / (node_w.unsqueeze(-1) * mw * chan_w).sum().clamp_min(1)

    ab2 = (alpha.view(B) ** 4)
    rects = z_to_rectangles(z0_hat, area, target_positions=target_positions,
                            constraints=cons, z_repr="xyaspect")
    gt_rects = torch.stack([fp[..., 2], fp[..., 3], fp[..., 0], fp[..., 1]], dim=-1)
    ov = (V1.overlap_fraction(rects, mask, scale) * ab2).mean()
    bd = (V1.boundary_touch(rects, cons, mask, scale) * ab2).mean()
    cg = (V1.cluster_gap(rects, gt_rects, cons, mask, scale) * ab2).mean()
    mb = (V1.mib_aspect(rects, cons, mask) * ab2).mean()

    # contest-metric alignment: hinge on GT wirelength
    hp_pred = hpwl_pair(rects, b2b, p2b, pins)
    with torch.no_grad():
        hp_gt = hpwl_pair(gt_rects, b2b, p2b, pins).clamp_min(1.0)
    hp_loss = (F.relu(hp_pred - hp_gt) / hp_gt * ab2).mean()

    # contest weighting: big cases carry exp(n/12) of the score
    n_blocks = mask.sum(dim=1).to(v.dtype)
    w_n = torch.exp((n_blocks - 60.0) / 60.0).clamp(0.5, 3.0).mean()

    loss = w_n * (args.v_loss_weight * v_loss
                  + args.x0_loss_weight * x0_loss
                  + args.overlap_loss_weight * ov
                  + args.boundary_loss_weight * bd
                  + args.cluster_loss_weight * cg
                  + args.mib_loss_weight * mb
                  + args.hpwl_loss_weight * hp_loss)

    with torch.no_grad():
        pe = ((z0_hat[..., :2] - z0[..., :2]).abs().sum(-1) * mask).sum() \
            / mask.sum().clamp_min(1)
    return {"loss": loss, "v": v_loss.detach(), "x0": x0_loss.detach(),
            "ov": ov.detach(), "bd": bd.detach(), "cg": cg.detach(),
            "mib": mb.detach(), "pos_l1": pe.detach(),
            "hp": hp_loss.detach()}


def main():
    # reuse v1's argument parser / training loop; override the deltas
    argv = sys.argv[1:]

    def has(flag):
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    defaults = {
        "--d-model": "640", "--layers": "14", "--heads": "10",
        "--lr": "8e-5", "--warmup": "4000", "--ema-decay": "0.9998",
        "--node-feat-dim": "32",
        "--max-steps": "800000", "--checkpoint-dir": "checkpoints/direct_v2",
    }
    for k, val in defaults.items():
        if not has(k):
            argv += [k, val]
    sys.argv = [sys.argv[0]] + argv

    # v2-only knobs, injected into the parsed args
    import argparse
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--hpwl-loss-weight", type=float, default=0.30)
    extra.add_argument("--low-t-frac", type=float, default=0.5)
    known, rest = extra.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest

    orig_parse = V1.parse_args

    def parse_with_extras():
        args = orig_parse()
        args.hpwl_loss_weight = known.hpwl_loss_weight
        args.low_t_frac = known.low_t_frac
        return args

    V1.parse_args = parse_with_extras
    V1.train_step = train_step_v2
    V1.main()


if __name__ == "__main__":
    main()
