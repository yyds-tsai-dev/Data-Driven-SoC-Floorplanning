#!/usr/bin/env python3
"""Trainer for the direct-prediction coordinate model (DirectDenoiser).

Recipe (see TRAIN_DIRECT_MODEL_claude.md for the full rationale):
  * v-prediction + min-SNR(gamma=5)-weighted x0 loss — precision pressure
    where it matters (low noise) without destabilizing high-noise training.
  * Self-conditioning: 50% of steps feed the model its own no-grad x0
    estimate, matching how the sampler runs.
  * EMA (0.9995) — the sampling weights.
  * Per-node loss weights: boundary-tagged blocks x2 (they define the bbox
    edges the evaluator scores against).
  * Auxiliary geometry losses (overlap fraction / boundary touch / cluster
    gap / MIB aspect) fully vectorized and weighted by alpha_bar(t)^2 so
    they only act where the x0 estimate is meaningful.  The Python-loop
    losses of diffusion_train.py were the throughput killer (80k steps in
    2 days); this trainer targets ~10x more steps per day.
  * D4 augmentation (mirror x/y, transpose) with consistent remapping of
    pins and boundary-tag bits — teaches the symmetry instead of memorizing
    an orientation.

Shared-GPU etiquette (the L4 is shared; both caps are on by default):
  * --vram-fraction 0.45  — hard cap via set_per_process_memory_fraction,
    leaving >50% of the 23 GiB for other users (allocations beyond the cap
    OOM us instead of squeezing them).
  * --gpu-util-cap 0.75   — after every optimizer step, sleep in proportion
    to the busy time so average GPU utilization stays below 75%.
  * Interruption-safe: SIGTERM / SIGINT / SIGHUP checkpoint interrupt.pt +
    latest.pt before exiting; a bare relaunch auto-resumes from latest.pt
    (only kill -9 can lose progress, at most --save-every steps, default
    1000).  step_*.pt files are pruned to the last --keep-recent plus
    permanent snapshots every --snapshot-every steps, so frequent saving
    cannot fill the shared disk.

Usage (1+ week run on the shared L4 — survives kills, auto-resumes):
  nohup ./train_forever_claude.sh checkpoints/direct_v1 \
      --batch-size 16 --max-steps 600000 --num-workers 4 \
      > direct_v1.log 2>&1 &

One-off / manual:
  python3 direct_diffusion_train.py --checkpoint-dir checkpoints/direct_v1 \
      --amp --batch-size 16 --max-steps 600000 --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (FloorplanDatasetLite, get_training_dataloader,
                                train_floorplan_collate)
from diffusion_data import (batch_to_device, build_relation_features,
                            fp_sol_to_z0, layout_scale, normalized_adjacency,
                            valid_block_mask, z_to_rectangles)
from diffusion_model import DiffusionSchedule
# diffusion_train is a training-only dependency (known_target_positions_from_fp
# is used by train_step alone), imported lazily so inference-side consumers of
# fast_condition do not pull in the v1 diffusion trainer.
from direct_diffusion_model import (DirectDenoiser, DirectModelConfig, EMA,
                                 known_z_channels)


# ---------------------------------------------------------------------------
# Vectorized replacement for diffusion_data.build_condition.  The original
# iterates over every b2b/p2b edge in Python with .item() calls — on CUDA
# tensors that is one device sync per edge (~9 s per batch, and the reason
# the old trainer managed only ~0.5 it/s).  This version produces identical
# outputs using scatter_add / gather only.
# ---------------------------------------------------------------------------
def dense_adj_fast(area: torch.Tensor, b2b: torch.Tensor) -> torch.Tensor:
    B, N = area.shape
    i = b2b[..., 0]
    j = b2b[..., 1]
    w = b2b[..., 2]
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N) & torch.isfinite(w) & (w > 0)
    w = torch.where(ok, w, torch.zeros_like(w)).to(area.dtype)
    ii = i.long().clamp(0, N - 1)
    jj = j.long().clamp(0, N - 1)
    adj = area.new_zeros(B, N * N)
    adj.scatter_add_(1, ii * N + jj, w)
    adj.scatter_add_(1, jj * N + ii, w)
    return adj.view(B, N, N)


def fast_condition(area, b2b, p2b, pins, cons, target_positions,
                   relation_feat_dim=9, node_feat_dim=26):
    device = area.device
    dtype = area.dtype
    B, N = area.shape
    mask = valid_block_mask(area)
    scale = layout_scale(area, mask).clamp_min(1.0)
    safe_area = torch.where(mask, area.clamp_min(1e-6), torch.ones_like(area))

    adj_raw = dense_adj_fast(area, b2b)
    adj = normalized_adjacency(adj_raw, mask)
    b2b_degree = (adj_raw > 0).sum(dim=-1).to(dtype)
    b2b_weight = adj_raw.sum(dim=-1)

    # pin aggregation via gather + scatter_add
    Np = pins.shape[1]
    p = p2b[..., 0]
    blk = p2b[..., 1]
    w = p2b[..., 2]
    pi = p.long().clamp(0, max(Np - 1, 0))
    px = torch.gather(pins[..., 0], 1, pi)
    py = torch.gather(pins[..., 1], 1, pi)
    ok = (p >= 0) & (p < Np) & (blk >= 0) & (blk < N) & (px != -1)
    w = torch.where(ok, w.clamp_min(0.0), torch.zeros_like(w)).to(dtype)
    one = torch.where(ok, torch.ones_like(w), torch.zeros_like(w))
    bi = blk.long().clamp(0, N - 1)
    p2b_degree = area.new_zeros(B, N).scatter_add_(1, bi, one)
    p2b_weight = area.new_zeros(B, N).scatter_add_(1, bi, w)
    pin_x_sum = area.new_zeros(B, N).scatter_add_(1, bi, w * px)
    pin_y_sum = area.new_zeros(B, N).scatter_add_(1, bi, w * py)
    pin_denom = p2b_weight.clamp_min(1e-6)
    pin_x = pin_x_sum / pin_denom / scale.unsqueeze(-1)
    pin_y = pin_y_sum / pin_denom / scale.unsqueeze(-1)

    c = cons
    zeros = torch.zeros((B, N), dtype=dtype, device=device)
    fixed = c[..., 0].to(dtype)
    preplaced = c[..., 1].to(dtype)
    mib = (c[..., 2].clamp_min(0) / 32.0).to(dtype)
    cluster = (c[..., 3].clamp_min(0) / 32.0).to(dtype)
    boundary_code = c[..., 4].long().clamp_min(0)
    boundary = (boundary_code.to(dtype) / 15.0).to(dtype)
    boundary_l = ((boundary_code & 1) != 0).to(dtype)
    boundary_r = ((boundary_code & 2) != 0).to(dtype)
    boundary_t = ((boundary_code & 4) != 0).to(dtype)
    boundary_b = ((boundary_code & 8) != 0).to(dtype)
    boundary_count = (boundary_l + boundary_r + boundary_t + boundary_b) / 4.0
    boundary_corner = (boundary_count > 0.25).to(dtype)
    tp = target_positions.to(device=device, dtype=dtype)
    has_target_wh = ((tp[..., 2] > 0) & (tp[..., 3] > 0) & mask).to(dtype)
    has_target_xy = ((tp[..., 0] >= 0) & (tp[..., 1] >= 0) & mask).to(dtype)
    target_w = torch.where(has_target_wh.bool(), tp[..., 2] / scale.unsqueeze(-1), zeros)
    target_h = torch.where(has_target_wh.bool(), tp[..., 3] / scale.unsqueeze(-1), zeros)
    target_x = torch.where(has_target_xy.bool(), tp[..., 0] / scale.unsqueeze(-1), zeros)
    target_y = torch.where(has_target_xy.bool(), tp[..., 1] / scale.unsqueeze(-1), zeros)
    target_log_aspect = torch.where(
        has_target_wh.bool(),
        torch.log((tp[..., 2] / tp[..., 3]).clamp_min(1e-6)).clamp(-3.0, 3.0) / 3.0,
        zeros)

    features = [
        torch.log(safe_area) / 10.0,
        torch.sqrt(safe_area) / scale.unsqueeze(-1),
        fixed, preplaced, mib, cluster, boundary,
        torch.log1p(b2b_degree) / 5.0,
        torch.log1p(b2b_weight) / 10.0,
        torch.log1p(p2b_degree) / 5.0,
        torch.log1p(p2b_weight) / 10.0,
        pin_x, pin_y,
        boundary_l, boundary_r, boundary_t, boundary_b,
        boundary_count, boundary_corner,
        has_target_wh, has_target_xy,
        target_w, target_h, target_x, target_y, target_log_aspect,
    ]
    if node_feat_dim > 26:
        # v2 extension (features 27-32), appended AFTER the original 26 so
        # node_feat_dim=26 checkpoints see a bit-identical input:
        #   27-30: global pin bbox — the die-frame anchor the model
        #          otherwise has to reconstruct via attention
        #   31-32: per-block pin extent — the weighted-centroid pin
        #          feature hides multi-pin spread
        pin_ok = (pins[..., 0] != -1) & (pins[..., 1] != -1)
        big = torch.tensor(1e18, dtype=dtype, device=device)
        any_pin = pin_ok.any(dim=1, keepdim=True).to(dtype)
        s = scale.unsqueeze(-1)
        gx0 = torch.where(pin_ok, pins[..., 0], big).min(dim=1, keepdim=True).values
        gy0 = torch.where(pin_ok, pins[..., 1], big).min(dim=1, keepdim=True).values
        gx1 = torch.where(pin_ok, pins[..., 0], -big).max(dim=1, keepdim=True).values
        gy1 = torch.where(pin_ok, pins[..., 1], -big).max(dim=1, keepdim=True).values
        for gv in (gx0, gy0, gx1, gy1):
            features.append(((gv * any_pin) / s).expand(B, N) * any_pin)
        neg = torch.full((B, N), -1e18, dtype=dtype, device=device)
        posi = torch.full((B, N), 1e18, dtype=dtype, device=device)
        px_hi = torch.where(ok, px, neg.gather(1, bi))
        px_lo = torch.where(ok, px, posi.gather(1, bi))
        py_hi = torch.where(ok, py, neg.gather(1, bi))
        py_lo = torch.where(ok, py, posi.gather(1, bi))
        bxmax = neg.clone().scatter_reduce_(1, bi, px_hi, reduce="amax")
        bxmin = posi.clone().scatter_reduce_(1, bi, px_lo, reduce="amin")
        bymax = neg.clone().scatter_reduce_(1, bi, py_hi, reduce="amax")
        bymin = posi.clone().scatter_reduce_(1, bi, py_lo, reduce="amin")
        has_p = p2b_degree > 0
        features.append(torch.where(has_p, (bxmax - bxmin).clamp_min(0.0), zeros) / s)
        features.append(torch.where(has_p, (bymax - bymin).clamp_min(0.0), zeros) / s)
    feat = torch.stack(features[:node_feat_dim], dim=-1)
    if node_feat_dim > feat.shape[-1]:
        pad = torch.zeros((*feat.shape[:-1], node_feat_dim - feat.shape[-1]),
                          dtype=dtype, device=device)
        feat = torch.cat([feat, pad], dim=-1)
    feat = torch.where(mask.unsqueeze(-1), feat, torch.zeros_like(feat))
    rel_feat = build_relation_features(area, b2b, cons, relation_feat_dim,
                                       adj_raw=adj_raw)
    return {"node_feat": feat, "adj": adj, "mask": mask, "scale": scale,
            "rel_feat": rel_feat}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="../")
    p.add_argument("--checkpoint-dir", default="checkpoints/direct_v1")
    p.add_argument("--resume", default=None, help="checkpoint path or 'latest'")
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing latest.pt (default auto-resumes)")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=25000,
                   help="step_*.pt at these steps are kept forever")
    p.add_argument("--keep-recent", type=int, default=3,
                   help="non-snapshot step_*.pt files retained")
    # -- shared-GPU etiquette -------------------------------------------------
    p.add_argument("--vram-fraction", type=float, default=0.45,
                   help="hard cap on this process's share of GPU memory")
    p.add_argument("--gpu-util-cap", type=float, default=0.75,
                   help="duty-cycle cap: sleep so avg GPU util stays below this")
    p.add_argument("--max-steps", type=int, default=600000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--node-feat-dim", type=int, default=26,
                   help="26 = v1 features; 32 adds pin-bbox/spread")
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    p.add_argument("--self-cond-prob", type=float, default=0.5)
    p.add_argument("--x0-loss-weight", type=float, default=1.0)
    p.add_argument("--v-loss-weight", type=float, default=0.5)
    p.add_argument("--overlap-loss-weight", type=float, default=0.5)
    p.add_argument("--boundary-loss-weight", type=float, default=0.3)
    p.add_argument("--cluster-loss-weight", type=float, default=0.3)
    p.add_argument("--mib-loss-weight", type=float, default=0.1)
    p.add_argument("--bnd-node-weight", type=float, default=2.0)
    p.add_argument("--augment", type=int, default=1, help="1: random D4 augmentation")
    p.add_argument("--file-shuffle", type=int, default=1,
                   help="1: shuffle at file granularity (cache-friendly)")
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# D4 augmentation on the raw batch tensors (fp_sol is [w, h, x, y])
# ---------------------------------------------------------------------------
def _tag_remap_table(kind: str, device) -> torch.Tensor:
    # bits: 1=L, 2=R, 4=T, 8=B
    tab = torch.zeros(16, dtype=torch.long, device=device)
    for c in range(16):
        L, R, T, B = c & 1, c & 2, c & 4, c & 8
        if kind == "mx":       # mirror x: L<->R
            o = (2 if L else 0) | (1 if R else 0) | T | B
        elif kind == "my":     # mirror y: T<->B
            o = L | R | (8 if T else 0) | (4 if B else 0)
        else:                  # transpose: L->B, R->T, T->R, B->L
            o = (8 if L else 0) | (4 if R else 0) | (2 if T else 0) | (1 if B else 0)
        tab[c] = o
    return tab


def augment_batch(area, pins, cons, fp, rng: torch.Generator):
    """Random per-sample transpose / mirror-x / mirror-y, applied
    consistently to rectangles, pins, and boundary-tag bits."""
    B, N = area.shape
    device = area.device
    valid = valid_block_mask(area)
    pin_valid = (pins[..., 0] != -1) & (pins[..., 1] != -1)
    cons = cons.clone()
    fp = fp.clone()
    pins = pins.clone()
    code = cons[..., 4].long().clamp(0, 15)

    do_t = torch.rand(B, generator=rng, device=device) < 0.5
    do_x = torch.rand(B, generator=rng, device=device) < 0.5
    do_y = torch.rand(B, generator=rng, device=device) < 0.5

    # transpose first: swap (w,h), (x,y), (px,py)
    tsel = do_t.view(B, 1)
    fp_t = torch.stack([fp[..., 1], fp[..., 0], fp[..., 3], fp[..., 2]], dim=-1)
    fp = torch.where(tsel.unsqueeze(-1), fp_t, fp)
    pins_t = torch.stack([pins[..., 1], pins[..., 0]], dim=-1)
    pins = torch.where((tsel & pin_valid).unsqueeze(-1) if False else
                       (do_t.view(B, 1) & pin_valid).unsqueeze(-1), pins_t, pins)
    tab = _tag_remap_table("t", device)
    code = torch.where(do_t.view(B, 1), tab[code], code)

    # bbox after transpose (from valid blocks)
    x = fp[..., 2]
    y = fp[..., 3]
    w = fp[..., 0]
    h = fp[..., 1]
    inf = torch.tensor(float("inf"), device=device)
    bx0 = torch.where(valid, x, inf).min(dim=1, keepdim=True).values
    bx1 = torch.where(valid, x + w, -inf).max(dim=1, keepdim=True).values
    by0 = torch.where(valid, y, inf).min(dim=1, keepdim=True).values
    by1 = torch.where(valid, y + h, -inf).max(dim=1, keepdim=True).values

    mx = do_x.view(B, 1)
    new_x = torch.where(mx & valid, bx0 + bx1 - (x + w), x)
    fp = torch.stack([w, h, new_x, y], dim=-1)
    new_px = torch.where(mx & pin_valid, bx0 + bx1 - pins[..., 0], pins[..., 0])
    pins = torch.stack([new_px, pins[..., 1]], dim=-1)
    tab = _tag_remap_table("mx", device)
    code = torch.where(mx, tab[code], code)

    x = fp[..., 2]
    my = do_y.view(B, 1)
    new_y = torch.where(my & valid, by0 + by1 - (y + h), y)
    fp = torch.stack([w, h, x, new_y], dim=-1)
    new_py = torch.where(my & pin_valid, by0 + by1 - pins[..., 1], pins[..., 1])
    pins = torch.stack([pins[..., 0], new_py], dim=-1)
    tab = _tag_remap_table("my", device)
    code = torch.where(my, tab[code], code)

    cons[..., 4] = code.to(cons.dtype)
    return pins, cons, fp


# ---------------------------------------------------------------------------
# Vectorized auxiliary losses (per-sample, weighted by alpha_bar^2 upstream)
# ---------------------------------------------------------------------------
def overlap_fraction(rects, mask, scale):
    x1, y1, w, h = rects.unbind(dim=-1)
    x2 = x1 + w
    y2 = y1 + h
    ox = (torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1))
          - torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))).clamp_min(0)
    oy = (torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1))
          - torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))).clamp_min(0)
    pm = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(rects.dtype)
    pm = pm * (1 - torch.eye(mask.shape[1], device=mask.device,
                             dtype=rects.dtype)).unsqueeze(0)
    tot = (ox * oy * pm).sum(dim=(1, 2)) * 0.5
    return tot / scale.pow(2).clamp_min(1.0)          # [B]


def boundary_touch(rects, cons, mask, scale):
    if cons.shape[-1] <= 4:
        return rects.new_zeros(rects.shape[0])
    x, y, w, h = rects.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    inf = torch.tensor(float("inf"), device=rects.device)
    mnx = torch.where(mask, x, inf).min(dim=1, keepdim=True).values
    mny = torch.where(mask, y, inf).min(dim=1, keepdim=True).values
    mxx = torch.where(mask, x2, -inf).max(dim=1, keepdim=True).values
    mxy = torch.where(mask, y2, -inf).max(dim=1, keepdim=True).values
    code = cons[..., 4].long()
    s = scale.view(-1, 1).clamp_min(1.0)
    loss = ((x - mnx).abs() / s) * ((code & 1) != 0)
    loss = loss + ((mxx - x2).abs() / s) * ((code & 2) != 0)
    loss = loss + ((mxy - y2).abs() / s) * ((code & 4) != 0)
    loss = loss + ((y - mny).abs() / s) * ((code & 8) != 0)
    active = ((code != 0) & mask).to(rects.dtype)
    return (loss * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


def _edge_dist(rects):
    x, y, w, h = rects.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    dx = torch.maximum(x.unsqueeze(2) - x2.unsqueeze(1),
                       x.unsqueeze(1) - x2.unsqueeze(2)).clamp_min(0)
    dy = torch.maximum(y.unsqueeze(2) - y2.unsqueeze(1),
                       y.unsqueeze(1) - y2.unsqueeze(2)).clamp_min(0)
    return dx + dy


def cluster_gap(rects, gt_rects, cons, mask, scale):
    """Match each cluster member's nearest-same-cluster gap to GT (≈0)."""
    if cons.shape[-1] <= 3:
        return rects.new_zeros(rects.shape[0])
    cl = cons[..., 3].long().clamp_min(0)
    same = (cl.unsqueeze(1) == cl.unsqueeze(2)) & (cl.unsqueeze(1) > 0)
    same = same & mask.unsqueeze(1) & mask.unsqueeze(2)
    same = same & ~torch.eye(mask.shape[1], dtype=torch.bool,
                             device=mask.device).unsqueeze(0)
    if not same.any():
        return rects.new_zeros(rects.shape[0])
    inf = torch.finfo(rects.dtype).max
    d = _edge_dist(rects).masked_fill(~same, inf).min(dim=2).values
    dg = _edge_dist(gt_rects).masked_fill(~same, inf).min(dim=2).values
    node = same.any(dim=2)
    s = scale.view(-1, 1).clamp_min(1.0)
    l = F.smooth_l1_loss(d / s, (dg / s).detach(), reduction="none", beta=0.05)
    return (l * node).sum(dim=1) / node.sum(dim=1).clamp_min(1)


def mib_aspect(rects, cons, mask):
    if cons.shape[-1] <= 2:
        return rects.new_zeros(rects.shape[0])
    g = cons[..., 2].long().clamp_min(0)
    same = (g.unsqueeze(1) == g.unsqueeze(2)) & (g.unsqueeze(1) > 0)
    same = same & mask.unsqueeze(1) & mask.unsqueeze(2)
    same = same & ~torch.eye(mask.shape[1], dtype=torch.bool,
                             device=mask.device).unsqueeze(0)
    if not same.any():
        return rects.new_zeros(rects.shape[0])
    la = torch.log(rects[..., 2].clamp_min(1e-6) / rects[..., 3].clamp_min(1e-6))
    diff = (la.unsqueeze(1) - la.unsqueeze(2)).abs()
    sm = same.to(rects.dtype)
    return (diff * sm).sum(dim=(1, 2)) / sm.sum(dim=(1, 2)).clamp_min(1)


# ---------------------------------------------------------------------------
def train_step(model, ema, schedule, batch, args, rng, amp_dtype, amp_on):
    """One loss computation.  NOTE: the self-conditioning no-grad forward and
    the main forward each get their own autocast region — sharing one region
    would poison torch's autocast weight cache with grad-less casts (the
    second forward would silently lose its autograd graph)."""
    from diffusion_train import known_target_positions_from_fp
    area, b2b, p2b, pins, cons, _tree, fp, _metrics = batch
    if args.augment:
        pins, cons, fp = augment_batch(area, pins, cons, fp, rng)
    target_positions = known_target_positions_from_fp(fp, cons)
    z0, mask, scale = fp_sol_to_z0(fp, area, z_repr="xyaspect")
    cond = fast_condition(
        area, b2b, p2b, pins, cons,
        target_positions=target_positions,
        relation_feat_dim=model.config.relation_feat_dim,
        node_feat_dim=model.config.node_feat_dim,
    )
    B = area.shape[0]
    device = area.device
    t = torch.randint(0, schedule.timesteps, (B,), device=device, generator=rng)
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
            v1 = model(z_t, t, cond["node_feat"], cond["adj"], mask,
                       rel_feat=cond["rel_feat"], self_cond=None)
        v1 = v1.float()
        sc = (alpha * z_t - sigma * v1)
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

    # auxiliary geometry losses on the decoded rectangles, gated by how
    # trustworthy the x0 estimate is at this noise level
    ab2 = (alpha.view(B) ** 4)
    rects = z_to_rectangles(z0_hat, area, target_positions=target_positions,
                            constraints=cons, z_repr="xyaspect")
    gt_rects = torch.stack([fp[..., 2], fp[..., 3], fp[..., 0], fp[..., 1]], dim=-1)
    ov = (overlap_fraction(rects, mask, scale) * ab2).mean()
    bd = (boundary_touch(rects, cons, mask, scale) * ab2).mean()
    cg = (cluster_gap(rects, gt_rects, cons, mask, scale) * ab2).mean()
    mb = (mib_aspect(rects, cons, mask) * ab2).mean()

    loss = (args.v_loss_weight * v_loss
            + args.x0_loss_weight * x0_loss
            + args.overlap_loss_weight * ov
            + args.boundary_loss_weight * bd
            + args.cluster_loss_weight * cg
            + args.mib_loss_weight * mb)

    with torch.no_grad():
        pe = ((z0_hat[..., :2] - z0[..., :2]).abs().sum(-1) * mask).sum() \
            / mask.sum().clamp_min(1)
    return {"loss": loss, "v": v_loss.detach(), "x0": x0_loss.detach(),
            "ov": ov.detach(), "bd": bd.detach(), "cg": cg.detach(),
            "mib": mb.detach(), "pos_l1": pe.detach()}


class FileShuffleSampler(torch.utils.data.Sampler):
    """Shuffle at file granularity + within each file.

    The lite dataset caches one 112-layout file per worker, so global index
    shuffling would reload a file per sample.  Files hold exactly 112 = 7x16
    layouts, so with batch_size 16 every batch stays inside one file: full
    decorrelation across epochs at zero cache cost (the sequential default
    feeds long same-n same-generator stretches, which is why the old runs'
    loss swung file-by-file)."""

    def __init__(self, n_files: int, per_file: int, seed: int):
        self.n_files = n_files
        self.per_file = per_file
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.n_files * self.per_file

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for f in torch.randperm(self.n_files, generator=g).tolist():
            base = f * self.per_file
            for k in torch.randperm(self.per_file, generator=g).tolist():
                yield base + k


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckdir = Path(args.checkpoint_dir)
    ckdir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda" and 0.0 < args.vram_fraction < 1.0:
        # hard cap: allocations beyond our share raise OOM here instead of
        # squeezing whoever else is on the card
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, dev_idx)
        total = torch.cuda.get_device_properties(dev_idx).total_memory
        print(f"VRAM cap: {args.vram_fraction:.0%} of "
              f"{total / 2**30:.1f} GiB = {args.vram_fraction * total / 2**30:.1f} GiB")

    cfg = DirectModelConfig(
        d_model=args.d_model, layers=args.layers, heads=args.heads,
        dropout=args.dropout, timesteps=args.timesteps,
        node_feat_dim=args.node_feat_dim)
    model = DirectDenoiser(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DirectDenoiser: {n_params/1e6:.1f}M params, device={device}")

    decay, no_decay = [], []
    for n, p_ in model.named_parameters():
        (no_decay if p_.dim() <= 1 else decay).append(p_)
    optimizer = AdamW([{"params": decay, "weight_decay": 0.01},
                       {"params": no_decay, "weight_decay": 0.0}],
                      lr=args.lr, betas=(0.9, 0.99))

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, args.ema_decay)
    diffusion = DiffusionSchedule(args.timesteps, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    step = 0
    # auto-resume: colleagues may kill this process at any time, so an
    # unattended relaunch must pick up where it left off unless --fresh
    if not args.resume and not args.fresh and (ckdir / "latest.pt").exists():
        args.resume = "latest"
        print("auto-resume: found latest.pt (pass --fresh to start over)")
    if args.resume:
        path = ckdir / "latest.pt" if args.resume == "latest" else Path(args.resume)
        if path.exists():
            ck = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["sched"])
            step = int(ck["step"])
            print(f"resumed from {path} at step {step}")
        else:
            print(f"resume not found: {path}; starting fresh")

    if args.file_shuffle and not args.num_samples:
        dataset = FloorplanDatasetLite(args.data_path)
        sampler = FileShuffleSampler(
            len(dataset) // dataset.layouts_per_file,
            dataset.layouts_per_file, args.seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=train_floorplan_collate,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0)
    else:
        loader = get_training_dataloader(
            data_path=args.data_path, batch_size=args.batch_size,
            num_samples=args.num_samples, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    def prune_step_checkpoints():
        """Keep latest.pt, permanent snapshots (multiples of snapshot-every),
        and the newest --keep-recent step files; delete the rest so frequent
        saves cannot fill the shared disk."""
        files = sorted(ckdir.glob("step_*.pt"))

        def step_of(p):
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1
        candidates = [p for p in files
                      if step_of(p) > 0 and step_of(p) % max(args.snapshot_every, 1)]
        for p in candidates[:-max(args.keep_recent, 0) or None]:
            try:
                p.unlink()
            except OSError:
                pass

    def checkpoint(name):
        state = {"model": model.state_dict(), "ema": ema.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "sched": sched.state_dict(), "step": step,
                 "model_config": cfg.__dict__, "args": vars(args)}
        atomic_save(state, ckdir / name)
        atomic_save(state, ckdir / "latest.pt")
        prune_step_checkpoints()
        print(f"saved {ckdir / name}", flush=True)

    stop = False

    def on_sig(signum, _f):
        nonlocal stop
        stop = True
        print(f"signal {signum}: will checkpoint and exit", flush=True)

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    try:
        signal.signal(signal.SIGHUP, on_sig)   # terminal/session closed
    except (AttributeError, ValueError):
        pass

    log_path = ckdir / "train_log.jsonl"
    model.train()
    accum = max(1, args.grad_accum_steps)
    t_last = time.time()
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # duty-cycle throttle: after each optimizer step sleep in proportion to
    # the time we kept the GPU busy, so average utilization stays below
    # --gpu-util-cap and the card remains responsive for other users
    throttle = device.type == "cuda" and 0.0 < args.gpu_util_cap < 1.0
    if throttle:
        print(f"GPU duty-cycle cap: {args.gpu_util_cap:.0%} "
              f"(sleep {1.0 / args.gpu_util_cap - 1.0:.2f}x busy time per step)")
    t_work = time.time()
    while step < args.max_steps and not stop:
        micro = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            out = train_step(model, ema, diffusion, batch, args, rng,
                             amp_dtype, args.amp)
            loss = out["loss"] / accum
            loss.backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sched.step()
            ema.update(model)
            step += 1
            if throttle:
                torch.cuda.synchronize()
                busy = time.time() - t_work
                pause = busy * (1.0 / args.gpu_util_cap - 1.0)
                if pause > 0:
                    time.sleep(min(pause, 5.0))
                t_work = time.time()
            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t_last
                t_last = time.time()
                row = {k: float(v.item()) for k, v in out.items()}
                row.update(step=step, lr=float(sched.get_last_lr()[0]),
                           sps=args.log_every / max(dt, 1e-9))
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"step {step} loss {row['loss']:.4f} v {row['v']:.4f} "
                      f"x0 {row['x0']:.4f} ov {row['ov']:.4f} bd {row['bd']:.4f} "
                      f"cg {row['cg']:.4f} pos_l1 {row['pos_l1']:.4f} "
                      f"lr {row['lr']:.2e} {row['sps']:.2f} it/s", flush=True)
            if step % args.save_every == 0:
                checkpoint(f"step_{step:08d}.pt")
            if stop or step >= args.max_steps:
                break
    checkpoint("final.pt" if step >= args.max_steps else "interrupt.pt")
    print(f"done at step {step}")


if __name__ == "__main__":
    main()
