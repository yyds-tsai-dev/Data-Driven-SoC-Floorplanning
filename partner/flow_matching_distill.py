#!/usr/bin/env python3
"""Post-hoc few-step distillation of a pre-trained flow-matching checkpoint.

Implements SCFM (ShortCut distillation for Flow Matching, arXiv 2510.17858)
against the ``flow_matching_v*`` teachers produced by ``flow_matching_train``.
The method needs no step-size embedding and no architectural change: it forces
the *velocity field itself* to be step-size independent, and a step-size
independent field is a straight one, which is exactly what a 1-2 step Euler
sampler needs.

Velocity-space consistency (paper Eq. 11-12), written in this repo's time
convention (``z_t = (1-t)*noise + t*z0``, so t=0 is noise, t=1 is data, and the
sampler integrates t upward -- the paper's t is our ``1 - t``, which flips the
sign of both the velocity and the step and therefore leaves the convex
combination below unchanged)::

    t1 < t2 < t3,  d1 = t2 - t1,  d2 = t3 - t2
    z_t2   = z_t1 + d1 * V_near(z_t1, t1)
    target = d1/(d1+d2) * V_near(z_t1, t1) + d2/(d1+d2) * V_far(z_t2, t2)
    loss   = || V_student(z_t1, t1) - target ||^2      (stopgrad on target)

``V_near`` is the frozen teacher for a ``--teacher-frac`` slice of the batch
(paper Eq. 21, "vanilla-mix") and the *fast* EMA of the student for the rest;
``V_far`` is always the *slow* EMA (paper Eq. 22, the dual fast-slow EMA that
Appendix E selects as the final algorithm).  The teacher slice uses the
finest grid skip (1) and anchors the student to the teacher's field; the
self-distilled slice uses coarse skips and does the actual straightening.

Deliberate deviations from the paper are marked ``DEVIATION`` below and are
argued in ``docs/superpowers/plans/2026-07-29-flow-distill-design.md``.

The optimizer, dataset, collate, augmentation, conditioning, checkpoint
payload and shared-GPU etiquette are the ones ``direct_diffusion_train`` /
``flow_matching_train`` already use, so a distilled checkpoint loads through the
unchanged ``checkpoint_method`` -> ``DirectDenoiser`` -> ``sample_flow`` path
(``contest_optimizer._load_flow_model`` and ``scripts/probes/flow_candidate_probe``
both sample the ``ema`` payload key, which is the slow EMA here).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))

from flow_matching_model import endpoint_from_velocity, flow_path, sample_flow

TRAINING_METHOD = "flow_distill_v1"


# ---------------------------------------------------------------------------
# Pure schedule / target helpers (no model, no globals -- unit testable)
# ---------------------------------------------------------------------------
def skip_ladder(grid_steps: int, max_skip: int) -> list[int]:
    """Power-of-two coarse skips ``{2, 4, ..., max_skip}`` on the grid.

    The paper draws the self-distilled skip from ``{2, 4, ..., T/4}``.  A skip
    of ``s`` asks the student to reproduce a jump of ``2*s/grid_steps`` of the
    whole path, so the ladder top sets how aggressive the shortcut is.
    """
    if grid_steps < 4 or grid_steps & (grid_steps - 1):
        raise ValueError("grid_steps must be a power of two >= 4")
    if max_skip < 2 or max_skip > grid_steps // 2:
        raise ValueError("max_skip must satisfy 2 <= max_skip <= grid_steps//2")
    ladder = []
    s = 2
    while s <= max_skip:
        ladder.append(s)
        s *= 2
    return ladder


def default_max_skip(grid_steps: int, steps_target: int) -> int:
    """Ladder top: ``grid/4`` (paper) for >=2 steps, ``grid/2`` for 1 step.

    DEVIATION.  At ``grid/2`` the only admissible window is ``t1=0, t2=1/2,
    t3=1``, i.e. exactly the 2-step-to-1-step consistency at the sampler's own
    entry point.  The paper never trains the full-path jump directly and lets
    1-step ability emerge from step-size independence; our teacher runs 8 steps
    rather than 32, so the emergent extrapolation is over a shorter ladder and
    the entry-point window is worth supervising when 1 step is the target.
    """
    return grid_steps // 2 if steps_target <= 1 else grid_steps // 4


def apply_shift(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 17 timestep shift, expressed in this repo's time direction.

    In paper time (``tp = 1 - t``, 1 = noise) the shift is
    ``S_s(tp) = s*tp / (1 + (s-1)*tp)``, which for ``s > 1`` moves mass toward
    the noise end.  Mapping back gives the transform below.  ``s == 1`` is the
    identity, which is the default here because our teacher is both trained and
    deployed on a uniform grid (DEVIATION: the paper always shifts, because
    Flux/SD3 are *sampled* with a shift).
    """
    tp = 1.0 - t
    return 1.0 - (s * tp) / (1.0 + (s - 1.0) * tp)


def sample_windows(
    batch: int,
    grid_steps: int,
    ladder: list[int],
    teacher_frac: float,
    jitter: bool,
    shift_range: tuple[float, float],
    device,
    generator=None,
) -> dict[str, torch.Tensor]:
    """Draw one ``(t1, t2, t3)`` consistency window per batch element.

    Returns ``t1/t2/t3`` in ``[0, 1]``, the boolean ``is_teacher`` slice and the
    integer ``skip`` (for logging).  Teacher-slice elements always use skip 1.
    """
    def rand(*shape):
        return torch.rand(shape, device=device, generator=generator)

    is_teacher = rand(batch) < teacher_frac
    ladder_t = torch.tensor(ladder, device=device, dtype=torch.long)
    pick = (rand(batch) * len(ladder)).long().clamp_max(len(ladder) - 1)
    skip = torch.where(is_teacher, torch.ones_like(pick), ladder_t[pick])

    jmax = grid_steps - 2 * skip                       # t3 index must stay <= grid
    j = (rand(batch) * (jmax + 1).to(torch.float32)).long().clamp(min=0)
    j = torch.minimum(j, jmax)
    # jitter only when the window does not touch the data end, so t3 <= 1 holds
    room = (j < jmax).to(torch.float32)
    u = rand(batch) * room / grid_steps if jitter else torch.zeros(batch, device=device)

    t1 = j.to(torch.float32) / grid_steps + u
    t2 = (j + skip).to(torch.float32) / grid_steps + u
    t3 = (j + 2 * skip).to(torch.float32) / grid_steps + u

    lo, hi = shift_range
    if not (lo == 1.0 and hi == 1.0):
        s = lo + (hi - lo) * rand(batch)
        t1, t2, t3 = apply_shift(t1, s), apply_shift(t2, s), apply_shift(t3, s)

    return {"t1": t1, "t2": t2, "t3": t3, "is_teacher": is_teacher, "skip": skip}


def scfm_target(
    v_near: torch.Tensor,
    v_far: torch.Tensor,
    d_near: torch.Tensor,
    d_far: torch.Tensor,
) -> torch.Tensor:
    """Paper Eq. 12: interval-weighted mean of the two sub-step velocities.

    Sanity property used by the smoke check: for a field whose velocity is
    constant along the path (an already-straight, perfectly rectified flow) the
    target equals that velocity for every window, so a converged straight field
    is a fixed point of the loss and distillation is a no-op on it.
    """
    w = (d_near / (d_near + d_far).clamp_min(1e-9)).view(-1, *([1] * (v_near.ndim - 1)))
    return w * v_near + (1.0 - w) * v_far


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Velocity l2 over valid blocks only (paper keeps the plain l2)."""
    w = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target).square() * w).sum() / (w.sum().clamp_min(1) * pred.shape[-1])


def _bt(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], *([1] * (like.ndim - 1))).to(like)


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------
def load_teacher(path: Path, device):
    """Return ``(model, config, payload)`` with the *sampled* (EMA) weights.

    Both the production loader and the candidate probe prefer ``ema`` over
    ``model``; distilling anything else would distil weights nobody samples.
    """
    from direct_diffusion_model import DirectDenoiser, DirectModelConfig
    from flow_matching_train import checkpoint_method

    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_method(payload)
    cfg = DirectModelConfig(**{k: v for k, v in payload["model_config"].items()
                               if k in DirectModelConfig.__dataclass_fields__})
    if cfg.z_repr != "xyaspect":
        raise ValueError(f"teacher must use the xyaspect decoder, got {cfg.z_repr!r}")
    model = DirectDenoiser(cfg).to(device)
    model.load_state_dict(payload.get("ema") or payload["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, payload


@torch.no_grad()
def refresh_from_shadow(model, shadow):
    """Copy an ``EMA`` shadow into a materialized model (dtype-preserving)."""
    sd = model.state_dict()
    for k in sd:
        sd[k].copy_(shadow[k].to(sd[k].dtype))


def ema_model(shadow, cfg, device):
    """Materialize a frozen ``DirectDenoiser`` that mirrors an EMA shadow."""
    from direct_diffusion_model import DirectDenoiser

    model = DirectDenoiser(cfg).to(device)
    refresh_from_shadow(model, shadow)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # -- distillation ---------------------------------------------------------
    p.add_argument("--teacher-checkpoint", required=True,
                   help="flow_matching_v1/v3 checkpoint (its EMA weights are the teacher)")
    p.add_argument("--steps-target", type=int, default=1, choices=(1, 2, 4),
                   help="sampler step count the student is aimed at (sets the skip ladder top)")
    p.add_argument("--grid-steps", type=int, default=32,
                   help="resolution of the discretized timestep grid L_n (power of two)")
    p.add_argument("--max-skip", type=int, default=0, help="0: derive from --steps-target")
    p.add_argument("--teacher-frac", type=float, default=0.4,
                   help="k/N in paper Eq. 13: batch share anchored to the teacher at skip 1")
    p.add_argument("--ema-fast", type=float, default=0.99)
    p.add_argument("--ema-slow", type=float, default=0.999)
    p.add_argument("--grid-jitter", type=int, default=1,
                   help="1: offset the grid by a continuous sub-cell phase (t coverage)")
    p.add_argument("--shift-min", type=float, default=1.0)
    p.add_argument("--shift-max", type=float, default=1.0)
    p.add_argument("--impose-known", type=int, default=1,
                   help="1: overwrite hard-anchor z channels in the rollout, as the sampler does")
    p.add_argument("--student-self-cond", type=int, default=0,
                   help="1: feed the stopgrad endpoint estimate as the student's self-cond")
    p.add_argument("--aux-weight", type=float, default=0.0,
                   help=">0: golden-endpoint smooth-l1 auxiliary (arm B; off by default)")
    p.add_argument("--overlap-aux-weight", type=float, default=0.0,
                   help=">0: overlap-fraction auxiliary on the implied endpoint (arm B)")
    p.add_argument("--aux-time-weight", choices=("none", "t2"), default="none",
                   help="'t2' reproduces flow_train's t**2 endpoint gating; 'none' (default) "
                        "keeps weight at t->0, which is where a 1-step student is read out")
    # -- lifecycle (mirrors direct_diffusion_train) ------------------------------
    p.add_argument("--data-path", default="FloorSet")
    p.add_argument("--checkpoint-dir", default="checkpoints/flow_distill_v1")
    p.add_argument("--resume", default=None, help="checkpoint path or 'latest'")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=None,
                   help="cap the training subset (few-shot ablation)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--augment", type=int, default=1)
    p.add_argument("--file-shuffle", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--vram-fraction", type=float, default=0.45)
    p.add_argument("--gpu-util-cap", type=float, default=0.75)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--probe-every", type=int, default=500,
                   help="0: off. Sample student@steps-target vs teacher@8 on a fixed batch")
    p.add_argument("--probe-teacher-steps", type=int, default=8)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=5000)
    p.add_argument("--keep-recent", type=int, default=3)
    args = p.parse_args(argv)
    if not args.max_skip:
        args.max_skip = default_max_skip(args.grid_steps, args.steps_target)
    if not 0.0 <= args.teacher_frac <= 1.0:
        raise ValueError("--teacher-frac must lie in [0, 1]")
    args.training_method = TRAINING_METHOD
    return args


# ---------------------------------------------------------------------------
# One distillation step
# ---------------------------------------------------------------------------
def distill_step(nets, batch, args, rng, amp_dtype, amp_on, ladder):
    """Return the SCFM loss and diagnostics for one batch.

    ``nets`` carries ``student`` (trainable), ``teacher`` (frozen), ``fast`` and
    ``slow`` (frozen EMA mirrors, refreshed by the caller).
    """
    import direct_diffusion_train as V1
    from diffusion_data import fp_sol_to_z0
    from direct_diffusion_model import known_z_channels
    from diffusion_train import known_target_positions_from_fp

    student, teacher, fast, slow = nets["student"], nets["teacher"], nets["fast"], nets["slow"]
    cfg = student.config

    area, b2b, p2b, pins, cons, _tree, fp, _metrics = batch
    if args.augment:
        pins, cons, fp = V1.augment_batch(area, pins, cons, fp, rng)
    target_positions = known_target_positions_from_fp(fp, cons)
    z0, mask, scale = fp_sol_to_z0(fp, area, z_repr="xyaspect")
    cond = V1.fast_condition(area, b2b, p2b, pins, cons,
                             target_positions=target_positions,
                             relation_feat_dim=cfg.relation_feat_dim,
                             node_feat_dim=cfg.node_feat_dim)
    device = area.device
    B = area.shape[0]
    z_known, known = known_z_channels(area, cons, target_positions, scale)

    win = sample_windows(B, args.grid_steps, ladder, args.teacher_frac,
                         bool(args.grid_jitter), (args.shift_min, args.shift_max),
                         device, rng)
    t1, t2, t3 = win["t1"], win["t2"], win["t3"]
    d1, d2 = t2 - t1, t3 - t2

    noise = torch.randn(z0.shape, device=device, generator=rng)
    z_t1, _ = flow_path(z0, noise, t1)
    z_t1 = z_t1 * mask.unsqueeze(-1)

    def run(net, z, t, self_cond):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
            out = net(z, t * (cfg.timesteps - 1), cond["node_feat"], cond["adj"], mask,
                      rel_feat=cond["rel_feat"], self_cond=self_cond)
        return out.float()

    with torch.no_grad():
        # V_near: teacher on the anchored slice, fast EMA on the self-distilled
        # slice (paper Eq. 21 / Eq. 22).  Whole-batch forwards keep the indexing
        # trivial; the degenerate fractions skip the pass they do not need.
        if args.teacher_frac >= 1.0:
            v_near = run(teacher, z_t1, t1, None)
        elif args.teacher_frac <= 0.0:
            v_near = run(fast, z_t1, t1, None)
        else:
            v_near = torch.where(win["is_teacher"].view(B, 1, 1),
                                 run(teacher, z_t1, t1, None),
                                 run(fast, z_t1, t1, None))

        # one teacher/EMA Euler sub-step, mirroring sample_flow exactly
        sc2 = endpoint_from_velocity(z_t1, v_near, t1)
        sc2[..., 2] = sc2[..., 2].clamp(-3.0, 3.0)
        z_t2 = z_t1 + _bt(d1, z_t1) * v_near
        if args.impose_known:
            imposed = (1.0 - _bt(t2, z_t1)) * noise + _bt(t2, z_t1) * z_known
            z_t2 = torch.where(known, imposed, z_t2)
            sc2 = torch.where(known, z_known, sc2)
        z_t2 = z_t2 * mask.unsqueeze(-1)

        v_far = run(slow, z_t2, t2, sc2)
        target = scfm_target(v_near, v_far, d1, d2)
        sc_student = sc2 if args.student_self_cond else None

    v_student = run(student, z_t1, t1, sc_student)
    velocity_loss = masked_mse(v_student, target, mask)
    loss = velocity_loss

    # Arm B (both weights default to 0 = paper-faithful pure velocity l2).  The
    # measured few-step deficit of the v1 teacher is almost entirely *overlap*
    # (st1 = 8.2x st8 raw overlap) and barely coordinate accuracy (st1 = 1.15x
    # st8 position L1), so the overlap term is the targeted auxiliary and the
    # endpoint term is the coarse anchor that keeps it from drifting.
    # Unlike flow_train's endpoint geometry, the default time weight is flat:
    # a 1-step student is read out at t -> 0, exactly where a t**2 gate is zero.
    z0_hat = endpoint_from_velocity(z_t1, v_student, t1)
    aux = torch.zeros((), device=device)
    ov = torch.zeros((), device=device)
    if args.aux_weight > 0.0 or args.overlap_aux_weight > 0.0:
        tw = _bt(t1.square(), z0_hat) if args.aux_time_weight == "t2" else 1.0
    if args.aux_weight > 0.0:
        chan = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device).view(1, 1, 4)
        w = mask.unsqueeze(-1).to(z0_hat.dtype) * chan * tw
        err = F.smooth_l1_loss(z0_hat, z0, reduction="none", beta=0.02) * w
        aux = err.sum() / (mask.unsqueeze(-1).to(z0_hat.dtype) * chan).sum().clamp_min(1)
        loss = loss + args.aux_weight * aux
    if args.overlap_aux_weight > 0.0:
        from diffusion_data import z_to_rectangles
        rects = z_to_rectangles(z0_hat, area, target_positions=target_positions,
                                constraints=cons, z_repr="xyaspect")
        per_sample = V1.overlap_fraction(rects, mask, scale)
        ov = (per_sample * (tw.reshape(-1) if torch.is_tensor(tw) else tw)).mean()
        loss = loss + args.overlap_aux_weight * ov

    with torch.no_grad():
        pos_l1 = ((z0_hat[..., :2] - z0[..., :2]).abs().sum(-1) * mask).sum()
        pos_l1 = pos_l1 / mask.sum().clamp_min(1)
        drift = masked_mse(v_student, v_near, mask)
    return {"loss": loss, "vel": velocity_loss.detach(), "aux": aux.detach(),
            "ov": ov.detach(), "pos_l1": pos_l1.detach(), "drift": drift.detach(),
            "skip": win["skip"].float().mean().detach()}


@torch.no_grad()
def rollout_probe(nets, probe, args, cfg, device, amp_dtype, amp_on):
    """Compare the few-step student rollout against the many-step teacher.

    This is the cheap fidelity rung: identical seeds, identical conditioning,
    ``--steps-target`` student steps versus ``--probe-teacher-steps`` teacher
    steps, reported as masked coordinate L1 between the two sampled ``z``s and
    each one's L1 against the golden layout.
    """
    cond, mask, z0, z_known, known = probe
    del amp_dtype, amp_on

    def draw(net, steps):
        gen = torch.Generator(device=device)
        gen.manual_seed(20260729)
        return sample_flow(net, cond, steps=steps, solver="euler", generator=gen,
                           z_known=z_known, known_mask=known).z

    zs = draw(nets["slow_model"], max(args.steps_target, 1))
    zt = draw(nets["teacher"], max(args.probe_teacher_steps, 1))
    denom = mask.sum().clamp_min(1)

    def l1(a, b):
        return float((((a[..., :2] - b[..., :2]).abs().sum(-1)) * mask).sum() / denom)

    return {"probe_st_vs_tea": l1(zs, zt), "probe_st_vs_gt": l1(zs, z0),
            "probe_tea_vs_gt": l1(zt, z0)}


def build_loader(args, device):
    import inspect

    import direct_diffusion_train as V1
    from iccad2026_evaluate import FloorplanDatasetLite, get_training_dataloader

    if args.file_shuffle and not args.num_samples:
        dataset = FloorplanDatasetLite(args.data_path)
        sampler = V1.FileShuffleSampler(len(dataset) // dataset.layouts_per_file,
                                        dataset.layouts_per_file, args.seed)
        from iccad2026_evaluate import train_floorplan_collate
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=train_floorplan_collate, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0)
    # the official train-only loader has no throughput kwargs; forward what it takes
    accepted = inspect.signature(get_training_dataloader).parameters
    kwargs = {"data_path": args.data_path, "batch_size": args.batch_size,
              "num_samples": args.num_samples, "shuffle": False,
              "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return get_training_dataloader(**kwargs)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckdir = Path(args.checkpoint_dir)
    ckdir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda" and 0.0 < args.vram_fraction < 1.0:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, idx)

    import direct_diffusion_train as V1
    from diffusion_data import batch_to_device, fp_sol_to_z0
    from direct_diffusion_model import DirectDenoiser, EMA, known_z_channels
    from diffusion_train import known_target_positions_from_fp

    teacher, cfg, _payload = load_teacher(Path(args.teacher_checkpoint), device)
    student = DirectDenoiser(cfg).to(device)
    student.load_state_dict(teacher.state_dict())          # init student <- teacher
    n_params = sum(p.numel() for p in student.parameters())
    ladder = skip_ladder(args.grid_steps, args.max_skip)
    print(f"SCFM distill: {n_params/1e6:.1f}M params, grid={args.grid_steps}, "
          f"ladder={ladder}, teacher_frac={args.teacher_frac}, "
          f"steps_target={args.steps_target}, device={device}", flush=True)

    decay, no_decay = [], []
    for _n, p_ in student.named_parameters():
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
    ema_f, ema_s = EMA(student, args.ema_fast), EMA(student, args.ema_slow)

    step = 0
    if not args.resume and not args.fresh and (ckdir / "latest.pt").exists():
        args.resume = "latest"
        print("auto-resume: found latest.pt (pass --fresh to start over)")
    if args.resume:
        path = ckdir / "latest.pt" if args.resume == "latest" else Path(args.resume)
        if path.exists():
            ck = torch.load(path, map_location=device, weights_only=False)
            student.load_state_dict(ck["model"])
            ema_s.load_state_dict(ck["ema"])
            ema_f.load_state_dict(ck.get("ema_fast", ck["ema"]))
            optimizer.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["sched"])
            step = int(ck["step"])
            print(f"resumed from {path} at step {step}")
        else:
            print(f"resume not found: {path}; starting fresh")

    fast = ema_model(ema_f.state_dict(), cfg, device)
    slow = ema_model(ema_s.state_dict(), cfg, device)
    nets = {"student": student, "teacher": teacher, "fast": fast, "slow": slow,
            "slow_model": slow}

    loader = build_loader(args, device)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    def prune():
        files = sorted(ckdir.glob("step_*.pt"))

        def step_of(p):
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1
        cands = [p for p in files
                 if step_of(p) > 0 and step_of(p) % max(args.snapshot_every, 1)]
        for p in cands[:-max(args.keep_recent, 0) or None]:
            try:
                p.unlink()
            except OSError:
                pass

    def checkpoint(name):
        state = {"model": student.state_dict(), "ema": ema_s.state_dict(),
                 "ema_fast": ema_f.state_dict(), "optimizer": optimizer.state_dict(),
                 "sched": sched.state_dict(), "step": step,
                 "model_config": cfg.__dict__, "args": vars(args)}
        V1.atomic_save(state, ckdir / name)
        V1.atomic_save(state, ckdir / "latest.pt")
        prune()
        print(f"saved {ckdir / name}", flush=True)

    stop = False

    def on_sig(signum, _f):
        nonlocal stop
        stop = True
        print(f"signal {signum}: will checkpoint and exit", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            try:
                signal.signal(sig, on_sig)
            except (OSError, ValueError):
                pass

    log_path = ckdir / "train_log.jsonl"
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    amp_on = bool(args.amp) and device.type == "cuda"
    accum = max(1, args.grad_accum_steps)
    throttle = device.type == "cuda" and 0.0 < args.gpu_util_cap < 1.0
    probe = None
    t_last = t_work = time.time()
    student.train()

    while step < args.max_steps and not stop:
        micro = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            if probe is None and args.probe_every:
                area, b2b, p2b, pins, cons, _tr, fp, _m = batch
                tp = known_target_positions_from_fp(fp, cons)
                pz0, pmask, pscale = fp_sol_to_z0(fp, area, z_repr="xyaspect")
                pcond = V1.fast_condition(area, b2b, p2b, pins, cons, target_positions=tp,
                                          relation_feat_dim=cfg.relation_feat_dim,
                                          node_feat_dim=cfg.node_feat_dim)
                pzk, pk = known_z_channels(area, cons, tp, pscale)
                probe = (pcond, pmask, pz0, pzk, pk)

            out = distill_step(nets, batch, args, rng, amp_dtype, amp_on, ladder)
            (out["loss"] / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sched.step()
            ema_f.update(student)
            ema_s.update(student)
            refresh_from_shadow(fast, ema_f.state_dict())
            refresh_from_shadow(slow, ema_s.state_dict())
            step += 1

            if throttle:
                torch.cuda.synchronize()
                pause = (time.time() - t_work) * (1.0 / args.gpu_util_cap - 1.0)
                if pause > 0:
                    time.sleep(min(pause, 5.0))
                t_work = time.time()

            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t_last
                t_last = time.time()
                row = {k: float(v.item()) for k, v in out.items()}
                row.update(step=step, lr=float(sched.get_last_lr()[0]),
                           sps=args.log_every / max(dt, 1e-9))
                if args.probe_every and (step % args.probe_every == 0 or step == 1):
                    student.eval()
                    row.update(rollout_probe(nets, probe, args, cfg, device,
                                             amp_dtype, amp_on))
                    student.train()
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                extra = ""
                if "probe_st_vs_tea" in row:
                    extra = (f" | st{args.steps_target}-vs-tea {row['probe_st_vs_tea']:.4f}"
                             f" st-gt {row['probe_st_vs_gt']:.4f}"
                             f" tea-gt {row['probe_tea_vs_gt']:.4f}")
                print(f"step {step} loss {row['loss']:.5f} vel {row['vel']:.5f} "
                      f"drift {row['drift']:.5f} pos_l1 {row['pos_l1']:.4f} "
                      f"skip {row['skip']:.2f} lr {row['lr']:.2e} "
                      f"{row['sps']:.2f} it/s{extra}", flush=True)
            if step % args.save_every == 0:
                checkpoint(f"step_{step:08d}.pt")
            if stop or step >= args.max_steps:
                break
    checkpoint("final.pt" if step >= args.max_steps else "interrupt.pt")
    print(f"done at step {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
