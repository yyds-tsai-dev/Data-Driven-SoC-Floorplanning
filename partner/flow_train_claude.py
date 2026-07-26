#!/usr/bin/env python3
"""Direct-v2-compatible conditional flow-matching trainer.

The optimizer, data loader, checkpoint lifecycle, EMA, and shared-GPU
throttling are deliberately delegated to ``direct_train_claude``.  This
module changes the direct trainer's stochastic path from diffusion
v-prediction to straight-path velocity matching while retaining v2's
conditioning, augmentation, geometry, HPWL, and large-case objectives.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from flow_matching_claude import (
    endpoint_from_velocity,
    endpoint_time_weight,
    flow_path,
    sample_flow_t,
)

# Accepted flow training-method tags.  v2.1 is the current recipe; v1 and v2
# stay loadable so the candidate probe can still read the earlier weights.
FLOW_METHODS = ("flow_matching_v1", "flow_matching_v2", "flow_matching_v2_1")
TRAINING_METHOD = "flow_matching_v2_1"


def checkpoint_method(checkpoint):
    """Validate that a checkpoint was produced by a flow trainer."""
    method = checkpoint.get("args", {}).get("training_method")
    if method not in FLOW_METHODS:
        raise ValueError(
            "checkpoint must declare training_method in "
            "{flow_matching_v1, flow_matching_v2, flow_matching_v2_1}"
        )
    return method


def validate_resume_checkpoint(args):
    """Reject a non-flow resume checkpoint before V1 restores its state."""
    from pathlib import Path

    checkpoint_dir = Path(args.checkpoint_dir)
    resume = args.resume
    if not resume and not args.fresh and (checkpoint_dir / "latest.pt").exists():
        resume = "latest"
    if not resume:
        return
    path = checkpoint_dir / "latest.pt" if resume == "latest" else Path(resume)
    if path.exists():
        checkpoint_method(torch.load(path, map_location="cpu", weights_only=False))


def masked_flow_loss(predicted, target, z_t, z0, t, mask):
    """Return masked velocity MSE and the implied data endpoint."""
    weight = mask.unsqueeze(-1).to(predicted.dtype)
    velocity_loss = ((predicted - target).square() * weight).sum()
    velocity_loss = velocity_loss / (weight.sum().clamp_min(1) * predicted.shape[-1])
    endpoint = endpoint_from_velocity(z_t, predicted, t)
    return velocity_loss, endpoint


def flow_train_step(model, ema, unused_schedule, batch, args, rng, amp_dtype, amp_on):
    """Train one DirectDenoiser step against a linear noise-to-data path."""
    import direct_train_claude as V1
    import direct_train_v2_claude as V2
    from diffusion_data import fp_sol_to_z0, z_to_rectangles
    from direct_model_claude import known_z_channels
    from diffusion_train import known_target_positions_from_fp

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
    batch_size = area.shape[0]
    device = area.device
    t = sample_flow_t(
        batch_size, device, rng,
        term_prob=args.term_t_prob, term_band=args.term_band,
    )
    noise = torch.randn(z0.shape, device=device, generator=rng)
    z_t, velocity_target = flow_path(z0, noise, t)
    z_t = z_t * mask.unsqueeze(-1)
    t_model = t * (model.config.timesteps - 1)
    z_known, known = known_z_channels(area, cons, target_positions, scale)

    self_condition = None
    if model.config.self_conditioning and (
        float(torch.rand((), generator=rng, device=device)) < args.self_cond_prob
    ):
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_on
        ):
            first_velocity = model(
                z_t, t_model, cond["node_feat"], cond["adj"], mask,
                rel_feat=cond["rel_feat"], self_cond=None,
            )
        self_condition = endpoint_from_velocity(z_t, first_velocity.float(), t)
        self_condition[..., 2] = self_condition[..., 2].clamp(-3, 3)
        self_condition = torch.where(known, z_known, self_condition).detach()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
        velocity = model(
            z_t, t_model, cond["node_feat"], cond["adj"], mask,
            rel_feat=cond["rel_feat"], self_cond=self_condition,
        )
    velocity_loss, z0_hat = masked_flow_loss(
        velocity.float(), velocity_target, z_t, z0, t, mask
    )

    mask_weight = mask.unsqueeze(-1).to(velocity.dtype)
    code = cons[..., 4].long() if cons.shape[-1] > 4 else torch.zeros_like(area).long()
    node_weight = torch.where(
        (code != 0) & mask,
        torch.full_like(area, args.bnd_node_weight),
        torch.ones_like(area),
    )
    channel_weight = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device).view(1, 1, 4)
    # Endpoint time weighting.  Default 'none' (= v1): on the straight path the
    # endpoint residual is (1-t)*(velocity residual), already attenuated at high
    # noise, so the unweighted loss is a well-conditioned coordinate-space
    # auxiliary over the whole path.  'snr' reproduces the disproved v2 arm.
    w_time = endpoint_time_weight(t, args.x0_time_weighting, args.min_snr_gamma)
    w_time = w_time.view(batch_size, 1, 1).to(z0_hat.dtype)
    endpoint_error = F.smooth_l1_loss(z0_hat, z0, reduction="none", beta=0.02)
    endpoint_error = (
        endpoint_error * channel_weight * node_weight.unsqueeze(-1) * mask_weight * w_time
    )
    x0_loss = endpoint_error.sum() / (
        node_weight.unsqueeze(-1) * mask_weight * channel_weight
    ).sum().clamp_min(1)

    # Endpoint geometry becomes reliable as the path approaches data time.
    endpoint_weight = t.square()
    rects = z_to_rectangles(
        z0_hat, area, target_positions=target_positions, constraints=cons, z_repr="xyaspect"
    )
    gt_rects = torch.stack([fp[..., 2], fp[..., 3], fp[..., 0], fp[..., 1]], dim=-1)
    ov = (V1.overlap_fraction(rects, mask, scale) * endpoint_weight).mean()
    bd = (V1.boundary_touch(rects, cons, mask, scale) * endpoint_weight).mean()
    cg = (V1.cluster_gap(rects, gt_rects, cons, mask, scale) * endpoint_weight).mean()
    mb = (V1.mib_aspect(rects, cons, mask) * endpoint_weight).mean()

    hp_pred = V2.hpwl_pair(rects, b2b, p2b, pins)
    with torch.no_grad():
        hp_gt = V2.hpwl_pair(gt_rects, b2b, p2b, pins).clamp_min(1.0)
    hp_loss = (F.relu(hp_pred - hp_gt) / hp_gt * endpoint_weight).mean()

    n_blocks = mask.sum(dim=1).to(velocity.dtype)
    large_case_weight = torch.exp((n_blocks - 60.0) / 60.0).clamp(0.5, 3.0).mean()
    loss = large_case_weight * (
        args.v_loss_weight * velocity_loss
        + args.x0_loss_weight * x0_loss
        + args.overlap_loss_weight * ov
        + args.boundary_loss_weight * bd
        + args.cluster_loss_weight * cg
        + args.mib_loss_weight * mb
        + args.hpwl_loss_weight * hp_loss
    )

    with torch.no_grad():
        position_l1 = ((z0_hat[..., :2] - z0[..., :2]).abs().sum(-1) * mask).sum()
        position_l1 = position_l1 / mask.sum().clamp_min(1)
    return {
        "loss": loss,
        "v": velocity_loss.detach(),
        "x0": x0_loss.detach(),
        "ov": ov.detach(),
        "bd": bd.detach(),
        "cg": cg.detach(),
        "mib": mb.detach(),
        "pos_l1": position_l1.detach(),
        "hp": hp_loss.detach(),
    }


def parse_flow_extras(argv):
    """Split the flow-only knobs off an argv list.

    v2.1 defaults deliberately reproduce the validated v1 objective:
    ``x0_time_weighting='none'`` and ``term_t_prob=0`` (uniform t).  Both v2
    arms remain reachable by flag so an ablation needs no code edit.
    """
    import argparse

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--hpwl-loss-weight", type=float, default=0.30)
    extra.add_argument("--x0-time-weighting", choices=("none", "snr"), default="none")
    extra.add_argument("--term-t-prob", type=float, default=0.0)
    extra.add_argument("--term-band", type=float, default=0.02)
    return extra.parse_known_args(argv)


FLOW_EXTRA_FIELDS = ("hpwl_loss_weight", "x0_time_weighting", "term_t_prob", "term_band")


def main():
    """Configure Direct-v2 defaults, then reuse the V1 training lifecycle."""
    import inspect

    import direct_train_claude as V1

    argv = sys.argv[1:]

    def has(flag):
        return any(arg == flag or arg.startswith(flag + "=") for arg in argv)

    defaults = {
        "--d-model": "640",
        "--layers": "14",
        "--heads": "10",
        "--lr": "8e-5",
        "--warmup": "4000",
        "--ema-decay": "0.9998",
        "--node-feat-dim": "32",
        "--max-steps": "800000",
        "--checkpoint-dir": "checkpoints/flow_matching_v2_1",
    }
    for flag, value in defaults.items():
        if not has(flag):
            argv += [flag, value]
    sys.argv = [sys.argv[0]] + argv

    known, rest = parse_flow_extras(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest

    original_parse = V1.parse_args

    def parse_flow_args():
        args = original_parse()
        args.training_method = TRAINING_METHOD
        for field in FLOW_EXTRA_FIELDS:
            if not hasattr(args, field):
                setattr(args, field, getattr(known, field))
        return args

    V1.parse_args = parse_flow_args
    V1.train_step = flow_train_step
    official_training_loader = V1.get_training_dataloader
    accepted = inspect.signature(official_training_loader).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in accepted.values()
    )

    def compatible_training_loader(*args, **kwargs):
        """Forward only kwargs supported by the official train-only loader.

        The official FloorSet loader has no ``num_workers`` or ``pin_memory``
        parameters, so V1's optional throughput knobs are unavailable on this
        subset path.  Its train-only data path, subset size, batch size, and
        shuffle flag are forwarded unchanged.
        """
        if not accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in accepted}
        return official_training_loader(*args, **kwargs)

    V1.get_training_dataloader = compatible_training_loader
    validate_resume_checkpoint(parse_flow_args())
    V1.main()


if __name__ == "__main__":
    main()
