#!/usr/bin/env python3
"""t-resolved comparison of flow v1 vs v2 checkpoints, CPU only.

Removes the two confounds in the raw train_log comparison:
  * log composition (v2 puts 10% of samples at t in [0.98,1), which inflates
    logged `v` and deflates logged geometry terms),
  * metric definition (v2's logged x0 carries w_snr, v1's does not).

Both models see IDENTICAL batches, IDENTICAL noise, and are scored with the
v1 (unweighted) metric definitions on a fixed grid of t.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    repo = Path(args.repo)
    sys.path.insert(0, str(repo / "partner"))
    sys.path.insert(0, str(repo / "FloorSet" / "iccad2026contest"))
    sys.path.insert(0, str(repo / "FloorSet"))

    import direct_diffusion_train as V1
    import direct_diffusion_train_v2 as V2
    from diffusion_data import batch_to_device, fp_sol_to_z0, z_to_rectangles
    from diffusion_train import known_target_positions_from_fp
    from direct_diffusion_model import DirectDenoiser, DirectModelConfig, EMA
    from flow_matching_model import endpoint_from_velocity, flow_path

    device = torch.device("cpu")

    # ---- fixed evaluation batches (deterministic, no augmentation) --------
    loader = V1.get_training_dataloader(
        data_path=str(repo / "FloorSet") + "/",
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=False,
    )
    batches = []
    for k, batch in enumerate(loader):
        if k >= args.batches:
            break
        batches.append(batch_to_device(batch, device))
    print(f"loaded {len(batches)} batches", flush=True)

    def load_model(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = DirectModelConfig(**{k: v for k, v in ck["model_config"].items()
                                   if k in DirectModelConfig.__dataclass_fields__})
        m = DirectDenoiser(cfg)
        m.load_state_dict(ck["model"])
        if "ema" in ck:
            ema = EMA(m)
            ema.load_state_dict(ck["ema"])
            ema.copy_to(m)
        m.eval()
        return m, cfg, int(ck.get("step", -1)), ck.get("args", {}).get("training_method")

    t_grid = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.97, 0.99]
    results = {}

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, cfg, step, method = load_model(path)
        print(f"[{name}] step={step} method={method}", flush=True)
        rows = []
        for tv in t_grid:
            acc = {k: 0.0 for k in ("v", "x0", "ov", "bd", "cg", "mib",
                                    "mib_gap", "mib_exc", "hp", "pos_l1")}
            n = 0
            for bi, batch in enumerate(batches):
                area, b2b, p2b, pins, cons, _tree, fp, _m = batch
                tp = known_target_positions_from_fp(fp, cons)
                z0, mask, scale = fp_sol_to_z0(fp, area, z_repr="xyaspect")
                cond = V1.fast_condition(
                    area, b2b, p2b, pins, cons, target_positions=tp,
                    relation_feat_dim=cfg.relation_feat_dim,
                    node_feat_dim=cfg.node_feat_dim)
                B = area.shape[0]
                # identical noise across models and t: seeded per batch only
                g = torch.Generator(device="cpu").manual_seed(1000 + bi)
                noise = torch.randn(z0.shape, generator=g)
                t = torch.full((B,), float(tv))
                z_t, v_target = flow_path(z0, noise, t)
                z_t = z_t * mask.unsqueeze(-1)
                with torch.no_grad():
                    v = model(z_t, t * (cfg.timesteps - 1), cond["node_feat"],
                              cond["adj"], mask, rel_feat=cond["rel_feat"],
                              self_cond=None).float()
                mw = mask.unsqueeze(-1).float()
                v_loss = ((v - v_target).square() * mw).sum() / (
                    mw.sum().clamp_min(1) * v.shape[-1])
                z0_hat = endpoint_from_velocity(z_t, v, t)
                code = cons[..., 4].long() if cons.shape[-1] > 4 else torch.zeros_like(area).long()
                nw = torch.where((code != 0) & mask, torch.full_like(area, 2.0),
                                 torch.ones_like(area))
                cw = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, 4)
                # v1 definition: NO w_snr
                err = F.smooth_l1_loss(z0_hat, z0, reduction="none", beta=0.02)
                x0 = (err * cw * nw.unsqueeze(-1) * mw).sum() / (
                    nw.unsqueeze(-1) * mw * cw).sum().clamp_min(1)
                rects = z_to_rectangles(z0_hat, area, target_positions=tp,
                                        constraints=cons, z_repr="xyaspect")
                gt = torch.stack([fp[..., 2], fp[..., 3], fp[..., 0], fp[..., 1]], -1)
                # UNGATED geometry (raw quality at this t, no t^2 factor)
                ov = V1.overlap_fraction(rects, mask, scale).mean()
                bd = V1.boundary_touch(rects, cons, mask, scale).mean()
                cg = V1.cluster_gap(rects, gt, cons, mask, scale).mean()
                mib_p = V1.mib_aspect(rects, cons, mask)
                # v3 note: raw `mib` RISES when the model stops being pushed
                # more symmetric than golden, so "lower mib is better" is the
                # wrong read for a hinged run.  `mib_gap` (distance to golden)
                # and `mib_exc` (the hinged excess) are the directional ones.
                mib_g = V1.mib_aspect(gt, cons, mask)
                mib_gap = (mib_p - mib_g).abs().mean()
                mib_exc = F.relu(mib_p - mib_g).mean()
                mb = mib_p.mean()
                hpp = V2.hpwl_pair(rects, b2b, p2b, pins)
                hpg = V2.hpwl_pair(gt, b2b, p2b, pins).clamp_min(1.0)
                hp = (F.relu(hpp - hpg) / hpg).mean()
                pl = ((z0_hat[..., :2] - z0[..., :2]).abs().sum(-1) * mask).sum() / \
                    mask.sum().clamp_min(1)
                for k, val in (("v", v_loss), ("x0", x0), ("ov", ov), ("bd", bd),
                               ("cg", cg), ("mib", mb), ("mib_gap", mib_gap),
                               ("mib_exc", mib_exc), ("hp", hp), ("pos_l1", pl)):
                    acc[k] += float(val)
                n += 1
            row = {k: acc[k] / n for k in acc}
            row["t"] = tv
            rows.append(row)
            print(f"  t={tv:.2f} " + " ".join(f"{k}={row[k]:.5f}" for k in
                  ("v", "x0", "ov", "mib", "mib_gap", "mib_exc", "pos_l1")),
                  flush=True)
        results[name] = {"step": step, "method": method, "rows": rows}
        del model

    Path(args.output).write_text(json.dumps(results, indent=1))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
