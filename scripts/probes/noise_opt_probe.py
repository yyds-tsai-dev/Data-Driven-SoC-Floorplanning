"""Initial-noise-optimization probe (flow OR direct-DDIM sampler).

Does optimizing the generative seed against a per-sample layout energy beat
spending the *same GPU time* on more random draws? And for the DDIM path: does a
seed optimized with a cheap 10-step unroll keep its advantage when rendered with
the production 25-step sampler?

Arms per case (raw candidate quality, no refine, no golden labels):
  (i)   baseline      : K antithetic random seeds, best-of-K.
  (ii)  treatment     : K_t antithetic seeds x noise-opt (differentiable unroll,
                        Adam + unit-grad + warmup/cosine + chi_d reg + argmin-
                        along-trajectory). Run for w_hpwl in {0, 0.1} to check
                        the per-sample normalization made hpwl safe. Rendered at
                        BOTH the unroll step count and --render-steps (transfer).
  (iv)  baseline_big  : K_big antithetic seeds -- same-compute control.

Health: energy_drop >= 0.30, ||z_T||/sqrt(d) in [0.9,1.1], anchors exact, no NaN.
Direction: optimized best overlap+viol beats same-compute baseline_big. Full-100
total_score_no_runtime is the only real gate (deferred to Phase B).

Run with cwd = FloorSet/iccad2026contest so litetestLoader resolves, e.g.
  cd FloorSet/iccad2026contest && PYTHONPATH=$PWD/..:. \
    ../../.venv/bin/python ../../scripts/probes/noise_opt_probe.py \
      --sampler direct --checkpoint <direct_control.pt>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "partner"))

DEFAULT_FLOW = ("/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/"
                "flow-matching-f1-f3/checkpoints/flow_matching_overfit/final.pt")
DEFAULT_DIRECT = str(ROOT / "partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt")
CASES = [0, 49, 79, 88, 93, 96, 99]


def load_case(dataset, idx):
    import torch
    sample = dataset[idx]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    polygons, _metrics = labels
    target_pos = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            target_pos.append((float(x_min), float(y_min),
                               float(x_max - x_min), float(y_max - y_min)))
        else:
            target_pos.append((0, 0, 1, 1))
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(block_count):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            if is_preplaced:
                tx, ty, tw, th = target_pos[i]
                opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
            elif is_fixed:
                _, _, tw, th = target_pos[i]
                opt_target_pos[i, 2] = tw
                opt_target_pos[i, 3] = th
    return (block_count, area_target, constraints, opt_target_pos,
            b2b_conn, p2b_conn, pins_pos)


def _overlap_area(P, n):
    import numpy as np
    x0, y0 = P[:n, 0], P[:n, 1]
    x1, y1 = x0 + P[:n, 2], y0 + P[:n, 3]
    ox = np.clip(np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None]), 0, None)
    oy = np.clip(np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None]), 0, None)
    m = ox * oy
    return (m.sum() - np.trace(m)) / 2.0


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    import numpy as np
    import torch
    from litetestLoader import FloorplanDatasetLiteTest

    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["flow", "direct"], default="direct")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cases", type=int, nargs="*", default=CASES)
    ap.add_argument("--rounds", type=int, default=10, help="noise-opt iters")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--unroll", type=int, default=10, help="opt unroll steps")
    ap.add_argument("--render-steps", type=int, default=25)
    ap.add_argument("--k-treat", type=int, default=8)
    ap.add_argument("--k-base", type=int, default=16)
    ap.add_argument("--k-big", type=int, default=160)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts/noise_opt/probe.json"))
    args = ap.parse_args()

    direct = args.sampler == "direct"
    ckpt = args.checkpoint or (DEFAULT_DIRECT if direct else DEFAULT_FLOW)
    if direct:
        os.environ["DIRECT_CKPT"] = ckpt
    else:
        os.environ["FLOW_CKPT"] = ckpt
        os.environ.setdefault("PARTNER_FLOW_SLOTS", "1")
    os.environ.setdefault("PARTNER_PRESCREEN_V", "1")

    from my_opt_claude import MyOptimizer
    from direct_train_claude import fast_condition
    from direct_model_claude import known_z_channels
    from diffusion_data import layout_scale, z_to_rectangles
    from physics_guidance_claude import build_context
    from noise_opt_claude import (NoiseOptConfig, optimize_noise,
                                  sample_flow_diff, sample_direct_diff,
                                  make_direct_sampler)

    opt = MyOptimizer()
    model = opt.direct_model if direct else opt.flow_model
    if model is None:
        raise SystemExit(f"{args.sampler} model failed to load from {ckpt}")
    dev = opt.device
    mcfg = opt.direct_cfg if direct else opt.flow_cfg
    sched = opt.direct_schedule if direct else None
    zdim = mcfg.z_dim
    print(f"sampler={args.sampler} device={dev} z_dim={zdim} z_repr={mcfg.z_repr} "
          f"unroll={args.unroll} render={args.render_steps} rounds={args.rounds}")

    dataset = FloorplanDatasetLiteTest("../")

    def build(n, at, cons, tpos, b2b, p2b, pins):
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), p2b.unsqueeze(0).to(dev),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=mcfg.relation_feat_dim, node_feat_dim=mcfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        ctx = build_context(at_d, cons_d, b2b.to(dev), scale, known)
        return cond, z_known, known, at_d, cons_d, tpos_d, ctx

    def ex(t, K):
        return t.expand(K, *t.shape[1:]).contiguous()

    def ex_cond(cond, K):
        return {k: (ex(v, K) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def antithetic(K, N, seed):
        gen = torch.Generator(device=dev).manual_seed(seed)
        half = (K + 1) // 2
        z = torch.randn((half, N, zdim), device=dev, generator=gen)
        return torch.cat([z, -z], dim=0)[:K]

    out = []
    for cid in args.cases:
        n, at, cons, tpos, b2b, p2b, pins = load_case(dataset, cid)
        cond, z_known, known, at_d, cons_d, tpos_d, ctx = build(
            n, at, cons, tpos, b2b, p2b, pins)
        N = cond["mask"].shape[1]
        valid = cond["mask"].unsqueeze(-1)
        has_known = bool(known.any())
        rec = {"case": cid, "n": n, "N_pad": N}

        def render(z_init, steps, seed):
            K = z_init.shape[0]
            with torch.no_grad():
                if direct:
                    kn = (torch.randn(steps, K, N, zdim, device=dev,
                                      generator=torch.Generator(device=dev).manual_seed(seed))
                          if has_known else None)
                    z0 = sample_direct_diff(
                        model, ex_cond(cond, K), sched, steps, z_init * ex(valid, K),
                        known_noise=kn, z_known=ex(z_known, K), known_mask=ex(known, K))
                else:
                    kn = (z_init * ex(known, K)) if has_known else None
                    z0 = sample_flow_diff(
                        model, ex_cond(cond, K), steps, "euler", z_init * ex(valid, K),
                        z_known=ex(z_known, K), known_mask=ex(known, K), known_noise=kn)
            rects = z_to_rectangles(
                z0, at_d.expand(K, -1), target_positions=tpos_d.expand(K, -1, -1),
                constraints=cons_d.expand(K, -1, -1), z_repr=mcfg.z_repr)
            return z0, [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]

        def metrics(preds):
            ov = [float(_overlap_area(p, n)) for p in preds]
            pen = opt._constraint_penalties(preds, n, at, cons) or [0.0] * len(preds)
            return dict(overlap_median=float(np.median(ov)), overlap_best=float(np.min(ov)),
                        viol_median=float(np.median(pen)), viol_best=float(np.min(pen)))

        def opt_sampler(K, unroll, seed):
            if direct:
                kn = (torch.randn(unroll, K, N, zdim, device=dev,
                                  generator=torch.Generator(device=dev).manual_seed(seed))
                      if has_known else None)
                return make_direct_sampler(
                    model, ex_cond(cond, K), sched, unroll, ex(z_known, K),
                    ex(known, K), kn, grad_checkpoint=args.grad_checkpoint)

            def _fn(seed_z):
                kn = (seed_z * ex(known, K)) if has_known else None
                return sample_flow_diff(
                    model, ex_cond(cond, K), unroll, "euler", seed_z,
                    z_known=ex(z_known, K), known_mask=ex(known, K), known_noise=kn)
            return _fn

        # ---- random arms -------------------------------------------------- #
        for tag, K, seed in (("baseline", args.k_base, 101),
                             ("baseline_big", args.k_big, 202)):
            try:
                _sync(); t0 = time.time()
                z_T = antithetic(K, N, seed) * ex(valid, K)
                _, preds = render(z_T, args.render_steps, seed + 7)
                _sync(); dt = time.time() - t0
                rec[tag] = dict(K=K, gpu_s=round(dt, 3), **metrics(preds))
            except RuntimeError as exc:
                rec[tag] = dict(K=K, error=str(exc)[:120])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---- treatment (hpwl A/B, transfer A/B) --------------------------- #
        K = args.k_treat
        ctx_k = ctx.expand(K)
        for hp in (0.0, 0.1):
            z_T0 = antithetic(K, N, 303) * ex(valid, K)
            cfg = NoiseOptConfig.from_env()
            cfg.rounds, cfg.lr, cfg.steps, cfg.solver, cfg.w_hpwl = (
                args.rounds, args.lr, args.unroll, "euler", hp)
            sampler = opt_sampler(K, args.unroll, 555)
            _sync(); t0 = time.time()
            z_T_opt, res = optimize_noise(
                None, ex_cond(cond, K), ctx_k, z_T0, ex(z_known, K), ex(known, K),
                cfg, sample_fn=sampler, return_result=True)
            _sync(); dt = time.time() - t0
            _, preds_u = render(z_T_opt, args.unroll, 900)
            z0r, preds_r = render(z_T_opt, args.render_steps, 901)

            energy = res.energy
            e0, e1 = float(energy[0].mean()), float(energy[-1].mean())
            drop = (e0 - e1) / (abs(e0) + 1e-9)
            free = ex(valid, K).expand(-1, -1, zdim) & (~ex(known, K))
            d_free = free.reshape(K, -1).sum(1).clamp_min(1).float()
            znorm = (z_T_opt * free).reshape(K, -1).norm(dim=1)
            sphere_dev = float((znorm / d_free.sqrt() - 1.0).abs().max())
            km = ex(known, K)
            anchor_err = float((z0r[km] - ex(z_known, K)[km]).abs().max()) if km.any() else 0.0

            key = f"treatment_hp{int(hp * 100):02d}"
            rec[key] = dict(
                K=K, w_hpwl=hp, iters_run=len(res.history), gpu_s=round(dt, 3),
                render_unroll=metrics(preds_u), render_full=metrics(preds_r),
                energy_first=e0, energy_last=e1, energy_drop_frac=round(drop, 4),
                sphere_dev_max=round(sphere_dev, 4), anchor_err_max=anchor_err,
                nan=bool(torch.isnan(z0r).any() or torch.isnan(energy).any()),
                history=res.history)

        printable = {k: ({kk: vv for kk, vv in v.items() if kk != "history"}
                         if isinstance(v, dict) else v) for k, v in rec.items()}
        print(json.dumps(printable, indent=1))
        out.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    # ---- summary ------------------------------------------------------- #
    def rollup(key, sub=None, field=None):
        vals = []
        for r in out:
            v = r.get(key)
            if not isinstance(v, dict):
                continue
            vv = v[sub][field] if sub else v.get(field)
            if vv is not None:
                vals.append(vv)
        return float(np.median(vals)) if vals else float("nan")

    print("\n=== mechanism health (per treatment arm) ===")
    for key in ("treatment_hp00", "treatment_hp10"):
        drops = [r[key]["energy_drop_frac"] for r in out if key in r]
        sph = [r[key]["sphere_dev_max"] for r in out if key in r]
        anc = [r[key]["anchor_err_max"] for r in out if key in r]
        nan = any(r[key]["nan"] for r in out if key in r)
        if drops:
            print(f"{key}: energy_drop min={min(drops):.3f} med={float(np.median(drops)):.3f} "
                  f"| sphere_max={max(sph):.4f} | anchor_max={max(anc):.1e} | nan={nan}")
    print("\n=== raw quality (median across cases): overlap_best / viol_best ===")
    print(f"baseline      : {rollup('baseline', field='overlap_best'):.4g} / "
          f"{rollup('baseline', field='viol_best'):.4g}  gpu={rollup('baseline', field='gpu_s'):.3g}")
    print(f"baseline_big  : {rollup('baseline_big', field='overlap_best'):.4g} / "
          f"{rollup('baseline_big', field='viol_best'):.4g}  gpu={rollup('baseline_big', field='gpu_s'):.3g}")
    for key in ("treatment_hp00", "treatment_hp10"):
        for sub in ("render_unroll", "render_full"):
            print(f"{key}/{sub}: {rollup(key, sub, 'overlap_best'):.4g} / "
                  f"{rollup(key, sub, 'viol_best'):.4g}  gpu={rollup(key, field='gpu_s'):.3g}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
