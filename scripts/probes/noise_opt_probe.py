"""Initial-noise-optimization probe: does optimizing the flow seed against a
per-sample layout energy beat spending the *same GPU time* on more random draws?

Arms per case (raw candidate quality, no refine, no golden labels):
  (i)   baseline      : flow, K antithetic random seeds, best-of-K.
  (ii)  treatment     : K_t antithetic seeds x noise-opt (differentiable Euler
                        unroll, Adam + unit-grad + warmup/cosine + chi_d reg +
                        argmin-along-trajectory), best-of-K_t.
  (iii) hybrid        : best-of-16 random -> top-4 by layout energy -> each
                        refined R_h iters -> best-of-4 (ReNO/DNO-style prescreen).
  (iv)  baseline_big  : flow, K_big antithetic seeds (~ treatment forward budget:
                        8 seeds x 10 iter x 2 (fwd+bwd) x steps ~= 160 x steps) --
                        the honest same-compute control (verdict gate on v1).

All seeds are antithetic (+/-z) pairs (arXiv 2506.06185): free negative-correlated
coverage. Mechanism health (this run, overfit ckpt -- quality NOT expected):
energy trace falls, anchors exact, no NaN, ||z_T||/sqrt(d) in [0.9,1.1], honest
GPU s. Verdict (v1 ckpt): treatment/hybrid best overlap+viol must beat the
same-compute baseline_big, not merely naive small-K, to earn a full-100 gate.

Run with cwd = FloorSet/iccad2026contest so litetestLoader resolves, e.g.
  cd FloorSet/iccad2026contest && \
  PYTHONPATH=.:$PWD/scripts .../.venv/bin/python \
    ../../scripts/probes/noise_opt_probe.py --checkpoint <flow.pt>
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

DEFAULT_CKPT = ("/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/"
                "flow-matching-f1-f3/checkpoints/flow_matching_overfit/final.pt")
CASES = [0, 49, 79, 88, 93, 96, 99]


# --------------------------------------------------------------------------- #
def load_case(dataset, idx):
    """Mirror ContestEvaluator's per-case unpacking (see pguide_quick_probe)."""
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


# --------------------------------------------------------------------------- #
def main():
    import numpy as np
    import torch
    from litetestLoader import FloorplanDatasetLiteTest

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--cases", type=int, nargs="*", default=CASES)
    ap.add_argument("--rounds", type=int, default=10, help="treatment iters")
    ap.add_argument("--hybrid-rounds", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=8, help="flow Euler steps")
    ap.add_argument("--k-treat", type=int, default=8)
    ap.add_argument("--k-base", type=int, default=16)
    ap.add_argument("--k-big", type=int, default=160)
    ap.add_argument("--hybrid-topk", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "artifacts/noise_opt/probe.json"))
    args = ap.parse_args()

    os.environ["FLOW_CKPT"] = args.checkpoint
    os.environ.setdefault("PARTNER_FLOW_SLOTS", "1")
    os.environ.setdefault("PARTNER_PRESCREEN_V", "1")   # enable viol estimate
    from my_opt_claude import MyOptimizer
    from direct_train_claude import fast_condition
    from direct_model_claude import known_z_channels
    from diffusion_data import layout_scale, z_to_rectangles
    from physics_guidance_claude import build_context, guidance_energy_per_sample
    from noise_opt_claude import NoiseOptConfig, optimize_noise, sample_flow_diff

    opt = MyOptimizer()
    if opt.flow_model is None:
        raise SystemExit(f"flow model failed to load from {args.checkpoint}")
    dev = opt.device
    fcfg = opt.flow_cfg
    model = opt.flow_model
    print(f"device={dev} flow z_dim={fcfg.z_dim} z_repr={fcfg.z_repr} "
          f"steps={args.steps} rounds={args.rounds} lr={args.lr}")

    dataset = FloorplanDatasetLiteTest("../")

    def build(n, at, cons, tpos, b2b, p2b, pins):
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), p2b.unsqueeze(0).to(dev),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=fcfg.relation_feat_dim,
            node_feat_dim=fcfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        ctx = build_context(at_d, cons_d, b2b.to(dev), scale, known)
        return cond, z_known, known, at_d, cons_d, tpos_d, ctx

    def expand(t, K):
        return t.expand(K, *t.shape[1:]).contiguous()

    def expand_cond(cond, K):
        return {k: (expand(v, K) if torch.is_tensor(v) else v)
                for k, v in cond.items()}

    def antithetic(K, N, zdim, seed):
        gen = torch.Generator(device=dev).manual_seed(seed)
        half = (K + 1) // 2
        z = torch.randn((half, N, zdim), device=dev, generator=gen)
        return torch.cat([z, -z], dim=0)[:K]

    def render(z0, at_d, tpos_d, cons_d, n, K):
        rects = z_to_rectangles(
            z0, at_d.expand(K, -1), target_positions=tpos_d.expand(K, -1, -1),
            constraints=cons_d.expand(K, -1, -1), z_repr=fcfg.z_repr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]

    def metrics(preds, n, at, cons):
        ov = [float(_overlap_area(p, n)) for p in preds]
        pen = opt._constraint_penalties(preds, n, at, cons) or [0.0] * len(preds)
        return dict(overlap_median=float(np.median(ov)), overlap_best=float(np.min(ov)),
                    viol_median=float(np.median(pen)), viol_best=float(np.min(pen)))

    out = []
    for cid in args.cases:
        n, at, cons, tpos, b2b, p2b, pins = load_case(dataset, cid)
        cond, z_known, known, at_d, cons_d, tpos_d, ctx = build(
            n, at, cons, tpos, b2b, p2b, pins)
        N, zdim = cond["mask"].shape[1], fcfg.z_dim
        valid = cond["mask"].unsqueeze(-1)
        rec = {"case": cid, "n": n, "N_pad": N}

        def sample_seeds(z_T):
            K = z_T.shape[0]
            zk = expand(z_known, K)
            km = expand(known, K)
            kn = z_T * km                      # known_noise carried by the seed
            with torch.no_grad():
                z0 = sample_flow_diff(model, expand_cond(cond, K), args.steps,
                                      "euler", z_T * expand(valid, K),
                                      z_known=zk, known_mask=km, known_noise=kn)
            return z0, km, zk

        # ---- random arms (i) baseline / (iv) baseline_big ---------------- #
        for tag, K, seed in (("baseline", args.k_base, 101),
                             ("baseline_big", args.k_big, 202)):
            try:
                _sync(); t0 = time.time()
                z_T = antithetic(K, N, zdim, seed) * expand(valid, K)
                z0, _, _ = sample_seeds(z_T)
                _sync(); dt = time.time() - t0
                preds = render(z0, at_d, tpos_d, cons_d, n, K)
                rec[tag] = dict(K=K, gpu_s=round(dt, 3), **metrics(preds, n, at, cons))
            except RuntimeError as exc:      # OOM guard
                rec[tag] = dict(K=K, error=str(exc)[:120])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---- (ii) treatment: K_t antithetic seeds x noise-opt ------------ #
        K = args.k_treat
        ctx_k = ctx.expand(K)
        z_T0 = antithetic(K, N, zdim, 303) * expand(valid, K)
        cfg = NoiseOptConfig.from_env()
        cfg.rounds, cfg.lr, cfg.steps, cfg.solver = (
            args.rounds, args.lr, args.steps, "euler")
        _sync(); t0 = time.time()
        z_T_opt, res = optimize_noise(
            model, expand_cond(cond, K), ctx_k, z_T0,
            expand(z_known, K), expand(known, K), cfg, return_result=True)
        _sync(); dt = time.time() - t0
        preds = render(res.z0, at_d, tpos_d, cons_d, n, K)

        energy = res.energy                       # [iters+1, K]
        e_first, e_last = float(energy[0].mean()), float(energy[-1].mean())
        drop = (e_first - e_last) / (abs(e_first) + 1e-9)
        free = expand(valid, K).expand(-1, -1, zdim) & (~expand(known, K))
        d_free = free.reshape(K, -1).sum(1).clamp_min(1).float()
        znorm = (z_T_opt * free).reshape(K, -1).norm(dim=1)
        sphere_dev = float((znorm / d_free.sqrt() - 1.0).abs().max())
        km = expand(known, K)
        anchor_err = float((res.z0[km] - expand(z_known, K)[km]).abs().max()) \
            if km.any() else 0.0

        rec["treatment"] = dict(
            K=K, iters_run=len(res.history), lr=args.lr, gpu_s=round(dt, 3),
            **metrics(preds, n, at, cons),
            energy_first=e_first, energy_last=e_last, energy_drop_frac=round(drop, 4),
            sphere_dev_max=round(sphere_dev, 4), anchor_err_max=anchor_err,
            nan=bool(torch.isnan(res.z0).any() or torch.isnan(energy).any()),
            history=res.history,
        )

        # ---- (iii) hybrid: best-of-16 -> top-k refine -------------------- #
        try:
            Kp = args.k_base
            z_T_pool = antithetic(Kp, N, zdim, 404) * expand(valid, Kp)
            z0_pool, _, _ = sample_seeds(z_T_pool)
            e_pool = guidance_energy_per_sample(
                z0_pool, ctx.expand(Kp), cfg.w_overlap, cfg.w_boundary,
                cfg.w_hpwl, cfg.w_group)
            top = torch.topk(e_pool, args.hybrid_topk, largest=False).indices
            z_T_top = z_T_pool[top].contiguous()
            Kh = z_T_top.shape[0]
            hcfg = NoiseOptConfig.from_env()
            hcfg.rounds, hcfg.lr, hcfg.steps, hcfg.solver = (
                args.hybrid_rounds, args.lr, args.steps, "euler")
            _sync(); t0 = time.time()
            z0_hyb = z0_pool                       # (pool already sampled)
            _, hres = optimize_noise(
                model, expand_cond(cond, Kh), ctx.expand(Kh), z_T_top,
                expand(z_known, Kh), expand(known, Kh), hcfg, return_result=True)
            _sync(); dt = time.time() - t0
            preds = render(hres.z0, at_d, tpos_d, cons_d, n, Kh)
            he_first = float(hres.energy[0].mean())
            he_last = float(hres.energy[-1].mean())
            rec["hybrid"] = dict(
                topk=Kh, iters_run=len(hres.history), gpu_s=round(dt, 3),
                **metrics(preds, n, at, cons),
                energy_first=he_first, energy_last=he_last,
                energy_drop_frac=round((he_first - he_last) / (abs(he_first) + 1e-9), 4))
        except RuntimeError as exc:
            rec["hybrid"] = dict(error=str(exc)[:120])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        printable = {k: (v if k != "treatment" else
                         {kk: vv for kk, vv in v.items() if kk != "history"})
                     for k, v in rec.items()}
        print(json.dumps(printable, indent=1))
        out.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    # ---- mechanism-health summary ------------------------------------- #
    tr = [r["treatment"] for r in out if "treatment" in r]
    drops = [t["energy_drop_frac"] for t in tr]
    sph = [t["sphere_dev_max"] for t in tr]
    anc = [t["anchor_err_max"] for t in tr]
    nan = any(t["nan"] for t in tr)
    print("\n=== mechanism health (overfit ckpt) ===")
    print(f"energy_drop_frac: min={min(drops):.3f} median={float(np.median(drops)):.3f}")
    print(f"sphere_dev_max:   max={max(sph):.4f} (target < 0.05)")
    print(f"anchor_err_max:   max={max(anc):.2e}")
    print(f"any NaN: {nan}")

    def rollup(arm, key):
        vals = [r[arm][key] for r in out if arm in r and key in r[arm]]
        return float(np.median(vals)) if vals else float("nan")
    print("\n=== raw quality (median across cases) ===")
    for arm in ("baseline", "treatment", "hybrid", "baseline_big"):
        print(f"{arm:13s} overlap_best={rollup(arm,'overlap_best'):.4g} "
              f"viol_best={rollup(arm,'viol_best'):.4g} "
              f"gpu_s={rollup(arm,'gpu_s'):.3g}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
