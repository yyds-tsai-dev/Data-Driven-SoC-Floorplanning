"""G1-a: data-free energy fine-tune of the direct-prediction denoiser.

Stage 2 of the DiOpt recipe.  Stage 1 is the shipped `direct_v2` checkpoint,
trained by imitation on golden coordinates; this stage never looks at a golden
coordinate again.  Each step:

    instances -> conditioning -> K differentiable samples (S=2, the production
    sampler) -> exact-area decode -> TFDL -> evaluator-faithful energy
    -> group soft-min -> direct gradient into the denoiser

Nothing here is RL.  The energy is a differentiable function of the layout by
construction, so the sample efficiency of a direct gradient beats a policy
gradient by one to two orders of magnitude and there is no KL coefficient to
tune between "no effect" and "collapse".

Objective per sample:

    obj = E                      evaluator-faithful log-cost (energy.py)
        + w_shape * shaping      long-range violation gradient, not faithful
        + w_pin   * drift/scale  preplaced blocks the compaction pushed off
        + w_disp  * disp/scale   how far TFDL had to move the raw prediction

The last two terms are what make the difference between this and "reward
fine-tuning on raw quality" (0805 survey's default-no): `drift` and `disp` are
exactly the parts of the prediction that the downstream legalizer cannot fix
for free, and `E` is measured *after* the projection, so the degenerate
overlapping blob that minimises a naive hpwl+area energy is not a minimum here.

Collapse defence (design memo Sec. 5), all three layers:
  * group soft-min over K samples, so diversity is rewarded rather than
    penalised -- only the best one or two samples carry weight, and making
    every sample identical wins nothing;
  * an L2 anchor to the warm-start weights;
  * `spread_K` monitoring with a hard stop: below 0.7x the base model's spread
    for `--collapse-steps` consecutive steps, training stops and the run is
    declared collapsed rather than quietly shipping a mode-collapsed sampler.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "partner", REPO / "FloorSet" / "iccad2026contest",
           REPO / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from icdc_engine import data as D                       # noqa: E402
from icdc_engine import energy as EN                    # noqa: E402
from icdc_engine import engine as G                     # noqa: E402
from icdc_engine import tfdl as T                       # noqa: E402
from icdc_engine.sampler import expand_cond, sample_differentiable   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(REPO / "partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"))
    p.add_argument("--out-dir", default=str(REPO / "partner/checkpoints/icdc_g1a"))
    p.add_argument("--index", default="")
    p.add_argument("--band", default="95,120")
    p.add_argument("--instances", type=int, default=3000,
                   help="size of the fixed instance pool drawn up front")
    p.add_argument("--batch", type=int, default=4, help="instances per step")
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--steps", type=int, default=2, help="denoising steps S")
    p.add_argument("--grad-steps", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--beta-group", type=float, default=8.0)
    p.add_argument("--w-shape", type=float, default=0.5)
    p.add_argument("--w-pin", type=float, default=1.0)
    p.add_argument("--w-disp", type=float, default=0.25)
    p.add_argument("--pin-tau", type=float, default=0.01,
                   help="saturation width of the drift/disp terms, in units "
                        "of the layout scale")
    p.add_argument("--w-anchor", type=float, default=1e4)
    p.add_argument("--collapse-ratio", type=float, default=0.7)
    p.add_argument("--collapse-steps", type=int, default=2000)
    p.add_argument("--train-last-k", type=int, default=0,
                   help="0 = full fine-tune; k = only the last k DiT blocks + head")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--vram-fraction", type=float, default=0.55)
    p.add_argument("--gpu-util-cap", type=float, default=0.95)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
def layout_objective(z, batch, args) -> Dict[str, torch.Tensor]:
    """z -> legal layout -> the per-sample objective and its parts."""
    rects = EN.decode_rects(z, batch["area"], batch["cons"], batch["tp"],
                            batch["scale"])
    mask = batch["area"] > 0
    pin = EN.preplaced_mask(batch["cons"], batch["tp"], batch["area"])
    code = (batch["cons"][..., 4].long() if batch["cons"].shape[-1] > 4
            else torch.zeros_like(batch["area"]).long())
    legal, drift = T.tfdl(rects, mask, pin, pin_xy=batch["tp"][..., :2],
                          boundary_code=code)
    out = EN.energy(legal, batch)
    scale = batch["scale"].clamp_min(1.0)
    drift_rel = drift.amax(dim=(1, 2)) / scale
    disp = ((legal[..., :2] - rects[..., :2]).abs().sum(dim=-1) * mask).sum(dim=1) \
        / mask.sum(dim=1).clamp_min(1) / scale
    out["drift"] = drift_rel
    out["disp"] = disp
    # BOTH auxiliary terms are saturated, and that is the whole lesson of the
    # first two runs.  Measured at lr 1e-5 and again at 2e-6: with a linear
    # `drift`/`disp` the objective blows up after a few hundred steps into
    # ever-larger layouts (area gap 0.18 -> 2.29, probe cost 2.92 -> 4.14).
    # Inflating the floorplan is the cheapest way to stop the compaction
    # pushing anything, so an unbounded "stay close to your own projection"
    # term has its optimum at infinite area and drags hpwl and bbox with it.
    # Saturating each at 1 lets them rank samples without ever outbidding the
    # evaluator-faithful quality term.
    tau = torch.full_like(drift_rel, args.pin_tau)
    out["obj"] = (out["E"] + args.w_shape * out["shaping"]
                  + args.w_pin * EN._sat(drift_rel, tau)
                  + args.w_disp * EN._sat(disp, tau))
    out["legal"] = legal
    return out


def spread_of(z, batch, K: int) -> torch.Tensor:
    """Mean pairwise L2 distance between the K samples' block centres,
    normalised by the layout scale -- the collapse tripwire."""
    with torch.no_grad():
        mask = batch["area"] > 0
        BK, N = mask.shape
        B = BK // K
        xy = z[..., :2].view(B, K, N, 2)
        m = mask.view(B, K, N)[:, :1]
        d = (xy.unsqueeze(1) - xy.unsqueeze(2)).norm(dim=-1)        # [B,K,K,N]
        d = (d * m.unsqueeze(1)).sum(dim=-1) / m.sum(dim=-1).clamp_min(1).unsqueeze(1)
        off = 1.0 - torch.eye(K, device=z.device).view(1, K, K)
        return ((d * off).sum(dim=(1, 2)) / max(K * (K - 1), 1)).mean()


def main():
    args = parse_args()
    band = tuple(int(v) for v in args.band.split(","))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if dev.type == "cuda" and 0 < args.vram_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction,
                                                   dev.index or 0)

    model, schedule, cfg, _ck = G.load_model(args.ckpt, device=dev)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point}
    from direct_diffusion_model import EMA
    ema = EMA(model, 0.999)

    params = []
    if args.train_last_k > 0:
        keep = {f"blocks.{i}." for i in
                range(max(0, cfg.layers - args.train_last_k), cfg.layers)}
        keep |= {"out.", "out_ada.", "out_norm."}
        for n, p_ in model.named_parameters():
            p_.requires_grad_(any(n.startswith(k) for k in keep))
            if p_.requires_grad:
                params.append(p_)
    else:
        params = [p_ for p_ in model.parameters()]
    n_train = sum(p_.numel() for p_ in params)
    print(f"fine-tuning {n_train/1e6:.1f}M / "
          f"{sum(p_.numel() for p_ in model.parameters())/1e6:.1f}M params",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)

    def lr_at(s):
        if s < args.warmup:
            return s / max(args.warmup, 1)
        p = (s - args.warmup) / max(args.max_steps - args.warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    print("drawing instance pool ...", flush=True)
    t0 = time.time()
    sampler = D.BandFileSampler(band=band, index_path=args.index or None,
                               seed=args.seed)
    pool = sampler.draw(args.instances)
    by_n: Dict[int, List[dict]] = {}
    for inst in pool:
        by_n.setdefault(inst["n"], []).append(inst)
    print(f"pool {len(pool)} instances, {len(by_n)} block counts, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- held-out validation instances (same distribution, never trained on)
    holdout = sampler.draw(64)

    rng = np.random.default_rng(args.seed)
    amp_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    log_path = out_dir / "train_log.jsonl"
    stop = {"flag": False}

    def on_sig(_s, _f):
        stop["flag"] = True
        print("signal: will checkpoint and exit", flush=True)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(s, on_sig)
        except (ValueError, AttributeError):
            pass

    def make_batch(insts):
        return D.collate(insts, device=dev, dtype=torch.float32)

    def run_group(batch, train: bool, seed=None):
        cond = G.build_cond(batch, cfg)
        z_known, known = G.known_channels(batch)
        condK = expand_cond(cond, args.K)
        zkK = z_known.repeat_interleave(args.K, dim=0)
        knK = known.repeat_interleave(args.K, dim=0)
        bK = {k: (v.repeat_interleave(args.K, dim=0) if torch.is_tensor(v) else v)
              for k, v in batch.items()}
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev).manual_seed(seed)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype,
                            enabled=bool(args.amp) and dev.type == "cuda"):
            z = sample_differentiable(model, condK, schedule, steps=args.steps,
                                      generator=gen, z_known=zkK, known_mask=knK,
                                      grad_steps=args.grad_steps if train else 0)
        z = z.float()
        out = layout_objective(z, bK, args)
        return z, out, bK

    # ---- base spread, measured with the warm-start weights
    with torch.no_grad():
        sp = []
        for i in range(4):
            b = make_batch(holdout[i * args.batch:(i + 1) * args.batch])
            z, _o, bK = run_group(b, train=False, seed=1000 + i)
            sp.append(float(spread_of(z, bK, args.K)))
        base_spread = float(np.mean(sp))
    print(f"base spread_K = {base_spread:.4f}", flush=True)

    # ---- scoring-band probe (real evaluator, the honest signal)
    probe = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "repo_ev", REPO / "scripts" / "iccad2026_evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        sys.modules["repo_ev"] = ev
        spec.loader.exec_module(ev)
        probe = [c for c in D.load_test_cases(ev) if c["n"] >= 100]
        print(f"probe: {len(probe)} scoring-band validation cases", flush=True)
    except Exception as exc:                                # pragma: no cover
        print("probe unavailable:", exc, flush=True)

    def probe_now(step):
        if not probe:
            return {}
        model.eval()
        pb = D.collate(probe, device=dev, dtype=torch.float32)
        bank, drift = G.sample_bank(model, schedule, pb, cfg, K=args.K,
                                    steps=args.steps, seed=7, device=dev)
        model.train()
        bank = bank.cpu().numpy()
        dr = drift.cpu().numpy()
        legal_costs, best = [], []
        for k, c in enumerate(probe):
            per = []
            for j in range(args.K):
                P = np.ascontiguousarray(bank[k, j, :c["n"]], dtype=np.float64)
                if not G.verify_hard_legal(P, c["area"].numpy(),
                                           c["cons"].numpy(), c["tp"].numpy())["ok"]:
                    continue
                m = ev.evaluate_solution(
                    {"positions": [tuple(map(float, r)) for r in P], "runtime": 1.0},
                    {"hpwl_baseline": c["hpwl_ref"], "area_baseline": c["area_ref"]},
                    c["cons"].to(torch.float32), c["b2b"].to(torch.float32),
                    c["p2b"].to(torch.float32), c["pins"].to(torch.float32),
                    c["area"].to(torch.float32), c["golden"], median_runtime=1.0)
                per.append(float(m.cost_no_runtime))
            legal_costs += per
            if per:
                best.append(min(per))
        row = {"step": step,
               "probe_legal_frac": float((dr <= 0).mean()),
               "probe_cost_med": float(np.median(legal_costs)) if legal_costs else float("nan"),
               "probe_bestK_med": float(np.median(best)) if best else float("nan"),
               "probe_cases_with_legal": len(best)}
        print(f"[probe {step}] legal {row['probe_legal_frac']*100:.0f}% "
              f"cost_med {row['probe_cost_med']:.4f} bestK_med "
              f"{row['probe_bestK_med']:.4f} cases {row['probe_cases_with_legal']}"
              f"/{len(probe)}", flush=True)
        return row

    model.train()
    collapse_run = 0
    hist = []
    probe0 = probe_now(0)
    t_last = time.time()
    for step in range(1, args.max_steps + 1):
        if stop["flag"]:
            break
        n = int(rng.choice(list(by_n)))
        picks = list(rng.choice(len(by_n[n]),
                                size=min(args.batch, len(by_n[n])),
                                replace=False))
        batch = make_batch([by_n[n][i] for i in picks])
        z, out, bK = run_group(batch, train=True)
        obj = out["obj"].view(-1, args.K)
        loss_group = -(1.0 / args.beta_group) * torch.logsumexp(
            -args.beta_group * obj, dim=1)
        loss = loss_group.mean()
        if args.w_anchor > 0:
            anc = sum(((p_ - base[n_]) ** 2).sum()
                      for n_, p_ in model.named_parameters()
                      if p_.requires_grad and n_ in base)
            loss = loss + args.w_anchor * anc / max(n_train, 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        ema.update(model)

        sp = float(spread_of(z, bK, args.K))
        collapse_run = collapse_run + 1 if sp < args.collapse_ratio * base_spread else 0
        if collapse_run >= args.collapse_steps:
            print(f"COLLAPSE STOP at step {step}: spread_K {sp:.4f} < "
                  f"{args.collapse_ratio}x{base_spread:.4f} for "
                  f"{collapse_run} steps", flush=True)
            break

        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "loss": float(loss), "n": n,
                   "obj": float(out["obj"].mean()), "E": float(out["E"].mean()),
                   "E_best": float(obj.min(dim=1).values.mean()),
                   "v_rel": float(out["v_rel"].mean()),
                   "g_h": float(out["g_h"].mean()), "g_a": float(out["g_a"].mean()),
                   "drift": float(out["drift"].mean()),
                   "legal_frac": float((out["drift"] <= 0).float().mean()),
                   "disp": float(out["disp"].mean()), "spread": sp,
                   "grad": gn, "lr": float(sched.get_last_lr()[0]),
                   "sps": args.log_every / max(time.time() - t_last, 1e-9)}
            t_last = time.time()
            hist.append(row)
            with log_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"step {step} loss {row['loss']:.4f} E {row['E']:.4f} "
                  f"Ebest {row['E_best']:.4f} V {row['v_rel']:.4f} "
                  f"gh {row['g_h']:.3f} ga {row['g_a']:.3f} "
                  f"legal {row['legal_frac']*100:.0f}% drift {row['drift']:.4f} "
                  f"disp {row['disp']:.4f} spread {sp:.4f} "
                  f"{row['sps']:.2f} it/s", flush=True)
        if step % args.eval_every == 0:
            r = probe_now(step)
            with log_path.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "model_config": cfg.__dict__, "step": step,
                        "args": vars(args), "base_spread": base_spread},
                       out_dir / "latest.pt")

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "model_config": cfg.__dict__, "step": args.max_steps,
                "args": vars(args), "base_spread": base_spread},
               out_dir / "final.pt")
    probe_now(-1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
