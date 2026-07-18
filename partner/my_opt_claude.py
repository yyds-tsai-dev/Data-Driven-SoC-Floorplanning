#!/usr/bin/env python3
"""Contest optimizer: diffusion-seeded column-slicing floorplanner.

Pipeline per test case:
  1. Deterministic heuristic seed layout (pin/graph-weighted centroids).
  2. Optional graph-conditioned diffusion refinement of the seed (the seed
     only provides relative-position hints; a few DDIM steps suffice).
  3. legalizer_claude.legalize_rectangles: column-slicing layout with
     hard-constraint guarantees plus a time-budgeted simulated-annealing
     search that minimizes HPWL / bbox area / soft violations.

The per-case time budget grows exponentially with block count (the contest
total score weights cases by exp(n/12), so large cases deserve nearly all
of the runtime), and averages roughly 5 s over the 100 validation cases.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import os

from iccad2026_evaluate import FloorplanOptimizer
from diffusion_data import build_condition, fp_sol_to_z0, layout_scale, z_to_rectangles
from diffusion_model import DiffusionSchedule, GraphDiffusionDenoiser, ModelConfig, ddim_refine
from legalizer_claude import (_ColumnOptimizer, _ensure_no_overlap,
                              _parse_constraints, _target, init_worker_pool,
                              legalize_rectangles, rectangles_from_z)
from refiner_claude import full_violations, refine_prediction

def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# Default keeps the historical half-core cap; PARTNER_POOL overrides.  On a
# 48-core box the eval leaves ~22 cores idle per case — a wider pool is
# more independent restart/refine draws per case at the SAME wall-clock
# (the same mechanism the doubled-budget oracle measured, minus the time).
N_RESTART_WORKERS = _env_int(
    "PARTNER_POOL", max(2, min(24, (os.cpu_count() or 4) // 2)))

Rect = Tuple[float, float, float, float]

DEFAULT_CHECKPOINT = (
    Path(__file__).parent / "checkpoints" / "diffusion_stable_xywh_order_2day" / "step_00080000.pt"
)
DIRECT_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "direct_v2"

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Historical defaults; env-overridable so experiments can rescale the
# per-case budget without editing code (defaults reproduce old behavior).
BUDGET_SCALE = _env_float("PARTNER_BUDGET_SCALE", 0.06)
BUDGET_TAU = _env_float("PARTNER_BUDGET_TAU", 20.0)
BUDGET_MIN = _env_float("PARTNER_BUDGET_MIN", 0.8)
BUDGET_MAX = _env_float("PARTNER_BUDGET_MAX", 24.0)


def _time_budget(block_count: int) -> float:
    """Exponential per-case budget: ~0.8 s for the smallest cases up to
    ~24 s for n=120; averages about 5 s over the validation set."""
    b = BUDGET_SCALE * math.exp(block_count / BUDGET_TAU)
    return max(BUDGET_MIN, min(BUDGET_MAX, b))


class MyOptimizer(FloorplanOptimizer):
    def __init__(
        self,
        verbose: bool = False,
        checkpoint_path: Optional[str] = None,
        steps: int = 4,
        refine_start_t: int = 250,
        device: Optional[str] = None,
    ):
        super().__init__(verbose=verbose)
        self.steps = steps
        self.refine_start_t = refine_start_t
        self.diffusion_seed = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[GraphDiffusionDenoiser] = None
        self.schedule: Optional[DiffusionSchedule] = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self._load_model()
        self.direct_model = None
        self._load_direct_model()
        # spawn the parallel-restart pool now so worker startup cost is not
        # charged to any test case
        init_worker_pool(N_RESTART_WORKERS)

    def _load_direct_model(self) -> None:
        """Load the direct-prediction denoiser (EMA weights) if a trained
        checkpoint exists.  Missing/broken checkpoints are silently ignored
        — the column pipeline works standalone."""
        if os.environ.get("DIRECT_OFF"):
            return
        try:
            from direct_model_claude import DirectDenoiser, DirectModelConfig
            env_path = os.environ.get("DIRECT_CKPT")
            if env_path:
                path = Path(env_path)
                if not path.exists():
                    return
            else:
                path = DIRECT_CHECKPOINT_DIR / "latest.pt"
                if not path.exists():
                    cands = sorted(DIRECT_CHECKPOINT_DIR.glob("step_*.pt"))
                    if not cands:
                        return
                    path = cands[-1]
            ck = torch.load(path, map_location=self.device, weights_only=False)
            cfg = DirectModelConfig(**{k: v for k, v in ck["model_config"].items()
                                       if k in DirectModelConfig.__dataclass_fields__})
            m = DirectDenoiser(cfg).to(self.device)
            m.load_state_dict(ck["model"])
            if "ema" in ck:
                sd = m.state_dict()
                for k in sd:
                    sd[k].copy_(ck["ema"][k].to(sd[k].dtype))
            m.eval()
            self.direct_model = m
            self.direct_cfg = cfg
            self.direct_schedule = DiffusionSchedule(cfg.timesteps, device=self.device)
            if self.verbose:
                print(f"loaded direct model step {ck.get('step')} from {path}")
        except Exception as exc:
            self.direct_model = None
            if self.verbose:
                print(f"direct model unavailable: {exc}")

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            if self.verbose:
                print(f"checkpoint not found: {self.checkpoint_path}; using heuristic init only")
            return
        for dev in ([self.device, torch.device("cpu")] if self.device.type == "cuda"
                    else [self.device]):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=dev, weights_only=False)
                cfg_dict = ckpt.get("model_config", {})
                cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                                     if k in ModelConfig.__dataclass_fields__})
                model = GraphDiffusionDenoiser(cfg).to(dev)
                model.load_state_dict(ckpt["model"])
                model.eval()
                self.device = dev
                self.model = model
                self.schedule = DiffusionSchedule(cfg.timesteps, device=dev)
                if self.verbose:
                    print(f"loaded diffusion checkpoint on {dev}: {self.checkpoint_path}")
                return
            except Exception as exc:
                if self.verbose:
                    print(f"failed to load checkpoint on {dev}: {exc}")
        self.model = None
        self.schedule = None

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Rect]:
        # self-heal: if a previous case's worker overrun killed the restart
        # pool, rebuild it (no-op when the pool is alive); losing the pool
        # for the rest of the run costs far more than one rebuild
        init_worker_pool(N_RESTART_WORKERS)
        start = time.time()
        budget = _time_budget(block_count)
        deadline = start + budget
        # Time-neutral vkill: the post-pass runs inside the SAME per-case
        # budget by carving a reserve out of the SA/refine deadline (total
        # wall-clock per case is unchanged).  VKILL_CARVE=0 restores the
        # legacy additive behavior; VKILL_OFF disables the pass entirely.
        vk_deadline = None
        if (os.environ.get("VKILL")
                and not os.environ.get("VKILL_OFF")
                and os.environ.get("VKILL_CARVE", "1") != "0"):
            if block_count >= int(_env_float("VKILL_MIN_N", 60.0)):
                reserve = min(0.25 * budget,
                              _env_float("VKILL_RESERVE_MAX", 5.0))
            else:
                reserve = min(0.1 * budget, 0.3)
            vk_deadline = deadline
            deadline = deadline - reserve

        area_targets = area_targets[:block_count].detach().float().cpu()
        constraints = constraints[:block_count].detach().float().cpu()
        target_positions = (
            target_positions[:block_count].detach().float().cpu()
            if target_positions is not None
            else torch.full((block_count, 4), -1.0)
        )
        b2b = b2b_connectivity.detach().float().cpu()
        p2b = p2b_connectivity.detach().float().cpu()
        pins = pins_pos.detach().float().cpu()

        try:
            seed_rects = _heuristic_init(area_targets, constraints, target_positions, b2b, p2b, pins)

            raw_rects = seed_rects
            if self.model is not None and self.schedule is not None:
                try:
                    raw_rects = self._diffusion_refine(
                        seed_rects, area_targets, b2b, p2b, pins,
                        constraints, target_positions, block_count,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"diffusion refine failed; using heuristic seed: {exc}")

            # direct-prediction candidates: when the restart pool is up,
            # hand a sampler to the legalizer — it reserves pool workers so
            # every prediction gets refined with a full budget in parallel.
            # The GPU sampling happens only after the seed-diffusion work
            # above AND after the column restarts are dispatched (GPU
            # contention here once starved mid-size cases' SA budget and
            # regressed the full validation to 1.262).
            import legalizer_claude as _lg
            direct_box: List = []
            th = None
            sample_fn = None
            # PARTNER_DIRECT_MIN (default 4.5 = historical): minimum
            # remaining budget for the direct channel.  The 4.5s gate
            # silently disables the pipeline's strongest channel for every
            # case below n~86 (budget = 0.06*e^(n/20)) — the whole 1.17x
            # mid band is column-only.  Sampling now runs concurrently
            # with the already-dispatched column restarts and small-n
            # batches are fast, so a much lower gate is viable.
            if self.direct_model is not None and (
                    deadline - time.time()) > _env_float(
                        "PARTNER_DIRECT_MIN", 4.5):
                if _lg._POOL is not None and _lg._POOL_READY:
                    sample_fn = (lambda K: self._sample_direct_preds(
                        block_count, area_targets, constraints,
                        target_positions, b2b, p2b, pins, K,
                        oversample=(deadline - time.time()) > 12.0))
                else:
                    # no pool: fall back to the sliced in-process thread
                    th = threading.Thread(
                        target=self._direct_worker,
                        args=(block_count, area_targets, constraints,
                              target_positions, b2b, p2b, pins,
                              deadline - 0.35, direct_box),
                        daemon=True)
                    th.start()

            column_out = legalize_rectangles(
                raw_rects, area_targets, constraints, target_positions,
                b2b_connectivity=b2b, p2b_connectivity=p2b, pins_pos=pins,
                deadline=deadline, sample_fn=sample_fn,
            )
            if th is not None:
                th.join(timeout=max(0.0, deadline - time.time()) + 0.1)
            out = (self._pick_best(column_out, direct_box)
                   if direct_box else column_out)
            t_kill = (vk_deadline if vk_deadline is not None
                      else time.time() + _env_float("VKILL_BUDGET", 6.0))
            return self._violation_kill(
                out, area_targets, constraints, target_positions,
                b2b, p2b, pins, t_kill)
        except Exception as exc:
            if self.verbose:
                print(f"column optimizer failed; using row fallback: {exc}")
            return _fallback_row(area_targets, constraints, target_positions)

    def _violation_kill(self, out, at, cons, tpos, b2b, p2b, pins,
                        t_kill: float):
        """Post-pass: targeted elimination of residual soft violations
        (boundary / grouping / MIB) via exact candidate enumeration with an
        evaluator-faithful acceptance test.  Contained: any failure —
        including an absent vkill_claude module — returns `out` unchanged.
        Runs until the absolute deadline `t_kill` (carved out of the case
        budget by solve, so per-case wall-clock is unchanged).  Opt-in via
        VKILL=1 (bare defaults keep the original pipeline byte-identical);
        VKILL_OFF=1 is a hard override."""
        if not os.environ.get("VKILL") or os.environ.get("VKILL_OFF"):
            return out
        try:
            from vkill_claude import kill_violations
            return kill_violations(
                out, at, cons, tpos, b2b, p2b, pins,
                budget=max(0.2, t_kill - time.time()),
                verbose=self.verbose)
        except Exception:
            return out

    def _sample_direct_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                             K, oversample: bool = True) -> List[np.ndarray]:
        """One batched DDIM pass (K layouts for ~1 layout's latency),
        prescreen-ordered: raw HPWL plus a strong penalty on total pairwise
        overlap (deeply overlapped predictions rarely survive the
        legalization rungs)."""
        from direct_train_claude import fast_condition
        from direct_model_claude import known_z_channels, sample_direct
        dev = self.device
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), p2b.unsqueeze(0).to(dev),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=self.direct_cfg.relation_feat_dim,
            node_feat_dim=self.direct_cfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        gen = torch.Generator(device=dev)
        gen.manual_seed(17)
        # sampling is batched, so oversample cheaply and let the prescreen
        # keep the best K for the (expensive) refine workers — but only
        # when the budget affords the extra sampling latency (on ~6 s
        # mid-size cases the doubled batch starved the refine workers and
        # lost cases direct used to win)
        # PARTNER_OVERSAMPLE: sampling is batched and runs concurrently
        # with the already-dispatched column restarts, so a bigger batch is
        # nearly free in wall-clock; the prescreen keeps the best K.
        os_f = 2
        try:
            os_f = max(1, int(float(os.environ.get("PARTNER_OVERSAMPLE",
                                                   "2"))))
        except ValueError:
            os_f = 2
        # cap the GPU batch: sampling happens before the refine dispatch,
        # so an oversized batch delays every refine worker (measured cliff
        # near ~2x the validated 48-sample batch)
        ks_cap = _env_int("PARTNER_KS_CAP", 56)
        K_s = max(K, min(os_f * K, ks_cap)) if oversample else K
        with torch.no_grad():
            cond_k = {k: (v.expand(K_s, *v.shape[1:]).contiguous()
                          if torch.is_tensor(v) else v)
                      for k, v in cond.items()}
            z = sample_direct(self.direct_model, cond_k,
                              self.direct_schedule, steps=50,
                              generator=gen,
                              z_known=z_known.expand(K_s, -1, -1),
                              known_mask=known.expand(K_s, -1, -1))
            rects = z_to_rectangles(
                z, at_d.expand(K_s, -1),
                target_positions=tpos_d.expand(K_s, -1, -1),
                constraints=cons_d.expand(K_s, -1, -1),
                z_repr=self.direct_cfg.z_repr)
        preds = [rects[k, :n].cpu().numpy().astype(np.float64)
                 for k in range(K_s)]

        w_b2b = (b2b[:n, :n].detach().cpu().numpy()
                 if b2b is not None else None)

        def _hpwl_of(P):
            # cheap standalone HPWL proxy for ordering (weighted b2b
            # Manhattan center distances; exact HPWL not needed to rank)
            if w_b2b is None:
                return 0.0
            cx = P[:, 0] + 0.5 * P[:, 2]
            cy = P[:, 1] + 0.5 * P[:, 3]
            iu, ju = np.nonzero(np.triu(w_b2b, 1))
            if not len(iu):
                return 0.0
            return float((w_b2b[iu, ju] * (np.abs(cx[iu] - cx[ju])
                                           + np.abs(cy[iu] - cy[ju]))).sum())

        def _ovl_frac(P):
            x0, y0 = P[:, 0], P[:, 1]
            x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
            ox = (np.minimum(x1[:, None], x1[None, :])
                  - np.maximum(x0[:, None], x0[None, :])).clip(min=0.0)
            oy = (np.minimum(y1[:, None], y1[None, :])
                  - np.maximum(y0[:, None], y0[None, :])).clip(min=0.0)
            ov = ox * oy
            ov[np.diag_indices(n)] = 0.0
            total = float((P[:, 2] * P[:, 3]).sum())
            return float(ov.sum()) / (2.0 * max(total, 1e-9))

        hps = [_hpwl_of(P) for P in preds]
        hp_min = max(min(hps), 1e-9)
        raw = [hps[k] / hp_min + 5.0 * _ovl_frac(preds[k])
               for k in range(len(preds))]

        # Constraint-aware prescreen (PARTNER_PRESCREEN_V=1): the plain
        # HPWL+overlap ranking is blind to boundary/grouping structure, so
        # predictions that legalize into violation-free layouts can lose
        # their refine slot to prettier-but-doomed ones.  Penalize coded
        # blocks far from their required wall and dispersed cluster groups.
        if os.environ.get("PARTNER_PRESCREEN_V"):
            try:
                w_v = float(os.environ.get("PARTNER_PRESCREEN_VW", "0.5"))
            except ValueError:
                w_v = 0.5
            _f, _p, _mib, clu, bnd = _parse_constraints(cons, n)
            groups = {}
            for i in range(n):
                if clu[i] > 0:
                    groups.setdefault(clu[i], []).append(i)
            diag = math.sqrt(max(float(at[:n].sum().item()), 1.0))

            def _viol_est(P):
                x0 = P[:, 0]
                y0 = P[:, 1]
                x1 = x0 + P[:, 2]
                y1 = y0 + P[:, 3]
                X0, Y0 = x0.min(), y0.min()
                X1, Y1 = x1.max(), y1.max()
                pen = 0.0
                n_b = 0
                for i in range(n):
                    code = bnd[i]
                    if not code:
                        continue
                    n_b += 1
                    d = 0.0
                    if code & 1:
                        d = max(d, float(x0[i] - X0))
                    if code & 2:
                        d = max(d, float(X1 - x1[i]))
                    if code & 4:
                        d = max(d, float(Y1 - y1[i]))
                    if code & 8:
                        d = max(d, float(y0[i] - Y0))
                    pen += min(d / diag, 1.0)
                if n_b:
                    pen /= n_b
                spread = 0.0
                for g in groups.values():
                    if len(g) < 2:
                        continue
                    gw = float(x1[g].max() - x0[g].min())
                    gh = float(y1[g].max() - y0[g].min())
                    asum = float((P[g, 2] * P[g, 3]).sum())
                    spread += min(max(0.0, gw * gh / max(asum, 1e-9) - 1.2),
                                  3.0)
                if groups:
                    spread /= max(sum(1 for g in groups.values()
                                      if len(g) >= 2), 1)
                return pen + spread

            for k in range(len(preds)):
                raw[k] += w_v * _viol_est(preds[k])

        order = sorted(range(len(preds)), key=lambda k: raw[k])
        return [preds[k] for k in order]

    def _direct_worker(self, n, at, cons, tpos, b2b, p2b, pins,
                       deadline, box) -> None:
        """No-pool fallback: sample K layouts and push each through the
        legalize+refine glue in this process; results land in `box`."""
        try:
            K = 8 if (deadline - time.time()) > 15.0 else 4
            preds = self._sample_direct_preds(n, at, cons, tpos,
                                              b2b, p2b, pins, K)
            for rank, P in enumerate(preds):
                left = deadline - time.time()
                if left < 1.0:
                    break
                rect_list = [tuple(map(float, P[i])) for i in range(n)]
                opt = _ColumnOptimizer(rect_list, at, cons, tpos,
                                       b2b, p2b, pins, deadline,
                                       seed=31 + rank)
                sub = time.time() + max(left / 3.0, min(left, 4.0))
                out = refine_prediction(opt, P, min(sub, deadline),
                                        seed=41 + rank)
                if out is not None:
                    box.append((out, opt))
        except Exception:
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _pick_best(self, column_out: List[Rect], box) -> List[Rect]:
        """Choose among the column result and direct candidates under the
        same proxy score the restart pool uses (plus full cluster checks)."""
        try:
            scorer = box[0][1]
            cands = [(np.array([list(r) for r in column_out]), column_out, False)]
            for out, _o in box:
                lst = [tuple(map(float, out[i])) for i in range(len(out))]
                cands.append((np.asarray(out, dtype=np.float64), lst, True))
            hps, areas, Vs = [], [], []
            for pos, _l, _d in cands:
                hps.append(scorer._hpwl(pos))
                areas.append(float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                                   * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min())))
                Vs.append(full_violations(scorer, pos))
            hp_ref = max(min(hps), 1e-9)
            scores = []
            for i in range(len(cands)):
                scores.append(
                    (1.0 + 0.5 * ((hps[i] - hp_ref) / hp_ref
                                  + max(0.0, areas[i] / scorer.area_ref - 1.0)))
                    * math.exp(2.0 * Vs[i] / scorer.n_soft_den))
            if os.environ.get("REFINER_DEBUG"):
                for i in range(len(cands)):
                    pos_i = cands[i][0]
                    px0, py0 = pos_i[:, 0], pos_i[:, 1]
                    px1, py1 = px0 + pos_i[:, 2], py0 + pos_i[:, 3]
                    b = scorer._bnd_idx
                    bnd = 0
                    if len(b):
                        codes = scorer._bnd_codes
                        eps = 1e-6
                        bad = ((codes & 1) != 0) & (np.abs(px0[b] - px0.min()) >= eps)
                        bad |= ((codes & 2) != 0) & (np.abs(px1[b] - px1.max()) >= eps)
                        bad |= ((codes & 4) != 0) & (np.abs(py1[b] - py1.max()) >= eps)
                        bad |= ((codes & 8) != 0) & (np.abs(py0[b] - py0.min()) >= eps)
                        bnd = int(bad.sum())
                    print(f"[pick] cand{i} direct={cands[i][2]} "
                          f"hp={hps[i]:.1f} area={areas[i]:.0f} "
                          f"V={Vs[i]} (bnd={bnd} rest={Vs[i]-bnd}) "
                          f"score={scores[i]:.4f}", flush=True)
            # a direct candidate must beat the column result by a clear
            # margin — marginal swaps are proxy-noise coin flips
            best_i = 0
            for i in range(1, len(cands)):
                if scores[i] < scores[best_i] and scores[i] < scores[0] * 0.985:
                    best_i = i
            pos, lst, is_direct = cands[best_i]
            if is_direct:
                locked = [scorer.kind[i] == 2 for i in range(scorer.n)]
                lst = _ensure_no_overlap(lst, locked)
            return lst
        except Exception:
            return column_out

    def _diffusion_refine(
        self,
        init_rects: List[Rect],
        area_targets: torch.Tensor,
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor,
        block_count: int,
    ) -> List[Rect]:
        at = area_targets.unsqueeze(0).to(self.device)
        cond = build_condition(
            at,
            b2b.unsqueeze(0).to(self.device),
            p2b.unsqueeze(0).to(self.device),
            pins.unsqueeze(0).to(self.device),
            constraints.unsqueeze(0).to(self.device),
            target_positions=target_positions.unsqueeze(0).to(self.device),
            relation_feat_dim=getattr(self.model.config, "relation_feat_dim", 0),
            node_feat_dim=getattr(self.model.config, "node_feat_dim", 13),
        )
        fp = torch.zeros((1, block_count, 4), dtype=torch.float32, device=self.device)
        for i, (x, y, w, h) in enumerate(init_rects):
            fp[0, i] = torch.tensor([w, h, x, y], dtype=torch.float32, device=self.device)
        z_repr = getattr(self.model.config, "z_repr", "xylogwh")
        z_init, _mask, _scale = fp_sol_to_z0(fp, at, z_repr=z_repr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.diffusion_seed)
        with torch.no_grad():
            z = ddim_refine(
                self.model,
                z_init,
                cond["node_feat"],
                cond["adj"],
                cond["mask"],
                start_t=self.refine_start_t,
                steps=max(1, int(self.steps)),
                schedule=self.schedule,
                generator=generator,
                rel_feat=cond.get("rel_feat"),
                z0_blend=0.20 if getattr(self.model.config, "predict_z0_head", False) else 0.0,
            )[0].cpu()
        return rectangles_from_z(z, area_targets, constraints, target_positions, z_repr=z_repr)


def _heuristic_init(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
) -> List[Rect]:
    """Pin/graph-weighted centroid layout used only as a relative-position
    seed for the diffusion model / column optimizer."""
    n = len(area_targets)
    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)

    shapes: List[Tuple[float, float]] = []
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            shapes.append((tw, th))
        else:
            area = max(float(area_targets[i]), 1e-6)
            shapes.append((math.sqrt(area), math.sqrt(area)))

    total_area = sum(w * h for w, h in shapes)
    scale = math.sqrt(max(total_area, 1.0))

    sx = [0.0] * n
    sy = [0.0] * n
    wsum = [0.0] * n
    n_pins = pins.shape[0]
    for edge in p2b:
        if edge[0] == -1:
            continue
        p = int(edge[0].item())
        b = int(edge[1].item())
        if not (0 <= b < n and 0 <= p < n_pins):
            continue
        px, py = float(pins[p, 0]), float(pins[p, 1])
        if px == -1.0 or py == -1.0:
            continue
        w = max(float(edge[2].item()), 0.0)
        sx[b] += w * px
        sy[b] += w * py
        wsum[b] += w

    no_pin = [i for i in range(n) if wsum[i] <= 1e-9]
    cols = max(1, int(math.sqrt(max(len(no_pin), 1))))
    cursor = 0
    cx = [0.0] * n
    cy = [0.0] * n
    for i in range(n):
        if wsum[i] > 1e-9:
            cx[i] = sx[i] / wsum[i]
            cy[i] = sy[i] / wsum[i]
        else:
            row, col = divmod(cursor, cols)
            cx[i] = (col + 0.5) * scale / cols
            cy[i] = (row + 0.5) * scale / cols
            cursor += 1

    nx = list(cx)
    ny = list(cy)
    deg = [0.0] * n
    for edge in b2b:
        if edge[0] == -1:
            continue
        i = int(edge[0].item())
        j = int(edge[1].item())
        if not (0 <= i < n and 0 <= j < n):
            continue
        w = max(float(edge[2].item()), 0.0)
        nx[i] += 0.25 * w * cx[j]
        ny[i] += 0.25 * w * cy[j]
        nx[j] += 0.25 * w * cx[i]
        ny[j] += 0.25 * w * cy[i]
        deg[i] += 0.25 * w
        deg[j] += 0.25 * w
    fcx = [nx[i] / (1.0 + deg[i]) for i in range(n)]
    fcy = [ny[i] / (1.0 + deg[i]) for i in range(n)]

    out: List[Rect] = []
    for i in range(n):
        w, h = shapes[i]
        tx, ty, _tw, _th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0:
            out.append((tx, ty, w, h))
        else:
            out.append((fcx[i] - 0.5 * w, fcy[i] - 0.5 * h, w, h))
    return out


def _fallback_row(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
) -> List[Rect]:
    """Guaranteed-feasible fallback: preplaced blocks stay put; every other
    block is placed in a single row strictly to the right of everything."""
    n = len(area_targets)
    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)
    out: List[Optional[Rect]] = [None] * n
    x_cursor = 0.0
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
            out[i] = (tx, ty, tw, th)
            x_cursor = max(x_cursor, tx + tw)
    x_cursor += 1.0
    for i in range(n):
        if out[i] is not None:
            continue
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            w, h = tw, th
        else:
            area = max(float(area_targets[i]), 1e-9)
            w = math.sqrt(area)
            h = area / w
        out[i] = (x_cursor, 0.0, w, h)
        x_cursor += w
    return [r for r in out]
