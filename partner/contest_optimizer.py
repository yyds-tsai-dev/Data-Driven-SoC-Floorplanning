#!/usr/bin/env python3
"""Contest optimizer: diffusion-seeded column-slicing floorplanner.

Pipeline per test case:
  1. Deterministic heuristic seed layout (pin/graph-weighted centroids).
  2. Optional graph-conditioned diffusion refinement of the seed (the seed
     only provides relative-position hints; a few DDIM steps suffice).
  3. column_sa_legalizer.legalize_rectangles: column-slicing layout with
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

from candidate_supply import CandidateBatch, allocate_quotas, rank_predictions
from iccad2026_evaluate import FloorplanOptimizer
from diffusion_data import build_condition, fp_sol_to_z0, layout_scale, z_to_rectangles
from diffusion_model import DiffusionSchedule, GraphDiffusionDenoiser, ModelConfig, ddim_refine
from column_sa_legalizer import (_ColumnOptimizer, _ensure_no_overlap,
                              _parse_constraints, _target, init_worker_pool,
                              legalize_rectangles, rectangles_from_z)
from layout_refiner import full_violations, refine_prediction

# The retrieval channel (retrieval_*) is opt-in: it only runs when both
# PARTNER_RETRIEVAL_INDEX and PARTNER_RETRIEVAL_SLOTS are set.  Its modules are
# imported lazily inside _sample_retrieval_preds so a deployment that ships
# only the active channels (Direct + flow + column) needs no retrieval sources.

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
# First R4 is intentionally fixed-capacity: retrieval may replace, but never
# add to, more than two Direct refinement slots.
FIRST_R4_RETRIEVAL_SLOTS = 2

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


def _select_ranked_source_quota(
    predictions: List[np.ndarray], sources: List[str], order: List[int],
    total: int, retrieval_quota: int,
) -> List[np.ndarray]:
    """Keep ranked candidates under fixed Direct/retrieval capacity quotas."""
    total = max(0, int(total))
    retrieval_quota = min(total, max(0, int(retrieval_quota)))
    direct_quota = total - retrieval_quota
    selected: List[np.ndarray] = []
    selected_indexes: set[int] = set()
    direct_count = 0
    retrieval_count = 0
    for index in order:
        source = sources[index]
        if source == "retrieval":
            if retrieval_count >= retrieval_quota:
                continue
            retrieval_count += 1
        elif source == "direct":
            if direct_count >= direct_quota:
                continue
            direct_count += 1
        else:
            continue
        selected.append(predictions[index])
        selected_indexes.add(index)

    # A missing retrieval source must not leave direct refinement capacity idle.
    if len(selected) < total and retrieval_count < retrieval_quota:
        for index in order:
            if sources[index] == "direct" and index not in selected_indexes:
                selected.append(predictions[index])
                selected_indexes.add(index)
                if len(selected) == total:
                    break
    return selected


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
        self.flow_model = None
        self._load_flow_model()
        self.retrieval_index = None
        self.retrieval_slots = 0
        configured_max_cost = _env_float("PARTNER_RETRIEVAL_MAX_COST", 2.0)
        self.retrieval_max_cost = configured_max_cost if math.isfinite(configured_max_cost) else 2.0
        retrieval_path = os.environ.get("PARTNER_RETRIEVAL_INDEX", "").strip()
        requested_slots = _env_int("PARTNER_RETRIEVAL_SLOTS", 0)
        if retrieval_path and requested_slots > 0:
            try:
                from retrieval_index import RetrievalIndex
                self.retrieval_index = RetrievalIndex.load(Path(retrieval_path))
                self.retrieval_slots = min(requested_slots, FIRST_R4_RETRIEVAL_SLOTS)
            except Exception as exc:
                self.retrieval_index = None
                self.retrieval_slots = 0
                if self.verbose:
                    print(f"retrieval index unavailable: {exc}")
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
            from direct_diffusion_model import DirectDenoiser, DirectModelConfig
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

    def _load_flow_model(self) -> None:
        """Load the opt-in flow-matching candidate source (off by default:
        requires both FLOW_CKPT and PARTNER_FLOW_SLOTS>0).  Any failure
        (missing file, wrong checkpoint tag, load error) silently disables
        the flow channel — it never harms the Direct/column pipeline."""
        self.flow_model = None
        path = os.environ.get("FLOW_CKPT", "").strip()
        slots = _env_int("PARTNER_FLOW_SLOTS", 0)
        if not path or slots <= 0 or not Path(path).exists():
            return
        try:
            from flow_matching_train import checkpoint_method
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            checkpoint_method(ckpt)
            from direct_diffusion_model import DirectDenoiser, DirectModelConfig
            cfg = DirectModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                                       if k in DirectModelConfig.__dataclass_fields__})
            model = DirectDenoiser(cfg).to(self.device)
            model.load_state_dict(ckpt.get("ema") or ckpt["model"])
            model.eval()
            self.flow_model = model
            self.flow_cfg = cfg
            if self.verbose:
                print(f"loaded flow model step {ckpt.get('step')} from {path}")
        except Exception as exc:
            self.flow_model = None
            if self.verbose:
                print(f"flow model unavailable: {exc}")

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
            import column_sa_legalizer as _lg
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
                    if self.retrieval_index is not None and self.retrieval_slots > 0:
                        sample_fn = (lambda K: self._sample_portfolio_preds(
                            block_count, area_targets, constraints,
                            target_positions, b2b, p2b, pins, K,
                            oversample=(deadline - time.time()) > 12.0))
                    else:
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
        including an absent violation_killer module — returns `out` unchanged.
        Runs until the absolute deadline `t_kill` (carved out of the case
        budget by solve, so per-case wall-clock is unchanged).  Opt-in via
        VKILL=1 (bare defaults keep the original pipeline byte-identical);
        VKILL_OFF=1 is a hard override."""
        if not os.environ.get("VKILL") or os.environ.get("VKILL_OFF"):
            return out
        try:
            from violation_killer import kill_violations
            return kill_violations(
                out, at, cons, tpos, b2b, p2b, pins,
                budget=max(0.2, t_kill - time.time()),
                verbose=self.verbose)
        except Exception:
            return out

    def _sample_direct_raw_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                                 K, oversample: bool = True) -> List[np.ndarray]:
        """Generate the baseline bounded Direct batch before prescreening."""
        from direct_diffusion_train import fast_condition
        from direct_diffusion_model import (known_z_channels, sample_direct,
                                         sample_direct_dpmpp)
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
        if (os.environ.get("PARTNER_NOISE_OPT") == "hybrid"
                and self.direct_model is not None):
            try:
                preds = self._noise_opt_hybrid_preds(
                    n, cond, z_known, known, scale, at_d, cons_d, tpos_d,
                    b2b.to(dev), K_s)
                if preds:
                    return preds
            except Exception as exc:  # never harm the default channel
                if self.verbose:
                    print(f"noise-opt hybrid failed: {exc}", file=sys.stderr)
        with torch.no_grad():
            cond_k = {k: (v.expand(K_s, *v.shape[1:]).contiguous()
                          if torch.is_tensor(v) else v)
                      for k, v in cond.items()}
            guide = None
            if os.environ.get("PARTNER_PHYSICS_GUIDE") == "1":
                from physics_guidance import (GuidanceConfig,
                                                     build_context,
                                                     make_guidance)
                ctx = build_context(at_d, cons_d, b2b.to(dev), scale,
                                    known).expand(K_s)
                guide = make_guidance(ctx, GuidanceConfig.from_env())
            solver = os.environ.get("PARTNER_DIRECT_SOLVER", "ddim")
            if solver == "dpmpp" and guide is None:
                z = sample_direct_dpmpp(
                    self.direct_model, cond_k, self.direct_schedule,
                    steps=_env_int("PARTNER_DDIM_STEPS", 50),
                    generator=gen,
                    z_known=z_known.expand(K_s, -1, -1),
                    known_mask=known.expand(K_s, -1, -1))
            else:
                z = sample_direct(self.direct_model, cond_k,
                                  self.direct_schedule,
                                  steps=_env_int("PARTNER_DDIM_STEPS", 50),
                                  generator=gen,
                                  z_known=z_known.expand(K_s, -1, -1),
                                  known_mask=known.expand(K_s, -1, -1),
                                  guidance=guide)
            rects = z_to_rectangles(
                z, at_d.expand(K_s, -1),
                target_positions=tpos_d.expand(K_s, -1, -1),
                constraints=cons_d.expand(K_s, -1, -1),
                z_repr=self.direct_cfg.z_repr)
        preds = [rects[k, :n].cpu().numpy().astype(np.float64)
                 for k in range(K_s)]

        # Opt-in flow-matching candidate source (default off): replaces a
        # fixed slice of the Direct batch with flow samples so the total
        # candidate count handed to the refine ladder is UNCHANGED
        # (replace-not-add) -- see candidate_supply.allocate_quotas.
        if self.flow_model is not None:
            flow_slots = _env_int("PARTNER_FLOW_SLOTS", 0)
            if flow_slots > 0 and preds:
                quotas = allocate_quotas(
                    len(preds),
                    {"direct": max(0, len(preds) - flow_slots), "flow": flow_slots},
                    ("direct", "flow"),
                )
                flow_n = quotas.get("flow", 0)
                if flow_n > 0:
                    try:
                        flow_preds = self._sample_flow_preds(
                            n, at, cons, tpos, b2b, p2b, pins, flow_n)
                        direct_n = quotas.get("direct", len(preds) - flow_n)
                        preds = preds[:direct_n] + flow_preds
                    except Exception:
                        pass  # flow failure never harms the Direct channel
        return preds

    def _sample_flow_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                           K) -> List[np.ndarray]:
        """Sample K layouts from the opt-in flow-matching model.

        Default path is unchanged. Opt-in flow-seed variants (all default off):
        PARTNER_FLOW_ANTITHETIC=1 (V-A, +/-z paired seeds), PARTNER_FLOW_ZORDER=1
        (V-B, zero-order neighborhood resample), PARTNER_FLOW_NOPT=hybrid
        (V-C/D, gradient noise-opt on the flow channel)."""
        from direct_diffusion_train import fast_condition
        from direct_diffusion_model import known_z_channels
        from flow_matching_model import sample_flow
        dev = self.device
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), p2b.unsqueeze(0).to(dev),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=self.flow_cfg.relation_feat_dim,
            node_feat_dim=self.flow_cfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        b2b_d = b2b.to(dev)
        if os.environ.get("PARTNER_FLOW_NOPT") == "hybrid":
            return self._flow_noise_opt_preds(
                n, cond, z_known, known, scale, at_d, cons_d, tpos_d, b2b_d, K)
        if os.environ.get("PARTNER_FLOW_ZORDER"):
            return self._flow_zorder_preds(
                n, cond, z_known, known, scale, at_d, cons_d, tpos_d, b2b_d, K)

        steps = _env_int("PARTNER_FLOW_STEPS", 8)
        solver = os.environ.get("PARTNER_FLOW_SOLVER", "euler")
        cond_k = {k: (v.expand(K, *v.shape[1:]).contiguous()
                      if torch.is_tensor(v) else v) for k, v in cond.items()}
        gen = torch.Generator(device=dev)
        gen.manual_seed(23)
        if os.environ.get("PARTNER_FLOW_ANTITHETIC"):
            # V-A: antithetic +/-z coverage (arXiv 2506.06185). known_noise is
            # shared within each pair so only the free seed is mirrored.
            from noise_optimization import sample_flow_diff
            N = cond["mask"].shape[1]
            zdim = self.flow_cfg.z_dim
            valid = cond_k["mask"].unsqueeze(-1)
            has_known = bool(known.any())
            half = (K + 1) // 2
            zf = torch.randn((half, N, zdim), device=dev, generator=gen)
            z_init = torch.cat([zf, -zf], dim=0)[:K] * valid
            kn = None
            if has_known:
                knh = torch.randn((half, N, zdim), device=dev, generator=gen)
                kn = torch.cat([knh, knh], dim=0)[:K] * known.expand(K, -1, -1)
            with torch.no_grad():
                z = sample_flow_diff(
                    self.flow_model, cond_k, steps, solver, z_init,
                    z_known=z_known.expand(K, -1, -1),
                    known_mask=known.expand(K, -1, -1), known_noise=kn)
        else:
            with torch.no_grad():
                z = sample_flow(
                    self.flow_model, cond_k, steps=steps, solver=solver,
                    generator=gen, z_known=z_known.expand(K, -1, -1),
                    known_mask=known.expand(K, -1, -1)).z
        rects = z_to_rectangles(
            z, at_d.expand(K, -1), target_positions=tpos_d.expand(K, -1, -1),
            constraints=cons_d.expand(K, -1, -1), z_repr=self.flow_cfg.z_repr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]

    def _flow_setup(self, cond, K):
        """Local (ex, ex_cond, N, zdim, valid, has_known, steps, solver)."""
        def ex(t, k):
            return t.expand(k, *t.shape[1:]).contiguous()

        def ex_cond(k):
            return {kk: (ex(v, k) if torch.is_tensor(v) else v)
                    for kk, v in cond.items()}
        return (ex, ex_cond, cond["mask"].shape[1], self.flow_cfg.z_dim,
                cond["mask"].unsqueeze(-1), bool(cond["mask"].numel()),
                _env_int("PARTNER_FLOW_STEPS", 8),
                os.environ.get("PARTNER_FLOW_SOLVER", "euler"))

    def _flow_zorder_preds(self, n, cond, z_known, known, scale,
                           at_d, cons_d, tpos_d, b2b, K) -> List[np.ndarray]:
        """V-B: zero-order neighborhood resample (Ma et al. 2501.09732).

        Sample a pool, pick the top-M by layout energy, draw sigma-Gaussian
        neighbors of each, re-render everything, and keep the best K by energy.
        Pure forward, stochastic (no gradient collapse -> dodges failure mode A;
        selection stays overlap-driven)."""
        import time as _time
        from physics_guidance import build_context, guidance_energy_per_sample
        from noise_optimization import sample_flow_diff
        dev = self.device
        z_repr = self.flow_cfg.z_repr
        ex, ex_cond, N, zdim, _v, _hk, steps, solver = self._flow_setup(cond, K)
        has_known = bool(known.any())
        M = _env_int("PARTNER_FLOW_ZORDER_M", 4)
        nb = _env_int("PARTNER_FLOW_ZORDER_NB", 3)
        sig = _env_float("PARTNER_FLOW_ZORDER_SIGMA", 0.1)
        Ksample = max(K, _env_int("PARTNER_FLOW_ZORDER_K", 16))
        w_ov = _env_float("PARTNER_FLOW_ZORDER_W_OVERLAP", 1.0)
        w_bd = _env_float("PARTNER_FLOW_ZORDER_W_BOUNDARY", 0.5)
        ctx = build_context(at_d, cons_d, b2b, scale, known)

        def render(z_init):
            k = z_init.shape[0]
            valid_k = cond["mask"].unsqueeze(-1).expand(k, -1, -1)
            kn = (z_init * ex(known, k)) if has_known else None
            with torch.no_grad():
                return sample_flow_diff(
                    self.flow_model, ex_cond(k), steps, solver, z_init * valid_k,
                    z_known=ex(z_known, k), known_mask=ex(known, k), known_noise=kn)

        def energy(z0):
            k = z0.shape[0]
            return guidance_energy_per_sample(z0, ctx.expand(k), w_ov, w_bd, 0.0, 0.0)

        t0 = _time.time()
        gen = torch.Generator(device=dev).manual_seed(23)
        valid_s = cond["mask"].unsqueeze(-1).expand(Ksample, -1, -1)
        z_seed = torch.randn((Ksample, N, zdim), device=dev, generator=gen) * valid_s
        z0 = render(z_seed)
        e = energy(z0)
        top = torch.topk(e, min(M, Ksample), largest=False).indices
        neigh = []
        for idx in top.tolist():
            base = z_seed[idx:idx + 1]
            for _ in range(nb):
                neigh.append(base + sig * torch.randn_like(base))
        if neigh:
            z_nb = torch.cat(neigh, dim=0)
            z0_nb = render(z_nb)
            z_all = torch.cat([z0, z0_nb], dim=0)
        else:
            z_all = z0
        e_all = energy(z_all)
        keep = torch.topk(e_all, min(K, z_all.shape[0]), largest=False).indices
        z_keep = z_all[keep].contiguous()
        Kk = z_keep.shape[0]
        rects = z_to_rectangles(
            z_keep, at_d.expand(Kk, -1), target_positions=tpos_d.expand(Kk, -1, -1),
            constraints=cons_d.expand(Kk, -1, -1), z_repr=z_repr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[flow-zorder] n={n} Ksample={Ksample} nb={len(neigh)} "
              f"K={Kk} gpu_s={_time.time() - t0:.2f}", file=sys.stderr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(Kk)]

    def _flow_noise_opt_preds(self, n, cond, z_known, known, scale,
                              at_d, cons_d, tpos_d, b2b, K) -> List[np.ndarray]:
        """V-C/D: gradient noise-opt on the flow channel with an INVERTED energy
        (hpwl-heavy). Tests whether targeting the refine-immovable wirelength
        dimension (not the refine-fixable overlap) revives gradient noise-opt,
        and whether the flow 8-step unroll avoids the DDIM-25 hpwl stall.
        Optional batch-repulsion (V-D) via PGUIDE_W_REP."""
        import time as _time
        from physics_guidance import build_context, guidance_energy_per_sample
        from noise_optimization import NoiseOptConfig, optimize_noise, sample_flow_diff
        dev = self.device
        z_repr = self.flow_cfg.z_repr
        ex, ex_cond, N, zdim, _v, _hk, steps, solver = self._flow_setup(cond, K)
        has_known = bool(known.any())
        M = min(_env_int("PARTNER_FLOW_NOPT_TOPM", 4), K)
        w_rep = _env_float("PGUIDE_W_REP", 0.0)
        sig_rep = _env_float("PGUIDE_SIGMA_REP", 0.1)
        ctx = build_context(at_d, cons_d, b2b, scale, known)
        cfg = NoiseOptConfig.from_env()
        cfg.rounds = _env_int("PARTNER_FLOW_NOPT_ITERS", 10)
        cfg.steps, cfg.solver = steps, solver
        cfg.w_hpwl = _env_float("PARTNER_FLOW_NOPT_W_HPWL", 1.0)
        cfg.w_overlap = _env_float("PARTNER_FLOW_NOPT_W_OVERLAP", 0.2)
        cfg.w_boundary = _env_float("PARTNER_FLOW_NOPT_W_BOUNDARY", 0.2)
        cfg.w_group = 0.0

        def render(z_init):
            k = z_init.shape[0]
            valid_k = cond["mask"].unsqueeze(-1).expand(k, -1, -1)
            kn = (z_init * ex(known, k)) if has_known else None
            with torch.no_grad():
                return sample_flow_diff(
                    self.flow_model, ex_cond(k), steps, solver, z_init * valid_k,
                    z_known=ex(z_known, k), known_mask=ex(known, k), known_noise=kn)

        def base_energy(z0):
            k = z0.shape[0]
            return guidance_energy_per_sample(
                z0, ctx.expand(k), cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, 0.0)

        t0 = _time.time()
        gen = torch.Generator(device=dev).manual_seed(23)
        valid_K = cond["mask"].unsqueeze(-1).expand(K, -1, -1)
        z_seed = torch.randn((K, N, zdim), device=dev, generator=gen) * valid_K
        z0 = render(z_seed)
        e = base_energy(z0)
        best = torch.topk(e, M, largest=False).indices
        worst = torch.topk(e, M, largest=True).indices

        def flow_sampler(seed):
            kn = (seed * ex(known, M)) if has_known else None
            return sample_flow_diff(
                self.flow_model, ex_cond(M), steps, solver, seed,
                z_known=ex(z_known, M), known_mask=ex(known, M), known_noise=kn)

        energy_fn = None
        if w_rep > 0.0:
            ctx_M = ctx.expand(M)

            def energy_fn(z0m):  # V-D: SVGD-style batch repulsion
                base = guidance_energy_per_sample(
                    z0m, ctx_M, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, 0.0)
                x = z0m[..., 0] * ctx_M.scale.view(-1, 1)
                y = z0m[..., 1] * ctx_M.scale.view(-1, 1)
                m = ctx_M.mask.to(z0m.dtype)
                cx = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
                cy = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
                diag = ctx_M.scale.clamp_min(1.0)
                d2 = ((cx[:, None] - cx[None]) ** 2 + (cy[:, None] - cy[None]) ** 2)
                rep = torch.exp(-d2 / (sig_rep * diag.mean()) ** 2)
                rep = rep - torch.diag(torch.diagonal(rep))
                return base + w_rep * rep.sum(dim=1)

        z_opt = optimize_noise(
            None, ex_cond(M), ctx.expand(M), z_seed[best].contiguous(),
            ex(z_known, M), ex(known, M), cfg, energy_fn=energy_fn,
            sample_fn=flow_sampler)
        z_ref = render(z_opt)

        z_out = z0.clone()
        for j, w in enumerate(worst.tolist()):
            z_out[w] = z_ref[j]
        rects = z_to_rectangles(
            z_out, at_d.expand(K, -1), target_positions=tpos_d.expand(K, -1, -1),
            constraints=cons_d.expand(K, -1, -1), z_repr=z_repr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[flow-nopt] n={n} K={K} M={M} iters={cfg.rounds} steps={steps} "
              f"w_rep={w_rep} gpu_s={_time.time() - t0:.2f}", file=sys.stderr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]

    def _noise_opt_hybrid_preds(self, n, cond, z_known, known, scale,
                                at_d, cons_d, tpos_d, b2b, K_s) -> List[np.ndarray]:
        """Opt-in (PARTNER_NOISE_OPT=hybrid) initial-noise optimization.

        Sample K_s candidates at the production step count, rank by the
        per-sample layout energy, gradient-optimize the initial noise of the
        best M via a cheap differentiable unroll, re-render those M at the
        production step count, and REPLACE the worst M of the pool with them
        (candidate count unchanged -- replace-not-add). Any failure raises and
        the caller falls back to the untouched default sampler."""
        import time as _time
        from physics_guidance import build_context, guidance_energy_per_sample
        from noise_optimization import (NoiseOptConfig, optimize_noise,
                                      sample_direct_diff, make_direct_sampler)
        dev = self.device
        model, sched = self.direct_model, self.direct_schedule
        zdim, z_repr = self.direct_cfg.z_dim, self.direct_cfg.z_repr
        N = cond["mask"].shape[1]
        valid = cond["mask"].unsqueeze(-1)
        has_known = bool(known.any())
        render_steps = _env_int("PARTNER_DDIM_STEPS", 50)
        unroll = _env_int("PARTNER_NOPT_UNROLL", 10)
        iters = _env_int("PARTNER_NOPT_ITERS", 10)
        M = min(_env_int("PARTNER_NOPT_TOPM", 4), K_s)
        gc = os.environ.get("PARTNER_NOPT_GRAD_CKPT", "1") != "0"

        def ex(t, k):
            return t.expand(k, *t.shape[1:]).contiguous()

        def ex_cond(k):
            return {kk: (ex(v, k) if torch.is_tensor(v) else v)
                    for kk, v in cond.items()}

        def render(z_init, k, seed):
            with torch.no_grad():
                kn = (torch.randn(render_steps, k, N, zdim, device=dev,
                                  generator=torch.Generator(device=dev).manual_seed(seed))
                      if has_known else None)
                return sample_direct_diff(
                    model, ex_cond(k), sched, render_steps, z_init * ex(valid, k),
                    known_noise=kn, z_known=ex(z_known, k), known_mask=ex(known, k))

        t0 = _time.time()
        ctx = build_context(at_d, cons_d, b2b, scale, known)
        cfg = NoiseOptConfig.from_env()
        cfg.rounds, cfg.steps = iters, unroll

        # 1. sample the K_s-candidate pool (production step count), keep seeds
        gen = torch.Generator(device=dev).manual_seed(_env_int("PARTNER_NOPT_SEED", 20))
        z_seed = torch.randn((K_s, N, zdim), device=dev, generator=gen) * ex(valid, K_s)
        z_pool = render(z_seed, K_s, _env_int("PARTNER_NOPT_SEED", 20) + 1)
        e_pool = guidance_energy_per_sample(
            z_pool, ctx.expand(K_s), cfg.w_overlap, cfg.w_boundary,
            cfg.w_hpwl, cfg.w_group)
        best = torch.topk(e_pool, M, largest=False).indices
        worst = torch.topk(e_pool, M, largest=True).indices

        # 2. optimize the best-M seeds with a cheap unroll
        kn_opt = (torch.randn(unroll, M, N, zdim, device=dev,
                              generator=torch.Generator(device=dev).manual_seed(41))
                  if has_known else None)
        sampler = make_direct_sampler(
            model, ex_cond(M), sched, unroll, ex(z_known, M), ex(known, M),
            kn_opt, grad_checkpoint=gc)
        z_opt = optimize_noise(
            None, ex_cond(M), ctx.expand(M), z_seed[best].contiguous(),
            ex(z_known, M), ex(known, M), cfg, sample_fn=sampler)

        # 3. re-render the optimized seeds at production step count, replace worst
        z_ref = render(z_opt, M, 71)
        rects_pool = z_to_rectangles(
            z_pool, at_d.expand(K_s, -1), target_positions=tpos_d.expand(K_s, -1, -1),
            constraints=cons_d.expand(K_s, -1, -1), z_repr=z_repr)
        rects_ref = z_to_rectangles(
            z_ref, at_d.expand(M, -1), target_positions=tpos_d.expand(M, -1, -1),
            constraints=cons_d.expand(M, -1, -1), z_repr=z_repr)
        preds = [rects_pool[k, :n].cpu().numpy().astype(np.float64) for k in range(K_s)]
        for j, w in enumerate(worst.tolist()):
            preds[w] = rects_ref[j, :n].cpu().numpy().astype(np.float64)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[noise-opt] n={n} K_s={K_s} M={M} iters={iters} unroll={unroll} "
              f"render={render_steps} gpu_s={_time.time() - t0:.2f}", file=sys.stderr)
        return preds

    def _constraint_penalties(self, preds, n, at, cons):
        if not os.environ.get("PARTNER_PRESCREEN_V"):
            return None
        _f, _p, _mib, clu, bnd = _parse_constraints(cons, n)
        groups = {}
        for i in range(n):
            if clu[i] > 0:
                groups.setdefault(clu[i], []).append(i)
        diag = math.sqrt(max(float(at[:n].sum().item()), 1.0))

        def _viol_est(P):
            x0, y0 = P[:, 0], P[:, 1]
            x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
            X0, Y0, X1, Y1 = x0.min(), y0.min(), x1.max(), y1.max()
            pen = 0.0
            n_b = 0
            for i in range(n):
                code = bnd[i]
                if not code:
                    continue
                n_b += 1
                distance = 0.0
                if code & 1:
                    distance = max(distance, float(x0[i] - X0))
                if code & 2:
                    distance = max(distance, float(X1 - x1[i]))
                if code & 4:
                    distance = max(distance, float(Y1 - y1[i]))
                if code & 8:
                    distance = max(distance, float(y0[i] - Y0))
                pen += min(distance / diag, 1.0)
            if n_b:
                pen /= n_b
            spread = 0.0
            for group in groups.values():
                if len(group) < 2:
                    continue
                group_width = float(x1[group].max() - x0[group].min())
                group_height = float(y1[group].max() - y0[group].min())
                group_area = float((P[group, 2] * P[group, 3]).sum())
                spread += min(max(0.0, group_width * group_height / max(group_area, 1e-9) - 1.2), 3.0)
            if groups:
                spread /= max(sum(len(group) >= 2 for group in groups.values()), 1)
            return pen + spread

        return [_viol_est(prediction) for prediction in preds]

    def _rank_portfolio(self, preds, n, at, cons, b2b):
        w_b2b = b2b[:n, :n].detach().cpu().numpy() if b2b is not None else None
        penalties = self._constraint_penalties(preds, n, at, cons)
        return rank_predictions(
            preds,
            at[:n].detach().cpu().numpy(),
            w_b2b if w_b2b is not None else np.zeros((n, n), dtype=np.float64),
            constraint_penalties=penalties,
            violation_weight=(_env_float("PARTNER_PRESCREEN_VW", 0.5)
                              if penalties is not None else 0.0),
        )

    def _sample_direct_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                             K, oversample: bool = True) -> List[np.ndarray]:
        """Prescreen the Direct batch exactly as the reviewed baseline does."""
        preds = self._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K, oversample)
        return [preds[index] for index in self._rank_portfolio(preds, n, at, cons, b2b)]

    def _empty_retrieval_batch(self, started: float, rejected: int = 0,
                               query_s: float = 0.0) -> CandidateBatch:
        return CandidateBatch(
            "retrieval", [], max(0.0, time.perf_counter() - started),
            {"source_ids": [], "retrieval_distances": [], "transforms": [],
             "match_costs": [], "match_confidences": [], "query_s": query_s,
             "match_s": 0.0, "transfer_s": 0.0, "rejected": rejected},
        )

    def _sample_retrieval_preds(self, n, at, cons, tpos, b2b, p2b, pins, K) -> CandidateBatch:
        """Query same-N training layouts and transfer one D4 match per source."""
        started = time.perf_counter()
        limit = min(
            max(0, int(K)), max(0, int(self.retrieval_slots)),
            FIRST_R4_RETRIEVAL_SLOTS,
        )
        if limit <= 0 or self.retrieval_index is None:
            return self._empty_retrieval_batch(started)

        try:
            from retrieval_features import extract_retrieval_features
            from retrieval_matching import match_blocks
            from retrieval_transfer import (remap_boundary_node_features,
                                                   transfer_layout)
            area = at[:n].detach().cpu().numpy()
            constraints = cons[:n].detach().cpu().numpy()
            target_positions = tpos[:n].detach().cpu().numpy()
            b2b_array = b2b.detach().cpu().numpy()
            p2b_array = p2b.detach().cpu().numpy()
            pins_array = pins.detach().cpu().numpy()
            target_features = extract_retrieval_features(
                area, b2b_array, p2b_array, pins_array, constraints, target_positions
            )
        except Exception:
            return self._empty_retrieval_batch(started)
        query_started = time.perf_counter()
        try:
            retrieved = self.retrieval_index.query(n, target_features.global_vector, top_k=limit)
        except Exception:
            return self._empty_retrieval_batch(
                started, query_s=time.perf_counter() - query_started
            )
        query_s = time.perf_counter() - query_started

        predictions: List[np.ndarray] = []
        source_ids: List[int] = []
        distances: List[float] = []
        transforms: List[str] = []
        match_costs: List[float] = []
        confidences: List[float] = []
        match_s = 0.0
        transfer_s = 0.0
        rejected = 0
        transform_names = ("identity", "mirror_x", "mirror_y", "transpose")
        for source_position in range(min(limit, len(retrieved.source_ids))):
            choices = []
            source_nodes = retrieved.node_features[source_position]
            for transform_order, transform in enumerate(transform_names):
                match_started = time.perf_counter()
                try:
                    result = match_blocks(
                        remap_boundary_node_features(source_nodes, transform),
                        target_features.node_matrix, max_cost=self.retrieval_max_cost,
                    )
                except Exception:
                    result = None
                match_s += time.perf_counter() - match_started
                if (result is not None and result.accepted
                        and math.isfinite(result.total_cost)):
                    choices.append((float(result.total_cost), transform_order, transform, result))
            if not choices:
                rejected += 1
                continue
            _cost, _order, transform, result = min(choices, key=lambda item: item[:2])
            transfer_started = time.perf_counter()
            try:
                prediction = transfer_layout(
                    retrieved.fp_xywh[source_position], result.target_to_source,
                    area, constraints, target_positions, transform,
                )
                fixed_or_preplaced = (constraints[:, 0] != 0) | (constraints[:, 1] != 0)
                has_shape = fixed_or_preplaced & (target_positions[:, 2] > 0) & (target_positions[:, 3] > 0)
                preplaced = ((constraints[:, 1] != 0) & (target_positions[:, 0] >= 0)
                             & (target_positions[:, 1] >= 0))
                if (prediction.shape != (n, 4) or not np.isfinite(prediction).all()
                        or not np.array_equal(prediction[has_shape, 2:4], target_positions[has_shape, 2:4])
                        or not np.array_equal(prediction[preplaced, :2], target_positions[preplaced, :2])):
                    raise ValueError("invalid transferred retrieval layout")
            except Exception:
                transfer_s += time.perf_counter() - transfer_started
                rejected += 1
                continue
            transfer_s += time.perf_counter() - transfer_started
            predictions.append(np.asarray(prediction, dtype=np.float64))
            source_ids.append(int(retrieved.source_ids[source_position]))
            distances.append(float(retrieved.distances[source_position]))
            transforms.append(transform)
            match_costs.append(float(result.total_cost))
            confidences.append(float(result.confidence))

        return CandidateBatch(
            "retrieval", predictions, max(0.0, time.perf_counter() - started),
            {"source_ids": source_ids, "retrieval_distances": distances,
             "transforms": transforms, "match_costs": match_costs,
             "match_confidences": confidences, "query_s": query_s,
             "match_s": match_s, "transfer_s": transfer_s, "rejected": rejected},
        )

    def _sample_portfolio_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                                K, oversample: bool = True) -> List[np.ndarray]:
        """Use one source-neutral rank for the bounded Direct/retrieval portfolio."""
        retrieval_quota = min(
            max(0, int(K)), max(0, int(self.retrieval_slots)),
            FIRST_R4_RETRIEVAL_SLOTS,
        )
        if retrieval_quota <= 0 or self.retrieval_index is None:
            return self._sample_direct_preds(n, at, cons, tpos, b2b, p2b, pins, K, oversample)
        direct = self._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K, oversample)
        retrieved = self._sample_retrieval_preds(n, at, cons, tpos, b2b, p2b, pins, retrieval_quota)
        predictions = direct + retrieved.predictions
        sources = ["direct"] * len(direct) + ["retrieval"] * len(retrieved.predictions)
        order = self._rank_portfolio(predictions, n, at, cons, b2b)
        return _select_ranked_source_quota(predictions, sources, order, K, retrieval_quota)

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
