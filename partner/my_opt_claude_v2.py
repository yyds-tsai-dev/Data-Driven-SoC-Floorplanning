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
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

import os

from iccad2026_evaluate import FloorplanOptimizer
from diffusion_data import build_condition, fp_sol_to_z0
from diffusion_model import DiffusionSchedule, GraphDiffusionDenoiser, ModelConfig, ddim_refine
from legalizer_claude import (_parse_constraints, _target, init_worker_pool,
                              legalize_rectangles, rectangles_from_z)

N_RESTART_WORKERS = max(2, min(12, (os.cpu_count() or 4) // 2))

Rect = Tuple[float, float, float, float]

DEFAULT_CHECKPOINT = (
    Path(__file__).parent / "checkpoints" / "diffusion_stable_xywh_order_2day" / "step_00080000.pt"
)

BUDGET_SCALE = 0.06
BUDGET_TAU = 20.0
BUDGET_MIN = 0.8
BUDGET_MAX = 24.0


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
        # spawn the parallel-restart pool now so worker startup cost is not
        # charged to any test case
        init_worker_pool(N_RESTART_WORKERS)

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
        start = time.time()
        deadline = start + _time_budget(block_count)

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

            return legalize_rectangles(
                raw_rects, area_targets, constraints, target_positions,
                b2b_connectivity=b2b, p2b_connectivity=p2b, pins_pos=pins,
                deadline=deadline,
            )
        except Exception as exc:
            if self.verbose:
                print(f"column optimizer failed; using row fallback: {exc}")
            return _fallback_row(area_targets, constraints, target_positions)

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
