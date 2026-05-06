from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from floorset_arch.constructive import construct_initial_placement
from floorset_arch.models import Rect, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.repair import repair_placement


ROOT = Path(__file__).resolve().parents[2]
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
if CONTEST_DIR.exists() and str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

try:
    from iccad2026_evaluate import FloorplanOptimizer
except Exception:
    class FloorplanOptimizer:  # type: ignore[no-redef]
        def __init__(self, verbose: bool = False):
            self.verbose = verbose


class ArchitectureV1Optimizer(FloorplanOptimizer):
    """Hard-constraint-safe architecture v1.0 solver."""

    def __init__(self, verbose: bool = False, config: Optional[SolverConfig] = None):
        super().__init__(verbose=verbose)
        self.config = config or SolverConfig()

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        inst = parse_instance(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        inst.model_hints = self._try_model_hints(inst)
        placement = construct_initial_placement(inst, self.config)
        repaired = repair_placement(inst, placement, self.config)
        return repaired.to_position_list(block_count)

    def _try_model_hints(self, inst) -> Optional[dict[int, Rect]]:
        checkpoint = os.environ.get(self.config.checkpoint_env)
        if not checkpoint:
            return None
        try:
            from floorset_arch.features import build_model_inputs
            from floorset_arch.nn.model import SimpleGraphFloorplanner
            from floorset_arch.nn.postprocess import predictions_to_rects
            from floorset_arch.training.checkpoint import load_checkpoint

            payload = load_checkpoint(checkpoint, map_location="cpu")
            cfg = payload.get("model_config", {})
            inputs = build_model_inputs(inst)
            model = SimpleGraphFloorplanner(
                input_dim=inputs.block_features.shape[1],
                hidden_dim=int(cfg.get("hidden_dim", 128)),
                layers=int(cfg.get("layers", 3)),
            )
            model.load_state_dict(payload["model_state"])
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                return predictions_to_rects(inst, pred)
        except Exception:
            return None

