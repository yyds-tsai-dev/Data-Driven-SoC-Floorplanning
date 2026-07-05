"""Scratch optimizer: measures ML-ordering-signal headroom for the column
backbone by replacing _heuristic_init's seed with the GROUND-TRUTH (fp_sol)
layout, WITHOUT touching repo files.

Mechanism:
  * At __init__, load the validation set once and build a
    block_count -> gt_seed_rects map. block_count is a unique key across the
    100 validation cases (21..120, verified all distinct), so incoming cases
    are matched by block_count. GT rects are the bounding boxes of each
    block's polygon in fp_sol, extracted exactly as the evaluator's
    _extract_baseline does.
  * In solve(), monkeypatch floorset_arch.legalizer.column_backbone
    ._heuristic_init to return the GT rects for the matching case, then call
    solve_with_column_backbone unchanged. The GT layout enters ONLY as the
    relative-position seed (unit seed_x/seed_y, column assignment); the
    column-slicing legalizer + SA + repair produce the final legal layout.

Run with FLOORSET_COLUMN_BACKBONE=1 and NO refine flags to match the
heuristic-seed reference band.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Ensure the repo's src/ (where floorset_arch lives) is importable, exactly as
# src/architecture_v11_optimizer.py does, so this scratch file is self-contained
# regardless of ambient PYTHONPATH.
_REPO_SRC = Path(__file__).resolve()
for _p in _REPO_SRC.parents:
    _cand = _p / "src" / "floorset_arch"
    if _cand.is_dir():
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
else:
    # Fall back to the known repo layout.
    _HARD = "/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/src"
    if _HARD not in sys.path:
        sys.path.insert(0, _HARD)

# Evaluator base class (import path set up by the eval scripts' PYTHONPATH).
from iccad2026_evaluate import FloorplanOptimizer

import floorset_arch.legalizer.column_backbone as cb
from floorset_arch.legalizer.column_backbone import solve_with_column_backbone

Rect = Tuple[float, float, float, float]


def _polys_to_bbox_rects(polygons, block_count: int) -> List[Rect]:
    """Replicate iccad2026_evaluate._extract_baseline polygon->bbox extraction."""
    out: List[Rect] = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            out.append(
                (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))
            )
        else:
            out.append((0.0, 0.0, 1.0, 1.0))
    return out


class GtSeedColumnOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        # Force the column backbone regardless of ambient env (belt & braces;
        # the run also exports FLOORSET_COLUMN_BACKBONE=1).
        os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")

        from litetestLoader import FloorplanDatasetLiteTest

        data_path = os.environ.get("FLOORSET_DATA_PATH", "../")
        ds = FloorplanDatasetLiteTest(str(data_path))

        self._gt_by_bc: Dict[int, List[Rect]] = {}
        for idx in range(len(ds)):
            sample = ds[idx]
            area_target = sample["input"][0]
            polygons = sample["label"][0]
            block_count = int((area_target != -1).sum().item())
            self._gt_by_bc[block_count] = _polys_to_bbox_rects(polygons, block_count)

        # Warm the parallel worker pool exactly like the production __init__.
        try:
            cb.warm_worker_pool()
        except Exception:
            pass

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
        gt_rects = self._gt_by_bc.get(block_count)

        if gt_rects is None or len(gt_rects) != block_count:
            # No matching GT (should not happen on the validation set) -> run
            # the unmodified backbone so this case is not silently broken.
            return solve_with_column_backbone(
                block_count, area_targets, b2b_connectivity, p2b_connectivity,
                pins_pos, constraints, target_positions,
            )

        original = cb._heuristic_init

        def _gt_init(area_targets_, constraints_, target_positions_, b2b_, p2b_, pins_):
            # Return a copy so downstream mutation cannot corrupt the cache.
            return [tuple(r) for r in gt_rects]

        cb._heuristic_init = _gt_init
        try:
            return solve_with_column_backbone(
                block_count, area_targets, b2b_connectivity, p2b_connectivity,
                pins_pos, constraints, target_positions,
            )
        finally:
            cb._heuristic_init = original


# Names the evaluator looks for.
MyOptimizer = GtSeedColumnOptimizer
ContestOptimizer = GtSeedColumnOptimizer
