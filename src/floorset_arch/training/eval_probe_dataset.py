from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
FLOORSET_DIR = ROOT / "FloorSet"


def _ensure_floorset_import_path(root: str | Path) -> None:
    for path in (Path(root).resolve(), FLOORSET_DIR):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _case_prefix(case_index: int | None) -> str:
    return f"case {case_index} " if case_index is not None else ""


def _polygon_block_count(fp_sol: Any) -> int:
    if isinstance(fp_sol, torch.Tensor):
        return int(fp_sol.shape[0]) if fp_sol.dim() > 0 else 0
    try:
        return len(fp_sol)
    except TypeError:
        return 0


def _polygon_for_block(fp_sol: Any, block: int) -> torch.Tensor:
    return torch.as_tensor(fp_sol[block]).detach().float()


def polygon_fp_sol_to_training_rects(
    fp_sol: Any,
    block_count: int,
    case_index: int | None = None,
) -> torch.Tensor:
    """Convert FloorSet evaluation polygons to training-style (w, h, x, y)."""
    prefix = _case_prefix(case_index)
    block_count = int(block_count)
    if isinstance(fp_sol, torch.Tensor):
        polygons = torch.as_tensor(fp_sol).detach().float()
        if polygons.dim() == 0:
            raise ValueError(
                f"{prefix}polygon fp_sol must have shape [blocks, vertices, 2], "
                f"got {tuple(polygons.shape)}"
            )
    elif _polygon_block_count(fp_sol) == 0:
        raise ValueError(f"{prefix}polygon fp_sol must have block polygons")

    available_blocks = _polygon_block_count(fp_sol)
    if available_blocks < block_count:
        raise ValueError(
            f"{prefix}polygon fp_sol has {available_blocks} blocks, "
            f"expected at least {block_count}"
        )

    rows: list[torch.Tensor] = []
    for block in range(block_count):
        polygon = _polygon_for_block(fp_sol, block)
        if polygon.dim() != 2 or polygon.shape[-1] != 2:
            raise ValueError(
                f"{prefix}block {block} polygon must have shape [vertices, 2], "
                f"got {tuple(polygon.shape)}"
            )
        valid = (polygon[:, 0] != -1) & (polygon[:, 1] != -1)
        points = polygon[valid]
        if points.numel() == 0:
            raise ValueError(f"{prefix}block {block} has no valid polygon points")
        min_xy = points.min(dim=0).values
        max_xy = points.max(dim=0).values
        width_height = max_xy - min_xy
        if bool((width_height <= 0).any().item()):
            raise ValueError(
                f"{prefix}block {block} has non-positive rectangle "
                f"width={float(width_height[0].item())} "
                f"height={float(width_height[1].item())}"
            )
        width, height = width_height
        x, y = min_xy
        rows.append(torch.stack((width, height, x, y)))
    return torch.stack(rows).to(dtype=torch.float32)


def adapt_eval_probe_sample(
    sample: dict[str, Any],
    case_index: int | None = None,
) -> dict[str, tuple[torch.Tensor, ...]]:
    inputs = sample["input"]
    labels = sample["label"]
    area_targets, b2b, p2b, pins, constraints = inputs
    fp_sol_polygons, metrics_sol = labels
    block_count = int((torch.as_tensor(area_targets).detach().flatten() != -1).sum().item())
    if block_count <= 0:
        raise ValueError(f"{_case_prefix(case_index)}has no valid blocks")

    fp_sol = polygon_fp_sol_to_training_rects(
        fp_sol_polygons,
        block_count=block_count,
        case_index=case_index,
    )
    tree_sol = torch.empty((0, 3), dtype=torch.float32)
    return {
        "input": (
            torch.as_tensor(area_targets),
            torch.as_tensor(b2b),
            torch.as_tensor(p2b),
            torch.as_tensor(pins),
            torch.as_tensor(constraints),
        ),
        "label": (
            tree_sol,
            fp_sol,
            torch.as_tensor(metrics_sol).detach().float(),
        ),
    }


class EvalProbeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        dataset: Dataset | None = None,
        expected_count: int | None = 100,
    ) -> None:
        self.root = Path(root)
        if dataset is None:
            _ensure_floorset_import_path(self.root)
            from lite_dataset_test import FloorplanDatasetLiteTest

            dataset = FloorplanDatasetLiteTest(str(self.root))

        actual = len(dataset)
        if expected_count is not None and actual != int(expected_count):
            raise ValueError(
                f"expected {int(expected_count)} evaluation cases, found {actual}"
            )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return adapt_eval_probe_sample(self.dataset[int(index)], case_index=int(index))
