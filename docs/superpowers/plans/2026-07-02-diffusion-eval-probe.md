# Diffusion Evaluation Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated overfit-probe path that trains v11 diffusion on the 100 local `LiteTensorDataTest` evaluation cases, disables missing tree supervision, saves probe checkpoints, and runs one final full evaluator pass.

**Architecture:** Add a focused evaluation-probe dataset adapter that converts validation polygon labels into the existing diffusion trainer sample contract. Keep `train_diffusion.py` as the only training loop, selected by `--dataset-mode eval-probe`, and expose the workflow through `scripts/train_diffusion_eval_probe.sh`.

**Tech Stack:** Python 3.12, PyTorch, `uv`, pytest, Bash, existing FloorSet `FloorplanDatasetLiteTest`, existing v11 diffusion trainer and evaluator scripts.

---

## Scope Check

This plan implements one subsystem: the evaluation-case overfit probe. It does not change the production solver, hidden-test behavior, normal diffusion training defaults, checkpoint promotion policy, or evaluator scoring.

## File Structure

- Create `src/floorset_arch/training/eval_probe_dataset.py`
  - Owns conversion from `FloorplanDatasetLiteTest` samples into the training-Lite sample shape.
  - Validates case count, polygon shape, block count, and positive rectangle dimensions.
- Modify `src/floorset_arch/training/train_diffusion.py`
  - Adds `--dataset-mode lite|eval-probe`.
  - Uses the adapter for `eval-probe`.
  - Disables val split semantics in probe mode.
  - Rejects tree loss and train-time evaluator in probe mode.
- Create `scripts/train_diffusion_eval_probe.sh`
  - Dedicated user command for the overfit probe.
  - Sets probe defaults and runs one final evaluator pass.
- Create `tests/test_diffusion_eval_probe_dataset.py`
  - Unit and integration coverage for adapter behavior.
- Create `tests/test_diffusion_eval_probe_training.py`
  - Trainer mode and guard coverage.
- Modify `tests/test_scripts.py`
  - Script UX/default coverage.

## Task 1: Add The Eval-Probe Dataset Adapter

**Files:**
- Create: `src/floorset_arch/training/eval_probe_dataset.py`
- Create: `tests/test_diffusion_eval_probe_dataset.py`

- [ ] **Step 1: Write the failing adapter tests**

Create `tests/test_diffusion_eval_probe_dataset.py`:

```python
from __future__ import annotations

import pytest
import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.targets import build_diffusion_targets
from floorset_arch.parser import parse_instance
from floorset_arch.training.eval_probe_dataset import (
    EvalProbeDataset,
    adapt_eval_probe_sample,
    polygon_fp_sol_to_training_rects,
)


def _polygon_sample():
    area = torch.tensor([4.0, 9.0])
    constraints = torch.zeros(2, 5)
    b2b = torch.tensor([[0.0, 1.0, 2.0]])
    p2b = torch.empty(0, 3)
    pins = torch.empty(0, 2)
    polygons = torch.tensor(
        [
            [[10.0, 20.0], [12.0, 20.0], [12.0, 22.0], [10.0, 22.0], [-1.0, -1.0]],
            [[3.0, 4.0], [6.0, 4.0], [6.0, 7.0], [3.0, 7.0], [-1.0, -1.0]],
        ]
    )
    metrics = torch.tensor([100.0, 0.0, 1.0, 1.0, 0.0, 0.0, 11.0, 13.0])
    return {
        "input": (area, b2b, p2b, pins, constraints),
        "label": (polygons, metrics),
    }


class _FakeEvalDataset:
    def __init__(self, count: int = 100):
        self._count = count

    def __len__(self):
        return self._count

    def __getitem__(self, index: int):
        return _polygon_sample()


def test_polygon_fp_sol_to_training_rects_returns_training_order_whxy():
    sample = _polygon_sample()
    polygons = sample["label"][0]

    fp_sol = polygon_fp_sol_to_training_rects(polygons, block_count=2, case_index=7)

    assert fp_sol.shape == (2, 4)
    assert torch.allclose(fp_sol[0], torch.tensor([2.0, 2.0, 10.0, 20.0]))
    assert torch.allclose(fp_sol[1], torch.tensor([3.0, 3.0, 3.0, 4.0]))


def test_adapt_eval_probe_sample_returns_training_lite_contract():
    adapted = adapt_eval_probe_sample(_polygon_sample(), case_index=3)
    inputs = adapted["input"]
    labels = adapted["label"]

    assert len(inputs) == 5
    assert len(labels) == 3
    tree_sol, fp_sol, metrics_sol = labels
    assert tree_sol.shape == (0, 3)
    assert tree_sol.dtype == torch.float32
    assert torch.allclose(fp_sol[0], torch.tensor([2.0, 2.0, 10.0, 20.0]))
    assert torch.allclose(metrics_sol, torch.tensor([100.0, 0.0, 1.0, 1.0, 0.0, 0.0, 11.0, 13.0]))


def test_eval_probe_dataset_requires_exactly_100_cases_by_default():
    with pytest.raises(ValueError, match="expected 100 evaluation cases, found 99"):
        EvalProbeDataset("FloorSet", dataset=_FakeEvalDataset(99))


def test_eval_probe_dataset_allows_custom_count_for_tests():
    dataset = EvalProbeDataset("FloorSet", dataset=_FakeEvalDataset(2), expected_count=2)

    sample = dataset[1]

    assert len(dataset) == 2
    assert sample["label"][0].shape == (0, 3)
    assert sample["label"][1].shape == (2, 4)


def test_polygon_adapter_reports_case_index_for_bad_polygon():
    bad = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])

    with pytest.raises(ValueError, match="case 12 block 0 has non-positive rectangle"):
        polygon_fp_sol_to_training_rects(bad, block_count=1, case_index=12)


def test_adapted_sample_builds_diffusion_targets_without_tree_edges():
    adapted = adapt_eval_probe_sample(_polygon_sample(), case_index=5)
    area, b2b, p2b, pins, constraints = adapted["input"]
    tree_sol, fp_sol, metrics_sol = adapted["label"]
    inst = parse_instance(2, area, b2b, p2b, pins, constraints, None)
    graph = build_diffusion_graph_inputs(inst)

    targets = build_diffusion_targets(inst, graph, fp_sol, tree_sol, metrics_sol)

    assert targets.center.shape == (2, 2)
    assert targets.log_aspect.shape == (2,)
    assert targets.tree_pair_mask.any().item() is False
    assert targets.quality_labels.numel() == 3
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run:

```bash
uv run pytest tests/test_diffusion_eval_probe_dataset.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'floorset_arch.training.eval_probe_dataset'`.

- [ ] **Step 3: Add the dataset adapter implementation**

Create `src/floorset_arch/training/eval_probe_dataset.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
FLOORSET_DIR = ROOT / "FloorSet"


def _ensure_floorset_import_path(root: str | Path) -> None:
    root_path = Path(root).resolve()
    for path in (root_path, FLOORSET_DIR):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _case_prefix(case_index: int | None) -> str:
    return f"case {case_index} " if case_index is not None else ""


def polygon_fp_sol_to_training_rects(
    fp_sol: torch.Tensor,
    block_count: int,
    case_index: int | None = None,
) -> torch.Tensor:
    polygons = torch.as_tensor(fp_sol).detach().float()
    prefix = _case_prefix(case_index)
    if polygons.dim() < 3 or polygons.shape[-1] != 2:
        raise ValueError(
            f"{prefix}polygon fp_sol must have shape [blocks, vertices, 2], "
            f"got {tuple(polygons.shape)}"
        )
    if polygons.shape[0] < int(block_count):
        raise ValueError(
            f"{prefix}polygon fp_sol has {polygons.shape[0]} blocks, "
            f"expected at least {int(block_count)}"
        )

    rows: list[torch.Tensor] = []
    for block in range(int(block_count)):
        poly = polygons[block]
        valid = (poly[:, 0] != -1) & (poly[:, 1] != -1)
        points = poly[valid]
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
) -> dict[str, tuple]:
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
            area_targets,
            b2b,
            p2b,
            pins,
            constraints,
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
```

- [ ] **Step 4: Run adapter tests to verify they pass**

Run:

```bash
uv run pytest tests/test_diffusion_eval_probe_dataset.py -q
```

Expected: PASS with all tests in `tests/test_diffusion_eval_probe_dataset.py`.

- [ ] **Step 5: Commit adapter work**

Run:

```bash
git add src/floorset_arch/training/eval_probe_dataset.py tests/test_diffusion_eval_probe_dataset.py
git commit -m "feat: add diffusion eval probe dataset adapter"
```

Expected: commit succeeds and `git status --short` no longer lists these two files.

## Task 2: Wire Eval-Probe Mode Into The Diffusion Trainer

**Files:**
- Modify: `src/floorset_arch/training/train_diffusion.py`
- Create: `tests/test_diffusion_eval_probe_training.py`

- [ ] **Step 1: Write failing trainer-mode tests**

Create `tests/test_diffusion_eval_probe_training.py`:

```python
from __future__ import annotations

import pytest
import torch

from floorset_arch.training import train_diffusion


def _adapted_sample(offset: float = 0.0):
    return {
        "input": (
            torch.tensor([4.0, 9.0]),
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.empty(0, 3),
            torch.empty(0, 2),
            torch.zeros(2, 5),
        ),
        "label": (
            torch.empty(0, 3),
            torch.tensor(
                [
                    [2.0, 2.0, 0.0 + offset, 0.0],
                    [3.0, 3.0, 3.0 + offset, 0.0],
                ]
            ),
            torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0]),
        ),
    }


class _TinyEvalProbeDataset:
    def __init__(self, _root):
        self.samples = [_adapted_sample(0.0), _adapted_sample(0.5)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def test_parse_args_defaults_to_lite_dataset_mode():
    args = train_diffusion.parse_args([])

    assert args.dataset_mode == "lite"


def test_eval_probe_mode_rejects_tree_weight():
    args = train_diffusion.parse_args(
        ["--dataset-mode", "eval-probe", "--tree-weight", "0.25"]
    )

    with pytest.raises(ValueError, match="eval-probe requires --tree-weight 0"):
        train_diffusion._validate_diffusion_training_args(args)


def test_eval_probe_mode_rejects_train_time_evaluator():
    args = train_diffusion.parse_args(
        [
            "--dataset-mode",
            "eval-probe",
            "--tree-weight",
            "0",
            "--train-evaluate-each-epoch",
        ]
    )

    with pytest.raises(ValueError, match="eval-probe uses final-only evaluator"):
        train_diffusion._validate_diffusion_training_args(args)


def test_make_loaders_uses_eval_probe_dataset_without_val_loader(monkeypatch):
    monkeypatch.setattr(train_diffusion, "EvalProbeDataset", _TinyEvalProbeDataset)
    args = train_diffusion.parse_args(
        [
            "--dataset-mode",
            "eval-probe",
            "--tree-weight",
            "0",
            "--num-samples",
            "100",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
        ]
    )

    train_loader, val_loader, ts, te, vs, ve, total = train_diffusion._make_loaders(args)

    assert val_loader is None
    assert (ts, te, vs, ve, total) == (0, 1, -1, -1, 2)
    first_batch = next(iter(train_loader))
    assert first_batch[5].shape[-1] == 3
    assert first_batch[6].shape[-1] == 4
```

- [ ] **Step 2: Run trainer-mode tests to verify they fail**

Run:

```bash
uv run pytest tests/test_diffusion_eval_probe_training.py -q
```

Expected: FAIL because `dataset_mode`, `_validate_diffusion_training_args`, and `EvalProbeDataset` wiring do not exist in `train_diffusion.py`.

- [ ] **Step 3: Add imports and argument validation**

Modify `src/floorset_arch/training/train_diffusion.py` imports near the other training imports:

```python
from floorset_arch.training.eval_probe_dataset import EvalProbeDataset
```

Add this helper near `_make_loaders()`:

```python
def _is_eval_probe_mode(args) -> bool:
    return getattr(args, "dataset_mode", "lite") == "eval-probe"


def _validate_diffusion_training_args(args) -> None:
    if not _is_eval_probe_mode(args):
        return
    if abs(float(getattr(args, "tree_weight", 0.0))) > 1e-12:
        raise ValueError("--dataset-mode eval-probe requires --tree-weight 0")
    if bool(getattr(args, "train_evaluate_each_epoch", False)):
        raise ValueError(
            "--dataset-mode eval-probe uses final-only evaluator; "
            "do not pass --train-evaluate-each-epoch"
        )
```

Add the parser argument near `--data-path`:

```python
parser.add_argument(
    "--dataset-mode",
    choices=("lite", "eval-probe"),
    default="lite",
    help="Use FloorSet Lite training data or the 100-case evaluation overfit probe.",
)
```

- [ ] **Step 4: Route `_make_loaders()` through the eval-probe dataset**

Modify `_make_loaders(args)` so it handles probe mode after synthetic smoke mode and before normal `FloorplanDatasetLite` loading:

```python
def _make_loaders(args):
    if int(args.synthetic_smoke_samples) > 0:
        train_loader = _synthetic_smoke_loader(args.synthetic_smoke_samples)
        val_loader = _synthetic_smoke_loader(
            max(1, min(args.val_samples, args.synthetic_smoke_samples))
        )
        return (
            train_loader,
            val_loader,
            0,
            len(train_loader) - 1,
            0,
            len(val_loader) - 1,
            len(train_loader),
        )

    if _is_eval_probe_mode(args):
        dataset = EvalProbeDataset(args.data_path)
        total = len(dataset)
        train_loader, ts, te = make_loader(
            dataset,
            0,
            total,
            True,
            args.seed,
            args.num_workers,
            args.batch_size,
        )
        return train_loader, None, ts, te, -1, -1, total

    dataset = FloorplanDatasetLite(args.data_path)
    total = len(dataset)
    val_start = choose_window_start(
        total, args.val_samples, args.seed + 777, 0, args.val_start
    )
    val_loader, vs, ve = make_loader(
        dataset,
        val_start,
        args.val_samples,
        False,
        args.seed,
        args.num_workers,
        args.batch_size,
    )
    train_start = choose_window_start(
        total, args.num_samples, args.seed, 1, args.window_start
    )
    train_loader, ts, te = make_loader(
        dataset,
        train_start,
        args.num_samples,
        True,
        args.seed + 1,
        args.num_workers,
        args.batch_size,
    )
    return train_loader, val_loader, ts, te, vs, ve, total
```

- [ ] **Step 5: Update `main()` probe semantics**

In `main(args)`, call validation immediately after device checks:

```python
_validate_diffusion_training_args(args)
probe_mode = _is_eval_probe_mode(args)
```

Update the header prints:

```python
print("Architecture v11 graph-conditioned diffusion training")
print(f"  dataset mode     = {getattr(args, 'dataset_mode', 'lite')}")
print(f"  dataset samples  = {total}")
print(f"  train window     = {ts}..{te}")
print(f"  val window       = {'disabled' if val_loader is None else f'{vs}..{ve}'}")
```

Update epoch re-windowing so normal Lite training keeps current behavior and probe mode keeps the fixed 100-case loader:

```python
if int(args.synthetic_smoke_samples) <= 0 and not probe_mode:
    dataset = FloorplanDatasetLite(args.data_path)
    train_start = choose_window_start(
        len(dataset), args.num_samples, args.seed, epoch, args.window_start
    )
    train_loader, ts, te = make_loader(
        dataset,
        train_start,
        args.num_samples,
        True,
        args.seed + epoch,
        args.num_workers,
        args.batch_size,
    )
    print(f"Epoch {epoch:03d}: train window {ts}..{te}", flush=True)
```

Update the val/probe stats section:

```python
if val_loader is None:
    val_stats = dict(train_stats)
    val_label = "probe"
else:
    val_stats = run_epoch(
        model, optimizer, val_loader, device, args, loss_config, epoch, train=False
    )
    val_label = "val"
```

Update the printed stats:

```python
print(
    f"Epoch {epoch:03d} {val_label:<5} loss={val_stats['loss']:.5f} "
    f"denoise={val_stats['denoise']:.5f} pair={val_stats['pair']:.5f} "
    f"tree={val_stats['tree']:.5f} quality={val_stats['quality']:.5f} "
    f"aspect={val_stats['aspect']:.5f} overlap={val_stats['layout_overlap']:.5f} "
    f"used={val_stats['used']:.0f} skipped={val_stats['skipped']:.0f}",
    flush=True,
)
```

Update W&B logging so probe stats do not appear as generalized validation:

```python
if wandb_run is not None:
    val_prefix = "probe" if val_loader is None else "val"
    wandb_run.log(
        {
            "epoch": epoch,
            **{f"train/{key}": value for key, value in train_stats.items()},
            **{f"{val_prefix}/{key}": value for key, value in val_stats.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
    )
```

Keep checkpoint payload keys as `train_stats` and `val_stats` for compatibility. In probe mode, `val_stats` is a copy of train stats.

- [ ] **Step 6: Give probe checkpoints probe-loss names**

Replace the best checkpoint path construction with mode-aware naming:

```python
latest_path = out_dir / f"{args.checkpoint_prefix}_latest_{run_tag}.pt"
best_metric_name = "best_probe_loss" if _is_eval_probe_mode(args) else "best_val_loss"
best_val_loss_path = out_dir / f"{args.checkpoint_prefix}_{best_metric_name}_{run_tag}.pt"
```

Replace final print labels:

```python
metric_label = "Best probe loss" if probe_mode else "Best val loss"
print(f"{metric_label}: {best_val:.5f}")
print(f"Latest checkpoint: {latest_path}")
print(f"Best {'probe-loss' if probe_mode else 'val-loss'} checkpoint: {best_val_loss_path}")
```

- [ ] **Step 7: Run trainer-mode tests**

Run:

```bash
uv run pytest tests/test_diffusion_eval_probe_training.py -q
```

Expected: PASS with all tests in `tests/test_diffusion_eval_probe_training.py`.

- [ ] **Step 8: Run existing diffusion training tests**

Run:

```bash
uv run pytest tests/test_diffusion_training_path.py tests/test_diffusion_eval_probe_dataset.py tests/test_diffusion_eval_probe_training.py -q
```

Expected: PASS. Existing synthetic smoke still writes `diffusion_latest_smoke.pt`.

- [ ] **Step 9: Commit trainer wiring**

Run:

```bash
git add src/floorset_arch/training/train_diffusion.py tests/test_diffusion_eval_probe_training.py
git commit -m "feat: wire diffusion eval probe trainer mode"
```

Expected: commit succeeds and `git status --short` no longer lists these two files.

## Task 3: Add The Dedicated Probe Script

**Files:**
- Create: `scripts/train_diffusion_eval_probe.sh`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Write failing script tests**

Append this test to `tests/test_scripts.py`:

```python
def test_train_diffusion_eval_probe_script_uses_probe_defaults_and_final_eval():
    text = Path("scripts/train_diffusion_eval_probe.sh").read_text(encoding="utf-8")

    assert 'DATASET_MODE="${DATASET_MODE:-eval-probe}"' in text
    assert 'NUM_SAMPLES="${NUM_SAMPLES:-100}"' in text
    assert 'VAL_SAMPLES="${VAL_SAMPLES:-100}"' in text
    assert 'EPOCHS="${EPOCHS:-100}"' in text
    assert 'TREE_WEIGHT="${TREE_WEIGHT:-0}"' in text
    assert 'TRAIN_EVALUATE_EACH_EPOCH="${TRAIN_EVALUATE_EACH_EPOCH:-0}"' in text
    assert 'CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-diffusion_eval_probe}"' in text
    assert '--dataset-mode "$DATASET_MODE"' in text
    assert '--tree-weight "$TREE_WEIGHT"' in text
    assert 'PROBE_CKPT="$OUTPUT_DIR/$CHECKPOINT_PREFIX' in text
    assert 'bash "$ROOT/scripts/eval_total.sh"' in text
    assert '--diffusion-checkpoint "$PROBE_CKPT"' in text
    assert '--diffusion-use-ema' in text
    assert '--output "$EVAL_OUTPUT"' in text
```

- [ ] **Step 2: Run script test to verify it fails**

Run:

```bash
uv run pytest tests/test_scripts.py::test_train_diffusion_eval_probe_script_uses_probe_defaults_and_final_eval -q
```

Expected: FAIL because `scripts/train_diffusion_eval_probe.sh` does not exist.

- [ ] **Step 3: Create the dedicated script**

Create `scripts/train_diffusion_eval_probe.sh`:

```bash
#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASET_MODE="${DATASET_MODE:-eval-probe}"
DATA_PATH="${DATA_PATH:-FloorSet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-.}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-100}"
EPOCHS="${EPOCHS:-100}"
VARIANT="${VARIANT:-hgt_lite}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-2}"
LR="${LR:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-0.02}"
NOISE_SAMPLES="${NOISE_SAMPLES:-1}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
PAIR_WEIGHT="${PAIR_WEIGHT:-0.25}"
TREE_WEIGHT="${TREE_WEIGHT:-0}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.01}"
ASPECT_WEIGHT="${ASPECT_WEIGHT:-0.05}"
OVERLAP_WEIGHT="${OVERLAP_WEIGHT:-0.05}"
BBOX_WEIGHT="${BBOX_WEIGHT:-0.01}"
NET_WEIGHT="${NET_WEIGHT:-0.01}"
CLUSTER_WEIGHT="${CLUSTER_WEIGHT:-0.02}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.02}"
MIB_WEIGHT="${MIB_WEIGHT:-0.02}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRINT_EVERY="${PRINT_EVERY:-25}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-diffusion_eval_probe}"
TRAIN_EVALUATE_EACH_EPOCH="${TRAIN_EVALUATE_EACH_EPOCH:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-floorset-v11-diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

if [ "$DATASET_MODE" != "eval-probe" ]; then
  echo "train_diffusion_eval_probe.sh requires DATASET_MODE=eval-probe" >&2
  exit 1
fi
case "$TREE_WEIGHT" in
  0|0.0|0.00|0.000|0.0000) ;;
  *)
    echo "eval-probe has no tree_sol labels; set TREE_WEIGHT=0" >&2
    exit 1
    ;;
esac
if [ "$TRAIN_EVALUATE_EACH_EPOCH" != "0" ]; then
  echo "eval-probe uses final-only evaluator; set TRAIN_EVALUATE_EACH_EPOCH=0" >&2
  exit 1
fi

DATE_STAMP=$(date +%m%d)
VARIANT_TAG="${VARIANT//-/_}"
DEFAULT_TAG="${DATE_STAMP}_eval_probe_ep${EPOCHS}_diff${VARIANT_TAG}_h${HIDDEN_DIM}_l${LAYERS}_steps${DIFFUSION_STEPS}_acc${ACCUMULATION_STEPS}_bs${BATCH_SIZE}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-$DEFAULT_TAG}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$CHECKPOINT_TAG}"
LOG_TAG="$CHECKPOINT_TAG"

EXTRA_ARGS=()
if [ "$WANDB" = "1" ]; then
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  if [ -n "$WANDB_ENTITY" ]; then
    EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [ -n "$WANDB_RUN_NAME" ]; then
    EXTRA_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  fi
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

uv run -m floorset_arch.training.train_diffusion \
  --dataset-mode "$DATASET_MODE" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --variant "$VARIANT" \
  --hidden-dim "$HIDDEN_DIM" \
  --layers "$LAYERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --max-diffusion-steps "$DIFFUSION_STEPS" \
  --noise-schedule "$NOISE_SCHEDULE" \
  --beta-start "$BETA_START" \
  --beta-end "$BETA_END" \
  --noise-samples "$NOISE_SAMPLES" \
  --ema-decay "$EMA_DECAY" \
  --pair-weight "$PAIR_WEIGHT" \
  --tree-weight "$TREE_WEIGHT" \
  --quality-weight "$QUALITY_WEIGHT" \
  --aspect-weight "$ASPECT_WEIGHT" \
  --overlap-weight "$OVERLAP_WEIGHT" \
  --bbox-weight "$BBOX_WEIGHT" \
  --net-weight "$NET_WEIGHT" \
  --cluster-weight "$CLUSTER_WEIGHT" \
  --boundary-weight "$BOUNDARY_WEIGHT" \
  --mib-weight "$MIB_WEIGHT" \
  --num-workers "$NUM_WORKERS" \
  --checkpoint-prefix "$CHECKPOINT_PREFIX" \
  --checkpoint-tag "$CHECKPOINT_TAG" \
  --print-every "$PRINT_EVERY" \
  "${EXTRA_ARGS[@]}" | tee "$LOG_DIR/train_arch_v11_diffusion_eval_probe_${LOG_TAG}.log"

PROBE_CKPT="$OUTPUT_DIR/$CHECKPOINT_PREFIX"_latest_"$CHECKPOINT_TAG.pt"
EVAL_OUTPUT="${EVAL_OUTPUT:-$ROOT/artifacts/eval_probe/${CHECKPOINT_TAG}_full_eval.json}"
mkdir -p "$(dirname "$EVAL_OUTPUT")"

bash "$ROOT/scripts/eval_total.sh" \
  --diffusion-checkpoint "$PROBE_CKPT" \
  --diffusion-use-ema \
  --output "$EVAL_OUTPUT"
```

- [ ] **Step 4: Make the script executable**

Run:

```bash
chmod +x scripts/train_diffusion_eval_probe.sh
```

Expected: `test -x scripts/train_diffusion_eval_probe.sh` exits 0.

- [ ] **Step 5: Run script tests**

Run:

```bash
uv run pytest tests/test_scripts.py::test_train_diffusion_eval_probe_script_uses_probe_defaults_and_final_eval -q
```

Expected: PASS.

- [ ] **Step 6: Commit script work**

Run:

```bash
git add scripts/train_diffusion_eval_probe.sh tests/test_scripts.py
git commit -m "feat: add diffusion eval probe script"
```

Expected: commit succeeds and `git status --short` no longer lists these two files.

## Task 4: Verify The Probe Workflow

**Files:**
- Modify only if a preceding test reveals a bug in files from Tasks 1-3.

- [ ] **Step 1: Run focused pytest suite**

Run:

```bash
uv run pytest \
  tests/test_diffusion_eval_probe_dataset.py \
  tests/test_diffusion_eval_probe_training.py \
  tests/test_diffusion_training_path.py \
  tests/test_scripts.py \
  -q
```

Expected: PASS for all selected tests.

- [ ] **Step 2: Run a CPU smoke probe**

Run:

```bash
EPOCHS=1 DEVICE=cpu WANDB=0 PRINT_EVERY=0 CHECKPOINT_TAG=smoke_eval_probe bash scripts/train_diffusion_eval_probe.sh
```

Expected:

- Training prints `dataset mode     = eval-probe`.
- Training prints `val window       = disabled`.
- Training writes `checkpoints/diffusion_eval_probe_latest_smoke_eval_probe.pt`.
- Final evaluator writes `artifacts/eval_probe/smoke_eval_probe_full_eval.json`.

- [ ] **Step 3: Check checkpoint loadability**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import torch

path = Path("checkpoints/diffusion_eval_probe_latest_smoke_eval_probe.pt")
payload = torch.load(path, map_location="cpu")
print(payload["variant"], payload["epoch"], payload["loss_config"]["tree_weight"])
assert payload["epoch"] == 1
assert payload["loss_config"]["tree_weight"] == 0.0
assert payload["ema_model_state_dict"]
PY
```

Expected: prints `hgt_lite 1 0.0` and exits 0.

- [ ] **Step 4: Check evaluator JSON shape**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/eval_probe/smoke_eval_probe_full_eval.json")
data = json.loads(path.read_text())
print(data["total_score"], len(data["test_results"]))
assert "total_score" in data
assert "total_score_no_runtime" in data
assert len(data["test_results"]) == 100
PY
```

Expected: prints a numeric score and `100`, then exits 0.

- [ ] **Step 5: Update graphify output**

Run:

```bash
graphify update .
```

Expected: command exits 0. If it reports changed graph files, inspect `git status --short` and include graphify output only if this repository normally tracks the updated files.

- [ ] **Step 6: Commit any verification fixes**

If Task 4 required code fixes, run:

```bash
git add src/floorset_arch/training/eval_probe_dataset.py src/floorset_arch/training/train_diffusion.py scripts/train_diffusion_eval_probe.sh tests/test_diffusion_eval_probe_dataset.py tests/test_diffusion_eval_probe_training.py tests/test_scripts.py
git commit -m "fix: stabilize diffusion eval probe workflow"
```

Expected: commit succeeds only if Task 4 changed files. If no fixes were needed, skip this commit.

## Task 5: Optional Full 100-Epoch Probe Run

**Files:**
- Generated outputs only: `checkpoints/`, `artifacts/eval_probe/`, and log files.

- [ ] **Step 1: Start the full probe after focused verification passes**

Run:

```bash
EPOCHS=100 DEVICE=cuda WANDB=1 bash scripts/train_diffusion_eval_probe.sh
```

Expected:

- A checkpoint named like `checkpoints/diffusion_eval_probe_latest_<tag>.pt`.
- A best probe-loss checkpoint named like `checkpoints/diffusion_eval_probe_best_probe_loss_<tag>.pt`.
- A final evaluator JSON under `artifacts/eval_probe/`.
- Logs that show probe loss over 100 epochs.

- [ ] **Step 2: Interpret the full run as diagnostic evidence**

Use this decision table:

```text
probe loss falls, evaluator improves:
  training and inference handoff have usable signal

probe loss falls, evaluator remains weak:
  inspect sampling, concretization, repair, and v10 ranking handoff

probe loss does not fall:
  inspect target construction, optimizer settings, model capacity, or loss weights

evaluator fails to load checkpoint:
  inspect checkpoint payload, dataset-mode args, and FLOORSET_DIFFUSION_* environment
```

Expected: no promotion decision is made from this run. The score is overfit-probe evidence only.

## Final Verification Checklist

- [ ] `uv run pytest tests/test_diffusion_eval_probe_dataset.py tests/test_diffusion_eval_probe_training.py tests/test_diffusion_training_path.py tests/test_scripts.py -q`
- [ ] `EPOCHS=1 DEVICE=cpu WANDB=0 PRINT_EVERY=0 CHECKPOINT_TAG=smoke_eval_probe bash scripts/train_diffusion_eval_probe.sh`
- [ ] Check `checkpoints/diffusion_eval_probe_latest_smoke_eval_probe.pt`
- [ ] Check `artifacts/eval_probe/smoke_eval_probe_full_eval.json`
- [ ] `graphify update .`
- [ ] `git status --short` shows only intentional generated artifacts or pre-existing unrelated changes.
