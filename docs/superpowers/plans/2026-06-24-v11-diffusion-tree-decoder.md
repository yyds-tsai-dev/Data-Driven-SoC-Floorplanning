# V11 Diffusion Tree Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v11 graph-conditioned diffusion solver path with A/B diffusion variants, direct diffusion decoder/repair/ranking, tree/metrics training targets, v11 wrappers, and eval floorplan PNG output.

**Architecture:** Keep `parse_instance()` returning `Instance`. Add a new `floorset_arch.diffusion` package that owns diffusion graph inputs, contracts, targets, model, sampling, concretization, repair hints, and ranking. Rename the production optimizer to `ArchitectureV11Optimizer`, keep v5/v4 aliases, and make diffusion inference use `DiffusionPlacementPrior` directly instead of converting to `AnchorGuidance`.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing FloorSet evaluator scripts, matplotlib for PNG output.

---

## Scope Check

This plan implements milestone 1 from `docs/superpowers/specs/2026-06-24-v11-diffusion-tree-decoder-design.md`.

Included:

- Variant A raw diffusion baseline.
- Variant B HGT-lite conditioned diffusion.
- v11 naming and v5/v4 compatibility.
- Diffusion graph input and output contracts.
- Training target utilities for `fp_sol`, `tree_sol`, and `metrics_sol`.
- Direct diffusion decoder, repair, and ranking path.
- Missing-checkpoint fallback with explicit metadata.
- Full-eval top-10 predicted floorplan PNG output and single-case PNG output.
- W&B defaults renamed to `floorset-v11-diffusion` for v11 diffusion training and related script/docs surfaces.

Excluded:

- Variant C or deeper/full HGT.
- Pure tree-only model state.
- Reinforcement learning.
- Evaluator-in-loop training.
- Feeding solution-only `metrics_sol` fields into inference.

## File Structure

Create:

- `src/floorset_arch/diffusion/__init__.py` exports public diffusion contracts and builders.
- `src/floorset_arch/diffusion/contracts.py` defines `Relation`, `DiffusionGraphInputs`, `DiffusionPlacementPrior`, `PlacementTensorBatch`, and validation helpers.
- `src/floorset_arch/diffusion/graph_inputs.py` builds deterministic `DiffusionGraphInputs` from `Instance`.
- `src/floorset_arch/diffusion/targets.py` builds diffusion training targets from `fp_sol`, `tree_sol`, and `metrics_sol`.
- `src/floorset_arch/diffusion/model.py` implements raw and HGT-lite diffusion denoiser components.
- `src/floorset_arch/diffusion/sampling.py` loads checkpoints and samples `DiffusionPlacementPrior`.
- `src/floorset_arch/diffusion/concretize.py` converts diffusion tensors to `PlacementTensorBatch` and selected `Placement` objects.
- `src/floorset_arch/diffusion/ranking.py` prefilters tensor candidates and ranks repaired placements with diffusion metadata.
- `src/floorset_arch/diffusion/training.py` provides a small diffusion training step API that can be exercised by tests before adding a long-running training script.
- `src/architecture_v11_optimizer.py` is the evaluator-facing v11 wrapper.
- `tests/test_diffusion_contracts.py` covers diffusion dataclass shapes and concretization.
- `tests/test_diffusion_graph_inputs.py` covers deterministic hetero graph inputs.
- `tests/test_diffusion_targets.py` covers `tree_sol` and `metrics_sol` target usage.
- `tests/test_diffusion_model.py` covers variant A/B forward and sampling shapes.
- `tests/test_architecture_v11.py` covers v11 aliases, fallback, and no `AnchorGuidance` conversion.
- `tests/test_eval_visualization.py` covers predicted PNG output helpers.

Modify:

- `src/floorset_arch/optimizer.py` renames the production class and adds the diffusion solve path.
- `src/floorset_arch/__init__.py` exports v11 plus v5/v4 aliases.
- `src/architecture_v5_optimizer.py` forwards to `ArchitectureV11Optimizer`.
- `scripts/iccad2026_evaluate.py` saves predicted floorplan PNGs from evaluation results.
- `scripts/eval_single.sh`, `scripts/eval_total.sh`, and `scripts/validate.sh` point at the v11 wrapper and pass floorplan output settings.
- `scripts/train.sh`, `scripts/train_transformer.sh`, `scripts/train_hgt.sh`, `src/floorset_arch/training/train.py`, and README training examples use v11 diffusion W&B defaults.
- `src/floorset_arch/training/train.py` preserves the anchor path and exposes `tree_sol` and `metrics_sol` to diffusion target preparation.

---

### Task 1: Rename Active Optimizer To V11 And Preserve V5/V4 Aliases

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `src/floorset_arch/__init__.py`
- Create: `src/architecture_v11_optimizer.py`
- Modify: `src/architecture_v5_optimizer.py`
- Modify: `scripts/validate.sh`
- Modify: `scripts/eval_single.sh`
- Modify: `scripts/eval_total.sh`
- Test: `tests/test_architecture_v11.py`

- [ ] **Step 1: Write failing v11 import and script tests**

Create `tests/test_architecture_v11.py` with:

```python
import importlib.util
from pathlib import Path

from floorset_arch.optimizer import (
    ArchitectureV4Optimizer,
    ArchitectureV5Optimizer,
    ArchitectureV11Optimizer,
)


def _load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v11_optimizer_aliases_preserve_legacy_names():
    assert ArchitectureV5Optimizer is ArchitectureV11Optimizer
    assert ArchitectureV4Optimizer is ArchitectureV11Optimizer


def test_v11_wrapper_exports_contest_optimizer():
    module = _load_wrapper(Path("src/architecture_v11_optimizer.py"))
    assert module.MyOptimizer is ArchitectureV11Optimizer
    assert module.ContestOptimizer is ArchitectureV11Optimizer


def test_v5_wrapper_forwards_to_v11_optimizer():
    module = _load_wrapper(Path("src/architecture_v5_optimizer.py"))
    assert module.MyOptimizer is ArchitectureV11Optimizer
    assert module.ContestOptimizer is ArchitectureV11Optimizer


def test_active_scripts_reference_v11_wrapper():
    for script in ["scripts/validate.sh", "scripts/eval_single.sh", "scripts/eval_total.sh"]:
        text = Path(script).read_text(encoding="utf-8")
        assert "architecture_v11_optimizer.py" in text
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
uv run pytest tests/test_architecture_v11.py -q
```

Expected: FAIL because `ArchitectureV11Optimizer` and `src/architecture_v11_optimizer.py` do not exist yet.

- [ ] **Step 3: Rename the optimizer class and add aliases**

In `src/floorset_arch/optimizer.py`, change:

```python
class ArchitectureV5Optimizer(FloorplanOptimizer):
    """Anchor-GNN guided hetero-graph beam solver with selectable checkpoint encoders."""
```

to:

```python
class ArchitectureV11Optimizer(FloorplanOptimizer):
    """Diffusion-ready floorplanning optimizer with legacy v5 fallback."""
```

At the bottom of `src/floorset_arch/optimizer.py`, replace the current alias with:

```python
ArchitectureV5Optimizer = ArchitectureV11Optimizer
ArchitectureV4Optimizer = ArchitectureV11Optimizer
```

Keep the existing method bodies unchanged in this task.

- [ ] **Step 4: Update package exports**

Replace `src/floorset_arch/__init__.py` with:

```python
"""Architecture v11 components for the ICCAD 2026 FloorSet challenge."""

from floorset_arch.optimizer import (
    ArchitectureV4Optimizer,
    ArchitectureV5Optimizer,
    ArchitectureV11Optimizer,
)

__all__ = [
    "ArchitectureV4Optimizer",
    "ArchitectureV5Optimizer",
    "ArchitectureV11Optimizer",
]
```

- [ ] **Step 5: Create the v11 wrapper and forward the v5 wrapper**

Create `src/architecture_v11_optimizer.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
for path in (SRC, CONTEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from floorset_arch.optimizer import ArchitectureV11Optimizer  # noqa: E402


MyOptimizer = ArchitectureV11Optimizer
ContestOptimizer = ArchitectureV11Optimizer
```

Replace `src/architecture_v5_optimizer.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
for path in (SRC, CONTEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from floorset_arch.optimizer import ArchitectureV11Optimizer  # noqa: E402


MyOptimizer = ArchitectureV11Optimizer
ContestOptimizer = ArchitectureV11Optimizer
```

- [ ] **Step 6: Point active scripts at v11**

In `scripts/validate.sh`, set:

```bash
uv run iccad2026_evaluate.py \
  --validate "$ROOT/src/architecture_v11_optimizer.py"
```

In `scripts/eval_single.sh`, set:

```bash
OPTIMIZER="$ROOT/src/architecture_v11_optimizer.py"
```

In `scripts/eval_total.sh`, set:

```bash
OPTIMIZER="$ROOT/src/architecture_v11_optimizer.py"
```

- [ ] **Step 7: Run tests and commit**

Run:

```bash
uv run pytest tests/test_architecture_v11.py tests/test_optimizer.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/optimizer.py src/floorset_arch/__init__.py src/architecture_v11_optimizer.py src/architecture_v5_optimizer.py scripts/validate.sh scripts/eval_single.sh scripts/eval_total.sh tests/test_architecture_v11.py
git commit -m "refactor: expose architecture v11 optimizer"
```

---

### Task 2: Add Diffusion Contracts And Deterministic Graph Inputs

**Files:**
- Create: `src/floorset_arch/diffusion/__init__.py`
- Create: `src/floorset_arch/diffusion/contracts.py`
- Create: `src/floorset_arch/diffusion/graph_inputs.py`
- Test: `tests/test_diffusion_contracts.py`
- Test: `tests/test_diffusion_graph_inputs.py`

- [ ] **Step 1: Write failing contract tests**

Create `tests/test_diffusion_contracts.py`:

```python
import pytest
import torch

from floorset_arch.diffusion.contracts import (
    DiffusionPlacementPrior,
    PlacementTensorBatch,
)


def test_diffusion_prior_validates_center_and_aspect_shapes():
    prior = DiffusionPlacementPrior(
        centers=torch.zeros(2, 3, 2),
        log_aspect=torch.zeros(2, 3),
        pairwise_axis_logits=torch.zeros(2, 4, 3),
        pair_index=torch.tensor([[0, 1], [0, 2], [1, 2], [2, 0]]),
        scale=10.0,
        variant="raw",
    )

    assert prior.sample_count == 2
    assert prior.block_count == 3
    assert prior.pair_count == 4


def test_diffusion_prior_rejects_mismatched_pair_logits():
    with pytest.raises(ValueError, match="pairwise_axis_logits"):
        DiffusionPlacementPrior(
            centers=torch.zeros(1, 3, 2),
            log_aspect=torch.zeros(1, 3),
            pairwise_axis_logits=torch.zeros(1, 2, 3),
            pair_index=torch.tensor([[0, 1]]),
        )


def test_placement_tensor_batch_exposes_topk_slice():
    batch = PlacementTensorBatch(
        rect_xywh=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        pairwise_axis_logits=torch.zeros(2, 1, 3),
        pair_index=torch.tensor([[0, 1]]),
        source="unit",
    )

    sliced = batch.select(torch.tensor([1]))

    assert sliced.rect_xywh.shape == (1, 3, 4)
    assert sliced.source == "unit"
```

- [ ] **Step 2: Write failing graph input tests**

Create `tests/test_diffusion_graph_inputs.py`:

```python
import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.parser import parse_instance


def _sample_instance():
    return parse_instance(
        4,
        torch.tensor([4.0, 9.0, 16.0, 25.0]),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 3.0, 3.0]]),
        torch.tensor([[10.0, 20.0], [30.0, 5.0]]),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 7.0, 1.0],
                [0.0, 0.0, 1.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 7.0, 2.0],
                [1.0, 1.0, 0.0, 0.0, 4.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 3.0, 3.0],
                [0.0, 0.0, 4.0, 4.0],
                [7.0, 8.0, 5.0, 5.0],
            ]
        ),
    )


def test_diffusion_graph_inputs_keep_raw_side_channels_and_typed_relations():
    graph = build_diffusion_graph_inputs(_sample_instance())

    assert {"block", "pin", "cluster", "mib", "boundary"} <= set(graph.node_features)
    assert graph.raw_block_features.shape[0] == 4
    assert graph.area.shape == (4,)
    assert graph.fixed_mask.tolist() == [False, False, False, True]
    assert graph.preplaced_mask.tolist() == [False, False, False, True]
    assert graph.movable_mask.tolist() == [True, True, True, False]
    assert ("block", "connects", "block") in graph.relation_specs
    assert ("boundary", "has_member", "block") in graph.relation_specs
    assert graph.pair_index.shape[1] == 2
    assert graph.raw_pair_features.shape[0] == graph.pair_index.shape[0]


def test_parse_instance_stays_model_agnostic():
    inst = _sample_instance()

    assert inst.__class__.__name__ == "Instance"
    assert not hasattr(inst, "node_features")
    assert not hasattr(inst, "pair_index")
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_diffusion_contracts.py tests/test_diffusion_graph_inputs.py -q
```

Expected: FAIL because `floorset_arch.diffusion` does not exist.

- [ ] **Step 4: Implement diffusion contracts**

Create `src/floorset_arch/diffusion/contracts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


Relation = tuple[str, str, str]


@dataclass(frozen=True)
class DiffusionGraphInputs:
    node_features: dict[str, torch.Tensor]
    edge_index: dict[Relation, torch.Tensor]
    edge_attr: dict[Relation, torch.Tensor]
    relation_specs: Sequence[Relation]
    raw_block_features: torch.Tensor
    raw_pair_features: torch.Tensor
    pair_index: torch.Tensor
    area: torch.Tensor
    scale: float
    fixed_mask: torch.Tensor
    preplaced_mask: torch.Tensor
    movable_mask: torch.Tensor
    boundary_codes: torch.Tensor
    cluster_ids: torch.Tensor
    mib_ids: torch.Tensor
    global_features: torch.Tensor

    def __post_init__(self) -> None:
        block_count = int(self.raw_block_features.shape[0])
        if self.area.shape != (block_count,):
            raise ValueError("area must have shape [N]")
        if self.fixed_mask.shape != (block_count,):
            raise ValueError("fixed_mask must have shape [N]")
        if self.preplaced_mask.shape != (block_count,):
            raise ValueError("preplaced_mask must have shape [N]")
        if self.movable_mask.shape != (block_count,):
            raise ValueError("movable_mask must have shape [N]")
        if self.pair_index.dim() != 2 or self.pair_index.shape[1] != 2:
            raise ValueError("pair_index must have shape [P, 2]")
        if self.raw_pair_features.shape[0] != self.pair_index.shape[0]:
            raise ValueError("raw_pair_features must align with pair_index")


@dataclass
class DiffusionPlacementPrior:
    centers: torch.Tensor
    log_aspect: torch.Tensor
    pairwise_axis_logits: torch.Tensor
    pair_index: torch.Tensor
    quality_pred: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None
    scale: float = 1.0
    variant: str = ""

    def __post_init__(self) -> None:
        if self.centers.dim() != 3 or self.centers.shape[-1] != 2:
            raise ValueError("centers must have shape [S, N, 2]")
        if self.log_aspect.shape != self.centers.shape[:2]:
            raise ValueError("log_aspect must have shape [S, N]")
        if self.pair_index.dim() != 2 or self.pair_index.shape[1] != 2:
            raise ValueError("pair_index must have shape [P, 2]")
        expected_pair_shape = (self.centers.shape[0], self.pair_index.shape[0])
        if self.pairwise_axis_logits.shape[:2] != expected_pair_shape:
            raise ValueError("pairwise_axis_logits must have shape [S, P, C]")
        if self.quality_pred is not None and self.quality_pred.shape != (self.centers.shape[0],):
            raise ValueError("quality_pred must have shape [S]")

    @property
    def sample_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def block_count(self) -> int:
        return int(self.centers.shape[1])

    @property
    def pair_count(self) -> int:
        return int(self.pair_index.shape[0])


@dataclass
class PlacementTensorBatch:
    rect_xywh: torch.Tensor
    pairwise_axis_logits: torch.Tensor
    pair_index: torch.Tensor
    score_features: torch.Tensor | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.rect_xywh.dim() != 3 or self.rect_xywh.shape[-1] != 4:
            raise ValueError("rect_xywh must have shape [S, N, 4]")
        expected_pair_shape = (self.rect_xywh.shape[0], self.pair_index.shape[0])
        if self.pairwise_axis_logits.shape[:2] != expected_pair_shape:
            raise ValueError("pairwise_axis_logits must align with pair_index")

    def select(self, indices: torch.Tensor | Sequence[int]) -> "PlacementTensorBatch":
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.rect_xywh.device)
        score_features = None if self.score_features is None else self.score_features[idx]
        return PlacementTensorBatch(
            rect_xywh=self.rect_xywh[idx],
            pairwise_axis_logits=self.pairwise_axis_logits[idx],
            pair_index=self.pair_index,
            score_features=score_features,
            source=self.source,
        )
```

Create `src/floorset_arch/diffusion/__init__.py`:

```python
from floorset_arch.diffusion.contracts import (
    DiffusionGraphInputs,
    DiffusionPlacementPrior,
    PlacementTensorBatch,
    Relation,
)
from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs

__all__ = [
    "DiffusionGraphInputs",
    "DiffusionPlacementPrior",
    "PlacementTensorBatch",
    "Relation",
    "build_diffusion_graph_inputs",
]
```

- [ ] **Step 5: Implement graph input builder**

Create `src/floorset_arch/diffusion/graph_inputs.py` with:

```python
from __future__ import annotations

from collections import defaultdict

import torch

from floorset_arch.diffusion.contracts import DiffusionGraphInputs, Relation
from floorset_arch.features import (
    ANCHOR_HGT_RELATION_SPECS,
    build_anchor_node_features,
)
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.models import Instance


def _empty_features(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.empty((rows, cols), dtype=torch.float32, device=device)


def _factor_features(groups: dict[int, list[int]], inst: Instance, device: torch.device) -> torch.Tensor:
    rows: list[list[float]] = []
    total_area = float(inst.area_targets[: inst.block_count].float().clamp_min(0).sum().clamp_min(1.0).item())
    for group_id in sorted(groups):
        members = [block for block in groups[group_id] if 0 <= block < inst.block_count]
        area = float(inst.area_targets[members].float().clamp_min(0).sum().item()) if members else 0.0
        rows.append([len(members) / max(float(inst.block_count), 1.0), area / total_area])
    if not rows:
        return _empty_features(0, 2, device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _boundary_features(inst: Instance, device: torch.device) -> torch.Tensor:
    rows = [[1.0 if int(code) & bit else 0.0 for bit in (1, 2, 4, 8)] for code in sorted(set(inst.boundary.values()))]
    if not rows:
        return _empty_features(0, 4, device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _block_group_ids(groups: dict[int, list[int]], block_count: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(block_count, dtype=torch.long, device=device)
    for local_id, group_id in enumerate(sorted(groups), start=1):
        for block in groups[group_id]:
            if 0 <= block < block_count:
                ids[block] = local_id
    return ids


def _pair_index(inst: Instance, device: torch.device) -> torch.Tensor:
    pairs: set[tuple[int, int]] = set()
    for i_f, j_f, _weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if 0 <= i < inst.block_count and 0 <= j < inst.block_count and i != j:
            pairs.add((min(i, j), max(i, j)))
    for groups in (inst.cluster_groups, inst.mib_groups):
        for blocks in groups.values():
            valid = sorted(block for block in blocks if 0 <= block < inst.block_count)
            for pos, i in enumerate(valid):
                for j in valid[pos + 1 :]:
                    pairs.add((i, j))
    if not pairs:
        for i in range(inst.block_count):
            for j in range(i + 1, inst.block_count):
                pairs.add((i, j))
    return torch.tensor(sorted(pairs), dtype=torch.long, device=device)


def _pair_features(inst: Instance, pairs: torch.Tensor, device: torch.device) -> torch.Tensor:
    b2b_weight: dict[tuple[int, int], float] = defaultdict(float)
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        key = (min(i, j), max(i, j))
        b2b_weight[key] += max(float(weight_f), 0.0)
    max_b2b = max(b2b_weight.values(), default=1.0)
    cluster = _block_group_ids(inst.cluster_groups, inst.block_count, device=torch.device("cpu"))
    mib = _block_group_ids(inst.mib_groups, inst.block_count, device=torch.device("cpu"))
    boundary = torch.zeros(inst.block_count, dtype=torch.long)
    for block, code in inst.boundary.items():
        if 0 <= block < inst.block_count:
            boundary[block] = int(code)
    area = inst.area_targets[: inst.block_count].float().clamp_min(1.0).cpu()
    rows: list[list[float]] = []
    for i, j in pairs.detach().cpu().tolist():
        weight = b2b_weight.get((i, j), 0.0)
        rows.append(
            [
                1.0 if weight > 0 else 0.0,
                weight / max(max_b2b, 1.0),
                1.0 if cluster[i] > 0 and cluster[i] == cluster[j] else 0.0,
                1.0 if mib[i] > 0 and mib[i] == mib[j] else 0.0,
                1.0 if boundary[i] > 0 and boundary[i] == boundary[j] else 0.0,
                float(torch.log(area[i] / area[j]).clamp(-4.0, 4.0).item()),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _global_features(inst: Instance, device: torch.device) -> torch.Tensor:
    n = max(float(inst.block_count), 1.0)
    values = [
        inst.block_count / 128.0,
        float(inst.valid_b2b.shape[0]) / max(n * n, 1.0),
        float(inst.valid_p2b.shape[0]) / max(n, 1.0),
        len(inst.fixed) / n,
        len(inst.preplaced) / n,
        len(inst.boundary) / n,
        len(inst.cluster_groups) / n,
        len(inst.mib_groups) / n,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_diffusion_graph_inputs(inst: Instance, device: torch.device | None = None) -> DiffusionGraphInputs:
    device = device or inst.area_targets.device
    block_features, scale = build_anchor_node_features(inst, device=device)
    graph = build_hetero_floorplan_graph(inst, block_features=block_features.detach().cpu())

    pins = inst.pins_pos.float().to(device)
    if pins.numel() > 0:
        valid_pin = ((pins[:, 0] != -1.0) & (pins[:, 1] != -1.0)).float().view(-1, 1)
        pin_features = torch.cat([pins / max(float(scale), 1.0), valid_pin], dim=1)
    else:
        pin_features = _empty_features(0, 3, device)

    node_features = {
        "block": block_features,
        "pin": pin_features,
        "cluster": _factor_features(inst.cluster_groups, inst, device),
        "mib": _factor_features(inst.mib_groups, inst, device),
        "boundary": _boundary_features(inst, device),
    }

    src_by_relation: dict[Relation, list[int]] = {}
    dst_by_relation: dict[Relation, list[int]] = {}
    weights_by_relation: dict[Relation, list[float]] = {}
    for edge in graph.edges:
        relation = (edge.src_type, edge.edge_type, edge.dst_type)
        if relation not in ANCHOR_HGT_RELATION_SPECS:
            continue
        if edge.src >= node_features[edge.src_type].shape[0]:
            continue
        if edge.dst >= node_features[edge.dst_type].shape[0]:
            continue
        src_by_relation.setdefault(relation, []).append(int(edge.src))
        dst_by_relation.setdefault(relation, []).append(int(edge.dst))
        weights_by_relation.setdefault(relation, []).append(max(float(edge.weight), 0.0))

    edge_index: dict[Relation, torch.Tensor] = {}
    edge_attr: dict[Relation, torch.Tensor] = {}
    for relation in ANCHOR_HGT_RELATION_SPECS:
        if relation not in src_by_relation:
            edge_index[relation] = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr[relation] = torch.empty((0, 1), dtype=torch.float32, device=device)
            continue
        edge_index[relation] = torch.tensor([src_by_relation[relation], dst_by_relation[relation]], dtype=torch.long, device=device)
        weights = torch.tensor(weights_by_relation[relation], dtype=torch.float32, device=device).view(-1, 1)
        edge_attr[relation] = weights / weights.max().clamp_min(1.0)

    area = inst.area_targets[: inst.block_count].float().to(device).clamp_min(1.0)
    fixed_mask = torch.tensor([i in inst.fixed for i in range(inst.block_count)], dtype=torch.bool, device=device)
    preplaced_mask = torch.tensor([i in inst.preplaced for i in range(inst.block_count)], dtype=torch.bool, device=device)
    movable_mask = ~(fixed_mask | preplaced_mask)
    boundary_codes = torch.tensor([int(inst.boundary.get(i, 0)) for i in range(inst.block_count)], dtype=torch.long, device=device)
    cluster_ids = _block_group_ids(inst.cluster_groups, inst.block_count, device)
    mib_ids = _block_group_ids(inst.mib_groups, inst.block_count, device)
    pairs = _pair_index(inst, device)

    return DiffusionGraphInputs(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        relation_specs=tuple(ANCHOR_HGT_RELATION_SPECS),
        raw_block_features=block_features,
        raw_pair_features=_pair_features(inst, pairs, device),
        pair_index=pairs,
        area=area,
        scale=max(float(scale), 1.0),
        fixed_mask=fixed_mask,
        preplaced_mask=preplaced_mask,
        movable_mask=movable_mask,
        boundary_codes=boundary_codes,
        cluster_ids=cluster_ids,
        mib_ids=mib_ids,
        global_features=_global_features(inst, device),
    )
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/test_diffusion_contracts.py tests/test_diffusion_graph_inputs.py tests/test_model.py::test_hgt_graph_inputs_keep_typed_local_relations -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/diffusion tests/test_diffusion_contracts.py tests/test_diffusion_graph_inputs.py
git commit -m "feat: add diffusion graph input contracts"
```

---

### Task 3: Add Diffusion Training Targets For fp_sol, tree_sol, And metrics_sol

**Files:**
- Create: `src/floorset_arch/diffusion/targets.py`
- Modify: `src/floorset_arch/training/train.py`
- Test: `tests/test_diffusion_targets.py`

- [ ] **Step 1: Write failing target tests**

Create `tests/test_diffusion_targets.py`:

```python
import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.targets import (
    build_diffusion_targets,
    parse_tree_sol_edges,
    split_metrics_sol,
)
from floorset_arch.parser import parse_instance


def _inst():
    return parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        None,
    )


def test_parse_tree_sol_edges_filters_padding_and_invalid_rows():
    tree_sol = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [-1.0, -1.0, -1.0],
            [3.0, 0.0, 0.0],
        ]
    )

    edges = parse_tree_sol_edges(tree_sol, block_count=3)

    assert edges.parent.tolist() == [0, 1]
    assert edges.child.tolist() == [1, 2]
    assert edges.side.tolist() == [0, 1]


def test_metrics_sol_split_keeps_solution_quality_out_of_inference_features():
    metrics = torch.tensor([100.0, 4.0, 7.0, 3.0, 4.0, 2.0, 55.0, 66.0])

    split = split_metrics_sol(metrics)

    assert split.instance_stats.tolist() == [4.0, 7.0, 3.0, 4.0, 2.0]
    assert split.quality_labels.tolist() == [100.0, 55.0, 66.0]


def test_build_diffusion_targets_aligns_pair_index_with_tree_labels():
    inst = _inst()
    graph_inputs = build_diffusion_graph_inputs(inst)
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0, 4.0],
        ]
    )
    tree_sol = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics_sol = torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0])

    targets = build_diffusion_targets(inst, graph_inputs, fp_sol, tree_sol, metrics_sol)

    assert targets.center.shape == (3, 2)
    assert targets.log_aspect.shape == (3,)
    assert targets.pair_axis_label.shape[0] == graph_inputs.pair_index.shape[0]
    assert targets.tree_pair_mask.any()
    assert targets.quality_labels.tolist() == [25.0, 10.0, 12.0]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_diffusion_targets.py -q
```

Expected: FAIL because `floorset_arch.diffusion.targets` does not exist.

- [ ] **Step 3: Implement target utilities**

Create `src/floorset_arch/diffusion/targets.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch

from floorset_arch.diffusion.contracts import DiffusionGraphInputs
from floorset_arch.models import Instance


@dataclass(frozen=True)
class TreeEdges:
    parent: torch.Tensor
    child: torch.Tensor
    side: torch.Tensor


@dataclass(frozen=True)
class MetricsSplit:
    instance_stats: torch.Tensor
    quality_labels: torch.Tensor


@dataclass(frozen=True)
class DiffusionTargets:
    center: torch.Tensor
    log_aspect: torch.Tensor
    pair_axis_label: torch.Tensor
    pair_axis_mask: torch.Tensor
    tree_side_label: torch.Tensor
    tree_pair_mask: torch.Tensor
    quality_labels: torch.Tensor


def parse_tree_sol_edges(tree_sol: torch.Tensor, block_count: int) -> TreeEdges:
    rows = torch.as_tensor(tree_sol).detach().cpu().float().reshape(-1, 3)
    parent: list[int] = []
    child: list[int] = []
    side: list[int] = []
    for parent_f, child_f, side_f in rows.tolist():
        p = int(parent_f)
        c = int(child_f)
        s = int(side_f)
        if not (0 <= p < block_count and 0 <= c < block_count and p != c):
            continue
        if s not in (0, 1):
            continue
        parent.append(p)
        child.append(c)
        side.append(s)
    return TreeEdges(
        parent=torch.tensor(parent, dtype=torch.long),
        child=torch.tensor(child, dtype=torch.long),
        side=torch.tensor(side, dtype=torch.long),
    )


def split_metrics_sol(metrics_sol: torch.Tensor) -> MetricsSplit:
    metrics = torch.as_tensor(metrics_sol).detach().cpu().float().flatten()
    padded = torch.zeros(8, dtype=torch.float32)
    padded[: min(8, metrics.numel())] = metrics[:8]
    instance_stats = padded[[1, 2, 3, 4, 5]]
    quality_labels = padded[[0, 6, 7]]
    return MetricsSplit(instance_stats=instance_stats, quality_labels=quality_labels)


def _fp_sol_xywh(fp_sol: torch.Tensor, block_count: int) -> torch.Tensor:
    raw = torch.as_tensor(fp_sol).detach().float().reshape(-1, 4)[:block_count]
    width = raw[:, 0].clamp_min(1.0)
    height = raw[:, 1].clamp_min(1.0)
    x = raw[:, 2].clamp_min(0.0)
    y = raw[:, 3].clamp_min(0.0)
    return torch.stack([x, y, width, height], dim=1)


def build_diffusion_targets(
    inst: Instance,
    graph_inputs: DiffusionGraphInputs,
    fp_sol: torch.Tensor,
    tree_sol: torch.Tensor,
    metrics_sol: torch.Tensor,
) -> DiffusionTargets:
    xywh = _fp_sol_xywh(fp_sol, inst.block_count).to(graph_inputs.area.device)
    center = torch.stack([xywh[:, 0] + xywh[:, 2] * 0.5, xywh[:, 1] + xywh[:, 3] * 0.5], dim=1)
    center = center / max(float(graph_inputs.scale), 1.0)
    log_aspect = torch.log((xywh[:, 2] / xywh[:, 3].clamp_min(1.0)).clamp_min(1e-6))

    pair_count = graph_inputs.pair_index.shape[0]
    pair_axis_label = torch.zeros(pair_count, dtype=torch.long, device=graph_inputs.area.device)
    pair_axis_mask = torch.zeros(pair_count, dtype=torch.bool, device=graph_inputs.area.device)
    for idx, (i, j) in enumerate(graph_inputs.pair_index.detach().cpu().tolist()):
        dx = center[j, 0] - center[i, 0]
        dy = center[j, 1] - center[i, 1]
        if abs(float(dx)) >= abs(float(dy)):
            pair_axis_label[idx] = 0 if dx >= 0 else 1
        else:
            pair_axis_label[idx] = 2
        pair_axis_mask[idx] = True

    tree_edges = parse_tree_sol_edges(tree_sol, inst.block_count)
    tree_side_label = torch.zeros(pair_count, dtype=torch.long, device=graph_inputs.area.device)
    tree_pair_mask = torch.zeros(pair_count, dtype=torch.bool, device=graph_inputs.area.device)
    pair_to_idx = {
        (min(int(i), int(j)), max(int(i), int(j))): idx
        for idx, (i, j) in enumerate(graph_inputs.pair_index.detach().cpu().tolist())
    }
    for p, c, side in zip(tree_edges.parent.tolist(), tree_edges.child.tolist(), tree_edges.side.tolist()):
        key = (min(p, c), max(p, c))
        if key in pair_to_idx:
            idx = pair_to_idx[key]
            tree_side_label[idx] = int(side)
            tree_pair_mask[idx] = True

    metrics = split_metrics_sol(metrics_sol)
    return DiffusionTargets(
        center=center,
        log_aspect=log_aspect,
        pair_axis_label=pair_axis_label,
        pair_axis_mask=pair_axis_mask,
        tree_side_label=tree_side_label,
        tree_pair_mask=tree_pair_mask,
        quality_labels=metrics.quality_labels.to(graph_inputs.area.device),
    )
```

- [ ] **Step 4: Expose tree_sol and metrics_sol from training unpack helpers**

Modify `src/floorset_arch/training/train.py` so `unpack_batch()` returns `tree_sol` and `metrics`:

```python
def unpack_batch(batch):
    area_targets, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics = batch
    return (
        area_targets.squeeze(0),
        b2b.squeeze(0),
        p2b.squeeze(0),
        pins.squeeze(0),
        constraints.squeeze(0),
        tree_sol.squeeze(0),
        fp_sol.squeeze(0),
        metrics.squeeze(0),
    )
```

Modify `unpack_batch_sample()` return shape the same way:

```python
return (
    area_targets[index],
    b2b[index],
    p2b[index],
    pins[index],
    constraints[index],
    tree_sol[index],
    fp_sol[index],
    metrics[index],
)
```

Then update anchor-only callers in `train.py` to unpack the extra value as `_tree_sol`. For example:

```python
area_targets, b2b, p2b, pins, constraints, _tree_sol, fp_sol, _metrics = sample
```

and:

```python
area_targets, b2b, p2b, pins, constraints, _tree_sol, _fp_sol, _metrics = (
    unpack_batch_sample(first, 0)
)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/test_diffusion_targets.py tests/test_model.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/diffusion/targets.py src/floorset_arch/training/train.py tests/test_diffusion_targets.py
git commit -m "feat: add diffusion training targets"
```

---

### Task 4: Implement Raw And HGT-Lite Diffusion Model With Sampling

**Files:**
- Create: `src/floorset_arch/diffusion/model.py`
- Create: `src/floorset_arch/diffusion/sampling.py`
- Modify: `src/floorset_arch/diffusion/__init__.py`
- Test: `tests/test_diffusion_model.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_diffusion_model.py`:

```python
import torch

from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.sampling import sample_diffusion_prior
from floorset_arch.parser import parse_instance


def _graph_inputs():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        torch.tensor([[0.0, 2.0, 1.0]]),
        torch.tensor([[5.0, 7.0]]),
        torch.zeros(3, 5),
        None,
    )
    return build_diffusion_graph_inputs(inst)


def test_raw_diffusion_forward_shapes():
    graph = _graph_inputs()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(graph, variant="raw", hidden_dim=24, layers=1)
    noisy = torch.zeros(2, 3, 3)
    timesteps = torch.tensor([4, 4], dtype=torch.long)

    out = model(graph, noisy, timesteps)

    assert out["eps_pred"].shape == (2, 3, 3)
    assert out["pairwise_axis_logits"].shape == (2, graph.pair_index.shape[0], 3)
    assert out["quality_pred"].shape == (2,)


def test_hgt_lite_diffusion_forward_shapes():
    graph = _graph_inputs()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(graph, variant="hgt_lite", hidden_dim=24, layers=1)
    noisy = torch.zeros(1, 3, 3)
    timesteps = torch.tensor([2], dtype=torch.long)

    out = model(graph, noisy, timesteps)

    assert out["eps_pred"].shape == (1, 3, 3)
    assert out["pairwise_axis_logits"].shape[1] == graph.pair_index.shape[0]


def test_sampling_returns_prior_without_anchor_guidance():
    graph = _graph_inputs()
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(graph, variant="raw", hidden_dim=16, layers=1)

    prior = sample_diffusion_prior(model, graph, samples=2, steps=3, seed=11)

    assert prior.centers.shape == (2, 3, 2)
    assert prior.log_aspect.shape == (2, 3)
    assert prior.pair_index.shape == graph.pair_index.shape
    assert not hasattr(prior, "rect_priors")
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_diffusion_model.py -q
```

Expected: FAIL because `model.py` and `sampling.py` do not exist.

- [ ] **Step 3: Implement the diffusion model**

Create `src/floorset_arch/diffusion/model.py`:

```python
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence

from floorset_arch.diffusion.contracts import DiffusionGraphInputs, Relation


def _time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
    args = timesteps.float().view(-1, 1) * freq.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class HGTLiteConditioner(nn.Module):
    def __init__(
        self,
        node_feat_dims: dict[str, int],
        relation_specs: Sequence[Relation],
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.node_types = tuple(node_feat_dims)
        self.relation_specs = tuple(relation_specs)
        self.input = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
                for node_type, dim in node_feat_dims.items()
            }
        )
        self.relation_msg = nn.ModuleDict(
            {"__".join(relation): nn.Linear(hidden_dim + 1, hidden_dim) for relation in relation_specs}
        )
        self.updates = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU(),
                        )
                        for node_type in self.node_types
                    }
                )
                for _ in range(max(1, layers))
            ]
        )

    def forward(self, graph: DiffusionGraphInputs) -> torch.Tensor:
        states = {
            node_type: self.input[node_type](features.float())
            for node_type, features in graph.node_features.items()
            if node_type in self.input
        }
        for update in self.updates:
            messages = {node_type: torch.zeros_like(state) for node_type, state in states.items()}
            for relation in self.relation_specs:
                src_type, _edge_type, dst_type = relation
                if src_type not in states or dst_type not in states:
                    continue
                edges = graph.edge_index[relation]
                if edges.numel() == 0:
                    continue
                src = edges[0].to(states[src_type].device)
                dst = edges[1].to(states[dst_type].device)
                attr = graph.edge_attr[relation].to(states[src_type].device)
                key = "__".join(relation)
                msg = self.relation_msg[key](torch.cat([states[src_type][src], attr], dim=1))
                messages[dst_type].index_add_(0, dst, msg)
            states = {node_type: states[node_type] + update[node_type](messages[node_type]) for node_type in states}
        return states["block"]


class GraphConditionedPlacementDiffusion(nn.Module):
    def __init__(
        self,
        raw_block_dim: int,
        raw_pair_dim: int,
        global_dim: int,
        node_feat_dims: dict[str, int],
        relation_specs: Sequence[Relation],
        variant: str = "raw",
        hidden_dim: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if variant not in {"raw", "hgt_lite"}:
            raise ValueError("variant must be raw or hgt_lite")
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.block_in = nn.Linear(raw_block_dim + 3 + hidden_dim + hidden_dim, hidden_dim)
        self.global_in = nn.Linear(global_dim, hidden_dim)
        self.time_dim = hidden_dim
        self.hgt = None
        if variant == "hgt_lite":
            self.hgt = HGTLiteConditioner(node_feat_dims, relation_specs, hidden_dim, layers)
        self.block_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + raw_pair_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.quality_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    @classmethod
    def from_graph_inputs(
        cls,
        graph: DiffusionGraphInputs,
        variant: str,
        hidden_dim: int = 128,
        layers: int = 2,
    ) -> "GraphConditionedPlacementDiffusion":
        return cls(
            raw_block_dim=int(graph.raw_block_features.shape[1]),
            raw_pair_dim=int(graph.raw_pair_features.shape[1]),
            global_dim=int(graph.global_features.numel()),
            node_feat_dims={key: int(value.shape[1]) for key, value in graph.node_features.items()},
            relation_specs=graph.relation_specs,
            variant=variant,
            hidden_dim=hidden_dim,
            layers=layers,
        )

    def _block_context(self, graph: DiffusionGraphInputs) -> torch.Tensor:
        if self.hgt is None:
            return graph.raw_block_features.new_zeros((graph.raw_block_features.shape[0], self.hidden_dim))
        return self.hgt(graph)

    def forward(self, graph: DiffusionGraphInputs, x_t: torch.Tensor, timesteps: torch.Tensor) -> dict[str, torch.Tensor]:
        samples, blocks, _dims = x_t.shape
        block_context = self._block_context(graph)
        global_context = self.global_in(graph.global_features.float()).view(1, 1, -1).expand(samples, blocks, -1)
        time_context = _time_embedding(timesteps.to(x_t.device), self.time_dim).view(samples, 1, -1).expand(samples, blocks, -1)
        raw_block = graph.raw_block_features.float().to(x_t.device).view(1, blocks, -1).expand(samples, blocks, -1)
        hgt_block = block_context.to(x_t.device).view(1, blocks, -1).expand(samples, blocks, -1)
        h = self.block_in(torch.cat([raw_block, x_t.float(), hgt_block, time_context + global_context], dim=2))
        eps_pred = self.block_mlp(h)

        pair_index = graph.pair_index.to(x_t.device)
        hi = h[:, pair_index[:, 0], :]
        hj = h[:, pair_index[:, 1], :]
        xi = x_t[:, pair_index[:, 0], :2]
        xj = x_t[:, pair_index[:, 1], :2]
        delta = xj - xi
        distance = torch.linalg.norm(delta, dim=2, keepdim=True)
        raw_pair = graph.raw_pair_features.float().to(x_t.device).view(1, pair_index.shape[0], -1).expand(samples, -1, -1)
        pair_features = torch.cat([hi, hj, raw_pair, delta, distance, distance.clamp_min(1e-6).reciprocal()], dim=2)
        pairwise_axis_logits = self.pair_mlp(pair_features)
        quality_pred = self.quality_head(h.mean(dim=1)).flatten()
        return {
            "eps_pred": eps_pred,
            "pairwise_axis_logits": pairwise_axis_logits,
            "quality_pred": quality_pred,
        }
```

- [ ] **Step 4: Implement minimal sampler and checkpoint loader**

Create `src/floorset_arch/diffusion/sampling.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch

from floorset_arch.diffusion.contracts import DiffusionGraphInputs, DiffusionPlacementPrior
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion


def sample_diffusion_prior(
    model: GraphConditionedPlacementDiffusion,
    graph: DiffusionGraphInputs,
    samples: int = 8,
    steps: int = 16,
    seed: int | None = None,
) -> DiffusionPlacementPrior:
    generator = torch.Generator(device=graph.area.device)
    if seed is not None:
        generator.manual_seed(int(seed))
    block_count = graph.area.shape[0]
    x_t = torch.randn((samples, block_count, 3), generator=generator, device=graph.area.device)
    pair_logits = torch.zeros((samples, graph.pair_index.shape[0], 3), device=graph.area.device)
    quality = torch.zeros(samples, device=graph.area.device)
    for step in reversed(range(max(1, steps))):
        t = torch.full((samples,), step, dtype=torch.long, device=graph.area.device)
        out = model(graph, x_t, t)
        eps = out["eps_pred"]
        x_t = x_t - eps / float(max(steps, 1))
        pair_logits = out["pairwise_axis_logits"]
        quality = out["quality_pred"]
    centers = x_t[:, :, :2] * max(float(graph.scale), 1.0)
    log_aspect = x_t[:, :, 2].clamp(-2.5, 2.5)
    return DiffusionPlacementPrior(
        centers=centers,
        log_aspect=log_aspect,
        pairwise_axis_logits=pair_logits,
        pair_index=graph.pair_index,
        quality_pred=quality,
        uncertainty=x_t.detach().std(dim=2),
        scale=graph.scale,
        variant=model.variant,
    )


def load_diffusion_checkpoint(path: Path, graph: DiffusionGraphInputs, map_location: str | torch.device = "cpu") -> GraphConditionedPlacementDiffusion:
    payload = torch.load(path, map_location=map_location)
    variant = str(payload.get("variant", "raw"))
    hidden_dim = int(payload.get("hidden_dim", 128))
    layers = int(payload.get("layers", 2))
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(graph, variant=variant, hidden_dim=hidden_dim, layers=layers)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model
```

Update `src/floorset_arch/diffusion/__init__.py` to export `GraphConditionedPlacementDiffusion`, `sample_diffusion_prior`, and `load_diffusion_checkpoint`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/test_diffusion_model.py tests/test_diffusion_contracts.py tests/test_diffusion_graph_inputs.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/diffusion tests/test_diffusion_model.py
git commit -m "feat: add graph-conditioned diffusion sampler"
```

---

### Task 5: Add Diffusion Concretization, Repair Hints, And Ranking

**Files:**
- Create: `src/floorset_arch/diffusion/concretize.py`
- Create: `src/floorset_arch/diffusion/ranking.py`
- Modify: `src/floorset_arch/diffusion/__init__.py`
- Test: `tests/test_diffusion_contracts.py`

- [ ] **Step 1: Extend failing contract tests for concretization and ranking**

Append to `tests/test_diffusion_contracts.py`:

```python
from floorset_arch.diffusion.concretize import (
    concretize_diffusion_prior,
    placement_from_tensor_candidate,
)
from floorset_arch.diffusion.ranking import select_tensor_shortlist
from floorset_arch.models import Rect
from floorset_arch.parser import parse_instance


def _concretize_instance():
    return parse_instance(
        2,
        torch.tensor([4.0, 9.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, -1.0, -1.0, -1.0], [10.0, 20.0, 3.0, 3.0]]),
    )


def test_concretize_preserves_area_and_preplaced_blocks():
    inst = _concretize_instance()
    prior = DiffusionPlacementPrior(
        centers=torch.tensor([[[5.0, 5.0], [1.0, 1.0]]]),
        log_aspect=torch.zeros(1, 2),
        pairwise_axis_logits=torch.zeros(1, 1, 3),
        pair_index=torch.tensor([[0, 1]]),
        variant="raw",
    )

    batch = concretize_diffusion_prior(inst, prior)
    placement = placement_from_tensor_candidate(inst, batch, 0)

    assert batch.rect_xywh.shape == (1, 2, 4)
    assert round(float(batch.rect_xywh[0, 0, 2] * batch.rect_xywh[0, 0, 3]), 5) == 4.0
    assert placement.rects[1] == Rect(10.0, 20.0, 3.0, 3.0)


def test_tensor_shortlist_prefers_lower_score_features():
    batch = PlacementTensorBatch(
        rect_xywh=torch.zeros(3, 2, 4),
        pairwise_axis_logits=torch.zeros(3, 1, 3),
        pair_index=torch.tensor([[0, 1]]),
        score_features=torch.tensor([[10.0], [1.0], [5.0]]),
    )

    selected = select_tensor_shortlist(batch, top_k=2)

    assert selected.tolist() == [1, 2]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_diffusion_contracts.py -q
```

Expected: FAIL because concretization modules do not exist.

- [ ] **Step 3: Implement concretization**

Create `src/floorset_arch/diffusion/concretize.py`:

```python
from __future__ import annotations

import math

import torch

from floorset_arch.diffusion.contracts import DiffusionPlacementPrior, PlacementTensorBatch
from floorset_arch.models import Instance, Placement, Rect


def concretize_diffusion_prior(inst: Instance, prior: DiffusionPlacementPrior) -> PlacementTensorBatch:
    centers = torch.nan_to_num(prior.centers.float(), nan=0.0, posinf=0.0, neginf=0.0)
    log_aspect = torch.nan_to_num(prior.log_aspect.float(), nan=0.0).clamp(-2.5, 2.5)
    area = inst.area_targets[: inst.block_count].float().to(centers.device).clamp_min(1.0)
    aspect = torch.exp(log_aspect)
    width = torch.sqrt(area.view(1, -1) * aspect).clamp_min(1.0)
    height = torch.sqrt(area.view(1, -1) / aspect.clamp_min(1e-6)).clamp_min(1.0)
    x = (centers[:, :, 0] - width * 0.5).clamp_min(0.0)
    y = (centers[:, :, 1] - height * 0.5).clamp_min(0.0)
    rects = torch.stack([x, y, width, height], dim=2)

    for block, target in inst.target_rects.items():
        if block in inst.fixed or block in inst.preplaced:
            rects[:, block, 2] = float(target.width)
            rects[:, block, 3] = float(target.height)
        if block in inst.preplaced:
            rects[:, block, 0] = float(target.x)
            rects[:, block, 1] = float(target.y)

    quality = prior.quality_pred
    if quality is None:
        score_features = torch.zeros((prior.sample_count, 1), dtype=rects.dtype, device=rects.device)
    else:
        score_features = quality.view(-1, 1).to(rects.device)
    return PlacementTensorBatch(
        rect_xywh=rects,
        pairwise_axis_logits=prior.pairwise_axis_logits,
        pair_index=prior.pair_index,
        score_features=score_features,
        source=f"diffusion:{prior.variant}",
    )


def placement_from_tensor_candidate(inst: Instance, batch: PlacementTensorBatch, index: int) -> Placement:
    rects: dict[int, Rect] = {}
    sample = batch.rect_xywh[index].detach().cpu()
    for block in range(inst.block_count):
        x, y, width, height = [float(value) for value in sample[block].tolist()]
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            x, y, width, height = 0.0, 0.0, 1.0, 1.0
        rects[block] = Rect(max(0.0, x), max(0.0, y), max(1.0, width), max(1.0, height))
    return Placement(rects)
```

- [ ] **Step 4: Implement tensor shortlist ranking**

Create `src/floorset_arch/diffusion/ranking.py`:

```python
from __future__ import annotations

import torch

from floorset_arch.diffusion.contracts import PlacementTensorBatch
from floorset_arch.diagnostics import placement_metrics
from floorset_arch.models import Instance, Placement
from floorset_arch.v10_proxy import v10_proxy_rank


def _overlap_proxy(rect_xywh: torch.Tensor) -> torch.Tensor:
    x1 = rect_xywh[:, :, 0]
    y1 = rect_xywh[:, :, 1]
    x2 = x1 + rect_xywh[:, :, 2].clamp_min(1.0)
    y2 = y1 + rect_xywh[:, :, 3].clamp_min(1.0)
    score = torch.zeros(rect_xywh.shape[0], dtype=rect_xywh.dtype, device=rect_xywh.device)
    for i in range(rect_xywh.shape[1]):
        for j in range(i + 1, rect_xywh.shape[1]):
            ix = (torch.minimum(x2[:, i], x2[:, j]) - torch.maximum(x1[:, i], x1[:, j])).clamp_min(0.0)
            iy = (torch.minimum(y2[:, i], y2[:, j]) - torch.maximum(y1[:, i], y1[:, j])).clamp_min(0.0)
            score = score + ix * iy
    return score


def tensor_prefilter_score(batch: PlacementTensorBatch) -> torch.Tensor:
    overlap = _overlap_proxy(batch.rect_xywh)
    bbox_right = batch.rect_xywh[:, :, 0] + batch.rect_xywh[:, :, 2]
    bbox_top = batch.rect_xywh[:, :, 1] + batch.rect_xywh[:, :, 3]
    bbox = bbox_right.max(dim=1).values * bbox_top.max(dim=1).values
    model_score = torch.zeros_like(overlap)
    if batch.score_features is not None and batch.score_features.numel() > 0:
        model_score = batch.score_features[:, 0].to(overlap.device)
    return overlap * 1000.0 + bbox * 0.001 + model_score


def select_tensor_shortlist(batch: PlacementTensorBatch, top_k: int) -> torch.Tensor:
    score = tensor_prefilter_score(batch)
    k = max(1, min(int(top_k), int(score.numel())))
    return torch.argsort(score)[:k]


def rank_repaired_placement(inst: Instance, placement: Placement) -> tuple:
    metrics = placement_metrics(inst, placement)
    return v10_proxy_rank(inst, placement, metrics)
```

Update `src/floorset_arch/diffusion/__init__.py` exports for `concretize_diffusion_prior`, `placement_from_tensor_candidate`, `select_tensor_shortlist`, and `rank_repaired_placement`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/test_diffusion_contracts.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/diffusion tests/test_diffusion_contracts.py
git commit -m "feat: add diffusion concretization and ranking"
```

---

### Task 6: Connect Diffusion Inference Into ArchitectureV11Optimizer

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Test: `tests/test_architecture_v11.py`

- [ ] **Step 1: Extend failing optimizer tests**

Append to `tests/test_architecture_v11.py`:

```python
import torch

from floorset_arch.diffusion.contracts import DiffusionPlacementPrior
from floorset_arch.models import Placement, Rect


def _tiny_problem():
    return {
        "block_count": 2,
        "area_targets": torch.tensor([4.0, 9.0]),
        "b2b_connectivity": torch.tensor([[0.0, 1.0, 1.0]]),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(2, 5),
        "target_positions": torch.full((2, 4), -1.0),
    }


def test_v11_falls_back_when_diffusion_checkpoint_is_missing(monkeypatch):
    monkeypatch.delenv("FLOORSET_DIFFUSION_CHECKPOINT", raising=False)
    optimizer = ArchitectureV11Optimizer()

    positions = optimizer.solve(**_tiny_problem())

    assert len(positions) == 2
    assert optimizer.last_solve_metadata["fallback"] == "v5_no_diffusion_checkpoint"


def test_v11_uses_diffusion_prior_without_anchor_guidance(monkeypatch):
    optimizer = ArchitectureV11Optimizer()

    def fake_prior(inst):
        return DiffusionPlacementPrior(
            centers=torch.tensor([[[1.0, 1.0], [5.0, 5.0]]]),
            log_aspect=torch.zeros(1, 2),
            pairwise_axis_logits=torch.zeros(1, 1, 3),
            pair_index=torch.tensor([[0, 1]]),
            variant="unit",
        )

    def fail_anchor(_inst):
        raise AssertionError("diffusion path must not request AnchorGuidance")

    monkeypatch.setattr(optimizer, "_try_diffusion_prior", fake_prior)
    monkeypatch.setattr(optimizer, "_try_anchor_guidance", fail_anchor)

    positions = optimizer.solve(**_tiny_problem())

    assert len(positions) == 2
    assert optimizer.last_solve_metadata["diffusion_variant"] == "unit"
    assert optimizer.last_solve_metadata["fallback"] == ""
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_architecture_v11.py -q
```

Expected: FAIL because `last_solve_metadata` and `_try_diffusion_prior()` are not implemented.

- [ ] **Step 3: Split legacy v5 solving into a helper**

In `src/floorset_arch/optimizer.py`, add imports near the other imports:

```python
from floorset_arch.diffusion.concretize import (
    concretize_diffusion_prior,
    placement_from_tensor_candidate,
)
from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.diffusion.ranking import rank_repaired_placement, select_tensor_shortlist
from floorset_arch.diffusion.sampling import load_diffusion_checkpoint, sample_diffusion_prior
```

Inside `ArchitectureV11Optimizer.__init__()`, add:

```python
self._diffusion_checkpoint_key: Optional[Path] = None
self._diffusion_model = None
self.last_solve_metadata: dict[str, str] = {}
```

Replace `solve()` with:

```python
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
    self.last_solve_metadata = {"fallback": "", "diffusion_variant": ""}
    prior = self._try_diffusion_prior(inst)
    if prior is not None:
        best = self._solve_from_diffusion_prior(inst, prior)
        self.last_solve_metadata["diffusion_variant"] = prior.variant
        return best.to_position_list(block_count)
    self.last_solve_metadata["fallback"] = "v5_no_diffusion_checkpoint"
    best = self._solve_v5_instance(inst)
    return best.to_position_list(block_count)
```

Add:

```python
def _solve_v5_instance(self, inst) -> Placement:
    inst.anchor_guidance = self._try_anchor_guidance(inst)
    if inst.anchor_guidance is None and self._uses_surrogate_guidance():
        inst.anchor_guidance = build_surrogate_guidance(inst)
    candidates = self._build_candidates(inst, self._candidate_specs(inst))
    return self._select_best_candidate(inst, candidates)
```

- [ ] **Step 4: Add diffusion checkpoint and candidate path**

Add methods to `ArchitectureV11Optimizer`:

```python
def _resolve_diffusion_checkpoint_path(self) -> Optional[Path]:
    value = os.environ.get("FLOORSET_DIFFUSION_CHECKPOINT", "").strip()
    if not value:
        return None
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw if raw.exists() else None
    for candidate in (ROOT / raw, ROOT / "checkpoints" / raw.name):
        if candidate.exists():
            return candidate.resolve()
    return None


def _diffusion_variant(self) -> str:
    variant = os.environ.get("FLOORSET_DIFFUSION_VARIANT", "hgt_lite").strip().lower()
    return variant if variant in {"raw", "hgt_lite"} else "hgt_lite"


def _try_diffusion_prior(self, inst):
    checkpoint = self._resolve_diffusion_checkpoint_path()
    if checkpoint is None:
        return None
    try:
        graph_inputs = build_diffusion_graph_inputs(inst, device=torch.device("cpu"))
        if self._diffusion_checkpoint_key != checkpoint or self._diffusion_model is None:
            model = load_diffusion_checkpoint(checkpoint, graph_inputs, map_location="cpu")
            self._diffusion_model = model
            self._diffusion_checkpoint_key = checkpoint
        samples = int(os.environ.get("FLOORSET_DIFFUSION_SAMPLES", "8"))
        steps = int(os.environ.get("FLOORSET_DIFFUSION_STEPS", "16"))
        seed = int(os.environ.get("FLOORSET_DIFFUSION_SEED", "0"))
        prior = sample_diffusion_prior(self._diffusion_model, graph_inputs, samples=samples, steps=steps, seed=seed)
        return prior
    except Exception as exc:
        self.last_solve_metadata["fallback"] = f"v5_diffusion_error:{exc.__class__.__name__}"
        return None


def _solve_from_diffusion_prior(self, inst, prior) -> Placement:
    batch = concretize_diffusion_prior(inst, prior)
    top_k = int(os.environ.get("FLOORSET_DIFFUSION_TOPK", "4"))
    selected = select_tensor_shortlist(batch, top_k=top_k)
    candidates: list[Placement] = []
    for idx in selected.detach().cpu().tolist():
        placement = placement_from_tensor_candidate(inst, batch, int(idx))
        repaired = self._repair_with_profile(inst, placement, "normal")
        repaired = refine_quality_candidate(inst, repaired, self.config, "default")
        candidates.append(repaired)
    if not candidates:
        return self._solve_v5_instance(inst)
    return min(candidates, key=lambda placement: rank_repaired_placement(inst, placement))
```

In `solve()`, when `_try_diffusion_prior()` returns `None`, preserve a diffusion error fallback reason:

```python
if not self.last_solve_metadata.get("fallback"):
    self.last_solve_metadata["fallback"] = "v5_no_diffusion_checkpoint"
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/test_architecture_v11.py tests/test_optimizer.py tests/test_diffusion_model.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/optimizer.py tests/test_architecture_v11.py
git commit -m "feat: connect diffusion path to v11 optimizer"
```

---

### Task 7: Add Diffusion Training Step API

**Files:**
- Create: `src/floorset_arch/diffusion/training.py`
- Modify: `src/floorset_arch/diffusion/__init__.py`
- Test: `tests/test_diffusion_targets.py`

- [ ] **Step 1: Add failing tiny training-step test**

Append to `tests/test_diffusion_targets.py`:

```python
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.training import diffusion_training_loss


def test_diffusion_training_loss_uses_tree_and_quality_targets():
    inst = _inst()
    graph_inputs = build_diffusion_graph_inputs(inst)
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0, 4.0],
        ]
    )
    tree_sol = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics_sol = torch.tensor([25.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 12.0])
    targets = build_diffusion_targets(inst, graph_inputs, fp_sol, tree_sol, metrics_sol)
    model = GraphConditionedPlacementDiffusion.from_graph_inputs(graph_inputs, variant="raw", hidden_dim=16, layers=1)

    loss, parts = diffusion_training_loss(model, graph_inputs, targets, seed=3)

    assert loss.requires_grad
    assert parts["tree"] >= 0.0
    assert parts["quality"] >= 0.0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run pytest tests/test_diffusion_targets.py::test_diffusion_training_loss_uses_tree_and_quality_targets -q
```

Expected: FAIL because `diffusion_training_loss` does not exist.

- [ ] **Step 3: Implement diffusion training loss**

Create `src/floorset_arch/diffusion/training.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from floorset_arch.diffusion.contracts import DiffusionGraphInputs
from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion
from floorset_arch.diffusion.targets import DiffusionTargets


def diffusion_training_loss(
    model: GraphConditionedPlacementDiffusion,
    graph: DiffusionGraphInputs,
    targets: DiffusionTargets,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    generator = torch.Generator(device=graph.area.device)
    generator.manual_seed(int(seed))
    clean = torch.cat([targets.center, targets.log_aspect.view(-1, 1)], dim=1).view(1, graph.area.shape[0], 3)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    timestep = torch.tensor([1], dtype=torch.long, device=clean.device)
    noisy = clean + 0.1 * noise
    pred = model(graph, noisy, timestep)
    denoise = F.mse_loss(pred["eps_pred"], noise)
    pair_loss = F.cross_entropy(
        pred["pairwise_axis_logits"][0][targets.pair_axis_mask],
        targets.pair_axis_label[targets.pair_axis_mask],
    )
    if targets.tree_pair_mask.any():
        tree_logits = pred["pairwise_axis_logits"][0][targets.tree_pair_mask, :2]
        tree_loss = F.cross_entropy(tree_logits, targets.tree_side_label[targets.tree_pair_mask])
    else:
        tree_loss = denoise * 0.0
    quality_target = targets.quality_labels.float().mean().view(1).to(pred["quality_pred"].device)
    quality_loss = F.mse_loss(pred["quality_pred"].view(1), quality_target)
    loss = denoise + 0.25 * pair_loss + 0.25 * tree_loss + 0.01 * quality_loss
    return loss, {
        "denoise": float(denoise.detach().cpu().item()),
        "pair": float(pair_loss.detach().cpu().item()),
        "tree": float(tree_loss.detach().cpu().item()),
        "quality": float(quality_loss.detach().cpu().item()),
    }
```

Update `src/floorset_arch/diffusion/__init__.py` to export `diffusion_training_loss`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run pytest tests/test_diffusion_targets.py tests/test_diffusion_model.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/diffusion/training.py src/floorset_arch/diffusion/__init__.py tests/test_diffusion_targets.py
git commit -m "feat: add diffusion training loss"
```

---

### Task 8: Save Predicted Floorplan PNGs During Evaluation

**Files:**
- Modify: `scripts/iccad2026_evaluate.py`
- Modify: `scripts/eval_single.sh`
- Modify: `scripts/eval_total.sh`
- Test: `tests/test_eval_visualization.py`

- [ ] **Step 1: Write failing visualization tests**

Create `tests/test_eval_visualization.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.iccad2026_evaluate import TestResult, save_predicted_floorplan_pngs


def test_save_predicted_floorplan_pngs_writes_single_case(tmp_path):
    pytest.importorskip("matplotlib")
    result = SimpleNamespace(
        submission_name="unit",
        test_results=[
            TestResult(
                test_id=7,
                block_count=2,
                is_feasible=True,
                hpwl_gap=0.0,
                area_gap=0.0,
                violations_relative=0.0,
                runtime_seconds=0.1,
                cost=3.25,
                positions=[(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 3.0, 3.0)],
            )
        ],
    )

    written = save_predicted_floorplan_pngs(result, tmp_path, top_k=10)

    assert len(written) == 1
    assert written[0].name == "case_7_cost_3.2500.png"
    assert written[0].exists()


def test_save_predicted_floorplan_pngs_writes_top_10_by_cost(tmp_path):
    pytest.importorskip("matplotlib")
    results = []
    for idx in range(12):
        results.append(
            TestResult(
                test_id=idx,
                block_count=1,
                is_feasible=True,
                hpwl_gap=0.0,
                area_gap=0.0,
                violations_relative=0.0,
                runtime_seconds=0.1,
                cost=float(idx),
                positions=[(float(idx), 0.0, 1.0, 1.0)],
            )
        )
    result = SimpleNamespace(submission_name="unit", test_results=results)

    written = save_predicted_floorplan_pngs(result, tmp_path, top_k=10)

    assert len(written) == 10
    assert written[0].name.startswith("top_cost_rank_01_case_11_cost_11.0000")
    assert written[-1].name.startswith("top_cost_rank_10_case_2_cost_2.0000")
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_eval_visualization.py -q
```

Expected: FAIL because `save_predicted_floorplan_pngs` does not exist.

- [ ] **Step 3: Add PNG helper functions**

In `scripts/iccad2026_evaluate.py`, near the visualization section, add:

```python
def _plot_positions(ax, positions, title: str):
    import matplotlib.patches as mpatches

    ax.set_title(title)
    if not positions:
        ax.text(0.5, 0.5, "no positions", ha="center", va="center", transform=ax.transAxes)
        return
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(positions), 1)))
    for i, (x, y, w, h) in enumerate(positions):
        rect = mpatches.Rectangle((x, y), w, h, fill=True, facecolor=colors[i % len(colors)], edgecolor="black", alpha=0.75)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=7)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def save_predicted_floorplan_pngs(result, output_dir: str | Path, top_k: int = 10) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("matplotlib required for floorplan PNG output")
        return []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solved = [r for r in result.test_results if getattr(r, "positions", None)]
    if not solved:
        return []
    single_case = len(solved) == 1
    ranked = sorted(solved, key=lambda row: float(row.cost), reverse=True)
    selected = ranked[: max(1, min(int(top_k), len(ranked)))]
    written: list[Path] = []
    for rank, row in enumerate(selected, start=1):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        _plot_positions(ax, row.positions, f"Case {row.test_id} predicted cost={row.cost:.4f}")
        if single_case:
            filename = f"case_{row.test_id}_cost_{row.cost:.4f}.png"
        else:
            filename = f"top_cost_rank_{rank:02d}_case_{row.test_id}_cost_{row.cost:.4f}.png"
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written
```

Ensure `Path` is already imported in the file. If `plt` is not module-global, move the `import matplotlib.pyplot as plt` into `save_predicted_floorplan_pngs()` and pass it to `_plot_positions`, or import inside `_plot_positions`.

- [ ] **Step 4: Add CLI flag and call helper after evaluation**

In `main()`, add an argument after `--save-solutions`:

```python
parser.add_argument('--floorplan-output-dir', default=None,
                   help='Directory for predicted floorplan PNG outputs')
```

After the result JSON is saved, add:

```python
floorplan_dir = args.floorplan_output_dir
if floorplan_dir is None:
    run_stem = Path(output).stem
    floorplan_dir = Path("artifacts") / "eval_v11" / "floorplans" / run_stem
written_pngs = save_predicted_floorplan_pngs(result, floorplan_dir, top_k=10)
if written_pngs:
    print(f"Floorplan PNGs saved to {Path(floorplan_dir)}")
```

- [ ] **Step 5: Update scripts to pass output directory**

In `scripts/eval_single.sh`, before running evaluator, define:

```bash
FLOORPLAN_DIR="${FLOORSET_EVAL_FLOORPLAN_DIR:-$ROOT/artifacts/eval_v11/floorplans/single_case_${TESTID}}"
```

Add to the evaluator command:

```bash
  --floorplan-output-dir "$FLOORPLAN_DIR" \
```

In `scripts/eval_total.sh`, inside `run_evaluator()`, define:

```bash
local floorplan_dir="${FLOORSET_EVAL_FLOORPLAN_DIR:-$ROOT/artifacts/eval_v11/floorplans/latest_total}"
```

Add to the evaluator command:

```bash
    --floorplan-output-dir "$floorplan_dir" \
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/test_eval_visualization.py tests/test_scripts.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/iccad2026_evaluate.py scripts/eval_single.sh scripts/eval_total.sh tests/test_eval_visualization.py
git commit -m "feat: save eval floorplan pngs"
```

---

### Task 9: Rename W&B Training Defaults For V11 Diffusion

**Files:**
- Modify: `src/floorset_arch/training/train.py`
- Modify: `scripts/train.sh`
- Modify: `scripts/train_transformer.sh`
- Modify: `scripts/train_hgt.sh`
- Modify: `README.md`
- Test: `tests/test_scripts.py`

- [ ] **Step 1: Add failing W&B default tests**

Append to `tests/test_scripts.py`:

```python
from pathlib import Path


def test_wandb_defaults_use_v11_diffusion_project_name():
    expected = "floorset-v11-diffusion"
    paths = [
        Path("scripts/train.sh"),
        Path("scripts/train_transformer.sh"),
        Path("scripts/train_hgt.sh"),
        Path("src/floorset_arch/training/train.py"),
    ]
    for path in paths:
        assert expected in path.read_text(encoding="utf-8")
        assert "floorset-arch-v5" not in path.read_text(encoding="utf-8")


def test_readme_wandb_examples_reference_v11_diffusion_project():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "floorset-v11-diffusion" in text
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_scripts.py::test_wandb_defaults_use_v11_diffusion_project_name tests/test_scripts.py::test_readme_wandb_examples_reference_v11_diffusion_project -q
```

Expected: FAIL because current defaults still reference `floorset-arch-v5` and README does not mention `floorset-v11-diffusion`.

- [ ] **Step 3: Update Python training CLI default**

In `src/floorset_arch/training/train.py`, change:

```python
parser.add_argument("--wandb-project", default="floorset-arch-v5")
```

to:

```python
parser.add_argument("--wandb-project", default="floorset-v11-diffusion")
```

Update the startup banner string from:

```python
print("Architecture v5 Anchor-GNN training")
```

to:

```python
print("Architecture v11 diffusion-ready training")
```

- [ ] **Step 4: Update shell training wrappers**

In `scripts/train.sh`, `scripts/train_transformer.sh`, and `scripts/train_hgt.sh`, change:

```bash
WANDB_PROJECT="${WANDB_PROJECT:-floorset-arch-v5}"
```

to:

```bash
WANDB_PROJECT="${WANDB_PROJECT:-floorset-v11-diffusion}"
```

If any default `CHECKPOINT_PREFIX` or `DEFAULT_TAG` still embeds `v5`, update it to use `v11` or `diffusion_ready` without changing script arguments. Preserve existing environment variable override behavior.

- [ ] **Step 5: Update README training examples**

In `README.md`, add or update the training configuration section so it includes:

```bash
WANDB_PROJECT=floorset-v11-diffusion
```

Keep existing `WANDB=0` CPU smoke-test examples intact, but mention that online runs default to the v11 diffusion W&B project unless overridden.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/test_scripts.py -q
```

Expected: PASS.

Commit:

```bash
git add src/floorset_arch/training/train.py scripts/train.sh scripts/train_transformer.sh scripts/train_hgt.sh README.md tests/test_scripts.py
git commit -m "chore: rename wandb project for v11 diffusion"
```

---

### Task 10: Add End-To-End Validation Commands And Refresh Graphify

**Files:**
- Modify only files required by fixes from verification failures.

- [ ] **Step 1: Run focused diffusion tests**

Run:

```bash
uv run pytest tests/test_diffusion_contracts.py tests/test_diffusion_graph_inputs.py tests/test_diffusion_targets.py tests/test_diffusion_model.py tests/test_architecture_v11.py tests/test_eval_visualization.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full pytest**

Run:

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 3: Run submission validation**

Run:

```bash
bash scripts/validate.sh
```

Expected: PASS and output identifies `ArchitectureV11Optimizer`.

- [ ] **Step 4: Run single-case eval and verify PNG output**

Run:

```bash
bash scripts/eval_single.sh 95 --output artifacts/eval_v11/single_95.json
```

Expected:

- evaluator writes `artifacts/eval_v11/single_95.json`
- one PNG exists under `artifacts/eval_v11/floorplans/single_case_95/`
- if no diffusion checkpoint exists, result metadata or trace shows `v5_no_diffusion_checkpoint`

- [ ] **Step 5: Run graphify update**

Run:

```bash
graphify update .
```

Expected: graphify completes without blocking implementation. Dirty `graphify-out/` files are acceptable evidence outputs.

- [ ] **Step 6: Commit verification fixes**

If verification required code or test fixes, commit them:

```bash
git add src scripts tests graphify-out
git commit -m "test: verify v11 diffusion milestone"
```

If verification passed without changes, do not create an empty commit.

---

## Self-Review

Spec coverage:

- v11 naming and v5 compatibility are covered by Task 1.
- `parse_instance() -> Instance -> build_diffusion_graph_inputs()` is covered by Task 2.
- A/B diffusion variants are covered by Task 4.
- `tree_sol` and `metrics_sol` training usage is covered by Task 3 and Task 7.
- Direct diffusion decoder, repair, and ranking are covered by Task 5 and Task 6.
- Missing-checkpoint fallback is covered by Task 6.
- Full eval top-10 PNG and single-case PNG output are covered by Task 8.
- W&B v11 diffusion naming is covered by Task 9.
- Final verification and `graphify update .` are covered by Task 10.

Placeholder scan:

- The plan uses no unfinished-marker patterns or incomplete placeholder sections.
- Code snippets define the functions/classes they later reference.

Type consistency:

- `DiffusionGraphInputs`, `DiffusionPlacementPrior`, and `PlacementTensorBatch` are defined before use by graph inputs, model, sampling, concretization, and optimizer tasks.
- The optimizer task uses `ArchitectureV11Optimizer` after Task 1 defines the class and aliases.
- The training target task defines `DiffusionTargets` before the diffusion training loss task consumes it.
