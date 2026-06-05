# HGT Ablation, Pseudo Targets, And Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-wave HGT improvement infrastructure: guidance ablations, evaluator-based checkpoint selection, dirty-sample repaired pseudo targets, and HGT relation gates.

**Architecture:** Keep the Production Solver Path and AnchorGuidance contract intact. Add small helper modules for training pseudo targets and checkpoint metric selection, then wire them into the existing optimizer, trainer, HGT model, scripts, and tests. Teacher residuals and teacher distillation remain out of scope for this plan.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing FloorSet evaluator scripts, shell training wrappers.

---

## File Structure

- Create: `src/floorset_arch/training/pseudo_targets.py`
  - Owns clean/dirty target provenance, online repaired pseudo-target creation, and cache-compatible record shapes.
- Create: `src/floorset_arch/training/selection.py`
  - Owns checkpoint metric records, comparison ordering, and JSONL manifest append/read helpers.
- Modify: `src/floorset_arch/optimizer.py`
  - Applies guidance ablation environment knobs after model prediction and before decoder consumption.
- Modify: `src/floorset_arch/training/train.py`
  - Uses pseudo-target records for target creation and dirty order/pairwise weighting.
  - Saves `best_val_loss` separately from evaluator-promoted metric records.
- Modify: `src/floorset_arch/training/checkpoint.py`
  - Persists relation-gate diagnostic values when available.
- Modify: `src/floorset_arch/nn/model.py`
  - Adds relation gates to `HGTLayer`.
- Modify: `scripts/train_hgt.sh`
  - Exposes conservative pseudo-target and checkpoint metric knobs.
- Modify: `README.md`
  - Documents HGT ablation, pseudo-target policy, relation gates, and promoted-checkpoint policy.
- Test: `tests/test_optimizer.py`
  - Guidance ablation behavior.
- Test: `tests/test_model.py`
  - Pseudo-target builder, dirty loss weighting, relation gates, checkpoint payload metadata.
- Test: `tests/test_scripts.py`
  - HGT script knobs.

## Task 1: Guidance Ablation Knobs

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for pairwise/aspect/priority ablation**

Add to `tests/test_optimizer.py`:

```python
def test_guidance_ablation_can_disable_pairwise_aspect_and_priority(monkeypatch):
    optimizer = ArchitectureV4Optimizer()
    inst = parse_instance(**_tiny_problem())
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    pred = {
        "anchor": torch.tensor([[0.5, 0.5], [2.5, 0.5]], dtype=torch.float32),
        "priority": torch.tensor([0.75, 0.25], dtype=torch.float32),
        "log_aspect": torch.tensor([0.4, -0.4], dtype=torch.float32),
        "pair_logits": torch.tensor([[2.0, -1.0]], dtype=torch.float32),
    }

    monkeypatch.setenv("FLOORSET_GUIDANCE_DISABLE_PAIRWISE", "1")
    monkeypatch.setenv("FLOORSET_GUIDANCE_DISABLE_ASPECT", "1")
    monkeypatch.setenv("FLOORSET_GUIDANCE_DISABLE_PRIORITY", "1")

    guidance = optimizer._anchor_predictions_to_guidance(inst, pred, scale=1.0, pairs=pairs)

    assert guidance.rect_priors
    assert guidance.pairwise_axis == {}
    assert guidance.log_aspect == {}
    assert guidance.priority == {}
```

- [ ] **Step 2: Write failing test for anchor-only ablation**

Add to `tests/test_optimizer.py`:

```python
def test_guidance_anchor_only_keeps_rect_priors(monkeypatch):
    optimizer = ArchitectureV4Optimizer()
    inst = parse_instance(**_tiny_problem())
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    pred = {
        "anchor": torch.tensor([[0.5, 0.5], [2.5, 0.5]], dtype=torch.float32),
        "priority": torch.tensor([0.75, 0.25], dtype=torch.float32),
        "log_aspect": torch.tensor([0.4, -0.4], dtype=torch.float32),
        "pair_logits": torch.tensor([[2.0, -1.0]], dtype=torch.float32),
    }

    monkeypatch.setenv("FLOORSET_GUIDANCE_ANCHOR_ONLY", "1")

    guidance = optimizer._anchor_predictions_to_guidance(inst, pred, scale=1.0, pairs=pairs)

    assert sorted(guidance.rect_priors) == [0, 1]
    assert guidance.pairwise_axis == {}
    assert guidance.log_aspect == {}
    assert guidance.priority == {}
```

- [ ] **Step 3: Run red tests**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_guidance_ablation_can_disable_pairwise_aspect_and_priority tests/test_optimizer.py::test_guidance_anchor_only_keeps_rect_priors -q
```

Expected: FAIL because `_anchor_predictions_to_guidance()` does not read the new environment knobs.

- [ ] **Step 4: Implement ablation helper in optimizer**

Add near `_anchor_predictions_to_guidance()` in `src/floorset_arch/optimizer.py`:

```python
def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}
```

At the start of `_anchor_predictions_to_guidance()`, add:

```python
anchor_only = _env_flag("FLOORSET_GUIDANCE_ANCHOR_ONLY")
disable_pairwise = anchor_only or _env_flag("FLOORSET_GUIDANCE_DISABLE_PAIRWISE")
disable_aspect = anchor_only or _env_flag("FLOORSET_GUIDANCE_DISABLE_ASPECT")
disable_priority = anchor_only or _env_flag("FLOORSET_GUIDANCE_DISABLE_PRIORITY")
```

Then guard aspect, priority, and pairwise storage by moving the existing aspect block under the first condition and moving the existing pairwise block under the third condition:

```python
if (
    not disable_aspect
    and aspect_tensor is not None
    and i not in inst.fixed
    and i not in inst.preplaced
):
    log_aspect = float(aspect_tensor.detach().cpu()[i])
    guidance.log_aspect[i] = log_aspect
    aspect = math.exp(max(-2.5, min(2.5, log_aspect)))
    area = max(1.0, float(inst.area_targets[i]))
    width = math.sqrt(area * aspect)
    height = math.sqrt(area / aspect)

if not disable_priority and priority_tensor is not None:
    guidance.priority[i] = float(priority_tensor.detach().cpu()[i])

if not disable_pairwise and pair_logits is not None and pairs is not None:
    logits = pair_logits.detach().cpu()
    for pair, logit in zip(pairs.detach().cpu().tolist(), logits.tolist()):
        guidance.pairwise_axis[(int(pair[0]), int(pair[1]))] = (
            float(logit[0]),
            float(logit[1]),
        )
```

- [ ] **Step 5: Run green tests**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_guidance_ablation_can_disable_pairwise_aspect_and_priority tests/test_optimizer.py::test_guidance_anchor_only_keeps_rect_priors -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/floorset_arch/optimizer.py tests/test_optimizer.py
git commit -m "feat: add HGT guidance ablation knobs"
```

## Task 2: Evaluator-Based Checkpoint Metric Selection

**Files:**
- Create: `src/floorset_arch/training/selection.py`
- Modify: `src/floorset_arch/training/train.py`
- Modify: `src/floorset_arch/training/checkpoint.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test for metric ordering**

Add imports to `tests/test_model.py`:

```python
from floorset_arch.training.selection import CheckpointMetricRecord, better_checkpoint_metric
```

Add test:

```python
def test_checkpoint_metric_prefers_no_runtime_over_val_loss():
    low_val_loss_bad_eval = CheckpointMetricRecord(
        checkpoint="bad_eval.pt",
        epoch=3,
        metric_source="tail_eval",
        feasible=100,
        val_loss=0.001,
        total_score_no_runtime=2.70,
        tail_weighted_no_runtime=2.70,
        soft_violations=10,
        avg_runtime=1.0,
    )
    higher_val_loss_good_eval = CheckpointMetricRecord(
        checkpoint="good_eval.pt",
        epoch=2,
        metric_source="tail_eval",
        feasible=100,
        val_loss=0.010,
        total_score_no_runtime=2.05,
        tail_weighted_no_runtime=2.05,
        soft_violations=4,
        avg_runtime=1.2,
    )

    assert better_checkpoint_metric(higher_val_loss_good_eval, low_val_loss_bad_eval)
    assert not better_checkpoint_metric(low_val_loss_bad_eval, higher_val_loss_good_eval)
```

- [ ] **Step 2: Write failing JSONL manifest test**

Add to `tests/test_model.py`:

```python
def test_checkpoint_metric_manifest_round_trips(tmp_path):
    from floorset_arch.training.selection import append_metric_record, read_metric_records

    manifest = tmp_path / "checkpoint_metrics.jsonl"
    record = CheckpointMetricRecord(
        checkpoint="ckpt.pt",
        epoch=4,
        metric_source="full_eval",
        feasible=100,
        val_loss=0.004,
        total_score_no_runtime=2.01,
        tail_weighted_no_runtime=2.03,
        soft_violations=5,
        avg_runtime=1.5,
    )

    append_metric_record(manifest, record)

    assert read_metric_records(manifest) == [record]
```

- [ ] **Step 3: Run red tests**

Run:

```bash
uv run pytest tests/test_model.py::test_checkpoint_metric_prefers_no_runtime_over_val_loss tests/test_model.py::test_checkpoint_metric_manifest_round_trips -q
```

Expected: FAIL because `floorset_arch.training.selection` does not exist.

- [ ] **Step 4: Implement selection helper**

Create `src/floorset_arch/training/selection.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointMetricRecord:
    checkpoint: str
    epoch: int
    metric_source: str
    feasible: int
    val_loss: float | None = None
    total_score_no_runtime: float | None = None
    tail_weighted_no_runtime: float | None = None
    soft_violations: int | None = None
    avg_runtime: float | None = None


def _score_value(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def _violation_value(value: int | None) -> int:
    return 10**9 if value is None else int(value)


def metric_sort_key(record: CheckpointMetricRecord) -> tuple:
    return (
        -int(record.feasible),
        _score_value(record.total_score_no_runtime),
        _score_value(record.tail_weighted_no_runtime),
        _violation_value(record.soft_violations),
        _score_value(record.avg_runtime),
        _score_value(record.val_loss),
        int(record.epoch),
    )


def better_checkpoint_metric(
    candidate: CheckpointMetricRecord, current: CheckpointMetricRecord | None
) -> bool:
    if current is None:
        return True
    return metric_sort_key(candidate) < metric_sort_key(current)


def append_metric_record(path: str | Path, record: CheckpointMetricRecord) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_metric_records(path: str | Path) -> list[CheckpointMetricRecord]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(CheckpointMetricRecord(**json.loads(line)))
    return records
```

- [ ] **Step 5: Rename validation-loss checkpoint path in trainer**

In `src/floorset_arch/training/train.py`, replace:

```python
best_path = out_dir / f"{args.checkpoint_prefix}_best_{run_tag}.pt"
```

with:

```python
best_val_loss_path = out_dir / f"{args.checkpoint_prefix}_best_val_loss_{run_tag}.pt"
```

Then update all `best_path` uses in the validation-loss save branch to `best_val_loss_path`. Keep `latest_path` unchanged. The final prints should become:

```python
print(f"Best val loss: {best_val:.5f}")
print(f"Best val-loss checkpoint: {best_val_loss_path}")
```

Keep stable checkpoint writes untouched unless `args.write_stable_checkpoints` is set.

- [ ] **Step 6: Persist best-val-loss naming in checkpoint payload**

In `src/floorset_arch/training/checkpoint.py`, add a payload field:

```python
"selection_metric": "val_loss",
```

inside `anchor_checkpoint_payload()`. This records that the training-loop best checkpoint is a health metric, not an evaluator-promoted checkpoint.

- [ ] **Step 7: Run green tests**

Run:

```bash
uv run pytest tests/test_model.py::test_checkpoint_metric_prefers_no_runtime_over_val_loss tests/test_model.py::test_checkpoint_metric_manifest_round_trips -q
```

Expected: PASS.

- [ ] **Step 8: Run checkpoint-focused tests**

Run:

```bash
uv run pytest tests/test_model.py::test_anchor_checkpoint_payload_records_hgt_config tests/test_model.py::test_hgt_run_tag_records_batch_size -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/floorset_arch/training/selection.py src/floorset_arch/training/train.py src/floorset_arch/training/checkpoint.py tests/test_model.py
git commit -m "feat: separate checkpoint promotion metrics from val loss"
```

## Task 3: Dirty Repaired Pseudo Targets

**Files:**
- Create: `src/floorset_arch/training/pseudo_targets.py`
- Modify: `src/floorset_arch/training/train.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing pseudo-target provenance test**

Add import to `tests/test_model.py`:

```python
from floorset_arch.training.pseudo_targets import (
    PseudoTargetConfig,
    TrainingTargetSource,
    build_training_target_record,
)
```

Add test:

```python
def test_dirty_sample_builds_repaired_pseudo_target():
    block_count = 2
    area_targets = torch.tensor([4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 8.0, 8.0],
        ]
    )
    inst = parse_instance(
        block_count,
        area_targets,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        fp_sol,
    )

    record = build_training_target_record(
        inst,
        fp_sol,
        PseudoTargetConfig(enabled=True, clean_enough_soft_violations=0),
    )

    assert record.source in {
        TrainingTargetSource.DIRTY_REPAIRED,
        TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH,
    }
    assert record.target_fp_sol.shape == (block_count, 4)
    assert record.original_soft_violations[1] > 0
```

- [ ] **Step 2: Write failing dirty loss weighting test**

Add to `tests/test_model.py`:

```python
def test_repaired_clean_enough_target_enables_dirty_order_weight():
    config = PseudoTargetConfig(
        enabled=True,
        dirty_pseudo_order_weight=0.25,
        dirty_pseudo_clean_enough_order_weight=0.35,
        clean_enough_soft_violations=0,
    )
    record = train_module._target_weight_policy(
        is_clean=False,
        target_source=TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH,
        config=config,
        existing_dirty_sample_weight=0.25,
    )

    assert record.sample_weight == 0.25
    assert record.order_weight_multiplier == 0.35
    assert record.pairwise_weight_multiplier == 0.35
```

- [ ] **Step 3: Run red tests**

Run:

```bash
uv run pytest tests/test_model.py::test_dirty_sample_builds_repaired_pseudo_target tests/test_model.py::test_repaired_clean_enough_target_enables_dirty_order_weight -q
```

Expected: FAIL because pseudo-target helpers and `_target_weight_policy()` do not exist.

- [ ] **Step 4: Implement pseudo-target helper**

Create `src/floorset_arch/training/pseudo_targets.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from floorset_arch.models import Instance, Placement, Rect, SolverConfig
from floorset_arch.repair import repair_placement, soft_violation_counts
from floorset_arch.training.losses import _placement_from_fp_sol


class TrainingTargetSource(StrEnum):
    CLEAN_FP_SOL = "clean_fp_sol"
    DIRTY_ORIGINAL = "dirty_original"
    DIRTY_REPAIRED = "dirty_repaired"
    DIRTY_REPAIRED_CLEAN_ENOUGH = "dirty_repaired_clean_enough"


@dataclass(frozen=True)
class PseudoTargetConfig:
    enabled: bool = False
    clean_enough_soft_violations: int = 0
    dirty_pseudo_order_weight: float = 0.20
    dirty_pseudo_clean_enough_order_weight: float = 0.35


@dataclass(frozen=True)
class TrainingTargetRecord:
    target_fp_sol: torch.Tensor
    source: TrainingTargetSource
    original_soft_violations: tuple[int, int, int]
    repaired_soft_violations: tuple[int, int, int] | None


def placement_to_fp_sol(placement: Placement, block_count: int) -> torch.Tensor:
    rows = []
    for block in range(block_count):
        rect = placement.rects[block]
        rows.append([rect.width, rect.height, rect.x, rect.y])
    return torch.tensor(rows, dtype=torch.float32)


def build_training_target_record(
    inst: Instance,
    fp_sol: torch.Tensor,
    config: PseudoTargetConfig,
) -> TrainingTargetRecord:
    block_count = inst.block_count
    original_placement = _placement_from_fp_sol(fp_sol, block_count)
    original_soft = soft_violation_counts(inst, original_placement)
    if sum(original_soft) == 0:
        return TrainingTargetRecord(
            target_fp_sol=fp_sol[:block_count].detach().cpu().float(),
            source=TrainingTargetSource.CLEAN_FP_SOL,
            original_soft_violations=original_soft,
            repaired_soft_violations=None,
        )
    if not config.enabled:
        return TrainingTargetRecord(
            target_fp_sol=fp_sol[:block_count].detach().cpu().float(),
            source=TrainingTargetSource.DIRTY_ORIGINAL,
            original_soft_violations=original_soft,
            repaired_soft_violations=None,
        )
    repaired = repair_placement(inst, original_placement.copy(), SolverConfig())
    repaired_soft = soft_violation_counts(inst, repaired)
    source = (
        TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH
        if sum(repaired_soft) <= config.clean_enough_soft_violations
        else TrainingTargetSource.DIRTY_REPAIRED
    )
    return TrainingTargetRecord(
        target_fp_sol=placement_to_fp_sol(repaired, block_count),
        source=source,
        original_soft_violations=original_soft,
        repaired_soft_violations=repaired_soft,
    )
```

- [ ] **Step 5: Implement target weight policy in trainer**

In `src/floorset_arch/training/train.py`, import:

```python
from dataclasses import dataclass
from floorset_arch.training.pseudo_targets import (
    PseudoTargetConfig,
    TrainingTargetSource,
    build_training_target_record,
)
```

Add near `decoder_ranking_multipliers()`:

```python
@dataclass(frozen=True)
class TargetWeightPolicy:
    sample_weight: float
    order_weight_multiplier: float
    pairwise_weight_multiplier: float


def _pseudo_target_config(args) -> PseudoTargetConfig:
    return PseudoTargetConfig(
        enabled=bool(getattr(args, "enable_repaired_pseudo_targets", False)),
        clean_enough_soft_violations=int(getattr(args, "pseudo_target_clean_enough_soft", 0)),
        dirty_pseudo_order_weight=float(getattr(args, "dirty_pseudo_order_weight", 0.20)),
        dirty_pseudo_clean_enough_order_weight=float(
            getattr(args, "dirty_pseudo_clean_enough_order_weight", 0.35)
        ),
    )


def _target_weight_policy(
    is_clean: bool,
    target_source: TrainingTargetSource,
    config: PseudoTargetConfig,
    existing_dirty_sample_weight: float,
) -> TargetWeightPolicy:
    if is_clean:
        return TargetWeightPolicy(1.0, 1.0, 1.0)
    if target_source == TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH:
        weight = config.dirty_pseudo_clean_enough_order_weight
        return TargetWeightPolicy(existing_dirty_sample_weight, weight, weight)
    if target_source == TrainingTargetSource.DIRTY_REPAIRED:
        weight = config.dirty_pseudo_order_weight
        return TargetWeightPolicy(existing_dirty_sample_weight, weight, weight)
    return TargetWeightPolicy(existing_dirty_sample_weight, 0.0, 0.0)
```

- [ ] **Step 6: Wire pseudo targets into `_prepare_training_sample()`**

In `_prepare_training_sample()`, immediately after the existing `inst = parse_instance` call, add:

```python
target_record = build_training_target_record(
    inst, fp_sol, _pseudo_target_config(args)
)
target_fp_sol = target_record.target_fp_sol.to(device=fp_sol.device)
```

Replace:

```python
targets = build_anchor_targets(fp_sol, block_count, scale, device)
```

with:

```python
targets = build_anchor_targets(target_fp_sol, block_count, scale, device)
```

Replace dirty weighting branch with:

```python
policy = _target_weight_policy(
    is_clean=is_clean,
    target_source=target_record.source,
    config=_pseudo_target_config(args),
    existing_dirty_sample_weight=args.dirty_sample_weight,
)
sample_weight = policy.sample_weight
order_weight_multiplier *= policy.order_weight_multiplier
pairwise_weight_multiplier *= policy.pairwise_weight_multiplier
```

Store these in the returned dict:

```python
"fp_sol": target_fp_sol,
"target_source": str(target_record.source),
"original_soft_violations": target_record.original_soft_violations,
"repaired_soft_violations": target_record.repaired_soft_violations,
```

- [ ] **Step 7: Add CLI args**

In `parse_args()` add:

```python
parser.add_argument("--enable-repaired-pseudo-targets", action="store_true")
parser.add_argument("--pseudo-target-clean-enough-soft", type=int, default=0)
parser.add_argument("--dirty-pseudo-order-weight", type=float, default=0.20)
parser.add_argument("--dirty-pseudo-clean-enough-order-weight", type=float, default=0.35)
```

- [ ] **Step 8: Run green tests**

Run:

```bash
uv run pytest tests/test_model.py::test_dirty_sample_builds_repaired_pseudo_target tests/test_model.py::test_repaired_clean_enough_target_enables_dirty_order_weight -q
```

Expected: PASS.

- [ ] **Step 9: Run training smoke**

Run:

```bash
uv run -m floorset_arch.training.train \
  --data-path FloorSet \
  --output-dir /tmp/floorset-hgt-pseudo-smoke \
  --num-samples 2 \
  --val-samples 1 \
  --epochs 1 \
  --device cpu \
  --hidden-dim 16 \
  --layers 1 \
  --encoder hgt \
  --num-heads 4 \
  --batch-size 1 \
  --enable-repaired-pseudo-targets \
  --checkpoint-prefix gnn_hgt_pseudo_smoke \
  --print-every 1
```

Expected: command exits 0 and writes a tiny HGT checkpoint under `/tmp/floorset-hgt-pseudo-smoke`.

- [ ] **Step 10: Commit Task 3**

```bash
git add src/floorset_arch/training/pseudo_targets.py src/floorset_arch/training/train.py tests/test_model.py
git commit -m "feat: train HGT with repaired dirty pseudo targets"
```

## Task 4: HGT Relation Gates

**Files:**
- Modify: `src/floorset_arch/nn/model.py`
- Modify: `src/floorset_arch/training/checkpoint.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing relation-gate tests**

Add to `tests/test_model.py`:

```python
def test_hgt_layers_create_identity_relation_gates():
    inst = _hgt_sample_instance()
    graph_inputs = features.build_anchor_hgt_graph_inputs(inst)
    model = FloorplanGNN(
        node_feat_dim=graph_inputs.node_features["block"].shape[1],
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        encoder_type="hgt",
        num_heads=4,
        hgt_node_feat_dims=graph_inputs.node_feat_dims,
        hgt_relation_specs=graph_inputs.relation_specs,
    )

    assert model.hgt_layers
    first_layer = model.hgt_layers[0]
    assert set(first_layer.rel_gate) == {
        model_module._relation_key(relation) for relation in graph_inputs.relation_specs
    }
    for parameter in first_layer.rel_gate.values():
        assert torch.allclose(parameter.detach(), torch.ones_like(parameter))
```

Add to `tests/test_model.py`:

```python
def test_checkpoint_payload_records_hgt_relation_gates():
    inst = _hgt_sample_instance()
    graph_inputs = features.build_anchor_hgt_graph_inputs(inst)
    model = FloorplanGNN(
        node_feat_dim=graph_inputs.node_features["block"].shape[1],
        hidden_dim=16,
        num_layers=1,
        encoder_type="hgt",
        num_heads=4,
        hgt_node_feat_dims=graph_inputs.node_feat_dims,
        hgt_relation_specs=graph_inputs.relation_specs,
    )
    args = type(
        "Args",
        (),
        {"encoder": "hgt", "num_heads": 4, "hidden_dim": 16, "layers": 1, "dropout": 0.0},
    )()

    payload = anchor_checkpoint_payload(model, args, 1, {}, {})

    assert payload["hgt_relation_gates"]
    assert set(payload["hgt_relation_gates"][0]) == {
        model_module._relation_key(relation) for relation in graph_inputs.relation_specs
    }
```

- [ ] **Step 2: Run red tests**

Run:

```bash
uv run pytest tests/test_model.py::test_hgt_layers_create_identity_relation_gates tests/test_model.py::test_checkpoint_payload_records_hgt_relation_gates -q
```

Expected: FAIL because `HGTLayer` has no `rel_gate` and checkpoint payload has no relation-gate metadata.

- [ ] **Step 3: Add gates to HGTLayer**

In `src/floorset_arch/nn/model.py`, inside `HGTLayer.__init__()`, add:

```python
self.rel_gate = nn.ParameterDict(
    {
        _relation_key(relation): nn.Parameter(torch.ones(1))
        for relation in self.relation_specs
    }
)
```

In `HGTLayer.forward()`, after:

```python
msg = (alpha.unsqueeze(-1) * v).reshape(-1, self.hidden_dim)
```

add:

```python
msg = msg * self.rel_gate[key].to(device=msg.device, dtype=msg.dtype)
```

- [ ] **Step 4: Add relation gate diagnostics to FloorplanGNN**

In `FloorplanGNN`, add method:

```python
def hgt_relation_gate_values(self) -> list[dict[str, float]]:
    if self.encoder_type != "hgt":
        return []
    values = []
    for layer in self.hgt_layers:
        values.append(
            {
                key: float(param.detach().cpu().item())
                for key, param in layer.rel_gate.items()
            }
        )
    return values
```

- [ ] **Step 5: Persist relation gates in checkpoint payload**

In `src/floorset_arch/training/checkpoint.py`, add to payload:

```python
"hgt_relation_gates": (
    model.hgt_relation_gate_values()
    if hasattr(model, "hgt_relation_gate_values")
    else []
),
```

- [ ] **Step 6: Run green tests**

Run:

```bash
uv run pytest tests/test_model.py::test_hgt_layers_create_identity_relation_gates tests/test_model.py::test_checkpoint_payload_records_hgt_relation_gates tests/test_model.py::test_floorplan_gnn_hgt_encoder_matches_output_contract -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/floorset_arch/nn/model.py src/floorset_arch/training/checkpoint.py tests/test_model.py
git commit -m "feat: add HGT relation gates"
```

## Task 5: Script And README Knobs

**Files:**
- Modify: `scripts/train_hgt.sh`
- Modify: `README.md`
- Test: `tests/test_scripts.py`

- [ ] **Step 1: Write failing script-default test**

Add to `tests/test_scripts.py`:

```python
def test_train_hgt_script_exposes_pseudo_target_knobs():
    script = Path("scripts/train_hgt.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENABLE_REPAIRED_PSEUDO_TARGETS="${ENABLE_REPAIRED_PSEUDO_TARGETS:-0}"' in text
    assert 'DIRTY_PSEUDO_ORDER_WEIGHT="${DIRTY_PSEUDO_ORDER_WEIGHT:-0.20}"' in text
    assert 'DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT="${DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT:-0.35}"' in text
    assert '--dirty-pseudo-order-weight "$DIRTY_PSEUDO_ORDER_WEIGHT"' in text
```

- [ ] **Step 2: Run red test**

Run:

```bash
uv run pytest tests/test_scripts.py::test_train_hgt_script_exposes_pseudo_target_knobs -q
```

Expected: FAIL because script knobs are not present.

- [ ] **Step 3: Add script environment knobs**

In `scripts/train_hgt.sh`, add default variables near the dirty-sample settings:

```bash
ENABLE_REPAIRED_PSEUDO_TARGETS="${ENABLE_REPAIRED_PSEUDO_TARGETS:-0}"
PSEUDO_TARGET_CLEAN_ENOUGH_SOFT="${PSEUDO_TARGET_CLEAN_ENOUGH_SOFT:-0}"
DIRTY_PSEUDO_ORDER_WEIGHT="${DIRTY_PSEUDO_ORDER_WEIGHT:-0.20}"
DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT="${DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT:-0.35}"
```

Before the `uv run` command, add:

```bash
PSEUDO_ARGS=()
if [ "$ENABLE_REPAIRED_PSEUDO_TARGETS" = "1" ]; then
  PSEUDO_ARGS+=(--enable-repaired-pseudo-targets)
fi
```

In the `uv run -m floorset_arch.training.train` argument list, add:

```bash
  --pseudo-target-clean-enough-soft "$PSEUDO_TARGET_CLEAN_ENOUGH_SOFT" \
  --dirty-pseudo-order-weight "$DIRTY_PSEUDO_ORDER_WEIGHT" \
  --dirty-pseudo-clean-enough-order-weight "$DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT" \
  "${PSEUDO_ARGS[@]}" \
```

- [ ] **Step 4: Update README HGT section**

In `README.md`, update the `scripts/train_hgt.sh` section with:

```markdown
HGT dirty-sample experiments can enable repaired pseudo targets:

```bash
ENABLE_REPAIRED_PSEUDO_TARGETS=1 bash scripts/train_hgt.sh
DIRTY_PSEUDO_ORDER_WEIGHT=0.20 DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT=0.35 ENABLE_REPAIRED_PSEUDO_TARGETS=1 bash scripts/train_hgt.sh
```

Guidance ablations are evaluator-time knobs:

```bash
FLOORSET_GUIDANCE_ANCHOR_ONLY=1 bash scripts/eval_total.sh checkpoints/model.pt
FLOORSET_GUIDANCE_DISABLE_PAIRWISE=1 bash scripts/eval_total.sh checkpoints/model.pt
```
```

- [ ] **Step 5: Run green tests**

Run:

```bash
uv run pytest tests/test_scripts.py::test_train_hgt_script_exposes_pseudo_target_knobs tests/test_scripts.py::test_train_hgt_script_defaults_to_hgt_encoder -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add scripts/train_hgt.sh README.md tests/test_scripts.py
git commit -m "docs: expose HGT pseudo-target and ablation knobs"
```

## Task 6: Final Verification And Ablation Smoke

**Files:**
- All changed files.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/test_model.py tests/test_optimizer.py tests/test_scripts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run CPU training smoke with pseudo targets**

Run:

```bash
DEVICE=cpu WANDB=0 NUM_SAMPLES=2 VAL_SAMPLES=1 EPOCHS=1 HIDDEN_DIM=16 LAYERS=1 BATCH_SIZE=1 ENABLE_REPAIRED_PSEUDO_TARGETS=1 bash scripts/train_hgt.sh
```

Expected: command exits 0, prints one training epoch, and writes an HGT checkpoint.

- [ ] **Step 4: Run guidance ablation smoke**

Run:

```bash
FLOORSET_GUIDANCE_ANCHOR_ONLY=1 uv run pytest tests/test_optimizer.py::test_optimizer_loads_hgt_checkpoint_for_anchor_guidance -q
```

Expected: PASS.

- [ ] **Step 5: Review git diff**

Run:

```bash
git diff --stat
git diff --check
git status --short
```

Expected: no whitespace errors. Status should show only intended source, test, script, and README changes from this implementation.

- [ ] **Step 6: Commit final verification note if docs changed during verification**

If verification requires README or docs adjustments, run:

```bash
git add README.md docs/superpowers/specs/2026-06-02-hgt-ablation-pseudo-target-selection-design.md
git commit -m "docs: record HGT first-wave verification"
```

Expected: commit created only if documentation changed.

## Self-Review

- Spec coverage: Task 1 covers guidance ablation; Task 2 covers evaluator-based selection; Task 3 covers hybrid online pseudo targets and cache-compatible record shape; Task 4 covers relation gates; Task 5 covers user-facing knobs and docs; Task 6 covers verification. Teacher residuals and distillation are excluded.
- Completeness scan: every task has exact files, test names, commands, and code snippets.
- Type consistency: `PseudoTargetConfig`, `TrainingTargetSource`, `TrainingTargetRecord`, `CheckpointMetricRecord`, and `hgt_relation_gate_values()` are named consistently across tests and implementation steps.
