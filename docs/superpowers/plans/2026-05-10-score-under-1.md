# Score Under 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime-calibrated scoring with real diagnostics and learned relative-order improvements toward honest total score below 1.

**Architecture:** Keep `floorset_arch` as the Production Solver Path. Add opt-in repair tracing around candidate repair, then add explicit pairwise relation learning and blend its logits into `relative_order.py`.

**Tech Stack:** Python, PyTorch, pytest, JSONL diagnostics, existing FloorSet evaluator.

**Execution Outcome:** Runtime calibration was removed, diagnostics and pairwise-head plumbing were implemented, and local-proxy overlap repair improved honest full evaluation from `2.5256` to the `2.3262`-`2.3991` range depending on runtime noise. The `< 1` target was not reached; next work should focus on removing remaining ID 99/98 soft violations and reducing bbox/HPWL, not on runtime tricks.

---

## File Structure

- `src/floorset_arch/optimizer.py`: remove runtime calibration and wrap candidate construction/repair with optional trace recording.
- `src/floorset_arch/diagnostics.py`: new focused metrics helpers for placements and repair deltas.
- `tests/test_optimizer.py`: regression test proving runtime calibration env vars cannot trigger sleep.
- `tests/test_diagnostics.py`: tests for metric calculation and JSONL trace shape.
- `src/floorset_arch/nn/model.py`: add optional pairwise relation head.
- `src/floorset_arch/training/losses.py`: add pairwise label extraction and loss.
- `src/floorset_arch/training/train.py`: log pairwise loss/accuracy and save pair-head-compatible checkpoints.
- `src/floorset_arch/relative_order.py`: blend pairwise logits into pair orientation decisions.
- `docs/optimization-notes.md`: record removed runtime calibration and future ablation notes.

## Task 1: Remove Runtime Calibration

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `tests/test_optimizer.py`
- Modify: `docs/optimization-notes.md`

- [ ] **Step 1: Write failing test**

```python
def test_runtime_calibration_env_does_not_sleep(monkeypatch):
    block_count = 21
    problem = {
        "block_count": block_count,
        "area_targets": torch.full((block_count,), 4.0),
        "b2b_connectivity": torch.empty(0, 3),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(block_count, 5),
        "target_positions": torch.full((block_count, 4), -1.0),
    }

    def fail_sleep(_seconds):
        raise AssertionError("runtime calibration must not sleep")

    monkeypatch.setattr("time.sleep", fail_sleep)
    monkeypatch.setenv("FLOORSET_RUNTIME_CALIBRATION_SECONDS", "10")

    optimizer = ArchitectureV3Optimizer()

    assert len(optimizer.solve(**problem)) == block_count
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_optimizer.py::test_runtime_calibration_env_does_not_sleep -q`

Expected before fix: FAIL with `AssertionError: runtime calibration must not sleep`.

- [ ] **Step 3: Remove calibration implementation**

Delete `import time`, `solve_started = time.perf_counter()`, `self._calibrate_runtime(...)`, and the `_calibrate_runtime()` method from `src/floorset_arch/optimizer.py`.

- [ ] **Step 4: Update docs**

Change `docs/optimization-notes.md` so it states runtime calibration is removed and future score work must come from placement quality, repair quality, or learned ordering/ranking.

- [ ] **Step 5: Run test to verify pass**

Run: `uv run pytest tests/test_optimizer.py::test_runtime_calibration_env_does_not_sleep -q`

Expected: PASS.

## Task 2: Add Repair Diagnostics

**Files:**
- Create: `src/floorset_arch/diagnostics.py`
- Create: `tests/test_diagnostics.py`
- Modify: `src/floorset_arch/optimizer.py`

- [ ] **Step 1: Write diagnostics metric tests**

```python
from floorset_arch.diagnostics import placement_metrics, repair_delta
from floorset_arch.models import Placement, Rect
from floorset_arch.parser import parse_instance


def test_placement_metrics_counts_overlap_and_bbox():
    inst = parse_instance(
        2,
        torch.tensor([4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(2, 5),
        torch.full((2, 4), -1.0),
    )
    placement = Placement({0: Rect(0, 0, 2, 2), 1: Rect(1, 0, 2, 2)})

    metrics = placement_metrics(inst, placement)

    assert metrics["overlap_count"] == 1
    assert metrics["bbox_area"] == 6.0
```

- [ ] **Step 2: Run test to verify missing module**

Run: `uv run pytest tests/test_diagnostics.py -q`

Expected: FAIL with `ModuleNotFoundError` or import error for `floorset_arch.diagnostics`.

- [ ] **Step 3: Implement diagnostics helpers**

Create `src/floorset_arch/diagnostics.py` with:

```python
from __future__ import annotations

from floorset_arch.geometry import bbox, overlaps
from floorset_arch.models import Instance, Placement
from floorset_arch.repair import soft_violation_counts
from floorset_arch.scoring import hpwl_proxy


def overlap_count(placement: Placement) -> int:
    rects = list(placement.rects.values())
    count = 0
    for idx, rect in enumerate(rects):
        for other in rects[idx + 1:]:
            if overlaps(rect, other):
                count += 1
    return count


def placement_metrics(inst: Instance, placement: Placement) -> dict[str, float | int]:
    bounds = bbox(list(placement.rects.values()))
    boundary, grouping, mib = soft_violation_counts(inst, placement)
    return {
        "overlap_count": overlap_count(placement),
        "boundary_violations": boundary,
        "group_violations": grouping,
        "mib_violations": mib,
        "hpwl_proxy": float(hpwl_proxy(inst, placement.rects)),
        "bbox_area": float(bounds.area),
    }


def repair_delta(before: Placement, after: Placement, before_metrics: dict, after_metrics: dict) -> dict[str, float]:
    moved = 0.0
    count = 0
    for block, before_rect in before.rects.items():
        after_rect = after.rects.get(block)
        if after_rect is None:
            continue
        moved += abs(after_rect.x - before_rect.x) + abs(after_rect.y - before_rect.y)
        count += 1
    return {
        "avg_moved_manhattan": moved / max(count, 1),
        "bbox_area_delta": float(after_metrics["bbox_area"]) - float(before_metrics["bbox_area"]),
        "hpwl_proxy_delta": float(after_metrics["hpwl_proxy"]) - float(before_metrics["hpwl_proxy"]),
    }
```

- [ ] **Step 4: Run diagnostics tests**

Run: `uv run pytest tests/test_diagnostics.py -q`

Expected: PASS.

## Task 3: Add Opt-In Repair Trace JSONL

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write trace test**

```python
def test_optimizer_writes_repair_trace_jsonl(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("FLOORSET_REPAIR_TRACE_JSONL", str(trace_path))

    optimizer = ArchitectureV3Optimizer(config=SolverConfig(checkpoint_repo_relative=False, default_checkpoint="missing.pt"))
    optimizer.solve(**_tiny_problem())

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows
    assert rows[0]["candidate"]["profile"] == "soft"
    assert "before_repair" in rows[0]
    assert "after_repair" in rows[0]
    assert "delta" in rows[0]
```

- [ ] **Step 2: Run trace test to verify failure**

Run: `uv run pytest tests/test_diagnostics.py::test_optimizer_writes_repair_trace_jsonl -q`

Expected: FAIL because trace file is not created.

- [ ] **Step 3: Implement trace writing**

Add helper methods to `ArchitectureV3Optimizer`:

```python
def _repair_candidate(self, inst, placement, profile: str, kind: str) -> Placement:
    before = placement
    after = repair_placement(inst, before, self.config)
    self._trace_repair(inst, before, after, {"profile": profile, "kind": kind})
    return after
```

Use `placement_metrics()` and `repair_delta()` from `floorset_arch.diagnostics`, append one JSON object per repaired candidate when `FLOORSET_REPAIR_TRACE_JSONL` is set.

- [ ] **Step 4: Run trace tests**

Run: `uv run pytest tests/test_diagnostics.py tests/test_optimizer.py -q`

Expected: PASS.

## Task 4: Pairwise Relation Labels And Loss

**Files:**
- Modify: `src/floorset_arch/training/losses.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write pairwise label test**

```python
def test_pairwise_relation_targets_ignore_ambiguous_pairs():
    fp_sol = torch.tensor([
        [2.0, 2.0, 0.0, 0.0],
        [2.0, 2.0, 5.0, 0.0],
        [2.0, 2.0, 5.5, 5.0],
    ])
    pairs = torch.tensor([[0, 1], [1, 2]])

    targets = build_pairwise_relation_targets(fp_sol, pairs, min_gap=1.0, clear_ratio=1.25)

    assert targets["x_label"].tolist() == [1, 0]
    assert targets["y_label"].tolist() == [0, 1]
    assert targets["mask"].tolist() == [True, True]
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_model.py::test_pairwise_relation_targets_ignore_ambiguous_pairs -q`

Expected: FAIL because helper is missing.

- [ ] **Step 3: Implement labels and pairwise loss**

Add `build_pairwise_relation_targets()` and `pairwise_relation_loss()` to `losses.py`. Labels should be derived from expert rectangle centers in `fp_sol`.

- [ ] **Step 4: Run model/loss tests**

Run: `uv run pytest tests/test_model.py -q`

Expected: PASS.

## Task 5: Add Pairwise Head To GNN

**Files:**
- Modify: `src/floorset_arch/nn/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write pair head shape test**

```python
def test_floorplan_gnn_pairwise_logits_shape():
    model = FloorplanGNN(node_feat_dim=18, hidden_dim=16, num_layers=1)
    node_feat = torch.randn(4, 18)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.empty(0, 1)
    pairs = torch.tensor([[0, 1], [2, 3]])

    output = model(node_feat, edge_index, edge_attr, pairs=pairs)

    assert output["pair_logits"].shape == (2, 2)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_model.py::test_floorplan_gnn_pairwise_logits_shape -q`

Expected: FAIL because `pairs` argument or `pair_logits` output is missing.

- [ ] **Step 3: Implement pair head**

Update `FloorplanGNN.forward()` to accept optional `pairs`. Use encoded node embeddings for `i`, `j`, absolute difference, and raw difference to produce two logits: x-precedence and y-precedence.

- [ ] **Step 4: Run model tests**

Run: `uv run pytest tests/test_model.py -q`

Expected: PASS.

## Task 6: Wire Pairwise Loss Into Training

**Files:**
- Modify: `src/floorset_arch/training/train.py`
- Modify: `src/floorset_arch/training/checkpoint.py`

- [ ] **Step 1: Add train args**

Add:

```python
parser.add_argument("--pairwise-weight", type=float, default=0.20)
parser.add_argument("--pairwise-pairs", type=int, default=4096)
```

- [ ] **Step 2: Compute pairs during training**

Sample pairs, request `model(..., pairs=pairs)`, compute pairwise relation loss from `fp_sol`, and add it to total loss.

- [ ] **Step 3: Save compatibility metadata**

Save `has_pair_head=True` in checkpoints when the model has the new head.

- [ ] **Step 4: Run training smoke test**

Run: `uv run -m floorset_arch.training.train --num-samples 4 --val-samples 2 --epochs 1 --output-dir /tmp/floorset-smoke --write-stable-checkpoints`

Expected: completes one epoch and saves a checkpoint.

## Task 7: Blend Pairwise Logits Into Decoder

**Files:**
- Modify: `src/floorset_arch/models.py`
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `src/floorset_arch/relative_order.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Add guidance field test**

```python
def test_anchor_guidance_can_store_pairwise_logits():
    guidance = AnchorGuidance()
    guidance.pairwise_axis[(0, 1)] = (2.0, -1.0)

    assert guidance.pairwise_axis[(0, 1)] == (2.0, -1.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_optimizer.py::test_anchor_guidance_can_store_pairwise_logits -q`

Expected: FAIL because `AnchorGuidance.pairwise_axis` is missing.

- [ ] **Step 3: Store pairwise logits**

Add `pairwise_axis: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)` to `AnchorGuidance`.

Update `_try_anchor_guidance()` to pass sampled or all block pairs to the model only when the checkpoint supports pair logits. For `n <= 120`, all unordered pairs are at most 7140 and acceptable.

- [ ] **Step 4: Blend logits in relative-order decisions**

In `relative_order.py`, when evaluating pair `(i, j)`, read `guidance.pairwise_axis.get((min(i, j), max(i, j)))`. Subtract a bounded model bias from the predicted axis score:

```python
if pair_logits is not None:
    x_logit, y_logit = pair_logits
    model_axis_bias = max(-1.0, min(1.0, x_logit - y_logit))
    h_score -= 0.20 * model_axis_bias
    v_score += 0.20 * model_axis_bias
```

Keep this blend small first so old checkpoints and current heuristics remain stable.

- [ ] **Step 5: Run optimizer tests**

Run: `uv run pytest tests/test_optimizer.py tests/test_constructive.py -q`

Expected: PASS.

## Task 8: Evaluate Honest Score And Trace Tail Cases

**Files:**
- Modify: `docs/optimization-notes.md`

- [ ] **Step 1: Run full tests**

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 2: Run honest evaluation**

Run: `bash scripts/eval_total.sh`

Expected: completes with no artificial sleep. Record total score, feasibility count, average runtime, and tail case costs.

- [ ] **Step 3: Run tail trace**

Run:

```bash
FLOORSET_REPAIR_TRACE_JSONL=/tmp/floorset-tail-trace.jsonl bash scripts/eval_single.sh 99
FLOORSET_REPAIR_TRACE_JSONL=/tmp/floorset-tail-trace.jsonl bash scripts/eval_single.sh 98
```

Expected: trace file contains JSONL rows with `before_repair`, `after_repair`, and `delta`.

- [ ] **Step 4: Update optimization notes**

Add an entry to `docs/optimization-notes.md` with the honest score and repair-trace summary. Include whether pairwise logits improved pre-repair overlap/soft counts or only shifted final repaired score.

## Self-Review

- Spec coverage: runtime calibration removal is Task 1; repair diagnostics are Tasks 2-3; pairwise relation learning is Tasks 4-7; evaluation and notes are Task 8.
- Placeholder scan: no `TBD`, `TODO`, or deferred implementation text remains.
- Type consistency: `AnchorGuidance.pairwise_axis`, `pair_logits`, `placement_metrics`, and `repair_delta` names are used consistently.
- Scope check: learned slot ranking is intentionally excluded from this plan and remains a later project.
