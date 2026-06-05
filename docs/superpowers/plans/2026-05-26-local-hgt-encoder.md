# Local HGT Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `encoder_type="hgt"` as a local-only heterogeneous graph transformer encoder for Anchor-GNN Guidance.

**Architecture:** Add tensorized HGT graph inputs built from the existing heterogeneous floorplan graph, add relation-specific local attention layers in `FloorplanGNN`, and wire training/inference/checkpoint config through the existing AnchorGuidance heads. Keep decoder, repair, candidate policy, and global attention unchanged.

**Tech Stack:** Python, PyTorch, pytest, existing `floorset_arch` training/evaluation scripts.

---

## File Structure

- `src/floorset_arch/features.py`: add fixed-dimension `AnchorHGTGraphInputs` and `build_anchor_hgt_graph_inputs()`.
- `src/floorset_arch/nn/model.py`: add `HGTLayer` and `encoder_type="hgt"` support while keeping MPNN and graph-transformer behavior compatible.
- `src/floorset_arch/training/train.py`: route HGT training batches through hetero graph inputs and persist HGT config.
- `src/floorset_arch/training/checkpoint.py`: save HGT node dimensions and relation specs in checkpoint payloads.
- `src/floorset_arch/optimizer.py`: load HGT checkpoint config and build HGT graph inputs during inference.
- `scripts/train_hgt.sh`: provide a user-facing HGT training wrapper.
- `README.md`: document the new HGT training command after implementation exists.
- `tests/test_model.py`: cover HGT forward behavior and relation-specific modules.
- `tests/test_parser.py` or `tests/test_optimizer.py`: cover HGT input construction/inference wiring.
- `tests/test_scripts.py`: cover the new training script defaults if current script tests support it.

## Task 1: HGT Input Builder

**Files:**
- Modify: `src/floorset_arch/features.py`
- Test: `tests/test_model.py`

- [x] **Step 1: Write failing tests**

Add tests that build a small parsed instance with b2b, p2b, cluster, MIB, and boundary constraints, then assert:

```python
from floorset_arch.features import build_anchor_hgt_graph_inputs

def test_hgt_graph_inputs_keep_typed_local_relations(sample_instance):
    graph = build_anchor_hgt_graph_inputs(sample_instance)
    assert set(graph.node_features) >= {"block", "pin", "cluster", "mib", "boundary"}
    assert ("block", "connects", "block") in graph.edge_index
    assert ("pin", "pin_connects", "block") in graph.edge_index
    assert ("block", "pin_connects", "pin") in graph.edge_index
    assert ("cluster", "has_member", "block") in graph.edge_index
    assert ("mib", "has_member", "block") in graph.edge_index
    assert ("boundary", "has_member", "block") in graph.edge_index
```

- [x] **Step 2: Run red test**

Run: `uv run pytest tests/test_model.py::test_hgt_graph_inputs_keep_typed_local_relations -q`

Expected: FAIL because `build_anchor_hgt_graph_inputs` does not exist.

- [x] **Step 3: Implement fixed HGT input builder**

Add:

```python
@dataclass(frozen=True)
class AnchorHGTGraphInputs:
    node_features: dict[str, torch.Tensor]
    edge_index: dict[tuple[str, str, str], torch.Tensor]
    edge_attr: dict[tuple[str, str, str], torch.Tensor]
    node_feat_dims: dict[str, int]
    relation_specs: tuple[tuple[str, str, str], ...]
```

and `build_anchor_hgt_graph_inputs(inst, device=None)` with fixed feature dimensions per node type.

Implementation note: relation specs are canonical and include empty relation tensors when a sample has no edges for a relation, so checkpoint modules are stable across train/infer samples.

- [x] **Step 4: Run green test**

Run: `uv run pytest tests/test_model.py::test_hgt_graph_inputs_keep_typed_local_relations -q`

Expected: PASS.

## Task 2: Relation-Specific Local HGT Layers

**Files:**
- Modify: `src/floorset_arch/nn/model.py`
- Test: `tests/test_model.py`

- [x] **Step 1: Write failing tests**

Add tests that instantiate `FloorplanGNN(..., encoder_type="hgt")`, pass HGT graph inputs, and assert output shapes plus relation-specific modules:

```python
def test_hgt_encoder_outputs_anchor_heads_for_blocks(sample_instance):
    graph = build_anchor_hgt_graph_inputs(sample_instance)
    model = FloorplanGNN(
        node_feat_dim=graph.node_features["block"].shape[1],
        hidden_dim=32,
        num_layers=2,
        encoder_type="hgt",
        num_heads=4,
        hgt_node_feat_dims=graph.node_feat_dims,
        hgt_relation_specs=graph.relation_specs,
    )
    pairs = torch.tensor([(0, 1)], dtype=torch.long)
    out = model(
        graph.node_features["block"],
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 1)),
        hgt_node_features=graph.node_features,
        hgt_edge_index=graph.edge_index,
        hgt_edge_attr=graph.edge_attr,
        pairs=pairs,
    )
    assert out["anchor"].shape == (sample_instance.block_count, 2)
    assert out["priority"].shape == (sample_instance.block_count,)
    assert out["log_aspect"].shape == (sample_instance.block_count,)
    assert out["pair_logits"].shape == (1, 2)
```

- [x] **Step 2: Run red test**

Run: `uv run pytest tests/test_model.py::test_hgt_encoder_outputs_anchor_heads_for_blocks -q`

Expected: FAIL because `encoder_type="hgt"` is unsupported.

- [x] **Step 3: Implement HGT model support**

Add `HGTLayer` with per-node-type query projection, per-relation key/value projection, per-relation edge bias, local edge softmax by destination, typed residual norms, and typed feed-forward networks.

- [x] **Step 4: Run green test**

Run: `uv run pytest tests/test_model.py::test_hgt_encoder_outputs_anchor_heads_for_blocks -q`

Expected: PASS.

## Task 3: Training and Checkpoint Wiring

**Files:**
- Modify: `src/floorset_arch/training/train.py`
- Modify: `src/floorset_arch/training/checkpoint.py`
- Test: `tests/test_model.py`

- [x] **Step 1: Write failing tests**

Add tests that checkpoint payloads preserve HGT config and that parser choices include `hgt`.

- [x] **Step 2: Run red tests**

Run: `uv run pytest tests/test_model.py -q`

Expected: FAIL on missing HGT config support.

- [x] **Step 3: Implement training/checkpoint wiring**

Route `args.encoder == "hgt"` through `build_anchor_hgt_graph_inputs()`, pass HGT tensors into `model(...)`, and save/load `hgt_node_feat_dims` plus `hgt_relation_specs`.

- [x] **Step 4: Run green tests**

Run: `uv run pytest tests/test_model.py -q`

Expected: PASS.

## Task 4: Inference Wiring

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Test: `tests/test_optimizer.py`

- [x] **Step 1: Write failing test**

Add a focused test or model-level smoke that HGT checkpoint config can be loaded and `_try_anchor_guidance()` calls the HGT graph builder.

- [x] **Step 2: Run red test**

Run: `uv run pytest tests/test_optimizer.py -q`

Expected: FAIL before optimizer HGT support.

- [x] **Step 3: Implement optimizer HGT support**

Load HGT config from checkpoint payload and use `build_anchor_hgt_graph_inputs()` when checkpoint encoder type is `hgt`.

- [x] **Step 4: Run green test**

Run: `uv run pytest tests/test_optimizer.py -q`

Expected: PASS.

## Task 5: Script and Docs

**Files:**
- Create: `scripts/train_hgt.sh`
- Modify: `README.md`
- Modify: `docs/optimization-notes.md` if implementation notes need updating.

- [x] **Step 1: Add script and docs**

Add `scripts/train_hgt.sh` mirroring `scripts/train_transformer.sh` with `ENCODER=hgt`, `CHECKPOINT_PREFIX=gnn_hgt`, and no global attention options.

Implementation note: `scripts/train_hgt.sh` also defaults high-risk decoder-aware order/pairwise multipliers to `1.5`.

- [x] **Step 2: Verify script help/smoke path**

Run a tiny CPU training smoke:

```bash
DEVICE=cpu WANDB=0 NUM_SAMPLES=2 VAL_SAMPLES=1 EPOCHS=1 HIDDEN_DIM=16 LAYERS=1 NUM_HEADS=4 OUTPUT_DIR=/tmp/floorset-hgt-smoke bash scripts/train_hgt.sh
```

Expected: command exits 0 and writes a tiny HGT checkpoint.

## Task 6: Final Verification

**Files:**
- All changed files.

- [x] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_model.py tests/test_optimizer.py tests/test_scripts.py -q
```

Expected: PASS.

- [x] **Step 2: Run full tests if focused tests pass**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [x] **Step 3: Review git diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: only intended files changed and no whitespace errors.

## Self-Review

- Spec coverage: implements local-only HGT, relation-specific attention, b2b/p2b typed locality, checkpoint compatibility, training/inference wiring, and docs.
- Placeholder scan: no placeholders remain.
- Type consistency: HGT graph input names are `node_features`, `edge_index`, `edge_attr`, `node_feat_dims`, and `relation_specs` across builder, model, training, optimizer, and checkpoint code.
