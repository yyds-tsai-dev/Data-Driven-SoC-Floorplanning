# Partner Train-Only Retrieval Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe, same-block-count retrieval prototype that transfers complete training layouts into unseen cases and measures whether they survive the existing partner prescreen/refine path under a fixed deadline.

**Architecture:** Precompute compact global and node features plus training `fp_sol` into per-`n` NumPy shards. At inference, query top-K within the same `n`, solve block correspondence, transfer a coherent normalized layout, then present it through the shared partner candidate-source boundary. This plan ends at Gate R4; CP-SAT work starts only if the gate passes.

**Tech Stack:** Python 3.12, NumPy, PyTorch, stdlib JSON, pytest, existing FloorSet loaders and partner evaluator.

## Global Constraints

- Complete `2026-07-23-partner-candidate-source-foundation.md` first.
- The production index is train-only; validation/evaluation inputs and solutions must be rejected at build and load time.
- First version supports only equal block count; no padding or cross-`n` transfer.
- Transfer `(x, y, w, h)` coherently; do not send order-only hints to column SA.
- Retrieval replaces existing candidate/refine capacity; it does not add time, GPU batch size, workers, or refine slots.
- Do not implement CP-SAT unless the final Gate R4 criteria pass.

---

## File Structure

- Create `src/solver/retrieval_features_claude.py`: global/node feature extraction and typed query/source records.
- Create `src/solver/retrieval_index_claude.py`: per-`n` train-only shards, manifest validation, and top-K query.
- Create `src/solver/retrieval_matching_claude.py`: O(n^3) Hungarian assignment with compatibility guards.
- Create `src/solver/retrieval_transfer_claude.py`: D4-aware coherent layout transfer and hard-anchor projection.
- Create `scripts/build_partner_retrieval_index.py`: bounded, reproducible training-index builder.
- Create `scripts/probes/retrieval_probe.py`: held-out query, raw-transfer, and repaired-candidate diagnostics.
- Modify `src/solver/my_opt_claude.py`: opt-in retrieval source using fixed quotas.
- Create focused tests under `tests/test_partner_retrieval_*.py`.

### Task 1: Permutation-Invariant Retrieval Features

**Files:**
- Create: `src/solver/retrieval_features_claude.py`
- Test: `tests/test_partner_retrieval_features.py`

**Interfaces:**
- Produces: `RetrievalFeatures(global_vector: np.ndarray, node_matrix: np.ndarray, block_count: int)`.
- Produces: `extract_retrieval_features(area, b2b, p2b, pins, constraints, target_positions) -> RetrievalFeatures`.
- Shapes: global vector `[24]`; node matrix `[N, 16]`; all values finite `float32`.

- [ ] **Step 1: Write failing invariance and shape tests**

Create `tests/test_partner_retrieval_features.py` with a three-block synthetic instance, then permute blocks and remap b2b/p2b indices:

```python
import numpy as np

from retrieval_features_claude import extract_retrieval_features


def _instance():
    area = np.array([4.0, 9.0, 16.0], dtype=np.float32)
    b2b = np.array([[0, 1, 2.0], [1, 2, 1.0], [-1, -1, -1]], dtype=np.float32)
    p2b = np.array([[0, 0, 1.0], [1, 2, 1.0], [-1, -1, -1]], dtype=np.float32)
    pins = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    constraints = np.array([
        [0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 2, 0, 2],
    ], dtype=np.float32)
    targets = np.array([[-1, -1, -1, -1], [-1, -1, 3, 3], [7, 2, 4, 4]], dtype=np.float32)
    return area, b2b, p2b, pins, constraints, targets


def test_feature_contract_is_finite_and_fixed_width():
    features = extract_retrieval_features(*_instance())
    assert features.global_vector.shape == (24,)
    assert features.node_matrix.shape == (3, 16)
    assert np.isfinite(features.global_vector).all()
    assert np.isfinite(features.node_matrix).all()


def test_global_features_are_block_permutation_invariant():
    area, b2b, p2b, pins, constraints, targets = _instance()
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    b2b2 = b2b.copy()
    valid = b2b2[:, 0] >= 0
    b2b2[valid, :2] = inverse[b2b2[valid, :2].astype(int)]
    p2b2 = p2b.copy()
    valid = p2b2[:, 0] >= 0
    p2b2[valid, 1] = inverse[p2b2[valid, 1].astype(int)]
    got = extract_retrieval_features(
        area[permutation], b2b2, p2b2, pins, constraints[permutation], targets[permutation]
    )
    expected = extract_retrieval_features(area, b2b, p2b, pins, constraints, targets)
    np.testing.assert_allclose(got.global_vector, expected.global_vector, atol=1e-6)
```

- [ ] **Step 2: Verify missing-module failure**

Run: `uv run pytest tests/test_partner_retrieval_features.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement typed features**

Create the dataclass and helpers. Use sorted distribution summaries rather than block-order concatenation:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalFeatures:
    global_vector: np.ndarray
    node_matrix: np.ndarray
    block_count: int

    def __post_init__(self) -> None:
        if self.global_vector.shape != (24,):
            raise ValueError("global retrieval feature must have shape [24]")
        if self.node_matrix.shape != (self.block_count, 16):
            raise ValueError("node retrieval features must have shape [N,16]")
        if not np.isfinite(self.global_vector).all() or not np.isfinite(self.node_matrix).all():
            raise ValueError("retrieval features must be finite")


def _summary(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(values.mean()), float(values.std()),
            float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))]


def extract_retrieval_features(area, b2b, p2b, pins, constraints, target_positions):
    area = np.asarray(area, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    n = len(area)
    degree = np.zeros(n, dtype=np.float64)
    weighted_degree = np.zeros(n, dtype=np.float64)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for edge in np.asarray(b2b):
        if edge[0] < 0:
            continue
        i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
        degree[[i, j]] += 1.0
        weighted_degree[[i, j]] += max(weight, 0.0)
        adjacency[i, j] += max(weight, 0.0)
        adjacency[j, i] += max(weight, 0.0)
    pin_degree = np.zeros(n, dtype=np.float64)
    for edge in np.asarray(p2b):
        if edge[0] >= 0 and 0 <= int(edge[1]) < n:
            pin_degree[int(edge[1])] += 1.0
    safe_area = np.maximum(area, 1e-9)
    area_share = safe_area / safe_area.sum()
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    mib = constraints[:, 2] if constraints.shape[1] > 2 else np.zeros(n)
    cluster = constraints[:, 3] if constraints.shape[1] > 3 else np.zeros(n)
    boundary = constraints[:, 4].astype(int) if constraints.shape[1] > 4 else np.zeros(n, dtype=int)
    neighbor_denominator = np.maximum((adjacency > 0).sum(axis=1), 1.0)
    neighbor_log_area = (adjacency > 0) @ np.log(safe_area) / neighbor_denominator
    neighbor_degree = (adjacency > 0) @ degree / neighbor_denominator
    global_values = (
        [float(n), float(np.log1p(safe_area.sum()))]
        + _summary(np.log(safe_area))
        + _summary(degree)
        + _summary(weighted_degree)
        + _summary(pin_degree)
        + [float(fixed.mean()), float(preplaced.mean()), float((mib > 0).mean()),
           float((cluster > 0).mean()), float((boundary > 0).mean()),
           float(np.unique(cluster[cluster > 0]).size)]
    )
    global_vector = np.asarray(global_values[:24], dtype=np.float32)
    node_matrix = np.stack([
        np.log(safe_area), area_share, degree, weighted_degree, pin_degree,
        fixed, preplaced, mib > 0, cluster > 0,
        (boundary & 1) != 0, (boundary & 2) != 0,
        (boundary & 4) != 0, (boundary & 8) != 0,
        np.asarray(target_positions)[:, 0] >= 0,
        neighbor_log_area, neighbor_degree,
    ], axis=1).astype(np.float32)
    return RetrievalFeatures(global_vector, node_matrix, n)
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_partner_retrieval_features.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/solver/retrieval_features_claude.py tests/test_partner_retrieval_features.py
git commit -m "feat: add permutation-invariant retrieval features"
```

### Task 2: Train-Only Sharded Index

**Files:**
- Create: `src/solver/retrieval_index_claude.py`
- Test: `tests/test_partner_retrieval_index.py`

**Interfaces:**
- Produces: `RetrievalShard(block_count, source_ids, global_features, node_features, fp_xywh, feature_mean, feature_scale)`.
- Produces: `RetrievalIndex.load(root: Path)`, `query(features, top_k, exclude_source_id=None)`.
- Storage: `manifest.json` plus `n_XXX.npz`; no pickle/object arrays. Each manifest shard entry records `n`, `count`, file SHA-256, and `fp_xywh` SHA-256.

- [ ] **Step 1: Write failing save/load, same-n, and leakage tests**

```python
import json
from pathlib import Path

import numpy as np
import pytest

from retrieval_index_claude import RetrievalIndex, RetrievalShard, save_index


def _shard():
    return RetrievalShard(
        block_count=2,
        source_ids=np.array([10, 11], dtype=np.int64),
        global_features=np.array([[0.0] * 24, [1.0] * 24], dtype=np.float32),
        node_features=np.zeros((2, 2, 16), dtype=np.float32),
        fp_xywh=np.ones((2, 2, 4), dtype=np.float32),
        feature_mean=np.zeros(24, dtype=np.float32),
        feature_scale=np.ones(24, dtype=np.float32),
    )


def test_index_queries_same_n_and_excludes_self(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)
    index = RetrievalIndex.load(tmp_path)
    result = index.query(2, np.zeros(24, dtype=np.float32), top_k=1, exclude_source_id=10)
    assert result.source_ids.tolist() == [11]


def test_index_rejects_evaluation_manifest(tmp_path: Path):
    save_index(tmp_path, [_shard()], source_split="train", feature_version=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["source_split"] = "evaluation"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="train-only"):
        RetrievalIndex.load(tmp_path)
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_partner_retrieval_index.py -q`

Expected: FAIL importing the missing module.

- [ ] **Step 3: Implement non-pickle shards and standardized cosine query**

Use these exact stored keys: `source_ids`, `global_features`, `node_features`, `fp_xywh`, `feature_mean`, `feature_scale`. `save_index()` must reject any `source_split != "train"`; `load()` must repeat the check rather than trusting the builder and verify every recorded SHA-256 before making a shard queryable.

```python
import hashlib
from pathlib import Path


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RetrievalResult:
    source_ids: np.ndarray
    distances: np.ndarray
    node_features: np.ndarray
    fp_xywh: np.ndarray


def _normalized(features, mean, scale):
    z = (features - mean) / np.maximum(scale, 1e-6)
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9)


def query(self, block_count, global_vector, top_k, exclude_source_id=None):
    shard = self.shards.get(int(block_count))
    if shard is None:
        return RetrievalResult(np.empty(0, dtype=np.int64), np.empty(0),
                               np.empty((0, block_count, 16)), np.empty((0, block_count, 4)))
    query = _normalized(global_vector[None], shard.feature_mean, shard.feature_scale)[0]
    keys = _normalized(shard.global_features, shard.feature_mean, shard.feature_scale)
    distance = 1.0 - keys @ query
    order = np.argsort(distance, kind="stable")
    if exclude_source_id is not None:
        order = order[shard.source_ids[order] != exclude_source_id]
    order = order[:max(0, int(top_k))]
    return RetrievalResult(shard.source_ids[order], distance[order],
                           shard.node_features[order], shard.fp_xywh[order])
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_partner_retrieval_index.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/solver/retrieval_index_claude.py tests/test_partner_retrieval_index.py
git commit -m "feat: add train-only retrieval index"
```

### Task 3: Block Matching and Coherent Transfer

**Files:**
- Create: `src/solver/retrieval_matching_claude.py`
- Create: `src/solver/retrieval_transfer_claude.py`
- Test: `tests/test_partner_retrieval_transfer.py`

**Interfaces:**
- Produces: `match_blocks(source_nodes, target_nodes, max_cost) -> MatchResult` with `target_to_source`, `total_cost`, `confidence`, `accepted`.
- Produces: `transfer_layout(source_fp_xywh, target_to_source, target_area, constraints, target_positions, transform) -> np.ndarray`.
- Contract: output is `[N,4]` in `[x,y,w,h]`, finite, movable shapes preserve target area within floating tolerance, fixed/preplaced shapes use their prescribed dimensions, and preplaced x/y is exact.

- [ ] **Step 1: Write failing assignment and transfer tests**

```python
import numpy as np

from retrieval_matching_claude import match_blocks
from retrieval_transfer_claude import transfer_layout


def test_matching_recovers_area_and_constraint_correspondence():
    source = np.zeros((3, 16), dtype=np.float64)
    source[:, 0] = [0.0, 1.0, 2.0]
    source[1, 5] = 1.0
    source[2, 7] = 1.0
    target = source[[2, 0, 1]]
    result = match_blocks(source, target, max_cost=0.1)
    assert result.accepted
    assert result.target_to_source.tolist() == [2, 0, 1]


def test_transfer_preserves_target_area_and_hard_anchor():
    source = np.array([[0, 0, 2, 2], [3, 0, 3, 3]], dtype=np.float64)
    area = np.array([9.0, 4.0])
    constraints = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float64)
    target_positions = np.array([[7, 5, 3, 3], [-1, -1, -1, -1]], dtype=np.float64)
    got = transfer_layout(source, np.array([1, 0]), area, constraints, target_positions, "identity")
    np.testing.assert_allclose(got[:, 2] * got[:, 3], area, rtol=1e-6)
    np.testing.assert_allclose(got[0, :2], [7, 5], atol=0.0)
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_partner_retrieval_transfer.py -q`

Expected: FAIL importing missing modules.

- [ ] **Step 3: Implement deterministic Hungarian matching**

Implement the following potentials-based O(n^3) assignment in `retrieval_matching_claude.py`; reject NaN costs and non-square same-`n` inputs. The first public cost combines standardized node L1 distance with hard incompatibility for fixed/preplaced flags and group-constraint membership. Boundary direction remains a soft feature until D4 remapping is applied, so an otherwise valid mirrored match is not forbidden.

```python
def hungarian_min_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Hungarian input must be finite square cost matrix")
    if not np.isfinite(cost).all():
        raise ValueError("Hungarian cost must be finite")
    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for row in range(1, n + 1):
        p[0] = row
        min_value = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = p[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, n + 1):
                if used[candidate]:
                    continue
                reduced = cost[active_row - 1, candidate - 1] - u[active_row] - v[candidate]
                if reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    way[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(n + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = np.empty(n, dtype=np.int64)
    for column in range(1, n + 1):
        assignment[p[column] - 1] = column - 1
    return assignment
```

```python
@dataclass(frozen=True)
class MatchResult:
    target_to_source: np.ndarray
    total_cost: float
    confidence: float
    accepted: bool


def match_blocks(source_nodes, target_nodes, max_cost):
    if source_nodes.shape != target_nodes.shape:
        raise ValueError("first retrieval version requires equal N and feature width")
    cost = np.abs(target_nodes[:, None, :] - source_nodes[None, :, :]).mean(axis=-1)
    hard = np.any(
        target_nodes[:, None, [5, 6, 7, 8]] != source_nodes[None, :, [5, 6, 7, 8]], axis=-1
    )
    cost = np.where(hard, 1e6, cost)
    assignment = hungarian_min_cost(cost)
    chosen = cost[np.arange(len(assignment)), assignment]
    second = np.partition(cost, 1, axis=1)[:, 1] if len(assignment) > 1 else chosen + 1.0
    confidence = float(np.mean(second - chosen))
    total = float(chosen.mean())
    return MatchResult(assignment, total, confidence,
                       bool(np.isfinite(total) and total <= max_cost and chosen.max() < 1e5))
```

- [ ] **Step 4: Implement coherent transfer**

`retrieval_transfer_claude.py` must reorder the complete source rectangles, apply one of `identity/mirror_x/mirror_y/transpose`, normalize source coordinates by `sqrt(sum(source w*h))`, reconstruct target shapes from target area and transferred log-aspect, then impose hard anchors:

```python
def apply_d4_rectangles(rectangles, transform):
    result = np.asarray(rectangles, dtype=np.float64).copy()
    if transform == "identity":
        return result
    x0 = result[:, 0].min()
    y0 = result[:, 1].min()
    x1 = (result[:, 0] + result[:, 2]).max()
    y1 = (result[:, 1] + result[:, 3]).max()
    if transform == "mirror_x":
        result[:, 0] = x0 + x1 - (result[:, 0] + result[:, 2])
    elif transform == "mirror_y":
        result[:, 1] = y0 + y1 - (result[:, 1] + result[:, 3])
    elif transform == "transpose":
        result = result[:, [1, 0, 3, 2]]
    else:
        raise ValueError(f"unsupported D4 transform: {transform}")
    return result


def remap_boundary_node_features(node_features, transform):
    result = np.asarray(node_features, dtype=np.float64).copy()
    left, right, top, bottom = [result[:, column].copy() for column in (9, 10, 11, 12)]
    if transform == "identity":
        return result
    if transform == "mirror_x":
        result[:, 9], result[:, 10] = right, left
    elif transform == "mirror_y":
        result[:, 11], result[:, 12] = bottom, top
    elif transform == "transpose":
        result[:, 9], result[:, 10], result[:, 11], result[:, 12] = bottom, top, right, left
    else:
        raise ValueError(f"unsupported D4 transform: {transform}")
    return result


def transfer_layout(source_fp_xywh, target_to_source, target_area,
                    constraints, target_positions, transform):
    source = np.asarray(source_fp_xywh, dtype=np.float64)[target_to_source].copy()
    source = apply_d4_rectangles(source, transform)
    source_scale = np.sqrt(np.maximum((source[:, 2] * source[:, 3]).sum(), 1.0))
    xy = source[:, :2] / source_scale
    log_aspect = np.clip(np.log(source[:, 2] / np.maximum(source[:, 3], 1e-9)), -3.0, 3.0)
    target_scale = np.sqrt(np.maximum(np.asarray(target_area).sum(), 1.0))
    aspect = np.exp(log_aspect)
    width = np.sqrt(target_area * aspect)
    height = np.sqrt(target_area / aspect)
    result = np.column_stack([xy * target_scale, width, height])
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    has_wh = (target_positions[:, 2] > 0) & (target_positions[:, 3] > 0)
    shape_known = (fixed | preplaced) & has_wh
    result[shape_known, 2:4] = target_positions[shape_known, 2:4]
    has_xy = preplaced & (target_positions[:, 0] >= 0) & (target_positions[:, 1] >= 0)
    result[has_xy, :2] = target_positions[has_xy, :2]
    if not np.isfinite(result).all() or np.any(result[:, 2:] <= 0):
        raise ValueError("transferred rectangles must be finite and positive")
    return result
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_partner_retrieval_transfer.py -q`

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/solver/retrieval_matching_claude.py src/solver/retrieval_transfer_claude.py tests/test_partner_retrieval_transfer.py
git commit -m "feat: match and transfer retrieved layouts"
```

### Task 4: Reproducible Index Builder and Offline Probe

**Files:**
- Create: `scripts/build_partner_retrieval_index.py`
- Create: `scripts/probes/retrieval_probe.py`
- Test: `tests/test_partner_retrieval_scripts.py`

**Interfaces:**
- Builder flags: `--data-path`, `--output`, `--max-per-n`, `--seed`.
- Probe flags: `--index`, `--data-path`, `--cases`, `--top-k`, `--output`.
- Default pilot cap: `2000` sources per block count using seeded reservoir sampling.

- [ ] **Step 1: Write parser and leakage-guard tests**

```python
from pathlib import Path
import subprocess
import sys


def test_retrieval_builder_help_is_noninteractive():
    result = subprocess.run(
        [sys.executable, "scripts/build_partner_retrieval_index.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--max-per-n" in result.stdout


def test_builder_source_split_is_not_configurable():
    text = Path("scripts/build_partner_retrieval_index.py").read_text()
    assert "--source-split" not in text
    assert 'source_split="train"' in text
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_partner_retrieval_scripts.py -q`

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Implement the bounded builder**

Load only `FloorplanDatasetLite`, never `FloorplanDatasetLiteTest`. For each sampled training record:

```python
sample = dataset[index]
area, b2b, p2b, pins, constraints = [tensor.cpu().numpy() for tensor in sample["input"]]
_tree, fp_sol, _metrics = sample["label"]
n = int((area != -1).sum())
fp = fp_sol[:n].cpu().numpy()
fp_xywh = fp[:, [2, 3, 0, 1]].astype(np.float32)
target_positions = np.full((n, 4), -1.0, dtype=np.float32)
fixed = constraints[:n, 0] != 0
preplaced = constraints[:n, 1] != 0
target_positions[fixed | preplaced, 2:4] = fp_xywh[fixed | preplaced, 2:4]
target_positions[preplaced, 0:2] = fp_xywh[preplaced, 0:2]
features = extract_retrieval_features(
    area[:n], b2b, p2b, pins, constraints[:n], target_positions
)
```

Use `np.random.default_rng(seed)` and reservoir sampling per `n`. Write a temporary directory, validate it with `RetrievalIndex.load()`, then rename it to the requested output so interrupted builds never look complete.

- [ ] **Step 4: Implement the offline probe**

For each requested evaluation query, report JSON rows containing `case_id`, `n`, retrieved source ids/distances, match cost/confidence, raw overlap/HPWL proxies, transform, cold index-load time, warm query time, matching time, and transfer time. The script may read evaluation inputs and labels for diagnostics, but must never write them into the index.

- [ ] **Step 5: Run focused tests and a small builder smoke**

Run: `uv run pytest tests/test_partner_retrieval_scripts.py -q`

Expected: both tests PASS.

Run: `uv run python scripts/build_partner_retrieval_index.py --data-path FloorSet --output artifacts/retrieval/pilot --max-per-n 2 --seed 17`

Expected: manifest reports `source_split=train`, at most two records per present block count, and `RetrievalIndex.load()` succeeds.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_partner_retrieval_index.py scripts/probes/retrieval_probe.py tests/test_partner_retrieval_scripts.py
git commit -m "feat: build and probe train-only retrieval index"
```

### Task 5: Opt-In Partner Candidate Source

**Files:**
- Modify: `src/solver/my_opt_claude.py`
- Create: `tests/test_partner_retrieval_integration.py`
- Create: `scripts/probes/run_retrieval_gate.sh`

**Interfaces:**
- New env: `PARTNER_RETRIEVAL_INDEX`, absent by default.
- New env: `PARTNER_RETRIEVAL_SLOTS`, default `0`.
- Produces: `_sample_retrieval_preds(self, n, at, cons, tpos, b2b, p2b, pins, K) -> CandidateBatch`.
- Consumes: `allocate_quotas()` and `rank_predictions()` from the foundation plan.

- [ ] **Step 1: Write opt-in and capacity tests**

```python
from candidate_supply_claude import allocate_quotas


def test_retrieval_disabled_preserves_all_direct_slots():
    assert allocate_quotas(12, {"direct": 12, "retrieval": 0}, ("direct",)) == {"direct": 12}


def test_retrieval_slots_replace_direct_capacity():
    got = allocate_quotas(12, {"direct": 8, "retrieval": 4}, ("direct", "retrieval"))
    assert got == {"direct": 8, "retrieval": 4}
    assert sum(got.values()) == 12
```

- [ ] **Step 2: Run tests before integration**

Run: `uv run pytest tests/test_partner_retrieval_integration.py -q`

Expected: allocation tests PASS; no optimizer behavior has changed yet.

- [ ] **Step 3: Load the index only when explicitly enabled**

In `MyOptimizer.__init__`, initialize `self.retrieval_index = None`; load only when both env values are present and slots are positive. Any load failure records a verbose diagnostic and leaves retrieval disabled, without affecting Direct-v2.

- [ ] **Step 4: Generate, match, transfer, and jointly rank candidates**

Implement `_sample_retrieval_preds` to:

1. extract target features;
2. query same-`n` top-K;
3. try `identity`, `mirror_x`, `mirror_y`, `transpose`, remapping source boundary-bit feature columns before each match;
4. reject low-confidence matches;
5. return finite transferred arrays in a `CandidateBatch`;
6. merge with Direct candidates before the single shared `rank_predictions()` call.

For a legalizer request of `K`, compute `retrieval_slots = min(K, PARTNER_RETRIEVAL_SLOTS)` and `direct_slots = K - retrieval_slots`; the merged list passed to the legalizer must be truncated to exactly the same `K` capacity that the Direct-only baseline receives.

- [ ] **Step 5: Add gate runner**

`scripts/probes/run_retrieval_gate.sh` must run, in order:

```bash
uv run pytest tests/test_partner_retrieval_features.py \
  tests/test_partner_retrieval_index.py \
  tests/test_partner_retrieval_transfer.py \
  tests/test_partner_retrieval_integration.py -q

PARTNER_RETRIEVAL_SLOTS=0 bash scripts/partner_eval_cont.sh retrieval_direct_control
PARTNER_RETRIEVAL_INDEX=artifacts/retrieval/pilot \
PARTNER_RETRIEVAL_SLOTS=4 \
bash scripts/partner_eval_cont.sh retrieval_r4
```

Do not hard-code a validation solution path into the optimizer or index builder.

- [ ] **Step 6: Verify focused, full, interface, and full-100 runs**

Run: `bash scripts/probes/run_retrieval_gate.sh`

Expected: focused tests PASS and two result JSON files are written.

Run: `uv run pytest`

Expected: full suite PASS.

Run: `bash scripts/validate.sh`

Expected: interface validation succeeds.

- [ ] **Step 7: Commit**

```bash
git add src/solver/my_opt_claude.py tests/test_partner_retrieval_integration.py scripts/probes/run_retrieval_gate.sh
git commit -m "feat: add time-neutral retrieval candidates"
```

## Gate R4 Decision

Promote retrieval to a CP-SAT planning phase only if at least one condition holds across full-100 paired repeats:

1. Fixed-deadline no-runtime score improves beyond repeat noise.
2. Aggregate is neutral but retrieval wins a stable, feature-identifiable case band.
3. Transferred HPWL/area advantage survives repair often enough to win final selection.

Stop retrieval and do not start CP-SAT if matching confidence is mostly low, repair erases geometry, final weighted winner contribution is negligible, or gains require validation self-retrieval. Record the rejection in `docs/experiments/` with raw/repaired/final and runtime breakdowns.

If R4 passes, write a new CP-SAT implementation plan scoped to medoid and local-window probes. If both R4 and Flow F4 pass, write a separate fixed-capacity D+R+F portfolio plan; do not fold either follow-up into this gate implementation.
