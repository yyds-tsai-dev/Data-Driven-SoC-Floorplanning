# V10 Budget Proxy Runtime Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved v10 proxy-first selection, opt-in conditional runtime budget, and opt-in narrow grouping pair bias without touching the GNN or checkpoint path.

**Architecture:** Add a focused `v10_proxy` module that owns hard legality, no-runtime proxy scoring, rank keys, and acceptance comparisons. Wire `optimizer.py`, `repair.py`, and `relative_order.py` to that shared policy while keeping Phase 2 and Phase 3 behind opt-in flags.

**Tech Stack:** Python 3, PyTorch tensors, pytest, existing `floorset_arch` dataclasses and parser helpers.

---

## File Structure

- Create `src/floorset_arch/v10_proxy.py`: shared V10 hard-legality, no-runtime proxy, rank key, and better-than comparison helpers.
- Create `tests/test_v10_proxy.py`: focused unit tests for hard legality, proxy ranking, and tie-break behavior.
- Modify `src/floorset_arch/optimizer.py`: replace local candidate ranking and quality refine acceptance with `v10_proxy`.
- Modify `tests/test_optimizer.py`: add rank-policy and quality-refine acceptance tests.
- Modify `src/floorset_arch/repair.py`: replace `_score_better_v10_soft()` internals with `v10_proxy`, add conditional runtime budget stats, and stop automatically applying hard clamp in the v10 repair path.
- Modify `tests/test_repair.py`: add repair acceptance and trace tests.
- Modify `src/floorset_arch/budget_layer.py`: add conditional runtime budget flag/default helpers and keep hard clamp helpers as explicit ablation only.
- Modify `tests/test_budget_layer.py`: cover conditional budget helpers and preserve hard clamp ablation behavior.
- Modify `src/floorset_arch/relative_order.py`: add opt-in narrow grouping pair bias and remove broad key blend from the narrow path.
- Modify `tests/test_relative_order.py`: add narrow-bias tests and update broad-bias expectations if needed.
- Modify `CONTEXT.md`: keep the glossary terms already resolved during grilling; stage only if execution scope includes documentation cleanup.

## Task 1: Shared V10 Proxy Helper

**Files:**
- Create: `src/floorset_arch/v10_proxy.py`
- Create: `tests/test_v10_proxy.py`

- [ ] **Step 1: Write failing tests for proxy hard legality and acceptance**

Add this file:

```python
from __future__ import annotations

import torch

from floorset_arch.models import Placement, Rect
from floorset_arch.parser import parse_instance
from floorset_arch.v10_proxy import (
    hard_legality_rank,
    v10_proxy_better,
    v10_proxy_cost,
    v10_proxy_rank,
)


def _inst(block_count: int = 3, *, boundary: int = 0, fixed: bool = False, preplaced: bool = False):
    constraints = torch.zeros(block_count, 5)
    if boundary:
        constraints[:boundary, 4] = 1.0
    if fixed:
        constraints[0, 0] = 1.0
    if preplaced:
        constraints[0, 1] = 1.0
    targets = torch.full((block_count, 4), -1.0)
    if fixed:
        targets[0] = torch.tensor([-1.0, -1.0, 2.0, 2.0])
    if preplaced:
        targets[0] = torch.tensor([0.0, 0.0, 2.0, 2.0])
    return parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        targets,
    )


def test_hard_legality_rank_penalizes_missing_overlap_fixed_and_preplaced():
    inst = _inst(3, fixed=True, preplaced=False)
    legal = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(3.0, 0.0, 2.0, 2.0), 2: Rect(6.0, 0.0, 2.0, 2.0)})
    missing = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(3.0, 0.0, 2.0, 2.0)})
    overlap = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(1.0, 0.0, 2.0, 2.0), 2: Rect(6.0, 0.0, 2.0, 2.0)})
    fixed_bad = Placement({0: Rect(0.0, 0.0, 1.0, 4.0), 1: Rect(3.0, 0.0, 2.0, 2.0), 2: Rect(6.0, 0.0, 2.0, 2.0)})

    assert hard_legality_rank(inst, legal) < hard_legality_rank(inst, missing)
    assert hard_legality_rank(inst, legal) < hard_legality_rank(inst, overlap)
    assert hard_legality_rank(inst, legal) < hard_legality_rank(inst, fixed_bad)

    preplaced_inst = _inst(3, preplaced=True)
    preplaced_bad = Placement({0: Rect(1.0, 0.0, 2.0, 2.0), 1: Rect(3.0, 0.0, 2.0, 2.0), 2: Rect(6.0, 0.0, 2.0, 2.0)})
    assert hard_legality_rank(preplaced_inst, legal) < hard_legality_rank(preplaced_inst, preplaced_bad)


def test_v10_proxy_cost_balances_quality_and_soft_penalty():
    inst = _inst(3, boundary=1)
    compact_dirty = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    huge_clean = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(200.0, 0.0, 2.0, 2.0), 2: Rect(202.0, 0.0, 2.0, 2.0)})

    assert v10_proxy_cost(inst, compact_dirty) < v10_proxy_cost(inst, huge_clean)
    assert v10_proxy_rank(inst, compact_dirty) < v10_proxy_rank(inst, huge_clean)


def test_v10_proxy_better_rejects_proxy_regression_even_when_soft_improves():
    inst = _inst(3, boundary=1)
    current = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    huge_clean = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(200.0, 0.0, 2.0, 2.0), 2: Rect(202.0, 0.0, 2.0, 2.0)})

    assert not v10_proxy_better(inst, huge_clean, current)


def test_v10_proxy_better_allows_soft_tie_within_tolerance(monkeypatch):
    inst = _inst(3, boundary=1)
    current = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    soft_better = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 3.0, 2.0, 2.0), 2: Rect(2.0, 3.0, 2.0, 2.0)})

    monkeypatch.setattr("floorset_arch.v10_proxy.v10_proxy_cost", lambda _inst, placement, metrics=None: 1.0)

    assert v10_proxy_better(inst, soft_better, current)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
uv run pytest tests/test_v10_proxy.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'floorset_arch.v10_proxy'`.

- [ ] **Step 3: Implement the proxy helper**

Create `src/floorset_arch/v10_proxy.py`:

```python
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from floorset_arch.diagnostics import placement_metrics
from floorset_arch.geometry import bbox
from floorset_arch.models import Instance, Placement
from floorset_arch.repair import soft_violation_counts


@dataclass(frozen=True)
class HardLegality:
    missing_blocks: int
    overlap_count: int
    area_violations: int
    fixed_violations: int
    preplaced_violations: int

    @property
    def rank(self) -> tuple[int, int, int, int, int]:
        return (
            self.missing_blocks,
            self.overlap_count,
            self.area_violations,
            self.fixed_violations,
            self.preplaced_violations,
        )

    @property
    def legal(self) -> bool:
        return self.rank == (0, 0, 0, 0, 0)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def hard_legality(inst: Instance, placement: Placement, metrics: dict[str, float | int] | None = None) -> HardLegality:
    metrics = metrics or placement_metrics(inst, placement)
    missing = sum(1 for block in range(inst.block_count) if block not in placement.rects)
    area_bad = 0
    fixed_bad = 0
    preplaced_bad = 0
    for block in range(inst.block_count):
        rect = placement.rects.get(block)
        if rect is None:
            continue
        target_area = float(inst.area_targets[block])
        if block not in inst.fixed and block not in inst.preplaced:
            rel = abs(rect.area - target_area) / max(target_area, 1.0)
            if rel > 0.010001:
                area_bad += 1
    for block in inst.fixed | inst.preplaced:
        target = inst.target_rects.get(block)
        rect = placement.rects.get(block)
        if target is None or rect is None:
            fixed_bad += int(block in inst.fixed)
            preplaced_bad += int(block in inst.preplaced)
            continue
        if abs(rect.width - target.width) > 1e-6 or abs(rect.height - target.height) > 1e-6:
            if block in inst.fixed:
                fixed_bad += 1
            if block in inst.preplaced:
                preplaced_bad += 1
        if block in inst.preplaced and (abs(rect.x - target.x) > 1e-6 or abs(rect.y - target.y) > 1e-6):
            preplaced_bad += 1
    return HardLegality(
        missing_blocks=missing,
        overlap_count=int(metrics["overlap_count"]),
        area_violations=area_bad,
        fixed_violations=fixed_bad,
        preplaced_violations=preplaced_bad,
    )


def hard_legality_rank(inst: Instance, placement: Placement, metrics: dict[str, float | int] | None = None) -> tuple[int, int, int, int, int]:
    return hard_legality(inst, placement, metrics).rank


def v10_proxy_cost(inst: Instance, placement: Placement, metrics: dict[str, float | int] | None = None) -> float:
    metrics = metrics or placement_metrics(inst, placement)
    rects = placement.rects
    bounds = bbox(list(rects.values()))
    total_area = float(torch.clamp(inst.area_targets[: inst.block_count], min=1.0).sum().item())
    edge_weight = sum(float(w) for *_ij, w in inst.valid_b2b.tolist()) + sum(float(w) for *_ij, w in inst.valid_p2b.tolist())
    hpwl_scale = max(1.0, edge_weight * max(1.0, total_area**0.5))
    area_score = bounds.area / max(total_area, 1.0)
    hpwl_score = float(metrics["hpwl_proxy"]) / hpwl_scale
    boundary, grouping, mib = soft_violation_counts(inst, placement)
    n_soft = max(1, len(inst.boundary))
    n_soft += sum(max(0, len(members) - 1) for members in inst.cluster_groups.values())
    n_soft += sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
    v_rel = (boundary + grouping + mib) / max(n_soft, 1)
    return (1.0 + 0.5 * (hpwl_score + area_score)) * math.exp(2.0 * v_rel)


def _soft_key(inst: Instance, placement: Placement) -> tuple[int, int, int, int]:
    boundary, grouping, mib = soft_violation_counts(inst, placement)
    return (boundary + grouping + mib, grouping, boundary, mib)


def _quality_key(metrics: dict[str, float | int]) -> tuple[float, float]:
    return (float(metrics["hpwl_proxy"]), float(metrics["bbox_area"]))


def v10_proxy_rank(inst: Instance, placement: Placement, metrics: dict[str, float | int] | None = None) -> tuple[Any, ...]:
    metrics = metrics or placement_metrics(inst, placement)
    return (
        hard_legality_rank(inst, placement, metrics),
        v10_proxy_cost(inst, placement, metrics),
        _soft_key(inst, placement),
        _quality_key(metrics),
    )


def v10_proxy_better(
    inst: Instance,
    candidate: Placement,
    current: Placement,
    *,
    candidate_metrics: dict[str, float | int] | None = None,
    current_metrics: dict[str, float | int] | None = None,
    allow_tie_soft: bool = True,
    allow_proxy_regression: bool = False,
) -> bool:
    candidate_metrics = candidate_metrics or placement_metrics(inst, candidate)
    current_metrics = current_metrics or placement_metrics(inst, current)
    if hard_legality_rank(inst, candidate, candidate_metrics) > hard_legality_rank(inst, current, current_metrics):
        return False
    if hard_legality_rank(inst, candidate, candidate_metrics) < hard_legality_rank(inst, current, current_metrics):
        return True
    cand_proxy = v10_proxy_cost(inst, candidate, candidate_metrics)
    cur_proxy = v10_proxy_cost(inst, current, current_metrics)
    tolerance = _env_float("FLOORSET_V10_PROXY_TIE_TOLERANCE", 0.001)
    if cand_proxy < cur_proxy * (1.0 - tolerance):
        return True
    if cand_proxy > cur_proxy * (1.0 + tolerance):
        return False
    if not allow_tie_soft:
        return False
    cand_soft = _soft_key(inst, candidate)
    cur_soft = _soft_key(inst, current)
    if cand_soft < cur_soft:
        return True
    if cand_soft > cur_soft:
        return False
    cand_quality = _quality_key(candidate_metrics)
    cur_quality = _quality_key(current_metrics)
    if cand_quality < cur_quality:
        return True
    if allow_proxy_regression and cand_proxy <= cur_proxy * (1.0 + tolerance):
        return True
    return False
```

- [ ] **Step 4: Run the helper tests and verify they pass**

Run:

```bash
uv run pytest tests/test_v10_proxy.py -v
```

Expected: PASS for all tests in `tests/test_v10_proxy.py`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/floorset_arch/v10_proxy.py tests/test_v10_proxy.py
git commit -m "feat: add v10 proxy acceptance helper"
```

## Task 2: Wire Proxy Ranking Into Optimizer

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Add optimizer failing tests**

Append these tests to `tests/test_optimizer.py`:

```python
def test_candidate_rank_defaults_to_v10_proxy(monkeypatch):
    inst = parse_instance(
        3,
        torch.full((3,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.full((3, 4), -1.0),
    )
    compact_dirty = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    huge_clean = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(200.0, 0.0, 2.0, 2.0), 2: Rect(202.0, 0.0, 2.0, 2.0)})
    monkeypatch.delenv("FLOORSET_CANDIDATE_RANK_POLICY", raising=False)
    optimizer = ArchitectureV4Optimizer()

    assert optimizer._candidate_rank(inst, compact_dirty) < optimizer._candidate_rank(inst, huge_clean)


def test_candidate_rank_can_restore_soft_first(monkeypatch):
    inst = parse_instance(
        3,
        torch.full((3,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.full((3, 4), -1.0),
    )
    compact_dirty = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    huge_clean = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(200.0, 0.0, 2.0, 2.0), 2: Rect(202.0, 0.0, 2.0, 2.0)})
    monkeypatch.setenv("FLOORSET_CANDIDATE_RANK_POLICY", "soft_first")
    optimizer = ArchitectureV4Optimizer()

    assert optimizer._candidate_rank(inst, huge_clean) < optimizer._candidate_rank(inst, compact_dirty)
```

- [ ] **Step 2: Run the optimizer tests and verify default-rank failure**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_candidate_rank_defaults_to_v10_proxy tests/test_optimizer.py::test_candidate_rank_can_restore_soft_first -v
```

Expected: first test FAILS because default rank is still soft-first; second test may pass.

- [ ] **Step 3: Replace optimizer rank and proxy cost calls**

In `src/floorset_arch/optimizer.py`, add this import:

```python
from floorset_arch.v10_proxy import v10_proxy_better, v10_proxy_cost, v10_proxy_rank
```

Replace `_candidate_rank()` with:

```python
    def _candidate_rank(self, inst, placement: Placement) -> tuple:
        metrics = self._placement_metrics(inst, placement)
        if os.environ.get("FLOORSET_CANDIDATE_RANK_POLICY", "v10_proxy") == "soft_first":
            soft = self._soft_total(metrics)
            return (
                int(metrics["overlap_count"]),
                float(soft),
                int(metrics["boundary_violations"]),
                int(metrics["group_violations"]),
                v10_proxy_cost(inst, placement, metrics),
            )
        return v10_proxy_rank(inst, placement, metrics)
```

Replace `_proxy_cost()` with:

```python
    def _proxy_cost(self, inst, placement: Placement) -> float:
        return v10_proxy_cost(inst, placement, placement_metrics(inst, placement))
```

Replace `_no_runtime_proxy_cost()` with a compatibility shim:

```python
    def _no_runtime_proxy_cost(
        self, inst, placement: Placement, metrics: dict[str, float | int]
    ) -> float:
        return v10_proxy_cost(inst, placement, metrics)
```

In `_quality_refine_candidate()`, replace the acceptance block:

```python
                if v10_proxy_better(
                    inst,
                    trial,
                    best,
                    candidate_metrics=metrics,
                    current_metrics=best_metrics,
                    allow_tie_soft=True,
                    allow_proxy_regression=False,
                ):
                    best = trial
                    best_metrics = metrics
                    best_soft = self._soft_total(metrics)
                    best_score = v10_proxy_cost(inst, best, best_metrics)
                    rect = candidate
                    others = [
                        other for idx, other in best.rects.items() if idx != block
                    ]
```

- [ ] **Step 4: Run optimizer tests**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_candidate_rank_defaults_to_v10_proxy tests/test_optimizer.py::test_candidate_rank_can_restore_soft_first -v
```

Expected: PASS.

- [ ] **Step 5: Run proxy and optimizer focused tests**

Run:

```bash
uv run pytest tests/test_v10_proxy.py tests/test_optimizer.py -v
```

Expected: PASS, or only pre-existing dirty-worktree failures unrelated to candidate rank. If unrelated failures appear, capture the failure names before continuing.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/floorset_arch/optimizer.py tests/test_optimizer.py
git commit -m "feat: use v10 proxy for optimizer ranking"
```

## Task 3: Wire Proxy Acceptance Into Repair

**Files:**
- Modify: `src/floorset_arch/repair.py`
- Modify: `tests/test_repair.py`

- [ ] **Step 1: Add repair acceptance failing tests**

Append these tests to `tests/test_repair.py`:

```python
def test_v10_soft_acceptance_rejects_proxy_regression_even_when_soft_improves(monkeypatch):
    inst = parse_instance(
        3,
        torch.full((3,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.full((3, 4), -1.0),
    )
    current = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    huge_clean = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(200.0, 0.0, 2.0, 2.0), 2: Rect(202.0, 0.0, 2.0, 2.0)})

    assert not _score_better_v10_soft(inst, SolverConfig(), huge_clean, current)


def test_v10_soft_acceptance_allows_soft_tie_when_proxy_is_equal(monkeypatch):
    inst = parse_instance(
        3,
        torch.full((3,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.full((3, 4), -1.0),
    )
    current = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(2.0, 0.0, 2.0, 2.0)})
    soft_better = Placement({0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 3.0, 2.0, 2.0), 2: Rect(2.0, 3.0, 2.0, 2.0)})
    monkeypatch.setattr("floorset_arch.v10_proxy.v10_proxy_cost", lambda _inst, placement, metrics=None: 1.0)

    assert _score_better_v10_soft(inst, SolverConfig(), soft_better, current)
```

- [ ] **Step 2: Run repair acceptance tests and verify failure**

Run:

```bash
uv run pytest tests/test_repair.py::test_v10_soft_acceptance_rejects_proxy_regression_even_when_soft_improves tests/test_repair.py::test_v10_soft_acceptance_allows_soft_tie_when_proxy_is_equal -v
```

Expected: at least the regression test FAILS under the old soft-slack acceptance.

- [ ] **Step 3: Replace repair acceptance with proxy helper**

In `src/floorset_arch/repair.py`, add:

```python
from floorset_arch.v10_proxy import v10_proxy_better
```

Replace `_score_better_v10_soft()` with:

```python
def _score_better_v10_soft(
    inst: Instance,
    config: SolverConfig,
    candidate: Placement,
    current: Placement,
) -> bool:
    del config
    try:
        return v10_proxy_better(
            inst,
            candidate,
            current,
            allow_tie_soft=True,
            allow_proxy_regression=False,
        )
    except Exception:
        return False
```

- [ ] **Step 4: Remove automatic hard clamp from v10 soft repair**

In `_v10_soft_repair()`, delete this line:

```python
    config = _runtime_tail_clamped_config(inst, config)
```

Keep `_runtime_tail_clamped_config()` defined for explicit hard-clamp ablation.

- [ ] **Step 5: Run repair tests**

Run:

```bash
uv run pytest tests/test_repair.py::test_v10_soft_acceptance_rejects_proxy_regression_even_when_soft_improves tests/test_repair.py::test_v10_soft_acceptance_allows_soft_tie_when_proxy_is_equal -v
```

Expected: PASS.

- [ ] **Step 6: Run repair and proxy focused tests**

Run:

```bash
uv run pytest tests/test_v10_proxy.py tests/test_repair.py -v
```

Expected: PASS, or only pre-existing dirty-worktree failures unrelated to v10 proxy acceptance. Record unrelated failure names.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/floorset_arch/repair.py tests/test_repair.py
git commit -m "feat: use v10 proxy for repair acceptance"
```

## Task 4: Conditional Runtime Budget

**Files:**
- Modify: `src/floorset_arch/budget_layer.py`
- Modify: `src/floorset_arch/repair.py`
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `tests/test_budget_layer.py`
- Modify: `tests/test_repair.py`

- [ ] **Step 1: Add budget-layer tests**

Append these tests to `tests/test_budget_layer.py`:

```python
def test_conditional_runtime_budget_defaults_off(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET", raising=False)

    from floorset_arch.budget_layer import conditional_runtime_budget_enabled

    assert not conditional_runtime_budget_enabled()


def test_conditional_runtime_limits_are_tier_specific(monkeypatch):
    from floorset_arch.budget_layer import conditional_runtime_budget_limits

    monkeypatch.setenv("FLOORSET_CONDITIONAL_RUNTIME_LIGHT_ATTEMPTS", "1")
    monkeypatch.setenv("FLOORSET_CONDITIONAL_RUNTIME_MEDIUM_ATTEMPTS", "2")
    monkeypatch.setenv("FLOORSET_CONDITIONAL_RUNTIME_HEAVY_ATTEMPTS", "3")

    assert conditional_runtime_budget_limits(BudgetTier.LIGHT).max_rejected_attempts == 1
    assert conditional_runtime_budget_limits(BudgetTier.MEDIUM).max_rejected_attempts == 2
    assert conditional_runtime_budget_limits(BudgetTier.HEAVY).max_rejected_attempts == 3
```

- [ ] **Step 2: Run budget-layer tests and verify failure**

Run:

```bash
uv run pytest tests/test_budget_layer.py::test_conditional_runtime_budget_defaults_off tests/test_budget_layer.py::test_conditional_runtime_limits_are_tier_specific -v
```

Expected: FAIL with missing `conditional_runtime_budget_enabled` or `conditional_runtime_budget_limits`.

- [ ] **Step 3: Implement conditional runtime budget helpers**

In `src/floorset_arch/budget_layer.py`, add:

```python
@dataclass(frozen=True)
class ConditionalRuntimeLimits:
    max_rejected_attempts: int
    max_elapsed_ms: int


def conditional_runtime_budget_enabled() -> bool:
    return env_flag("FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET")


def conditional_runtime_budget_limits(tier: BudgetTier) -> ConditionalRuntimeLimits:
    if tier is BudgetTier.HEAVY:
        attempts_default = 3
        elapsed_default = 4000
        prefix = "HEAVY"
    elif tier is BudgetTier.MEDIUM:
        attempts_default = 2
        elapsed_default = 2500
        prefix = "MEDIUM"
    else:
        attempts_default = 1
        elapsed_default = 1200
        prefix = "LIGHT"
    return ConditionalRuntimeLimits(
        max_rejected_attempts=env_int(f"FLOORSET_CONDITIONAL_RUNTIME_{prefix}_ATTEMPTS", attempts_default),
        max_elapsed_ms=env_int(f"FLOORSET_CONDITIONAL_RUNTIME_{prefix}_ELAPSED_MS", elapsed_default),
    )
```

- [ ] **Step 4: Run budget-layer tests**

Run:

```bash
uv run pytest tests/test_budget_layer.py::test_conditional_runtime_budget_defaults_off tests/test_budget_layer.py::test_conditional_runtime_limits_are_tier_specific -v
```

Expected: PASS.

- [ ] **Step 5: Add repair trace test for conditional budget fields**

Append this test to `tests/test_repair.py`:

```python
def test_v10_soft_repair_trace_reports_conditional_runtime_budget(tmp_path, monkeypatch):
    inst = _soft_test_instance()
    placement = Placement({0: Rect(4.0, 0.0, 2.0, 2.0), 1: Rect(0.0, 0.0, 2.0, 2.0), 2: Rect(8.0, 0.0, 2.0, 2.0), 3: Rect(12.0, 0.0, 2.0, 2.0)})
    trace = tmp_path / "repair.jsonl"
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")
    monkeypatch.setenv("FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET", "1")
    monkeypatch.setenv("FLOORSET_REPAIR_TRACE_JSONL", str(trace))

    repaired = repair_placement(inst, placement, SolverConfig(max_repair_passes=2))

    assert repaired.rects
    rows = [line for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert '"runtime_budget"' in rows[-1]
    assert '"attempts"' in rows[-1]
    assert '"accepted"' in rows[-1]
    assert '"elapsed_ms"' in rows[-1]
    assert '"stop_reason"' in rows[-1]
```

- [ ] **Step 6: Run the new repair trace test and verify failure**

Run:

```bash
uv run pytest tests/test_repair.py::test_v10_soft_repair_trace_reports_conditional_runtime_budget -v
```

Expected: FAIL because repair trace rows do not include `runtime_budget`.

- [ ] **Step 7: Add conditional runtime stats in repair**

In `src/floorset_arch/repair.py`, add imports:

```python
import time
from floorset_arch.budget_layer import (
    conditional_runtime_budget_enabled,
    conditional_runtime_budget_limits,
)
from floorset_arch.risk_budget import instance_risk_budget
```

Add a small stats helper near `_v10_soft_repair()`:

```python
def _runtime_budget_trace(extra_path: str, attempts: int, accepted: int, elapsed_ms: float, stop_reason: str, tier: str) -> dict[str, object]:
    return {
        "extra_path": extra_path,
        "attempts": attempts,
        "accepted": accepted,
        "elapsed_ms": round(float(elapsed_ms), 3),
        "stop_reason": stop_reason,
        "tier": tier,
    }
```

Modify `_v10_soft_repair()` to track attempts:

```python
    attempts = 0
    accepted = 0
    stop_reason = "max_passes"
    started = time.perf_counter()
    try:
        budget = instance_risk_budget(inst)
        tier = budget.tier
    except Exception:
        tier = BudgetTier.LIGHT
    limits = conditional_runtime_budget_limits(tier)
    rejected = 0
    max_passes = max(1, min(config.max_repair_passes, 3))
    for _ in range(max_passes):
        if conditional_runtime_budget_enabled():
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms >= limits.max_elapsed_ms:
                stop_reason = "elapsed_budget"
                break
            if rejected >= limits.max_rejected_attempts:
                stop_reason = "rejected_attempts"
                break
        attempts += 1
        trial = best.copy()
        _repair_boundary(inst, trial)
        _connect_clusters(inst, trial, config)
        _resolve_overlaps(inst, trial, config)
        _snap_boundary_components(inst, trial, config)
        _repair_boundary(inst, trial)
        _snap_hard(inst, trial)
        if _score_better_v10_soft(inst, config, trial, best):
            best = trial
            accepted += 1
            rejected = 0
        else:
            rejected += 1
            if not conditional_runtime_budget_enabled():
                stop_reason = "rejected"
                break
    best.runtime_budget_trace = _runtime_budget_trace(
        "v10_soft_repair",
        attempts,
        accepted,
        (time.perf_counter() - started) * 1000.0,
        stop_reason,
        str(tier.value if isinstance(tier, BudgetTier) else tier),
    )
```

Because `Placement` is a dataclass without slots, attaching `runtime_budget_trace` at runtime is acceptable for trace-only metadata.

- [ ] **Step 8: Add trace propagation in optimizer**

In `src/floorset_arch/optimizer.py`, inside `_trace_repair()`, add:

```python
        runtime_budget = getattr(after, "runtime_budget_trace", None)
        if runtime_budget is not None:
            row["runtime_budget"] = runtime_budget
```

- [ ] **Step 9: Run budget and repair trace tests**

Run:

```bash
uv run pytest tests/test_budget_layer.py::test_conditional_runtime_budget_defaults_off tests/test_budget_layer.py::test_conditional_runtime_limits_are_tier_specific tests/test_repair.py::test_v10_soft_repair_trace_reports_conditional_runtime_budget -v
```

Expected: PASS.

- [ ] **Step 10: Commit Task 4**

```bash
git add src/floorset_arch/budget_layer.py src/floorset_arch/repair.py src/floorset_arch/optimizer.py tests/test_budget_layer.py tests/test_repair.py
git commit -m "feat: add conditional runtime budget tracing"
```

## Task 5: Narrow Grouping Pair Bias

**Files:**
- Modify: `src/floorset_arch/relative_order.py`
- Modify: `tests/test_relative_order.py`

- [ ] **Step 1: Add narrow grouping pair bias tests**

Append these tests to `tests/test_relative_order.py`:

```python
def test_narrow_grouping_pair_bias_does_not_enable_global_key_blend(monkeypatch):
    constraints = torch.zeros(3, 5)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
    constraints[2, 3] = 1.0
    inst = parse_instance(
        3,
        torch.full((3,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((3, 4), -1.0),
    )
    inst.anchor_guidance = AnchorGuidance(rect_priors={0: Rect(0.0, 0.0, 2.0, 2.0), 1: Rect(30.0, 0.0, 2.0, 2.0), 2: Rect(60.0, 0.0, 2.0, 2.0)})
    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "1")
    monkeypatch.setenv("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "0")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert placement.rects[2].x >= placement.rects[1].right


def test_narrow_grouping_pair_bias_only_applies_to_ambiguous_same_cluster_pairs(monkeypatch):
    constraints = torch.zeros(4, 5)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
    constraints[2, 3] = 2.0
    constraints[3, 3] = 2.0
    inst = parse_instance(
        4,
        torch.full((4,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((4, 4), -1.0),
    )
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.1, 2.0, 2.0, 2.0),
            2: Rect(20.0, 0.0, 2.0, 2.0),
            3: Rect(80.0, 20.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "1")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_MIN_MEMBERS", "2")
    monkeypatch.setenv("FLOORSET_NARROW_GROUPING_AMBIGUITY_MARGIN", "0.30")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert edge_touch_length(placement.rects[0], placement.rects[1]) > 0.0
    assert edge_touch_length(placement.rects[2], placement.rects[3]) == 0.0
```

- [ ] **Step 2: Run relative-order tests and verify failure**

Run:

```bash
uv run pytest tests/test_relative_order.py::test_narrow_grouping_pair_bias_does_not_enable_global_key_blend tests/test_relative_order.py::test_narrow_grouping_pair_bias_only_applies_to_ambiguous_same_cluster_pairs -v
```

Expected: FAIL because narrow grouping flag is not implemented.

- [ ] **Step 3: Implement narrow grouping helpers**

In `src/floorset_arch/relative_order.py`, add:

```python
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _narrow_grouping_pair_bias_enabled() -> bool:
    return os.environ.get("FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS", "").strip().lower() in {"1", "true", "yes", "on"}


def _cluster_grouping_pressure(inst: Instance, cluster: int, members: list[int], raw_x: list[float], raw_y: list[float], widths: list[float], heights: list[float]) -> bool:
    if len(members) < _env_int("FLOORSET_NARROW_GROUPING_MIN_MEMBERS", 3):
        return False
    span_x = max(raw_x[i] for i in members) - min(raw_x[i] for i in members)
    span_y = max(raw_y[i] for i in members) - min(raw_y[i] for i in members)
    avg_w = sum(widths[i] for i in members) / max(len(members), 1)
    avg_h = sum(heights[i] for i in members) / max(len(members), 1)
    pressure = max(span_x / max(avg_w, 1e-6), span_y / max(avg_h, 1e-6))
    return pressure >= _env_float("FLOORSET_NARROW_GROUPING_PRESSURE", 1.4)
```

- [ ] **Step 4: Disable broad key blend unless broad flag is explicitly on**

In `_bias_order_keys()`, keep compact key blend only under the old broad flag:

```python
    if profile == "compact" and _grouping_adjacency_bias_enabled(profile) and not _narrow_grouping_pair_bias_enabled():
        blend = max(
            0.0,
            min(
                1.0,
                float(os.environ.get("FLOORSET_GROUPING_ADJACENCY_KEY_BLEND", "0.65")),
            ),
        )
```

Keep the existing `else` unchanged:

```python
    else:
        blend = 0.0 if profile == "compact" else 0.18
```

- [ ] **Step 5: Add narrow pair bonus inside the pair loop**

Before the pair loop in `construct_relative_order_placement()`, build pressure state:

```python
    cluster_pressure = {
        cluster: _cluster_grouping_pressure(inst, cluster, members, raw_x, raw_y, widths, heights)
        for cluster, members in inst.cluster_groups.items()
    }
    narrow_bonus = _env_float("FLOORSET_NARROW_GROUPING_AXIS_BONUS", 0.12)
    ambiguity_margin = _env_float("FLOORSET_NARROW_GROUPING_AMBIGUITY_MARGIN", 0.15)
```

Inside the existing same-cluster `if ci and ci == cj:` block, add the narrow logic before the broad `_grouping_adjacency_bias_enabled(profile)` bonus:

```python
                if _narrow_grouping_pair_bias_enabled() and cluster_pressure.get(ci, False) and abs(h_score - v_score) <= ambiguity_margin:
                    if orient.get(ci, "H") == "H":
                        h_score += narrow_bonus
                    else:
                        v_score += narrow_bonus
                elif orient.get(ci, "H") == "H":
                    h_score += 0.45 if _grouping_adjacency_bias_enabled(profile) else 0.0
                else:
                    v_score += 0.45 if _grouping_adjacency_bias_enabled(profile) else 0.0
```

Remove the old nested `if orient.get(...` block that this replaces.

- [ ] **Step 6: Prevent whole-group chaining in narrow mode**

Change the chain block condition:

```python
    if _grouping_adjacency_bias_enabled(profile) and not _narrow_grouping_pair_bias_enabled():
```

- [ ] **Step 7: Run relative-order tests**

Run:

```bash
uv run pytest tests/test_relative_order.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/floorset_arch/relative_order.py tests/test_relative_order.py
git commit -m "feat: add narrow grouping pair bias"
```

## Task 6: Integration Checks and Documentation Hygiene

**Files:**
- Modify: `CONTEXT.md` only if the executor decides to keep the glossary updates from the design session in the implementation branch.
- Modify: `graphify-out/*` only through `graphify update .`.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/test_v10_proxy.py tests/test_optimizer.py tests/test_repair.py tests/test_budget_layer.py tests/test_relative_order.py -v
```

Expected: PASS. If failures are unrelated to touched code and existed before execution, record exact test names and failure summaries in the task notes.

- [ ] **Step 2: Run broader unit tests for changed areas**

Run:

```bash
uv run pytest tests/test_model.py tests/test_checkpoint_promotion.py tests/test_optimizer.py tests/test_repair.py tests/test_relative_order.py tests/test_budget_layer.py -v
```

Expected: PASS, or documented pre-existing dirty-worktree failures.

- [ ] **Step 3: Run static diff check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Update Graphify index**

Run:

```bash
graphify update .
```

Expected: graph update completes without fatal errors.

- [ ] **Step 5: Inspect final worktree scope**

Run:

```bash
git status --short
```

Expected: only files touched by this plan and expected `graphify-out` update artifacts are modified. Do not revert unrelated dirty files that were present before execution.

- [ ] **Step 6: Commit integration cleanup**

If `CONTEXT.md` glossary changes are intentionally included, commit them with:

```bash
git add CONTEXT.md graphify-out
git commit -m "docs: update v10 proxy budget terminology"
```

If `CONTEXT.md` is left dirty because it contains pre-existing user changes, skip this commit and note that the glossary changes remain unstaged.

- [ ] **Step 7: Final validation command for handoff**

Run:

```bash
uv run pytest tests/test_v10_proxy.py tests/test_optimizer.py tests/test_repair.py tests/test_budget_layer.py tests/test_relative_order.py -v
```

Expected: PASS.

## Execution Notes

- The current branch is `v10-budget-proxy-runtime-grouping`.
- The worktree had dirty files before this plan was written. Executors must inspect `git status --short` before each commit and stage only files from the current task.
- Do not modify GNN architecture, checkpoint files, training checkpoint promotion, or broad quality portfolio defaults.
- Do not use validation IDs in production code.
- Keep Phase 2 and Phase 3 behind opt-in flags.

## Self-Review

- Spec coverage: Phase 1 is covered by Tasks 1-3; Phase 2 is covered by Task 4; Phase 3 is covered by Task 5; validation and graph update are covered by Task 6.
- Placeholder scan: no deferred implementation markers are used in task steps.
- Type consistency: `Placement`, `Rect`, `Instance`, `SolverConfig`, `BudgetTier`, and helper names match the existing codebase or are introduced before first use.
