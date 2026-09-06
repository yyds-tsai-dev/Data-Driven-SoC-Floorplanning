# Partner Candidate Source Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a behavior-preserving candidate ranking and fixed-slot allocation boundary from `src/solver/my_opt_claude.py` so Direct-v2, retrieval, and Flow Matching can be compared without changing total runtime capacity.

**Architecture:** A small `candidate_supply_claude.py` module owns source-neutral candidate records, deterministic quota allocation, and the existing HPWL/overlap prescreen. `MyOptimizer` remains the orchestrator and keeps the current Direct-v2 behavior byte-for-byte when no new source is enabled.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, existing partner evaluator.

## Global Constraints

- `partner/` is the primary pipeline; do not change `src/` behavior.
- Bare defaults must reproduce the current Direct-v2 candidate path.
- Candidate-source changes must not increase the per-case deadline, GPU batch cap, prescreen count, worker count, or refine slots.
- No validation/evaluation solution may enter a production candidate source.
- Preserve existing `PARTNER_OVERSAMPLE`, `PARTNER_KS_CAP`, `PARTNER_PRESCREEN_V`, and `PARTNER_PRESCREEN_VW` semantics.

---

## File Structure

- Create `src/solver/candidate_supply_claude.py`: source-neutral candidate records, quota allocation, and prescreen ranking.
- Modify `src/solver/my_opt_claude.py`: call the extracted functions without changing default behavior.
- Modify `tests/conftest.py`: make standalone `partner/` modules importable in tests.
- Create `tests/test_partner_candidate_supply.py`: unit and parity tests.

### Task 1: Candidate Records and Fixed-Slot Allocation

**Files:**
- Create: `src/solver/candidate_supply_claude.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_partner_candidate_supply.py`

**Interfaces:**
- Produces: `CandidateBatch(source: str, predictions: list[np.ndarray], generation_s: float, metadata: dict[str, object])`.
- Produces: `allocate_quotas(total: int, requested: dict[str, int], enabled: Sequence[str]) -> dict[str, int]`.
- Contract: returned quotas are non-negative, sum to at most `total`, and never silently create extra slots.

- [ ] **Step 1: Add the partner import path and failing allocation tests**

Add to `tests/conftest.py` after `SRC` setup:

```python
PARTNER = ROOT / "partner"
if str(PARTNER) not in sys.path:
    sys.path.insert(0, str(PARTNER))
```

Create `tests/test_partner_candidate_supply.py`:

```python
import numpy as np

from candidate_supply_claude import CandidateBatch, allocate_quotas


def test_candidate_batch_rejects_non_rectangles():
    bad = np.zeros((3, 3), dtype=np.float64)
    try:
        CandidateBatch("direct", [bad], 0.01, {})
    except ValueError as exc:
        assert "[N,4]" in str(exc)
    else:
        raise AssertionError("invalid candidate was accepted")


def test_allocate_quotas_never_expands_total_capacity():
    got = allocate_quotas(
        total=12,
        requested={"direct": 6, "retrieval": 3, "flow": 3},
        enabled=("direct", "retrieval", "flow"),
    )
    assert got == {"direct": 6, "retrieval": 3, "flow": 3}
    assert sum(got.values()) == 12


def test_allocate_quotas_preserves_direct_only_baseline():
    assert allocate_quotas(12, {"direct": 12}, ("direct",)) == {"direct": 12}
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `uv run pytest tests/test_partner_candidate_supply.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'candidate_supply_claude'`.

- [ ] **Step 3: Implement the records and allocator**

Create `src/solver/candidate_supply_claude.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateBatch:
    source: str
    predictions: list[np.ndarray]
    generation_s: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("candidate source must be non-empty")
        if self.generation_s < 0:
            raise ValueError("generation_s must be non-negative")
        for prediction in self.predictions:
            if prediction.ndim != 2 or prediction.shape[1] != 4:
                raise ValueError("candidate prediction must have shape [N,4]")
            if not np.isfinite(prediction).all():
                raise ValueError("candidate prediction must be finite")


def allocate_quotas(
    total: int,
    requested: Mapping[str, int],
    enabled: Sequence[str],
) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    names = tuple(name for name in enabled if requested.get(name, 0) > 0)
    if not names or total == 0:
        return {}
    weights = {name: int(requested[name]) for name in names}
    weight_sum = sum(weights.values())
    raw = {name: total * weights[name] / weight_sum for name in names}
    result = {name: int(raw[name]) for name in names}
    left = total - sum(result.values())
    order = sorted(names, key=lambda name: (-(raw[name] - result[name]), names.index(name)))
    for name in order[:left]:
        result[name] += 1
    return result
```

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/test_partner_candidate_supply.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the contracts**

```bash
git add src/solver/candidate_supply_claude.py tests/conftest.py tests/test_partner_candidate_supply.py
git commit -m "refactor: add partner candidate source contracts"
```

### Task 2: Extract Deterministic Prescreen Ranking

**Files:**
- Modify: `src/solver/candidate_supply_claude.py`
- Modify: `tests/test_partner_candidate_supply.py`

**Interfaces:**
- Produces: `rank_predictions(predictions, area_targets, b2b, constraint_penalties=None, violation_weight=0.0) -> list[int]`.
- Consumes: rectangle arrays in `[x, y, w, h]` order.
- Contract: direct-only ordering must equal the current HPWL-normalized plus `5 * overlap_fraction` formula.

- [ ] **Step 1: Write failing score-order tests**

Append:

```python
from candidate_supply_claude import rank_predictions


def test_rank_predictions_penalizes_overlap_after_hpwl_normalization():
    separated = np.array([[0, 0, 2, 2], [2, 0, 2, 2]], dtype=np.float64)
    overlapping = np.array([[0, 0, 2, 2], [1, 1, 2, 2]], dtype=np.float64)
    area = np.array([4.0, 4.0])
    b2b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert rank_predictions([overlapping, separated], area, b2b) == [1, 0]


def test_rank_predictions_accepts_source_neutral_constraint_penalties():
    p0 = np.array([[0, 0, 1, 1]], dtype=np.float64)
    p1 = np.array([[0, 0, 1, 1]], dtype=np.float64)
    order = rank_predictions(
        [p0, p1], np.array([1.0]), np.zeros((1, 1)),
        constraint_penalties=[2.0, 0.0], violation_weight=0.5,
    )
    assert order == [1, 0]
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_partner_candidate_supply.py -q`

Expected: FAIL importing `rank_predictions`.

- [ ] **Step 3: Implement source-neutral ranking**

Append to `src/solver/candidate_supply_claude.py`:

```python
def _hpwl_proxy(prediction: np.ndarray, b2b: np.ndarray) -> float:
    n = prediction.shape[0]
    centers_x = prediction[:, 0] + 0.5 * prediction[:, 2]
    centers_y = prediction[:, 1] + 0.5 * prediction[:, 3]
    i, j = np.nonzero(np.triu(b2b[:n, :n], 1))
    if len(i) == 0:
        return 0.0
    return float((b2b[i, j] * (
        np.abs(centers_x[i] - centers_x[j])
        + np.abs(centers_y[i] - centers_y[j])
    )).sum())


def _overlap_fraction(prediction: np.ndarray, area_targets: np.ndarray) -> float:
    n = prediction.shape[0]
    x0, y0 = prediction[:, 0], prediction[:, 1]
    x1 = x0 + prediction[:, 2]
    y1 = y0 + prediction[:, 3]
    overlap_x = np.maximum(
        0.0, np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    )
    overlap_y = np.maximum(
        0.0, np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    )
    overlap = overlap_x * overlap_y
    overlap[np.diag_indices(n)] = 0.0
    return float(overlap.sum()) / (2.0 * max(float(area_targets[:n].sum()), 1e-9))


def rank_predictions(
    predictions: Sequence[np.ndarray],
    area_targets: np.ndarray,
    b2b: np.ndarray,
    constraint_penalties: Sequence[float] | None = None,
    violation_weight: float = 0.0,
) -> list[int]:
    if not predictions:
        return []
    hpwl = [_hpwl_proxy(prediction, b2b) for prediction in predictions]
    hpwl_ref = max(min(hpwl), 1e-9)
    penalties = constraint_penalties or [0.0] * len(predictions)
    if len(penalties) != len(predictions):
        raise ValueError("constraint penalty count must match predictions")
    score = [
        hpwl[k] / hpwl_ref
        + 5.0 * _overlap_fraction(predictions[k], area_targets)
        + violation_weight * float(penalties[k])
        for k in range(len(predictions))
    ]
    return sorted(range(len(predictions)), key=lambda k: (score[k], k))
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_partner_candidate_supply.py -q`

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/solver/candidate_supply_claude.py tests/test_partner_candidate_supply.py
git commit -m "refactor: extract partner candidate prescreen"
```

### Task 3: Wire the Boundary into the Direct-v2 Baseline

**Files:**
- Modify: `src/solver/my_opt_claude.py:311-445`
- Modify: `tests/test_partner_candidate_supply.py`

**Interfaces:**
- Consumes: `rank_predictions()` from Task 2.
- Produces: unchanged `_sample_direct_preds(self, n, at, cons, tpos, b2b, p2b, pins, K, oversample=True) -> list[np.ndarray]`.

- [ ] **Step 1: Add a parity test using the extracted ranker**

Append a test with three deterministic layouts and assert the rank order equals a frozen expected list:

```python
def test_direct_baseline_frozen_rank_order():
    predictions = [
        np.array([[0, 0, 2, 2], [1, 1, 2, 2]], dtype=np.float64),
        np.array([[0, 0, 2, 2], [2, 0, 2, 2]], dtype=np.float64),
        np.array([[0, 0, 2, 2], [8, 0, 2, 2]], dtype=np.float64),
    ]
    area = np.array([4.0, 4.0])
    b2b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert rank_predictions(predictions, area, b2b) == [1, 0, 2]
```

- [ ] **Step 2: Run the test before refactoring**

Run: `uv run pytest tests/test_partner_candidate_supply.py -q`

Expected: PASS; this freezes the intended ordering before integration.

- [ ] **Step 3: Replace only the inline HPWL/overlap ordering**

At the top of `src/solver/my_opt_claude.py`, import:

```python
from candidate_supply_claude import rank_predictions
```

Inside `_sample_direct_preds`, keep prediction generation and the existing constraint-aware `_viol_est` calculation. Replace the local `hps`, `hp_min`, `raw`, and final `sorted(range(len(preds)), key=lambda k: raw[k])` block with:

```python
penalties = [_viol_est(prediction) for prediction in preds] if os.environ.get(
    "PARTNER_PRESCREEN_V"
) else None
order = rank_predictions(
    preds,
    at[:n].detach().cpu().numpy(),
    w_b2b if w_b2b is not None else np.zeros((n, n), dtype=np.float64),
    constraint_penalties=penalties,
    violation_weight=_env_float("PARTNER_PRESCREEN_VW", 0.5) if penalties is not None else 0.0,
)
return [preds[k] for k in order]
```

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_partner_candidate_supply.py -q`

Expected: all focused tests PASS.

Run: `uv run pytest`

Expected: full suite PASS.

- [ ] **Step 5: Run behavior smoke**

Run: `bash scripts/validate.sh`

Expected: submission-interface validation succeeds with 100% feasible smoke output.

- [ ] **Step 6: Commit**

```bash
git add src/solver/my_opt_claude.py src/solver/candidate_supply_claude.py tests/test_partner_candidate_supply.py
git commit -m "refactor: route direct candidates through shared prescreen"
```

## Completion Gate

Before starting retrieval or Flow Matching work:

1. Full pytest and `scripts/validate.sh` pass.
2. Bare-default Direct-v2 full-100 no-runtime result remains inside the established stochastic repeat band around `1.1162`.
3. Average and maximum runtime do not increase beyond run-to-run noise.
4. `git diff` contains no `src/` solver behavior change.

If the baseline changes materially, fix the extraction before adding any candidate source.
