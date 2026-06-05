# v10 Evaluator Selection Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make evaluator tests target the repository-maintained scripts evaluator, verify v10 scoring behavior, and recalibrate selection guidance before algorithm changes.

**Architecture:** `tests/test_evaluator_scoring.py` directly loads `scripts/iccad2026_evaluate.py` with `importlib.util.spec_from_file_location`, while temporarily adding `FloorSet/` to `sys.path` so official helper modules resolve. Scoring tests assert v10 feasible caps and `exp(n/12)` weighting. Selection recalibration updates docs and, only if needed, training-selection tests to use full v10 no-runtime score before tail-only evidence.

**Tech Stack:** Python 3.12, `uv run`, pytest, local FloorSet helper modules, repository docs.

---

## File Structure

- Modify: `tests/test_evaluator_scoring.py`
  - Responsibility: unit tests for the local scripts evaluator API and v10 score behavior.
- Modify: `scripts/iccad2026_evaluate.py`
  - Responsibility: local no-runtime diagnostics evaluator, already patched to v10 scoring. This plan only adds import-path robustness if tests expose it as necessary.
- Modify: `docs/optimization-notes.md`
  - Responsibility: human-readable scoring and selection guidance; remove stale v9 tail dominance claims.
- Modify: `docs/evaluation/2026-06-05-v10-recalibration.md`
  - Responsibility: capture the recalibration command output and decision notes. Create it if a full or partial recalibration run is performed.
- Create: `src/floorset_arch/risk_budget.py`
  - Responsibility: shared v10-aware risk and budget calculation for high-risk repair and quality portfolio gating.
- Create: `tests/test_risk_budget.py`
  - Responsibility: unit tests for v10 score share, constraint density, net density, and budget tier assignment.
- Modify: `src/floorset_arch/optimizer.py:420-509`
  - Responsibility: consume shared risk budget for high-risk candidate gating instead of relying on block-count-heavy thresholds alone.
- Modify: `src/floorset_arch/quality_portfolio.py:24-48`
  - Responsibility: consume shared risk budget for quality portfolio gating.
- Modify: `tests/test_model.py`
  - Responsibility: checkpoint-selection behavior tests, only if selection policy needs explicit v10 evidence checks beyond existing tests.
- Modify: `src/floorset_arch/training/selection.py`
  - Responsibility: checkpoint metric ordering, only if tests show the current ordering is insufficient.

Do not move `FloorSet` files into `src`. Do not copy `litetestLoader.py`, `liteLoader.py`, `lite_dataset.py`, `cost.py`, or `utils.py` into a new package.

## Task 1: Point Evaluator Tests At The Scripts Evaluator

**Files:**
- Modify: `tests/test_evaluator_scoring.py:1-13`

- [ ] **Step 1: Replace the existing import bootstrap with a direct file loader**

Replace the current top-of-file import bootstrap:

```python
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

CONTEST_DIR = Path(__file__).resolve().parents[1] / "FloorSet" / "iccad2026contest"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

import iccad2026_evaluate as evaluator  # noqa: E402
```

with:

```python
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
FLOORSET_ROOT = ROOT / "FloorSet"
SCRIPTS_EVALUATOR = ROOT / "scripts" / "iccad2026_evaluate.py"


def _load_scripts_evaluator():
    if str(FLOORSET_ROOT) not in sys.path:
        sys.path.insert(0, str(FLOORSET_ROOT))
    spec = importlib.util.spec_from_file_location(
        "scripts_iccad2026_evaluate",
        SCRIPTS_EVALUATOR,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_scripts_evaluator()
```

- [ ] **Step 2: Run the focused tests and verify the import failure is gone**

Run:

```bash
uv run pytest -q tests/test_evaluator_scoring.py
```

Expected: tests no longer fail with errors such as:

```text
compute_cost() got an unexpected keyword argument 'use_runtime'
module 'iccad2026_evaluate' has no attribute 'compute_cost_breakdown'
TestResult.__init__() got an unexpected keyword argument 'cost_no_runtime'
```

Some assertions may still fail because they encode v9 expectations. Those failures are handled in Task 2.

- [ ] **Step 3: Commit the import-target fix**

```bash
git add tests/test_evaluator_scoring.py
git commit -m "test: load scripts evaluator directly"
```

## Task 2: Update Evaluator Tests For v10 Scoring

**Files:**
- Modify: `tests/test_evaluator_scoring.py:16-70`

- [ ] **Step 1: Update the runtime-disabled cost test for feasible cap behavior**

Change `test_compute_cost_can_disable_runtime_adjustment` to:

```python
def test_compute_cost_can_disable_runtime_adjustment():
    with_runtime = evaluator.compute_cost(0.1, 0.2, 0.25, 4.0, True)
    without_runtime = evaluator.compute_cost(
        0.1,
        0.2,
        0.25,
        4.0,
        True,
        use_runtime=False,
    )

    expected_quality = 1 + evaluator.ALPHA * (0.1 + 0.2)
    expected_violation = math.exp(evaluator.BETA * 0.25)

    assert with_runtime > without_runtime
    assert without_runtime == expected_quality * expected_violation
```

This test remains mostly unchanged because the formula result is below the cap.

- [ ] **Step 2: Add a feasible-cap test**

Add this test after `test_compute_cost_breakdown_exposes_formula_factors`:

```python
def test_feasible_cost_is_capped_below_infeasible_penalty():
    cost = evaluator.compute_cost(100.0, 100.0, 1.0, 100.0, True)

    assert cost == evaluator.M_PENALTY - 1e-6
    assert cost < evaluator.compute_cost(0.0, 0.0, 0.0, 1.0, False)
```

- [ ] **Step 3: Update total-score test to assert v10 `exp(n/12)` math**

Replace `test_no_runtime_total_uses_no_runtime_costs` with:

```python
def test_no_runtime_total_uses_v10_exp_n_over_12_weighting():
    results = [
        evaluator.TestResult(
            0,
            21,
            True,
            0.0,
            0.0,
            0.0,
            1.0,
            cost=2.0,
            cost_no_runtime=1.0,
        ),
        evaluator.TestResult(
            1,
            120,
            True,
            0.0,
            0.0,
            0.0,
            8.0,
            cost=8.0,
            cost_no_runtime=3.0,
        ),
    ]

    total = evaluator.compute_total_score(
        [r.cost_no_runtime for r in results],
        [r.block_count for r in results],
    )
    weights = [math.exp((21 - 120) / 12), math.exp((120 - 120) / 12)]
    expected = (1.0 * weights[0] + 3.0 * weights[1]) / sum(weights)

    assert total == expected
    assert total < 3.0
```

- [ ] **Step 4: Update score contributor expectation**

Replace the old two-case score-contributor expectation with a full 21-120
weighting check:

```python
results = [
    evaluator.TestResult(
        idx,
        block_count,
        True,
        0.0,
        0.0,
        0.0,
        1.0,
        cost=1.0,
        cost_no_runtime=1.0,
    )
    for idx, block_count in enumerate(range(21, 121))
]
contributors = evaluator.summarize_score_contributors(results, use_runtime=False)
expected_weight_120 = math.exp(0.0) / sum(
    math.exp((n - 120) / 12) for n in range(21, 121)
)

assert contributors[0]["block_count"] == 120
assert contributors[0]["score_weight"] == expected_weight_120
assert 0.079 < contributors[0]["score_weight"] < 0.081
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest -q tests/test_evaluator_scoring.py
```

Expected: all tests in `tests/test_evaluator_scoring.py` pass.

- [ ] **Step 6: Commit v10 scoring test updates**

```bash
git add tests/test_evaluator_scoring.py
git commit -m "test: assert v10 evaluator scoring"
```

## Task 3: Make Scripts Evaluator Import Robust When Loaded From Tests

**Files:**
- Modify: `scripts/iccad2026_evaluate.py:46-57`
- Test: `tests/test_evaluator_scoring.py`

- [ ] **Step 1: Run the direct import smoke**

Run:

```bash
uv run python - <<'PY'
import importlib.util
import sys
from pathlib import Path

root = Path.cwd()
floorset = root / "FloorSet"
target = root / "scripts" / "iccad2026_evaluate.py"
sys.path.insert(0, str(floorset))
spec = importlib.util.spec_from_file_location("scripts_iccad2026_evaluate", target)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module.compute_cost(100, 100, 1, 100, True))
PY
```

Expected:

```text
9.999999
```

- [ ] **Step 2: If the smoke fails without manual `FloorSet` path insertion, add explicit helper path setup**

Only if needed, replace this block in `scripts/iccad2026_evaluate.py`:

```python
# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
```

with:

```python
ROOT_DIR = Path(__file__).resolve().parents[1]
FLOORSET_DIR = ROOT_DIR / "FloorSet"

for import_path in (ROOT_DIR, FLOORSET_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
```

- [ ] **Step 3: Run compile and focused tests**

Run:

```bash
uv run python -m py_compile scripts/iccad2026_evaluate.py
uv run pytest -q tests/test_evaluator_scoring.py
```

Expected: compile succeeds and focused tests pass.

- [ ] **Step 4: Commit import robustness only if Step 2 changed code**

```bash
git add scripts/iccad2026_evaluate.py tests/test_evaluator_scoring.py
git commit -m "test: make scripts evaluator import robust"
```

## Task 4: Recalibrate Selection Guidance Documentation

**Files:**
- Modify: `docs/optimization-notes.md:15-25`
- Create: `docs/evaluation/2026-06-05-v10-recalibration.md`

- [ ] **Step 1: Update stale tail dominance wording**

In `docs/optimization-notes.md`, replace:

```markdown
- Score was dominated by validation IDs 99 and 98 because total score is exponentially weighted by block count.
- ID 99 contributed about `1.76 / 2.65`; ID 98 contributed about `0.59 / 2.65`.
```

with:

```markdown
- 2026-06-05 v10 scoring changes total weighting from `exp(n)` to `exp(n/12)`. Large cases still matter, but ID 99 alone no longer dominates the total score under the official formula.
- Under v10 weights, treat IDs 95-99 as diagnostics for the largest bucket, not as the whole promotion surface. Full-validation `total_score_no_runtime` remains the primary architecture metric.
```

- [ ] **Step 2: Create recalibration note**

Create `docs/evaluation/2026-06-05-v10-recalibration.md` with:

```markdown
# v10 Recalibration Notes

## Purpose

Recompute local evaluator expectations after the ICCAD 2026 FloorSet v10 scoring update.

## Formula Checks

- Feasible pathological case: `compute_cost(100, 100, 1, 100, True)` returns `9.999999`.
- Infeasible case: `compute_cost(0, 0, 0, 1, False)` returns `10.0`.
- Total score uses `exp(n/12)` weights.
- With block counts 21-120, the 120-block case carries about `7.9975%` of total weight.
- With block counts 21-120, the 116-120 bucket carries about `34.0841%` of total weight.

## Decision Impact

The old v9 assumption that ID 99 alone dominates the validation total is stale. Continue reporting IDs 95-99 because they are useful diagnostics, but promote checkpoints and algorithm changes using full-validation `total_score_no_runtime` first.

## Required Next Run

Run a full validation evaluation using the copied scripts evaluator path:

```bash
scripts/update.sh
bash scripts/eval_total.sh
```

Record:

- `total_score`
- `total_score_no_runtime`
- feasible count
- average runtime
- median runtime
- p90 runtime
- max runtime
- top score contributors
- top no-runtime score contributors

## Algorithm Gate

Do not change `floorset_arch` algorithm code until the full v10 recalibration run identifies whether the leading regression is HPWL gap, area gap, soft violations, raw runtime, or feasibility.
```

- [ ] **Step 3: Run docs grep to ensure no old `exp(n)` guidance remains in active notes**

Run:

```bash
rg -n "ID 99 contributed|exp\\(n\\)|e\\^\\{n_i\\}|close to 0\\.1" docs/optimization-notes.md docs/evaluation docs/superpowers/specs docs/superpowers/plans
```

Expected: no stale active guidance in `docs/optimization-notes.md` or the new recalibration note. Older archived specs may still mention historical context; do not rewrite unrelated history unless it is presented as current guidance.

- [ ] **Step 4: Commit documentation recalibration**

```bash
git add docs/optimization-notes.md docs/evaluation/2026-06-05-v10-recalibration.md
git commit -m "docs: recalibrate selection guidance for v10 scoring"
```

## Task 5: Check Checkpoint Selection Policy Against v10 Evidence

**Files:**
- Read: `src/floorset_arch/training/selection.py:1-64`
- Read: `tests/test_model.py`
- Modify: `tests/test_model.py` only if coverage is missing
- Modify: `src/floorset_arch/training/selection.py` only if tests expose a policy bug

- [ ] **Step 1: Inspect existing selection tests**

Run:

```bash
rg -n "CheckpointMetricRecord|better_checkpoint_metric|tail_weighted_no_runtime|total_score_no_runtime" tests/test_model.py src/floorset_arch/training/selection.py
```

Expected: tests cover feasible count, full no-runtime score, tail weighted no-runtime score, soft violations, runtime, validation loss, and epoch ordering.

- [ ] **Step 2: Add a test that full v10 no-runtime beats tail-only evidence when both are available**

If no equivalent test exists, add this to `tests/test_model.py` near other selection tests:

```python
def test_checkpoint_selection_prefers_full_no_runtime_over_tail_only():
    from floorset_arch.training.selection import (
        CheckpointMetricRecord,
        better_checkpoint_metric,
    )

    full_eval = CheckpointMetricRecord(
        checkpoint="full.pt",
        epoch=2,
        metric_source="full_eval",
        feasible=100,
        total_score_no_runtime=2.05,
        tail_weighted_no_runtime=2.20,
        soft_violations=8,
        avg_runtime=1.2,
        val_loss=0.5,
    )
    tail_better_but_full_worse = CheckpointMetricRecord(
        checkpoint="tail.pt",
        epoch=3,
        metric_source="full_eval",
        feasible=100,
        total_score_no_runtime=2.10,
        tail_weighted_no_runtime=2.00,
        soft_violations=4,
        avg_runtime=1.0,
        val_loss=0.4,
    )

    assert not better_checkpoint_metric(tail_better_but_full_worse, full_eval)
    assert better_checkpoint_metric(full_eval, tail_better_but_full_worse)
```

- [ ] **Step 3: Run selection tests**

Run:

```bash
uv run pytest -q tests/test_model.py -k "checkpoint or selection or tail_weighted"
```

Expected: selection-related tests pass.

- [ ] **Step 4: Commit selection policy test**

```bash
git add tests/test_model.py src/floorset_arch/training/selection.py
git commit -m "test: lock checkpoint selection to full v10 score"
```

## Task 6: Add Shared v10 Risk Budget Calculation

**Files:**
- Create: `src/floorset_arch/risk_budget.py`
- Create: `tests/test_risk_budget.py`

- [ ] **Step 1: Write tests for v10 score share and densities**

Create `tests/test_risk_budget.py` with:

```python
import math
from types import SimpleNamespace

import torch

from floorset_arch.risk_budget import (
    BudgetTier,
    instance_risk_budget,
    v10_score_share,
)


def _inst(
    block_count,
    *,
    b2b=0,
    p2b=0,
    boundary=0,
    cluster_sizes=(),
    mib_sizes=(),
    fixed=0,
    preplaced=0,
):
    return SimpleNamespace(
        block_count=block_count,
        valid_b2b=torch.zeros(b2b, 3),
        valid_p2b=torch.zeros(p2b, 3),
        boundary={i: 1 for i in range(boundary)},
        cluster_groups={
            idx + 1: list(range(size))
            for idx, size in enumerate(cluster_sizes)
        },
        mib_groups={
            idx + 1: list(range(size))
            for idx, size in enumerate(mib_sizes)
        },
        fixed=set(range(fixed)),
        preplaced=set(range(fixed, fixed + preplaced)),
    )


def test_v10_score_share_matches_exp_n_over_12():
    share_120 = v10_score_share(120)
    expected = math.exp((120 - 120) / 12) / sum(
        math.exp((n - 120) / 12) for n in range(21, 121)
    )

    assert share_120 == expected
    assert 0.079 < share_120 < 0.081


def test_medium_large_dense_case_gets_budget_without_118_block_gate():
    inst = _inst(
        104,
        b2b=4300,
        p2b=2100,
        boundary=30,
        cluster_sizes=(18, 12, 10),
        mib_sizes=(8, 7),
        fixed=6,
        preplaced=4,
    )

    budget = instance_risk_budget(inst)

    assert budget.block_count == 104
    assert budget.score_share > 0.0
    assert budget.constraint_density > 0.5
    assert budget.net_density > 60.0
    assert budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}


def test_120_block_low_density_case_is_not_automatically_heavy():
    inst = _inst(
        120,
        b2b=120,
        p2b=120,
        boundary=2,
        cluster_sizes=(),
        mib_sizes=(),
        fixed=0,
        preplaced=0,
    )

    budget = instance_risk_budget(inst)

    assert 0.079 < budget.score_share < 0.081
    assert budget.constraint_density < 0.05
    assert budget.net_density == 2.0
    assert budget.tier in {BudgetTier.NONE, BudgetTier.LIGHT}
```

- [ ] **Step 2: Run tests and verify they fail because module is missing**

Run:

```bash
uv run pytest -q tests/test_risk_budget.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'floorset_arch.risk_budget'`.

- [ ] **Step 3: Implement shared risk budget module**

Create `src/floorset_arch/risk_budget.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class BudgetTier(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


@dataclass(frozen=True)
class RiskBudget:
    block_count: int
    score_share: float
    constraint_density: float
    net_density: float
    boundary_count: int
    grouping_budget: int
    mib_budget: int
    hard_shape_count: int
    tier: BudgetTier


def v10_score_share(block_count: int, min_block_count: int = 21, max_block_count: int = 120) -> float:
    weights = [
        math.exp((n - max_block_count) / 12)
        for n in range(min_block_count, max_block_count + 1)
    ]
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return math.exp((block_count - max_block_count) / 12) / denom


def _edge_count(tensor) -> int:
    if tensor is None:
        return 0
    shape = getattr(tensor, "shape", None)
    if not shape:
        return 0
    return int(shape[0])


def _group_budget(groups) -> int:
    return sum(max(0, len(members) - 1) for members in groups.values())


def _tier(score_share: float, constraint_density: float, net_density: float) -> BudgetTier:
    if score_share >= 0.045 and constraint_density >= 0.45 and net_density >= 55.0:
        return BudgetTier.HEAVY
    if score_share >= 0.020 and (
        net_density >= 60.0 or (constraint_density >= 0.50 and net_density >= 20.0)
    ):
        return BudgetTier.MEDIUM
    if score_share >= 0.006 and constraint_density >= 0.55:
        return BudgetTier.LIGHT
    if score_share >= 0.015 and (constraint_density >= 0.20 or net_density >= 35.0):
        return BudgetTier.LIGHT
    return BudgetTier.NONE


def instance_risk_budget(inst) -> RiskBudget:
    block_count = int(inst.block_count)
    boundary_count = len(getattr(inst, "boundary", {}))
    grouping_budget = _group_budget(getattr(inst, "cluster_groups", {}))
    mib_budget = _group_budget(getattr(inst, "mib_groups", {}))
    hard_shape_count = len(getattr(inst, "fixed", set())) + len(
        getattr(inst, "preplaced", set())
    )
    b2b_count = _edge_count(getattr(inst, "valid_b2b", None))
    p2b_count = _edge_count(getattr(inst, "valid_p2b", None))
    score_share = v10_score_share(block_count)
    constraint_density = (
        boundary_count + grouping_budget + mib_budget + hard_shape_count
    ) / max(block_count, 1)
    net_density = (b2b_count + p2b_count) / max(block_count, 1)

    return RiskBudget(
        block_count=block_count,
        score_share=score_share,
        constraint_density=constraint_density,
        net_density=net_density,
        boundary_count=boundary_count,
        grouping_budget=grouping_budget,
        mib_budget=mib_budget,
        hard_shape_count=hard_shape_count,
        tier=_tier(score_share, constraint_density, net_density),
    )
```

- [ ] **Step 4: Run risk budget tests**

Run:

```bash
uv run pytest -q tests/test_risk_budget.py
```

Expected: PASS.

- [ ] **Step 5: Commit shared risk budget**

```bash
git add src/floorset_arch/risk_budget.py tests/test_risk_budget.py
git commit -m "feat: add v10 risk budget tiers"
```

## Task 7: Recalibrate High-Risk And Quality Portfolio Gating

**Files:**
- Modify: `src/floorset_arch/optimizer.py:420-509`
- Modify: `src/floorset_arch/quality_portfolio.py:24-48`
- Test: `tests/test_optimizer.py`
- Test: `tests/test_risk_budget.py`

- [ ] **Step 1: Add failing tests for gating behavior**

Add tests to `tests/test_risk_budget.py`:

```python
def test_budget_tier_can_drive_high_risk_without_id_specific_logic():
    dense_medium = _inst(
        104,
        b2b=4300,
        p2b=2100,
        boundary=30,
        cluster_sizes=(18, 12, 10),
        mib_sizes=(8, 7),
        fixed=6,
        preplaced=4,
    )
    sparse_tail = _inst(120, b2b=120, p2b=120, boundary=2)

    assert instance_risk_budget(dense_medium).tier in {
        BudgetTier.MEDIUM,
        BudgetTier.HEAVY,
    }
    assert instance_risk_budget(sparse_tail).tier in {
        BudgetTier.NONE,
        BudgetTier.LIGHT,
    }
```

Run:

```bash
uv run pytest -q tests/test_risk_budget.py
```

Expected: PASS after Task 6 implementation. This locks the desired budget behavior before wiring it into the solver.

- [ ] **Step 2: Wire high-risk gating to shared budget**

In `src/floorset_arch/optimizer.py`, add import:

```python
from floorset_arch.risk_budget import BudgetTier, instance_risk_budget
```

Then replace `_uses_high_risk_portfolio`, `_is_targeted_high_risk_case`, and `_is_high_risk_case` internals with budget-aware checks while preserving explicit env overrides:

```python
    def _uses_high_risk_portfolio(self, inst) -> bool:
        mode = (
            os.environ.get("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", "auto")
            .strip()
            .lower()
        )
        if mode in {"0", "false", "off", "no"}:
            return False
        if mode in {"1", "true", "on", "yes"}:
            return self._is_high_risk_case(inst)
        return self._is_targeted_high_risk_case(inst)

    def _is_targeted_high_risk_case(self, inst) -> bool:
        budget = instance_risk_budget(inst)
        return budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}

    def _is_high_risk_case(self, inst) -> bool:
        budget = instance_risk_budget(inst)
        return budget.tier is not BudgetTier.NONE
```

Keep `_is_dense_medium_risk_case` and `_uses_dense_shape_profiles` only if existing tests or profile logic still use them. If they become unused, leave them in place for this task rather than doing a broad cleanup.

- [ ] **Step 3: Wire quality portfolio to shared budget**

In `src/floorset_arch/quality_portfolio.py`, add import:

```python
from floorset_arch.risk_budget import BudgetTier, instance_risk_budget
```

Replace the automatic trigger block in `is_quality_portfolio_case`:

```python
    b2b_count = int(inst.valid_b2b.shape[0]) if inst.valid_b2b is not None else 0
    p2b_count = int(inst.valid_p2b.shape[0]) if inst.valid_p2b is not None else 0
    boundary_count = len(inst.boundary)
    grouping_budget = sum(max(0, len(members) - 1) for members in inst.cluster_groups.values())
    mib_budget = sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
    edge_density = (b2b_count + p2b_count) / max(inst.block_count, 1)

    min_blocks = int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_MIN_BLOCKS", "116"))
    extreme_density = float(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_EDGE_DENSITY", "80.0"))
    pin_heavy_min = int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_PIN_HEAVY_MIN", "3000"))
    b2b_heavy_min = int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_B2B_HEAVY_MIN", "5000"))
    if inst.block_count >= min_blocks and edge_density >= extreme_density:
        return True
    if inst.block_count >= min_blocks and p2b_count >= pin_heavy_min and b2b_count >= b2b_heavy_min:
        return True
    if inst.block_count >= 90 and boundary_count + grouping_budget + mib_budget >= 70 and edge_density >= 60.0:
        return True
    return False
```

with:

```python
    budget = instance_risk_budget(inst)
    return budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}
```

- [ ] **Step 4: Run gating and optimizer tests**

Run:

```bash
uv run pytest -q tests/test_risk_budget.py tests/test_optimizer.py
```

Expected: PASS. If existing optimizer tests encode old block-count-only behavior, update them to assert v10 budget behavior instead of validation-ID or 118-block assumptions.

- [ ] **Step 5: Commit gating recalibration**

```bash
git add src/floorset_arch/risk_budget.py src/floorset_arch/optimizer.py src/floorset_arch/quality_portfolio.py tests/test_risk_budget.py tests/test_optimizer.py
git commit -m "feat: recalibrate portfolio gating for v10 scoring"
```

## Task 8: Run Recalibration Smoke Commands

**Files:**
- Modify: `docs/evaluation/2026-06-05-v10-recalibration.md`

- [ ] **Step 1: Run evaluator scoring tests**

```bash
uv run pytest -q tests/test_evaluator_scoring.py
```

Expected: all evaluator scoring tests pass.

- [ ] **Step 2: Run selection policy tests**

```bash
uv run pytest -q tests/test_model.py -k "checkpoint or selection or tail_weighted"
```

Expected: selection policy tests pass.

- [ ] **Step 3: Run risk-budget gating tests**

```bash
uv run pytest -q tests/test_risk_budget.py tests/test_optimizer.py
```

Expected: risk-budget and optimizer gating tests pass.

- [ ] **Step 4: Run full evaluator only if the local dataset/checkpoint is ready**

```bash
scripts/update.sh
bash scripts/eval_total.sh
```

Expected: command prints `Total Score`, `Total Score (No Runtime)`, feasibility count, runtime summaries, and score contributors.

- [ ] **Step 5: Record full-run result if Step 4 was run**

Append a `## Full Run Result` section to `docs/evaluation/2026-06-05-v10-recalibration.md`:

```markdown
## Full Run Result

- Command: `scripts/update.sh && bash scripts/eval_total.sh`
- Total score: `<numeric output>`
- Total score no-runtime: `<numeric output>`
- Feasible: `<count>/100`
- Average runtime: `<seconds>`
- Median runtime: `<seconds>`
- P90 runtime: `<seconds>`
- Max runtime: `<seconds>`
- Top v10 score contributors: `<case list>`
- Top v10 no-runtime contributors: `<case list>`
- High-risk / quality portfolio budget tiers observed: `<case list with tier, score_share, constraint_density, net_density>`

## Algorithm Recommendation

Based on this run, the next algorithm patch should target `<HPWL gap | area gap | soft violations | runtime | feasibility>`, because `<specific observed evidence>`.
```

If the full run is not performed, write:

```markdown
## Full Run Result

Not run in this pass. Evaluator unit tests and selection-policy tests passed, but algorithm changes remain blocked until a full v10 recalibration run is recorded.
```

- [ ] **Step 5: Commit recalibration result**

```bash
git add docs/evaluation/2026-06-05-v10-recalibration.md
git commit -m "docs: record v10 recalibration result"
```

- [ ] **Step 6: Commit recalibration result**

```bash
git add docs/evaluation/2026-06-05-v10-recalibration.md
git commit -m "docs: record v10 recalibration result"
```

## Task 9: Update The Code Graph After Changes

**Files:**
- Generated: `graphify-out/graph.json`
- Generated: `graphify-out/graph.html`
- Generated: `graphify-out/GRAPH_REPORT.md`

- [ ] **Step 1: Refresh graphify output**

Run:

```bash
graphify update .
```

Expected:

```text
Code graph updated.
```

- [ ] **Step 2: Commit graph update only if project practice requires generated graph artifacts in commits**

```bash
git add graphify-out
git commit -m "chore: update code graph"
```

If graph artifacts are locally generated and not committed in this branch, leave them unstaged.

## Self-Review

Spec coverage:

- Direct scripts evaluator import is covered by Task 1.
- v10 feasible cap and `exp(n/12)` assertions are covered by Task 2.
- Avoiding `FloorSet` duplication is stated in File Structure and enforced by the import approach.
- Documentation recalibration is covered by Task 4.
- Selection policy evidence is covered by Task 5.
- Shared v10 risk/budget tiers are covered by Task 6.
- High-risk and quality portfolio gating recalibration is covered by Task 7.
- Algorithm changes are gated by Task 8.
- Graph update is covered by Task 9.

Placeholder scan:

- The only angle-bracket values appear inside a result-recording template in Task 8. They are not implementation placeholders; they are fields to be filled from command output after the full run.

Type consistency:

- `CheckpointMetricRecord`, `better_checkpoint_metric`, `total_score_no_runtime`, and `tail_weighted_no_runtime` match existing names in `src/floorset_arch/training/selection.py`.
- `compute_cost`, `compute_cost_breakdown`, `compute_total_score`, and `summarize_score_contributors` match the scripts evaluator API.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-06-05-v10-evaluator-selection-recalibration.md`. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task and review between tasks.
2. Inline Execution - execute tasks in this session with checkpoints.

Do not start algorithm patches until Task 8 records v10 recalibration evidence.
