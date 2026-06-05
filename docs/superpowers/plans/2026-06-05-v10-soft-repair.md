# v10 Risk-Gated Soft Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in v10 soft-repair acceptance path that targets boundary, grouping, and MIB soft pressure without changing the production checkpoint, decoder, or default runtime behavior.

**Architecture:** Implement small private helpers in `src/floorset_arch/repair.py` for v10 soft-repair eligibility and acceptance. Reuse existing repair primitives and `src/floorset_arch/risk_budget.py`; keep the feature behind `FLOORSET_ENABLE_V10_SOFT_REPAIR=1`. Validate with focused unit tests first, then full v10 evaluation against the Graph Transformer 0521 baseline.

**Tech Stack:** Python 3.12, pytest, uv, FloorSet evaluator scripts, existing `floorset_arch` repair/risk-budget modules.

---

## File Structure

- Modify: `src/floorset_arch/repair.py`
  - Responsibility: v10 soft-repair eligibility, acceptance policy, bounded repair pass, and integration into `repair_placement()`.
- Modify: `tests/test_repair.py`
  - Responsibility: unit coverage for opt-in gating, risk tier behavior, v10 acceptance slack, hard/overlap rejection, env overrides, and integration.
- Modify after validation: `docs/optimization-notes.md`
  - Responsibility: record whether the opt-in repair improves or regresses the Graph Transformer 0521 baseline.
- Modify after validation: `docs/evaluation/2026-06-05-v10-recalibration.md`
  - Responsibility: record full-validation score/runtime evidence and artifact path.
- Create after validation: `artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json`
  - Responsibility: full validation artifact for `FLOORSET_ENABLE_V10_SOFT_REPAIR=1`.

Do not modify decoder ranking, checkpoint promotion, `quality_portfolio.py`, or high-risk repair profile defaults in this phase.

## Task 1: Add Failing Unit Tests For V10 Soft Repair Helpers

**Files:**
- Modify: `tests/test_repair.py`

- [ ] **Step 1: Add imports for the new private helpers**

Add `SimpleNamespace`, import the repair module for monkeypatching, and add the new helpers to the existing `from floorset_arch.repair import (...)` block.

```python
from types import SimpleNamespace

import floorset_arch.repair as repair_module
```

Update the repair import block to include:

```python
    _score_better_v10_soft,
    _v10_soft_repair_eligible,
```

- [ ] **Step 2: Add a small fixture helper for v10 soft tests**

Append this helper near the top of `tests/test_repair.py`, after imports:

```python
def _soft_test_instance():
    areas = torch.full((4,), 4.0)
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return parse_instance(
        4,
        areas,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
```

- [ ] **Step 3: Add failing tests for opt-in and risk-tier eligibility**

Append these tests after `test_guarded_repair_reduces_soft_violation_counts()`:

```python
def test_v10_soft_repair_is_disabled_by_default(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.delenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", raising=False)
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.HEAVY),
    )

    assert not _v10_soft_repair_eligible(inst, placement)


def test_v10_soft_repair_allows_medium_and_heavy_when_enabled(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")

    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.MEDIUM),
    )
    assert _v10_soft_repair_eligible(inst, placement)

    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.HEAVY),
    )
    assert _v10_soft_repair_eligible(inst, placement)


def test_v10_soft_repair_light_tier_requires_soft_pressure(monkeypatch):
    inst = _soft_test_instance()
    low_soft = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    high_soft = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")
    monkeypatch.setenv("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_SOFT", "3")
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.LIGHT),
    )

    assert not _v10_soft_repair_eligible(inst, low_soft)
    assert _v10_soft_repair_eligible(inst, high_soft)


def test_v10_soft_repair_skips_none_risk_tier(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.NONE),
    )

    assert not _v10_soft_repair_eligible(inst, placement)
```

- [ ] **Step 4: Add failing tests for acceptance policy**

Append these tests after the eligibility tests:

```python
def test_v10_soft_accepts_grouping_improvement_with_grouping_slack(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(7.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    scores = {id(current): 100.0, id(candidate): 111.0}
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, placement: scores[id(placement)],
    )

    assert _score_better_v10_soft(inst, SolverConfig(), candidate, current)


def test_v10_soft_rejects_boundary_only_improvement_beyond_boundary_slack(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(7.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    scores = {id(current): 100.0, id(candidate): 107.0}
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, placement: scores[id(placement)],
    )

    assert not _score_better_v10_soft(inst, SolverConfig(), candidate, current)


def test_v10_soft_rejects_overlap_regression_even_when_soft_improves(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, _placement: 1.0,
    )

    assert not _score_better_v10_soft(inst, SolverConfig(), candidate, current)


def test_v10_soft_requires_geometry_improvement_when_soft_does_not_improve(monkeypatch):
    inst = _soft_test_instance()
    current = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )
    candidate = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(8.0, 0.0, 2.0, 2.0),
        }
    )
    scores = {id(current): 100.0, id(candidate): 99.98}
    monkeypatch.setattr(
        repair_module,
        "_geometry_quality_proxy",
        lambda _inst, placement: scores[id(placement)],
    )

    assert _score_better_v10_soft(inst, SolverConfig(), candidate, current)

    scores[id(candidate)] = 99.995

    assert not _score_better_v10_soft(inst, SolverConfig(), candidate, current)
```

- [ ] **Step 5: Run focused tests and verify they fail for missing helpers**

Run:

```bash
uv run pytest -q tests/test_repair.py -k "v10_soft"
```

Expected: failure during import with a message like:

```text
ImportError: cannot import name '_score_better_v10_soft'
```

- [ ] **Step 6: Commit failing tests**

```bash
git add tests/test_repair.py
git commit -m "test: cover v10 soft repair policy"
```

## Task 2: Implement Eligibility And Acceptance Helpers

**Files:**
- Modify: `src/floorset_arch/repair.py`
- Test: `tests/test_repair.py`

- [ ] **Step 1: Add risk-budget imports**

At the top of `src/floorset_arch/repair.py`, after existing imports, add:

```python
from floorset_arch.risk_budget import BudgetTier, instance_risk_budget
```

- [ ] **Step 2: Add environment and overlap helper functions**

Add these helpers after `_geometry_quality_proxy()`:

```python
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _overlap_count(placement: Placement) -> int:
    rects = list(placement.rects.values())
    count = 0
    for idx, rect in enumerate(rects):
        for other in rects[idx + 1 :]:
            if overlaps(rect, other):
                count += 1
    return count
```

- [ ] **Step 3: Add soft-capacity and hard-regression helpers**

Add these helpers after `_overlap_count()`:

```python
def _soft_capacity(inst: Instance) -> int:
    boundary_budget = len(inst.boundary)
    grouping_budget = sum(
        max(0, len(members) - 1) for members in inst.cluster_groups.values()
    )
    mib_budget = sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
    return max(boundary_budget + grouping_budget + mib_budget, 1)


def _hard_or_overlap_regressed(
    inst: Instance,
    candidate: Placement,
    current: Placement,
) -> bool:
    if _overlap_count(candidate) > _overlap_count(current):
        return True
    for block in range(inst.block_count):
        if block not in candidate.rects:
            return True
    for block in inst.fixed | inst.preplaced:
        target = inst.target_rects.get(block)
        rect = candidate.rects.get(block)
        if target is None or rect is None:
            return True
        if block in inst.preplaced and (
            abs(rect.x - target.x) > 1e-6 or abs(rect.y - target.y) > 1e-6
        ):
            return True
        if abs(rect.width - target.width) > 1e-6 or abs(rect.height - target.height) > 1e-6:
            return True
    return False
```

- [ ] **Step 4: Add v10 soft-repair eligibility**

Add this helper after `_hard_or_overlap_regressed()`:

```python
def _v10_soft_repair_eligible(inst: Instance, placement: Placement) -> bool:
    if not _env_flag("FLOORSET_ENABLE_V10_SOFT_REPAIR"):
        return False
    try:
        budget = instance_risk_budget(inst)
    except Exception:
        return False

    if budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}:
        return True
    if budget.tier is not BudgetTier.LIGHT:
        return False

    soft_total = sum(soft_violation_counts(inst, placement))
    soft_relative = soft_total / _soft_capacity(inst)
    min_soft = _env_int("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_SOFT", 6)
    min_relative = _env_float("FLOORSET_V10_SOFT_REPAIR_LIGHT_MIN_RELATIVE", 0.12)
    return soft_total >= min_soft or soft_relative >= min_relative
```

- [ ] **Step 5: Add v10 acceptance helper**

Add this helper after `_v10_soft_repair_eligible()`:

```python
def _score_better_v10_soft(
    inst: Instance,
    config: SolverConfig,
    candidate: Placement,
    current: Placement,
) -> bool:
    if _hard_or_overlap_regressed(inst, candidate, current):
        return False

    cand_counts = soft_violation_counts(inst, candidate)
    cur_counts = soft_violation_counts(inst, current)
    cand_soft = sum(cand_counts)
    cur_soft = sum(cur_counts)
    try:
        cand_proxy = _geometry_quality_proxy(inst, candidate)
        cur_proxy = _geometry_quality_proxy(inst, current)
    except Exception:
        return False

    if cand_soft < cur_soft:
        if cand_counts[1] < cur_counts[1]:
            slack = _env_float("FLOORSET_V10_SOFT_REPAIR_GROUPING_SLACK", 0.12)
        elif cand_counts[1] > cur_counts[1]:
            return False
        elif cand_counts[0] < cur_counts[0] and cand_counts[2] >= cur_counts[2]:
            slack = _env_float("FLOORSET_V10_SOFT_REPAIR_BOUNDARY_SLACK", 0.06)
        else:
            slack = _env_float("FLOORSET_V10_SOFT_REPAIR_GENERAL_SLACK", 0.08)
        return cand_proxy <= cur_proxy * (1.0 + slack)

    if cand_soft == cur_soft:
        min_improvement = _env_float(
            "FLOORSET_V10_SOFT_REPAIR_MIN_GEOMETRY_IMPROVEMENT",
            0.0001,
        )
        return cand_proxy <= cur_proxy * (1.0 - min_improvement)

    return False
```

Note: `config` is accepted for signature consistency with `_score_better_soft_first()` and future config use. It is intentionally unused in the first implementation.

- [ ] **Step 6: Run the focused v10 soft tests**

Run:

```bash
uv run pytest -q tests/test_repair.py -k "v10_soft"
```

Expected: all `v10_soft` tests pass.

- [ ] **Step 7: Run the full repair test file**

Run:

```bash
uv run pytest -q tests/test_repair.py
```

Expected: all tests in `tests/test_repair.py` pass.

- [ ] **Step 8: Commit helper implementation**

```bash
git add src/floorset_arch/repair.py tests/test_repair.py
git commit -m "feat: add v10 soft repair policy helpers"
```

## Task 3: Integrate Opt-In V10 Soft Repair Into `repair_placement()`

**Files:**
- Modify: `src/floorset_arch/repair.py`
- Modify: `tests/test_repair.py`

- [ ] **Step 1: Add a failing integration test**

Append this test after the v10 helper tests:

```python
def test_v10_soft_repair_path_runs_only_when_opted_in(monkeypatch):
    inst = _soft_test_instance()
    placement = Placement(
        {
            0: Rect(5.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 4.0, 2.0, 2.0),
            3: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    marker = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(2.0, 0.0, 2.0, 2.0),
            2: Rect(4.0, 0.0, 2.0, 2.0),
            3: Rect(6.0, 0.0, 2.0, 2.0),
        }
    )

    def fake_v10_repair(_inst, _placement, _config):
        return marker

    monkeypatch.setattr(repair_module, "_v10_soft_repair", fake_v10_repair)
    monkeypatch.delenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", raising=False)

    repaired_without_flag = repair_placement(inst, placement, SolverConfig())

    assert repaired_without_flag is not marker

    monkeypatch.setenv("FLOORSET_ENABLE_V10_SOFT_REPAIR", "1")

    repaired_with_flag = repair_placement(inst, placement, SolverConfig())

    assert repaired_with_flag is marker
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run:

```bash
uv run pytest -q tests/test_repair.py::test_v10_soft_repair_path_runs_only_when_opted_in
```

Expected: failure because `_v10_soft_repair` is not defined or not called.

- [ ] **Step 3: Implement bounded v10 soft repair**

Add this helper after `_guarded_soft_repair()`:

```python
def _v10_soft_repair(inst: Instance, placement: Placement, config: SolverConfig) -> Placement:
    if not _v10_soft_repair_eligible(inst, placement):
        return placement

    best = placement.copy()
    max_passes = max(1, min(config.max_repair_passes, 3))
    for _ in range(max_passes):
        trial = best.copy()
        _repair_boundary(inst, trial)
        _connect_clusters(inst, trial, config)
        _resolve_overlaps(inst, trial, config)
        _snap_boundary_components(inst, trial, config)
        _repair_boundary(inst, trial)
        _snap_hard(inst, trial)
        if _score_better_v10_soft(inst, config, trial, best):
            best = trial
        else:
            break
    return best
```

- [ ] **Step 4: Integrate the opt-in repair in `repair_placement()`**

In `repair_placement()`, replace:

```python
    repaired = _guarded_soft_repair(inst, repaired, config)
    repaired = _large_case_boundary_refine(inst, repaired, config)
```

with:

```python
    repaired = _guarded_soft_repair(inst, repaired, config)
    if _env_flag("FLOORSET_ENABLE_V10_SOFT_REPAIR"):
        repaired = _v10_soft_repair(inst, repaired, config)
    repaired = _large_case_boundary_refine(inst, repaired, config)
```

- [ ] **Step 5: Run the focused integration test**

Run:

```bash
uv run pytest -q tests/test_repair.py::test_v10_soft_repair_path_runs_only_when_opted_in
```

Expected: pass.

- [ ] **Step 6: Run all repair tests**

Run:

```bash
uv run pytest -q tests/test_repair.py
```

Expected: pass.

- [ ] **Step 7: Run related risk-budget and optimizer tests**

Run:

```bash
uv run pytest -q tests/test_risk_budget.py tests/test_optimizer.py -k "risk or high_risk or repair or large_case"
```

Expected: pass.

- [ ] **Step 8: Commit integration**

```bash
git add src/floorset_arch/repair.py tests/test_repair.py
git commit -m "feat: gate v10 soft repair behind opt-in flag"
```

## Task 4: Evaluate And Document The Opt-In Repair

**Files:**
- Create: `artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json`
- Modify: `docs/evaluation/2026-06-05-v10-recalibration.md`
- Modify: `docs/optimization-notes.md`

- [ ] **Step 1: Run full validation with the opt-in flag**

Run:

```bash
FLOORSET_ENABLE_V10_SOFT_REPAIR=1 \
bash scripts/eval_total.sh --output artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json
```

Expected:

```text
Feasible: 100
Results saved to artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json
```

If the evaluator writes relative to `FloorSet/iccad2026contest`, copy the artifact back to the root artifact directory:

```bash
mkdir -p artifacts/eval_v10
cp FloorSet/iccad2026contest/artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json \
  artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json
```

- [ ] **Step 2: Compare against the Graph Transformer 0521 baseline**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("artifacts/eval_v10/best_since_0512/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.json")
new = Path("artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json")
b = json.loads(base.read_text())
n = json.loads(new.read_text())

def summary(data):
    s = data.get("summary", {})
    return {
        "total": float(data["total_score"]),
        "no_runtime": float(data.get("total_score_no_runtime", data["total_score"])),
        "feasible": int(s.get("num_feasible", 0)),
        "avg_runtime": float(s.get("avg_runtime", 0.0)),
        "median_runtime": float(s.get("median_runtime", 0.0)),
        "p90_runtime": float(s.get("p90_runtime", 0.0)),
        "max_runtime": float(s.get("max_runtime", 0.0)),
    }

bs = summary(b)
ns = summary(n)
print("baseline", bs)
print("v10_soft", ns)
print("delta_no_runtime", ns["no_runtime"] - bs["no_runtime"])
print("delta_total", ns["total"] - bs["total"])
print("delta_avg_runtime", ns["avg_runtime"] - bs["avg_runtime"])
print("delta_p90_runtime", ns["p90_runtime"] - bs["p90_runtime"])
PY
```

Expected success criteria:

```text
v10_soft feasible == 100
delta_no_runtime < 0
delta_total <= 0.05
delta_p90_runtime <= 0.25
```

If `delta_no_runtime >= 0`, keep the feature opt-in and document the regression.

- [ ] **Step 3: Update evaluation notes**

Generate the exact markdown subsection from the baseline and new artifact:

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("artifacts/eval_v10/best_since_0512/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.json")
new = Path("artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json")
b = json.loads(base.read_text())
n = json.loads(new.read_text())

def row(data):
    s = data.get("summary", {})
    return {
        "total": float(data["total_score"]),
        "no_runtime": float(data.get("total_score_no_runtime", data["total_score"])),
        "feasible": int(s.get("num_feasible", 0)),
        "avg_runtime": float(s.get("avg_runtime", 0.0)),
        "p90_runtime": float(s.get("p90_runtime", 0.0)),
        "max_runtime": float(s.get("max_runtime", 0.0)),
    }

bs = row(b)
ns = row(n)
decision = (
    "candidate for default promotion"
    if ns["feasible"] == 100
    and ns["no_runtime"] < bs["no_runtime"]
    and ns["total"] <= bs["total"] + 0.05
    and ns["p90_runtime"] <= bs["p90_runtime"] + 0.25
    else "keep opt-in only"
)

print("### V10 Soft Repair Opt-In")
print()
print("Run with `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` on the Graph Transformer 0521 checkpoint.")
print()
print("| Run | v10 no-runtime | v10 total | Feasible | Avg runtime | P90 runtime | Max runtime |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
print(f"| Graph Transformer 0521 baseline | `{bs['no_runtime']:.4f}` | `{bs['total']:.4f}` | `{bs['feasible']}/100` | `{bs['avg_runtime']:.2f}s` | `{bs['p90_runtime']:.2f}s` | `{bs['max_runtime']:.2f}s` |")
print(f"| Graph Transformer 0521 + `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` | `{ns['no_runtime']:.4f}` | `{ns['total']:.4f}` | `{ns['feasible']}/100` | `{ns['avg_runtime']:.2f}s` | `{ns['p90_runtime']:.2f}s` | `{ns['max_runtime']:.2f}s` |")
print()
print(f"Decision: {decision}.")
print("Artifact: `artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json`.")
PY
```

Append the generated markdown under `## Algorithm Gate` in `docs/evaluation/2026-06-05-v10-recalibration.md`.

- [ ] **Step 4: Update optimization notes**

Generate the exact optimization-notes bullet:

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("artifacts/eval_v10/best_since_0512/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.json")
new = Path("artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json")
b = json.loads(base.read_text())
n = json.loads(new.read_text())

def row(data):
    s = data.get("summary", {})
    return {
        "total": float(data["total_score"]),
        "no_runtime": float(data.get("total_score_no_runtime", data["total_score"])),
        "feasible": int(s.get("num_feasible", 0)),
        "p90_runtime": float(s.get("p90_runtime", 0.0)),
    }

bs = row(b)
ns = row(n)
verb = "improved" if ns["no_runtime"] < bs["no_runtime"] else "regressed"
decision = (
    "Treat it as a promotion candidate after one repeat run."
    if ns["feasible"] == 100
    and ns["no_runtime"] < bs["no_runtime"]
    and ns["total"] <= bs["total"] + 0.05
    and ns["p90_runtime"] <= bs["p90_runtime"] + 0.25
    else "Keep it opt-in only."
)
print(
    "- 2026-06-05 `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` follow-up on the promoted "
    f"Graph Transformer 0521 checkpoint {verb} v10 no-runtime from "
    f"`{bs['no_runtime']:.4f}` to `{ns['no_runtime']:.4f}` and total from "
    f"`{bs['total']:.4f}` to `{ns['total']:.4f}` while staying "
    f"`{ns['feasible']}/100` feasible. {decision}"
)
PY
```

Add the generated bullet to `## Current Baseline` in `docs/optimization-notes.md`.

- [ ] **Step 5: Run documentation sanity checks**

Run:

```bash
rg -n "fill from run|value from run|replace with" docs/evaluation/2026-06-05-v10-recalibration.md docs/optimization-notes.md
```

Expected: no matches.

- [ ] **Step 6: Run focused regression tests one more time**

Run:

```bash
uv run pytest -q tests/test_repair.py tests/test_risk_budget.py tests/test_optimizer.py -k "v10_soft or risk or high_risk or repair or large_case"
```

Expected: pass.

- [ ] **Step 7: Update graphify**

Run:

```bash
graphify update .
```

Expected: graph update completes without blocking. Dirty `graphify-out/` files are expected.

- [ ] **Step 8: Commit evaluation notes**

```bash
git add \
  artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json \
  docs/evaluation/2026-06-05-v10-recalibration.md \
  docs/optimization-notes.md
git commit -m "docs: record v10 soft repair evaluation"
```

## Self-Review

- Spec coverage: The plan covers opt-in flag, per-instance risk gate, light-tier soft-pressure gate, conservative acceptance slack, hard/overlap rejection, tests, full validation, and documentation.
- Scope check: Runtime-Tail Budget Clamp and Decoder-Side Grouping Adjacency Bias are explicitly excluded from implementation tasks.
- Placeholder scan: Documentation updates are generated from JSON artifacts, and Task 4 includes an explicit `rg` check for unfinished wording.
- Type consistency: Helper names are consistently `_v10_soft_repair_eligible`, `_score_better_v10_soft`, and `_v10_soft_repair`; all live in `src/floorset_arch/repair.py`.
