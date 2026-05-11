# Runtime-Aware V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report no-runtime quality score beside local runtime-aware score, rename the active wrapper/class to architecture v4, and load `.env` through `python-dotenv`.

**Architecture:** Keep `floorset_arch` as the production package. Add no-runtime score as a second evaluator channel while preserving existing runtime-aware `total_score`. Rename only public active architecture labels and wrappers from v3 to v4.

**Tech Stack:** Python 3.12, pytest, PyTorch, `python-dotenv`, shell scripts, ICCAD evaluator.

---

## File Structure

- `FloorSet/iccad2026contest/iccad2026_evaluate.py`: add `use_runtime`, `cost_no_runtime`, `total_score_no_runtime`, and runtime summary fields.
- `src/floorset_arch/optimizer.py`: rename the previous optimizer class to `ArchitectureV4Optimizer` and load repo `.env`.
- `src/architecture_v4_optimizer.py`: create the new contest wrapper.
- Previous architecture wrapper: remove after scripts/tests use v4.
- `src/floorset_arch/__init__.py`: export `ArchitectureV4Optimizer`.
- `tests/test_evaluator_scoring.py`: add focused evaluator scoring tests.
- `tests/test_optimizer.py`, `tests/test_diagnostics.py`: update imports and class names.
- `scripts/eval_total.sh`, `scripts/eval_single.sh`, `scripts/validate.sh`, `scripts/train.sh`: update active wrapper/training labels to v4.
- `pyproject.toml`, `requirements.txt`: add `python-dotenv`.
- `CONTEXT.md`, `docs/optimization-notes.md`, current `docs/superpowers/specs/*.md`, current `docs/superpowers/plans/*.md`: update active architecture and scoring guidance.

### Task 1: Evaluator No-Runtime Scoring

**Files:**
- Modify: `FloorSet/iccad2026contest/iccad2026_evaluate.py`
- Create: `tests/test_evaluator_scoring.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
import math
import sys
from pathlib import Path


CONTEST_DIR = Path(__file__).resolve().parents[1] / "FloorSet" / "iccad2026contest"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

import iccad2026_evaluate as evaluator  # noqa: E402


def test_compute_cost_can_disable_runtime_adjustment():
    with_runtime = evaluator.compute_cost(0.1, 0.2, 0.25, 4.0, True)
    without_runtime = evaluator.compute_cost(0.1, 0.2, 0.25, 4.0, True, use_runtime=False)

    expected_quality = 1 + evaluator.ALPHA * (0.1 + 0.2)
    expected_violation = math.exp(evaluator.BETA * 0.25)

    assert with_runtime > without_runtime
    assert without_runtime == expected_quality * expected_violation


def test_no_runtime_total_uses_no_runtime_costs():
    results = [
        evaluator.TestResult(0, 21, True, 0.0, 0.0, 0.0, 1.0, cost=2.0, cost_no_runtime=1.0),
        evaluator.TestResult(1, 120, True, 0.0, 0.0, 0.0, 8.0, cost=8.0, cost_no_runtime=3.0),
    ]

    assert evaluator.compute_total_score([r.cost_no_runtime for r in results], [r.block_count for r in results]) > 2.99
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_evaluator_scoring.py -q`

Expected: FAIL because `compute_cost()` has no `use_runtime` parameter and `TestResult` has no `cost_no_runtime`.

- [ ] **Step 3: Implement scoring fields**

Change `compute_cost()` to:

```python
def compute_cost(
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    runtime_factor: float,
    is_feasible: bool,
    use_runtime: bool = True,
) -> float:
    if not is_feasible:
        return M_PENALTY

    quality_factor = 1 + ALPHA * (max(0, hpwl_gap) + max(0, area_gap))
    violation_factor = math.exp(BETA * violations_relative)
    runtime_adjustment = max(0.7, math.pow(max(0.01, runtime_factor), GAMMA)) if use_runtime else 1.0

    return quality_factor * violation_factor * runtime_adjustment
```

Add `cost_no_runtime: float = M_PENALTY` to `TestResult`.

In `evaluate_solution()`, compute `cost_no_runtime = compute_cost(..., use_runtime=False)` and include it in `SolutionMetrics` if needed only through `TestResult`.

In `ContestEvaluator.evaluate()`, set `cost_no_runtime` on every non-error result and compute `total_score_no_runtime`.

Add summary fields:

```python
"avg_cost_no_runtime": ...,
"median_runtime": ...,
"p90_runtime": ...,
"max_runtime": ...,
"tail_runtimes": ...,
```

Print `Total Score (No Runtime)` and `Avg Cost (No Runtime)` in CLI output.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `uv run pytest tests/test_evaluator_scoring.py -q`

Expected: PASS.

### Task 2: Rename Active Architecture V3 To V4

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `src/floorset_arch/__init__.py`
- Create: `src/architecture_v4_optimizer.py`
- Delete: previous architecture wrapper file
- Modify: `tests/test_optimizer.py`
- Modify: `tests/test_diagnostics.py`
- Modify: `scripts/eval_total.sh`
- Modify: `scripts/eval_single.sh`
- Modify: `scripts/validate.sh`
- Modify: `scripts/train.sh`
- Modify: `src/floorset_arch/training/train.py`

- [ ] **Step 1: Write failing import expectations**

Update tests to import:

```python
from floorset_arch.optimizer import ArchitectureV4Optimizer, CandidateSpec
```

and instantiate `ArchitectureV4Optimizer()`.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_optimizer.py tests/test_diagnostics.py -q`

Expected: FAIL because `ArchitectureV4Optimizer` does not exist yet.

- [ ] **Step 3: Rename class and wrapper**

Rename class in `src/floorset_arch/optimizer.py`:

```python
class ArchitectureV4Optimizer(FloorplanOptimizer):
    """Anchor-GNN guided hetero-graph beam solver."""
```

Update package export:

```python
"""Architecture v4 components for the ICCAD 2026 FloorSet challenge."""

from floorset_arch.optimizer import ArchitectureV4Optimizer

__all__ = ["ArchitectureV4Optimizer"]
```

Create `src/architecture_v4_optimizer.py` importing `ArchitectureV4Optimizer` and assigning:

```python
MyOptimizer = ArchitectureV4Optimizer
ContestOptimizer = ArchitectureV4Optimizer
```

Update scripts to evaluate `src/architecture_v4_optimizer.py`.

Update training labels to v4.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `uv run pytest tests/test_optimizer.py tests/test_diagnostics.py -q`

Expected: PASS.

### Task 3: Load `.env` With python-dotenv

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `src/architecture_v4_optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing dotenv test**

Add:

```python
def test_optimizer_loads_repo_dotenv_for_checkpoint_env(tmp_path, monkeypatch):
    checkpoint = tmp_path / "from-dotenv.pt"
    _write_anchor_checkpoint(checkpoint)
    env_file = Path(".env")
    original = env_file.read_text() if env_file.exists() else None
    try:
        env_file.write_text(f"FLOORSET_GNN_CHECKPOINT={checkpoint}\\n", encoding="utf-8")
        monkeypatch.delenv("FLOORSET_GNN_CHECKPOINT", raising=False)
        optimizer = ArchitectureV4Optimizer()
        inst = parse_instance(**_tiny_problem())
        assert optimizer._try_anchor_guidance(inst) is not None
    finally:
        if original is None:
            env_file.unlink(missing_ok=True)
        else:
            env_file.write_text(original, encoding="utf-8")
```

- [ ] **Step 2: Run test and verify RED**

Run: `uv run pytest tests/test_optimizer.py::test_optimizer_loads_repo_dotenv_for_checkpoint_env -q`

Expected: FAIL because Python does not load `.env`.

- [ ] **Step 3: Add dependency and loader**

Add `python-dotenv>=1.0.1` to `pyproject.toml` dependencies and `requirements.txt`.

In `src/floorset_arch/optimizer.py`, add a guarded import:

```python
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)
```

- [ ] **Step 4: Run test and verify GREEN**

Run: `uv run pytest tests/test_optimizer.py::test_optimizer_loads_repo_dotenv_for_checkpoint_env -q`

Expected: PASS.

### Task 4: Documentation And Current Guidance

**Files:**
- Modify: `CONTEXT.md`
- Modify: `docs/optimization-notes.md`
- Modify: `docs/superpowers/specs/2026-05-10-score-under-1-design.md`
- Modify: `docs/superpowers/specs/2026-05-11-large-case-candidates-clean-training-design.md`
- Modify: `docs/superpowers/plans/2026-05-10-score-under-1.md`
- Modify: `docs/superpowers/plans/2026-05-11-large-case-candidates-clean-training.md`
- Modify: `docs/superpowers/specs/2026-05-11-runtime-aware-v4-design.md`
- Modify: `docs/superpowers/plans/2026-05-11-runtime-aware-v4.md`

- [ ] **Step 1: Update domain language**

In `CONTEXT.md`, replace active v3 language with Architecture v4 and add:

```markdown
**No-Runtime Quality Score**:
The local architecture-tuning score that uses official quality and soft-violation factors with runtime adjustment fixed to `1.0`.
_Avoid_: treating local runtime-aware score as the primary architecture metric.
```

- [ ] **Step 2: Update optimization notes**

State the local tuning order:

```text
feasible count -> large-case no-runtime score -> total no-runtime score -> soft violations -> HPWL/area -> raw runtime -> local runtime-aware score
```

- [ ] **Step 3: Update current v3 references**

Change active/current references to v4. Keep historical v3 mentions only when the sentence explicitly records past results.

- [ ] **Step 4: Verify docs search**

Run: `rg -n "stale active architecture labels" CONTEXT.md docs src scripts tests`

Expected: no active references remain.

### Task 5: Final Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_evaluator_scoring.py tests/test_optimizer.py tests/test_diagnostics.py -q`

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 3: Check status**

Run: `git status --short`

Expected: only intended files modified/deleted/created.

## Self-Review

- Spec coverage: evaluator no-runtime scoring, v4 rename, dotenv dependency, docs, and tests each have a task.
- Placeholder scan: no placeholders remain.
- Type consistency: plan uses `ArchitectureV4Optimizer`, `cost_no_runtime`, and `total_score_no_runtime` consistently.
