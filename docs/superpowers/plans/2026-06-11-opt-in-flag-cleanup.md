# Opt-In Flag Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove deprecated opt-in paths and dead helpers across the codebase, document the full flag inventory, and update README to match the current solver, training, and evaluator behavior.

**Architecture:** Use an evidence-led cleanup. First preserve an audit trail, then remove superseded live code paths, then sweep the remaining codebase for flags, CLI options, debug toggles, numeric overrides, and unused helpers. Keep mixed-evidence controls as explicit ablations and remove only paths with clear negative, stale, or no-caller evidence.

**Tech Stack:** Python 3.12, pytest, uv, shell scripts, graphify, Semble, Markdown docs.

---

## File Structure

- Create: `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`
  - Audit table for every checked flag, option, numeric override, debug toggle, and helper.
- Modify: `README.md`
  - Update the existing Chinese project sections to reflect v5 wrapper, removed legacy files, current score policy, retained ablations, and active scripts.
- Modify: `docs/optimization-notes.md`
  - Convert removed harmful flags from live opt-in guidance to historical evidence.
- Modify: `docs/evaluation/2026-06-05-v10-recalibration.md`
  - Keep historical full-validation evidence, but mark removed paths as historical.
- Modify: `src/floorset_arch/budget_layer.py`
  - Remove hard runtime-tail clamp live decision fields if the hard clamp path is deleted.
- Modify: `src/floorset_arch/optimizer.py`
  - Stop importing or applying hard runtime-tail clamp config. Remove dead helpers.
- Modify: `src/floorset_arch/repair.py`
  - Remove hard runtime-tail clamp helper path and `_hard_or_overlap_regressed`.
- Modify: `src/floorset_arch/relative_order.py`
  - Remove broad grouping adjacency bias path while preserving default-on narrow grouping pair bias and its disable switch.
- Modify: `tests/test_budget_layer.py`, `tests/test_optimizer.py`, `tests/test_repair.py`, `tests/test_relative_order.py`
  - Remove tests that only protect deleted behavior; add tests for no stale broad/hard-clamp behavior.
- Accept deletion: `src/arch_old/*`, `src/architecture_v4_optimizer.py`
  - User already deleted these. Verify no active imports remain and update docs.

---

### Task 1: Capture Inventory And Deleted Wrapper Baseline

**Files:**
- Create: `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`
- Modify: `README.md`
- Stage existing deletions: `src/arch_old/*`, `src/architecture_v4_optimizer.py`

- [ ] **Step 1: Confirm worktree starts with only user deletions**

Run:

```bash
git status --short --branch
```

Expected: the only unstaged source changes are deleted `src/arch_old/*` and deleted `src/architecture_v4_optimizer.py`.

- [ ] **Step 2: Write the inventory audit document**

Create `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md` with this initial content:

```markdown
# Opt-In Flag Cleanup Audit

## Decision Policy

- Solver promotion/removal uses full-validation `total_score_no_runtime` first.
- Raw runtime, p90 runtime, max runtime, and runtime-aware total are gating or sanity signals.
- Training, checkpoint, debug, and evaluator controls are verified with targeted tests or CLI smoke checks unless they change solver evaluation output.
- Mixed-evidence controls remain explicit ablations.
- Removed controls keep historical evidence in docs, but no live usage instructions.

## Inventory

| Name | Kind | Owner | Default | Caller / Source | Evidence | Metric impact | Decision | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `src/arch_old/` | reference directory | legacy/reference | absent | no active imports expected | user removed; verify by exact search | not part of Production Solver Path | remove | accept deletion and remove README tree references |
| `src/architecture_v4_optimizer.py` | wrapper | evaluator wrapper | absent | current scripts use `src/architecture_v5_optimizer.py` | exact search and script check | v5 wrapper is current eval path | remove | accept deletion and update README/docs |
| `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP` | env flag | solver repair budget | off | `budget_layer.py`, `repair.py`, tests | 2026-06-05 full validation over-clamped quality | no-runtime regressed to `2.2741`; total regressed to `3.0120` | remove | delete hard clamp live path |
| `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS` | env flag | relative-order decoder | off | `relative_order.py`, tests | 2026-06-05 full validation regressed; narrow grouping supersedes it | no-runtime regressed to `2.4989`; total regressed to `3.3102` | remove | delete broad grouping path |
| `FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS` | env flag | relative-order decoder | on | `relative_order.py`, tests | 2026-06-10 full validation improved no-runtime and total | no-runtime `2.1538`, total `2.6993` | keep default | retain `=0` ablation switch |
| `FLOORSET_ENABLE_V10_SOFT_REPAIR` | env flag | repair | off | `budget_layer.py`, `repair.py`, tests | mixed full-validation evidence | no-runtime can improve, runtime tail regresses without budget | keep explicit ablation | document as ablation-only |
| `FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET` | env flag | repair budget | off | `budget_layer.py`, `repair.py`, tests | improves soft-repair runtime-aware path but not no-runtime default | total/runtime useful, no-runtime not promoted | keep explicit ablation | document as ablation pair with v10 soft repair |
```

- [ ] **Step 3: Verify deleted wrapper and legacy references**

Run:

```bash
rg -n "arch_old|architecture_v4_optimizer" . --glob '!graphify-out/**' --glob '!.git/**'
```

Expected before implementation: matches in README and historical docs only. Active scripts should point to `src/architecture_v5_optimizer.py`.

- [ ] **Step 4: Update README module tree for deleted paths**

In `README.md`, replace the `src/` tree block with this structure:

```text
src/
├── architecture_v5_optimizer.py
└── floorset_arch/
    ├── __init__.py
    ├── budget_layer.py
    ├── constructive.py
    ├── diagnostics.py
    ├── features.py
    ├── geometry.py
    ├── hetero_graph.py
    ├── models.py
    ├── optimizer.py
    ├── parser.py
    ├── quality_portfolio.py
    ├── relative_order.py
    ├── repair.py
    ├── risk_budget.py
    ├── scoring.py
    ├── surrogate_guidance.py
    ├── v10_proxy.py
    ├── nn/
    │   ├── __init__.py
    │   └── model.py
    └── training/
        ├── __init__.py
        ├── checkpoint.py
        ├── losses.py
        ├── promote_checkpoint.py
        ├── pseudo_targets.py
        ├── selection.py
        └── train.py
```

Replace the active wrapper paragraph with:

```markdown
`src/architecture_v5_optimizer.py` 是目前 evaluator scripts 使用的 contest-facing wrapper。它把 repo root、`src/` 和 `FloorSet/iccad2026contest` 加到 `sys.path`，並把 `MyOptimizer` / `ContestOptimizer` 指向 `floorset_arch.optimizer.ArchitectureV5Optimizer`。

`floorset_arch.optimizer.ArchitectureV4Optimizer` 目前保留為 `ArchitectureV5Optimizer` 的相容 alias，供舊 tests 或歷史 docs 讀懂演進脈絡；新的 eval/validate script 一律使用 v5 wrapper。
```

Remove the `Legacy/reference files` bullet for `src/arch_old/`.

- [ ] **Step 5: Run wrapper smoke tests**

Run:

```bash
uv run python - <<'PY'
import importlib.util
from pathlib import Path

wrapper = Path("src/architecture_v5_optimizer.py")
spec = importlib.util.spec_from_file_location("architecture_v5_optimizer", wrapper)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
assert mod.MyOptimizer is mod.ContestOptimizer
print(mod.MyOptimizer.__name__)
PY
```

Expected: prints `ArchitectureV5Optimizer`.

- [ ] **Step 6: Commit wrapper and legacy cleanup baseline**

Run:

```bash
git add docs/evaluation/2026-06-11-opt-in-flag-cleanup.md README.md src/arch_old src/architecture_v4_optimizer.py
git commit -m "chore: remove legacy wrappers from active tree"
```

Expected: commit includes deleted legacy/reference files, deleted v4 wrapper, README updates, and the initial audit doc.

---

### Task 2: Remove Hard Runtime-Tail Clamp Path

**Files:**
- Modify: `src/floorset_arch/budget_layer.py`
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `src/floorset_arch/repair.py`
- Modify: `tests/test_budget_layer.py`
- Modify: `tests/test_optimizer.py`
- Modify: `tests/test_repair.py`
- Modify: `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`

- [ ] **Step 1: Write failing stale-flag tests**

Replace `tests/test_budget_layer.py::test_budget_trace_context_exposes_unified_decision_fields` with:

```python
def test_budget_trace_context_exposes_unified_decision_fields(monkeypatch):
    inst = _inst()
    monkeypatch.setenv("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "auto")
    monkeypatch.setenv("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP", "1")

    trace = budget_trace_context(inst)

    assert trace["risk_tier"] in {"medium", "heavy"}
    assert trace["candidate_budget_tier"] in {"medium", "heavy"}
    assert trace["quality_portfolio_allowed"] is True
    assert "runtime_tail_clamp_enabled" not in trace
    assert trace["score_share"] > 0.0
    assert trace["net_density"] > 0.0
```

Delete these complete test functions from `tests/test_repair.py`:

- `test_runtime_tail_clamp_is_opt_in`
- `test_runtime_tail_clamp_lowers_heavy_budget_knobs`
- `test_runtime_tail_clamp_never_increases_existing_caps`

Delete `tests/test_optimizer.py::test_runtime_tail_clamp_applies_after_heavy_repair_profile`.

- [ ] **Step 2: Run stale-flag tests to verify current code fails**

Run:

```bash
uv run pytest -q tests/test_budget_layer.py::test_budget_trace_context_exposes_unified_decision_fields
```

Expected before implementation: FAIL because `runtime_tail_clamp_enabled` is still present in the trace.

- [ ] **Step 3: Remove clamp fields from `budget_layer.py`**

In `src/floorset_arch/budget_layer.py`, change `V10BudgetDecision` to:

```python
@dataclass(frozen=True)
class V10BudgetDecision:
    risk: RiskBudget
    candidate_budget_tier: BudgetTier
    quality_portfolio_allowed: bool
```

Delete the complete functions named `runtime_tail_clamp_enabled` and
`runtime_tail_clamp_limits`. The first currently returns
`env_flag("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP")`; the second currently returns
tier-specific clamp limits from `FLOORSET_RUNTIME_CLAMP_*` overrides.

```python
def runtime_tail_clamp_enabled() -> bool:
    return env_flag("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP")
```

Change `budget_decision()` to:

```python
def budget_decision(inst, *, budget: RiskBudget | None = None) -> V10BudgetDecision:
    budget = budget or instance_risk_budget(inst)
    return V10BudgetDecision(
        risk=budget,
        candidate_budget_tier=candidate_budget_tier(inst, budget=budget),
        quality_portfolio_allowed=quality_portfolio_allowed(inst, budget=budget),
    )
```

Remove this key from `budget_trace_context()`:

```python
"runtime_tail_clamp_enabled": decision.runtime_tail_clamp_enabled,
```

- [ ] **Step 4: Remove clamp application from `optimizer.py`**

Change the repair import in `src/floorset_arch/optimizer.py` from:

```python
from floorset_arch.repair import _runtime_tail_clamped_config, repair_placement
```

to:

```python
from floorset_arch.repair import repair_placement
```

Change the end of `_repair_profile_config()` from:

```python
if inst is None:
    return profile_config
return _runtime_tail_clamped_config(inst, profile_config)
```

to:

```python
return profile_config
```

- [ ] **Step 5: Remove clamp helpers from `repair.py`**

Delete the complete functions named `_runtime_tail_clamp_limits` and
`_runtime_tail_clamped_config` from `src/floorset_arch/repair.py`.

Remove `replace` from this import if it is no longer used:

```python
from dataclasses import replace
```

- [ ] **Step 6: Update audit entry**

In `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`, change the `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP` action to:

```markdown
| `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP` | env flag | solver repair budget | removed | historical docs only | 2026-06-05 full validation over-clamped quality | no-runtime regressed to `2.2741`; total regressed to `3.0120` | remove | deleted hard clamp live path and tests |
```

- [ ] **Step 7: Verify no live hard clamp references remain**

Run:

```bash
rg -n "FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP|RUNTIME_CLAMP|runtime_tail_clamp|_runtime_tail_clamped_config" src tests scripts README.md docs/evaluation docs/optimization-notes.md
```

Expected: matches remain only in historical docs and the cleanup audit.

- [ ] **Step 8: Run targeted tests**

Run:

```bash
uv run pytest -q tests/test_budget_layer.py tests/test_repair.py tests/test_optimizer.py
```

Expected: PASS.

- [ ] **Step 9: Commit hard clamp removal**

Run:

```bash
git add src/floorset_arch/budget_layer.py src/floorset_arch/optimizer.py src/floorset_arch/repair.py tests/test_budget_layer.py tests/test_optimizer.py tests/test_repair.py docs/evaluation/2026-06-11-opt-in-flag-cleanup.md
git commit -m "refactor: remove hard runtime clamp opt-in"
```

---

### Task 3: Remove Broad Grouping Adjacency Bias Path

**Files:**
- Modify: `src/floorset_arch/relative_order.py`
- Modify: `tests/test_relative_order.py`
- Modify: `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`

- [ ] **Step 1: Replace broad-bias tests with stale-flag guard**

Delete `tests/test_relative_order.py::test_grouping_adjacency_bias_can_chain_compact_profile`.

Replace `test_narrow_grouping_pair_bias_does_not_enable_global_key_blend` with:

```python
def test_removed_broad_grouping_flag_does_not_enable_global_key_blend(monkeypatch):
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
    inst.anchor_guidance = AnchorGuidance(
        rect_priors={
            0: Rect(60.0, 0.0, 2.0, 2.0),
            1: Rect(30.0, 0.0, 2.0, 2.0),
            2: Rect(0.0, 0.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "1")
    monkeypatch.setenv("FLOORSET_GROUPING_ADJACENCY_KEY_BLEND", "1.0")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert placement.rects[0].x >= placement.rects[1].right
```

- [ ] **Step 2: Run stale-flag guard to verify current code fails**

Run:

```bash
uv run pytest -q tests/test_relative_order.py::test_removed_broad_grouping_flag_does_not_enable_global_key_blend
```

Expected before implementation: FAIL if the broad flag still enables key blend or chaining.

- [ ] **Step 3: Remove broad key blend from `_bias_order_keys()`**

In `src/floorset_arch/relative_order.py`, replace this block:

```python
if profile == "compact" and _grouping_adjacency_bias_enabled(profile) and not _narrow_grouping_pair_bias_enabled():
    blend = max(
        0.0,
        min(
            1.0,
            _env_float("FLOORSET_GROUPING_ADJACENCY_KEY_BLEND", 0.65),
        ),
    )
else:
    blend = 0.0 if profile == "compact" else 0.18
```

with:

```python
blend = 0.0 if profile == "compact" else 0.18
```

- [ ] **Step 4: Delete broad grouping flag helper**

Delete `_grouping_adjacency_bias_enabled()` from `src/floorset_arch/relative_order.py`.

Replace this same-cluster branch:

```python
elif orient.get(ci, "H") == "H":
    h_score += 0.45 if _grouping_adjacency_bias_enabled(profile) else 0.0
else:
    v_score += 0.45 if _grouping_adjacency_bias_enabled(profile) else 0.0
```

with no broad fallback branch. After the change, same-cluster pairs only receive
extra grouping pressure through `_narrow_grouping_axis_bonus()` when
`narrow_enabled` and `cluster_pressure` are true.

Then remove the final chain block that begins with
`if _grouping_adjacency_bias_enabled(profile) and not narrow_enabled:`.

- [ ] **Step 5: Update audit entry**

In `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`, change the broad grouping entry to:

```markdown
| `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS` | env flag | relative-order decoder | removed | historical docs only | 2026-06-05 full validation regressed; narrow grouping supersedes it | no-runtime regressed to `2.4989`; total regressed to `3.3102` | remove | deleted broad key blend, chaining path, and tests |
```

- [ ] **Step 6: Verify no live broad grouping references remain**

Run:

```bash
rg -n "FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS|FLOORSET_GROUPING_ADJACENCY|grouping_adjacency|_grouping_adjacency_bias_enabled" src tests scripts README.md docs/evaluation docs/optimization-notes.md
```

Expected: matches remain only in historical docs and the cleanup audit.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
uv run pytest -q tests/test_relative_order.py
```

Expected: PASS.

- [ ] **Step 8: Commit broad grouping removal**

Run:

```bash
git add src/floorset_arch/relative_order.py tests/test_relative_order.py docs/evaluation/2026-06-11-opt-in-flag-cleanup.md
git commit -m "refactor: remove broad grouping adjacency flag"
```

---

### Task 4: Remove Dead Helpers And Recheck Duplicate Logic

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `src/floorset_arch/repair.py`
- Modify: `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`

- [ ] **Step 1: Confirm current no-caller evidence**

Run:

```bash
rg -n "_build_candidate_worker|_proxy_cost|_hard_or_overlap_regressed" src tests
```

Expected before removal: definitions only for `_build_candidate_worker`, `_proxy_cost`, and `_hard_or_overlap_regressed`, plus direct uses of `_no_runtime_proxy_cost`.

- [ ] **Step 2: Delete `_build_candidate_worker()`**

Remove the complete module-level function `_build_candidate_worker` from
`src/floorset_arch/optimizer.py`.

Keep class method `_build_candidate()` because it is the active path.

- [ ] **Step 3: Delete `_proxy_cost()`**

Remove this method from `src/floorset_arch/optimizer.py`:

```python
def _proxy_cost(self, inst, placement: Placement) -> float:
    return v10_proxy_cost(inst, placement, placement_metrics(inst, placement))
```

Keep `_no_runtime_proxy_cost()` because tests use it and it names the current proxy purpose more clearly.

- [ ] **Step 4: Delete `_hard_or_overlap_regressed()`**

Remove the complete function `_hard_or_overlap_regressed` from
`src/floorset_arch/repair.py`.

If `_overlap_count()` becomes unused after this deletion, keep it only if another helper still calls it. Otherwise remove `_overlap_count()` too.

- [ ] **Step 5: Update audit entries**

Append these rows to `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`:

```markdown
| `_build_candidate_worker` | helper | optimizer | unused | definition only | exact search and active class method check | no metric surface | remove | deleted unused module-level candidate worker |
| `_proxy_cost` | helper | optimizer | unused | definition only | exact search; `_no_runtime_proxy_cost` remains active | no metric surface | remove | deleted unused legacy proxy method |
| `_hard_or_overlap_regressed` | helper | repair | unused | definition only | exact search | no metric surface | remove | deleted unused hard/overlap guard |
```

- [ ] **Step 6: Verify helper removal**

Run:

```bash
rg -n "_build_candidate_worker|_proxy_cost|_hard_or_overlap_regressed" src tests
```

Expected: no matches for the deleted helpers. `_no_runtime_proxy_cost` may still match if included in a broader search.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
uv run pytest -q tests/test_optimizer.py tests/test_repair.py
```

Expected: PASS.

- [ ] **Step 8: Commit dead helper cleanup**

Run:

```bash
git add src/floorset_arch/optimizer.py src/floorset_arch/repair.py docs/evaluation/2026-06-11-opt-in-flag-cleanup.md
git commit -m "refactor: remove unused solver helpers"
```

---

### Task 5: Complete Full-Codebase Flag Inventory

**Files:**
- Modify: `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`

- [ ] **Step 1: Generate env-var inventory**

Run:

```bash
python3 - <<'PY'
import ast
from pathlib import Path

roots = [Path("src"), Path("scripts"), Path("tests")]
files = [p for root in roots if root.exists() for p in root.rglob("*.py")]

def parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents

def enclosing(node, parents):
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return cur.name
    return "<module>"

env_vars = {}
for path in files:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        continue
    parents = parent_map(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("FLOORSET_"):
            env_vars.setdefault(node.value, []).append((str(path), node.lineno, enclosing(node, parents)))

for name in sorted(env_vars):
    locs = "; ".join(f"{p}:{line}:{owner}" for p, line, owner in env_vars[name])
    print(f"{name}\t{locs}")
PY
```

Expected: a tab-separated inventory of all Python-visible `FLOORSET_*` variables.

- [ ] **Step 2: Generate shell/docs flag inventory**

Run:

```bash
rg -o "FLOORSET_[A-Z0-9_]+" scripts README.md docs tests src | sed 's/.*FLOORSET_/FLOORSET_/' | sort -u
```

Expected: a unique list that includes shell-only flags such as `FLOORSET_EVAL_REFRESH` and training script variables.

- [ ] **Step 3: Search CLI options**

Run:

```bash
rg -n "add_argument\\(|^[A-Z0-9_]+=\"\\$\\{|ENABLE_|CHECKPOINT_|WANDB|DEVICE|NUM_|EPOCHS|BATCH_SIZE|CLEAN_SAMPLE_POLICY" src/floorset_arch/training scripts README.md
```

Expected: argparse and shell-script options for training, evaluator, and checkpoint workflows.

- [ ] **Step 4: Append remaining keep/remove decisions**

Add rows to `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md` for remaining controls. Use this exact decision language:

```markdown
| `FLOORSET_GNN_CHECKPOINT` | env var | evaluator/solver checkpoint | configured/default checkpoint | scripts and `SolverConfig` | required evaluator plumbing | controls checkpoint under evaluation | keep default | document in README |
| `FLOORSET_EVAL_REFRESH` | env flag | eval script cache refresh | off | `scripts/eval_total.sh` | shell caller check | no solver behavior change; refreshes cached JSON | keep debug/evaluator | document as evaluator cache control |
| `FLOORSET_REPAIR_TRACE_JSONL` | env var | diagnostics | unset | optimizer trace writer | diagnostics tests | debug output only | keep debug | document as trace-only |
| `FLOORSET_ENABLE_QUALITY_PORTFOLIO` | env flag | quality portfolio | off | `budget_layer.py`, `quality_portfolio.py`, optimizer | current checkpoint regression evidence | current Graph Transformer no-runtime regressed | keep explicit ablation | document as ablation-only |
| `FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO` | env flag | candidate budget | auto | `budget_layer.py`, optimizer | budget-layer tests | generalized risk-gated candidate expansion | keep ablation/control | document as budget-control flag |
| `FLOORSET_INCLUDE_BEAM_CANDIDATES` | env flag | candidate generation | off | optimizer | historical no-gain notes | increased runtime without score win | keep explicit ablation | document as ablation-only or remove if no longer needed after exact search review |
| `FLOORSET_INCLUDE_NO_GUIDANCE_CANDIDATE` | env flag | candidate generation | off | optimizer | no-guidance diagnostics | useful ablation, not default | keep explicit ablation | document as ablation-only |
| `FLOORSET_ENABLE_SURROGATE_GUIDANCE` | env flag | no-checkpoint guidance | off | optimizer/surrogate guidance | optimization notes warn current surrogate worsened ID99 | diagnostic only | keep explicit ablation | document as no-checkpoint ablation |
| `FLOORSET_ENABLE_NO_GUIDANCE_GEOMETRY_REFINE` | env flag | repair/no-guidance | on for qualifying no-guidance cases | repair | no-guidance repair notes | no-checkpoint diagnostic path | keep ablation/control | document as no-guidance-only |
```

For each numeric override, classify it under the owning retained or removed path. Do not leave unclassified `FLOORSET_*` names.

- [ ] **Step 5: Verify every env var is represented in the audit**

Run:

```bash
python3 - <<'PY'
import re
from pathlib import Path

sources = "\n".join(
    p.read_text(errors="ignore")
    for root in ["src", "scripts", "tests", "README.md", "docs"]
    for p in ([Path(root)] if Path(root).is_file() else Path(root).rglob("*"))
    if p.is_file() and ".git" not in p.parts and "graphify-out" not in p.parts
)
found = sorted(set(re.findall(r"FLOORSET_[A-Z0-9_]+", sources)))
audit = Path("docs/evaluation/2026-06-11-opt-in-flag-cleanup.md").read_text()
missing = [name for name in found if name not in audit]
if missing:
    print("\n".join(missing))
    raise SystemExit(1)
print(f"all {len(found)} FLOORSET_* names represented")
PY
```

Expected: prints `all N FLOORSET_* names represented`.

- [ ] **Step 6: Commit inventory completion**

Run:

```bash
git add docs/evaluation/2026-06-11-opt-in-flag-cleanup.md
git commit -m "docs: audit opt-in flags across codebase"
```

---

### Task 6: Update README And Historical Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/optimization-notes.md`
- Modify: `docs/evaluation/2026-06-05-v10-recalibration.md`

- [ ] **Step 1: Update README score policy section**

Ensure `README.md` says:

```markdown
`total_score_no_runtime` 是本 repo 目前做 solver 架構、repair policy、candidate matrix 與 checkpoint promotion 的主要本地指標。提交前仍要看 raw runtime、p90 runtime、max runtime 與 runtime-aware `total_score`，但它們是 gating / sanity check，不取代 full-validation no-runtime promotion。
```

- [ ] **Step 2: Update README script and training sections**

Add or refresh examples so retained controls are labeled:

````markdown
Evaluator / submission sanity controls:

```bash
FLOORSET_EVAL_REFRESH=1 bash scripts/eval_total.sh --best-since-0512
FLOORSET_REPAIR_TRACE_JSONL=artifacts/repair_trace.jsonl bash scripts/eval_single.sh 99
```

Training / checkpoint controls:

```bash
CHECKPOINT_METRICS_MANIFEST=artifacts/checkpoint_metrics.jsonl \
EVALUATOR_BEST_CHECKPOINT=checkpoints/gnn_hgt_best_evaluator.pt \
bash scripts/train_hgt.sh
```

Ablation-only solver controls:

```bash
FLOORSET_ENABLE_V10_SOFT_REPAIR=1 FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET=1 bash scripts/eval_total.sh
FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS=0 bash scripts/eval_total.sh
FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto bash scripts/eval_total.sh
```
````

Do not include deleted controls `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP` or `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS` as runnable instructions.

- [ ] **Step 3: Update README active path steps**

Change the solve flow line:

```markdown
6. `_select_best_candidate()` 使用 hard legality gate 和 V10 no-runtime proxy 選出 best placement。
```

Ensure it no longer says `soft-first ranking`.

- [ ] **Step 4: Update historical docs from live guidance to history**

In `docs/optimization-notes.md`, replace the current runtime clamp and broad grouping bullets with:

```markdown
- Historical only: `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1` reduced tail runtime in the first v10 run, but over-clamped quality and has been removed from the live solver path. Use Conditional Runtime Budget experiments instead.
- Historical only: `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1` regressed full validation and has been removed from the live decoder. Narrow Grouping Pair Bias is the retained default path.
```

In `docs/evaluation/2026-06-05-v10-recalibration.md`, add one sentence after the old clamp/grouping evidence:

```markdown
These two flags are historical evidence only after the 2026-06-11 cleanup; their live code paths were removed in favor of conditional runtime budget and default-on narrow grouping pair bias.
```

- [ ] **Step 5: Verify README has no deleted flag instructions**

Run:

```bash
rg -n "FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP|FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS|architecture_v4_optimizer|arch_old|soft-first ranking" README.md
```

Expected: no matches.

- [ ] **Step 6: Commit README and docs updates**

Run:

```bash
git add README.md docs/optimization-notes.md docs/evaluation/2026-06-05-v10-recalibration.md
git commit -m "docs: align README with cleaned opt-in surface"
```

---

### Task 7: Final Verification And Graph Update

**Files:**
- Modified by graphify: `graphify-out/*`

- [ ] **Step 1: Run exact stale-reference checks**

Run:

```bash
rg -n "FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP|FLOORSET_RUNTIME_CLAMP|FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS|FLOORSET_GROUPING_ADJACENCY|_runtime_tail_clamped_config|_grouping_adjacency_bias_enabled|_build_candidate_worker|_proxy_cost|_hard_or_overlap_regressed|architecture_v4_optimizer|arch_old" src tests scripts README.md docs --glob '!docs/superpowers/plans/**' --glob '!docs/superpowers/specs/**'
```

Expected: matches only in historical docs and the cleanup audit. No matches in `src`, `tests`, `scripts`, or `README.md`.

- [ ] **Step 2: Run targeted test suite**

Run:

```bash
uv run pytest -q tests/test_budget_layer.py tests/test_relative_order.py tests/test_repair.py tests/test_optimizer.py tests/test_checkpoint_promotion.py tests/test_model.py tests/test_evaluator_scoring.py tests/test_diagnostics.py
```

Expected: PASS.

- [ ] **Step 3: Run evaluator wrapper smoke**

Run:

```bash
bash scripts/validate.sh
```

Expected: validator starts with `src/architecture_v5_optimizer.py` and exits successfully. If this command is too slow or data-dependent, record the exact failure in the final notes and run the Python wrapper smoke from Task 1 again.

- [ ] **Step 4: Run graphify update**

Run:

```bash
graphify update .
```

Expected: graphify completes and updates `graphify-out/`.

- [ ] **Step 5: Commit graph update if files changed**

Run:

```bash
git status --short graphify-out
```

If graphify changed files, run:

```bash
git add graphify-out
git commit -m "chore: update graph after opt-in cleanup"
```

If graphify did not change files, do not create a commit.

- [ ] **Step 6: Final status**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: clean worktree, with cleanup commits visible on `v10-budget-proxy-runtime-grouping`.

---

## Self-Review

- Spec coverage: The plan covers full-codebase inventory, user-deleted legacy paths, deprecated solver flags, dead helpers, README rewrite, historical docs, tests, evaluator smoke, and graphify update.
- Placeholder scan: The plan uses concrete paths, command lines, expected outputs, and code snippets; it contains no placeholder implementation steps.
- Type consistency: The plan keeps `ArchitectureV4Optimizer` only as a package alias, uses `ArchitectureV5Optimizer` for the wrapper, keeps `FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS` as the retained default switch, and removes only the hard clamp and broad grouping live paths.
