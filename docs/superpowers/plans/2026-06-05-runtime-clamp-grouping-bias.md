# Runtime Clamp and Grouping Bias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in runtime-tail budget clamp and opt-in decoder-side grouping adjacency bias without changing production defaults.

**Architecture:** Runtime clamp lives in `repair.py` near the v10 soft-repair gate and is exposed to `optimizer.py` profile expansion. Grouping adjacency bias lives in `relative_order.py`, where cluster chain constraints are already built. Both controls are enabled only by environment variables.

**Tech Stack:** Python, pytest, existing `floorset_arch` solver modules, ICCAD v10 evaluator scripts.

---

### Task 1: Runtime-Tail Budget Clamp

**Files:**
- Modify: `src/floorset_arch/repair.py`
- Modify: `src/floorset_arch/optimizer.py`
- Test: `tests/test_repair.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing repair clamp tests**

Add to `tests/test_repair.py`:

```python
def test_runtime_tail_clamp_is_opt_in(monkeypatch):
    config = SolverConfig(
        max_repair_passes=8,
        max_boundary_component_snaps=30,
        max_cluster_component_moves=26,
        max_pair_candidates_per_component=40,
    )
    monkeypatch.delenv("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP", raising=False)

    clamped = repair_module._runtime_tail_clamped_config(_soft_test_instance(), config)

    assert clamped is config


def test_runtime_tail_clamp_lowers_heavy_budget_knobs(monkeypatch):
    config = SolverConfig(
        max_repair_passes=8,
        max_boundary_component_snaps=30,
        max_cluster_component_moves=26,
        max_pair_candidates_per_component=40,
    )
    monkeypatch.setenv("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP", "1")
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.HEAVY),
    )

    clamped = repair_module._runtime_tail_clamped_config(_soft_test_instance(), config)

    assert clamped.max_repair_passes == 2
    assert clamped.max_boundary_component_snaps == 20
    assert clamped.max_cluster_component_moves == 18
    assert clamped.max_pair_candidates_per_component == 24


def test_runtime_tail_clamp_never_increases_existing_caps(monkeypatch):
    config = SolverConfig(
        max_repair_passes=1,
        max_boundary_component_snaps=7,
        max_cluster_component_moves=6,
        max_pair_candidates_per_component=5,
    )
    monkeypatch.setenv("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP", "1")
    monkeypatch.setattr(
        repair_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=repair_module.BudgetTier.HEAVY),
    )

    clamped = repair_module._runtime_tail_clamped_config(_soft_test_instance(), config)

    assert clamped.max_repair_passes == 1
    assert clamped.max_boundary_component_snaps == 7
    assert clamped.max_cluster_component_moves == 6
    assert clamped.max_pair_candidates_per_component == 5
```

- [ ] **Step 2: Run failing repair tests**

Run:

```bash
uv run pytest -q tests/test_repair.py -k "runtime_tail_clamp"
```

Expected: fails because `_runtime_tail_clamped_config` does not exist.

- [ ] **Step 3: Implement repair clamp helper**

In `src/floorset_arch/repair.py`, import `replace`:

```python
from dataclasses import replace
```

Add:

```python
def _runtime_tail_clamp_limits(tier: BudgetTier) -> tuple[int, int, int, int]:
    if tier is BudgetTier.HEAVY:
        defaults = (2, 20, 18, 24)
    elif tier is BudgetTier.MEDIUM:
        defaults = (2, 16, 14, 20)
    else:
        defaults = (1, 10, 8, 12)
    return (
        _env_int("FLOORSET_RUNTIME_CLAMP_MAX_REPAIR_PASSES", defaults[0]),
        _env_int("FLOORSET_RUNTIME_CLAMP_BOUNDARY_SNAPS", defaults[1]),
        _env_int("FLOORSET_RUNTIME_CLAMP_CLUSTER_MOVES", defaults[2]),
        _env_int("FLOORSET_RUNTIME_CLAMP_PAIR_CANDIDATES", defaults[3]),
    )


def _runtime_tail_clamped_config(inst: Instance, config: SolverConfig) -> SolverConfig:
    if not _env_flag("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP"):
        return config
    try:
        tier = instance_risk_budget(inst).tier
    except Exception:
        tier = BudgetTier.LIGHT
    max_passes, boundary_snaps, cluster_moves, pair_candidates = _runtime_tail_clamp_limits(tier)
    return replace(
        config,
        max_repair_passes=min(config.max_repair_passes, max_passes),
        max_boundary_component_snaps=min(config.max_boundary_component_snaps, boundary_snaps),
        max_cluster_component_moves=min(config.max_cluster_component_moves, cluster_moves),
        max_pair_candidates_per_component=min(
            config.max_pair_candidates_per_component,
            pair_candidates,
        ),
    )
```

In `_v10_soft_repair()`, after eligibility:

```python
    config = _runtime_tail_clamped_config(inst, config)
```

- [ ] **Step 4: Run repair clamp tests**

Run:

```bash
uv run pytest -q tests/test_repair.py -k "runtime_tail_clamp or v10_soft"
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing optimizer clamp test**

Add to `tests/test_optimizer.py`:

```python
def test_runtime_tail_clamp_applies_after_heavy_repair_profile(monkeypatch):
    monkeypatch.setenv("FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP", "1")
    monkeypatch.setattr(
        optimizer_module,
        "instance_risk_budget",
        lambda _inst: SimpleNamespace(tier=optimizer_module.BudgetTier.HEAVY),
    )
    inst = parse_instance(
        4,
        torch.full((4,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(4, 5),
        torch.full((4, 4), -1.0),
    )

    config = optimizer_module._repair_profile_config(SolverConfig(), "grouping_first", inst)

    assert config.max_cluster_component_moves == 18
    assert config.max_pair_candidates_per_component == 24
```

- [ ] **Step 6: Run failing optimizer clamp test**

Run:

```bash
uv run pytest -q tests/test_optimizer.py::test_runtime_tail_clamp_applies_after_heavy_repair_profile
```

Expected: fails because `_repair_profile_config` does not accept `inst`.

- [ ] **Step 7: Wire clamp into optimizer profile expansion**

In `src/floorset_arch/optimizer.py`, import:

```python
from floorset_arch.repair import repair_placement, _runtime_tail_clamped_config
```

Change `_repair_profile_config` signature:

```python
def _repair_profile_config(config: SolverConfig, repair_profile: str, inst=None) -> SolverConfig:
```

Store the expanded config in a local variable and return:

```python
    return _runtime_tail_clamped_config(inst, profile_config) if inst is not None else profile_config
```

Update both worker paths to call:

```python
profile_config = _repair_profile_config(config, repair_profile, inst)
```

- [ ] **Step 8: Run optimizer clamp test**

Run:

```bash
uv run pytest -q tests/test_optimizer.py::test_runtime_tail_clamp_applies_after_heavy_repair_profile
```

Expected: passes.

- [ ] **Step 9: Commit runtime clamp**

Run:

```bash
git add src/floorset_arch/repair.py src/floorset_arch/optimizer.py tests/test_repair.py tests/test_optimizer.py
git commit -m "feat: add opt-in runtime tail clamp"
```

### Task 2: Decoder-Side Grouping Adjacency Bias

**Files:**
- Modify: `src/floorset_arch/relative_order.py`
- Test: `tests/test_relative_order.py`

- [ ] **Step 1: Write failing compact grouping bias test**

Add to `tests/test_relative_order.py`:

```python
from floorset_arch.geometry import edge_touch_length
```

Add:

```python
def test_grouping_adjacency_bias_can_chain_compact_profile(monkeypatch):
    constraints = torch.zeros(4, 5)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
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
            1: Rect(20.0, 20.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
            3: Rect(0.0, 2.0, 2.0, 2.0),
        }
    )
    monkeypatch.setenv("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "1")

    placement = construct_relative_order_placement(inst, profile="compact")

    assert edge_touch_length(placement.rects[0], placement.rects[1]) > 0.0
```

- [ ] **Step 2: Run failing grouping bias test**

Run:

```bash
uv run pytest -q tests/test_relative_order.py::test_grouping_adjacency_bias_can_chain_compact_profile
```

Expected: fails because compact profile does not currently chain cluster members.

- [ ] **Step 3: Implement opt-in grouping bias**

In `src/floorset_arch/relative_order.py`, add:

```python
def _grouping_adjacency_bias_enabled(profile: str) -> bool:
    if os.environ.get("FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return profile != "compact"
    mode = os.environ.get("FLOORSET_GROUPING_ADJACENCY_BIAS_MODE", "auto").strip().lower()
    if mode == "soft_only":
        return profile != "compact"
    if mode == "compact_only":
        return profile == "compact"
    return True
```

Replace:

```python
    if profile != "compact":
```

with:

```python
    if _grouping_adjacency_bias_enabled(profile):
```

- [ ] **Step 4: Run relative-order tests**

Run:

```bash
uv run pytest -q tests/test_relative_order.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit grouping bias**

Run:

```bash
git add src/floorset_arch/relative_order.py tests/test_relative_order.py
git commit -m "feat: add opt-in grouping adjacency bias"
```

### Task 3: Validation and Documentation

**Files:**
- Modify: `docs/evaluation/2026-06-05-v10-recalibration.md`
- Modify: `docs/optimization-notes.md`
- Create: `artifacts/eval_v10/runtime_clamp_grouping_bias_graph_transformer_0521_v10.json`

- [ ] **Step 1: Run focused regression**

Run:

```bash
uv run pytest -q tests/test_repair.py tests/test_optimizer.py tests/test_relative_order.py -k "runtime_tail_clamp or grouping_adjacency_bias or v10_soft or high_risk or relative_order"
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run combined opt-in validation**

Run:

```bash
FLOORSET_ENABLE_V10_SOFT_REPAIR=1 \
FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1 \
FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1 \
bash scripts/eval_total.sh \
  --output /nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/artifacts/eval_v10/runtime_clamp_grouping_bias_graph_transformer_0521_v10.json
```

Expected: evaluator completes and writes JSON.

- [ ] **Step 4: Compare against baseline and v10 soft repair**

Run:

```bash
uv run python - <<'PY'
import json

paths = {
    "baseline": "artifacts/eval_v10/best_since_0512/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.json",
    "v10_soft": "artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json",
    "clamp_bias": "artifacts/eval_v10/runtime_clamp_grouping_bias_graph_transformer_0521_v10.json",
}
for name, path in paths.items():
    data = json.load(open(path))
    summary = data["summary"]
    print(
        name,
        f"no_rt={data['total_score_no_runtime']:.4f}",
        f"total={data['total_score']:.4f}",
        f"feasible={summary['num_feasible']}/{summary['num_tests']}",
        f"avg={summary['avg_runtime']:.2f}s",
        f"p90={summary['p90_runtime']:.2f}s",
        f"max={summary['max_runtime']:.2f}s",
    )
PY
```

Expected: printed comparison table.

- [ ] **Step 5: Update docs**

Add the combined run row to `docs/evaluation/2026-06-05-v10-recalibration.md`
and summarize the decision in `docs/optimization-notes.md`.

- [ ] **Step 6: Update graphify**

Run:

```bash
graphify update .
```

Expected: command completes.

- [ ] **Step 7: Commit validation notes**

Run:

```bash
git add docs/evaluation/2026-06-05-v10-recalibration.md docs/optimization-notes.md
git add -f artifacts/eval_v10/runtime_clamp_grouping_bias_graph_transformer_0521_v10.json
git commit -m "docs: record runtime clamp grouping bias evaluation"
```
