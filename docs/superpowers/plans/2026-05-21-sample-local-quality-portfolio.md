# Sample-Local Quality Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in/default-gated sample-local portfolio that generates multiple non-GNN placement candidates in parallel for high-impact cases and selects lower HPWL/area no-runtime quality without hurting checkpoint total score.

**Architecture:** Keep `floorset_arch` as the Production Solver Path. Add a small quality portfolio module that defines trigger policy, candidate variants, and soft-safe post-refine operations; `optimizer.py` expands candidate specs and runs them with sample-local worker parallelism. Default placement remains a candidate, and promotion requires full configured-checkpoint evaluation beating total `2.3086` and no-runtime `2.0362`.

**Tech Stack:** Python 3.12, PyTorch tensors for parsed instance data, existing `ThreadPoolExecutor`, pytest, local FloorSet evaluator scripts.

---

## File Structure

- Modify `src/floorset_arch/models.py`: extend `CandidateSpec` only if the existing dataclass in `optimizer.py` needs more metadata; prefer keeping the dataclass local to `optimizer.py`.
- Create `src/floorset_arch/quality_portfolio.py`: pure helpers for trigger policy, quality profiles, and soft-safe placement refiners. No checkpoint loading and no evaluator imports.
- Modify `src/floorset_arch/optimizer.py`: add portfolio candidate specs, route quality candidates through post-refine hooks, and use bounded sample-local parallelism.
- Modify `tests/test_optimizer.py`: verify portfolio trigger, candidate spec expansion, worker count, and ranking behavior.
- Create `tests/test_quality_portfolio.py`: unit-test HPWL/area refiners on small synthetic placements.
- Modify `docs/optimization-notes.md`: record accepted/rejected eval results.
- Create `docs/evaluation/2026-05-21-quality-portfolio-v1.md`: document full configured-checkpoint and no-checkpoint ablation results.

## Task 1: Define Portfolio Policy and Candidate Metadata

**Files:**
- Create: `src/floorset_arch/quality_portfolio.py`
- Modify: `src/floorset_arch/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for the trigger and specs**

Add to `tests/test_optimizer.py`:

```python
def test_quality_portfolio_targets_high_impact_cases(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_QUALITY_PORTFOLIO", raising=False)
    monkeypatch.delenv("FLOORSET_QUALITY_PORTFOLIO_PROFILES", raising=False)
    block_count = 120
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.tensor([[float(i % block_count), float((i + 1) % block_count), 1.0] for i in range(7200)]),
        torch.tensor([[float(i % 64), float(i % block_count), 1.0] for i in range(64)]),
        torch.zeros(64, 2),
        torch.zeros(block_count, 5),
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert optimizer._uses_quality_portfolio(inst)
    assert any(spec.name.startswith("quality_") for spec in specs)
    assert "default" in {spec.quality_profile for spec in specs}
    assert "hpwl_refine" in {spec.quality_profile for spec in specs}
    assert "area_refine" in {spec.quality_profile for spec in specs}


def test_quality_portfolio_skips_low_impact_cases(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_QUALITY_PORTFOLIO", raising=False)
    inst = parse_instance(
        32,
        torch.full((32,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(32, 5),
        torch.full((32, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert not optimizer._uses_quality_portfolio(inst)
    assert all(spec.quality_profile == "default" for spec in specs)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_quality_portfolio_targets_high_impact_cases tests/test_optimizer.py::test_quality_portfolio_skips_low_impact_cases -q
```

Expected: FAIL because `_uses_quality_portfolio` and `quality_profile` do not exist yet.

- [ ] **Step 3: Add policy helper**

Create `src/floorset_arch/quality_portfolio.py`:

```python
from __future__ import annotations

import os

from floorset_arch.models import Instance


QUALITY_PROFILES = ("default", "hpwl_refine", "area_refine", "balanced_refine")


def enabled_quality_profiles() -> list[str]:
    raw = os.environ.get("FLOORSET_QUALITY_PORTFOLIO_PROFILES", "default,hpwl_refine,area_refine,balanced_refine")
    profiles = [part.strip() for part in raw.split(",") if part.strip()]
    filtered = [profile for profile in profiles if profile in QUALITY_PROFILES]
    return filtered or ["default"]


def is_quality_portfolio_case(inst: Instance) -> bool:
    mode = os.environ.get("FLOORSET_ENABLE_QUALITY_PORTFOLIO", "auto").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes", "always", "force"}:
        return True

    b2b_count = int(inst.valid_b2b.shape[0]) if inst.valid_b2b is not None else 0
    p2b_count = int(inst.valid_p2b.shape[0]) if inst.valid_p2b is not None else 0
    boundary_count = len(inst.boundary)
    grouping_budget = sum(max(0, len(members) - 1) for members in inst.cluster_groups.values())
    mib_budget = sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
    edge_density = (b2b_count + p2b_count) / max(inst.block_count, 1)

    if inst.block_count >= int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_MIN_BLOCKS", "116")):
        return True
    if inst.block_count >= 90 and boundary_count + grouping_budget + mib_budget >= 50:
        return True
    if inst.block_count >= 90 and edge_density >= float(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_EDGE_DENSITY", "55.0")):
        return True
    return False
```

- [ ] **Step 4: Extend candidate spec metadata and candidate expansion**

In `src/floorset_arch/optimizer.py`, import the helpers and extend `CandidateSpec`:

```python
from floorset_arch.quality_portfolio import enabled_quality_profiles, is_quality_portfolio_case

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    profile: str
    kind: str = "relative_order"
    repair_profile: str = "normal"
    disable_guidance: bool = False
    quality_profile: str = "default"
```

Add methods:

```python
def _uses_quality_portfolio(self, inst) -> bool:
    return is_quality_portfolio_case(inst)

def _quality_profiles(self) -> list[str]:
    return enabled_quality_profiles()
```

After the base candidate list is assembled in `_candidate_specs`, expand only when `_uses_quality_portfolio(inst)` is true:

```python
return self._with_quality_portfolio_specs(inst, specs)
```

Implement:

```python
def _with_quality_portfolio_specs(self, inst, specs: list[CandidateSpec]) -> list[CandidateSpec]:
    if not self._uses_quality_portfolio(inst):
        return specs
    profiles = self._quality_profiles()
    expanded: list[CandidateSpec] = []
    seen: set[tuple[str, str, str, str, bool]] = set()
    for spec in specs:
        for quality_profile in profiles:
            key = (spec.profile, spec.kind, spec.repair_profile, quality_profile, spec.disable_guidance)
            if key in seen:
                continue
            seen.add(key)
            name = spec.name if quality_profile == "default" else f"quality_{spec.profile}_{quality_profile}"
            expanded.append(replace(spec, name=name, quality_profile=quality_profile))
    return expanded or specs
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_quality_portfolio_targets_high_impact_cases tests/test_optimizer.py::test_quality_portfolio_skips_low_impact_cases -q
```

Expected: PASS.

## Task 2: Implement Soft-Safe Quality Refiners

**Files:**
- Modify: `src/floorset_arch/quality_portfolio.py`
- Modify: `src/floorset_arch/optimizer.py`
- Create: `tests/test_quality_portfolio.py`

- [ ] **Step 1: Write failing tests for HPWL and area refiners**

Create `tests/test_quality_portfolio.py`:

```python
import torch

from floorset_arch.geometry import Rect, bbox, has_overlaps
from floorset_arch.models import Placement, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.quality_portfolio import refine_quality_candidate
from floorset_arch.repair import soft_violation_counts
from floorset_arch.scoring import hpwl_proxy


def test_hpwl_refine_moves_connected_block_closer_without_soft_regression(monkeypatch):
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.tensor([[0.0, 1.0, 20.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        torch.full((3, 4), -1.0),
    )
    placement = Placement({
        0: Rect(0.0, 0.0, 2.0, 2.0),
        1: Rect(80.0, 0.0, 2.0, 2.0),
        2: Rect(40.0, 0.0, 2.0, 2.0),
    })
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "8")
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "32")

    refined = refine_quality_candidate(inst, placement, SolverConfig(), "hpwl_refine")

    assert not has_overlaps(list(refined.rects.values()))
    assert soft_violation_counts(inst, refined) == soft_violation_counts(inst, placement)
    assert hpwl_proxy(inst, refined.rects) < hpwl_proxy(inst, placement.rects)


def test_area_refine_reduces_bbox_without_soft_regression(monkeypatch):
    inst = parse_instance(
        4,
        torch.full((4,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(4, 5),
        torch.full((4, 4), -1.0),
    )
    placement = Placement({
        0: Rect(0.0, 0.0, 2.0, 2.0),
        1: Rect(2.0, 0.0, 2.0, 2.0),
        2: Rect(4.0, 0.0, 2.0, 2.0),
        3: Rect(100.0, 100.0, 2.0, 2.0),
    })
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "8")
    monkeypatch.setenv("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "32")

    refined = refine_quality_candidate(inst, placement, SolverConfig(), "area_refine")

    assert not has_overlaps(list(refined.rects.values()))
    assert soft_violation_counts(inst, refined) == soft_violation_counts(inst, placement)
    assert bbox(list(refined.rects.values())).area < bbox(list(placement.rects.values())).area
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_quality_portfolio.py -q
```

Expected: FAIL because `refine_quality_candidate` is not implemented.

- [ ] **Step 3: Implement quality proxy and refiners**

Append to `src/floorset_arch/quality_portfolio.py`:

```python
from floorset_arch.geometry import Rect, bbox, candidate_frontier_points, first_non_overlapping
from floorset_arch.models import Placement, SolverConfig
from floorset_arch.repair import soft_violation_counts
from floorset_arch.scoring import hpwl_proxy


def _edge_weight(inst: Instance, block: int) -> float:
    return sum(weight for _other, weight in inst.b2b_by_block.get(block, [])) + sum(
        weight for _pin, weight in inst.p2b_by_block.get(block, [])
    )


def _quality_score(inst: Instance, placement: Placement, profile: str) -> float:
    bounds = bbox(list(placement.rects.values()))
    hpwl = hpwl_proxy(inst, placement.rects)
    if profile == "area_refine":
        return bounds.area + 0.001 * hpwl
    if profile == "hpwl_refine":
        return hpwl + 0.01 * bounds.area
    return hpwl + 0.08 * bounds.area


def refine_quality_candidate(
    inst: Instance,
    placement: Placement,
    config: SolverConfig,
    profile: str,
) -> Placement:
    if profile == "default" or not placement.rects:
        return placement

    best = placement.copy()
    best_soft = sum(soft_violation_counts(inst, best))
    best_score = _quality_score(inst, best, profile)
    max_blocks = max(0, int(os.environ.get("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "20")))
    max_slots = max(0, int(os.environ.get("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "48")))
    if max_blocks == 0 or max_slots == 0:
        return best

    movable = [block for block in best.rects if block not in inst.preplaced]
    if profile == "area_refine":
        center_x = sum(rect.center_x for rect in best.rects.values()) / max(len(best.rects), 1)
        center_y = sum(rect.center_y for rect in best.rects.values()) / max(len(best.rects), 1)
        movable.sort(
            key=lambda block: (
                -(abs(best.rects[block].center_x - center_x) + abs(best.rects[block].center_y - center_y)),
                -_edge_weight(inst, block),
                block,
            )
        )
    else:
        movable.sort(key=lambda block: (-_edge_weight(inst, block), block))

    for block in movable[:max_blocks]:
        rect = best.rects[block]
        others = [other for idx, other in best.rects.items() if idx != block]
        slots = candidate_frontier_points(others)[:max_slots]
        for x, y in slots:
            candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
            if abs(candidate.x - rect.x) <= 1e-9 and abs(candidate.y - rect.y) <= 1e-9:
                continue
            if not first_non_overlapping(candidate, others):
                continue
            trial = best.copy()
            trial.rects[block] = candidate
            if sum(soft_violation_counts(inst, trial)) > best_soft:
                continue
            score = _quality_score(inst, trial, profile)
            if score < best_score * (1.0 - 1e-4):
                best = trial
                best_soft = sum(soft_violation_counts(inst, best))
                best_score = score
                rect = candidate
                others = [other for idx, other in best.rects.items() if idx != block]
    return best
```

- [ ] **Step 4: Hook refiners into optimizer candidate repair**

In `src/floorset_arch/optimizer.py`, import and call:

```python
from floorset_arch.quality_portfolio import refine_quality_candidate

def _repair_candidate(self, inst, placement: Placement, spec: CandidateSpec) -> Placement:
    before = placement
    after = self._repair_with_profile(inst, before, spec.repair_profile)
    if spec.repair_profile == "quality_refine":
        after = self._quality_refine_candidate(inst, after)
    after = refine_quality_candidate(inst, after, self.config, spec.quality_profile)
    self._trace_repair(...)
    return after
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
uv run pytest tests/test_quality_portfolio.py tests/test_optimizer.py -q
```

Expected: PASS.

## Task 3: Add Bounded Sample-Local Parallelism

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing worker-count test**

Add to `tests/test_optimizer.py`:

```python
def test_quality_portfolio_uses_bounded_sample_local_workers(monkeypatch):
    monkeypatch.setenv("FLOORSET_QUALITY_PORTFOLIO_WORKERS", "4")
    specs = [
        CandidateSpec(name="a", profile="soft", quality_profile="default"),
        CandidateSpec(name="b", profile="soft", quality_profile="hpwl_refine"),
        CandidateSpec(name="c", profile="soft", quality_profile="area_refine"),
        CandidateSpec(name="d", profile="soft", quality_profile="balanced_refine"),
        CandidateSpec(name="e", profile="compact", quality_profile="hpwl_refine"),
    ]
    optimizer = ArchitectureV4Optimizer()

    assert optimizer._candidate_worker_count_for_specs(specs) == 4
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_quality_portfolio_uses_bounded_sample_local_workers -q
```

Expected: FAIL because current worker policy returns `1` unless `FLOORSET_CANDIDATE_WORKERS` is set.

- [ ] **Step 3: Implement portfolio worker policy**

Update `_candidate_worker_count_for_specs`:

```python
def _candidate_worker_count_for_specs(self, specs: list[CandidateSpec]) -> int:
    if any(spec.repair_profile != "normal" for spec in specs):
        return 1
    if any(spec.quality_profile != "default" for spec in specs):
        raw = os.environ.get("FLOORSET_QUALITY_PORTFOLIO_WORKERS", "auto").strip().lower()
        if raw == "auto":
            cpu_count = os.cpu_count() or 1
            return max(1, min(len(specs), 4, cpu_count))
        try:
            requested = int(raw)
        except ValueError:
            return 1
        return max(1, min(requested, len(specs), os.cpu_count() or 1))
    return self._candidate_workers(len(specs))
```

- [ ] **Step 4: Run worker-count test**

Run:

```bash
uv run pytest tests/test_optimizer.py::test_quality_portfolio_uses_bounded_sample_local_workers -q
```

Expected: PASS.

## Task 4: Ranking and Verification

**Files:**
- Modify: `src/floorset_arch/optimizer.py`
- Modify: `docs/optimization-notes.md`
- Create: `docs/evaluation/2026-05-21-quality-portfolio-v1.md`

- [ ] **Step 1: Make no-runtime proxy ranking the portfolio default only for portfolio candidates**

Update `_candidate_rank` so portfolio candidates can be selected by no-runtime proxy without changing non-portfolio default behavior:

```python
def _select_best_candidate(self, inst, candidates: list[Placement]) -> Placement:
    return min(candidates, key=lambda placement: self._candidate_rank(inst, placement))
```

If spec-aware ranking is needed, change `_build_candidates` to return `(spec, placement)` pairs and rank with spec metadata. Prefer avoiding that unless tests show no-runtime candidates are generated but not selected.

- [ ] **Step 2: Run unit suite**

Run:

```bash
uv run pytest tests/test_optimizer.py tests/test_repair.py tests/test_relative_order.py tests/test_quality_portfolio.py -q
```

Expected: PASS.

- [ ] **Step 3: Run configured-checkpoint full evaluation**

Run:

```bash
env -u FLOORSET_ENABLE_SURROGATE_GUIDANCE bash scripts/eval_total.sh --output /home/yyds/Data-Driven-SoC-Floorplanning/artifacts/configured_checkpoint_quality_portfolio_v1.json
```

Promotion criteria:

- Total score is less than `2.3086`.
- No-runtime total is less than `2.0362`.
- Feasible count is `100/100`.

- [ ] **Step 4: Run no-checkpoint ablation**

Run:

```bash
env FLOORSET_GNN_CHECKPOINT=/tmp/floorset-no-gnn-missing.pt bash scripts/eval_total.sh --output /home/yyds/Data-Driven-SoC-Floorplanning/artifacts/no_checkpoint_quality_portfolio_v1.json
```

Expected: feasibility remains `100/100`. Treat no-checkpoint improvement as supporting evidence only; do not promote if configured-checkpoint quality regresses.

- [ ] **Step 5: Document result**

Create `docs/evaluation/2026-05-21-quality-portfolio-v1.md` with:

```markdown
# Quality Portfolio v1 Evaluation

## Decision

Accepted or rejected as default based on configured-checkpoint full evaluation.

## Configured Checkpoint

| Run | Total | No-runtime | Feasible | Avg runtime | Median runtime | P90 runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline high-risk normal-only | `2.3086` | `2.0362` | `100/100` | `1.04s` | `1.05s` | `1.99s` |
| Quality Portfolio v1 | `...` | `...` | `...` | `...` | `...` | `...` |

## No-Checkpoint Ablation

| Run | Total | No-runtime | Feasible |
| --- | ---: | ---: | ---: |
| No-checkpoint edge shrink | `6.5204` | `4.5038` | `100/100` |
| Quality Portfolio v1 | `...` | `...` | `...` |
```

Also update `docs/optimization-notes.md` with one bullet summarizing the decision and key numbers.

## Self-Review

- Spec coverage: The plan covers gated high-risk portfolio, non-GNN candidates, sample-local parallelism, checkpoint/no-checkpoint verification, and documentation.
- Placeholder scan: The only ellipses are in the documentation template for post-evaluation values; replace them before completing Task 4.
- Type consistency: `quality_profile` belongs to `CandidateSpec`; `refine_quality_candidate(inst, placement, config, profile)` is the shared portfolio hook; portfolio policy lives in `quality_portfolio.py`.
