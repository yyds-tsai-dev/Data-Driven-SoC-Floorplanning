# G0-v2 Transient-fp Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the formal G0-v2 oracle over the complete receipt-bound eligible heldout population using the frozen production `3 Direct + 3 Flow` base and one transient training-only `fp_sol` exact-TFDL candidate.

**Architecture:** Put fail-closed per-case conversion, admission, official scoring, sparse-label extraction, and weighted population accounting in a small importable module. Keep the CLI responsible for verified one-pass shard discovery, frozen production optimizer construction, canonical streaming evidence, and atomic finalization. Dense fp coordinates never cross the per-case call boundary.

**Tech Stack:** Python 3.12, PyTorch CPU float64 for teacher geometry, SQLite/JSONL for bounded evidence, existing `partner.icdc` TFDL and sparse-label code, submission `MyOptimizer`, and `scripts/iccad2026_evaluate.py`.

## Global Constraints

- Formal population: every receipt-bound row with `n >= 100` and `split_for_id(instance_id, heldout_mod=10) == "heldout"`; `--max-files` is a non-authorizing tracer only.
- Frozen base: exactly six production candidates, `PARTNER_NREF=6`, `PARTNER_FLOW_SLOTS=3`, `PARTNER_OVERSAMPLE=1`, Direct DPM++/2, Flow Euler/8.
- Candidate set: exactly `0=production-base`, `1=transient-fp-exact-tfdl`; no blind mutation bank.
- `fp_sol` is training-only and converted exactly `(w,h,x,y) -> (x,y,w,h)` on CPU float64. Soft V is scoreable, not a hard rejection.
- Dense fp coordinates may not be serialized, persisted in SQLite, used as a student target/input, or reach validation/test or inference.
- The provided/local evaluator source SHA256 is `7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1`.
- QA PDF SHA256 is `60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db`.
- G0 hard gain is `0.0181504738793652`; training authority requires target gain `0.0261247299384228` over the complete population.
- No package review or validation full100 is run before G0 and subsequent G1 justify production integration.

---

### Task 1: Fail-closed two-slot oracle core

**Files:**
- Create: `partner/icdc/g0_v2.py`
- Create: `tests/test_icdc_g0_v2.py`

**Interfaces:**
- Consumes: sanitized case mappings, receipt-bound training `fp_sol`, a production base `(N,4)` tensor, and injected admission/scorer callables.
- Produces: `transient_fp_xywh(source, row, n) -> torch.Tensor`, `evaluate_case(base, fp_seed, case, scorer, ...) -> G0CaseResult`, `PopulationAccumulator.add(result)`, and `PopulationAccumulator.finish(complete=True) -> G0Summary`.

- [ ] **Step 1: Write failing source-role and transient-lifetime tests**

```python
def test_transient_fp_converts_whxy_to_xywh_and_tree_is_invariant():
    source_a, receipt = bound_source(tree_value=0)
    source_b, _ = bound_source(tree_value=9)
    assert transient_fp_xywh(source_a, receipt.layout_index, 2).tolist() == [
        [10.0, 20.0, 3.0, 4.0], [30.0, 40.0, 5.0, 6.0]
    ]
    assert semantic_case(source_a, receipt) == semantic_case(source_b, receipt)

def test_case_result_and_canonical_json_contain_no_dense_coordinates():
    result = evaluate_case(...)
    encoded = canonical_case_record(result)
    assert b"fp_sol" not in encoded
    assert b"fp_xywh" not in encoded
    assert b"positions" not in encoded
    assert b"rects" not in encoded
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `uv run pytest tests/test_icdc_g0_v2.py -q`

Expected: collection or import failure because `partner.icdc.g0_v2` does not exist.

- [ ] **Step 3: Add immutable result types and exact conversion**

```python
@dataclass(frozen=True)
class G0CaseResult:
    instance_id: str
    n: int
    base_cost: float
    teacher_cost: float
    winner: Literal["production-base", "transient-fp-exact-tfdl"]
    teacher_status: str
    sparse_label: TopologyLabel
    hard_audit: Mapping[str, bool]

def transient_fp_xywh(source, row, n):
    raw = source[5][row, :n].to(device="cpu", dtype=torch.float64).contiguous()
    return torch.stack((raw[:, 2], raw[:, 3], raw[:, 0], raw[:, 1]), dim=1).contiguous()
```

Validate exact source width/order, CPU/f64/finite/positive dimensions, immutable inputs, and exact row bounds. Do not retain the returned tensor in `G0CaseResult`.

- [ ] **Step 4: Add RED tests for hard legality, soft-V acceptance, and base retention**

```python
def test_soft_v_fp_is_admitted_and_official_cost_selects_winner(...):
    out = evaluate_case(base, fp, case_with_soft_group_v, scorer)
    assert out.winner == "transient-fp-exact-tfdl"

@pytest.mark.parametrize("teacher_cost,status", [(2.0, "not_improved"), (None, "rejected")])
def test_non_improving_or_rejected_teacher_retains_base(teacher_cost, status, ...):
    out = evaluate_case(...)
    assert out.winner == "production-base"
    assert out.teacher_cost == out.base_cost
```

The injected call trace must prove both layouts use official `cost_no_runtime`, exact admission occurs before teacher scoring, preplaced origins remain bit-exact, and diagnostic energy cannot affect selection.

- [ ] **Step 5: Implement minimal two-slot evaluation and sparse extraction**

Use `pin_feasible_then_exact_tfdl`, `engine.verify_hard_legal`, and `extract_sparse_label`. Score the base once; score the teacher only if admitted and hard legal. Select by `(cost_no_runtime, ordinal, name)`. Convert results immediately to scalar audit data plus `TopologyLabel`, then delete local dense tensors before returning.

- [ ] **Step 6: Add RED population accounting tests**

```python
def test_population_uses_exp_n_over_12_and_requires_complete_registration():
    pop = PopulationAccumulator()
    pop.register("a", 100)
    pop.register("b", 112)
    pop.add(case_result("a", 100, 1.3, 1.1))
    with pytest.raises(ValueError, match="coverage"):
        pop.finish(complete=True)
    pop.add(case_result("b", 112, 1.2, 1.0))
    summary = pop.finish(complete=True)
    assert summary.delta_h == pytest.approx(weighted([1.3, 1.2], [100, 112]) - weighted([1.1, 1.0], [100, 112]))
```

Cover duplicate IDs, nonfinite weight/product/sum, base-unavailable, incomplete coverage, terminal-state precedence, and `TARGET_GAIN_MET` only at `Delta_H >= 0.0261247299384228`.

- [ ] **Step 7: Implement bounded exact population accounting and run GREEN**

Store registration and results in SQLite or constant-memory compensated sums. Bind sorted `(instance_id,n,weight)` to `population_sha256`, require one result for every registered ID, and reject nonfinite intermediate/final values.

Run: `uv run pytest tests/test_icdc_g0_v2.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add partner/icdc/g0_v2.py tests/test_icdc_g0_v2.py
git commit -m "feat: add transient fp g0 oracle core"
```

### Task 2: Streaming formal runner and sealed evidence

**Files:**
- Create: `scripts/probes/icdc_fp_teacher_g0.py`
- Modify: `tests/test_icdc_g0_v2.py`

**Interfaces:**
- Consumes: canonical `FloorSet/floorset_lite`, frozen submission wrapper/checkpoints, official scorer, `partner.icdc.g0_v2`.
- Produces: CLI `main(argv) -> int`, canonical `cases.jsonl`, sparse `labels.jsonl`, `population.json`, and `manifest.json` in an atomically published directory.

- [ ] **Step 1: Write RED tests for the exact frozen production contract**

```python
def test_production_environment_is_exact_3d3f():
    assert production_environment() == {
        "PARTNER_NREF": "6", "PARTNER_FLOW_SLOTS": "3",
        "PARTNER_OVERSAMPLE": "1",
        "PARTNER_DIRECT_SOLVER": "dpmpp", "PARTNER_DDIM_STEPS": "2",
        "PARTNER_FLOW_SOLVER": "euler", "PARTNER_FLOW_STEPS": "8",
    }
```

Add a fake optimizer trace requiring exactly one base solve per eligible case and an explicit receipt proving six candidates split 3/3. Reject fallback, pool failure, any other mix, source/checkpoint/wrapper/scorer hash mismatch, symlink root, duplicate/noncanonical shards, and validation/test loaders before optimizer construction.

- [ ] **Step 2: Write RED streaming and artifact-hygiene tests**

Run two real tiny receipt-bound shards through injected optimizer/admission/scorer seams. Assert numeric `(worker,layout,row)` order, one verified bytes read per shard, and immediate `register -> solve -> record` processing for each eligible row without whole-corpus retention or a source reread. Recursively reject dense coordinate keys and any numeric `[N,4]` value in every persisted artifact. Reverse discovery order and require byte-identical outputs.

- [ ] **Step 3: Run the focused suite and confirm RED**

Run: `uv run pytest tests/test_icdc_g0_v2.py -q`

Expected: runner import/API failures while Task 1 remains green.

- [ ] **Step 4: Implement verified discovery and lazy production setup**

Reuse the fail-closed byte-loading and raw-source validation semantics from `icdc_topology_teacher.py`, but do not import validation/test data loaders. Set and verify the six frozen environment values before importing `submission/cadc1013/op_wrapper.py`. Hash the wrapper, source, Direct and Flow checkpoints, scorer, QA PDF, and current commit into the manifest.

- [ ] **Step 5: Implement per-case production base and transient teacher flow**

For every registered eligible row:

```python
base = optimizer.solve(n, area, b2b, p2b, pins, cons, target_positions=tp)
fp_seed = transient_fp_xywh(source, row, n)
result = evaluate_case(base, fp_seed, case, scorer)
del fp_seed
writer.write(canonical_case_record(result))
labels.write(canonical_sparse_label(result.sparse_label))
population.add(result)
```

The real implementation must normalize exact tensor shapes/dtypes and verify the optimizer portfolio trace before accepting `base`.

- [ ] **Step 6: Implement atomic evidence publication and gate exit codes**

Write support files into a private same-parent staging directory, flush/fsync/close them, compute hashes, write `manifest.json` last, fsync the staging directory, and publish with no-replace rename. Any exception leaves no completed destination. Formal `TARGET_GAIN_MET` returns zero; every killed/stop state returns nonzero. A `--max-files` run always records `authorizing=false` and returns a distinct nonzero tracer status even if numeric thresholds pass.

- [ ] **Step 7: Run GREEN and relevant regressions**

Run:

```bash
uv run pytest tests/test_icdc_g0_v2.py -q
uv run pytest tests/test_icdc_tfdl.py tests/test_icdc_topology_prior.py -q
uv run python -m py_compile partner/icdc/g0_v2.py scripts/probes/icdc_fp_teacher_g0.py
git diff --check
graphify update .
```

Expected: all tests pass, compile succeeds, diff check is empty; graph updates stay unstaged unless intentionally committed.

- [ ] **Step 8: Commit Task 2**

```bash
git add scripts/probes/icdc_fp_teacher_g0.py tests/test_icdc_g0_v2.py
git commit -m "feat: add streaming g0 v2 runner"
```

### Task 3: Tracer, full eligible heldout G0, and training handoff

**Files:**
- Create: `docs/experiments/2026-08-14-g0-v2-transient-fp.md`
- Generated, not committed: `artifacts/icdc_g0_v2_tracer/`, `artifacts/icdc_g0_v2_full/`

**Interfaces:**
- Consumes: the Task 2 CLI and frozen source/production identities.
- Produces: a non-authorizing tracer receipt, then the formal complete-H terminal manifest and sparse training authority only if `TARGET_GAIN_MET`.

- [ ] **Step 1: Run a 100-shard non-authorizing tracer**

```bash
uv run python scripts/probes/icdc_fp_teacher_g0.py \
  --data-root FloorSet/floorset_lite \
  --out-dir artifacts/icdc_g0_v2_tracer \
  --max-files 100
```

Expected: complete deterministic tracer evidence with `authorizing=false`, no dense coordinates, 100% base coverage, and no contract/hash failures. Numeric gain is diagnostic only.

- [ ] **Step 2: Inspect only gate and hygiene evidence**

Run the runner's manifest verifier and compare counts/hashes, not validation/full100 outcomes. If the tracer fails, fix the runner or stop on a real contract violation; do not tune thresholds or candidate policy from its score.

- [ ] **Step 3: Run formal G0 over complete H**

```bash
uv run python scripts/probes/icdc_fp_teacher_g0.py \
  --data-root FloorSet/floorset_lite \
  --out-dir artifacts/icdc_g0_v2_full
```

Expected: every eligible ID registered and scored exactly once, complete population SHA/denominator, no missing base, no dense persistence, and one immutable terminal state.

- [ ] **Step 4: Apply the predeclared gate without reinterpretation**

Advance to same-shape student training only when the formal manifest says `TARGET_GAIN_MET` and `Delta_H >= 0.0261247299384228`. Otherwise record the exact killed/stop state and do not train or run G1.

- [ ] **Step 5: Record experiment evidence and commit**

Document command, source commit, immutable artifact hashes, population count/SHA, `B_H`, `T_H`, `Delta_H`, coverage, terminal state, elapsed time, and the next authorized action.

```bash
git add docs/experiments/2026-08-14-g0-v2-transient-fp.md
git commit -m "docs: record formal g0 v2 result"
```
