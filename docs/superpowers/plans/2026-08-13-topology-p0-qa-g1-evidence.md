# Topology P0: QA and G1 Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the source-role, QA, no-dense-artifact, and sealed G1 evidence contracts that `fp_topology_v1` and the offline topology teacher will consume.

**Architecture:** Keep raw FloorSet interpretation in `topology_data`, contest/QA truth in one read-only contract adapter, artifact hygiene in one recursive validator, and G1 adjudication in a standalone canonical-evidence module. The existing teacher probe becomes a thin caller of those public contracts; P0 adds no optimizer behavior, topology extraction, online work, or full100 execution.

**Tech Stack:** Python 3.12, PyTorch CPU tensors, NumPy, pytest, `uv`, JSON/JSONL, SHA256, the checked-in contest evaluator, and the checked-in QA PDF/Alpha CSV.

## Global Constraints

- P0 is first: P1 may not start until every P0 targeted test, the P0 full test command, `git diff --check`, and the P0 API handoff review pass.
- The raw source is exactly seven CPU finite tensor entries in this order: `source[0]` input `[B,N,6]`; `source[1]` b2b `[B,*,3]`; `source[2]` p2b `[B,*,3]`; `source[3]` pins `[B,*,2]`; `source[4]` `tree_sol` `[B,N-1,3]`; `source[5]` `fp_sol` `[B,N,4]` stored `(w,h,x,y)`; and `source[6]` `metrics_sol` `[B,8]`.
- `tree_sol` is schema/dtype/shape/padding evidence only. It is never a model input, label, topology input, candidate, loss, cost-selection, or checkpoint-selection input. A valid tree-only mutation may change the raw receipt SHA but must not change a semantic teacher/training result.
- Training `fp_sol` is receipt-bound, transient, training-only supervision. Convert it exactly once as `(w,h,x,y) -> (x,y,w,h)`; never serialize it, put it into the student, seed an online proposal from it, or use it as a coordinate/value loss.
- Validation and test data are forbidden for training gradients, thresholds, proposals, tuning, checkpoint selection, and full100 feedback. Production source admission accepts only the canonical non-validation root `FloorSet/floorset_lite`; tests may inject a trust policy only through a Python-only keyword.
- QA authority is `docs/official/C_QA_20260804.pdf`, SHA256 `60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db`. Missing bytes, a changed path, or a changed digest is a preflight failure and cannot produce a completed manifest.
- QA A4 makes grouping/MIB/boundary violations scoreable soft V behavior. QA A5 makes preplaced `(x,y,w,h)` hard and boundary soft. QA A6 accepts area ratios exactly `99/100` and `101/100`, and hard-fails `98.99/100` and `101.01/100`. QA A15 fixes `max(0.7, max(0.01, r/m)**0.3)`. QA A16 makes `fp_sol`, not `tree_sol`, the training layout source.
- The provided/local scorer contract is literal `iccad2026_evaluate_cost_no_runtime_v1`: call `evaluate_solution({"positions": positions, "runtime": 1.0}, baseline_metrics, target_constraints, b2b_connectivity, p2b_connectivity, pins_pos, target_areas, target_positions, median_runtime=1.0)` and select only finite `.cost_no_runtime`.
- Every JSON/JSONL evidence payload is UTF-8, `json.dumps(..., sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)`. `assert_no_dense_fp_artifact` rejects dense fp/rectangle/coordinate fields before any artifact write.
- A G1 arm has exactly 100 ordered rows `(case_id, n, runtime, cost_no_runtime)`. Runtime p90 is `sorted(runtimes)[89]`, never an interpolated percentile.
- Alpha medians are only `docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv`, SHA256 `804c3432febb88a8f8ee0a8c0ede4b4598a5cf3c109d68de6a6ca86f05d211bd`. G1 weighs rows by `exp(n/12)` and passes only when `candidate_combined <= control_combined + 1e-12`.
- `internal_runtime_target_met = (mean_runtime <= 0.300)` is a recorded diagnostic, never a G1 pass/fail condition. Feasibility, zero errors, matching ordered IDs and Alpha medians, frozen identities, causal smoke, and per-case exact 3 Direct/3 Flow remain mandatory gates.
- P0 unit and integration tests use temporary fixtures only. Do not invoke package review, `scripts/eval_total.sh`, a full100 evaluator, a blind arm, G0, or a production checkpoint action without a separately authorized goal policy.

---

## Locked file map

| File | Responsibility |
| --- | --- |
| `src/icdc_engine/topology_data.py` | Validate the seven-tensor source and expose an in-memory, receipt-bound training row containing sanitized input plus transient `(x,y,w,h)` fp geometry. |
| `src/icdc_engine/topology_artifact_guard.py` | Canonically encode evidence and reject any dense fp/rectangle/coordinate payload before persistence. |
| `src/icdc_engine/qa_contract.py` | Pin the QA PDF and provided/local evaluator identities, expose QA manifest fields, and produce a score/audit record with hard versus soft semantics. |
| `src/icdc_engine/g1_evidence.py` | Validate, canonically serialize, seal, and compare matched 100-row G1 arm evidence. |
| `scripts/probes/icdc_topology_teacher.py` | Remain an I/O and transaction adapter; replace private raw-source and QA checks with P0 imports and add no duplicate policy logic. |
| `scripts/probes/icdc_topology_g1_evidence.py` | Fixture-safe CLI that reads two sealed arm JSON files and writes one canonical comparison evidence file. |
| `tests/test_topology_p0_contracts.py` | Source-role, QA, artifact-hygiene, and evaluator regression fixtures. |
| `tests/test_icdc_g1_evidence.py` | G1 row, serialization, Alpha, comparator, and CLI fixtures. |

## P1 handoff contracts

P1 consumes these exact P0 APIs. Do not rename, widen, or copy their behavior in P1.

```python
@dataclass(frozen=True)
class VerifiedTrainingFpRow:
    receipt: CorpusSourceReceipt
    instance_id: str
    input_fingerprint: str
    case: Mapping[str, object]          # sanitized input only; no dense target
    fp_xywh: torch.Tensor                # CPU float64 [N,4], transient only

`validate_raw_source(source: Sequence[torch.Tensor]) -> tuple[int, int]`.

`verified_training_fp_row(source: Sequence[torch.Tensor], receipt: CorpusSourceReceipt) -> VerifiedTrainingFpRow`.

`assert_no_dense_fp_artifact(value: object) -> None`.

@dataclass(frozen=True)
class QAContractEvidence:
    qa_relative_path: str
    qa_sha256: str
    scorer_relative_path: str
    scorer_sha256: str
    scorer_contract: str
    shapely_version: str

@dataclass(frozen=True)
class LocalScoreAudit:
    feasible: bool
    cost_no_runtime: float
    boundary_violations: int
    grouping_violations: int
    mib_violations: int
    area_violations: int
    dimension_violations: int

`preflight_qa_contract(repo_root: Path) -> QAContractEvidence`.

`qa_manifest_fields(evidence: QAContractEvidence) -> dict[str, str]`.

`score_provided_local_no_runtime(case: Mapping[str, object], rects_xywh: torch.Tensor, evidence: QAContractEvidence) -> LocalScoreAudit`.
```

`VerifiedTrainingFpRow.fp_xywh` may flow only to P1's extractor in memory. P1 must call `assert_no_dense_fp_artifact` on every JSONL row, label record, proposal record, and manifest before writing, and must use `QAContractEvidence` rather than re-reading the PDF/scorer identity.

## Execution constraints and review gates

For every task below, the scheduler first performs the RED-review gate: a separate Terra reviewer (`gpt-5.6-terra`, `xhigh`, read-only) checks the quoted test, exact contract, file ownership, and expected RED reason. A fresh Luna implementer (`gpt-5.6-luna`, `low`) then owns only the named files, preserves concurrent edits, runs the listed commands, and creates the listed atomic commit. A different Terra reviewer performs the re-review gate on the actual diff, test output, no-dense scan, and interface compatibility; Important or Critical findings return to a fresh Luna before the next task. Sol integrates only after inspecting the files and evidence itself.

Do not edit `src/icdc_engine/{tfdl.py,engine.py,energy.py}`, production optimizer/submission files, evaluator source, checkpoints, artifacts, or unrelated tests. Never inspect or edit `scratchpad/`. After every code modification run `graphify update .`; leave its expected dirty graph artifacts out of each task commit.

### Task 1: Make raw-source roles and training-only fp conversion explicit

**Files:**

- Modify: `src/icdc_engine/topology_data.py`
- Modify: `scripts/probes/icdc_topology_teacher.py:1597-1649,1991-2010,2222-2245`
- Test: `tests/test_topology_p0_contracts.py`

**Interfaces:**

- Produces `VerifiedTrainingFpRow`, `validate_raw_source(source) -> tuple[int, int]`, and `verified_training_fp_row(source, receipt) -> VerifiedTrainingFpRow` exactly as in the P1 handoff.
- The teacher probe consumes only these functions for source validation and row construction; it may retain its transaction code.
- `fp_xywh` is CPU `torch.float64`, shape `[N,4]`, and is never placed in `case`.

- [ ] **Step 1: Write the failing source-role and tree-invariance test.**

```python
def _raw_source(tree_value: float = 0.0) -> list[torch.Tensor]:
    inp = torch.tensor([[[4., 0., 0., 0., 0., 0.], [9., 1., 1., 0., 0., 0.]]])
    b2b = torch.tensor([[[0., 1., 1.]]])
    p2b = torch.empty((1, 0, 3))
    pins = torch.empty((1, 0, 2))
    tree = torch.tensor([[[tree_value, 0., 1.]]])
    fp_sol = torch.tensor([[[2., 2., 11., 13.], [3., 3., 17., 19.]]])
    metrics = torch.tensor([[25., 0., 0., 0., 0., 0., 3., 5.]])
    return [inp, b2b, p2b, pins, tree, fp_sol, metrics]

def test_verified_training_row_uses_fp_not_tree_and_keeps_fp_transient():
    receipt_a = CorpusSourceReceipt("worker_0/layouts_0.th", "a" * 64, 0, "b" * 64)
    receipt_b = CorpusSourceReceipt("worker_0/layouts_0.th", "c" * 64, 0, "b" * 64)
    first = verified_training_fp_row(_raw_source(0.0), receipt_a)
    second = verified_training_fp_row(_raw_source(9.0), receipt_b)
    assert validate_raw_source(_raw_source()) == (1, 2)
    assert first.fp_xywh.dtype is torch.float64 and first.fp_xywh.device.type == "cpu"
    assert first.fp_xywh.tolist() == [[11.0, 13.0, 2.0, 2.0], [17.0, 19.0, 3.0, 3.0]]
    assert first.case == second.case
    assert torch.equal(first.fp_xywh, second.fp_xywh)
    assert first.receipt.file_sha256 != second.receipt.file_sha256
    assert "fp_xywh" not in first.case and "tree_sol" not in first.case
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_p0_contracts.py::test_verified_training_row_uses_fp_not_tree_and_keeps_fp_transient -q`

Expected: FAIL because `VerifiedTrainingFpRow` or `verified_training_fp_row` is not exported, or because the old probe-local parser still exposes a tree/fp target through the sanitized case.

- [ ] **Step 3: Add the minimal source-contract implementation.**

```python
@dataclass(frozen=True)
class VerifiedTrainingFpRow:
    receipt: CorpusSourceReceipt
    instance_id: str
    input_fingerprint: str
    case: Mapping[str, object]
    fp_xywh: torch.Tensor

def validate_raw_source(source: Sequence[torch.Tensor]) -> tuple[int, int]:
    if not isinstance(source, (list, tuple)) or len(source) != 7:
        raise ValueError("source schema")
    inp, b2b, p2b, pins, tree_sol, fp_sol, metrics_sol = source
    expected = ((inp, 3, 6), (b2b, 3, 3), (p2b, 3, 3), (pins, 3, 2),
                (tree_sol, 3, 3), (fp_sol, 3, 4), (metrics_sol, 2, 8))
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.requires_grad
           or not t.is_floating_point() or not bool(torch.isfinite(t).all())
           or t.ndim != rank or t.shape[-1] != width for t, rank, width in expected):
        raise ValueError("source tensors")
    batch, n = int(inp.shape[0]), int(inp.shape[1])
    if batch < 1 or n < 1 or tree_sol.shape[1] != n - 1:
        raise ValueError("source shape")
    if any(t.shape[0] != batch for t in (b2b, p2b, pins, tree_sol, fp_sol, metrics_sol)):
        raise ValueError("source batch")
    return batch, n

def _fp_sol_xywh(fp_sol: torch.Tensor, row: int, n: int) -> torch.Tensor:
    raw = fp_sol[row, :n].to(dtype=torch.float64, device="cpu").contiguous()
    return torch.stack((raw[:, 2], raw[:, 3], raw[:, 0], raw[:, 1]), dim=1)
```

Add `verified_training_fp_row` by calling the existing padding validators, deriving the sanitized input-only `case`, computing `input_fingerprint = fingerprint_case(case)`, and returning `_fp_sol_xywh(source[5], receipt.layout_index, case["n"])`. Do not read `source[4]` after `validate_raw_source`; replace the teacher probe's `_validate_source_shard` and `_source_case_from_shard` bodies with calls to these public functions.

- [ ] **Step 4: Run the green and source-regression commands.**

Run: `uv run pytest tests/test_topology_p0_contracts.py::test_verified_training_row_uses_fp_not_tree_and_keeps_fp_transient -q`

Expected: PASS.

Run: `uv run pytest tests/test_topology_p0_contracts.py -q && uv run pytest tests/test_icdc_topology_prior.py -q`

Expected: PASS; the legacy receipt tests still receive the same sanitized schema and raw source validation rejects malformed tree shape/padding.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add src/icdc_engine/topology_data.py scripts/probes/icdc_topology_teacher.py tests/test_topology_p0_contracts.py && git commit -m "feat: bind topology source roles"`

### Task 2: Block dense fp data from evidence artifacts

**Files:**

- Create: `src/icdc_engine/topology_artifact_guard.py`
- Test: `tests/test_topology_p0_contracts.py`

**Interfaces:**

- Produces `canonical_json_bytes(value: object) -> bytes`, `assert_no_dense_fp_artifact(value: object) -> None`, and `write_checked_json(path: Path, value: object) -> str`.
- P1 consumes the first two functions before every proposal/label/manifest write; `write_checked_json` returns the lowercase SHA256 of exactly the written bytes.

- [ ] **Step 1: Write the failing dense-leak test.**

```python
def test_checked_artifacts_reject_dense_fp_but_keep_sparse_topology(tmp_path):
    sparse = {"schema": "fp_topology_v1", "version": 1,
              "axis_edges": [{"src": 0, "dst": 1, "axis": 0}],
              "contacts": []}
    assert_no_dense_fp_artifact(sparse)
    digest = write_checked_json(tmp_path / "sparse.json", sparse)
    assert digest == hashlib.sha256((tmp_path / "sparse.json").read_bytes()).hexdigest()
    for bad in ({"fp_sol": [[1., 2., 3., 4.]]},
                {"rects": [[0., 0., 1., 1.]]},
                {"axis_edges": [], "origin": 2.0},
                {"contacts": [], "overlap_magnitude": 1.0}):
        with pytest.raises(ValueError, match="dense fp artifact"):
            assert_no_dense_fp_artifact(bad)
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_p0_contracts.py::test_checked_artifacts_reject_dense_fp_but_keep_sparse_topology -q`

Expected: FAIL with `ModuleNotFoundError: icdc.topology_artifact_guard`.

- [ ] **Step 3: Add the recursive canonical guard.**

```python
_DENSE_KEYS = frozenset({
    "fp_sol", "fp_xywh", "dense_fp", "rects", "positions", "origin", "width",
    "height", "x", "y", "w", "h", "gap", "overlap", "overlap_magnitude",
})

def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("canonical JSON") from exc

def assert_no_dense_fp_artifact(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key in _DENSE_KEYS:
                raise ValueError("dense fp artifact")
            assert_no_dense_fp_artifact(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_dense_fp_artifact(child)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        return
    else:
        raise ValueError("canonical JSON")

def write_checked_json(path: Path, value: object) -> str:
    assert_no_dense_fp_artifact(value)
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 4: Run the green and artifact-regression commands.**

Run: `uv run pytest tests/test_topology_p0_contracts.py::test_checked_artifacts_reject_dense_fp_but_keep_sparse_topology -q`

Expected: PASS.

Run: `uv run pytest tests/test_topology_p0_contracts.py -q`

Expected: PASS; canonical bytes reject NaN, tensors, and forbidden dense key paths before a file is created.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add src/icdc_engine/topology_artifact_guard.py tests/test_topology_p0_contracts.py && git commit -m "feat: guard topology evidence artifacts"`

### Task 3: Pin QA authority and hard-versus-soft local scoring

**Files:**

- Create: `src/icdc_engine/qa_contract.py`
- Test: `tests/test_topology_p0_contracts.py`

**Interfaces:**

- Produces `QAContractEvidence`, `LocalScoreAudit`, `preflight_qa_contract(repo_root)`, `qa_manifest_fields(evidence)`, and `score_provided_local_no_runtime(case, rects_xywh, evidence)`.
- P1 receives a preflighted `QAContractEvidence`; the scorer adapter returns only aggregate evaluator facts and never coordinates.

- [ ] **Step 1: Write the failing QA preflight and semantic fixtures.**

```python
def test_qa_preflight_and_qa_a4_a5_a6_semantics(tmp_path, monkeypatch):
    evidence = preflight_qa_contract(Path.cwd())
    assert qa_manifest_fields(evidence)["qa_sha256"] == "60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db"
    case = _qa_case_with_preplaced_boundary_and_soft_group()
    soft_v = score_provided_local_no_runtime(
        case, torch.tensor([[0., 0., 10., 10.], [20., 0., 10., 10.]], dtype=torch.float64), evidence)
    assert soft_v.feasible and soft_v.boundary_violations == 1 and soft_v.grouping_violations == 1
    assert soft_v.cost_no_runtime < 10.0
    for side, feasible in ((99.0, True), (101.0, True), (98.99, False), (101.01, False)):
        area_case = _qa_area_case(100.0)
        audit = score_provided_local_no_runtime(
            area_case, torch.tensor([[0., 0., side, 1.]], dtype=torch.float64), evidence)
        assert audit.feasible is feasible
    moved = torch.tensor([[1., 0., 10., 10.], [20., 0., 10., 10.]], dtype=torch.float64)
    assert not score_provided_local_no_runtime(case, moved, evidence).feasible
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"changed" if self.name == "C_QA_20260804.pdf" else Path.read_bytes(self))
    with pytest.raises(ValueError, match="QA PDF SHA256"):
        preflight_qa_contract(Path.cwd())
```

`_qa_case_with_preplaced_boundary_and_soft_group` returns a two-row full five-column `cons` with block 0 preplaced and boundary-constrained, a disconnected nonzero cluster, evaluator baselines, empty legal relation tails, and `tp[0] == [0,0,10,10]`. `_qa_area_case` returns one soft block with target area `100.0`; both helpers are defined in this test file before this node.

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_p0_contracts.py::test_qa_preflight_and_qa_a4_a5_a6_semantics -q`

Expected: FAIL with `ModuleNotFoundError: icdc.qa_contract`.

- [ ] **Step 3: Add the QA/scorer adapter.**

```python
QA_RELATIVE_PATH = "docs/official/C_QA_20260804.pdf"
QA_SHA256 = "60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db"
SCORER_RELATIVE_PATH = "scripts/iccad2026_evaluate.py"
SCORER_CONTRACT = "iccad2026_evaluate_cost_no_runtime_v1"

def preflight_qa_contract(repo_root: Path) -> QAContractEvidence:
    qa_path = repo_root / QA_RELATIVE_PATH
    if not qa_path.is_file() or hashlib.sha256(qa_path.read_bytes()).hexdigest() != QA_SHA256:
        raise ValueError("QA PDF SHA256")
    scorer_path = repo_root / SCORER_RELATIVE_PATH
    if not scorer_path.is_file():
        raise ValueError("scorer source")
    return QAContractEvidence(QA_RELATIVE_PATH, QA_SHA256, SCORER_RELATIVE_PATH,
                              hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
                              SCORER_CONTRACT, shapely.__version__)

def qa_manifest_fields(evidence: QAContractEvidence) -> dict[str, str]:
    return {"qa_relative_path": evidence.qa_relative_path, "qa_sha256": evidence.qa_sha256,
            "scorer_relative_path": evidence.scorer_relative_path,
            "scorer_sha256": evidence.scorer_sha256,
            "scorer_contract": evidence.scorer_contract,
            "shapely_version": evidence.shapely_version}
```

Implement `score_provided_local_no_runtime` by validating CPU float64 `[N,4]`, converting each row to four Python floats, importing the checked-in evaluator, making the literal call from Global Constraints, requiring finite `metrics.cost_no_runtime`, and returning only `is_feasible`, the no-runtime cost, and the five listed integer violation fields. Require the current scorer file digest to equal `evidence.scorer_sha256` before the call.

- [ ] **Step 4: Run the green and evaluator regression commands.**

Run: `uv run pytest tests/test_topology_p0_contracts.py::test_qa_preflight_and_qa_a4_a5_a6_semantics -q`

Expected: PASS.

Run: `uv run pytest tests/test_topology_p0_contracts.py tests/test_evaluator_scoring.py -q`

Expected: PASS; the fixture proves soft V remains feasible, hard preplacement remains immovable, and the inclusive QA A6 endpoints retain the evaluator's hard boundary.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add src/icdc_engine/qa_contract.py tests/test_topology_p0_contracts.py && git commit -m "feat: pin topology QA contract"`

### Task 4: Define canonical sealed G1 rows and Alpha binding

**Files:**

- Create: `src/icdc_engine/g1_evidence.py`
- Test: `tests/test_icdc_g1_evidence.py`

**Interfaces:**

- Produces frozen `G1CaseRow(case_id: str, n: int, runtime: float, cost_no_runtime: float)`, `G1PortfolioReceipt`, `G1ArmEvidence`, `canonical_g1_bytes(value)`, `load_alpha_medians(repo_root)`, and `seal_g1_arm(...)`.
- `seal_g1_arm` returns a record with exactly 100 validated, ordered rows, Alpha medians, runtime `sum/mean/p90/max`, weighted combined score, calculator SHA256, and canonical evidence SHA256.

- [ ] **Step 1: Write the failing row/Alpha/percentile test.**

```python
def _g1_rows(offset: float = 0.0) -> tuple[G1CaseRow, ...]:
    return tuple(G1CaseRow(str(i), 100 + (i % 3), 0.10 + i / 1000.0, 1.0 + offset)
                 for i in range(100))

def _g1_receipts() -> tuple[G1PortfolioReceipt, ...]:
    return tuple(G1PortfolioReceipt(str(i), 3, 3, "dpmpp", 2, "euler", 8, True)
                 for i in range(100))

def test_g1_arm_is_canonical_binds_alpha_and_uses_nearest_rank_p90():
    arm = seal_g1_arm("C0", _g1_rows(), _g1_receipts(), Path.cwd(),
                      feasible_count=100, error_count=0, freeze_sha256="a" * 64,
                      calculator_path=Path("src/icdc_engine/g1_evidence.py"))
    assert arm.runtime_p90 == pytest.approx(sorted(row.runtime for row in _g1_rows())[89])
    assert arm.alpha_relative_path == "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"
    assert arm.alpha_sha256 == "804c3432febb88a8f8ee0a8c0ede4b4598a5cf3c109d68de6a6ca86f05d211bd"
    assert arm.evidence_sha256 == hashlib.sha256(canonical_g1_bytes(arm.to_record())).hexdigest()
    with pytest.raises(ValueError, match="ordered rows"):
        seal_g1_arm("C0", tuple(reversed(_g1_rows())), _g1_receipts(), Path.cwd(),
                    feasible_count=100, error_count=0, freeze_sha256="a" * 64,
                    calculator_path=Path("src/icdc_engine/g1_evidence.py"))
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_icdc_g1_evidence.py::test_g1_arm_is_canonical_binds_alpha_and_uses_nearest_rank_p90 -q`

Expected: FAIL with `ModuleNotFoundError: icdc.g1_evidence`.

- [ ] **Step 3: Add the canonical row and arm implementation.**

```python
ALPHA_RELATIVE_PATH = "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"
ALPHA_SHA256 = "804c3432febb88a8f8ee0a8c0ede4b4598a5cf3c109d68de6a6ca86f05d211bd"

@dataclass(frozen=True)
class G1CaseRow:
    case_id: str
    n: int
    runtime: float
    cost_no_runtime: float

@dataclass(frozen=True)
class G1PortfolioReceipt:
    case_id: str
    direct_count: int
    flow_count: int
    direct_sampler: str
    direct_steps: int
    flow_sampler: str
    flow_steps: int
    normal_pool: bool

def canonical_g1_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def _runtime_summary(rows: Sequence[G1CaseRow]) -> tuple[float, float, float, float]:
    values = sorted(float(row.runtime) for row in rows)
    return sum(values), sum(values) / 100.0, values[89], values[-1]
```

Complete `load_alpha_medians` with `csv.DictReader`, exact path/hash validation, IDs `"0"` through `"99"` once each, and finite positive `median_runtime_s`. Complete `seal_g1_arm` by requiring the exact ordered row IDs `"0"` through `"99"`, matching one receipt per row, `3/3`, `dpmpp/2`, `euler/8`, and `normal_pool=True`; calculate `combined = q * max(0.7, max(0.01, r/max(m, 0.01)) ** 0.3)` and the `exp(n/12)` weighted mean; then hash the complete canonical record with its `evidence_sha256` field omitted.

- [ ] **Step 4: Run the green and canonicalization regression commands.**

Run: `uv run pytest tests/test_icdc_g1_evidence.py::test_g1_arm_is_canonical_binds_alpha_and_uses_nearest_rank_p90 -q`

Expected: PASS.

Run: `uv run pytest tests/test_icdc_g1_evidence.py -q`

Expected: PASS; invalid floats, duplicate/missing IDs, Alpha mismatch, non-100 rows, non-normal pool, and non-3D/3F receipts fail closed.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add src/icdc_engine/g1_evidence.py tests/test_icdc_g1_evidence.py && git commit -m "feat: seal topology G1 evidence"`

### Task 5: Compare matched frozen G1 arms without a runtime-only pass

**Files:**

- Modify: `src/icdc_engine/g1_evidence.py`
- Create: `scripts/probes/icdc_topology_g1_evidence.py`
- Test: `tests/test_icdc_g1_evidence.py`

**Interfaces:**

- Produces frozen `G1Comparison`, `compare_g1_arms(control: G1ArmEvidence, candidate: G1ArmEvidence) -> G1Comparison`, and CLI `main(argv: Sequence[str] | None = None) -> int`.
- `G1Comparison` includes `candidate_combined`, `control_combined`, `binding_pass`, `internal_runtime_target_met`, all mandatory gate booleans, `state`, and a canonical digest. It has no tuning output.

- [ ] **Step 1: Write the failing matched-comparator and CLI test.**

```python
def test_g1_comparison_requires_all_matched_gates_and_keeps_point300_diagnostic(tmp_path):
    control = _sealed_arm("C0", offset=0.0, runtime_offset=0.25)
    candidate = _sealed_arm("C1", offset=-0.01, runtime_offset=0.31)
    result = compare_g1_arms(control, candidate)
    assert result.binding_pass is True
    assert result.internal_runtime_target_met is False
    assert result.state == "HIGH_TAIL_CAUSAL_PROOF"
    bad_ids = dataclasses.replace(candidate, rows=tuple(
        dataclasses.replace(candidate.rows[0], case_id="wrong"), *candidate.rows[1:]))
    with pytest.raises(ValueError, match="matched IDs"):
        compare_g1_arms(control, bad_ids)
    too_expensive = _sealed_arm("C1", offset=0.01, runtime_offset=0.01)
    assert compare_g1_arms(control, too_expensive).state == "KILLED_G1_COMBINED"
```

`_sealed_arm` is a test helper defined in this file: it calls `seal_g1_arm` with the 100 generated rows/receipts from Task 4, `feasible_count=100`, `error_count=0`, matching freeze/environment/causal-smoke digests, and a different arm ID.

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_icdc_g1_evidence.py::test_g1_comparison_requires_all_matched_gates_and_keeps_point300_diagnostic -q`

Expected: FAIL because `compare_g1_arms` and `G1Comparison` do not exist.

- [ ] **Step 3: Add the comparator and fixture-safe CLI.**

```python
@dataclass(frozen=True)
class G1Comparison:
    control_combined: float
    candidate_combined: float
    binding_pass: bool
    internal_runtime_target_met: bool
    matched_ids: bool
    matched_alpha_medians: bool
    portfolio_ok: bool
    feasibility_ok: bool
    freeze_ok: bool
    causal_smoke_ok: bool
    state: str
    comparison_sha256: str

def compare_g1_arms(control: G1ArmEvidence, candidate: G1ArmEvidence) -> G1Comparison:
    if tuple(row.case_id for row in control.rows) != tuple(row.case_id for row in candidate.rows):
        raise ValueError("matched IDs")
    if tuple(row.alpha_median for row in control.rows) != tuple(row.alpha_median for row in candidate.rows):
        raise ValueError("matched Alpha medians")
    binding = candidate.weighted_combined <= control.weighted_combined + 1e-12
    runtime_diagnostic = candidate.runtime_mean <= 0.300
    mandatory = control.portfolio_ok and candidate.portfolio_ok and control.feasibility_ok and candidate.feasibility_ok and control.freeze_ok and candidate.freeze_ok and control.causal_smoke_ok and candidate.causal_smoke_ok
    state = "HIGH_TAIL_CAUSAL_PROOF" if binding and mandatory else "KILLED_G1_COMBINED"
    return _hashed_comparison(control, candidate, binding, runtime_diagnostic, mandatory, state)
```

Make `_hashed_comparison` construct the frozen dataclass with all named booleans and SHA256 of canonical JSON excluding `comparison_sha256`. The CLI accepts only `--control`, `--candidate`, and `--out`; it validates two `G1ArmEvidence.from_record` payloads, calls `compare_g1_arms`, calls `assert_no_dense_fp_artifact` on `comparison.to_record()`, writes canonical JSON, and returns `0` only for `HIGH_TAIL_CAUSAL_PROOF`. It never launches an evaluator, writes a checkpoint, or changes a schedule.

- [ ] **Step 4: Run the green and comparison regression commands.**

Run: `uv run pytest tests/test_icdc_g1_evidence.py::test_g1_comparison_requires_all_matched_gates_and_keeps_point300_diagnostic -q`

Expected: PASS.

Run: `uv run pytest tests/test_icdc_g1_evidence.py -q`

Expected: PASS; failures cover a `+1e-12` comparison boundary, unmatched IDs/medians, a bad receipt mix, non-100 feasibility, nonzero errors, missing freeze evidence, and a failed causal smoke receipt.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add src/icdc_engine/g1_evidence.py scripts/probes/icdc_topology_g1_evidence.py tests/test_icdc_g1_evidence.py && git commit -m "feat: compare sealed topology G1 arms"`

## P0 acceptance evidence and kill behavior

- Preserve the source-role test output proving a valid tree-only raw mutation changes receipt identity but not sanitized case/fp semantics. Any tree reference in P0 model, label, extractor, candidate, loss, scorer, or checkpoint path is `KILLED_INPUT_CHECKPOINT_OR_SCORER`.
- Preserve QA preflight fields in every later manifest. Missing/mismatched QA PDF, scorer source, or literal scorer contract kills before sample/admission/scoring.
- Preserve the no-dense scanner result for every evidence payload. A dense fp coordinate, rectangle, origin, size, gap, or overlap-magnitude key kills the artifact write before it is published.
- Preserve G1 arm and comparison JSON plus their canonical hashes. Non-100 rows, incomplete receipts, non-3D/3F pool behavior, error/feasibility/freeze/causal-smoke failure, or `candidate_combined > control_combined + 1e-12` yields a kill state; `.300` only changes the diagnostic field.

## P0 final verification and handoff

- [ ] Run `uv run pytest tests/test_topology_p0_contracts.py tests/test_icdc_g1_evidence.py -q`; expect PASS.
- [ ] Run `uv run pytest`; expect PASS. This is the full test suite, not a package review or full100 evaluator.
- [ ] Run `git diff --check`; expect no output.
- [ ] Run `git status --short`; confirm only intended P0 source/tests and expected untracked/dirty graph artifacts are present before integration.
- [ ] Sol checks each P1 handoff signature above against imports and canonical field names. Do not execute P1, G0, package review, blind full100, or any final evaluator workflow without separately authorized policy.

## P0 self-review

- [ ] Source contract coverage: seven tensor indices/types/order, schema-only `tree_sol`, exact fp conversion, and training/validation trust boundaries map to Task 1.
- [ ] QA coverage: path/SHA preflight, A4 soft V, A5 preplaced hard/boundary soft, A6 inclusive endpoints, and A15 scorer formula map to Task 3.
- [ ] Dense-leak coverage: recursive pre-write scan and SHA-pinned canonical bytes map to Task 2.
- [ ] G1 coverage: row type, canonical validation, Alpha binding, sorted `[89]` p90, `exp(n/12)`, combined comparison, `.300` diagnostic, IDs/medians/3D3F/feasibility/freeze/causal-smoke gates map to Tasks 4-5.
- [ ] Placeholder scan: run `rg -n -i 'to'"'"'do|tb'"'"'d|implement[[:space:]]+later|fill[[:space:]]+in[[:space:]]+details|similar[[:space:]]+to' docs/superpowers/plans/2026-08-13-topology-p0-qa-g1-evidence.md`; expect no matches.
- [ ] Type scan: check the P1 handoff block against Tasks 1-3 and the P1 plan before committing documentation.

Plan complete and saved to `docs/superpowers/plans/2026-08-13-topology-p0-qa-g1-evidence.md`. Its execution must precede P1; use a fresh-worker Subagent-Driven loop with the stated independent reviews, or execute inline only with the required checkpointed review gates.
