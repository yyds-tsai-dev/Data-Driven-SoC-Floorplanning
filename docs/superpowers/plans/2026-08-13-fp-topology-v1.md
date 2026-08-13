# FP Topology V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert receipt-verified training `fp_sol` into canonical sparse `fp_topology_v1`, realize it through an offline exact-teacher boundary, and emit only winner-linked sparse labels and pinned evidence.

**Architecture:** P1 consumes P0's transient verified fp row, QA preflight, local scorer adapter, and artifact guard. A focused extractor owns canonical sparse topology; a focused realizer accepts only that topology, a base seed, and sanitized input; the existing teacher probe remains a transaction adapter that calls one offline lifecycle module. P1 is non-blocking for the initial G0 critical path and adds no online production runtime behavior.

**Tech Stack:** Python 3.12, PyTorch CPU float64 tensors, NumPy, pytest, `uv`, SHA256/canonical JSON, existing `partner/icdc/tfdl.py`, `partner/icdc/engine.py`, and the P0 QA scorer contract.

## Global Constraints

- Dependency: P0 must be complete and accepted before P1 imports `VerifiedTrainingFpRow`, `QAContractEvidence`, `LocalScoreAudit`, `assert_no_dense_fp_artifact`, or canonical evidence helpers.
- P1 is offline teacher work only. It does not execute G0, package review, `scripts/eval_total.sh`, a full100 evaluator, a blind arm, a checkpoint mutation, online TFDL, online proposal generation, dense coordinate serialization, coordinate/value loss, or optimizer integration.
- `fp_topology_v1` contains only `schema`, `version`, `receipt`, `instance_id`, `input_fingerprint`, `axis_edges`, `contacts`, and `topology_sha256`. It contains no origin, width, height, gap, overlap magnitude, dense rectangle, `fp_sol`, `fp_xywh`, or coordinate field.
- Logical record field order is `schema`, `version`, `receipt`, `instance_id`, `input_fingerprint`, `axis_edges`, `contacts`, `topology_sha256`; `receipt` order is `relative_path`, `source_sha256`, `row`; edge order is `(axis,src,dst)` and contact order is `(axis,a,b,a_before_b)`.
- Before hashing, omit `topology_sha256`, use UTF-8 `json.dumps(..., sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)`, SHA256 the exact bytes, add the lowercase digest, then canonicalize the complete record with the same serializer.
- Validators require nonempty `instance_id`, lowercase 64-hex digests, `row >= 0`, `0 <= src,dst,a,b < N`, unequal edge endpoints, `a < b`, `axis in {0,1}`, boolean `a_before_b`, canonical sorting, no duplicate relation, no unknown field, and a matching digest.
- Extraction consumes CPU float64 `(x,y,w,h)`. For pair `i,j`, use `gx=max(x_i-(x_j+w_j), x_j-(x_i+w_i))`, `gy=max(y_i-(y_j+h_j), y_j-(y_i+h_i))`; select x when `gx >= gy`; direction is increasing selected-axis center, with lower ID first on an exact center tie; apply transitive reduction independently to the two axis DAGs.
- Contacts consider only same nonzero cluster IDs. Compute `S=sqrt(sum(input_area_i for every valid input block i))` on CPU float64, `contact_eps=1e-9*S`, accept absolute face gap `<= contact_eps` plus strictly positive perpendicular overlap, and select a maximum-overlap Kruskal forest with exact tie key `(axis,min_id,max_id)`. Detection tolerance never relaxes realization legality: realized faces are bit-equal.
- Extraction must be invariant under translation and positive uniform scaling; `S` and geometry scale together. The manifest pins CPU float64, the S formula, `1e-9`, axis/direction ties, per-axis reduction, and Kruskal selection in `extractor_config_sha256`.
- The realizer signature accepts sparse topology, a base seed, and sanitized case input only. It must not accept fp rectangles, `fp_xywh`, `fp_sol`, a validation case, or an evaluator/scorer.
- Slots are fixed and never renumbered: `0=base`, `1=fp-axis`, `2=existing axis`, `3=pin`, `4=fp-contact`, `5=existing contact`. Each slot records exactly one of `candidate`, `rejected`, or `duplicate`; a duplicate points to its earliest `duplicate_of` ordinal and receives no evaluator call. Base must be an admitted candidate exactly once, else `KILLED_LEGALITY_OR_COVERAGE` before scoring.
- The exact offline pipeline is verified training shard -> transient canonical fp -> sparse `fp_topology_v1` -> realize from sparse topology/base seed/sanitized input -> SHA-pinned provided/local evaluator hard audit (soft V accepted and recorded) -> exact TFDL with no shelf fallback -> hard audit -> SHA-pinned provided/local no-runtime scorer -> deterministic winner. Score each candidate exactly once; select `(cost_no_runtime, ordinal, name)`; soft grouping/MIB/boundary failures are scoreable, not rejection; preplaced hard constraints override boundary soft constraints.
- Winner `TopologyLabel` carries immutable `fp_topology_sha256`, proposal identity, ordinal, name, and `cost_no_runtime` linkage metadata; those fields are neither student input nor loss target. P1 writes no dense artifact.

---

## Locked file map

| File | Responsibility |
| --- | --- |
| `partner/icdc/topology_data.py` | Own `TopologyLabel` and its immutable winner-linkage dataclasses; retain the P0 source row contract. |
| `partner/icdc/fp_topology.py` | Extract, validate, canonically serialize, and hash `fp_topology_v1` sparse records. |
| `partner/icdc/topology_realizer.py` | Translate sparse topology plus a base seed and sanitized input into deterministic offline realization attempts, without audit/scoring/fp-rectangle arguments. |
| `partner/icdc/topology_teacher_pipeline.py` | Turn exactly six ordered realization attempts into admitted/rejected/duplicate slots, exact-audit/scorer evidence, and one winner-linked label. |
| `partner/icdc/topology_manifest.py` | Construct and validate the no-dense, SHA-pinned teacher manifest and its extractor/TFDL/scorer evidence fields. |
| `scripts/probes/icdc_topology_teacher.py` | Call the P1 lifecycle/manifest APIs from its existing staged transaction; do not add geometry, selector, or scorer copies. |
| `tests/test_fp_topology_v1.py` | Canonical record, extraction, invariance, contact, and forbidden-field tests. |
| `tests/test_topology_teacher_pipeline.py` | Realizer boundary, fixed slots, exact admission, scorer count, winner, and linkage tests. |
| `tests/test_topology_manifest.py` | Pinned digest, artifact-hygiene, and manifest evidence tests. |

## Exact P0 imports and P1 public interfaces

P1 imports only the P0 handoff types/functions named below. It must not duplicate their conversion, QA, scorer, Alpha, or dense-artifact logic.

```python
@dataclass(frozen=True)
class WinnerTopologyLinkage:
    fp_topology_sha256: str
    proposal_id: str
    ordinal: int
    name: str
    cost_no_runtime: float

@dataclass(frozen=True)
class TopologyLabel:
    instance_id: str
    n: int
    sample_seed: int
    teacher_cost: float
    base_cost: float
    record_weight: float
    edges: tuple[SparseEdge, ...]
    contacts: tuple[ContactLabel, ...]
    pin_paths: tuple[tuple[int, ...], ...]
    linkage: WinnerTopologyLinkage

def extract_fp_topology_v1(
    row: VerifiedTrainingFpRow,
) -> dict[str, object]:
    """Return a validated sparse record; never serialize row.fp_xywh."""

def validate_fp_topology_v1(record: Mapping[str, object], n: int) -> None:
    """Raise ValueError unless the canonical sparse record and digest match."""

def realize_sparse_topology(
    topology: Mapping[str, object], base_seed_xywh: torch.Tensor, case: Mapping[str, object]
) -> torch.Tensor:
    """Return CPU float64 [N,4]; no evaluator/fp rectangle argument exists."""

def evaluate_topology_case(
    topology: Mapping[str, object], base_seed_xywh: torch.Tensor,
    case: Mapping[str, object], sample_seed: int, qa: QAContractEvidence,
) -> CaseTopologyEvidence:
    """Return six fixed slot records and exactly one winner label or a kill state."""
```

## Execution constraints and review gates

For every task, first perform a RED-review with an independent read-only Terra (`gpt-5.6-terra`, `xhigh`) against the specified test, source boundary, and P0 handoff. Then use a fresh Luna (`gpt-5.6-luna`, `low`) with ownership limited to that task's files; it preserves concurrent work and creates the named atomic commit. A different Terra re-reviews the actual diff, command output, no-dense scan, and P0 interface use. Important or Critical findings are corrected by a fresh Luna and re-reviewed before the next task; Sol accepts only inspected evidence.

Never edit `partner/icdc/{tfdl.py,engine.py,energy.py}`, the contest evaluator, production optimizer/submission modules, checkpoints, artifacts, or `scratchpad/`. After every code modification run `graphify update .` and never stage expected graph artifacts. P1 never runs G0.

### Task 1: Add immutable winner linkage to sparse labels

**Files:**

- Modify: `partner/icdc/topology_data.py:42-91`
- Test: `tests/test_topology_teacher_pipeline.py`

**Interfaces:**

- Produces `WinnerTopologyLinkage` and the exact `TopologyLabel` constructor in the public interface block.
- Existing loss code receives `label.edges`, `label.contacts`, `label.pin_paths`, and `label.record_weight` unchanged; it never reads `label.linkage`.

- [ ] **Step 1: Write the failing immutable-linkage test.**

```python
def test_topology_label_requires_immutable_winner_linkage():
    link = WinnerTopologyLinkage("a" * 64, "case#0:7:4:fp-contact", 4,
                                 "fp-contact", 0.91)
    label = TopologyLabel("case#0", 2, 7, 0.91, 1.0, 1.0 / 0.91,
                          (), (), (), link)
    assert label.linkage.fp_topology_sha256 == "a" * 64
    assert label.linkage.proposal_id == "case#0:7:4:fp-contact"
    with pytest.raises(FrozenInstanceError):
        label.linkage.name = "base"
    with pytest.raises(ValueError, match="linkage"):
        WinnerTopologyLinkage("not-a-digest", "x", 0, "base", 1.0)
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_teacher_pipeline.py::test_topology_label_requires_immutable_winner_linkage -q`

Expected: FAIL because `WinnerTopologyLinkage` is absent and `TopologyLabel` has no mandatory linkage field.

- [ ] **Step 3: Add the frozen linkage type and label field.**

```python
@dataclass(frozen=True)
class WinnerTopologyLinkage:
    fp_topology_sha256: str
    proposal_id: str
    ordinal: int
    name: str
    cost_no_runtime: float

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.fp_topology_sha256) is None:
            raise ValueError("linkage digest")
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise ValueError("linkage proposal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("linkage ordinal")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("linkage name")
        if not isinstance(self.cost_no_runtime, float) or not math.isfinite(self.cost_no_runtime):
            raise ValueError("linkage cost")
```

Append `linkage: WinnerTopologyLinkage` to `TopologyLabel`; update every existing in-repository topology-label fixture to construct a frozen linkage with a valid digest. Do not add linkage tensors to `SparseTopologyBatch` or `collate_labels`.

- [ ] **Step 4: Run the green and label-regression commands.**

Run: `uv run pytest tests/test_topology_teacher_pipeline.py::test_topology_label_requires_immutable_winner_linkage -q`

Expected: PASS.

Run: `uv run pytest tests/test_icdc_topology_prior.py tests/test_topology_teacher_pipeline.py -q`

Expected: PASS; loss-batch tensors remain unchanged and label provenance cannot become a training feature.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add partner/icdc/topology_data.py tests/test_topology_teacher_pipeline.py tests/test_icdc_topology_prior.py && git commit -m "feat: link topology labels to winners"`

### Task 2: Canonicalize and validate `fp_topology_v1`

**Files:**

- Create: `partner/icdc/fp_topology.py`
- Test: `tests/test_fp_topology_v1.py`

**Interfaces:**

- Produces `canonical_fp_topology_bytes(record, include_digest: bool) -> bytes`, `topology_sha256(record) -> str`, `validate_fp_topology_v1(record, n) -> None`, and `extract_fp_topology_v1(row) -> dict[str, object]`.
- Consumes the P0 `VerifiedTrainingFpRow` only in memory and calls `assert_no_dense_fp_artifact` before returning a record.

- [ ] **Step 1: Write the failing canonical-schema test.**

```python
def test_fp_topology_v1_has_exact_schema_sorting_and_digest():
    record = {
        "schema": "fp_topology_v1", "version": 1,
        "receipt": {"relative_path": "worker_0/layouts_0.th", "source_sha256": "a" * 64, "row": 0},
        "instance_id": "worker_0/layouts_0.th#0", "input_fingerprint": "b" * 64,
        "axis_edges": [{"src": 0, "dst": 1, "axis": 0}],
        "contacts": [{"a": 0, "b": 1, "axis": 0, "a_before_b": True}],
        "topology_sha256": "",
    }
    record["topology_sha256"] = topology_sha256(record)
    validate_fp_topology_v1(record, 2)
    assert list(record) == ["schema", "version", "receipt", "instance_id", "input_fingerprint", "axis_edges", "contacts", "topology_sha256"]
    assert record["topology_sha256"] == hashlib.sha256(canonical_fp_topology_bytes(record, False)).hexdigest()
    for bad in ({**record, "x": 1}, {**record, "axis_edges": [{"src": 1, "dst": 0, "axis": 0}]},
                {**record, "contacts": [{"a": 1, "b": 0, "axis": 0, "a_before_b": True}]}):
        with pytest.raises(ValueError):
            validate_fp_topology_v1(bad, 2)
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_fp_topology_v1.py::test_fp_topology_v1_has_exact_schema_sorting_and_digest -q`

Expected: FAIL with `ModuleNotFoundError: icdc.fp_topology`.

- [ ] **Step 3: Add exact canonical validation and hashing.**

```python
_FIELDS = ("schema", "version", "receipt", "instance_id", "input_fingerprint",
           "axis_edges", "contacts", "topology_sha256")

def canonical_fp_topology_bytes(record: Mapping[str, object], include_digest: bool) -> bytes:
    payload = dict(record) if include_digest else {key: value for key, value in record.items()
                                                    if key != "topology_sha256"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def topology_sha256(record: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_fp_topology_bytes(record, False)).hexdigest()

def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(field)
    return value
```

Implement `validate_fp_topology_v1` with exact-key equality, logical insertion order, schema/version checks, receipt checks, list-only edges/contacts, endpoint/range/duplicate checks, exact sorted order, and `topology_sha256(record) == record["topology_sha256"]`. It must call P0 `assert_no_dense_fp_artifact(record)` after validating allowed field names, so an otherwise valid sparse relation cannot hide dense data.

- [ ] **Step 4: Run the green and canonicalization regression commands.**

Run: `uv run pytest tests/test_fp_topology_v1.py::test_fp_topology_v1_has_exact_schema_sorting_and_digest -q`

Expected: PASS.

Run: `uv run pytest tests/test_fp_topology_v1.py -q`

Expected: PASS; records reject bad type/order/digest/duplicate/endpoint/axis/boolean fields and all dense fields.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add partner/icdc/fp_topology.py tests/test_fp_topology_v1.py && git commit -m "feat: add canonical fp topology records"`

### Task 3: Extract invariant axis DAGs and contact forests

**Files:**

- Modify: `partner/icdc/fp_topology.py`
- Test: `tests/test_fp_topology_v1.py`

**Interfaces:**

- Completes `extract_fp_topology_v1(row)` and adds `extractor_config_sha256() -> str`.
- Produces only the sparse record from Task 2; its returned value contains no input rectangles.

- [ ] **Step 1: Write the failing extraction/invariance/contact test.**

```python
def test_extractor_is_float64_translation_scale_invariant_and_uses_contact_kruskal():
    row = _verified_row([[0., 0., 2., 2.], [2., .5, 2., 2.], [7., 0., 2., 2.]],
                        area=[4., 4., 4.], clusters=[1, 1, 1])
    base = extract_fp_topology_v1(row)
    moved = extract_fp_topology_v1(_replace_fp(row, row.fp_xywh + torch.tensor([17., -3., 0., 0.])))
    scaled = extract_fp_topology_v1(_replace_fp(row, row.fp_xywh * 5.0, area=[100., 100., 100.]))
    assert base["axis_edges"] == moved["axis_edges"] == scaled["axis_edges"]
    assert base["contacts"] == moved["contacts"] == scaled["contacts"]
    assert base["contacts"] == [{"a": 0, "b": 1, "axis": 0, "a_before_b": True}]
    assert all(edge["axis"] == 0 for edge in base["axis_edges"])
    assert extractor_config_sha256() == extractor_config_sha256()
```

`_verified_row` and `_replace_fp` are concrete test helpers in this file: they create a P0 `VerifiedTrainingFpRow` with CPU float64 fp tensor, a valid receipt, and full five-column sanitized `cons`; `_replace_fp` changes only in-memory fp and matching input area.

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_fp_topology_v1.py::test_extractor_is_float64_translation_scale_invariant_and_uses_contact_kruskal -q`

Expected: FAIL because extraction is not implemented or returns no invariant sparse record.

- [ ] **Step 3: Add the exact CPU float64 extractor.**

```python
def _pair_axis_and_direction(rects: torch.Tensor, first: int, second: int) -> tuple[int, int, int]:
    xi, yi, wi, hi = (float(v) for v in rects[first])
    xj, yj, wj, hj = (float(v) for v in rects[second])
    gx = max(xi - (xj + wj), xj - (xi + wi))
    gy = max(yi - (yj + hj), yj - (yi + hi))
    axis = 0 if gx >= gy else 1
    ci = (xi + wi / 2.0, yi + hi / 2.0)[axis]
    cj = (xj + wj / 2.0, yj + hj / 2.0)[axis]
    src, dst = (first, second) if (ci, first) <= (cj, second) else (second, first)
    return axis, src, dst

def _contact_eps(case: Mapping[str, object]) -> float:
    areas = torch.as_tensor(case["area"], dtype=torch.float64, device="cpu")
    return 1e-9 * math.sqrt(float(areas[areas > 0].sum()))
```

For every unordered pair, call `_pair_axis_and_direction`; build two acyclic edge sets and remove edge `(u,v)` iff `v` remains reachable from `u` without that edge. For contacts, evaluate both axes only within the same positive cluster, require `abs(face_gap) <= _contact_eps(case)` and perpendicular overlap `> 0`, normalize endpoints to `a < b` while correcting `a_before_b`, sort candidates by `(-overlap, axis, a, b)`, and use union-find per cluster to retain a maximum-overlap forest. Sort final relations, assemble the Task 2 field order, hash, validate, and return. `extractor_config_sha256` hashes exact canonical JSON with CPU `float64`, S formula, coefficient, tie rules, reduction, and Kruskal rule.

- [ ] **Step 4: Run the green and extractor-regression commands.**

Run: `uv run pytest tests/test_fp_topology_v1.py::test_extractor_is_float64_translation_scale_invariant_and_uses_contact_kruskal -q`

Expected: PASS.

Run: `uv run pytest tests/test_fp_topology_v1.py -q`

Expected: PASS; tests cover gx/gy tie-to-x, center tie-to-lower-ID, transitive reduction, strict perpendicular overlap, face tolerance only at detection, duplicate elimination, and translation/positive-scaling invariance.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add partner/icdc/fp_topology.py tests/test_fp_topology_v1.py && git commit -m "feat: extract invariant fp topology"`

### Task 4: Build a sparse-only topology realizer boundary

**Files:**

- Create: `partner/icdc/topology_realizer.py`
- Test: `tests/test_topology_teacher_pipeline.py`

**Interfaces:**

- Produces frozen `RealizationAttempt(ordinal, name, intended_kind, rects_xywh, rejection_reason)` and `realize_sparse_topology(topology, base_seed_xywh, case) -> torch.Tensor`.
- The public realizer signature has exactly the three arguments in the interface block. It accepts no scorer, evaluator, fp rectangle, fp source, or validation input.

- [ ] **Step 1: Write the failing sparse-boundary test.**

```python
def test_realizer_accepts_sparse_topology_not_fp_rectangles_and_preserves_fixed_geometry():
    topology = _topology_record(axis_edges=[{"src": 0, "dst": 1, "axis": 0}], contacts=[])
    seed, case = _realizer_case()
    result = realize_sparse_topology(topology, seed, case)
    assert result.dtype is torch.float64 and result.device.type == "cpu"
    assert result.shape == seed.shape
    assert torch.equal(result[0, 2:], torch.tensor([2., 3.], dtype=torch.float64))
    assert result[1, 0] >= result[0, 0] + result[0, 2]
    parameters = tuple(inspect.signature(realize_sparse_topology).parameters)
    assert parameters == ("topology", "base_seed_xywh", "case")
```

`_realizer_case` returns CPU float64 `[N,4]`, full five-column `cons`, `area`, and `tp`; its fixed/preplaced geometry is defined from input, never fp output.

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_teacher_pipeline.py::test_realizer_accepts_sparse_topology_not_fp_rectangles_and_preserves_fixed_geometry -q`

Expected: FAIL with `ModuleNotFoundError: icdc.topology_realizer`.

- [ ] **Step 3: Add the sparse realizer.**

```python
@dataclass(frozen=True)
class RealizationAttempt:
    ordinal: int
    name: str
    intended_kind: str
    rects_xywh: torch.Tensor | None
    rejection_reason: str | None

def realize_sparse_topology(topology: Mapping[str, object], base_seed_xywh: torch.Tensor,
                            case: Mapping[str, object]) -> torch.Tensor:
    n = int(case["n"])
    validate_fp_topology_v1(topology, n)
    if base_seed_xywh.device.type != "cpu" or base_seed_xywh.shape != (n, 4):
        raise ValueError("base seed")
    out = base_seed_xywh.to(dtype=torch.float64).clone()
    for index, flags in enumerate(case["cons"]):
        if flags[0] or flags[1]:
            out[index, 2:] = torch.as_tensor(case["tp"][index][2:], dtype=torch.float64)
        if flags[1]:
            out[index, :2] = torch.as_tensor(case["tp"][index][:2], dtype=torch.float64)
    for edge in topology["axis_edges"]:
        src, dst, axis = edge["src"], edge["dst"], edge["axis"]
        out[dst, axis] = max(float(out[dst, axis]), float(out[src, axis] + out[src, axis + 2]))
    return out
```

Add private deterministic helpers for pin-support and contact intents; they operate solely from `topology["axis_edges"]`, `topology["contacts"]`, base seed, and case. They reject a cycle, a fixed/preplaced conflict, or a contact unable to preserve strictly positive perpendicular overlap; they do not call TFDL, engine hard audit, energy, or a scorer.

- [ ] **Step 4: Run the green and boundary-regression commands.**

Run: `uv run pytest tests/test_topology_teacher_pipeline.py::test_realizer_accepts_sparse_topology_not_fp_rectangles_and_preserves_fixed_geometry -q`

Expected: PASS.

Run: `uv run pytest tests/test_topology_teacher_pipeline.py -k 'realizer or sparse_boundary' -q`

Expected: PASS; AST/signature checks reject fp rectangle arguments and tests cover fixed/preplaced geometry, cycles, and nonpositive contact overlap.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add partner/icdc/topology_realizer.py tests/test_topology_teacher_pipeline.py && git commit -m "feat: add sparse topology realizer"`

### Task 5: Evaluate six fixed slots through exact admission and local scoring

**Files:**

- Create: `partner/icdc/topology_teacher_pipeline.py`
- Test: `tests/test_topology_teacher_pipeline.py`

**Interfaces:**

- Produces frozen `TopologySlotEvidence`, `CaseTopologyEvidence`, `evaluate_topology_case`, and `topology_label_from_winner`.
- Consumes P0 QA scoring and P1 realizer/extractor record. It calls the scorer exactly once for each `candidate` status and never for `rejected`/`duplicate`.

- [ ] **Step 1: Write the failing fixed-slot/winner test.**

```python
def test_fixed_slots_exact_pipeline_scores_candidates_once_and_keeps_base():
    topology, seed, case, qa = _pipeline_fixture()
    evidence = evaluate_topology_case(topology, seed, case, sample_seed=9, qa=qa)
    assert [(slot.ordinal, slot.name) for slot in evidence.slots] == [
        (0, "base"), (1, "fp-axis"), (2, "existing axis"),
        (3, "pin"), (4, "fp-contact"), (5, "existing contact"),
    ]
    assert {slot.status for slot in evidence.slots} <= {"candidate", "rejected", "duplicate"}
    assert evidence.slots[0].status == "candidate"
    assert evidence.label.linkage.ordinal == evidence.winner_ordinal
    assert evidence.label.linkage.cost_no_runtime == evidence.winner_cost_no_runtime
    assert evidence.scorer_calls == sum(slot.status == "candidate" for slot in evidence.slots)
    for slot in evidence.slots:
        if slot.status == "duplicate":
            assert slot.duplicate_of is not None and slot.scored is False
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_teacher_pipeline.py::test_fixed_slots_exact_pipeline_scores_candidates_once_and_keeps_base -q`

Expected: FAIL with `ModuleNotFoundError: icdc.topology_teacher_pipeline`.

- [ ] **Step 3: Add the fixed lifecycle and exact boundary.**

```python
_SLOTS = ((0, "base"), (1, "fp-axis"), (2, "existing axis"),
          (3, "pin"), (4, "fp-contact"), (5, "existing contact"))

@dataclass(frozen=True)
class TopologySlotEvidence:
    ordinal: int
    name: str
    status: str
    duplicate_of: int | None
    reason: str | None
    scored: bool
    cost_no_runtime: float | None
    hard: Mapping[str, bool] | None
    soft: Mapping[str, int] | None

def _exact_admit(rects: torch.Tensor, case: Mapping[str, object]) -> tuple[torch.Tensor, Mapping[str, bool]]:
    legal, drift = tfdl.tfdl(_batch(rects), _mask(case), _pinned(case), pin_xy=_pin_xy(case),
                             boundary_code=_boundary(case), exact=True)
    if torch.count_nonzero(drift).item() != 0:
        raise ValueError("pin drift")
    hard = engine.verify_hard_legal(legal[0].numpy(), _area(case), _cons(case), _tp(case))
    if not hard or not all(hard.values()):
        raise ValueError("hard audit")
    return legal[0], hard
```

For each `_SLOTS` entry generate the intended sparse realization, first run nonexact TFDL from the original seed with literal zero drift, then `_exact_admit` from that same original seed. Canonically fingerprint the realized sparse topology; first fingerprint is candidate and later equal fingerprints become `duplicate` with earliest ordinal. Any failure becomes `rejected` with an explicit reason. If slot 0 is not candidate, return `state="KILLED_LEGALITY_OR_COVERAGE"` before a scorer call. For every candidate, call P0 `score_provided_local_no_runtime` once, retain soft V counts even when feasible, and select the winner by `(cost_no_runtime, ordinal, name)`. Build `WinnerTopologyLinkage(topology["topology_sha256"], f"{case['instance_id']}:{sample_seed}:{ordinal}:{name}", ordinal, name, cost)` and derive the sparse label only after winning. Parse `partner/icdc/tfdl.py` with `ast` before processing and reject if the chosen public path can call `shelf_fallback`.

- [ ] **Step 4: Run the green and lifecycle-regression commands.**

Run: `uv run pytest tests/test_topology_teacher_pipeline.py::test_fixed_slots_exact_pipeline_scores_candidates_once_and_keeps_base -q`

Expected: PASS.

Run: `uv run pytest tests/test_topology_teacher_pipeline.py -q`

Expected: PASS; fixtures cover base rejection kill before scoring, duplicate no-scorer, no-improvement base winner, tie by ordinal/name, exact TFDL/no shelf fallback, hard audit, soft V acceptance, and preplaced-over-boundary priority.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add partner/icdc/topology_teacher_pipeline.py tests/test_topology_teacher_pipeline.py && git commit -m "feat: evaluate fixed topology slots"`

### Task 6: Bind P1 manifests and wire the existing teacher adapter

**Files:**

- Create: `partner/icdc/topology_manifest.py`
- Modify: `scripts/probes/icdc_topology_teacher.py:63-65,794-1014,2015-2058,2329-2331`
- Test: `tests/test_topology_manifest.py`

**Interfaces:**

- Produces `build_topology_manifest(...) -> dict[str, object]` and `validate_topology_manifest(manifest) -> None`.
- The teacher probe passes P0 `QAContractEvidence`, P1 extractor config/source hash, exact TFDL source/config hash, scorer fields, ordered receipt digest, and already-safe lifecycle records; it retains staged fsync/atomic publication ownership.

- [ ] **Step 1: Write the failing concrete-manifest test.**

```python
def test_topology_manifest_binds_concrete_sources_and_rejects_dense_payload(tmp_path):
    manifest = build_topology_manifest(
        receipts=[{"relative_path": "worker_0/layouts_0.th", "source_sha256": "a" * 64, "row": 0}],
        qa=_qa_evidence(), extractor_path=Path("partner/icdc/fp_topology.py"),
        tfdl_path=Path("partner/icdc/tfdl.py"), scorer_path=Path("scripts/iccad2026_evaluate.py"),
    )
    validate_topology_manifest(manifest)
    assert set(("source_receipt_set_sha256", "extractor_source_sha256", "extractor_config_sha256",
                "exact_tfdl_source_sha256", "exact_tfdl_config_sha256", "scorer_source_sha256",
                "scorer_contract_sha256", "qa_sha256")) <= set(manifest)
    with pytest.raises(ValueError, match="dense fp artifact"):
        validate_topology_manifest({**manifest, "rects": [[0., 0., 1., 1.]]})
    with pytest.raises(ValueError, match="extractor source"):
        validate_topology_manifest({**manifest, "extractor_source_sha256": "0" * 64})
```

- [ ] **Step 2: Run the node to verify RED.**

Run: `uv run pytest tests/test_topology_manifest.py::test_topology_manifest_binds_concrete_sources_and_rejects_dense_payload -q`

Expected: FAIL with `ModuleNotFoundError: icdc.topology_manifest`.

- [ ] **Step 3: Add the pinned manifest builder and thin adapter wiring.**

```python
def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError("source file")
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _receipt_set_sha256(receipts: Sequence[Mapping[str, object]]) -> str:
    rows = sorted(receipts, key=lambda row: (str(row["relative_path"]), int(row["row"])))
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()

def build_topology_manifest(receipts: Sequence[Mapping[str, object]], qa: QAContractEvidence,
                            extractor_path: Path, tfdl_path: Path, scorer_path: Path) -> dict[str, object]:
    manifest = {
        "schema": "icdc_fp_topology_evidence_v1", "source_receipt_set_sha256": _receipt_set_sha256(receipts),
        "extractor_source_sha256": _sha256_file(extractor_path),
        "extractor_config_sha256": extractor_config_sha256(),
        "exact_tfdl_source_sha256": _sha256_file(tfdl_path),
        "exact_tfdl_config_sha256": exact_tfdl_config_sha256(),
        "scorer_source_sha256": _sha256_file(scorer_path),
        "scorer_contract_sha256": hashlib.sha256(qa.scorer_contract.encode("utf-8")).hexdigest(),
        **qa_manifest_fields(qa),
    }
    assert_no_dense_fp_artifact(manifest)
    return manifest
```

Define `exact_tfdl_config_sha256` as canonical JSON for CPU float64, bit-equal faces, and `shelf_fallback=false`. `validate_topology_manifest` requires lowercase 64-hex values, recomputes all source/config/receipt digests from supplied immutable inputs, checks the literal scorer contract, and calls P0's dense guard. In the teacher probe, replace only imports/calls to its private proposal fingerprint, admission/scorer lifecycle, topology SHA, and manifest extras with the P1 APIs; retain all existing staging, spool, fsync, and atomic rename code. No fp coordinate is added to JSONL or the manifest.

- [ ] **Step 4: Run the green and adapter-regression commands.**

Run: `uv run pytest tests/test_topology_manifest.py::test_topology_manifest_binds_concrete_sources_and_rejects_dense_payload -q`

Expected: PASS.

Run: `uv run pytest tests/test_fp_topology_v1.py tests/test_topology_teacher_pipeline.py tests/test_topology_manifest.py tests/test_icdc_topology_prior.py -q`

Expected: PASS; manifest output contains concrete P0/P1 SHA fields and no dense fp payload, while the teacher's staged transaction tests retain atomic output behavior.

- [ ] **Step 5: Update graph and commit the self-contained slice.**

Run: `graphify update .`

Run: `git add partner/icdc/topology_manifest.py scripts/probes/icdc_topology_teacher.py tests/test_topology_manifest.py && git commit -m "feat: bind fp topology evidence"`

## P1 acceptance evidence and kill behavior

- Save canonical `fp_topology_v1` records, their receipt/input hashes, extractor config/source hashes, and invariance test output. Invalid schema/order/digest, coordinate/dense field, duplicate relation, failed reduction, or contact tolerance used as realized legality is a pre-artifact kill.
- Save six slot rows per case with ordinal/name/status/reason/duplicate target/scorer flag/hard/soft data. Base missing/rejected/duplicate is `KILLED_LEGALITY_OR_COVERAGE` before any scorer invocation; duplicates never receive a second score.
- Save exact-TFDL and hard-audit evidence. Drift, hard violation, shelf fallback reachability, finite-cost failure, or missing P0 scorer/QA evidence rejects that slot; soft grouping/MIB/boundary violations remain scoreable evidence when feasibility holds.
- Save a winner label only when it has a valid sparse topology SHA and immutable proposal linkage. `record_weight=base_cost/teacher_cost` is finite positive; no-improvement retains base with `teacher_cost == base_cost` and weight `1`.
- P1 is non-blocking for initial G0. It must not claim a G0/G1 pass, execute G0, or schedule a full100 run.

## P1 final verification and handoff

- [ ] Run `uv run pytest tests/test_fp_topology_v1.py tests/test_topology_teacher_pipeline.py tests/test_topology_manifest.py -q`; expect PASS.
- [ ] Run `uv run pytest`; expect PASS. This is a full test suite only; it is not package review, G0, or full100 evaluation.
- [ ] Run `git diff --check`; expect no output.
- [ ] Run `git status --short`; confirm only intended P1 sources/tests plus expected untracked/dirty graph artifacts are present before integration.
- [ ] Confirm no P1 command executed G0, package review, blind evaluation, `scripts/eval_total.sh`, or online production behavior.

## P1 self-review

- [ ] Canonical schema/types/order/hash, validators, no-dense serializer, CPU float64 gx/gy/direction/reduction/contact S+epsilon/Kruskal/sorts/invariance map to Tasks 2-3.
- [ ] Sparse-only realizer signature and fixed 0..5 slot/status/duplicate/base rules map to Tasks 4-5.
- [ ] Exact TFDL/hard audit/provided-local scorer pipeline, soft/hard semantics, deterministic winner, and immutable `TopologyLabel` linkage map to Tasks 1 and 5.
- [ ] Concrete receipt/extractor/TFDL/scorer/QA manifest hashes map to Task 6.
- [ ] P0 dependency and initial-G0 non-blocking restriction appear in Global Constraints, interfaces, execution constraints, and acceptance evidence.
- [ ] Placeholder scan: run `rg -n -i 'to'"'"'do|tb'"'"'d|implement[[:space:]]+later|fill[[:space:]]+in[[:space:]]+details|similar[[:space:]]+to' docs/superpowers/plans/2026-08-13-fp-topology-v1.md`; expect no matches.
- [ ] Type scan: compare every public interface in this plan with P0's handoff block and the Task 1-6 definitions before committing documentation.

Plan complete and saved to `docs/superpowers/plans/2026-08-13-fp-topology-v1.md`. Execute it only after P0 and only when the initial G0 critical path has left room for the non-blocking sparse-topology work.
