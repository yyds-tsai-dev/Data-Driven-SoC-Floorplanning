# Track B: Data-Free Sparse Topology Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate a data-free sparse topology teacher and same-shape Direct student, then perform one sealed high-tail G1 causal test under the user-approved six-candidate 3-Direct/3-Flow portfolio. Stop before G2 or package work.

**Architecture:** Freeze `partner/checkpoints/direct_v2_cont/eval_step1p2M.pt`; deterministically propose axis/order exchanges, pin repairs, and grouping contacts; admit only float64 exact-TFDL, exact-hard-legal, zero-drift proposals. Distill transitive-reduced sparse topology labels through three samples of the production Direct DPM++ two-step differentiable sampler, with detached teacher weights and base/EMA anchors. Test the selected same-shape Direct checkpoint only inside a frozen total-six portfolio of exactly 3 Direct + 3 Flow.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, `uv`, JSONL/SHA256, existing `partner/icdc/{tfdl,energy,engine,sampler}.py`.

## Global Constraints

- Development baseline: `artifacts/partner_eval/gbridge_package_full100.json`; no-runtime `1.1437448258795715`, average runtime `0.29881621031556277` s/case, `100/100` feasible, zero errors. Because its normal pool was 0 Direct/6 Flow, it supplies the absolute G1 bar but is not a causal control for a changed Direct checkpoint.
- Frozen causal portfolio `P_B_3D3F`: total `K=6`, exactly 3 Direct DPM++/2 plus 3 Flow Euler/8, with the Flow checkpoint and all non-quota solver policy frozen. No-pool, Flow failure, oversampling, non-six K, or any non-3D/3F mix invalidates the run.
- G1 uses a newly measured concealed matched pair under `P_B_3D3F`: C0 is the production Direct EMA and C1 is the held-out-selected Track-B EMA. Except for `DIRECT_CKPT`, the arms are byte-for-byte/config-for-config identical.
- Decomposition: `n>=100` weight `0.8264247038229129`, current-band average `1.1011247299384228`, outside contribution `0.2337481470681254`; perfecting only `n>=100` reaches `1.0601728508910383`. The approved portfolio leaves `PARTNER_DIRECT_MIN=.3`, so Direct remains closed for `n<=98`; G2 requires a separate approved production-policy spec.
- Track A leaves approximately `0.434` ms from its `0.75` ms allowance; the matched candidate arm must bind runtime to the actual full100 result.
- Training IDs/fingerprints and deterministic held-out IDs are allowed; validation IDs, validation loaders, and validation outcomes are forbidden for proposal tuning, training, thresholds, or checkpoint selection.
- Never edit existing TFDL, energy, engine, sampler, optimizer, submission, or package code. No online TFDL, proposal bank, dense head, extra graph pass, or changed Direct/Flow sampler. The only approved production-policy difference from the historical wrapper is the exact 3D/3F quota at unchanged total six.
- Every admitted, named-intent-surviving proposal (including base) is scored by official `evaluate_solution(runtime=1, median_runtime=1)`; `EN.energy` is diagnostic only after admission and never a shortlist/filter. Select deterministic `(cost,candidate_ordinal,name)`, always include base, and retain an explicit no-improvement base winner with `teacher_cost==base_cost` and `record_weight=1`. Corpus stores hpwl/area references, never golden coordinate targets.
- G0 uses eligible held-out `H` from the immutable receipt-bound index (`n>=100`, `split_for_id mod 10`), one fixed Direct seed/case, and `w=exp(n/12)`. Bind ordered IDs/ns/weights/denominator/population SHA and `B_H`, `T_H`, `Delta_H=B_H-T_H`. Hard `Delta_H>=0.0181504738793652`; target `Delta_H>=0.0261247299384228`; these are predictive gates, not absolute full100 comparisons. `T_H>1.5` is diagnostic kill only. Between hard/target STOP with no training authority; only target advances Task 5. G1 remains sole causal transfer proof.
- G1 requires retained teacher gain `>=75%`; non-validation causal smoke; then candidate no-runtime both `<=1.1287448258795715` and `<=C0-0.015`, runtime `<=.300`, `100/100`, zero errors, and valid 3D/3F receipts.
- Kill on oracle `>1.5`, failed coverage/pin support, lost held-out benefit, receipt mismatch, runtime failure, drift, shelf fallback, hard illegality, missing blind arm, or either G1 score gate. On G1 pass write `HIGH_TAIL_CAUSAL_PROOF` followed by `STOP_REQUIRES_SEPARATE_APPROVAL`; do not start G2 and do not package.

---

## File map

Create only `partner/icdc/topology_data.py`, `partner/icdc/topology_prior.py`, `partner/icdc/train_topology_prior.py`, `tests/test_icdc_topology_prior.py`, `scripts/probes/icdc_topology_teacher.py`, `scripts/probes/icdc_topology_gate.py`, `scripts/probes/icdc_topology_3d3f_wrapper.py`, and `scripts/probes/run_icdc_topology_stage.sh`; generated evidence is under `artifacts/icdc_topology/` or `.superpowers/sdd/`, and generated gate artifacts are not committed. The Track-B wrapper may set the frozen environment and add fail-closed source-count/Flow-success receipts around inherited production methods; it may not change sampling, ranking, refinement, or selection.

### Task 1: Data schema, sanitized corpus, split, manifest

The implemented corpus API is receipt-bound.  Callers must pass the immutable
canonical `source_root=FloorSet/floorset_lite` and one
`CorpusSourceReceipt(relative_path, file_sha256, layout_index, fingerprint)`
per case; a bare `save_sanitized_corpus(path, cases)` call is invalid and must
not appear in fixtures or documentation.  Verification hashes each exact
source file, loads its bytes once from a `BytesIO` buffer using
`weights_only=True`, validates the exact seven-tensor shard schema (input
`[B,N,6]`, tree `[B,N-1,3]`, fingerprint `[B,N,4]`, metrics `[B,8]`, plus
the remaining exact validated tensor widths), and reconstructs canonical
geometry by mapping raw `(w,h,x,y)` to `(x,y,w,h)`.
The receipt's relative path, SHA256, layout index, and sanitized fingerprint
are immutable provenance.  `source_root` and receipts are required save-time
inputs and define the source verification boundary; later sealed index/manifest
artifacts carry the experiment-level reproducibility boundary.

**Files:** Create `partner/icdc/topology_data.py`; create/modify `tests/test_icdc_topology_prior.py`.

**Interfaces:** Frozen dataclasses exactly as specified:
`SparseEdge(src:int,dst:int,axis:int,margin:float,kind:str,weight:float)`,
`ContactLabel(a:int,b:int,axis:int,a_before_b:bool,perp_margin:float,weight:float)`,
`TopologyLabel(instance_id:str,n:int,sample_seed:int,teacher_cost:float,base_cost:float,record_weight:float,edges:Tuple[SparseEdge,...],contacts:Tuple[ContactLabel,...],pin_paths:Tuple[Tuple[int,...],...])`,
and `SparseTopologyBatch` with `torch.Tensor` fields
`edge_batch,edge_src,edge_dst,edge_axis,edge_margin,edge_weight,contact_batch,contact_a,contact_b,contact_axis,contact_order,contact_margin,contact_weight`.
Export `fingerprint_case(case: Mapping[str, Any])->str`,
`split_for_id(instance_id: str, heldout_mod: int = 10)->str`,
`collate_labels(labels: Sequence[TopologyLabel], device: torch.device,
dtype: torch.dtype)->SparseTopologyBatch`, canonical JSONL read/write,
sanitized corpus save/load, and SHA256 manifest helpers.

- [ ] Write this concrete RED test (the repository test helper pattern):
```python
def test_sanitize_excludes_golden_and_masks_non_input_geometry(tmp_path):
    case = _case(golden=[[99.0, 99.0, 99.0, 99.0]])
    _save(tmp_path / "c.jsonl", [case])
    row = load_sanitized_corpus(tmp_path / "c.jsonl")[0]
    assert set(row) == {
        "instance_id", "n", "area", "cons", "tp", "b2b", "p2b",
        "pins", "hpwl_ref", "area_ref",
    }
    assert row["tp"] == [
        [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 3.0, 4.0],
        [5.0, 6.0, 5.0, 6.0],
    ]
    assert "golden" not in row and "test_id" not in row
```

Here `_save` constructs a matching exact seven-tensor temporary source shard,
receipt-derived `instance_id` and fingerprint, and binds the test-only
canonical root; the fixture therefore exercises the real receipt-bound API
without undefined `source_root`/`receipt` variables.
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py::test_sanitize_excludes_golden_and_masks_non_input_geometry -q`; expect RED.
- [ ] Implement `@dataclass(frozen=True)`, canonical sorted-key JSON, and a
  SHA256 manifest. The exact corpus key allowlist is the set asserted above.
  For a soft row, serialize `tp=[-1,-1,-1,-1]`; for fixed-only, serialize only
  `w,h`; for preplaced, serialize input-authorized `x,y,w,h`. Reject a source
  manifest outside `FloorSet/floorset_lite`, any `test_id`, validation
  loader provenance, duplicate fingerprint, non-finite value, or shape
  mismatch. Ignore, never serialize, and never use a source `golden` field.
  Training instance IDs/fingerprints are retained for deterministic splitting
  and are not coordinate targets.
- [ ] Run the targeted test plus `uv run pytest tests/test_icdc_topology_prior.py -q`; expect PASS; run `graphify update .` without staging unrelated `graphify-out/` dirt.
- [ ] Commit `git add partner/icdc/topology_data.py tests/test_icdc_topology_prior.py && git commit -m "feat: add topology schemas and corpus manifests"`.

### Task 2: Sparse labels and topology losses

**Files:** Create `partner/icdc/topology_prior.py`; modify `tests/test_icdc_topology_prior.py`.

**Interfaces:** Import `Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple`,
`torch`, and Task 1 types. Define
`topology_losses(rects: torch.Tensor, labels: SparseTopologyBatch,
scale: torch.Tensor)->Dict[str,torch.Tensor]` and
`extract_sparse_label(legal: torch.Tensor, case: Mapping[str, Any],
instance_id: str, sample_seed: int, teacher_cost: float,
base_cost: float)->TopologyLabel`.

- [ ] Write concrete RED gradient test:
```python
def test_topology_losses_are_zero_when_constraints_hold_and_grad_when_broken():
    r = torch.tensor([[[0.,0.,2.,2.],[3.,0.,2.,2.]]], requires_grad=True)
    b = collate_labels([TopologyLabel("x",2,1,1.,1.,1.,(SparseEdge(0,1,0,1.,"sep",1.),),(),())], r.device, r.dtype)
    good = topology_losses(r, b, torch.tensor([1.]))
    assert good["separation"].item() == 0
    bad = r.detach().clone()
    bad[0, 1, 0] = 1.
    bad.requires_grad_()
    out = topology_losses(bad, b, torch.tensor([1.]))
    out["total"].backward()
    assert torch.isfinite(bad.grad).all() and bad.grad.abs().sum() > 0


def test_contact_loss_normalizes_both_axis_gap_and_overlap_deficit():
    rects = torch.tensor([[[0., 0., 2., 2.], [2.5, .5, 2., 2.]]],
                         requires_grad=True)
    label = TopologyLabel(
        "contact", 2, 1, 1., 1., 1., (),
        (ContactLabel(0, 1, 0, True, 2., 1.),), (),
    )
    batch = collate_labels([label], rects.device, rects.dtype)
    out = topology_losses(rects, batch, torch.tensor([10.]))
    torch.testing.assert_close(out["contact"], torch.tensor(.1))
    out["contact"].backward()
    assert torch.isfinite(rects.grad).all()
    assert rects.grad[0, 1, 0] > 0 and rects.grad[0, 1, 1] > 0


def test_contact_loss_respects_order_and_is_zero_at_exact_positive_contact():
    rects = torch.tensor([[[2., 0., 2., 2.], [0., 0., 2., 2.]]])
    label = TopologyLabel(
        "reverse", 2, 2, 1., 1., 1., (),
        (ContactLabel(0, 1, 0, False, 2., 1.),), (),
    )
    batch = collate_labels([label], rects.device, rects.dtype)
    torch.testing.assert_close(
        topology_losses(rects, batch, torch.tensor([10.]))["contact"],
        torch.tensor(0.),
    )
```
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py -k topology_losses -q`; expect RED.
- [ ] Implement sparse gathers. For axis `0`, signed separation is
  `x_dst-(x_src+w_src)`; for axis `1`, it is `y_dst-(y_src+h_src)`. Apply
  `relu((margin-signed_separation)/scale[edge_batch])`. Separation and contact
  are global detached-effective-weight normalized means; contact itself is the
  sum (not average) of normalized `abs(face_gap)` and overlap-deficit terms,
  with `total = separation + contact`. Empty classes return differentiable
  zero. The finite, nonnegative `[B]` scale is clamped with `clamp_min(1e-6)`.
  Contact order defines signed abutment gap. Exact teacher contacts require
  bit-equal face abutment and strictly positive perpendicular overlap; extract
  the grouping forest from observed teacher contacts using deterministic
  maximum-overlap Kruskal. Margin is the nonnegative physical face gap.
  Transitive-reduce separation edges before expanding pin edges. For each axis,
  `pin_paths` is audit-only incoming critical support chains terminating at
  each preplaced block (never p2b); recursively choose the incoming
  predecessor by greatest physical end, tie-breaking by lower block ID.
  Expand every adjacent path edge into `edges` with `kind="pin"`, retaining
  duplicate sep+pin edges so their weights intentionally add; the fixed batch
  schema has no separate pin loss. In `collate_labels`, multiply each
  edge/contact weight by detached `record_weight`, with frozen
  `record_weight = base_cost / teacher_cost` (finite positive; reject
  `teacher_cost > base_cost`). Proposal identity remains the teacher JSONL
  envelope/manifest keyed by `(instance_id, sample_seed)`, not TopologyLabel
  fields. Detach weights only, never `rects`; no energy import/call is allowed.
- [ ] Run focused tests; expect PASS; run `graphify update .` without staging graph dirt.
- [ ] Commit `git add partner/icdc/topology_prior.py tests/test_icdc_topology_prior.py && git commit -m "feat: add sparse topology labels and losses"`.

### Task 3: Bounded proposals and exact teacher path

**Files:** Modify `partner/icdc/topology_prior.py`; modify `tests/test_icdc_topology_prior.py`.

**Interfaces:** Define frozen
`ProposalConfig(axis_exchange_cap:int=8,pin_repair_cap:int=8,group_contact_cap:int=8,total_cap:int=32)`
and `ProposalResult(name: str, rects: torch.Tensor, legal: torch.Tensor,
drift: torch.Tensor, hard_checks: Dict[str,bool], cost: Optional[float],
label: Optional[TopologyLabel])`. Export
`generate_proposals(raw_rects: torch.Tensor, case: Mapping[str, Any],
cfg: ProposalConfig)->Iterator[Tuple[str,torch.Tensor]]` and
`pin_feasible_then_exact_tfdl(proposal: torch.Tensor,
case: Mapping[str, Any])->Optional[Tuple[torch.Tensor,torch.Tensor]]`. Keep the
admission predicates independently testable as
`is_acyclic(n:int, edges:Sequence[Tuple[int,int]])->bool`,
`matches_preplaced_origins(rects:torch.Tensor, case:Mapping[str,Any])->bool`,
and `has_exact_positive_contact(rects:torch.Tensor, a:int, b:int, axis:int,
a_before_b:bool, perp_margin:float)->bool`.

- [ ] Write these concrete RED tests (the local `proposal_fixture()` returns a
  four-block sanitized case with one preplaced block and one disconnected
  cluster pair (the full five-column `cons` schema, not a two-column group
  shorthand):
```python
def test_proposals_are_deterministic_named_and_capped():
    raw, case = proposal_fixture()
    cfg = ProposalConfig(1, 1, 1, 3)
    first = list(generate_proposals(raw, case, cfg))
    second = list(generate_proposals(raw.clone(), case, cfg))
    assert [name for name, _ in first] == [name for name, _ in second]
    assert [r.tolist() for _, r in first] == [r.tolist() for _, r in second]
    assert len(first) <= 3
    assert len({name for name, _ in first}) == len(first)
    assert all(name == "base" or name.startswith(("axis:", "pin:", "contact:"))
               for name, _ in first)


def test_cycle_pin_drift_and_corner_only_contact_are_rejected():
    assert not is_acyclic(3, [(0, 1), (1, 2), (2, 0)])
    _, case = proposal_fixture()
    drifted = torch.tensor([[[.25, 0., 1., 1.], [1., 0., 1., 1.],
                              [2., 1., 1., 1.], [3., 0., 1., 1.]]],
                            dtype=torch.float64)
    assert not matches_preplaced_origins(drifted, case)
    corner_only = torch.tensor([[[0., 0., 1., 1.], [1., 1., 1., 1.]]],
                               dtype=torch.float64)
    assert not has_exact_positive_contact(corner_only, 0, 1, 0, True, .1)


def test_group_contact_requires_exact_abutment_and_positive_overlap():
    exact = torch.tensor([[[0., 0., 1., 2.], [1., .5, 1., 2.]]],
                         dtype=torch.float64)
    gap = exact.clone(); gap[0, 1, 0] += 1e-4
    assert has_exact_positive_contact(exact, 0, 1, 0, True, 1.)
    assert not has_exact_positive_contact(gap, 0, 1, 0, True, 1.)
```
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py -k 'proposal or feasible' -q`; expect RED.
- [ ] Implement deterministic proposal names and stable lexicographic ordering:
  base; at most eight axis/order exchanges made by minimally crossing the
  selected separation decision boundary; at most eight preplaced-aware path
  repairs; and at most eight missing grouping contacts with exact abutment and
  positive perpendicular-overlap seeds. Deduplicate by topology fingerprint
  and stop at `total_cap`. The generator consumes a CPU floating `[N,4]`
  coordinate seed and emits CPU float64 `[N,4]` proposals; coordinate
  mutations must realize the requested topology when the public TFDL path
  recomputes it, with no hidden graph override. Apply mutations in the fixed
  base → axis/order → pin → contact order. Axis/order proposals cross the
  selected pair threshold by moving only unpinned origins. Pin repairs set
  all preplaced origins and repair the reverse incoming relation through the
  unpinned peer. Contacts bridge components only through exact face abutment
  with positive perpendicular overlap; dimensions and preplaced geometry stay
  fixed. The realized fingerprint includes every pair axis/direction and exact
  cluster-contact relation; the first realized fingerprint wins.

  The Task 3 case adapter consumes the full sanitized schema with
  `cons[:,0]=fixed`, `cons[:,1]=preplaced`, `cons[:,2]=MIB`,
  `cons[:,3]=cluster`, and `cons[:,4]=boundary`, plus `area[N]` and
  `tp[N,4]`. It batches each proposal as `[1,N,4]`, masks `area > 0`, and
  derives `pinned=(cons[:,1] != 0) & authorized_tp_origins`,
  `pin_xy=tp[:,:2]`, and `boundary_code=cons[:,4]` (or zeros for a uniform
  two-column boundary input). The adapter preserves dimensions and fixed
  geometry from the case rather than inferring them from proposal coordinates.

  Admission runs `T.tfdl` nonexact from the original seed as the
  equality-pinned feasibility check, requiring literal zero drift and bit-exact
  preplaced origins, then runs exact TFDL from that same original seed and
  calls `engine.verify_hard_legal(P_np, area_np, cons_np, tp_np)`, whose hard
  dictionary must be all true. That verifier checks shape, positive area,
  dimensions, preplaced origins, and overlap only; it does not verify
  boundary, grouping, or MIB semantics. The nonexact and exact calls each use
  the same original seed independently and each must pass its own literal
  zero-drift and pin-equality checks.
  Reject cycles, non-finite output, any drift, overlap/hard failure, corner-only
  contact, loss of the intended exact grouping contact after projection, or
  shelf use. The public TFDL path must not call `shelf_fallback`; enforce that
  boundary with an AST check rather than runtime detection. Task 3 verifies
  only hard legality and contact intent; it does not verify group/boundary/MIB
  or recompute the full soft profile. Keep `cost=None` and `label=None` in
  every `ProposalResult`; Task 3 never imports/calls energy or
  `extract_sparse_label`. Task 4 recomputes the evaluator-faithful soft
  profile/cost and extracts labels.
- [ ] Run focused tests; expect PASS; run `graphify update .` without staging graph dirt; commit `git add partner/icdc/topology_prior.py tests/test_icdc_topology_prior.py && git commit -m "feat: add exact topology proposal path"`.

### Task 4: Deterministic teacher and G0

**Files:** Create `scripts/probes/icdc_topology_teacher.py`; modify `tests/test_icdc_topology_prior.py`.

- [ ] Write these RED tests. `write_teacher_fixture(tmp_path)` creates one
  train and one held-out non-validation case plus a tiny same-shape checkpoint;
  it is test data, never a validation loader:
```python
def test_teacher_cli_writes_split_evidence(tmp_path):
    data_root, checkpoint = write_teacher_fixture(tmp_path)
    out = tmp_path / "out"
    rc = teacher_main([
        "--data-root", str(data_root), "--index-out", str(out / "training_index.json"),
        "--checkpoint", str(checkpoint), "--out-dir", str(out),
        "--seed", "20260813", "--heldout-mod", "2", "--n-min", "1",
    ])
    assert rc == 0
    assert {p.name for p in out.iterdir()} == {
        "train_corpus.jsonl", "heldout_corpus.jsonl", "train_labels.jsonl",
        "heldout_labels.jsonl", "proposals.jsonl", "rejections.jsonl",
        "training_index.json", "g0_manifest.json",
    }


def test_teacher_rejects_validation_provenance_before_model_load(tmp_path,
                                                                 monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("checkpoint load happened before provenance rejection")
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="validation provenance"):
        teacher_main(["--data-root", str(tmp_path / "validation"),
                      "--checkpoint", str(tmp_path / "x.pt"),
                      "--out-dir", str(tmp_path / "out")])


def test_teacher_ast_forbids_validation_and_golden_reads():
    forbidden = {"load_test_cases", "FloorplanDatasetLiteTest"}
    paths = [Path("scripts/probes/icdc_topology_teacher.py")]
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in forbidden:
                    findings.append((path, node.lineno, node.id))
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                findings.append((path, node.lineno, node.attr))
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "golden"
                    and isinstance(node.ctx, ast.Load)):
                findings.append((path, node.lineno, "golden"))
    assert findings == []
```
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py -k teacher -q`; expect RED.
- [ ] Implement exact CLI:
```bash
uv run python scripts/probes/icdc_topology_teacher.py \
  --data-root FloorSet/floorset_lite \
  --index-out artifacts/icdc_topology/training_index.json \
  --checkpoint partner/checkpoints/direct_v2_cont/eval_step1p2M.pt \
  --out-dir artifacts/icdc_topology \
  --seed 20260813 --heldout-mod 10 --n-min 100
```
Build the index by sorted approved worker/layout paths and in-file row,
recording every discovered row (including `excluded_n_min`), exact train plus
held-out partition of eligible rows, source bytes SHA, row count, block count,
and input fingerprint before deterministic `split_for_id`. Use a stable
hash-derived seed from global seed, instance, and sample ordinal. The exact
canonical source root is checked before any checkpoint load; reject symlinks or
noncanonical roots in production. Hash exact checkpoint bytes, then deserialize
the same bytes with `torch.load(BytesIO(...), weights_only=True)`.

Process source/corpus rows as a stream; never materialize the approximately
million-case connectivity set. Production rejects missing scorer dependencies
(including Shapely) or scorer source SHA/version mismatches before G0. Verify
default checkpoint SHA
`508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659`, canonical
config, state keys/shapes/dtypes, and selected EMA tensor identity; missing EMA
or mismatch fails closed. Fixtures may inject an expected hash only through a
test helper, never a permissive CLI bypass.

Derive `tp` only from fixed/preplaced input geometry and retain only metric
references, never golden coordinates. Every generated candidate envelope in
canonical `proposals.jsonl` binds receipt, partition, instance, seed, ordinal,
name, intended/seed/realized fingerprint or named intent, admission status and
reason, drift/hard evidence, diagnostic energy, official score/feasibility,
winner/status. Opaque public Task 3 admission failures may be
`admission_failed`; zero-drift/hard claims apply only to admitted, scored,
winner records and coverage. Recheck axis/pin/contact intent locally and
publicly after projection; do not reopen the Task 3 API.

Call official `evaluate_solution(runtime=1, median_runtime=1)` on every exact
admitted intent-surviving candidate; call `EN.energy` only afterward for
diagnostics. Deterministically select `(cost,candidate_ordinal,name)` with
baseline included. If no improvement, record the baseline winner,
`teacher_cost==base_cost`, explicit status and evidence, and `record_weight=1`.
Coverage is exactly one official winner per eligible held-out case; separately
record admission, intent survival, and positive-gain weighted coverage. Compute
`B_H`, `T_H`, and `Delta_H` over ordered eligible `H` with `w=exp(n/12)`, bind
population hash and denominator, and emit the six terminal G0 states. `T_H>1.5`
is diagnostic kill only; hard/target delta gates are
`0.0181504738793652`/`0.0261247299384228`, and no training is authorized below
the target.

Write outputs transactionally to staging, require `index-out` to equal exactly
`out-dir/training_index.json`, and never write a completed manifest on an
infrastructure/write failure. A complete gate manifest has `status=complete`
and a terminal G0 state. Canonical artifacts contain no timestamps or absolute
paths. Manifest hashes all eight artifacts: `train_corpus`, `heldout_corpus`,
`train_labels`, `heldout_labels`, `proposals`, `rejections`,
`training_index`, and `g0_manifest`.
- [ ] Run `uv run python scripts/probes/icdc_topology_teacher.py --help` and fixture test; expect PASS; run `graphify update .` without staging graph dirt.
- [ ] Add red-team tests for root/symlink rejection before load, verified-byte TOCTOU, exact partition/manifest, soft golden masking, byte-identical rerun, official-vs-energy selection, delta/population hash, intent/rejection accounting, no-improvement baseline, EMA identity, transactional output, and fail-closed gate. Keep vertical fixture/`--help`; an explicit bounded `--max-files` plumbing slice must never authorize G0. The real full-corpus command remains unchanged.
- [ ] Commit `git add scripts/probes/icdc_topology_teacher.py tests/test_icdc_topology_prior.py && git commit -m "feat: add deterministic topology teacher"`.

### Task 5: Same-shape trainer and contract

**Files:** Create `partner/icdc/train_topology_prior.py`; modify `tests/test_icdc_topology_prior.py`.

**Interface:** `checkpoint_contract(base: Mapping[str, Any],
candidate: Mapping[str, Any], *, sampler_method: str = "dpmpp",
sampler_steps: int = 2, candidate_count: int = 6,
portfolio_contract_sha256: str)->Dict[str, Any]` verifies config, Direct
sampler, total candidate count, portfolio hash, state keys, shapes, and dtypes.
Also export `frozen_portfolio_contract(flow_slots:int=3,nref:int=6)`,
`canonical_checkpoint_identity(checkpoint)->Dict[str,str]`, and the versioned
canonical state encoder. The encoder hashes sorted UTF-8 keys followed by NUL,
dtype, NUL, compact JSON shape, NUL, and CPU-contiguous tensor bytes.

- [ ] Write this concrete RED contract test:
```python
def test_checkpoint_contract_requires_six_candidates_and_3d3f_portfolio_hash():
    base = tiny_checkpoint()
    candidate = copy.deepcopy(base)
    portfolio = frozen_portfolio_contract(flow_slots=3, nref=6)
    assert checkpoint_contract(
        base, candidate, sampler_method="dpmpp", sampler_steps=2,
        candidate_count=6, portfolio_contract_sha256=portfolio["sha256"],
    )["ok"]
    assert not checkpoint_contract(
        base, candidate, sampler_method="dpmpp", sampler_steps=2,
        candidate_count=4, portfolio_contract_sha256=portfolio["sha256"],
    )["ok"]
    assert not checkpoint_contract(
        base, candidate, sampler_method="dpmpp", sampler_steps=2,
        candidate_count=6, portfolio_contract_sha256="wrong",
    )["ok"]
    changed_shape = copy.deepcopy(candidate)
    changed_shape["model"][next(iter(changed_shape["model"]))] = torch.zeros(3)
    assert not checkpoint_contract(
        base, changed_shape, sampler_method="dpmpp", sampler_steps=2,
        candidate_count=6, portfolio_contract_sha256=portfolio["sha256"],
    )["ok"]


def test_trainer_ast_forbids_validation_and_golden_reads():
    path = Path("partner/icdc/train_topology_prior.py")
    tree = ast.parse(path.read_text())
    forbidden = {"load_test_cases", "FloorplanDatasetLiteTest"}
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in forbidden:
                findings.append((node.lineno, node.id))
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            findings.append((node.lineno, node.attr))
        if (isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "golden"
                and isinstance(node.ctx, ast.Load)):
            findings.append((node.lineno, "golden"))
    assert findings == []


def test_source_contract_rejects_c0_that_is_not_source_ema():
    source = tiny_checkpoint(model_fill=1., ema_fill=2.)
    c0 = tiny_checkpoint(model_fill=3., ema_fill=3.)
    flow = tiny_flow_checkpoint()
    result = verify_source_contract(source, c0, flow)
    assert not result["ok"]
    assert "source_ema_equals_c0" in result["failed"]
```
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py -k trainer -q`; expect RED.
- [ ] Implement production-aligned `sample_differentiable` using exactly two
  DPM++ steps and exactly three samples per instance, stored label seeds/noise,
  `energy.decode_rects`, sparse
  separation/contact/pin-edge losses, base-weight/EMA anchors, and
  collapse/output-diversity metrics. Train only on the training split; select
  `best.pt` only on deterministic held-out constraint/teacher-gain evidence.
  Save model/EMA/config with the base key set, tensor shapes and dtypes, plus a
  contract metadata block declaring `sampler="dpmpp"`, `sampler_steps=2`,
  `student_samples=3`, `production_candidates=6`, and the canonical
  `P_B_3D3F` hash. Never compute energy on raw
  coordinates or import/load validation.
- [ ] Before training, write `source_contract.json` and stop unless canonical
  identities prove
  `source.ema_state_sha256 == c0.model_state_sha256 == c0.ema_state_sha256`.
  Bind these known file identities: source training checkpoint
  `508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659`,
  C0 submission Direct
  `2b9ce827aed93443e442a002d178e8e6282cb4c6148818c9122a6ff0411c8a02`,
  their canonical EMA/model state
  `0efb3c706d627f6230e6f550d83e88741dc1f5a95e6c3450d7ed1e4a882a4d87`,
  and Flow file
  `110c1d84d74ee88d94cf8d3be9ac464602747db8c301d95b3ca69a2cb8bd2f09`.
- [ ] Run exact CLI:
```bash
uv run python -m partner.icdc.train_topology_prior \
  --checkpoint partner/checkpoints/direct_v2_cont/eval_step1p2M.pt \
  --control-checkpoint submission/cadc1013/checkpoints/direct_v2_final.pt \
  --flow-checkpoint submission/cadc1013/checkpoints/flow_matching_v1_final.pt \
  --train-corpus artifacts/icdc_topology/train_corpus.jsonl \
  --train-labels artifacts/icdc_topology/train_labels.jsonl \
  --heldout-corpus artifacts/icdc_topology/heldout_corpus.jsonl \
  --heldout-labels artifacts/icdc_topology/heldout_labels.jsonl \
  --out-dir artifacts/icdc_topology/checkpoints \
  --sampler-steps 2 --student-samples 3 --production-candidates 6 \
  --max-steps 5000 --batch 4 \
  --eval-every 250 --seed 20260813
```
  Expect `latest.pt`, held-out-selected `best.pt`, `source_contract.json`,
  `contract.json`, and `train_log.jsonl`; run `graphify update .` without
  staging graph dirt. Neither mutable checkpoint is authorized for full100.
- [ ] Commit `git add partner/icdc/train_topology_prior.py tests/test_icdc_topology_prior.py && git commit -m "feat: train same-shape topology prior"`.

### Task 6: Freeze G1 contract, implement fail-closed gates, then run once

**Files:** Create `scripts/probes/icdc_topology_gate.py`,
`scripts/probes/icdc_topology_3d3f_wrapper.py`, and
`scripts/probes/run_icdc_topology_stage.sh`; modify
`tests/test_icdc_topology_prior.py`.

**Interfaces:** Define `PortfolioContractError` and `StageFreezeError`; export
`build_solver_environment(inherited,direct_ckpt,flow_ckpt)`,
`environment_identities(execution_env)`,
`validate_portfolio_receipt(receipt)`,
`validate_arm_receipt(receipt,expected_cases=100)`,
`validate_matched_receipts(control,candidate)`,
`authorize_blind_pair(schedule,stage,freeze_manifest)`,
`freeze_stage_checkpoint(selected,stage_path,manifest_path,bindings)`, and
`adjudicate_g1(control_arm,candidate_arm)`, where each arm record includes
score, runtime, feasibility, errors, receipt, freeze, environment, and
checkpoint validity. The canonical schedule has exactly one stage,
`g1_n100_3d3f`; a pass returns `HIGH_TAIL_CAUSAL_PROOF` and then the terminal
`STOP_REQUIRES_SEPARATE_APPROVAL`. There are no G2 stages.

- [ ] Write these concrete RED gate tests:
```python
def test_3d3f_normal_pool_contract_is_exact_and_failure_is_not_a_fallback():
    quota = allocate_quotas(6, {"direct": 3, "flow": 3}, ("direct", "flow"))
    assert quota == {"direct": 3, "flow": 3}
    valid = {
        "pool_ready": True, "requested_K": 6, "raw_direct_count": 6,
        "requested_quota": quota, "resolved_quota": quota,
        "successful_flow_count": 3, "post_mix": quota,
        "post_candidate_count": 6, "oversample": False,
        "retrieval_count": 0, "gpu_second_wave": False,
        "flow_exception": None,
    }
    assert validate_portfolio_receipt(valid)["ok"]
    with pytest.raises(PortfolioContractError, match="flow_exception"):
        validate_portfolio_receipt({**valid, "flow_exception": "boom"})
    with pytest.raises(PortfolioContractError, match="pool_ready"):
        validate_portfolio_receipt({**valid, "pool_ready": False})


def test_solver_environment_scrubs_inherited_solver_namespaces():
    inherited = {"PATH": "/bin", "HOME": "/tmp/user", "PARTNER_EVIL": "1",
                 "DIRECT_OFF": "1", "FLOW_EXTRA": "1", "VKILL_FAKE": "1"}
    env = build_solver_environment(inherited, Path("opaque.pt"), Path("flow.pt"))
    assert env["PARTNER_NREF"] == "6"
    assert env["PARTNER_FLOW_SLOTS"] == "3"
    assert env["PARTNER_DIRECT_SOLVER"] == "dpmpp"
    assert env["PARTNER_DDIM_STEPS"] == "2"
    assert env["PARTNER_FLOW_SOLVER"] == "euler"
    assert env["PARTNER_FLOW_STEPS"] == "8"
    assert not ({"PARTNER_EVIL", "FLOW_EXTRA", "VKILL_FAKE"} & env.keys())
    assert not env.get("DIRECT_OFF")


def test_matched_environment_hash_masks_only_declared_arm_transport_fields():
    control = build_solver_environment({}, Path("control.pt"), Path("flow.pt"))
    candidate = build_solver_environment({}, Path("candidate.pt"), Path("flow.pt"))
    control.update({"FLOORSET_TOPOLOGY_RECEIPT": "/tmp/A.json",
                    "FLOORSET_OPAQUE_ARM_ID": "A"})
    candidate.update({"FLOORSET_TOPOLOGY_RECEIPT": "/tmp/B.json",
                      "FLOORSET_OPAQUE_ARM_ID": "B"})
    c_id = environment_identities(control)
    x_id = environment_identities(candidate)
    assert c_id["full_execution_env_sha256"] != x_id["full_execution_env_sha256"]
    assert c_id["matched_solver_env_sha256"] == x_id["matched_solver_env_sha256"]
    candidate["PARTNER_FLOW_STEPS"] = "7"
    with pytest.raises(PortfolioContractError, match="environment delta"):
        validate_matched_receipts(
            valid_arm_receipt(environment=control),
            valid_arm_receipt(environment=candidate),
        )


def test_schedule_rejects_blind_pair_before_sealed_stage_freeze(tmp_path):
    schedule = frozen_schedule_3d3f()
    with pytest.raises(StageFreezeError, match="freeze manifest"):
        authorize_blind_pair(schedule, "g1_n100_3d3f", tmp_path / "missing.json")
    freeze = write_test_freeze_manifest(tmp_path, stage="g1_n100_3d3f")
    assert authorize_blind_pair(schedule, "g1_n100_3d3f", freeze)["authorized"]


def test_arm_receipts_require_all_cases_and_identical_status_vectors():
    control = valid_arm_receipt(case_statuses=["direct_gate_closed"] * 99
                                + ["normal_pool_valid"])
    candidate = copy.deepcopy(control)
    assert validate_arm_receipt(control)["ok"]
    assert validate_matched_receipts(control, candidate)["ok"]
    candidate["cases"][-1]["status"] = "direct_gate_closed"
    with pytest.raises(PortfolioContractError, match="status vector"):
        validate_matched_receipts(control, candidate)
    with pytest.raises(PortfolioContractError, match="100"):
        validate_arm_receipt({**control, "cases": control["cases"][:-1]})


def test_g1_rejects_invalid_control_before_score_comparison():
    control = valid_arm(score=1.145, runtime=.299)
    candidate = valid_arm(score=1.128, runtime=.300)
    control["errors"] = 1
    decision = adjudicate_g1(control, candidate)
    assert decision["status"] == "terminate"
    assert "control_errors" in decision["failed"]
    control = valid_arm(score=1.145, runtime=.301)
    decision = adjudicate_g1(control, candidate)
    assert "control_runtime" in decision["failed"]


def test_g1_requires_both_absolute_and_matched_causal_gain():
    decision = adjudicate_g1(
        valid_arm(score=1.120, runtime=.299),
        valid_arm(score=1.110, runtime=.299),
    )
    assert decision["status"] == "terminate"
    assert "causal_gain" in decision["failed"]
    decision = adjudicate_g1(
        valid_arm(score=1.145, runtime=.300),
        valid_arm(score=1.128, runtime=.300),
    )
    assert decision["status"] == "HIGH_TAIL_CAUSAL_PROOF"
    assert decision["next"] == "STOP_REQUIRES_SEPARATE_APPROVAL"


def test_freeze_rejects_candidate_changed_after_audit_or_smoke(tmp_path):
    selected = write_tiny_checkpoint(tmp_path / "best.pt", fill=1.)
    audit = write_identity_artifact(tmp_path / "audit.json", selected)
    smoke = write_identity_artifact(tmp_path / "smoke.json", selected)
    write_tiny_checkpoint(selected, fill=2.)
    with pytest.raises(StageFreezeError, match="candidate identity"):
        freeze_stage_checkpoint(
            selected, tmp_path / "g1.pt", tmp_path / "g1.freeze.json",
            {"heldout_audit": audit, "causal_smoke": smoke},
        )
```
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py -k 'portfolio or
  environment or schedule or g1' -q`; expect RED before implementation.
- [ ] Implement `P_B_3D3F` as a canonical exact mapping. Build each evaluator
  subprocess environment from an allowlist of generic process keys plus only
  these solver values; never mutate or inherit a solver namespace:
```text
DIRECT_OFF=""; DIRECT_CKPT=<opaque arm>; FLOW_CKPT=<frozen Flow>
PARTNER_POOL=24; PARTNER_NREF=6; PARTNER_NREF_MIN_N=95
PARTNER_DIRECT_MIN=.3; PARTNER_DIRECT_SEAT_FIX=0
PARTNER_OVERSAMPLE=4; PARTNER_KS_CAP=56
PARTNER_DIRECT_SOLVER=dpmpp; PARTNER_DDIM_STEPS=2
PARTNER_FLOW_SLOTS=3; PARTNER_FLOW_SOLVER=euler; PARTNER_FLOW_STEPS=8
PARTNER_FLOW_ANTITHETIC=1
VKILL_OFF=1; PARTNER_PRESCREEN_V=1; PARTNER_TAG_ANCHOR_EXTRA=3
PARTNER_BUDGET_SCALE=8.498e-5; PARTNER_BUDGET_TAU=12
PARTNER_BUDGET_MIN=.05; PARTNER_BUDGET_MAX=1.22; PARTNER_POOL_GATE=0
PARTNER_REFINE_STALL_STOP=1; PARTNER_SA_KERNEL=numba
PARTNER_REFINE_KERNEL=numba; PARTNER_REFINE_FASTBUILD=1
PARTNER_FAST_SETUP=1; PARTNER_EDGE_SEAT_V2=1; PARTNER_FRAME_WPIN=1
PARTNER_COORD_POLISH=1; PARTNER_FRAME_SCALE_LADDER=1
PARTNER_FRAME_SCALE_SET=1.02; PARTNER_SEAT_FINAL=1
PARTNER_TAG_COMPRESS=1; PARTNER_GROUP_BRIDGE=1
PARTNER_FLOW_ZORDER=""; PARTNER_FLOW_NOPT=""; PARTNER_NOISE_OPT=""
PARTNER_PHYSICS_GUIDE=""; PARTNER_GPU_ARM=0
PARTNER_RETRIEVAL_INDEX=""; PARTNER_RETRIEVAL_SLOTS=0
PARTNER_GROUP_DAG_BRIDGE=""; PARTNER_GROUP_DAG_BRIDGE_DEBUG=""
PARTNER_ORACLE_PRED_FILE=""; PARTNER_PSEL_DUMP=""; PARTNER_SEAT_DEBUG=""
PARTNER_DIRECT_WARM=""; PARTNER_TAG_COMPRESS_DEBUG=""
PARTNER_GROUP_BRIDGE_DEBUG=""; PYTHONHASHSEED=0
```
  Empty strings above are intentional: several inherited modules use raw
  `bool(os.environ.get(...))`, so the string `"0"` would incorrectly enable
  or disable behavior (notably `DIRECT_OFF`). Tests must lock this truthiness.
  Allowlist only the generic keys required to launch Python (`PATH`, `HOME`,
  `TMPDIR`, and `CUDA_VISIBLE_DEVICES`) and record them; they may not control
  solver behavior. Do not inherit `PYTHONPATH`; the wrapper bootstraps its exact
  repository dependency paths. For every arm retain
  `full_execution_env_sha256` over the literal launched environment. Separately
  compute `matched_solver_env_sha256` by replacing `DIRECT_CKPT` with the fixed
  token `<OPAQUE_DIRECT_ARM>` and excluding exactly
  `FLOORSET_TOPOLOGY_RECEIPT` and `FLOORSET_OPAQUE_ARM_ID`. The closed allowlist
  of per-arm deltas is therefore those three fields only; validate the actual
  Direct identities separately against the sealed freeze mapping. Reject any
  other key/value delta. Also hash the evaluator, wrapper, source commit,
  hardware receipt, and both checkpoints.
- [ ] Implement behavior-preserving tracing at the real call boundaries; do
  not rely on an exception raised inside inherited sampling, because production
  intentionally swallows Flow, sampler, parallel-pool, and outer-solve failures.
  The wrapper must bootstrap its absolute repository `partner/` path before
  importing `contest_optimizer`; it must work from
  `FloorSet/iccad2026contest/` with an empty ambient `PYTHONPATH`.

  Install narrowly scoped hooks that return the original value unchanged:

  1. wrap `contest_optimizer._direct_seat_ok` to record the actual per-case
     Direct-gate decision;
  2. while subclass `_sample_direct_raw_preds` is active, wrap the module's
     `z_to_rectangles` boundary to hash and count all six decoded raw Direct
     candidates before Flow replacement;
  3. override `_sample_flow_preds` to record the requested/returned three Flow
     arrays and capture/re-raise its exception so the inherited catch remains
     behavior-identical;
  4. after inherited `_sample_direct_raw_preds` returns, prove its exact ordered
     output is raw Direct `[0:3]` followed by the recorded Flow `[0:3]`, with
     `K=6`, `oversample=False`, and six total;
  5. wrap `column_sa_legalizer._parallel_solve` to record entry, successful
     return, or exception/fallback; and
  6. wrap the actual `contest_optimizer._fallback_row` boundary to record any
     outer row fallback.

  The subclass `solve` owns a per-case trace. At entry assert both models are
  loaded and record ordinal/block count/pool readiness. If the Direct gate is
  false, emit `direct_gate_closed` and require no sampling event. If true,
  require the normal pool and exactly one valid K6 3D/3F call; any no-pool
  `_direct_worker`, K4/K8, Flow exception, empty sampler, parallel exception,
  sequential fallback, second wave, retrieval, source/count/order mismatch, or
  outer row fallback marks the case invalid. Never raise from an inner hook.
  After `super().solve` returns but before returning positions to the evaluator,
  an invalid case atomically writes an invalid receipt (`fsync` temporary file,
  `os.replace`) and calls `os._exit(86)`. Thus the evaluator process ends
  non-zero before that case can be scored, rather than converting the failure
  to an error row. Valid hooks must not modify arrays, source order, ranking,
  refinement, or selection.

  Accumulate valid case records in memory. Register one `atexit` writer that
  atomically emits the normal arm receipt after all timed evaluator calls. The
  runner sets an arm-specific `FLOORSET_TOPOLOGY_RECEIPT` path and accepts it
  only with `schema_version`, opaque arm ID, `complete=true`, 100 ordered case
  entries, environment/checkpoint hashes, and no invalid/fallback events. A
  missing/incomplete receipt or non-zero process is an invalid arm. Receipt
  hashing is runner work after subprocess exit and is not charged to a case.
- [ ] Add real-boundary regression tests before implementation. Use a child
  process launched from `FloorSet/iccad2026contest/` with `PYTHONPATH` absent;
  its fixture imports the actual wrapper/base modules and injects one fault at
  a named real hook. The parameterized test must prove `returncode==86`, an
  atomic `complete=false` receipt, and no scored output for each of
  `flow_exception`, `pool_not_ready`, `parallel_solve_exception`, and
  `outer_fallback_row`. A success fixture must return zero, produce
  `complete=true`, and preserve byte-identical candidate hashes/order through
  every tracing hook. Also unit-test that `validate_arm_receipt` rejects a
  missing normal-pool record, direct-only fallback, duplicated case ordinal,
  unexpected K4/K8, and incomplete completion marker.
- [ ] Implement the held-out audit and non-validation causal smoke before any
  freeze or full100:
```bash
uv run python scripts/probes/icdc_topology_gate.py audit \
  --stage g1_n100_3d3f \
  --corpus artifacts/icdc_topology/heldout_corpus.jsonl \
  --labels artifacts/icdc_topology/heldout_labels.jsonl \
  --teacher-manifest artifacts/icdc_topology/g0_manifest.json \
  --base partner/checkpoints/direct_v2_cont/eval_step1p2M.pt \
  --candidate artifacts/icdc_topology/checkpoints/best.pt \
  --out artifacts/icdc_topology/g1_heldout.json
uv run python scripts/probes/icdc_topology_gate.py causal-smoke \
  --corpus artifacts/icdc_topology/heldout_corpus.jsonl \
  --control submission/cadc1013/checkpoints/direct_v2_final.pt \
  --candidate artifacts/icdc_topology/checkpoints/best.pt \
  --flow submission/cadc1013/checkpoints/flow_matching_v1_final.pt \
  --out artifacts/icdc_topology/g1_causal_smoke.json
```
  Audit requires the same-shape contract, exact TFDL, zero drift/shelf/hard
  errors, and weighted retained gain
  `(base_cost-student_cost)/(base_cost-teacher_cost)>=.75` over positive teacher
  gain. Smoke uses only predeclared non-validation witnesses and fixed seeds;
  require different Direct raw hashes, identical Flow raw hashes, exact 3D/3F
  receipts, and at least one changed post-rank candidate or final-layout hash.
  It is causal reachability evidence, never a quality/checkpoint-selection gate.
  Both JSON artifacts must embed the candidate's full canonical
  file/model/EMA/keyset/config identity, not only its path.
- [ ] Freeze `best.pt` atomically as
  `artifacts/icdc_topology/checkpoints/g1_n100_3d3f.pt`, then write canonical
  `g1_n100_3d3f.freeze.json`; chmod both `0444`. Bind file/model/EMA/keyset/config
  identities, C0 and Flow identities, portfolio/environment, teacher manifest,
  training index, held-out audit, causal smoke, canonical schedule, source
  commit, and seeds. Recompute every binding before each arm; permissions alone
  never establish immutability. Reject unless the canonical candidate identity
  embedded in both the prior held-out audit and causal smoke exactly equals the
  selected bytes being frozen. The mutation-between-smoke-and-freeze RED test
  above must fail before copy. Mutable `best.pt` is never accepted by runner.
- [ ] Commit the complete gate, fail-closed wrapper, runner, tests, and canonical
  schedule **before** any full100:
```bash
git add scripts/probes/icdc_topology_gate.py \
  scripts/probes/icdc_topology_3d3f_wrapper.py \
  scripts/probes/run_icdc_topology_stage.sh tests/test_icdc_topology_prior.py
git commit -m "feat: freeze 3d3f topology prior g1 gate"
```
- [ ] Prepare a sealed A/B mapping and run the only authorized pair through one
  non-interactive command. The runner must not print, parse, or expose arm A's
  score before arm B completes. A failure/missing arm terminates; there is no
  rerun, reorder, replacement checkpoint, or adaptation:
```bash
bash scripts/probes/run_icdc_topology_stage.sh sealed-pair \
  --stage g1_n100_3d3f \
  --freeze artifacts/icdc_topology/checkpoints/g1_n100_3d3f.freeze.json \
  --out-dir artifacts/icdc_topology/g1_blind
```
  Both subprocesses use the official evaluator and
  `scripts/probes/icdc_topology_3d3f_wrapper.py`; solver behavior differs only
  in `DIRECT_CKPT`, while the two runner-owned receipt destinations and opaque
  arm IDs may differ as transport metadata. Both the literal full-execution
  hashes and normalized matched-solver hashes are sealed.
  The wrapper bootstraps its own repository dependency paths; do not inherit an
  ambient `PYTHONPATH`. Seal output JSON, log, exit status, environment and
  portfolio receipt hashes for both arms before unmasking once.
- [ ] Adjudicate once. First require **each** arm independently to have process
  exit zero, output/receipt completion, freeze/environment/checkpoint validity,
  `100/100`, zero errors, runtime `<=.300`, and no fallback event. Then require
  identical ordered per-case Direct-gate/pool-status vectors and exact valid
  3D/3F receipts for every gate-open case. Only then compare candidate
  no-runtime `<=1.1287448258795715` and `<=control-0.015`. Write `terminate` or
  `HIGH_TAIL_CAUSAL_PROOF` followed by `STOP_REQUIRES_SEPARATE_APPROVAL`:
```bash
uv run python scripts/probes/icdc_topology_gate.py adjudicate-sealed \
  --stage g1_n100_3d3f \
  --freeze artifacts/icdc_topology/checkpoints/g1_n100_3d3f.freeze.json \
  --pair-dir artifacts/icdc_topology/g1_blind \
  --out artifacts/icdc_topology/g1_gate.json
```
- [ ] Run `bash scripts/probes/run_icdc_topology_stage.sh --help`, focused
  tests, full `uv run pytest`, `git diff --check`, and `graphify update .`;
  expect PASS without staging unrelated graph dirt. No package review follows.

### Task 7: Record the authority stop; do not execute G2

**Files:** Create
`docs/experiments/2026-08-13-tfdl-data-free-topology-prior-gates.md` only after
G1 adjudication. Generated evidence remains under `artifacts/icdc_topology/`.

- [ ] Record immutable source/checkpoint/portfolio/environment/schedule hashes,
  G0 and held-out results, causal smoke, sealed pair hashes, the one unmasked G1
  decision, and rollback identities. Preserve failures as evidence; do not
  rewrite thresholds or omit a failed arm.
- [ ] If G1 fails, record `terminate`. If it passes, record
  `HIGH_TAIL_CAUSAL_PROOF` and the mandatory terminal
  `STOP_REQUIRES_SEPARATE_APPROVAL` with reason: Direct is not sampled for
  `n<=98` under the frozen policy.
- [ ] Do not change `PARTNER_DIRECT_MIN`, budget curve/cap, K, candidate mix,
  online topology mechanism, loss, seed, schedule, or checkpoint based on the
  blind result. Do not execute `g2_60_99`, `g2_all_1p05`, `g2_all_1p00`, final
  confirmation, production integration, or package review without a new
  user-approved specification and a fresh matched control/runtime study.
- [ ] Run `uv run pytest tests/test_icdc_topology_prior.py -q`, full
  `uv run pytest`, `git diff --check`, and the placeholder scan over Track-B
  source; run `graphify update .` without staging unrelated graph dirt.

## Self-review

- [ ] Exact interfaces, ownership, metrics, runtime composition, validation
  hygiene, scorer separation, gates, kill/rollback, and artifact paths checked
  against the approved spec after independent review.
- [ ] Concrete RED/GREEN code and command blocks checked for every code-changing
  task; the complete schedule is committed before the first blind full100.
- [ ] Placeholder scan and `git diff --check` completed; graph updates are
  required after each code task without staging unrelated graph dirt.
