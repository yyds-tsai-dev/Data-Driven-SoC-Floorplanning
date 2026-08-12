# Track B: Data-Free Sparse Topology-Constraint Distillation

## Decision and baseline

Track B is approved as a teacher-to-student topology prior, with no production
optimizer integration until Sol serializes that change after the independent
Track-B gates. The
authoritative development baseline is
`artifacts/partner_eval/gbridge_package_full100.json`:

- weighted no-runtime: `1.1437448258795715`
- average runtime: `0.29881621031556277` s/case
- feasibility: `100/100`
- hard errors: `0`
- runtime headroom: `1.1837896844372198` ms/case

No package review is requested until a final result reaches exactly `1.00` at
`<=0.300` s/case.

### Approved 3-Direct / 3-Flow amendment

Preflight found that the historical wrapper requested six refinement seats but
`PARTNER_FLOW_SLOTS=10`; on the normal successful path, quota allocation
therefore replaced all six Direct samples with Flow samples. A changed Direct
checkpoint could not causally affect the historical portfolio. The user
approved this narrow amendment on 2026-08-13:

- total refinement candidates remain exactly `6`;
- the normal pool path contains exactly `3` Direct and `3` Flow candidates;
- Direct uses DPM++ with exactly `2` steps;
- Flow uses Euler with exactly `8` steps;
- the Flow checkpoint, seeds, candidate ranker, six refinement seats, and all
  other solver policy remain frozen;
- retrieval, GPU second-wave, Direct/Flow noise optimization, Flow Z-order,
  physics guidance, and oracle paths are disabled for the causal gate; and
- a no-pool path, Flow failure, oversampling, or any observed mix other than
  `3/3` is a contract failure, not a valid blind result.

The legacy optimizer intentionally swallows Flow, pool, and solver failures,
so raising inside a sampling override is not a reliable invalidation boundary.
The experiment wrapper must instead trace the actual Direct-decode, Flow-call,
Direct-gate, pool, parallel-solve, and outer-row-fallback boundaries without
changing their returned arrays or order. If a contract violation is observed,
the wrapper atomically writes an invalid receipt and terminates the evaluator
subprocess non-zero from the outer subclass `solve`, after inherited `solve`
returns but before the evaluator scores that case. The runner rejects any
non-zero arm, incomplete receipt, missing completion marker, or fallback trace;
an evaluator error row is never accepted as a substitute.

Changing the Flow quota invalidates the historical artifact as a causal
control. Its `1.1437448258795715` score remains the authoritative development
baseline and the source of the absolute G1 outcome bar, but G1 attribution uses
a newly frozen matched pair under the same 3-Direct/3-Flow contract: production
Direct EMA versus the held-out-selected Track-B EMA. Both arms stay concealed
until both full100 artifacts and manifests are immutable.

Historical closure is commit `d31781e`. The old TFDL energy-pooling result had
Spearman `0.9904`; an untrained bank was legal on only `40%` of cases, with the
sole preplaced-drift failure; the best fine-tuned median was `2.65` versus a
`1.09–1.16` break-even range. Coordinate-only energy fine-tuning is dead and
must not be rerun.

## Why the previous approach cannot work

`partner/icdc/tfdl.py::extract_topology` runs under `no_grad` and returns
detached topology features. Axis, order, and contact decisions therefore
cannot receive gradient. Pins are lower bounds, and frozen chains drift; the
saturated auxiliary terms only prevent an infinite bounding box. Relaxed
energy grouping cannot distinguish two disconnected halves that are each
internally connected. Track B must supervise the missing topology constraints,
not imitate coordinates.

## Chosen architecture

The development path is:

1. Freeze the Direct baseline checkpoint and configuration.
2. Generate bounded, deterministic offline topology proposals.
3. Run an equality-pinned preplaced-feasibility check.
4. Evaluate admitted proposals with exact TFDL (no shelf fallback), official
   evaluator scoring, diagnostic energy, and exact boundary/grouping geometry.
5. Convert only sparse critical constraints into labels.
6. Distill those labels into a `DirectDenoiser` checkpoint with the SAME SHAPE.

Production changes the Direct checkpoint inside the approved fixed portfolio:
same configuration and tensor shapes, same DPM++ two-step sampler, exactly
three Direct plus three Flow candidates, and no online TFDL, proposal bank,
dense head, or extra graph pass. The quota amendment changes no model or online
topology machinery and keeps the total candidate/refinement count at six.

### Proposal policy

Proposal generators are limited to:

- low-confidence axis/order exchanges;
- preplaced-aware edge repair; and
- necessary grouping contacts.

Reject a proposal on any cycle, equality-pin infeasibility, preplaced drift,
overlap, shape/area/fixed/preplaced violation, or other exact hard legality
failure. Record deterministic seeds, per-generator caps, source instance,
teacher configuration, and train/held-out provenance for every proposal.

### Sparse label schema

Each label record contains instance and proposal identifiers plus only the
critical constraints retained after transitive reduction:

- x/y separation edge, direction, and minimum margin;
- exact grouping-contact equality and perpendicular positive-overlap margin;
- pin-support paths; and
- critical edges after transitive reduction.

Student outputs remain coordinates and aspect variables. No golden coordinates
are teacher targets. The teacher explains topology constraints, rather than
providing dense labels or exact-coordinate imitation.

#### Frozen sparse-label and loss contract

Topology separation and contact losses are global, detached-effective-weight
normalized means. Separation uses the normalized hinge on every retained edge.
The contact term is a sum (not an average) of normalized absolute face-gap and
normalized overlap-deficit terms; `total = separation + contact`. An empty edge
or contact class contributes a differentiable zero. The loss scale accepts a
finite, nonnegative `[B]` tensor and uses `clamp_min(1e-6)` where it is a
denominator.

Exact contacts require bit-equal face abutment and strictly positive overlap on
the perpendicular axis. The extracted grouping forest consists of observed
teacher contacts selected deterministically by maximum-overlap Kruskal. Margin
is the nonnegative physical face gap. Transitive reduction precedes pin-edge
expansion.

`pin_paths` is audit-only metadata: for each axis it stores incoming critical
support chains terminating at each preplaced block, never p2b. The incoming
predecessor is chosen recursively by greatest physical end, breaking ties by
lower block ID. Duplicate separation and pin edges intentionally remain and
therefore add weight. The fixed batch schema has no separate pin loss.

Record weight is frozen as `base_cost / teacher_cost`, finite and positive;
extraction rejects `teacher_cost > base_cost`. Proposal identity remains the
teacher JSONL envelope/manifest keyed by `instance_id` and `sample_seed`, not
frozen `TopologyLabel` fields.

Run three samples from the production-aligned differentiable two-step sampler
before computing the student losses (this Task 5 trajectory count is unrelated
to Task 4, which uses exactly one fixed teacher sample per eligible case). The
student objective is a weighted sum
of:

- differentiable normalized hinge separation loss
  `mean(max(0, (margin - signed_separation) / scale))` over labeled edges;
- contact loss combining `abs(contact_axis_gap)/scale` with
  `max(0, (perpendicular_margin-overlap)/scale)`;
- pin/topology consistency loss for pin-support paths and required ordering;
- a detached teacher-quality/ranking weight for each sparse-label record; and
- base-checkpoint and EMA-anchor penalties to retain the Direct solution.

Official `cost_no_runtime` is used offline to select proposals and derive the
detached record weight. `EN.energy` is evaluated only after admission as a
diagnostic; it never selects proposals or derives weights. It is not evaluated directly on the student's raw,
possibly overlapping coordinates: doing so would restore the degenerate blob
minimum that killed pre-legalization energy training. The sparse separation,
contact, and pin-margin losses are the only topology gradients supplied to the
same-shape student.

### Task 3 admission boundary

Task 3 is intentionally narrower than teacher scoring. Its generator consumes
a CPU floating `[N,4]` coordinate seed and emits CPU float64 `[N,4]`
proposals. Mutations are deterministic in base → axis/order → pin → contact
order and must realize their topology through the public TFDL recomputation;
there is no hidden graph override. Axis/order mutations move only unpinned
origins across the selected pair threshold. Pin repairs set every preplaced
origin and repair the reverse incoming relation through the unpinned peer.
Contact mutations bridge components only with exact face abutment and positive
perpendicular overlap; dimensions and preplaced geometry remain fixed. The
realized fingerprint covers every pair axis/direction and every exact
cluster-contact relation, and the first realized fingerprint wins.

The Task 3 case adapter consumes the full sanitized case schema:
`cons[:,0]=fixed`, `cons[:,1]=preplaced`, `cons[:,2]=MIB`,
`cons[:,3]=cluster`, `cons[:,4]=boundary`, with `area[N]` and `tp[N,4]`.
Each proposal is batched as `[1,N,4]` with `mask=area>0`,
`pinned=(cons[:,1]!=0)&authorized_tp_origins`, `pin_xy=tp[:,:2]`, and
`boundary_code=cons[:,4]` (or zeros for uniform two-column boundary input).
The proposal fixture therefore uses the full five-column `cons` schema; a
disconnected cluster pair is not represented by a two-column group shortcut.

Admission runs nonexact TFDL and exact TFDL from the original seed, requiring
literal zero drift and bit-exact preplaced origins at both checks, then requires
the engine call `verify_hard_legal(P_np, area_np, cons_np, tp_np)` to return an
all-true hard dictionary. This verifier checks shape, positive area,
dimensions, preplaced origins, and overlap only; it does not verify
boundary/group/MIB semantics. The public TFDL path does not call
`shelf_fallback`; this is enforced with an AST boundary check. Task 3 verifies
hard legality and contact intent only. It does not recompute group/boundary/MIB
soft relations, call energy, or call `extract_sparse_label`; every
`ProposalResult` has `cost=None` and `label=None`. Task 4 recomputes the
evaluator-faithful soft profile/cost before extracting labels.

Monitor constraint satisfaction, drift, hard legality, energy/ranking gain,
collapse (loss diversity and output variance), and proposal/student diversity.
Reject a run that wins by collapsing all outputs to one topology.

## Data and validation hygiene

### Task 1 source receipt contract (implemented)

The sanitized corpus is admitted only from the immutable canonical source root
`FloorSet/floorset_lite`.  Each row is bound to a `CorpusSourceReceipt` carrying
`relative_path`, the source-file SHA256, `layout_index`, and the sanitized input
`fingerprint`; the receipt-derived instance identity is
`relative_path#layout_index`.  The source root and every receipt are required
arguments to `save_sanitized_corpus`—there is no unbound corpus-writing API.

Receipt verification hashes the exact source bytes first, then loads those
bytes through a single in-memory buffer with `torch.load(...,
weights_only=True, map_location="cpu")`.  Only the exact seven-tensor raw shard
schema is accepted: input rows `[B,N,6]`, tree `[B,N-1,3]`, fingerprint
 `[B,N,4]`, and metrics `[B,8]` (with the remaining raw tensors retaining their
 exact validated widths).  All tensors must be finite CPU tensors with
 matching batch dimensions; no arbitrary pickle/object source is trusted.

The source adapter maps raw `(w,h,x,y)` geometry to the canonical
`(x,y,w,h)` representation and masks non-input coordinates before fingerprinting.
Golden coordinates are never serialized or used as targets; raw `fp` is read
only to derive input-authorized fixed/preplaced geometry.  The implementation
verifies `source_root` and the complete receipt list while saving; later sealed
index/manifest artifacts provide the experiment-level reproducibility boundary.

Training and held-out data come from a non-validation instance pool. Validation
cases are never used for proposal tuning, repeated feedback, threshold choice,
or checkpoint selection. The checkpoint is selected solely by held-out gates.
Before the first full100 run, freeze the complete currently authorized
schedule: held-out selection, the immutable `g1_n100_3d3f.pt` checkpoint, 3/3
portfolio/environment contract, matched control and candidate hashes, arm
concealment/order, thresholds, and stop rule. G1 receives exactly one blind
matched pair after both arms are frozen. Its result may only establish the
predeclared high-tail causal proof or terminate; it may not tune training,
thresholds, proposals, losses, seeds, quota, or checkpoint choice.

The exact current weighted decomposition is:

- `n >= 100` weight: `0.8264247038229129`;
- current-band average: `1.1011247299384228`;
- outside contribution: `0.2337481470681254`.

`q_band = 1.0829742560590576` corresponds to full100 minus `0.015` under
100%-replacement. Even perfecting `n >= 100` while leaving all other cases
unchanged yields only `1.0601728508910383`. The approved quota amendment does
not change `PARTNER_DIRECT_MIN=0.3`; under the frozen budget, Direct opens at
`n=99` and remains closed for `n<=98`. Therefore the currently authorized
Track-B work can establish only the high-tail G1 proof. Expansion to `60–98`
and the final `1.00` requires a separate, explicitly approved production-policy
specification for the Direct gate/portfolio and a new matched runtime/control
study. It must not be inferred from the 3/3 approval.

## Gates and stop conditions

### G0 — teacher oracle

Define the eligible held-out population `H` by the immutable receipt-bound
index (`n >= 100`, `split_for_id(instance_id, heldout_mod=10) == "heldout"`), with exactly one fixed
Direct seed per case.  Do not compare an absolute held-out statistic with the
validation/full100-derived `1.075` or `1.0829742560590576` bars.  With
`w=exp(n/12)`, bind the ordered IDs, `n`, weights, denominator, population
SHA, official weighted base mean `B_H`, teacher mean `T_H`, and gain
`Delta_H=B_H-T_H`.

The predictive held-out gates are hard minimum
`Delta_H >= 0.0181504738793652` (the predeclared
`1.1011247299384228 - 1.0829742560590576`) and target
`Delta_H >= 0.0261247299384228` (the predeclared
`1.1011247299384228 - 1.075`).  These are selection gates, not claims that
`T_H` equals the full100 high-tail statistic.  `T_H > 1.5` remains a
diagnostic kill only.  Between hard and target, stop with no training
authority; only the target permits Task 5.  G1 remains the sole causal
transfer proof.

For every generated proposal (base and mutations) that passes exact admission
and named-intent survival, call official `evaluate_solution({"positions": ..., "runtime": 1.0}, ...,
median_runtime=1.0).cost_no_runtime`;
`EN.energy` is called only after exact legal admission, recorded diagnostically,
and never shortlists or filters. Select by deterministic
`(cost_no_runtime, ordinal, name)`, always including baseline; `B_H`, `T_H`,
labels, and record weights use `cost_no_runtime`. If no proposal
improves, select baseline with `teacher_cost == base_cost`, explicit
no-improvement status, and retained label/evidence (`record_weight=1`).  The
coverage denominator is every eligible held-out case and exactly one officially
scored winner per case; separately report mutation admission, intent survival,
and positive-gain weighted coverage.

G0 terminal states are `KILLED_INPUT_CHECKPOINT_OR_SCORER`,
`KILLED_LEGALITY_OR_COVERAGE`, `KILLED_TEACHER_GT_1_5`,
`STOP_HARD_GAIN_MISSED`, `TARGET_GAIN_MET`, and
`TARGET_GAIN_MISSED_NO_TRAINING_AUTHORITY`. Precedence is exactly that order;
retain all secondary reasons. The last state applies only when a policy
condition clears hard but not target, avoiding impossible overlap. Only
`TARGET_GAIN_MET` advances.

### G1 — same-shape student

Require held-out zero drift and hard legality under an exact TFDL audit and
retain at least `75%` of the teacher gain over records with positive teacher
gain. Before the blind pair, a deterministic non-validation causal smoke must
prove that both arms execute the normal pool with exactly six candidates split
3 Direct/3 Flow, Direct raw hashes differ, Flow raw hashes are identical, and
at least one predeclared witness changes after ranking or in the final layout.

Freeze the held-out-selected candidate at
`artifacts/icdc_topology/checkpoints/g1_n100_3d3f.pt` plus an immutable manifest.
The manifest binds the candidate, the C0 submission Direct checkpoint, the Flow
checkpoint, source training checkpoint, portfolio, teacher data, held-out
audit, schedule, seeds, and source commit by SHA256. The known input identities
are:

- training source file:
  `508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659`;
- C0 submission Direct file:
  `2b9ce827aed93443e442a002d178e8e6282cb4c6148818c9122a6ff0411c8a02`;
- source EMA and C0 model/EMA canonical state:
  `0efb3c706d627f6230e6f550d83e88741dc1f5a95e6c3450d7ed1e4a882a4d87`;
  and
- frozen Flow file:
  `110c1d84d74ee88d94cf8d3be9ac464602747db8c301d95b3ca69a2cb8bd2f09`.

Run one concealed matched full100 pair: C0 uses the frozen production Direct
EMA and C1 uses the frozen Track-B EMA, with every other input identical. Do not
unmask or inspect either outcome until both artifacts, logs, environment
receipts, and hashes are sealed. C1 passes only if all of the following hold:

- C0 and C1 each independently have `100/100` feasibility, zero errors,
  average runtime `<=0.300` s/case, valid freeze/environment/checkpoint hashes,
  and complete valid portfolio receipts;
- the arms have identical ordered `(case ordinal, block count, Direct-gate
  status, pool status)` vectors; every Direct-gate-open case has exactly one
  normal-pool receipt and every Direct-gate-closed case is explicitly recorded;
- weighted no-runtime `<=1.1287448258795715`;
- weighted no-runtime `<= C0 - 0.015`;
- the receipts prove Direct DPM++/2, Flow Euler/8, total six, and exact 3D/3F.

Verify that model configuration, state-dict keys, tensor shapes, and dtypes
match the frozen Direct contract. G1 is a high-tail causal proof, not the final
goal.

### G2 — separate authority required

No G2 blind run, Direct-gate change, or all-count checkpoint schedule is
authorized by the 3D/3F amendment. Under the frozen contract, Direct is not
sampled for `n<=98`; reaching `1.10`, `1.05`, or exact `1.00` therefore needs a
separate production-policy specification and explicit user approval. That new
specification must predeclare a lower-count Direct gate/portfolio, matched
control and runtime study, non-validation causal smoke, checkpoints, held-out
counterparts, blind schedule, and confirmation policy. G1 may only transition
to `STOP_REQUIRES_SEPARATE_APPROVAL`, never silently into G2.

Kill the track if the oracle is above `1.5`, proposal coverage or pin support
fails, the student cannot retain the held-out benefit, any receipt violates the
normal-pool contract, either blind arm fails, or the candidate misses either G1
score gate, legality, or runtime. Track A may spend at most `0.75` ms, leaving
approximately `0.434` ms. Track B changes only the Direct checkpoint within the
approved fixed six-candidate 3D/3F portfolio; actual matched full100 runtime is
binding.

## Artifacts, reproducibility, and observability

Task 4 emits exactly eight canonical files: `train_corpus.jsonl`,
`heldout_corpus.jsonl`, `train_labels.jsonl`, `heldout_labels.jsonl`,
`proposals.jsonl`, `rejections.jsonl`, `training_index.json`, and
`g0_manifest.json`. Every proposal envelope binds receipt/partition/instance,
seed/ordinal/name, intended/seed/realized fingerprint or named intent,
admission status/reason, available drift/hard evidence, diagnostic energy,
official score/feasible, and winner/status. The manifest hashes exactly the
seven support artifacts by fixed relative basename, excluding itself. It stores
`self_sha256 = SHA256(canonical g0_manifest JSON with the self_sha256 field
omitted)`; an external freeze may later hash final manifest bytes. No absolute
paths and no `topology_data.write_sha256_manifest` are allowed.
Opaque public admission failures may be `admission_failed`; zero drift/hard
claims apply only to admitted/scored/winner records and coverage. Every
eligible held-out case contributes exactly one official winner, including the
baseline when there is no improvement.

The source boundary is the exact canonical root, checked before any checkpoint
load; sorted approved worker/layout paths are hashed by exact bytes and loaded
via the same `BytesIO` with `torch.load(weights_only=True)`, one sorted shard at
a time; process all rows, append canonical partition JSONL incrementally,
`fsync`, and evict tensors before the next shard. Retain only index metadata,
digests, and gate aggregates; no whole-corpus save/load API is permitted.
Production rejects symlinks/noncanonical roots and records an
environment identity without claiming cross-hardware bit identity. Output is
staged in a sibling directory, fsynced, and atomically renamed into a previously
absent destination; `index-out` must be exactly `out-dir/training_index.json`,
and a failed write cannot produce a completed manifest. Completion requires
manifest `status=complete` and a terminal G0 state. Artifacts contain no
timestamps or absolute paths. The production checkpoint expected SHA is
`508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659`; verify
exact bytes before deserialization and bind canonical config, key/shape/dtype,
and selected EMA tensor identity. Missing EMA, mismatch, Shapely dependency,
or scorer source SHA/version mismatch fails closed. Fixtures may use only a
test-injected expected hash helper.

Before model or G0 scoring, freeze scorer identity: `scripts/iccad2026_evaluate.py`
SHA256 is `7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1`,
with the declared scorer schema/version identifier, `SHAPELY_AVAILABLE=true`,
and Shapely `2.0.5`. A legitimate source change requires a written
specification amendment and new freeze; it is never auto-accepted.

Persist a versioned manifest containing baseline commit/checkpoint identities,
canonical model/EMA/keyset/config hashes, Flow identity, portfolio/environment
hash, instance split hash, the predeclared G1 checkpoint/full100 schedule,
deterministic seeds, proposal caps, accepted/rejected counts and reasons, exact
scorer version, label schema version, normalization scales, training config,
held-out metrics, causal-smoke receipt, and the concealed paired full100
results. Emit per-case proposal coverage, drift, overlap, hard-error,
topology-satisfaction, energy, runtime, candidate-source counts, and diversity
metrics.

Environment identity has two distinct hashes. Each arm retains its literal
`full_execution_env_sha256` for audit. For matched comparison, compute
`matched_solver_env_sha256` after replacing `DIRECT_CKPT` by one fixed token
and excluding only the runner-owned receipt destination and opaque arm ID.
Those three fields are the closed allowlist of per-arm differences; the Direct
checkpoint identity is separately bound to the sealed mapping/freeze manifest.
Any other environment difference, especially any additional solver namespace
key or value, invalidates the pair.

Every artifact must be regenerable from the manifest without validation data.
Retain teacher proposals, sparse labels, rejection logs, checkpoints, freeze
manifests, environment receipts, and gate reports as immutable evidence. Before
either blind arm, start from a scrubbed solver environment and admit only the
declared 3D/3F contract; inherited `PARTNER_`, `DIRECT_`, `FLOW_`, or `VKILL`
state is forbidden. False values for flags consumed through raw environment
truthiness must be empty/unset, never the non-empty string `"0"`; in
particular, `DIRECT_OFF` must remain false so the Direct arm is actually
loaded. Rollback restores both the frozen production Direct
checkpoint and the pre-amendment portfolio; production has no new online model
state to migrate. No package review follows G1.

Normal receipts are written atomically at interpreter exit, after the evaluator
has finished all timed cases; invalid receipts are written immediately before
process termination. The runner owns the receipt path and requires a schema,
arm ID, `complete=true` marker, exactly 100 ordered case-status entries, no
fallback/exception events, and matching output/log/environment hashes before
sealing an arm. A mutable selection artifact is never trusted by path: the
held-out audit and causal smoke record the candidate's canonical identity, and
the freeze gate rejects unless both identities equal the frozen candidate.

## File ownership and rejected alternatives

Later Track-B work may create only:

- `partner/icdc/topology_prior.py`
- `partner/icdc/topology_data.py`
- `partner/icdc/train_topology_prior.py`
- `tests/test_icdc_topology_prior.py`
- Track-B scripts and their generated artifacts.

Existing TFDL, energy, and engine modules are dependencies; any later narrow
change to them requires review. No Track-A file may be edited, and no optimizer
integration may occur until Sol serializes it.

Repaired-coordinate fine-tuning is rejected because it repeats the closed,
drift-prone coordinate target failure. A dense straight-through pairwise head
is rejected because it adds production shape/runtime risk, learns non-critical
and contradictory pair labels, and still does not provide exact pin/grouping
legality. The sparse, exact-teacher, same-shape prior is the only accepted
Track-B direction.

## Completion condition

The currently approved Track-B phase is complete only as
`HIGH_TAIL_CAUSAL_PROOF` after G0 and every G1 prerequisite pass, the one sealed
matched pair clears both score gates plus runtime/legality, and the immutable
record transitions to `STOP_REQUIRES_SEPARATE_APPROVAL`. This does not complete
the overall goal. Exact `1.00`, `<=0.300` s/case, `100/100`, zero errors, and
the final confirmation remain pending a separately approved G2 policy. Until
then, report the current baseline and gate status, never Track-B or submission
completion.
