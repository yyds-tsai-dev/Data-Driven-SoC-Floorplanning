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
4. Evaluate admitted proposals with exact TFDL (no shelf fallback), the
   evaluator-faithful energy shortlist, and exact boundary/grouping geometry.
5. Convert only sparse critical constraints into labels.
6. Distill those labels into a `DirectDenoiser` checkpoint with the SAME SHAPE.

Production changes the checkpoint only: same configuration and tensor shapes,
same two-step sampler, same candidate count, and no online TFDL, proposal bank,
dense head, or extra graph pass.

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

Run the production-aligned differentiable two-step sampler before computing
the student losses. The student objective is a weighted sum of:

- differentiable normalized hinge separation loss
  `mean(max(0, (margin - signed_separation) / scale))` over labeled edges;
- contact loss combining normalized equality error with a hinge enforcing the
  perpendicular positive-overlap margin;
- pin/topology consistency loss for pin-support paths and required ordering;
- a detached teacher-quality/ranking weight for each sparse-label record; and
- base-checkpoint and EMA-anchor penalties to retain the Direct solution.

Evaluator-faithful energy is used offline to select proposals and derive that
detached record weight. It is not evaluated directly on the student's raw,
possibly overlapping coordinates: doing so would restore the degenerate blob
minimum that killed pre-legalization energy training. The sparse separation,
contact, and pin-margin losses are the only topology gradients supplied to the
same-shape student.

Monitor constraint satisfaction, drift, hard legality, energy/ranking gain,
collapse (loss diversity and output variance), and proposal/student diversity.
Reject a run that wins by collapsing all outputs to one topology.

## Data and validation hygiene

Training and held-out data come from a non-validation instance pool. Validation
cases are never used for proposal tuning, repeated feedback, threshold choice,
or checkpoint selection. The checkpoint is selected solely by held-out gates.
Before the first full100 run, freeze a complete stage schedule: the held-out
selection rule, one checkpoint per stage, thresholds, stop rules, and the final
confirmation rule. Each predeclared stage may receive at most one blind paired
full100 injection after its checkpoint is frozen. A stage result may only
promote the already-declared next stage or terminate the track; it may not tune
training, thresholds, proposals, losses, seeds, or checkpoint choice. All
later checkpoints remain selected solely from non-validation held-out evidence.
Reserve a separate final paired full100 confirmation for the final checkpoint,
which must likewise be frozen without using earlier full100 outcomes as a
selection signal.

The exact current weighted decomposition is:

- `n >= 100` weight: `0.8264247038229129`;
- current-band average: `1.1011247299384228`;
- outside contribution: `0.2337481470681254`.

`q_band = 1.0829742560590576` corresponds to full100 minus `0.015` under
100%-replacement. Even perfecting `n >= 100` while leaving all other cases
unchanged yields only `1.0601728508910383`; therefore the track must expand
after proof to `60–99` and ultimately all cases. `n >= 100` alone cannot reach
`1.00`.

## Gates and stop conditions

### G0 — teacher oracle

On held-out, non-validation instances, require 100% per-case proposal
coverage; every admitted proposal must be exactly hard-legal, with zero drift,
zero shelf fallback, and exact scoring. Target `q_band <= 1.075` to leave
student margin. The hard minimum is
`q_band <= 1.0829742560590576`; stop if it is missed.

### G1 — same-shape student

Require held-out zero drift and hard legality under an exact TFDL audit. The
one-time blind full100 paired injection must remain `100/100` feasible and
achieve no-runtime `<=1.1287448258795715` (at least `0.015` gain) with runtime
`<=0.300` s/case. Verify that model configuration, sampler method and steps,
candidate count, state-dict key set, tensor shapes, and dtypes match the frozen
production contract. G1 is not the final goal.

### G2 — expansion

Expand weighted coverage to `60–99` and then all block counts using sampling
aligned to `exp(n/12)`. The implementation plan must predeclare the checkpoints
and held-out counterparts for milestones `<=1.10`, `<=1.05`, and ultimately
exactly verified `1.00`. Each stage gets at most its scheduled blind paired
full100 promotion-or-termination run, never an adaptation loop. The final
checkpoint gets the reserved confirmation for `1.00`, `<=0.300 s`, `100/100`
feasibility, and zero errors. Never claim the goal at G1.

Kill the track if the oracle is above `1.5` with no trend, proposal coverage or
pin support fails, the student cannot retain the held-out benefit, or runtime
changes. Track A may spend at most `0.75` ms, leaving approximately `0.434` ms;
Track B production remains replace-only, with actual full100 runtime binding.

## Artifacts, reproducibility, and observability

Persist a versioned manifest containing baseline commit/checkpoint hash,
instance split hash, the predeclared stage/checkpoint/full100 schedule,
deterministic seeds, proposal caps, accepted/rejected counts and reasons, exact
scorer version, label schema version, normalization scales, training config,
held-out metrics, checkpoint tensor-key hash, and paired full100 results. Emit
per-case proposal coverage, drift, overlap, hard-error, topology-satisfaction,
energy, runtime, and diversity metrics.

Every artifact must be regenerable from the manifest without validation data.
Retain teacher proposals, sparse labels, rejection logs, checkpoints, and gate
reports as immutable evidence. Rollback is a single checkpoint replacement to
the frozen Direct baseline; production has no new runtime state to migrate.

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

Track B is complete only after G0, G1, and G2 pass with reproducible manifests,
held-out evidence, one blind paired full100 verification, exact `1.00` no-runtime
and `<=0.300` s/case, `100/100` feasibility, zero hard errors, and a tested
one-checkpoint rollback. Until then, report the current baseline and gate
status, not completion.
