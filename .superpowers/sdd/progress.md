# Subagent-Driven Development Progress

Branch: `codex/5.6-sol-partner-candidate-research`
Plan sequence:

1. `docs/superpowers/plans/2026-07-23-partner-candidate-source-foundation.md`
2. `docs/superpowers/plans/2026-07-23-partner-retrieval-gate.md`
3. `docs/superpowers/plans/2026-07-23-partner-flow-matching-gate.md`

Baseline: `uv run pytest` — 382 passed, 1 skipped (2026-07-23).

Foundation Task 1: complete (commits `4598a85..a4bff1e`, Terra task review clean; focused 3 passed, full 385 passed/1 skipped; 46 warnings match baseline provenance).

Foundation Task 2: complete (commits `a4bff1e..e6b7425`, Terra task review clean; focused 6 passed, full 388 passed/1 skipped; 46 warnings match baseline provenance).

Foundation Task 3: complete (commits `e6b7425..920f3e8`, Terra task review clean; focused 7 passed, full 389 passed/1 skipped; src and partner validation passed).

Foundation completion gate: passed on T14 config. Full-100 artifact `artifacts/partner_eval/foundation_shared_prescreen_full100.json`: no-runtime 1.1138297 vs 1.1162007 baseline (-0.0023710), total 1.9789078, 100/100 feasible, avg runtime 4.9331s vs 4.8682s (+1.33%), max 23.8134s vs 23.6964s (+0.49%); runtime deltas accepted as run-to-run noise. No src solver diff.

Retrieval Task 1: complete (commits `920f3e8..637897d` plus review fix `c1569ca`; Terra re-review approved; focused 3 passed, full 392 passed/1 skipped; float32 public-record contract enforced).

Retrieval Task 2: complete (commit `ea78239` plus numerical review fix `c4b6498`; Terra re-review approved under accidental/misconfiguration leakage threat model; focused 8 passed, full 400 passed/1 skipped; train-only save/load and finite hash-verified query contract enforced).

Retrieval Task 3: complete (commit `74e8cec` plus hard-match review fix `bb82672`; Terra re-review approved; focused 19 passed, full 419 passed/1 skipped; target-to-source Hungarian matching, dynamic hard-incompatibility sentinel, D4 transfer, anchors, and extreme/tie cases verified).

Retrieval Task 4 implementation/review: complete (commit `c18c2da`, publication fixes `4668665`/`917d50d`, and evaluator-anchor fix `856da31`; Terra re-reviews approved; focused 12 passed, full 431 passed/1 skipped; train-only bounded builder, NFS-safe no-replace atomic publication, and evaluator-aligned read-only probe verified).

Retrieval Task 4 pilot evidence: `artifacts/retrieval/pilot` contains 100 shards for N=21..120, exactly 2 seeded train records each, hash/load verified; first cold scan exposed NFS `renameat2` EINVAL and failed closed, reviewed symlink fallback then published successfully. `pilot_probe100.json`: 0/100 compatible under strict four-role matching; query p50 0.067 ms, matching p50 217.86 ms/p90 663.20 ms/max 1051.62 ms. `pilot64` (64/N, 6400 records) also had 0/100 exact four-role signatures; fixed+preplaced signatures existed for 76/100 (median rank 7), fixed-only for 100/100. Official semantics and independent Terra analysis reject role identity as a hard invariant; Retrieval Task 3b all-soft correction in progress before Task 5.

Retrieval Task 3b: complete (commit `78c873e` plus review test fix `a203016`; Terra re-review approved; focused 30 passed, full 430 passed/1 skipped). Same-N/shape/finite are the only hard matching preconditions; all 16 node features are soft and target hard anchors are post-projected.

Retrieval all-soft performance evidence: `artifacts/retrieval/pilot64_all_soft_probe100.json` matched 100/100 at top-2 with costs max 0.606 (<2.0 threshold). Query p95 0.103 ms and transfer p95 0.211 ms, but scalar-Python Hungarian across 2 sources x 4 D4 transforms costs p50 231.61 ms/p95 889.94 ms/max 1065.94 ms (N=101..120 p50 809.55 ms). Confidence diagnostic was negative in 59 cases. Task 3c exact NumPy-vectorized assignment and bounded confidence correction in progress; Task 5 remains blocked on runtime re-probe.

Retrieval Task 3c: complete (commit `7226c0b`; Terra review approved; focused 27 passed, full 439 passed/1 skipped; exact vectorized potentials solver and bounded diagnostic confidence).

Retrieval optimized performance gate: passed with `artifacts/retrieval/pilot64_all_soft_vector_probe100.json`. Outputs/source/transform/cost/raw proxies exactly equal the scalar probe, coverage 100/100. Matching p50 91.62 ms/p95 223.18 ms/max 277.70 ms; N=101..120 p50 211.20 ms/p95 255.99 ms/max 277.70 ms. Total matching 32.33 -> 10.31 s (3.13x); query p95 0.093 ms; transfer p95 0.182 ms; confidence finite 0.036..0.148. Task 5 may proceed using 2 retrieval slots/top-2 for the first R4 gate.

Retrieval Task 5: complete (commit `8a72481` plus capacity/path fix `4ce8b96`; independent Terra re-review clean; focused 60 passed, full 449 passed/1 skipped, evaluator validation passed). Retrieval is opt-in and shares the existing Direct candidate budget; the first R4 gate is defensively capped at 2 retrieval slots, preserves Direct backfill for K=1/2/5, and uses an absolute train-only `pilot64` index path that remains valid after the evaluator changes directory. The locally excluded implementation report is not tracked. R4 paired full-100 control/treatment is next; CP-SAT remains blocked unless R4 passes.

Retrieval R4 forced-quota pair 1: failed. Same step-1,139,000 checkpoint and fixed deadline/capacity produced control `total=2.0003461328`, `no-runtime=1.1257981382` versus two-slot treatment `total=2.0131576739`, `no-runtime=1.1328798292`; deltas are `+0.0128115411` and `+0.0070816910` (lower is better). Both are 100/100 feasible and runtime summaries are effectively identical. No-runtime case outcomes are 34 wins / 44 losses / 22 ties; the weighted regression is dominated by cases 93, 88, 98, and 95, while the best gain at case 96 is insufficient. Existing selection reserves two Retrieval slots even when shared rank puts them below Direct, and evaluator JSON lacks raw/prescreen/refined source provenance. Independent Terra verdict: do not run a blind repeat and do not start CP-SAT. First run a small stratified, opt-in provenance trace to decide whether an unconstrained max-two selector is feature-identifiable; otherwise record final R4 rejection.

Retrieval provenance trace: complete (commits `fd16bc5`, `56b489a`, `2f6f3b1`, `9a69398`; independent Terra re-reviews clean; focused 14 passed, full 463 passed/1 skipped). One-case real CUDA smoke passed with production K=8. The 1-second stratified trace showed maximum quota selecting only 3 Retrieval candidates across cases 93/96; the 4-second tail trace treated 99 union candidates with one failure and produced zero Retrieval refined winners across six cases. Maximum quota was better than forced on cases 79/98 and equal on the other four solely because it preserved additional Direct candidates. Top-ranked Retrieval false positives at cases 93/96 refined to about 2.79--3.00 versus Direct 1.02--1.13; case 99's best distance/confidence also did not predict utility. Final independent verdict: no feature-identifiable selector, no maximum-policy full-100, R4 fail, and CP-SAT golden-teacher/medoid/local-window path remains blocked. Formal report: `docs/experiments/2026-07-23-retrieval-r4-gate.md`.

---

Branch: `5.6-sol-reduce-time`
Plan: `docs/superpowers/plans/2026-08-12-fast-grouping-bridge.md`
Approved spec: `docs/superpowers/specs/2026-08-11-fast-grouping-bridge-design.md`
Preflight: Terra xhigh approved implementation; explicit commit invariant is grouping V down, total V down, and `_Ctx` score down. Package remains untouched through G0/G1.
Baseline: `uv run pytest` passed with 1253 passed, 5 skipped, 0 failed (985.93s); report `.superpowers/sdd/g01-baseline-report.md`.
Task 1: complete (commits `ea9f72d..d77cf81`, Terra re-review clean after two test-fix rounds; Sol focused verification 10 passed).
Task 2: complete (commits `d77cf81..11a9e1d`; Terra re-review clean after three focused fix rounds; Sol focused verification 27 passed; integration full suite 1268 passed, 5 skipped).
Task 3: complete (G0 PASS; Sol replay and fresh Terra xhigh evidence review clean: 100/100 feasible, 7 accepts, no-runtime `1.152956051348 -> 1.148936876390`, gain `0.004019174958`, mean wrapper `0.002013s`).
Task 4: accepted under the user-approved 2026-08-12 amendment. Causal call
means: 0.001732350000s/0.001988140000s/0.001787850000s; aggregate
0.001836113333s. Projected averages: 0.2912198360s/0.2901608443s/0.2899212596s;
margins: 0.0087801640s/0.0098391557s/0.0100787404s. Historical debug-inclusive
rule FAIL (pair3 0.3000952487s; mean delta +0.0068471677s). Quality, legality,
and package gates remain binding.
Task 5: unblocked/pending; fresh package gate still requires actual,
non-projected average runtime <=0.300s and 100/100 feasible.

---

Branch: `5.6-sol-reduce-time`
Approved specs:

1. `docs/superpowers/specs/2026-08-13-lp-free-changed-contact-dag-bridge-design.md`
2. `docs/superpowers/specs/2026-08-13-tfdl-data-free-topology-prior-design.md`

Track A plan: `docs/superpowers/plans/2026-08-13-lp-free-changed-contact-dag-bridge.md` (`9a10a14`, independent Terra xhigh plan review approved).
Track A Task 1: complete (commits `9a10a14..00ce698`; Luna low TDD plus
three fail-closed review-fix rounds; Terra xhigh final spec/task review clean;
focused axis 35 passed, full bridge 45 passed, graph refresh succeeded).

Track B preflight: paused pending a user-approved Direct/Flow production
portfolio amendment. The active wrapper requests `PARTNER_NREF=6` and
`PARTNER_FLOW_SLOTS=10`; `allocate_quotas` therefore supplies six Flow and zero
Direct candidates, so a Direct-checkpoint-only topology prior cannot affect the
authoritative `1.1437448258795715` baseline. No Track B blind evaluation is
authorized until this contract is resolved and frozen.

No package review/work before exact `1.00 @ <=0.300s`, per user instruction.

Track B portfolio amendment: approved — production total remains six candidates,
fixed at 3 Direct DPM++/2 plus 3 Flow Euler/8.
Track B Task 1: complete (commits `4bf1b8c..087f6f6`; independent Terra
re-review clean; receipt-bound seven-tensor shard reconstruction and sanitized
corpus contracts accepted).
Track B Task 2: complete (commits `e16f84a..3f97151`; independent Terra
re-review clean; sparse topology labels/losses accepted).
Track B Task 3: complete (commits `2cda0e5..358f2b6`; independent Terra
re-review clean; bounded topology proposals and exact admission accepted;
focused suite and randomized legality audit passed).
Track B Task 4: in progress. Helper slice complete (commits
`526319e..180f161`; fail-closed correction rounds `d527e02`, `d8d3bd0`,
`180f161`; independent Terra re-review approved; Sol focused verification
335 passed plus CLI help/pycompile/diff-check). Source/evidence boundary A is
complete (RED commits `4647b07`, `19e9750`, `8b7c1ef`; production commits
`91306af`, `b03916b`; two Terra xhigh review/fix rounds clean; Sol verification
112 Task-4A tests and 459 focused-file tests passed). It binds canonical ASCII
shard paths, validates padded source/outcome evidence fail-closed, uses bounded
per-tensor finite checks, and rejects noncanonical JSON. Streaming publication
boundary B, real teacher execution, and G0 remain pending. No package review or
blind full-100 is authorized before the exact score/runtime goal.

Track B Task 4 P1-A: complete (approved RED range `9e82f3c..e701fc6` plus
test-oracle/security corrections through `7f26062`; GREEN source commits
`0a1924c`, `dfdebc2`, `36c1a3d`; independent Terra xhigh final re-review clean).
The verified in-memory EMA is materialized lazily without checkpoint reopen or
global-RNG drift, builds one actual-config condition, samples one Direct
DPM++/2 trajectory, validates every tensor seam, and decodes once to CPU
float64 before the explicit candidate-lifecycle placeholder. Sol verification:
Task4 78 passed, teacher 277 passed, CLI help/pycompile/diff-check passed.
Track B Task 4 P1-B: complete (final RED commits `1d223eb`, `9c1f50f`;
GREEN `a6b4bf9`; production-gap fix `8f805c9`; independent Terra xhigh final
re-review approved with no Critical/Important). Every bounded proposal is
validated before sinks and follows admission -> hard -> intent -> official
score -> diagnostic energy; only official cost selects the deterministic
winner. The frozen CPU-f64 energy batch, canonical empty relations,
`hpwl_ref == 0`, total/per-kind caps, semantic proposal names, base-unavailable
accounting, and no-alias behavior are covered. Sol focused verification:
P1-B 99 passed; Terra review selection 101 passed; implementer full focused
file 695 passed. P1-C evidence integration and G0 remain pending.

Track B Task 4 P1-C runtime/compiler/evidence: complete through source commit
`06a5d5e`; focused P1-C 57 passed, evidence/validator 135 passed, P1-A/B 99
passed, and B2 16 passed before the spool slice. The single-read SQLite case
spool / streamed-index RED contract is complete through tests-only commit
`5e83034`; independent Terra xhigh re-review approved with no
Critical/Important findings. Sol replayed six key REDs and all six failed at
the intended missing production boundaries (no private spool, double source
read, retained index rows, delayed index emission, Python population roster,
and non-SQLite incomplete finish). Source GREEN is next.

Continuation GREEN: trusted shard summaries are persisted in the lease-owned
SQLite spool and joined during ordered replay; no Python summary sequence/map
remains. `tests/test_icdc_topology_prior.py -q`: 786 passed, 14 warnings.
Py_compile, CLI help, and `git diff --check` passed. No full100/package review.

Task 4 cumulative source hardening complete: independent builder provenance,
finite population aggregates, inode-safe SQLite cleanup, descriptor-bound
staging cleanup, and foreign replacement preservation. Exact cumulative RED
selection: 15 passed. `tests/test_icdc_topology_prior.py -q`: 801 passed, 14
warnings. `uv run python -m py_compile scripts/probes/icdc_topology_teacher.py`,
CLI `--help`, and `git diff --check` passed. No package/full100 run.
