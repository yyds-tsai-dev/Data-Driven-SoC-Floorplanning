# Fast Final Grouping Bridge Design

Date: 2026-08-12 (amended by user approval)

## Goal

Add a default-off, deterministic final post-pass that reconnects disconnected
cluster components without running the SciPy LP repair pipeline.  The pass must
move the verified 0811d submission toward the final objective of
`total_score_no_runtime = 1.00` while keeping average runtime at or below
`0.3s` and preserving 100/100 hard feasibility.

The mechanism is not expected to close the entire remaining score gap by
itself.  It is promotable only if it improves the actual evaluator-facing
solver under the gates below; a successful promotion becomes the new baseline
for subsequent work toward 1.00.

## Evidence and Decision

The analytical A/B/C repair probe showed that its useful high-weight edits
were grouping bridges, but the full pipeline cost `99.794s` across the 21
`n >= 100` cases.  Replaying only the repository's existing pure-NumPy
component-reattachment primitive on the final 0811d layouts gives:

- baseline no-runtime score: `1.152956051348`;
- grouping-only score over all 100 cases: `1.148936876`;
- weighted gain: `0.004019175`;
- grouping-violating cases inspected: 37;
- accepted cases: 7;
- total bridge computation time: `0.161719s`;
- projected average runtime: `0.291887s` from `0.290270s`.

The accepted cases span block counts 28, 32, 35, 53, 78, 110, and 115, so
activation does not depend on validation case IDs.  The two high-weight
accepts contribute `0.003850328` of the measured gain.

Three designs were considered:

1. Reuse the existing grouping candidate generator and proxy acceptance at the
   final post-pass.  This retains the measured `0.004019` gain and is the
   recommended design.
2. Require HPWL and bbox area to be individually non-increasing in addition to
   lowering grouping violations.  This is structurally safer but rejects the
   useful n=110 candidate and leaves only about `0.002179` measured gain.
3. Replace the LP repair with a new DAG projection constructor.  This may
   recover more of the `0.01086` analytical headroom, but it is a separate,
   higher-risk successor rather than the smallest test of the observed bridge
   mechanism.

The chosen design is option 1, protected by default-off containment, exact
hard guards, a soft 20ms deadline, and evaluator-level paired promotion gates.

## Architecture

### Grouping-only public operation

Add `bridge_grouping_violations(opt, out, budget_s=0.02)` to
`src/solver/violation_killer.py`.  It is a narrow public wrapper around the
existing `_components`, `_Ctx`, `_fix_grouping`, and `_final_guards_ok`
machinery.  It does not invoke boundary repair, MIB repair, sliver repair,
LNS, stage-2 refinement, or any VKILL carve logic.

Its contract is:

- return the same `out` object when its shape is invalid, no grouping
  violation exists, the deadline is exhausted, no candidate strictly
  improves the proxy, a final guard fails, or an exception occurs;
- never move a preplaced block;
- never change fixed/preplaced dimensions;
- keep every soft-block area within the existing hard tolerance;
- return an overlap-free layout;
- require the evaluator-faithful total violation count to decrease and the
  existing `_Ctx` score to improve strictly;
- be deterministic and consume no random numbers.

The 20ms value is a soft deadline because the existing candidate builder is
not pre-emptible inside one candidate batch.  Saved-layout measurement showed
a maximum of about 20.2ms.  Online G1 evidence, rather than the nominal value,
decides whether runtime remains within the final 0.3s requirement.

### Final-pipeline integration

Add these default-off environment controls:

- `PARTNER_GROUP_BRIDGE=1` enables the post-pass;
- `PARTNER_GROUP_BRIDGE_BUDGET=0.02` controls its per-case soft deadline;
- `PARTNER_GROUP_BRIDGE_DEBUG=1` enables self-paired diagnostic output.

The existing `MyOptimizer._tag_compress` hook in
`src/solver/contest_optimizer.py` already constructs or reuses the final
`_ColumnOptimizer` scorer.  Extend that hook as follows:

1. If both `PARTNER_TAG_COMPRESS` and `PARTNER_GROUP_BRIDGE` are off, return
   the exact same input object before imports or scorer construction.
2. Construct or reuse one scorer when either flag is on.
3. Run tag compression first when enabled.
4. Run the grouping bridge on the resulting layout when enabled, using the
   same scorer.
5. Return the final result.  No downstream layout-changing stage follows this
   hook.

This keeps the two flags independently usable while avoiding the 50--300ms
`_ColumnOptimizer` construction cost measured in an intentionally cold replay.
The constructor warm-up condition must likewise cover either flag so that the
grouping wrapper and exact violation counter import outside evaluator-timed
`solve()`.

### Packaging boundary

Experiment first against `src/solver/contest_optimizer.py`; do not change the
verified submission package during G0/G1.  Only after promotion gates pass:

- synchronize `src/solver/contest_optimizer.py` to
  `submission/cadc1013/op_src.py`;
- synchronize `src/solver/violation_killer.py` to the packaged copy;
- add the promoted setdefault to `submission/cadc1013/op_wrapper.py`;
- rebuild the archive and verify the synchronized sources byte-for-byte;
- evaluate a freshly extracted archive rather than the working tree.

## Data Flow and Acceptance

For a final layout `out`:

1. The shared scorer supplies block kinds, areas, cluster groups, HPWL, and
   the evaluator-faithful soft-violation denominator.
2. `_grouping_count` is the cheap activation gate.  Zero returns immediately.
3. `_fix_grouping` enumerates exact component-contact placements, including
   rigid component translations and shove-made room where necessary.
4. Candidate overlap and proxy score are checked under the soft deadline.
5. `_final_guards_ok` rechecks shapes, preplaced equality, areas, and overlap.
6. Only a strict total-violation decrease plus strict proxy improvement is
   returned.  Otherwise the original object is returned unchanged.

The proxy is not treated as proof of official score improvement.  Promotion
depends on official evaluator results across complete online runs.  Debug mode
must report, on the same final input layout, elapsed milliseconds, grouping
count before/after, total violations before/after, HPWL before/after, bbox
before/after, and whether the candidate committed.

## Testing Strategy

Implementation follows red-green-refactor.  Tests extend the patterns in
`tests/test_partner_tag_compress.py` and add a focused grouping-bridge file.

Required unit and integration behaviours are:

1. both flags off returns the identical list object and performs no import or
   scorer construction;
2. bridge on with tag compression off still works independently;
3. a connected cluster and an instance without clusters return the identical
   input object;
4. a disconnected movable component is reattached and grouping violations
   decrease;
5. a disconnected component containing a preplaced block is handled only by
   moving an entirely movable counterpart, or is rejected;
6. preplaced coordinates, fixed/preplaced dimensions, soft areas, and
   overlap legality remain valid;
7. a proxy-non-improving candidate is rolled back;
8. an exception is contained and returns the input unchanged;
9. scorer construction occurs at most once when both final passes are on;
10. the soft deadline bounds candidate evaluation without introducing RNG;
11. partner and packaged sources are byte-identical after promotion.

Run the focused tests first, then the relevant partner suite, then
`uv run pytest`.  The known order-sensitive anytime-ladder test must be
rerun in isolation if it is the only full-suite failure; it may not be
silently called passing.

## Experimental Gates

### G0: saved-layout mechanism gate

Already observed; reproduce after implementation through the public wrapper:

- 100/100 hard feasible;
- official weighted no-runtime gain at least `0.003` against the exact saved
  0811d layouts;
- measured bridge computation adds at most `0.003s` to the 100-case average;
- no case ID or saved-layout data is referenced by production code.

Failure of any item closes the mechanism without online promotion work.

### G1: online paired evaluator gate (amended 2026-08-12)

Run three full100 ON/OFF pairs with reversed ordering across pairs.  Promotion
requires all of the following:

- every arm is 100/100 hard feasible;
- ON improves no-runtime score in all three pairs;
- mean paired no-runtime delta is at most `-0.002`;
- the aggregate mean of all 300 causal public bridge-call `ms` values is at
  most `0.003s` (per-run means are reported evidence, not separate gates);
- each pair's projected production average runtime is its official OFF
  average runtime plus that ON log's mean bridge-call seconds, and is at most
  `0.300s`;
- self-paired diagnostics show grouping decreases only on committed cases and
  no hard-guard failure is committed.

The original debug-inclusive end-to-end runtime rule remains recorded as a
historical diagnostic, not a production gate: pair 3 ON was `0.3000952487s`
and mean paired runtime delta was `+0.0068471677s` (therefore FAIL under that
original rule).  The debug path performs post-call grouping/exact-V/HPWL/bbox
rescoring that production does not enable, so binding G1 uses causal call
timing and OFF-plus-call projections.  From the six evidence logs: pair means
are `0.001732350000s`, `0.001988140000s`, and `0.001787850000s`; aggregate mean
is `0.001836113333s`.  Projected runtimes are respectively
`0.2912198360s` (margin `0.0087801640s`), `0.2901608443s` (margin
`0.0098391557s`), and `0.2899212596s` (margin `0.0100787404s`).

If variance prevents complete separation but the self-paired official replay
still exceeds the G0 threshold, retain the branch as evidence but do not alter
the verified submission package.

### Promotion and package gate

After G1 passes, enable the flag in the package wrapper, build a new dated
archive, extract it into a fresh directory, and run full100 evaluation.
Promotion requires:

- 100/100 hard feasible;
- `total_score_no_runtime` no worse than the best promoted online ON result;
- average runtime at or below `0.300s`;
- packaged source closure and checksum recorded in an experiment note.

The existing `cadc1013_0811d_tagcompress.tar.gz` remains the fallback until
all promotion evidence is complete.

## Error Handling and Rollback

The operation is best-effort and exception-contained.  Any unexpected input,
deadline exhaustion, missing dependency, or guard failure returns the exact
stage input.  The flag stays default off until G1 passes.  A failed G0 or G1
requires no production rollback because the verified package is not modified
during experimentation.  Debug-only timing/rescoring must not be substituted
for the causal-call rule or used to waive the final package gate.

## Self-Review

- No case IDs, validation layouts, or golden metrics enter production code.
- Activation is derived only from the current layout's actual cluster
  connectivity.
- The design reuses a scorer already paid for by the current final pipeline;
  projected runtime does not assume a free new scorer construction.
- The 20ms deadline is explicitly soft, and online runtime is authoritative.
- G1 records 100 causal call timings per ON log and binds to their aggregate
  300-call mean plus OFF-plus-call projections;
  the fresh-extracted package gate remains actual (non-projected) average
  runtime at most `0.300s` with 100/100 feasibility.
- Proxy acceptance is explicitly separated from official evaluator evidence.
- Success is not redefined as the local `0.004` gain: the full thread goal
  remains no-runtime 1.00 at average runtime at most 0.3s.
