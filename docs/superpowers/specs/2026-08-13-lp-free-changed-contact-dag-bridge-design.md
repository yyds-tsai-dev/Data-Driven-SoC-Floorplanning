# Track A: LP-Free Changed-Contact Separation-DAG Grouping Bridge

## Status and baseline

This is the approved Track A design specification. The authoritative development
baseline is `artifacts/partner_eval/gbridge_package_full100.json`:

- weighted no-runtime score: `1.1437448258795715`;
- average runtime: `0.29881621031556277` seconds;
- feasibility: `100/100`;
- hard violations/errors: zero;
- runtime headroom: `1.1837896844372198` ms/case.

No package review is performed until the final goal. Track A is a bounded,
default-off experiment and must preserve the baseline on failure.

## Problem and scope

The promoted local `bridge_grouping_violations` path is component-shove/local.
It cannot create a missing grouping contact. The former full-LP approach showed
gain but is too slow. Fixed-topology `coord_polish` and CSA are documented
NO-GO. Track A is distinct only because it creates one missing grouping contact
and changes the feasible constraint model. It does not change dimensions,
preplaced origins, package policy, or any Track B files, and it cannot reach
`1.00` alone.

The current local bridge remains the preceding path and fallback. Track A runs
only when that path leaves a disconnected grouping.

## Mechanism

1. Return immediately before graph construction when residual grouping is zero.
   Otherwise run after the current local grouping bridge has completed.
2. Discover connected components using evaluator-semantics grouping, not
   validation IDs, saved coordinates, or block-count rules.
3. Generate deterministic, bounded contact candidates between components. A
   candidate chooses a horizontal or vertical abutment and records the two
   components, contact axis, perpendicular axis, and required positive shared
   edge.
4. Reject candidates before projection if they are infeasible, cyclic, or pin
   conflicting. Candidate selection is deterministic and bounded.
5. Split the disconnected group into component variables. Represent the exact
   new contact without an LP: affine-eliminate/merge the two contact variables
   on the contact axis, and impose a bounded positive-overlap relation on the
   perpendicular axis. Corner touch is invalid; shared-edge overlap must be
   strictly positive.
6. Rebuild the reduced component separation DAG after the contact choice.
7. Apply global forward and reverse longest-path/slack projection on the
   reduced DAG, with dimensions fixed and preplaced origins fixed.
8. Recount evaluator-semantics grouping and all soft/hard constraints. On any
   failed invariant, return the input placement unchanged (identity on failure).

The implementation must not call `_fix_grouping` or fixed-topology
`coord_polish`; mechanism tests must prove this exclusion.

## Strict acceptance invariants

Accept a candidate only when all of the following hold:

- grouping violation count strictly decreases;
- exact total soft-V strictly decreases;
- the `_Ctx` proxy strictly improves;
- every final hard guard passes;
- no pre-existing soft relation is broken;
- exact positive shared-edge contact is present for the selected pair;
- no corner-only contact, pin conflict, cycle, or evaluator-semantic mismatch
  remains.

The current local bridge is always attempted first and remains the fallback.
Any failed invariant, exception, non-finite value, or recount disagreement
kills the candidate and preserves the prior placement.

## Runtime and instrumentation gates

The feature is controlled by default-off `PARTNER_GROUP_DAG_BRIDGE`. Add
deterministic caps for selected contacts, movable components, reduced edges,
and projection iterations. Warmed `time.perf_counter` instrumentation must
include activation, candidate build, projection, recount, and guard time.

The causal mean overhead gate is `<= 0.75 ms/case`; report median, p95, and
maximum as evidence. The binding runtime gate is actual full100 average runtime
`<= 0.300 s`. A zero-residual case must pay no graph-build cost.

## Validation gates

### G0: exact replay and gain hypothesis

Replay the exact current post-group-bridge evaluation. Require `100/100` hard
feasible cases and zero errors. The score-gain hypothesis is at least `0.0025`,
meaning score `<= 1.1412448258795715`; this is a hypothesis, not a forecast.
There is no per-case policy: every accepted candidate must satisfy the strict
invariants above. Kill the track on any failure.

### G1: paired full100 confirmation

Run three reversed paired full100 ON/OFF runs. Require all runs feasible, mean
ON-minus-OFF `<= -0.002`, and actual ON average runtime `<= 0.300 s`. If any
gate fails, retain the default-off setting and do not package the track.

## Required tests

Add focused tests covering:

- a synthetic coordinated multi-block redistribution where local shove cannot
  solve the grouping;
- a preplaced anchor whose origin must remain fixed;
- impossible contact and cycle rejection;
- positive-edge connectivity, with corner touch rejected;
- deterministic identity-on-failure and repeatability;
- mechanism proof of no `_fix_grouping` or fixed-topology `coord_polish` call;
- exact soft/hard acceptance and preservation of pre-existing soft relations.

Anticipated implementation ownership is `partner/violation_killer.py`, Track-A
probes, `tests/test_partner_group_bridge.py`, and
`tests/test_partner_tag_compress.py`. Any final hook in
`partner/contest_optimizer.py` is serialized and owned by Sol after independent
gates. No Track B files may be changed.

## Alternatives rejected

- Full LP: potentially effective, but runtime is unacceptable.
- Fixed-topology `coord_polish`: cannot represent the changed-contact feasible
  model; documented NO-GO.
- CSA: same fixed-topology limitation and documented NO-GO.
- Unbounded candidate search or arbitrary coordinate nudging: nondeterministic,
  difficult to audit, and incompatible with the runtime gate.
- Reusing validation IDs, saved coordinates, or block counts for components:
  these do not encode evaluator-semantics connectivity.

## Failure modes and observability

Expected failures include no valid contact, cycle creation, pin conflict,
insufficient positive overlap, hard-guard regression, soft-relation breakage,
recount disagreement, cap exhaustion, non-finite projection, and runtime
regression. Each rejected candidate must expose a stable reason code and case
identifier in diagnostics, together with candidate counts, selected contact,
component/edge/iteration counts, timing buckets, before/after grouping, exact
soft-V, `_Ctx` proxy, and hard-guard results. Full100 evidence must retain ON,
OFF, paired deltas, feasibility/errors, and runtime percentile summaries.

## Rollout and rollback

Land the implementation behind `PARTNER_GROUP_DAG_BRIDGE=0`. Run unit and
mechanism tests, then G0, then G1. Enable only after both gates pass and the
evidence is reviewed. Rollback is setting the flag to `0` (or removing the
final serialized hook); identity-on-failure independently protects each case.
The baseline artifact remains the comparison authority, and no package review
occurs before the final goal.
