# 2026-08-11 Preplaced-frame reinsertion G0: NO-GO

## Question

Can the final layout satisfy residual preplaced boundary tags by removing only
the movable rectangles beyond the tag-defined wall and reinserting them into
obstacle-edge slots inside that frame?  The acceptance test is independent of
unknown golden baselines: exact soft violations must strictly decrease while
raw HPWL and bbox area do not increase.

The production input was the freshly extracted final submission result
`/tmp/tag_package_full100.json`: `total_score_no_runtime=1.152956051`, average
runtime `0.290269905 s`, and 100/100 hard feasible.

## Implementation

- `src/solver/frame_reinsert.py` derives attainable wall targets exclusively from
  preplaced constraint geometry, enumerates deterministic obstacle-edge slots,
  and contains every failure by returning the original layout.
- `scripts/probes/frame_reinsert_g0.py` applies the pure pass to saved layouts,
  re-scores both arms with the official evaluator, and aggregates with the
  full-100 `exp(n/12)` denominator.
- `tests/test_partner_frame_reinsert.py` covers target derivation, locked
  outliers, cluster contact, dimension preservation, monotone acceptance,
  failure containment, and weighted aggregation.

No evaluator-facing hook or environment flag was added.

## Gate

- full-score-equivalent gain at least `0.005`;
- at least three strict wins among `n>=100`;
- 100/100 hard feasible;
- median internal time at most `10 ms` and n=120 at most `30 ms`.

## Results

| Arm | Accepted | Strict wins | Weighted before | Weighted after | Gain | Mean internal | Median internal | Max internal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conservative cluster skip | 0/100 | 0 | 1.152956051 | 1.152956051 | 0.000000000 | 17.55 ms | 0.058 ms | 1.191 s |
| Exact-V guarded cluster moves | 0/100 | 0 | 1.152956051 | 1.152956051 | **0.000000000** | 49.80 ms | 0.058 ms | **1.796 s** |

Both arms remained 100/100 hard feasible because no candidate was accepted.
Evidence is in `/tmp/frame_reinsert_g0_full.json` and
`/tmp/frame_reinsert_g0_full_cluster.json`.

The one allowed mechanism correction removed the conservative rule that
discarded a target whenever an outlier belonged to a movable cluster.  Safety
did not weaken: the final exact grouping count still had to fall as part of
the strict total-V gate.  This exposed more slot searches but produced no
candidate and made the expensive cases slower.

## Attribution

The attainable high-weight walls require simultaneous relocation of many
rectangles, not one exceptional block:

| Test | n | Target | Movable outliers |
|---|---:|---|---:|
| 88 | 109 | right / top | 10 / 6 |
| 89 | 110 | right / bottom / top | 7 / 6 / 8 |
| 94 | 115 | left | 13 |
| 99 | 120 | right / top | 16 / 6 |

At the current density, deterministic obstacle-edge insertion cannot place the
whole outlier set without moving retained incumbents.  This is the same
realizability bottleneck seen by the older DAG topology search (91% hard
reject), now reproduced specifically on the residual preplaced-frame cases.
The measured 0.76--1.80 s tail also rules out broadening the slot search under
the 0.3 s submission operating point.

## Decision

**NO-GO.  Do not implement G1 and do not merge this module into production.**

A successor must change the global construction topology before dense packing,
or provide a bounded simultaneous legalizer that can displace incumbents.  A
larger greedy/beam slot search is not a successor: it attacks the same failed
realizer and already exceeds the runtime budget without one accepted layout.
