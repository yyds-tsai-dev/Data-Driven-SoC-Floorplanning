# Post-bridge residual and fast-boundary audit

Date: 2026-08-11

## Baseline under study

This audit is a saved-layout replay, not a promoted solver result.  It starts
from the verified 0811d tag-compress full100 output and substitutes the seven
officially improving candidates found by the 20ms fast grouping-bridge probe.

- 0811d no-runtime score: `1.152956051348`
- grouping-replay score: `1.148936876390`
- grouping-replay gain: `0.004019174958`
- projected average runtime: `0.291887s`

## Residual decomposition

Re-evaluating every substituted layout with the official evaluator gives:

| Counterfactual | Weighted score | Headroom from 1.148936876 |
| --- | ---: | ---: |
| Actual grouping replay | 1.148936876 | -- |
| All soft violations removed, quality unchanged | 1.092540194 | 0.056396682 |
| Perfect quality, violations unchanged | 1.050799692 | 0.098137184 |
| Perfect quality and no violations | 1.000000000 | 0.148936876 |

Residual raw soft counts are 126 boundary, 37 grouping, and 1 MIB.  Both
quality and violations remain score-bearing; eliminating either dimension
alone cannot reach 1.00.

The largest excess-score contributors after the grouping replay are:

| Test | n | Cost | Excess contribution | Main residual |
| ---: | ---: | ---: | ---: | --- |
| 99 | 120 | 1.151982 | 0.012155 | boundary 2, grouping 1, quality 1.0533 |
| 89 | 110 | 1.265454 | 0.009226 | boundary 5, quality 1.0479 |
| 97 | 118 | 1.114437 | 0.007747 | violation-free quality 1.1144 |
| 98 | 119 | 1.098888 | 0.007276 | boundary 1, grouping 1 |
| 96 | 117 | 1.112049 | 0.006979 | violation-free quality 1.1120 |
| 86 | 107 | 1.240667 | 0.006515 | grouping 1, quality 1.1964 |

## Boundary-only micro-pass

The existing pure-NumPy `_fix_boundary` primitive was replayed after the fast
grouping candidates, using the same scorer-reuse assumption and a 20ms soft
deadline.

- boundary-violating cases inspected: 63;
- accepted cases: 2 (`n=29`, `n=79`);
- total boundary computation: `0.05797s`;
- additional weighted gain: `0.000175689`;
- combined grouping+boundary replay score: `1.148761187`;
- boundary pass by itself projects average runtime to `0.290850s`; including
  the grouping pass projects about `0.29247s`.

**Decision: NO-GO as a standalone promotion target.**  The primitive does not
move the high-weight boundary residuals at n=110 or n=120, and its weighted
gain is an order of magnitude below the grouping G0 threshold.

## What the LP bridge is doing

The high-weight grouping cases repaired by the analytical LP are not mostly
rigid component translations:

| n | Bridge | Blocks moved | Dominant movement |
| ---: | --- | ---: | --- |
| 101 | 24--76, y contact | 77/101 | 77 y coordinates |
| 111 | 30--110, x contact | 34/111 | 32 x coordinates |
| 120 | 19--33, y contact | 97/120 | 88 x and 45 y coordinates |

All keep dimensions fixed.  The n=120 repair also reduces both HPWL and bbox
area.  This explains why the local component-shove primitive cannot reproduce
them: the contact is enabled by a global separation-topology projection that
redistributes slack across long chains.

The appropriate successor is therefore a bounded, LP-free DAG projection for
a small number of grouping contacts, not further tuning of the local boundary
or component-translation enumerators.  It should be considered only after the
fast grouping bridge completes its own G1 gate.

## Session evidence

- `/tmp/current_0811d_fast_grouping_all100.json`
- `/tmp/current_0811d_fast_grouping_residual.json`
- `/tmp/current_0811d_fast_group_boundary_all100.json`
- `/tmp/current_0811d_gr_prod_n100.json`
- `/tmp/tag_package_full100.json`
