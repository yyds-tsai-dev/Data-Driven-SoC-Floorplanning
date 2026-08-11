# 2026-08-11 Column split/merge LNS G0: NO-GO

## Question

Can a one-step multi-unit column split/merge escape partitions that the
production column SA cannot reach economically through single-unit relocates?
The probe preserves the production representation and evaluates every
contiguous split and adjacent merge with `_ColumnOptimizer._evaluate`.

The default-off implementation and tests are archived on
`feat/column-split-merge-g0` at commit `cde1204`; it was not merged into the
production solver.

## Protocol and gate

- Production 0.3 s configuration and shipped direct/flow checkpoints.
- Full 100-case evaluator with two reversed pairs: OFF→ON, then ON→OFF.
- Required full-score-equivalent weighted gain from `n >= 100` of at least
  0.005, at least three strict tail wins, 100/100 hard feasibility, and a
  practical enumeration time before proceeding to stochastic G1.

## Results

| Pair | OFF noRT | ON noRT | ON−OFF | OFF avg runtime | ON avg runtime | Tail weighted contribution | Tail W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1, OFF→ON | 1.154704780 | 1.150123840 | -0.004580940 | 0.292306 s | 0.298317 s | -0.002424510 | 6/2/13 |
| 2, ON→OFF | 1.150745238 | 1.151197830 | +0.000452593 | 0.296076 s | 0.296511 s | +0.003679810 | 9/3/9 |
| Paired mean | — | — | **-0.002064174** | — | **+0.003223 s delta** | **+0.000627650** | **6/3/12 on mean case deltas** |

Both arms were 100/100 hard feasible. A debug run on test 99 enumerated
96–124 candidates per worker in roughly 18–37 ms. Nine of 18 workers improved
the internal proxy, but the final official cost improved by only about 0.0001.

The pair-1 apparent win was dominated by test 86 (`n=107`, official cost
delta -0.22537, weighted -0.00610) and did not repeat. Pair 2 instead regressed
test 83 by +0.11847 and test 89 by +0.08132. The mean tail contribution is
positive even though the mean total score delta is slightly negative, so the
headline improvement comes from low-weight cases and run-to-run deadline
variation rather than the target tail.

## Verdict

**NO-GO. Do not implement or ship G1.**

The structural move can improve the solver's internal proxy, but those gains
do not transfer consistently to evaluator no-runtime score. Revisit only if a
candidate scorer with independently demonstrated official-score alignment is
available, or if split/merge becomes part of a construction method with a
legal layout-quality guarantee.
