# 2026-08-11 Column split/merge LNS G0: NO-GO

## Question

Can a one-step multi-unit column split/merge escape partitions that the
production column SA cannot reach economically through single-unit relocates?
The probe preserves the production representation and evaluates every
contiguous split and adjacent merge with `_ColumnOptimizer._evaluate`.

The historical implementation is preserved for reproducibility and remains
default-off behind `PARTNER_COL_LNS_ORACLE=1`; the submission path is unchanged
unless the experiment flag is set.

## Protocol

- Production 0.3 s configuration: frame scale 1.02, final seat, numba SA and
  refinement kernels, six direct refinements, and the shipped direct/flow
  checkpoints.
- Full 100-case evaluator, 100/100 hard feasibility required.
- Two paired evaluations with reversed arm order: OFF→ON, then ON→OFF. The
  second pair was added because the first result landed within 0.0005 of the
  preregistered gate while single-repetition noise is much larger.
- Primary gate: full-score-equivalent weighted gain from `n >= 100` of at
  least 0.005, plus at least three strict tail wins. The plan also required
  median oracle time at most 25 ms/worker and n=120 maximum at most
  60 ms/worker before proceeding to G1.

Raw evaluator records are `/tmp/col_lns_full_{control,on}.json` and
`/tmp/col_lns_full_{control,on}2.json` for this run.

## Results

| Pair | OFF noRT | ON noRT | ON−OFF | OFF avg runtime | ON avg runtime | Tail weighted contribution | Tail W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1, OFF→ON | 1.154704780 | 1.150123840 | -0.004580940 | 0.292306 s | 0.298317 s | -0.002424510 | 6/2/13 |
| 2, ON→OFF | 1.150745238 | 1.151197830 | +0.000452593 | 0.296076 s | 0.296511 s | +0.003679810 | 9/3/9 |
| Paired mean | — | — | **-0.002064174** | — | **+0.003223 s delta** | **+0.000627650** | **6/3/12 on mean case deltas** |

Both arms were 100/100 hard feasible in both pairs. A debug run on test 99
enumerated 96–124 candidates per worker in roughly 18–37 ms. Nine of 18
workers improved the internal proxy, but the final official cost improved by
only about 0.0001. Timing was therefore plausible, but the quality gate failed
before a full tail timing sweep was warranted.

The pair-1 apparent win was dominated by test 86 (`n=107`, official cost
delta -0.22537, weighted -0.00610) and did not repeat (pair-2 delta was
effectively zero). Pair 2 instead regressed test 83 by +0.11847 and test 89 by
+0.08132. The mean tail contribution is positive even though the mean total
score delta is slightly negative, so the headline improvement comes from
low-weight cases and run-to-run deadline variation rather than the target
tail.

## Verdict

**NO-GO. Do not implement G1 or enable the oracle hook in production.**

The structural move can improve the solver's internal proxy, but those gains
do not transfer consistently to evaluator no-runtime score. Adding it as a
stochastic SA move would spend the same 0.3 s budget on a neighborhood whose
measured tail expectation is non-positive. It would also remove the oracle's
only safeguard—exhaustive comparison—without fixing the proxy/official-score
mismatch.

Revisit only if either:

1. a candidate scorer with independently demonstrated official-score
   alignment becomes available; or
2. split/merge is embedded in a construction method that provides a legal
   layout-quality guarantee rather than selected by the current proxy.
