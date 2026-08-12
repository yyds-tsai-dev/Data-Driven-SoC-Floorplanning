# 2026-08-13 LP-free changed-contact DAG bridge: G0 HOLD

## Decision

Track A is **HOLD / KILLED** at G0.  The causal bridge evidence did not reach
the registered G0 score or latency contract, so there is no G1 run and no
promotion or package review.  This note records the gate result only; it does
not authorize a retry, integration, or package review.

## Gate evidence

The exact paired replay was unchanged by the Track-A switch:

- OFF score: `1.1437448258795715`;
- ON score: `1.1437448258795715`;
- changed cases: `0/100`;
- feasible: `100/100` for both arms;
- evaluator errors: `0`;
- residual grouping cases: `34`, with identity behavior on those cases.

The causal latency measurement was a mean of
`1.1877225316129625 ms/case`, exceeding the registered `0.75 ms/case` gate.
The latency gate therefore misses, independently of the zero score delta.
Because G0 is not passed, G1 was not run.

## Closure

The baseline remains the authoritative development result.  Track A remains
default-off/identity-preserving, with no package review and no final-submission
change.  Track B documentation may use the baseline score, but must not treat
this Track-A experiment as a passed causal control.
