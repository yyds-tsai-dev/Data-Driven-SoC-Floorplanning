# Selective joint-repair G0

Date: 2026-08-11

## Question

Can the existing analytical A/B/C repair pipeline be applied only to the
largest, highest-weight cases while keeping the final 0811d submission at an
average runtime of at most 0.3 seconds?

The probe used the freshly verified output of
`submission/cadc1013_0811d_tagcompress.tar.gz` (MD5
`aab0c6553e387275301674095a0cd709`) as the input layout.  Every repaired
candidate was accepted only when the official evaluator reported a strict
`cost_no_runtime` improvement and full hard feasibility.

## Result

**NO-GO.**  Applying the repair to all 21 cases with `n >= 100` changes the
official weighted no-runtime score from `1.152956051348` to `1.142096359`, a
gain of only `0.010859693`, while consuming `99.794` seconds of repair time.
That projects the average runtime from `0.290270` to `1.288210` seconds.

The block-count threshold frontier is also outside the runtime budget:

| Activation | Cases | Repair time | Projected average runtime | No-runtime gain |
| --- | ---: | ---: | ---: | ---: |
| `n >= 100` | 21 | 99.794 s | 1.288210 s | 0.010859693 |
| `n >= 110` | 11 | 60.414 s | 0.894408 s | 0.007915334 |
| `n >= 118` | 3 | 31.579 s | 0.606062 s | 0.003171181 |
| `n >= 120` | 1 | 23.216 s | 0.522427 s | 0.002910660 |

Even an oracle that selects the best single measured case cannot satisfy both
gates.  Case 89 (`n=110`) contributes the largest weighted improvement,
`0.003301697`, but requires `1.791` seconds of repair time and would raise the
average runtime to about `0.308177` seconds before paying for any generic
selection logic.  The full measured gain is concentrated in four accepted
wall/topology edits (`n=101, 110, 111, 120`); most other cases spend 1--9
seconds for negligible improvement.

## Decision

Do not integrate the SciPy LP A/B/C repair path into the evaluator-facing
solver.  It misses the stated G0 gain/runtime frontier by a large margin, and
a validation-case-specific gate would not be a legitimate architecture
improvement.  Keep the final 0811d package unchanged.

The useful architectural signal is narrower: topology-edit candidates can
occasionally remove a large soft-constraint penalty, but the current LP-based
candidate evaluation is too expensive.  A successor must predict or construct
the edit without running the full repair pipeline and must be judged as a new
mechanism, not as selective deployment of this one.

## Reproduction assets

- Repair implementation: `scratchpad/icdc/gr_prod.py`
- Prior full-run evidence: `scratchpad/icdc/gr_prod.json`
- Current-run raw evidence (session-local):
  `/tmp/current_0811d_gr_prod_n100.json`
- Current verified evaluator output (session-local):
  `/tmp/tag_package_full100.json`
