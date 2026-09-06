# Constructive G0/G1 Gate Result

Date: 2026-08-11

Branch: `feat/constructive-g0-g1`

Decision: **NO-GO — do not integrate into the evaluator-facing path**

## Question

Can a hard-legal constructive decoder improve the current repaired-production
portfolio by combining:

- dynamic block ordering (`net_closure`, `constraint_first`, `large_first`,
  and `pin_gravity`),
- joint exact-area aspect decisions with MIB shape synchronization,
- consecutive edge-connected cluster construction, and
- sparse legal region/obstacle slots?

This is not a rerun of the retired `src/floorset_arch/constructive.py` beam.
The prototype uses raw `partner/` tensors, makes MIB/cluster decisions jointly,
and compares four fixed-capacity order arms. Production defaults were not
changed.

## Implementation

- `src/solver/constructive_g01.py`: isolated NumPy constructor and gate math.
- `scripts/probes/constructive_g01_probe.py`: official evaluator G0/G1 runner.
- `tests/test_partner_constructive_g01.py`: hard legality, MIB, cluster,
  policy-diversity, oracle-code, boundary, and aggregation tests.

The G0 oracle representation contains no source rectangles or per-block source
coordinates. It retains an order, exact-area log-aspects, 16x16 normalized
region cells, and quantized global frame aspect/utilization. Thus the reported
G0 score measures representation-to-realizer capacity rather than coordinate
passthrough. The repaired-golden layout is also scored directly as a loader and
evaluator calibration.

## Commands

```bash
uv run python scripts/probes/constructive_g01_probe.py \
  --mode g0 \
  --data-path /nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/FloorSet \
  --golden-layouts /nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/scratchpad/icdc/gr_layouts3.json \
  --out /tmp/constructive-g01-g0-full.json

uv run python scripts/probes/constructive_g01_probe.py \
  --mode g1 \
  --data-path /nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/FloorSet \
  --production-layouts /nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/scratchpad/icdc/gr_prod_layouts.json \
  --out /tmp/constructive-g01-g1-full.json
```

## G0: Representation Capacity

| Metric | Gate | Measured | Result |
|---|---:|---:|---|
| Hard-legal cases | 100/100 | **100/100** | pass |
| Weighted no-runtime | <=1.0300 | **6.8159** | fail |
| n=120 decode | <=20 ms | **1194 ms** | fail |
| Repaired-golden calibration | reference | **1.011256** | valid input/evaluator |

Decode timing across 100 cases was 12.9 ms minimum, 94.2 ms median,
604.5 ms p95, 1.194 s maximum, and 172.8 ms mean. The slow tail is therefore
not initialization noise; the sparse-candidate loop scales superlinearly.

The quality miss is also structural rather than one bad constraint family.
The worst cases simultaneously inflate HPWL, bbox area, and boundary
violations. For example, test 98 (n=119) scored 9.2967 with HPWL gap 2.1139,
area gap 3.5628, relative violation 0.4423, and 23 boundary misses. The
representation's useful geometry is being erased by sequential collision
resolution.

One mechanism-level correction was tried after the two-case smoke attribution:
anchor-relative obstacle snaps, explicit wall seating, and a quantized global
frame. It did not repair the channel: the two-case score moved from 3.5200 to
4.7326. No parameter sweep was performed.

## G1: Four-Policy No-Training Probe, n>=100

| Metric | Gate | Measured | Result |
|---|---:|---:|---|
| Wins over repaired production | >=4/21 | **0/21** | fail |
| Full-total-equivalent gain | >=0.005 | **0.000** | fail |
| Hard-legal candidate coverage | >=90% | **84/84 (100%)** | pass |
| Repaired-production tail score | reference | **1.1127** | — |
| Raw constructive best-of-four tail score | lower is better | **5.8076** | fail |
| Four-arm generation time | <=submission budget | **2.078 s/case mean** | fail |

Per-arm mean raw costs were 6.1152 (`net_closure`), 6.7696
(`constraint_first`), 8.7514 (`large_first`), and 6.8139 (`pin_gravity`). A
single candidate took 0.519 s on average. Even the best individual result,
4.4219 on test 80, was far behind that case's 1.2575 production cost.

## Interpretation

The experiment separates two facts:

1. Hard legality is not the blocker. Exact area, fixed/preplaced invariants,
   non-overlap, MIB shape sharing, and cluster construction produced 100%
   admissible candidates.
2. Greedy sparse-slot realization is the blocker. It destroys global packing
   topology and net geometry faster than dynamic ordering can recover them,
   while its per-block obstacle enumeration consumes far more than the runtime
   available after the current 0.28 s solver.

This agrees with, and strengthens, the earlier beam result: the old beam could
have been blamed on soft group handling or one ordering heuristic; G0/G1 removes
those confounders and still fails by several score points. The whole family of
sequential sparse-slot constructive decoders is therefore closed for the final
submission.

## Decision and Next Search Boundary

- Keep this code isolated for reproducibility; do not expose an environment
  flag or candidate slot in production.
- Do not tune beam width, slot count, or the four scoring weights. The gap is
  orders of magnitude larger than plausible tuning gain, and runtime already
  exceeds budget.
- A successor must preserve global topology during realization and be bounded
  near O(n^2) with vectorized/batched operations. It must first pass the same
  repaired-golden G0 before any training or production integration.
- In particular, future work should test a compact global discrete
  representation (for example, coupled sequence-pair/constraint-DAG decisions
  with a single bounded compaction solve), not another sequential block placer.
