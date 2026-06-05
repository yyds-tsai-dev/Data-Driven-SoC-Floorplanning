# Runtime Clamp and Grouping Bias Design

## Goal

Improve ICCAD v10 submission score under a conservative runtime budget by adding
two opt-in controls:

- `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1`
- `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1`

The current production default must remain unchanged.

## Context

The promoted Graph Transformer 0521 checkpoint remains the baseline:

- v10 no-runtime: `2.1194`
- v10 total: `2.6641`
- feasible: `100/100`
- avg/p90/max runtime: `1.23s` / `2.23s` / `6.80s`

The first opt-in v10 soft-repair run improved no-runtime slightly to `2.1082`,
but worsened runtime-aware total to `2.7881` and pushed max runtime to `9.47s`.
This means extra repair has some quality signal, but it is not submission-safe
until runtime tails are clamped.

## Design

### Runtime-Tail Budget Clamp

Add a small helper in `repair.py` that derives a clamped `SolverConfig` from the
existing v10 `RiskBudget` tier when `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1`.

Default clamp values:

| Tier | max repair passes | boundary snaps | cluster moves | pair candidates |
| --- | ---: | ---: | ---: | ---: |
| none/light | `1` | `10` | `8` | `12` |
| medium | `2` | `16` | `14` | `20` |
| heavy | `2` | `20` | `18` | `24` |

The clamp only lowers expensive knobs. It never increases them, and it is a
no-op when the flag is disabled.

Apply the clamp after repair acceptance / v10 risk gate:

- `_v10_soft_repair()` uses the clamped config after `_v10_soft_repair_eligible()`
  returns true.
- `_repair_profile_config()` in `optimizer.py` applies the clamp after profile
  expansion so heavier profiles cannot escape the runtime cap.

### Decoder-Side Grouping Adjacency Bias

`relative_order.py` already chains cluster members for non-compact profiles.
Add an opt-in path that enables that chain for compact profiles too, controlled
by `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1`.

Add `FLOORSET_GROUPING_ADJACENCY_BIAS_MODE` with defaults:

- `auto`: enable for compact and non-compact when the main flag is on.
- `soft_only`: preserve current compact behavior.
- `compact_only`: apply only to compact.

This is deliberately cheaper than repair: it changes ordering constraints before
packing, so grouping pressure can be reduced before `_connect_clusters()` spends
runtime.

## Acceptance Criteria

- With both flags disabled, existing tests and behavior remain unchanged.
- With `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1`, clamped configs lower expensive
  repair knobs according to tier and never increase user-provided smaller caps.
- With `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` and runtime clamp enabled,
  `_v10_soft_repair()` uses no more than the clamped pass budget.
- With `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1`, compact relative-order
  placements can connect cluster members that are split without the flag.
- Full validation records the combined opt-in result against the Graph
  Transformer 0521 checkpoint.

## Decision

Keep both controls opt-in until full validation shows a runtime-aware total-score
win. If no-runtime improves but runtime-aware total regresses, keep the controls
as ablation tools and do not promote them into submission defaults.
