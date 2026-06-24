# v10 Recalibration Notes

## Purpose

Recompute local evaluator expectations after the ICCAD 2026 FloorSet v10 scoring update.

## Formula Checks

- Feasible pathological case: `compute_cost(100, 100, 1, 100, True)` returns `9.999999`.
- Infeasible case: `compute_cost(0, 0, 0, 1, False)` returns `10.0`.
- Total score uses `exp(n/12)` weights.
- With block counts 21-120, the 120-block case carries about `7.9975%` of total weight.
- With block counts 21-120, the 116-120 bucket carries about `34.0841%` of total weight.

## Decision Impact

The old v9 assumption that ID 99 alone dominates the validation total is stale. Continue reporting IDs 95-99 because they are useful diagnostics, but promote checkpoints and algorithm changes using full-validation `total_score_no_runtime` first.

## Gating Impact

High-risk repair and quality portfolio budget should use v10 score share plus instance structure:

- `score_share`: normalized `exp(n/12)` contribution.
- `constraint_density`: boundary, grouping, MIB, fixed, and preplaced pressure per block.
- `net_density`: block-to-block plus pin-to-block edge count per block.

A sparse 120-block case is no longer automatically a heavy-budget case. A medium-large case with high constraint and net density can receive more budget even when it is below 118 blocks.

## Required Next Run

Run a full validation evaluation using the copied scripts evaluator path:

```bash
scripts/update.sh
bash scripts/eval_total.sh
```

Record:

- `total_score`
- `total_score_no_runtime`
- feasible count
- average runtime
- median runtime
- p90 runtime
- max runtime
- top score contributors
- top no-runtime score contributors
- high-risk / quality portfolio budget tiers

## Full Run Result

Run on 2026-06-05 with `bash scripts/eval_total.sh ... --output <absolute-path>`.
The official v10 evaluator path reports neutral-runtime `cost`, so these totals are
the current no-runtime/quality comparison surface.

| Run | v10 neutral total | Feasible | Avg runtime | ID98 cost | ID99 cost | ID98 contribution | ID99 contribution | IDs95-99 contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `.env` checkpoint `gnn_best_0512_ns500000_ep4_h192_l6_acc32.pt` | `2.2337` | `100/100` | `1.14s` | `2.4619` | `2.1024` | `8.11%` | `7.53%` | `31.23%` |
| `gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt` | `2.3390` | `100/100` | `1.19s` | `2.1357` | `2.4078` | `6.72%` | `8.23%` | `30.63%` |
| `.env` checkpoint + `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1` | `2.2337` | `100/100` | `1.14s` | `2.4619` | `2.1024` | `8.11%` | `7.53%` | `31.23%` |
| `.env` checkpoint + `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` | `2.2299` | `100/100` | `1.17s` | `2.4437` | `2.1019` | `8.06%` | `7.54%` | `31.16%` |
| No-Checkpoint Guidance Mode | `5.0509` | `100/100` | `1.13s` | `6.5866` | `3.3906` | `9.60%` | `5.37%` | `29.57%` |

Artifacts are stored under `artifacts/eval_v10/`.

The fixed v10 score weights are `7.3580%` for ID 98, `7.9975%` for ID 99,
and `34.0841%` for IDs 95-99 together.

## Best Checkpoint Sweep

Run with:

```bash
bash scripts/eval_total.sh --best-since-0512
```

The script reuses valid JSON results already present under
`artifacts/eval_v10/best_since_0512/` unless `FLOORSET_EVAL_REFRESH=1` is set.

| Rank | Checkpoint | v10 no-runtime | v10 total | Feasible | Avg runtime |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt` | `2.1194` | `2.6641` | `100/100` | `1.23s` |
| 2 | `gnn_best_0514_ns800000_ep4_h192_l6_acc32.pt` | `2.1749` | `2.7450` | `100/100` | `1.16s` |
| 3 | `gnn_hgt_best_val_loss_0603_ns800000_ep10_enchgt_h256_l4_acc32_bs8_heads4.pt` | `2.2144` | `2.7476` | `100/100` | `1.22s` |
| 4 | `gnn_best_0512_ns500000_ep4_h192_l6_acc32.pt` | `2.2337` | `2.2337` | `100/100` | `1.14s` |
| 5 | `gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt` | `2.2563` | `2.8308` | `100/100` | `1.19s` |
| 6 | `gnn_hgt_best_0601_ns500000_ep3_enchgt_h256_l4_acc32_bs8_heads4.pt` | `2.3006` | `2.9253` | `100/100` | `1.27s` |
| 7 | `gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt` | `2.3390` | `2.3390` | `100/100` | `1.19s` |

Current global best checkpoint:
`checkpoints/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt`.

## Algorithm Gate

Quality portfolio auto is not a production-default candidate for the current
Graph Transformer checkpoint. It slightly improved the older 0512 checkpoint
from `2.2337` to `2.2299`, but the follow-up full validation on the current
global-best Graph Transformer 0521 checkpoint regressed:

| Run | v10 no-runtime | v10 total | Feasible | Avg runtime | P90 runtime | Max runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph Transformer 0521 baseline | `2.1194` | `2.6641` | `100/100` | `1.23s` | `2.23s` | `6.80s` |
| Graph Transformer 0521 + `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` | `2.2446` | `2.9168` | `100/100` | `1.24s` | `2.53s` | `5.97s` |
| Graph Transformer 0521 + `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` | `2.1082` | `2.7881` | `100/100` | `1.40s` | `2.54s` | `9.47s` |
| Graph Transformer 0521 + soft repair + `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1` | `2.2741` | `3.0120` | `100/100` | `1.24s` | `1.89s` | `8.11s` |
| Graph Transformer 0521 + soft repair + runtime clamp + `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1` | `2.4989` | `3.3102` | `100/100` | `1.24s` | `1.95s` | `8.65s` |

This is a `+0.1252` no-runtime regression and a `+0.2527` total-score
regression, so keep `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` opt-in only.
The artifact is
`artifacts/eval_v10/quality_auto_graph_transformer_0521_v10.json`.

The first v10 risk-gated soft-repair run produced a small no-runtime gain
against the Graph Transformer 0521 baseline (`-0.0112`), but it regressed the
runtime-aware total by `+0.1240`. The largest diagnostic bucket did not benefit:
IDs 95-99 each had flat or worse no-runtime cost, and the runtime tail worsened
on IDs 95, 96, 98, and 99. The worst runtime regression was ID 88, rising from
`6.02s` to `9.47s` while also worsening total cost. Keep
`FLOORSET_ENABLE_V10_SOFT_REPAIR=1` opt-in and do not combine it with production
submission defaults until conditional runtime budget also preserves no-runtime
quality. The artifact is
`artifacts/eval_v10/v10_soft_repair_graph_transformer_0521_v10.json`.

The first runtime-tail clamp reduced tail runtime but over-clamped quality. With
soft repair enabled, p90 improved from `2.54s` to `1.89s` and max runtime
improved from `9.47s` to `8.11s`, but no-runtime score regressed to `2.2741`
and runtime-aware total regressed to `3.0120`. The largest bucket did not recover:
IDs 95, 96, 98, and 99 were worse than baseline on no-runtime cost. The former
`FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP=1` hard-cap opt-in was removed; future work
should use per-case conditional caps instead of resurrecting that flag. The artifact is
`artifacts/eval_v10/runtime_clamp_graph_transformer_0521_v10.json`.

The first decoder-side grouping adjacency bias was also not submission-safe.
Combined with soft repair and runtime clamp, no-runtime score regressed to
`2.4989` and total score regressed to `3.3102`. Runtime stayed bounded, but many
mid/large cases gained soft violations; the largest no-runtime regressions
included IDs 81, 82, 49, 64, 68, and 92. The former
`FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS=1` broad opt-in was removed; current
work should avoid globally compacting cluster keys and instead use the retained
narrow bias only when grouping soft pressure is high and the local order gap is ambiguous. The
artifact is
`artifacts/eval_v10/runtime_clamp_grouping_bias_graph_transformer_0521_v10.json`.
These two flags are historical evidence only after the 2026-06-11 cleanup; their
live code paths were removed in favor of conditional runtime budget and
default-on narrow grouping pair bias.

Large-case candidates still have no measured v10 gain. No-Checkpoint Guidance
Mode remains useful for deterministic repair diagnosis but is not competitive
with the configured checkpoint path.

## 2026-06-10 Proxy / Runtime / Grouping Ablation

This run uses fresh same-machine full validation with clean detached worktrees:

- baseline worktree: `78b8f86022753874a0e6d86e5ef70cb480544ec8`
- current worktree: `1907b6386c0bfc7fb8cc71f51fd22b3dcde43cdf`
- checkpoint:
  `gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt`
- output directory: `artifacts/eval_v10_ablation/`

Treat this table as the current apples-to-apples ablation surface. The older
`artifacts/eval_v10/` results above remain useful history, but their baseline
does not numerically match this rerun and should not be mixed into delta claims.

| Run | v10 no-runtime | v10 total | Feasible | Avg runtime | P90 runtime | Max runtime | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| merge-base baseline | `2.2184` | `2.7762` | `100/100` | `1.25s` | `2.27s` | `7.01s` | comparison baseline |
| Phase 1 V10 proxy default | `2.1634` | `2.7471` | `100/100` | `1.24s` | `2.12s` | `7.42s` | keep default |
| Phase 1 + v10 soft repair, no conditional budget | `2.1446` | `2.8314` | `100/100` | `1.40s` | `2.46s` | `9.64s` | keep opt-in only |
| Phase 1 + v10 soft repair + conditional budget | `2.1670` | `2.7207` | `100/100` | `1.22s` | `2.56s` | `5.70s` | useful for total/runtime, not no-runtime |
| Phase 1 + narrow grouping pair bias | `2.1538` | `2.6993` | `100/100` | `1.23s` | `2.24s` | `6.58s` | promote to default |
| Phase 1 + conditional budget + narrow grouping | `2.1926` | `2.7515` | `100/100` | `1.18s` | `2.13s` | `5.79s` | do not combine by default |

Phase 1 is confirmed as a production default: it lowers no-runtime by `0.0549`
and total by `0.0291` against the fresh merge-base rerun. The remaining tail
risk is runtime noise and case-specific geometry, not a reason to return to
soft-first selection.

Conditional runtime budget improves the soft-repair path's runtime-aware total
relative to unbudgeted soft repair (`2.8314 -> 2.7207`) and cuts max runtime
(`9.64s -> 5.70s`), but it slightly regresses no-runtime relative to Phase 1
alone (`2.1634 -> 2.1670`). Keep
`FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET=1` and
`FLOORSET_ENABLE_V10_SOFT_REPAIR=1` as an explicit ablation pair. The next
optimization should gate the extra soft-repair attempt more tightly on proxy
acceptance and geometry deltas, especially for regressions like IDs 68, 55, 62,
73, and 27.

Narrow grouping pair bias is the first grouping-side variant that improves both
primary surfaces in this rerun. It lowers no-runtime from `2.1634` to `2.1538`
and total from `2.7471` to `2.6993` relative to Phase 1 alone, while avoiding
the old broad global key-blend failure mode. Promote narrow grouping pair bias
to default, but keep `FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS=0` as an
ablation switch. Remaining regressions are local geometry/soft tradeoffs, with
IDs 59, 27, 5, 73, 25, 89, 71, and 76 the first targets if this bias is tuned
again.

Do not promote the combined conditional-budget plus narrow-grouping preset. It
reduces raw runtime tail, but no-runtime regresses to `2.1926` and total
regresses to `2.7515` versus narrow grouping alone. This suggests the current
soft-repair extra path fights the narrow decoder bias on high-weight cases such
as ID 99.
