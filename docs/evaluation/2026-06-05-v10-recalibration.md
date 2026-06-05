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

This is a `+0.1252` no-runtime regression and a `+0.2527` total-score
regression, so keep `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` opt-in only.
The artifact is
`artifacts/eval_v10/quality_auto_graph_transformer_0521_v10.json`.

Large-case candidates still have no measured v10 gain. No-Checkpoint Guidance
Mode remains useful for deterministic repair diagnosis but is not competitive
with the configured checkpoint path.
