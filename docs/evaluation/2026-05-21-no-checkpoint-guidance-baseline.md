# No-Checkpoint Guidance Baseline

## Purpose

Measure how far the deterministic decoder and repair path can go without Anchor-GNN Guidance. This isolates repair headroom from checkpoint quality after larger GNN training runs showed limited quality gains.

## Runs

| Run | Checkpoint | Repair trace | Total score | No-runtime total | Feasible | Avg runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| No-Checkpoint Guidance Mode | `/tmp/floorset-no-gnn-missing.pt` | `artifacts/no_checkpoint_guidance_repair_trace.jsonl` | `6.2174` | `4.6289` | `100/100` | `1.59s` |
| Configured Anchor-GNN path | `.env`: `checkpoints/gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt` | `artifacts/configured_checkpoint_repair_trace.jsonl` | `3.2697` | `2.0326` | `100/100` | `1.57s` |

## Repair Trace Summary

| Run | Scope | Before overlap avg | After overlap avg | Before soft avg | After soft avg | Avg move |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No-Checkpoint | All traced candidates | `3.53` | `0.00` | `23.50` | `7.10` | `101.19` |
| No-Checkpoint | Blocks 116-120 | `4.95` | `0.00` | `36.21` | `11.68` | `229.44` |
| Configured GNN | All traced candidates | `5.62` | `0.00` | `22.23` | `6.12` | `42.68` |
| Configured GNN | Blocks 116-120 | `10.37` | `0.00` | `31.53` | `6.84` | `55.58` |

## Tail Case Delta

Delta is No-Checkpoint minus configured GNN.

| Test ID | Blocks | No-runtime delta | Cost delta | Vrel delta | HPWL gap delta | Area gap delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 95 | 116 | `+1.9986` | `+2.9162` | `-0.0455` | `+1.1169` | `+2.3015` |
| 96 | 117 | `+3.5377` | `+7.0770` | `+0.1538` | `+1.9564` | `+1.6901` |
| 97 | 118 | `+1.9619` | `+1.2985` | `+0.0333` | `+1.0366` | `+1.6804` |
| 98 | 119 | `+6.0253` | `+5.2434` | `+0.0577` | `+2.1992` | `+5.6348` |
| 99 | 120 | `+1.3816` | `+2.1068` | `-0.0149` | `+0.7260` | `+1.9080` |

## Interpretation

No-Checkpoint Guidance Mode is viable as a repair ablation, not as a replacement path yet. Repair can remove traced overlaps and reduce many boundary/group violations without GNN guidance, but it needs much larger movement and leaves worse HPWL/area quality. The next repair work should target large-case geometry preservation while keeping soft-first acceptance: reduce boundary/group/MIB violations without expanding bbox or destroying HPWL, with ID 98 as the sharpest regression signal.

## Follow-Up: Boundary-Edge Shrink

Added a no-guidance-only geometry refinement that tries to shrink already-satisfied movable boundary edges inward when doing so preserves overlap freedom and does not increase soft violations. The expensive frontier polish remains opt-in through `FLOORSET_GEOMETRY_REFINE_MAX_BLOCKS`; the default path uses the cheaper edge shrink only.

| Run | Total score | No-runtime total | Avg runtime | Median runtime | P90 runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-Checkpoint baseline | `6.2174` | `4.6289` | `1.59s` | `1.18s` | `2.35s` |
| No-Checkpoint + edge shrink | `6.5204` | `4.5038` | `1.47s` | `1.04s` | `2.17s` |

ID 98 improved from `8.2411` to `7.7030` no-runtime cost. Its area gap improved from `6.3340` to `6.0228`, HPWL gap from `2.8857` to `2.8754`, and relative soft violation from `0.1923` to `0.1731`.

The configured Anchor-GNN ID 98 check stayed at `2.2158` no-runtime cost, confirming the new repair branch is isolated to `AnchorGuidance=None`.
