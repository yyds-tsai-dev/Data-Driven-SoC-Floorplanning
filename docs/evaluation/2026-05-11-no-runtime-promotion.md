# No-Runtime Checkpoint Promotion And Large-Case Matrix Evidence

## Goal

Use `total_score_no_runtime` as the promotion metric for available Anchor-GNN checkpoints and re-check the large-case candidate matrix without local runtime-factor distortion.

## Evaluation Setup

- Evaluator: `FloorSet/iccad2026contest/iccad2026_evaluate.py`
- Optimizer wrapper: `src/architecture_v4_optimizer.py`
- Validation cases: 100/100
- Promotion metric: `total_score_no_runtime`
- Runtime source: `time.perf_counter()` after fixing wall-clock timing.

## Checkpoint Promotion

| Variant | Checkpoint | Large-Case Matrix | Feasible | Total Score | No-Runtime Total | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| default_gnn_best_perf | `checkpoints/gnn_best.pt` | off | 100 | 2.0608 | 1.9848 | Keep default |
| default_200k_ep10_h192 | `checkpoints/gnn_best_0510_ns200000_ep10_h192_l6_acc32.pt` | off | 100 | 2.4602 | 2.2768 | Do not promote |
| default_500k_ep4_h192 | `checkpoints/gnn_best_0510_ns500000_ep4_h192_l6_acc32.pt` | off | 100 | 1.6128 | 1.9953 | Do not promote |

`gnn_best_0510_ns500000_ep4_h192_l6_acc32.pt` had a better local runtime-aware total in one run, but its no-runtime total was worse than `gnn_best.pt`. Under the current score policy, it is not promoted.

## Large-Case Candidate Matrix

| Variant | Checkpoint | Matrix Setting | Feasible | Total Score | No-Runtime Total | Avg Runtime | P90 Runtime | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| default_gnn_best_perf | `checkpoints/gnn_best.pt` | `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=0` | 100 | 2.0608 | 1.9848 | 0.959s | 1.797s | Keep default |
| large_matrix_gnn_best_perf | `checkpoints/gnn_best.pt` | `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1` | 100 | 2.2894 | 1.9848 | 1.016s | 1.837s | Keep opt-in |

Tail-case no-runtime deltas were all zero for IDs 95-99:

| Test ID | Blocks | Default No-Runtime | Matrix No-Runtime | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 95 | 116 | 2.3925 | 2.3925 | +0.0000 |
| 96 | 117 | 1.6700 | 1.6700 | +0.0000 |
| 97 | 118 | 1.6855 | 1.6855 | +0.0000 |
| 98 | 119 | 2.3653 | 2.3653 | +0.0000 |
| 99 | 120 | 1.8918 | 1.8918 | +0.0000 |

The matrix remains useful as an opt-in experiment, but this evidence does not justify enabling it by default.

## Timing Fix

While extracting evidence, saved results showed occasional negative runtimes under `time.time()`. The evaluator now uses `time.perf_counter()` for optimizer timing. This does not change `total_score_no_runtime`, but it makes raw runtime and local runtime-aware totals meaningful.
