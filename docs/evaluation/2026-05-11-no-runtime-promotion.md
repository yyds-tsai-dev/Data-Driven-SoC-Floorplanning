# No-Runtime Checkpoint Promotion And Large-Case Matrix Evidence

## Goal

Use `total_score_no_runtime` as the promotion metric for available Anchor-GNN checkpoints and re-check the large-case candidate matrix without local runtime-factor distortion.

## Evaluation Setup

- Evaluator: `FloorSet/iccad2026contest/iccad2026_evaluate.py`
- Optimizer wrapper: historical `src/architecture_v4_optimizer.py` path; current evaluator scripts use `src/architecture_v5_optimizer.py`.
- Validation cases: 100/100
- Promotion metric: `total_score_no_runtime`
- Runtime source: `time.perf_counter()` after fixing wall-clock timing.

## Checkpoint Promotion

| Variant | Checkpoint | Large-Case Matrix | Feasible | Total Score | No-Runtime Total | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| latest_200k_ep10_h192 | `checkpoints/gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt` | off | 100 | 2.0002 | 1.9560 | Promote |
| default_gnn_best_perf | `checkpoints/gnn_best.pt` | off | 100 | 2.0209 | 1.9848 | Superseded |
| default_gnn_latest_perf | `checkpoints/gnn_latest.pt` | off | 100 | 2.0311 | 1.9848 | Do not promote |
| best_500k_ep4_h192 | `checkpoints/gnn_best_0510_ns500000_ep4_h192_l6_acc32.pt` | off | 100 | 1.9936 | 1.9953 | Do not promote |
| latest_500k_ep4_h192 | `checkpoints/gnn_latest_0510_ns500000_ep4_h192_l6_acc32.pt` | off | 100 | 2.0640 | 1.9953 | Do not promote |
| best_200k_ep10_h192 | `checkpoints/gnn_best_0510_ns200000_ep10_h192_l6_acc32.pt` | off | 100 | 2.4110 | 2.2768 | Do not promote |

`gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt` wins the full validation `total_score_no_runtime` despite not being the supervised-val-loss best checkpoint. Promotion should follow evaluator evidence, not training loss alone.

## Large-Case Candidate Matrix

| Variant | Checkpoint | Matrix Setting | Feasible | Total Score | No-Runtime Total | Avg Runtime | P90 Runtime | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| latest_200k_ep10_h192 | `checkpoints/gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt` | `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=0` | 100 | 2.0002 | 1.9560 | 0.973s | 1.814s | Keep default |
| large_matrix_latest_200k_ep10_h192 | `checkpoints/gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt` | `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1` | 100 | 2.3021 | 1.9560 | 1.059s | 1.969s | Keep opt-in |
| default_gnn_best_perf | `checkpoints/gnn_best.pt` | `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=0` | 100 | 2.0209 | 1.9848 | 1.003s | 1.859s | Superseded |

Tail-case no-runtime deltas for the promoted checkpoint were all zero for IDs 95-99:

| Test ID | Blocks | Default No-Runtime | Matrix No-Runtime | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 95 | 116 | 2.7948 | 2.7948 | +0.0000 |
| 96 | 117 | 1.6946 | 1.6946 | +0.0000 |
| 97 | 118 | 1.8599 | 1.8599 | +0.0000 |
| 98 | 119 | 2.1289 | 2.1289 | +0.0000 |
| 99 | 120 | 1.9000 | 1.9000 | +0.0000 |

The matrix remains useful as an opt-in experiment, but this evidence does not justify enabling it by default.

## Training Script Readiness

`scripts/train.sh` is aligned with the current v4 direction: h192/l6 defaults, pairwise head enabled, weighted dirty-sample handling in the Python trainer, and no stable-checkpoint overwrite unless `WRITE_STABLE_CHECKPOINTS=1` is set. Remote runs can override `OUTPUT_DIR`, `WANDB_MODE`, `WANDB_ENTITY`, `NUM_WORKERS`, `PRINT_EVERY`, `VAL_START`, clean-sample policy, and resume behavior through environment variables.

Strict clean-only filtering is not a safe default: a 500-sample training-data probe found only 1 clean sample. The default `CLEAN_SAMPLE_POLICY=weighted` keeps dirty samples as low-weight geometry references (`DIRTY_SAMPLE_WEIGHT=0.25`) while suppressing dirty order/pairwise losses (`DIRTY_ORDER_WEIGHT=0.0`).

## Timing Fix

While extracting evidence, saved results showed occasional negative runtimes under `time.time()`. The evaluator now uses `time.perf_counter()` for optimizer timing. This does not change `total_score_no_runtime`, but it makes raw runtime and local runtime-aware totals meaningful.
