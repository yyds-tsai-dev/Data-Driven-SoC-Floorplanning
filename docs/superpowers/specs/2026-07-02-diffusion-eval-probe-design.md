# Diffusion Evaluation Probe Checkpoint Design

## Goal

Create a deliberately scoped overfit probe that trains the current v11 diffusion training path on the 100 local evaluation cases from `LiteTensorDataTest`, saves a quick checkpoint, and runs one final full evaluator pass to verify that the current training, sampling, repair, and ranking pipeline can learn useful signal.

This checkpoint is diagnostic only. It must not be treated as generalized validation evidence, a hidden-test proxy, or an evaluator-best promotion candidate.

## Decisions

1. Treat the run as an overfit probe / pipeline sanity check.
2. Train on all 100 local evaluation cases; do not create a separate supervised validation split.
3. Disable tree supervision because `LiteTensorDataTest` labels provide polygon floorplans and metrics, but no `tree_sol`.
4. Use `TREE_WEIGHT=0` and reject eval-probe runs that try to enable tree loss.
5. Save checkpoints during training, but do not run the full evaluator every epoch.
6. Run the full evaluator once after training completes.
7. Default to 100 epochs, with an easy path to extend to 200 epochs if the probe loss is still improving.
8. Add a dedicated script so probe settings do not blend into normal diffusion training.

## Architecture

Add an evaluation-probe dataset adapter that converts each `FloorplanDatasetLiteTest` sample into the existing diffusion trainer's 8-field sample contract:

```text
area_targets
b2b_connectivity
p2b_connectivity
pins_pos
placement_constraints
tree_sol
fp_sol
metrics_sol
```

The adapter must keep the training loop narrow:

- `tree_sol` is an empty tensor.
- `fp_sol` is converted from validation polygons to training-style `(w, h, x, y)` rows.
- `metrics_sol` is passed through.
- The existing `_prepare_diffusion_sample()` and `build_diffusion_targets()` path remains the target construction boundary.

The training entrypoint should expose a small dataset-mode switch, for example `--dataset-mode lite|eval-probe`, while the user-facing entry is a dedicated `scripts/train_diffusion_eval_probe.sh` wrapper.

## Data Flow

`scripts/train_diffusion_eval_probe.sh` sets a probe profile:

- `DATASET_MODE=eval-probe`
- `NUM_SAMPLES=100`
- `EPOCHS=100`
- `TREE_WEIGHT=0`
- train-time evaluator disabled
- checkpoint prefix or tag includes `eval_probe`

`train_diffusion.py --dataset-mode eval-probe` loads `FloorplanDatasetLiteTest(FloorSet)`, adapts all 100 cases, and trains on the full adapted set. The run reports probe training stats rather than supervised validation stats.

After training, the script runs one full evaluator pass against the chosen probe checkpoint, normally the latest or best probe-loss checkpoint. The resulting JSON is labeled as overfit probe evidence.

## Error Handling

The probe should fail fast on contract drift:

- `LiteTensorDataTest` must contain exactly 100 cases.
- Each adapted label must include polygon coordinates with final dimension 2.
- The adapted `fp_sol` must contain one positive-width and positive-height rectangle per valid block.
- `eval-probe` with nonzero tree weight must fail, because there is no real `tree_sol`.
- `eval-probe` with train-time full evaluator enabled must fail, because this design uses final-only evaluator verification.

Failures should include the case index when the issue is sample-specific.

## Testing

Add focused tests instead of long training runs:

- Adapter unit test for a synthetic `FloorplanDatasetLiteTest`-style sample.
- Field-order test proving polygon labels become `(w, h, x, y)`.
- Integration test proving the adapted sample can enter `_prepare_diffusion_sample()` / `build_diffusion_targets()`.
- Config guard tests for `TREE_WEIGHT != 0` and train-time evaluator enabled under `eval-probe`.
- A one-epoch CPU smoke path that writes a loadable checkpoint.

Manual verification should start with a short run:

```bash
EPOCHS=1 DEVICE=cpu WANDB=0 bash scripts/train_diffusion_eval_probe.sh
```

The full 100-epoch probe can run after the short command passes.

## Interpretation

Success means the latest diffusion training machinery can overfit the 100 local evaluation cases and produce a checkpoint that exercises the full diffusion sampling, repair, ranking, visualization, and evaluator path.

Failure narrows the debugging target. If probe loss does not fall, inspect target construction and model training. If probe loss falls but final evaluator remains weak, inspect sampling, concretization, repair, and v10 ranking handoff. If evaluator execution fails, inspect checkpoint loading and environment isolation.

Do not compare this probe score against normal architecture-promotion runs except as a diagnostic sanity check.
