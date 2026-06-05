# v10 Evaluator And Selection Recalibration Design

## Goal

Align the local evaluator and checkpoint-selection evidence with the ICCAD 2026 FloorSet v10 scoring rules before making any algorithm changes.

This design has two outcomes:

1. Tests exercise `scripts/iccad2026_evaluate.py` directly, because that is the repository-maintained evaluator copy.
2. Checkpoint and ablation decisions use v10 score semantics, especially feasible-cost capping and `exp(n/12)` total-score weighting.

Algorithm patches are intentionally out of scope until recalibrated v10 evidence shows which failure mode matters most.

## Context

The official FloorSet v10 PDF changed two scoring details that affect local decision-making:

- Feasible per-case costs are capped below the infeasible penalty: `min(formula_cost, M - 1e-6)`.
- Total score weights are normalized from `exp(n_i / 12)`, not `exp(n_i)`.

The production solver path does not need to move files or change package ownership for this. `FloorSet/` is treated as the official contest repository/submodule. The local scripts evaluator is an intentional repository-maintained copy used for no-runtime diagnostics and copy-over workflow.

The existing test file imports `FloorSet/iccad2026contest/iccad2026_evaluate.py` by putting that directory first on `sys.path`. That official evaluator copy has a smaller API than the scripts evaluator, so tests expecting `compute_cost_breakdown`, `cost_no_runtime`, and env-loading helpers fail against the wrong module.

## Decisions

1. Do not move `FloorSet` files into `src`.
2. Do not duplicate `FloorSet` helper modules under `src`.
3. Test `scripts/iccad2026_evaluate.py` by loading it from its file path with `importlib.util.spec_from_file_location`.
4. Add `FloorSet/` root to `sys.path` during that test import so bare imports such as `litetestLoader`, `liteLoader`, `lite_dataset`, `cost`, and `utils` resolve.
5. Keep `FloorSet/iccad2026contest/iccad2026_evaluate.py` as the official or copied execution target used by shell scripts after `scripts/update.sh`.
6. Update tests to assert v10 behavior directly: feasible cap, `exp(n/12)` weights, and contributor shares that are no longer `>99%` for the largest case.
7. Recalibrate selection docs and metric interpretation before touching `floorset_arch` algorithm code.

## Architecture

The evaluator boundary remains:

```text
tests/test_evaluator_scoring.py
  -> direct file load: scripts/iccad2026_evaluate.py
      -> imports official helper modules from FloorSet/

scripts/update.sh
  -> copies scripts/iccad2026_evaluate.py
     into FloorSet/iccad2026contest/iccad2026_evaluate.py
     for evaluator CLI runs
```

This avoids a new package layer and keeps the existing contest workflow intact.

Selection recalibration should treat `src/floorset_arch/training/selection.py` as the checkpoint promotion policy surface, and `docs/optimization-notes.md` as the human-readable source of local scoring guidance. The solver implementation remains unchanged until new v10-weighted evidence exists.

## Test Requirements

`tests/test_evaluator_scoring.py` should:

- import the scripts evaluator directly;
- verify `compute_cost(..., is_feasible=True)` caps pathological feasible cases at `9.999999`;
- verify infeasible cases still score `10.0`;
- verify no-runtime cost still uses runtime adjustment `1.0`;
- verify total score uses `exp(n/12)` by checking a hand-computed case;
- verify score contributors use the same `exp(n/12)` shares;
- keep monotonic-runtime and env-loader tests pointed at the scripts evaluator API.

## Recalibration Requirements

After tests are fixed, run or document a recalibration pass that reports:

- `total_score`;
- `total_score_no_runtime`;
- feasible count;
- average, median, p90, and max runtime;
- top contributors under v10 weights;
- large-case bucket summaries for 101-120 and 116-120;
- high-risk / quality-portfolio trigger decisions with the block count, v10 score
  contribution, constraint density, and net density that caused each trigger;
- tail case details for IDs 95-99 only as diagnostics, not as the whole decision surface.

The previous mental model that case 120 alone dominates the score is no longer valid. Under v10 weights, the 120-block case is still important but no longer overwhelms the rest of the validation set. The 116-120 bucket remains important enough to guide targeted analysis, but full-range total score should be the promotion metric.

## High-Risk And Quality Portfolio Gating

High-risk and quality-portfolio gating should be recalibrated as part of this work, not deferred to a later algorithm patch.

The current code already uses more than one signal, but the thresholds were tuned under an older tail-dominance assumption. v10 changes the budget logic:

- A 120-block case is no longer about `63%` of total weight; it is about `8%`.
- The 116-120 bucket is still important at about `34%`, but not enough to justify treating every 118-120 case as automatically worth maximum extra search.
- Medium-large cases with high constraint density or net density can deserve extra budget even when they are below 118 blocks.

Define a shared risk/budget interpretation for both high-risk repair and quality portfolio:

```text
v10_score_share = exp((block_count - max_block_count) / 12) / sum_j exp((n_j - max_block_count) / 12)
constraint_density = (boundary_count + grouping_budget + mib_budget + hard_shape_count) / block_count
net_density = (b2b_count + p2b_count) / block_count
```

Use these signals to choose budget tiers:

- `none`: normal single-candidate path.
- `light`: one additional profile or cheap refinement only.
- `medium`: high-risk normal repair / one quality refinement profile.
- `heavy`: multiple profiles or larger repair caps, only when v10 contribution and densities both justify the runtime.

The recalibration should produce diagnostic rows showing why each case entered a tier. The goal is not to hard-code validation IDs, but to make the trigger reusable for hidden cases with similar block-count, constraint-density, and net-density structure.

## Algorithm Decision Gate

No algorithm patch should be selected until a v10 recalibration report answers:

1. Is the current default still `100/100` feasible under the scripts evaluator and copied official evaluator path?
2. Which v10-weighted cases are top contributors after the `exp(n/12)` change?
3. Is the regression driven by hard feasibility, soft violations, HPWL gap, area gap, or runtime?
4. Do existing opt-in knobs improve v10 full-score metrics, not just old tail-only diagnostics?

Only after those answers should the next algorithm patch be chosen.

## Candidate Algorithm Directions After Recalibration

These are possible follow-up directions, not part of this implementation:

1. Replace block-count-only high-risk / quality-portfolio gating with shared v10
   budget tiers driven by score contribution, constraint density, and net density.
2. Adjust candidate ranking to penalize v10 score factors more directly.
3. Run HGT guidance-head ablations to identify whether pairwise, aspect, or priority guidance harms the decoder.
4. Promote checkpoints only from evaluator evidence, not supervised validation loss alone.

## Self-Review

- No new duplication of `FloorSet` files is introduced.
- The test import target is explicit and does not depend on ambient `PYTHONPATH` order.
- The evaluator change is separated from algorithm changes.
- The recalibration gate prevents tuning against stale v9 tail dominance assumptions.
