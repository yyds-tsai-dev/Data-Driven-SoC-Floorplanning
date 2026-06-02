# HGT Ablation, Pseudo Targets, And Selection Design

## Goal

Improve Local HGT evaluation quality without changing the Production Solver Path contract. The first implementation wave will make HGT failures diagnosable, stop treating supervised validation loss as the promoted checkpoint metric, use repaired pseudo targets for dirty training samples, and add relation gates to the HGT encoder.

The first-wave promotion target remains evaluator evidence, not training loss:

- Beat the current GNN baseline on `total_score_no_runtime`.
- Preserve `100/100` feasibility.
- Report raw runtime separately from runtime-aware total.

Teacher residuals and teacher distillation are explicitly out of scope for the first wave.

## Context

The HGT checkpoint `gnn_hgt_best_0601_ns500000_ep3_enchgt_h256_l4_acc32_bs8_heads4.pt` loaded correctly, but evaluated worse than the GNN and Graph Transformer baselines:

- HGT: `3.0849` total, `2.5166` no-runtime.
- Existing v4/GNN local result: about `2.0856` no-runtime.
- Documented GNN baseline: `2.0864` no-runtime.
- Documented Graph Transformer result: `2.2024` no-runtime.

The failure is not feasibility. HGT produced `100/100` feasible layouts with zero hard violations. The score regression is driven by HPWL, area, and soft-constraint trade-offs. The dominant tail failure is `test_id=99`, where HGT increased no-runtime cost from `2.1011` to `2.8555`, mostly through area gap, HPWL gap, and doubled soft violations.

The training data is also highly noisy for constraint-sensitive supervision. In the 500k HGT training window, only `491` samples were constraint-clean, while `499,509` were dirty. The current weighted dirty policy keeps dirty geometry as low-weight anchor/aspect reference but suppresses dirty order and pairwise supervision entirely. That protects against bad labels, but leaves HGT with too little reliable relative-order signal.

## Research Basis

The design follows three known directions:

- HGT uses node- and edge-type dependent attention over heterogeneous relations, so relation gates are a local extension of the same typed-relation idea: <https://arxiv.org/abs/2003.01332>.
- Knowledge distillation can transfer guidance behavior from a stronger teacher to another model, but this remains a later wave here: <https://arxiv.org/abs/1503.02531>.
- Pseudo-label and self-training methods can add supervision when original labels are scarce or noisy. This design uses deterministic repaired pseudo targets instead of model-generated labels for the first wave: <https://arxiv.org/abs/1909.13788>.

## Decisions

1. Add HGT guidance ablation controls before changing training behavior.
2. Keep `latest` checkpoint saving by epoch, but stop treating supervised validation loss as the promoted checkpoint metric.
3. Add an evaluator/tail metric manifest for checkpoint selection and promotion evidence.
4. Use hybrid dirty-sample handling: online repaired pseudo targets first, with a cache-compatible target builder API reserved for later.
5. Dirty order and pairwise losses may become non-zero only when derived from repaired pseudo targets, not raw dirty `fp_sol`.
6. Add relation gates to HGT layers, initialized close to identity so existing behavior is not disrupted at startup.
7. Do not add teacher residuals, teacher checkpoint loading, or teacher distillation in the first implementation wave.

## HGT Guidance Ablation

Add inference-time knobs that alter only `AnchorGuidance` consumption after checkpoint prediction:

- `FLOORSET_GUIDANCE_DISABLE_PAIRWISE=1`
- `FLOORSET_GUIDANCE_DISABLE_ASPECT=1`
- `FLOORSET_GUIDANCE_DISABLE_PRIORITY=1`
- `FLOORSET_GUIDANCE_ANCHOR_ONLY=1`

`ANCHOR_ONLY` disables pairwise, aspect, and priority guidance while preserving anchor rectangles. The model still loads and runs normally, so the ablation isolates which guidance head harms the decoder and repair path.

Each ablation should report:

- `total_score`
- `total_score_no_runtime`
- feasible count
- average and tail runtime
- tail IDs 95-99 costs
- HPWL gap, area gap, and boundary/group/MIB soft counts

## Checkpoint Selection

Training will continue to save:

- `latest`: most recent epoch.
- `best_val_loss`: lowest supervised validation loss.

Promotion will use a separate metrics manifest rather than overwriting best by validation loss. The manifest will be JSONL, one record per evaluated checkpoint or epoch:

```json
{
  "checkpoint": "checkpoints/example.pt",
  "epoch": 2,
  "metric_source": "tail_eval",
  "total_score": 3.08,
  "total_score_no_runtime": 2.51,
  "feasible": 100,
  "avg_runtime": 1.08,
  "tail_weighted_no_runtime": 2.52,
  "tail_cases": {
    "99": {
      "cost_no_runtime": 2.8555,
      "hpwl_gap": 0.784,
      "area_gap": 1.7138,
      "soft_violations": 8
    }
  }
}
```

The promoted checkpoint is the best record under this ordering:

1. Feasible count.
2. `total_score_no_runtime` when full eval is available.
3. Tail weighted no-runtime score when only tail eval is available.
4. Soft violations on weighted tail cases.
5. Raw runtime as a tie-breaker, not the primary metric.

The training loop should not run full evaluator by default every epoch. It may write checkpoint records from explicitly invoked eval scripts or a small configured tail-eval command.

## Dirty Sample Pseudo Targets

Add a target builder that returns both the target rectangles and their provenance:

- `clean_fp_sol`: original clean `fp_sol`.
- `dirty_original`: original dirty `fp_sol`, geometry-only.
- `dirty_repaired`: repaired pseudo target derived from dirty `fp_sol`.
- `dirty_repaired_clean_enough`: repaired target whose remaining soft violations are below the configured threshold.

For dirty samples, the online path is:

1. Convert raw `fp_sol` to a placement.
2. Run the existing repair/constraint cleanup path with deterministic settings.
3. Convert repaired placement back to target rectangles.
4. Measure remaining boundary/group/MIB soft violations.
5. Use the repaired target for order and pairwise labels with controlled low weight.

Initial weighting policy:

- Clean samples: anchor, aspect, order, and pairwise use weight `1.0`.
- Dirty original geometry: anchor/aspect keep the existing low weight, initially `0.25`.
- Dirty repaired pseudo target: order/pairwise use `0.10` to `0.25`.
- Dirty repaired clean-enough target: order/pairwise may use up to `0.35`.
- Dirty repaired target that still has high soft violations contributes no order/pairwise loss.

The implementation should expose these as CLI/env-configurable values, but defaults must be conservative.

## Cache-Compatible Interface

The first wave implements online repaired pseudo targets only. However, the target builder should be shaped so an offline cache can be added later without rewriting training:

```text
sample index + target policy version + repair config hash -> pseudo target record
```

The pseudo target record should be able to store:

- target rectangles
- source/provenance
- remaining soft violation counts
- block count
- repair policy version

The training loop asks the target builder for targets; it should not know whether targets came from online repair or cache.

## HGT Relation Gates

Add relation gates inside HGT layers:

- One learnable gate per canonical relation.
- The gate scales relation messages before destination aggregation.
- Initialize gates close to identity so a fresh model starts near current HGT behavior.
- Log or expose gate values in checkpoints so later diagnostics can see which relation types dominate.

The relation gates preserve the Locality Invariant. They do not add global block attention and do not create edges between unrelated blocks.

## Testing

Add focused tests for:

- Guidance ablation removes only the requested guidance fields.
- `ANCHOR_ONLY` leaves anchor priors but removes pairwise/aspect/priority.
- Checkpoint metric comparison selects by evaluator no-runtime evidence, not validation loss.
- Dirty `fp_sol` can produce a repaired pseudo target and provenance.
- Dirty order/pairwise loss becomes non-zero only for repaired pseudo targets that pass the configured quality threshold.
- HGT relation gates exist for every canonical relation and preserve output shapes.
- Relation gates initialized to identity-like values do not produce NaNs and keep a small HGT forward pass stable.

Run focused tests first, then the full pytest suite before claiming implementation completion.

## Success Criteria

- HGT ablation can identify whether pairwise, aspect, or priority guidance is hurting tail score.
- Promoted checkpoint selection is based on evaluator/tail no-runtime evidence rather than supervised validation loss.
- Dirty samples no longer provide raw dirty order/pairwise labels, but repaired pseudo targets can contribute controlled order/pairwise supervision.
- HGT relation gates are checkpoint-compatible and preserve local typed message passing.
- Teacher residuals and distillation are not implemented in this wave.

## Risks

- Online repair during training may slow HGT training. Keep it configurable and measure throughput on a small run before long remote jobs.
- Repaired pseudo targets may still encode poor geometry if repair fixes soft violations by expanding area too much. Gate order/pairwise weights by remaining soft counts and keep anchor/aspect weight conservative.
- Tail-only checkpoint selection may overfit validation tail if used alone. Full validation remains required before production promotion.
- Relation gates can suppress useful constraint relations if regularized poorly. Start identity-like and inspect gate values after training.

## Out Of Scope

- Teacher residuals.
- Teacher checkpoint loading.
- Teacher distillation losses.
- Global attention or HGT v2 global refinement.
- Changing the Production Solver Path decoder contract.
- Promoting a new checkpoint without full evaluator evidence.

## Self-Review

- Placeholder scan: no TODO or TBD placeholders remain.
- Internal consistency: the first wave includes ablation, checkpoint metric selection, repaired pseudo targets, and relation gates; teacher residuals are consistently excluded.
- Scope check: the design is one implementation slice across optimizer guidance handling, training targets, checkpoint metrics, HGT layers, tests, and docs.
- Ambiguity check: dirty order/pairwise labels come only from repaired pseudo targets that pass a configurable quality threshold, never from raw dirty `fp_sol`.
