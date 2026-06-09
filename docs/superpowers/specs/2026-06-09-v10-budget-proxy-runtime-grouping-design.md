# V10 Budget Proxy Runtime Grouping Design

## Goal

Improve ICCAD 2026 v10 validation score by changing the solver's local acceptance surface from soft-first selection to a shared V10 no-runtime proxy, then adding opt-in controls for conditional runtime budget and narrow grouping pair bias.

The design intentionally does not change the GNN, checkpoint, decoder model heads, or broad quality portfolio defaults.

Baseline evidence:

- Graph Transformer 0521 baseline: no-runtime `2.1194`, total `2.6641`, max runtime `6.80s`.
- `quality_portfolio=auto`: no-runtime regressed to `2.2446`.
- `v10_soft_repair=1`: no-runtime improved slightly to `2.1082`, but total regressed to `2.7881` and max runtime increased to `9.47s`.
- `runtime_tail_clamp=1`: lowered p90 runtime but regressed no-runtime to `2.2741`.
- `grouping_adjacency_bias=1`: regressed no-runtime to `2.4989`.

## Context

The current no-runtime score driver is:

```text
no-runtime score = quality_factor(HPWL + area) * exp(2 * soft_vrel)
total score      = no-runtime score * runtime_adjustment
```

The largest baseline contributors show mixed HPWL, area, and soft pressure rather than pure soft-violation failure:

| ID | blocks | cost_no | HPWL | area | vrel | Main issue |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 99 | 120 | 1.925 | 0.417 | 0.803 | 0.090 | high area |
| 98 | 119 | 2.021 | 0.651 | 0.684 | 0.096 | high HPWL and area |
| 93 | 114 | 2.777 | 0.768 | 1.003 | 0.194 | high HPWL, area, and soft |
| 97 | 118 | 1.828 | 0.472 | 0.622 | 0.083 | moderate |
| 95 | 116 | 1.961 | 0.565 | 0.608 | 0.106 | runtime tail also bad |

The existing `_candidate_rank()` default is soft-first. The existing v10 soft repair acceptance also allows substantial geometry proxy slack when soft decreases. That can accept placements whose soft count improves while HPWL or bbox area regresses enough to hurt the v10 no-runtime score.

## Decisions

1. Use option B: design all three phases, but promote only Phase 1 as the default behavior.
2. Make Phase 1 V10 proxy selection and acceptance the production default.
3. Preserve an environment override for the old soft-first candidate rank policy as an ablation path.
4. Keep Phase 2 Conditional Runtime Budget behind `FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET=1`.
5. Keep the old hard runtime clamp flag as ablation only; do not automatically apply it in the new Phase 2 path.
6. Keep Phase 3 Narrow Grouping Pair Bias behind `FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS=1`.
7. Do not touch GNN architecture, checkpoint promotion, broad quality portfolio defaults, or validation-ID-specific production logic.

## Architecture

### Phase 1: V10 Proxy Selection and Acceptance

Add a scoring-oriented helper module, such as `src/floorset_arch/v10_proxy.py`, that owns the shared local acceptance surface:

```text
Hard Legality Gate
  -> V10 No-Runtime Proxy
  -> soft tie-break
  -> HPWL / area tie-break
```

The helper should provide:

- a hard-legality summary or rank key
- no-runtime proxy cost
- proxy-first better-than comparison
- soft tie-break comparison
- HPWL and bbox area fallback tie-break comparison

The proxy should approximate the evaluator-facing no-runtime score, using geometric quality and `exp(2 * soft_vrel)`. It remains a local selection proxy, not a replacement for full evaluator validation.

`optimizer.py` should use the helper for `_candidate_rank()` and quality refine acceptance. The default candidate rank policy becomes proxy-first. The old `soft_first` rank stays available through an environment override.

`repair.py` should use the helper for `_score_better_v10_soft()`. A repair trial is accepted when:

- hard legality does not regress
- the V10 no-runtime proxy clearly improves, or
- the proxy is effectively tied within `0.1%` and soft tie-breaks prefer the trial

If the proxy regresses by more than `0.1%`, the trial is rejected even when soft violations decrease.

Quality refine is stricter than repair: it should accept proxy improvements and proxy ties with better soft tie-breaks, but it should not accept proxy regression.

### Phase 2: Conditional Runtime Budget

Add an opt-in conditional runtime policy that preserves baseline repair and stops only extra repair paths or profile expansion when they are not producing accepted proxy improvements.

The budget is local to each candidate and repair profile. It is not a global `solve()` timeout and does not let one slow candidate suppress another unrelated candidate.

Each trace row in the existing `FLOORSET_REPAIR_TRACE_JSONL` stream should include:

- `runtime_budget.attempts`
- `runtime_budget.accepted`
- `runtime_budget.elapsed_ms`
- `runtime_budget.stop_reason`
- `runtime_budget.tier`
- `runtime_budget.extra_path`

The old `_runtime_tail_clamped_config()` behavior remains available only when its existing hard clamp flag is explicitly enabled for ablation. The new conditional budget path should not automatically apply hard clamp limits.

### Phase 3: Narrow Grouping Pair Bias

Add an opt-in narrow pairwise grouping bias in `relative_order.py`.

It applies only when all conditions are true:

- the pair is in the same cluster
- the cluster has high cluster-level grouping pressure
- the pair is axis-ambiguous, for example `abs(h_score - v_score) <= 0.15`
- the narrow grouping flag is enabled

When enabled, the pair loop gives a small axis bonus toward the cluster orientation. It does not globally blend all cluster keys and does not chain the entire group by default.

This replaces the broad grouping-bias direction for future promotion decisions because global key blending and default chaining already showed no-runtime regression.

## Data Flow

Phase 1 data flow:

```text
placement metrics
  -> hard legality summary
  -> V10 no-runtime proxy
  -> rank or acceptance decision
  -> selected candidate / accepted repair / accepted refine
```

Phase 2 data flow:

```text
candidate or repair profile
  -> extra repair attempt
  -> V10 proxy acceptance decision
  -> attempts / accepted / elapsed update
  -> continue or stop extra path
  -> append trace to existing repair JSONL
```

Phase 3 data flow:

```text
raw anchors and shapes
  -> cluster-level grouping pressure
  -> same-cluster ambiguous pair detection
  -> small axis bonus
  -> relative-order graph constraints
```

## Error Handling

Proxy calculation failures should reject the trial or rank the candidate behind hard-legal proxy-scored candidates. They should not raise during evaluator runs.

Missing block data, fixed-shape mismatches, preplaced position or dimension mismatches, area violations, and overlaps should fail the Hard Legality Gate or rank behind hard-legal alternatives.

Conditional runtime trace write failures should not change solver output.

Invalid environment variable values should fall back to conservative defaults.

When Phase 2 and Phase 3 flags are disabled, their behavior should not change solver output. Phase 1 is the intentional default behavior change.

## Testing

Unit tests for the V10 proxy helper should cover:

- hard legality outranks soft or proxy improvements
- proxy improvement is accepted
- proxy ties within `0.1%` use soft tie-breaks
- proxy regression beyond `0.1%` is rejected even when soft improves
- missing blocks, overlaps, fixed-shape mismatch, and preplaced mismatch fail or rank behind legal candidates

Optimizer tests should cover:

- default candidate rank is proxy-first
- an environment override can restore `soft_first`
- quality refine does not accept proxy regression

Repair tests should cover:

- `_score_better_v10_soft()` rejects soft-improving proxy regression
- proxy ties can be accepted by soft tie-breaks
- hard or overlap regression is rejected

Conditional runtime budget tests should cover:

- trace rows include attempts, accepted, elapsed, stop reason, tier, and extra path
- rejected extra attempts stop only the local candidate or repair profile path
- baseline normal repair is preserved
- the hard clamp flag remains available for explicit ablation

Narrow grouping pair bias tests should cover:

- no global cluster key blend in the narrow path
- no default whole-group chain in the narrow path
- only same-cluster ambiguous pairs under cluster-level pressure receive a bonus
- pairs outside the ambiguity margin are unchanged

Full validation should run in sequence:

1. Phase 1 default against Graph Transformer 0521 baseline.
2. Phase 1 plus Phase 2 opt-in.
3. Phase 1 plus Phase 2 and Phase 3 opt-in.

Each run should report no-runtime score, total score, feasible count, average runtime, p90 runtime, max runtime, top no-runtime contributors, and repair trace acceptance summaries.

## Out Of Scope

- GNN architecture changes.
- New checkpoint promotion.
- Broad quality portfolio default promotion.
- Global runtime timeout.
- Hard runtime clamp promotion.
- Broad cluster key blending promotion.
- Whole-group chaining by default.
- Validation-ID-specific production logic.

## Self-Review

- The design has one default behavior change: Phase 1 proxy-first selection and acceptance.
- Runtime and grouping changes remain opt-in until full-validation evidence supports promotion.
- The design separates evaluator-facing no-runtime score from the solver-internal V10 no-runtime proxy.
- The design preserves baseline repair while allowing conditional stopping of extra paths.
- No phase requires touching the GNN or training pipeline.
