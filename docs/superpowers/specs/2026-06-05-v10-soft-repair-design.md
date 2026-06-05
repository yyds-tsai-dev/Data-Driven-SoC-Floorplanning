# v10 Risk-Gated Soft Repair Design

## Goal

Improve ICCAD 2026 v10 submission score by adding an opt-in repair acceptance path that targets boundary, grouping, and MIB soft-constraint regressions without changing the production checkpoint, decoder, or default runtime behavior.

The baseline is the current Graph Transformer 0521 checkpoint:

- v10 no-runtime total: `2.1194`
- v10 total: `2.6641`
- feasible count: `100/100`
- average runtime: `1.23s`
- p90 runtime: `2.23s`

The first implementation is experimental and must stay behind `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` until full-validation evidence shows it beats the baseline.

## Context

The v10 scoring update makes fixed-shape and preplaced violations hard infeasibility while keeping boundary, grouping, and MIB as soft constraints inside an exponential penalty. Current Graph Transformer 0521 validation is already `100/100` feasible, so the highest-leverage safe target is not hard legality but v10-weighted soft pressure on medium-large and large cases.

The current largest 116-120 bucket is not the only decision surface under `exp(n/12)` weights. Several medium-large cases, such as IDs 93, 89, 83, 78, 70, and 61, have high soft pressure or high no-runtime contribution. The repair change must therefore use reusable v10 risk signals rather than validation IDs or a `block_count >= 118` gate.

## Decisions

1. Keep Graph Transformer 0521 as the production default checkpoint.
2. Keep `FLOORSET_ENABLE_QUALITY_PORTFOLIO` opt-in; the Graph Transformer 0521 follow-up run regressed no-runtime and total score.
3. Add the first repair experiment behind `FLOORSET_ENABLE_V10_SOFT_REPAIR=1`.
4. Gate the repair per instance with the existing v10 risk budget.
5. Do not enable heavy high-risk repair profiles by default.
6. Do not change decoder ranking, grouping adjacency, or model training in this phase.
7. Defer Runtime-Tail Budget Clamp until after this repair acceptance experiment.
8. Defer Decoder-Side Grouping Adjacency Bias until after repair acceptance and runtime-tail budget work.

## Architecture

The production solver flow remains:

```text
Graph Transformer guidance
  -> relative-order placement
  -> repair_placement()
  -> candidate ranking
  -> evaluator output
```

This design modifies only the repair acceptance surface. The implementation should reuse existing repair primitives in `src/floorset_arch/repair.py`, especially `soft_violation_counts()`, `_geometry_quality_proxy()`, `_guarded_soft_repair()`, and existing hard legality checks. It should avoid a new solver branch.

The high-level control should be:

```text
if FLOORSET_ENABLE_V10_SOFT_REPAIR is off:
    use current repair behavior
else:
    compute instance_risk_budget(inst)
    decide whether the current placement has enough soft pressure
    run bounded v10 soft repair only when both gates pass
    accept only placements that satisfy v10 soft-aware acceptance rules
```

## Enablement

`FLOORSET_ENABLE_V10_SOFT_REPAIR` is the global opt-in flag.

Per-instance behavior:

- `BudgetTier.MEDIUM` and `BudgetTier.HEAVY`: eligible.
- `BudgetTier.LIGHT`: eligible only when the current placement has high soft pressure.
- `BudgetTier.NONE`: not eligible.

The light-tier soft-pressure threshold should be explicit and environment-overridable. A reasonable first default is to enable when current soft total is at least `6` or current relative soft pressure is at least `0.12`.

## Acceptance Rules

The candidate repair result must never worsen hard legality or overlap. A result with hard infeasibility or increased overlap must be rejected.

For soft and geometry trade-offs:

- If soft total decreases and grouping does not get worse, allow `_geometry_quality_proxy()` to regress by at most `8%`.
- If grouping violations decrease, allow `_geometry_quality_proxy()` to regress by at most `12%`.
- If only boundary violations decrease, allow `_geometry_quality_proxy()` to regress by at most `6%`.
- If soft total does not decrease, require `_geometry_quality_proxy()` to improve by at least `0.01%`.
- If MIB improves without grouping or boundary improvement, treat it like general soft total improvement and use the `8%` slack.

These slack values should be environment-overridable for ablation, but the defaults should remain conservative.

## Data Flow

Inputs:

- `Instance`
- current `Placement`
- `SolverConfig`
- `RiskBudget`
- environment flags and slack overrides

Metrics:

- boundary violation count
- grouping violation count
- MIB violation count
- total soft count
- overlap count or equivalent hard-legality signal
- `_geometry_quality_proxy()`

Output:

- either the original placement or a repaired placement accepted under the v10 soft-aware rules.

The accepted placement then continues through the existing candidate ranking path. Candidate ranking remains soft-first for this phase.

## Error Handling

The repair path must degrade gracefully:

- If risk-budget calculation fails, skip the opt-in v10 soft repair and use current repair behavior.
- If proxy scoring fails, reject the trial and keep the original placement.
- If a trial placement is missing blocks, violates preplaced immutability, violates fixed-shape dimensions, or creates overlap, reject it.
- The opt-in path must not raise during evaluator runs; failures should fall back to the baseline repair output.

## Testing

Unit tests should cover:

- `FLOORSET_ENABLE_V10_SOFT_REPAIR` defaults to off.
- medium/heavy risk instances are eligible when the flag is on.
- light risk instances require soft pressure before eligibility.
- none risk instances remain ineligible.
- grouping improvement allows larger proxy slack than boundary-only improvement.
- hard or overlap regression is rejected even when soft improves.
- no-soft-improvement trials require geometry proxy improvement.
- environment overrides for slack and soft-pressure thresholds are honored.

Full validation should compare against Graph Transformer 0521 baseline:

- feasible count remains `100/100`
- v10 no-runtime total improves below `2.1194`
- v10 total does not materially regress
- average and p90 runtime do not materially regress
- top no-runtime contributors and soft deltas are reported

## Out Of Scope

- Promoting a new checkpoint.
- Enabling `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` by default.
- Re-enabling heavy high-risk repair profiles by default.
- Decoder-side grouping adjacency changes.
- Runtime-tail budget clamp.
- Any validation-ID-specific production behavior.

## Self-Review

- No implementation detail requires a new solver branch.
- The feature is opt-in and conservative by default.
- The design uses v10 full-validation evidence instead of old tail-dominance assumptions.
- Follow-up work is ordered but not included in this phase.
