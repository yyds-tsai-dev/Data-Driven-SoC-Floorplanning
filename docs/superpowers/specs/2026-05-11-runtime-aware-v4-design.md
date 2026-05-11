# Runtime-Aware V4 Evaluation Design

## Goal

Make local architecture tuning reflect placement quality first, while preserving runtime-aware contest scoring for final risk checks, then rename the active contest wrapper generation from architecture v3 to architecture v4 and load repo-root `.env` through `python-dotenv`.

## Context

The contest cost formula is correct in `iccad2026_evaluate.py`: quality gap and soft violations are multiplied by a runtime adjustment. The local full evaluator, however, estimates runtime factor as each validation case runtime divided by the median runtime from the same local validation run. The official contest runtime factor is different: it is each hidden case runtime divided by the median runtime of all submissions for that same hidden case.

That local denominator mixes case-size effects into score. Large validation cases tend to run slower than the solver's own median case, so local runtime-aware score can over-penalize large-case strategies even though official scoring compares large cases against other large-case submissions.

## Decisions

1. Keep the official `compute_cost()` formula and keep local runtime-aware score available.
2. Add an explicit no-runtime cost path by making runtime adjustment optional in `compute_cost()`.
3. During `--evaluate`, record both per-case local-runtime cost and per-case no-runtime cost.
4. Report both total scores in CLI/JSON output: local runtime-aware total remains the official-like local score, and no-runtime total becomes the primary architecture-tuning metric.
5. Add raw runtime summary statistics, at least average, median, p90, max, and selected block-count tail cases when available.
6. Treat `--score` saved-solution evaluation as no-runtime unless stored runtimes are intentionally supplied and recomputed later.
7. Rename the active contest wrapper generation from architecture v3 to architecture v4 across wrapper file, optimizer class, tests, scripts, package export, training labels, `CONTEXT.md`, and current docs.
8. Add `python-dotenv` to project dependencies and load repo-root `.env` from Python entrypoints so local runs and scripts share one environment-default mechanism.

## Architecture

`floorset_arch` remains the Production Solver Path. The version rename affects the contest-facing wrapper and active optimizer class name only: `ArchitectureV4Optimizer` replaces `ArchitectureV3Optimizer`, while reusable solver modules remain under the existing `floorset_arch` package.

The evaluator will carry two score channels:

- `cost`: local runtime-aware cost, using the current local median runtime recomputation.
- `cost_no_runtime`: quality and soft-violation cost with runtime adjustment fixed to `1.0`.

The total score output will mirror this split:

- `total_score`: local runtime-aware total for continuity with current evaluator output.
- `total_score_no_runtime`: primary local architecture-tuning total.

This keeps final submission checks honest about runtime risk without allowing local runtime median artifacts to dominate architecture choices.

## Documentation Updates

`CONTEXT.md` should define Architecture v4 and no-runtime quality score as project language. Optimization notes should state that previous v3 runtime-weighted ablations are not sufficient evidence against large-case quality improvements unless no-runtime score and raw runtime are also reviewed.

Existing design and plan docs should be updated only where they describe the current active architecture or current scoring policy. Historical execution notes may keep v3 references when they clearly refer to past work.

## Testing

- Unit-test `compute_cost(..., use_runtime=False)` returns the quality/violation product without runtime adjustment.
- Unit-test `--evaluate` result objects can carry both runtime-aware and no-runtime costs.
- Unit-test total no-runtime score calculation uses `cost_no_runtime` values and the same exponential block-count weighting.
- Unit-test dotenv loading through the v4 wrapper or optimizer entry path without requiring shell sourcing.
- Update existing optimizer/tests imports from `ArchitectureV3Optimizer` to `ArchitectureV4Optimizer`.
- Run focused scoring/optimizer tests, then the full pytest suite.

## Success Criteria

- Full local evaluation reports both runtime-aware and no-runtime totals.
- Architecture comparison docs prioritize feasibility, large-case no-runtime score, total no-runtime score, soft violations, HPWL/area, raw runtime, then local runtime-aware score.
- No artificial sleeping or runtime calibration is reintroduced.
- `architecture_v4_optimizer.py` is the active wrapper used by eval scripts.
- `.env` can be loaded by Python through `python-dotenv`.

## Self-Review

- Placeholder scan: no placeholders remain.
- Internal consistency: runtime-aware scoring is preserved while no-runtime scoring is made primary for tuning.
- Scope check: one implementation slice across evaluator scoring, version naming, dotenv dependency, tests, and docs.
- Ambiguity check: v4 rename is limited to active wrapper/class labels, not the `floorset_arch` package name.
