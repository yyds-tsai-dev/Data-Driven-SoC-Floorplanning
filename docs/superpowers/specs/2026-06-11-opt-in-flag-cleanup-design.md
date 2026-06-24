# Opt-In Flag Cleanup Design

## Goal

Inventory and clean up every opt-in flag, environment knob, CLI option, debug toggle, and unused or duplicate helper function across the codebase. Remove only paths with enough evidence that they are deprecated, harmful, unused, or superseded. Keep useful ablation and diagnostic controls explicit and documented.

## Context

This branch already contains the v10 proxy, conditional runtime budget, and narrow grouping work. Before this cleanup, the existing branch changes were committed into focused commits:

- HGT evaluator-backed checkpoint selection.
- Narrow grouping default promotion with v10 budget evidence.
- Removal of superseded v10 evaluation JSON artifacts.
- HGT checkpoint artifact refresh.

The cleanup starts from that clean branch state. During design review, the user
removed `src/arch_old/` and `src/architecture_v4_optimizer.py`; implementation
must treat those deletions as part of the cleanup surface and verify imports,
scripts, tests, and README/docs against the current `architecture_v5_optimizer.py`
wrapper and `ArchitectureV4Optimizer = ArchitectureV5Optimizer` compatibility
alias.

## Decisions

1. Use an evidence-led cleanup rather than aggressive removal or deprecate-only no-ops.
2. Treat the whole codebase as in scope, not only the Production Solver Path.
3. Use full-validation `total_score_no_runtime` as the primary solver architecture decision metric.
4. Use raw runtime, p90 runtime, max runtime, and runtime-aware total as gating or sanity signals.
5. Remove production code paths and tests for flags with full-validation negative evidence when a newer path supersedes them.
6. Keep mixed-evidence paths as explicit ablations when they still have diagnostic value.
7. Update `README.md` through its existing structure, not only by adding a knobs appendix.
8. Keep `CONTEXT.md` glossary-only; do not put implementation inventory there.

## Scope

The inventory covers:

- Solver and runtime flags: candidate generation, repair, quality portfolio, grouping, runtime budget, surrogate guidance, no-guidance modes, beam candidates, and numeric overrides.
- Training CLI and environment controls: encoder options, HGT relation gates, pseudo targets, loss weights, checkpoint output, resume behavior, and stable checkpoint writes.
- Checkpoint selection and promotion controls: metrics manifests, evaluator-best checkpoint paths, promotion helpers, and selection records.
- Debug and trace controls: repair trace output, evaluator refresh, diagnostics, and local smoke-test switches.
- Evaluator plumbing: checkpoint resolution, `.env` loading, runtime/no-runtime evaluator behavior, and script-level options.
- Active code, scripts, tests, docs, and deleted reference or wrapper paths.

`src/arch_old` has been deleted by the user. The cleanup must confirm it has no
active imports, remove stale README references, and record it as reference-code
removal rather than refactoring duplicate helpers in place.

`src/architecture_v4_optimizer.py` has also been deleted. Because this path was a
contest-facing wrapper historically, the cleanup must verify that current eval
and validation scripts use `src/architecture_v5_optimizer.py`, that tests can
continue importing the package-level compatibility alias, and that README/docs
no longer describe v4 as the active wrapper.

## Inventory

Create `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md` as the audit trail. Each entry should include:

- name
- kind: env flag, CLI option, numeric override, debug trace, function/helper
- owner or module
- default behavior
- caller path
- evidence source
- metric impact, when relevant
- decision
- action taken

Decision values:

- `keep default`: validated default behavior.
- `keep explicit ablation`: mixed evidence or diagnostic value, not a production recommendation.
- `remove`: delete the runtime code path, tests that only protect the removed behavior, and user-facing usage docs.
- `docs/history only`: keep historical evidence but no live usage path.
- `needs validation`: production-impact behavior with insufficient evidence.

## Evidence Rules

Solver-impact decisions require full-validation evidence when changing production behavior. Existing trusted docs and artifacts count; do not rerun expensive validation when the current evidence already answers the question.

Training, checkpoint, debug, and evaluator controls use targeted tests, CLI smoke checks, manifest tests, and caller checks unless they change solver evaluation output.

Debug and trace flags should be kept only if they do not alter solver output and still have a caller or documented use. Dead trace paths should be removed.

Numeric overrides stay only when their owning path stays. Overrides under removed paths are removed with that path.

Helper functions are removed or consolidated only after AST inventory, Semble/graphify caller search, exact string search, and targeted tests confirm the decision.

## Initial High-Priority Candidates

The first expected cleanup candidates are:

- Broad `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS`: full-validation regression and superseded by default-on Narrow Grouping Pair Bias.
- Hard `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP`: full-validation over-clamping and superseded as the preferred direction by Conditional Runtime Budget.
- `_build_candidate_worker`: static caller check currently suggests it is unused.
- `_proxy_cost`: static caller check currently suggests it is unused.
- `_hard_or_overlap_regressed`: static caller check currently suggests it is unused.
- Duplicate runtime clamp helpers in `budget_layer.py` and `repair.py`: remove if the hard clamp path is removed.
- Deleted `src/arch_old/`: verify no active callers remain and clean README/docs references.
- Deleted `src/architecture_v4_optimizer.py`: verify scripts and docs point at the v5 wrapper or compatibility alias.

These are only starting points. The implementation must still inspect all other flags and helpers before declaring the cleanup complete.

## Removal Policy

When removing a deprecated opt-in, remove the whole live path:

- production branch or helper
- direct tests for the deprecated behavior
- README usage instructions
- numeric overrides that only affect that path

Keep historical evidence in evaluation docs so the team does not repeat the same experiment.

For mixed evidence, keep the code and tests but label it as ablation-only or debug-only in README and the cleanup inventory.

For duplicate helpers, prefer the established shared modules:

- budget and gating decisions in `budget_layer.py`
- v10 proxy and hard legality logic in `v10_proxy.py`
- evaluator-facing scoring in evaluator/scoring helpers

Do not consolidate duplicate-looking code if it belongs to reference-only legacy code and has no active caller.

## Testing

Use three verification layers:

1. Static and caller verification:
   AST inventory, Semble/graphify search, exact string search, and import/caller checks.
2. Targeted tests:
   Run pytest slices for changed modules, especially budget layer, relative order, repair, optimizer, checkpoint promotion, training, model, evaluator scoring, and diagnostics.
3. Evaluator validation:
   Run solver smoke or full validation only for production-impact flags whose evidence is insufficient after the inventory.

At the end, run `graphify update .` because code will be modified.

## README Update

Update `README.md` according to its existing structure:

- refresh project overview and current production path if cleanup changes defaults
- refresh score policy and validation language
- refresh script descriptions and CLI/env examples
- refresh training and checkpoint promotion sections
- refresh active module architecture
- remove instructions for deleted flags
- mark retained controls as production default, ablation only, debug only, training only, or evaluator plumbing

If a new cleanup result does not fit the current README structure, add a new concise section.

## Out Of Scope

- New solver algorithm experiments.
- GNN architecture changes.
- Checkpoint retraining.
- Changing v10 metric policy.
- Hard-coding validation IDs into production logic.

## Self-Review

- The design covers the whole codebase, not only solver flags.
- The README requirement updates existing sections instead of adding only a knobs appendix.
- The removal threshold uses evidence and avoids deleting mixed-evidence ablations blindly.
- The inventory has a concrete output path and decision vocabulary.
- The design keeps `CONTEXT.md` glossary-only.
