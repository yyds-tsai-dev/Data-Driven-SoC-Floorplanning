# Opt-In Flag Cleanup Audit

## Decision Policy

- Solver promotion/removal uses full-validation `total_score_no_runtime` first.
- Raw runtime, p90 runtime, max runtime, and runtime-aware total are gating or sanity signals.
- Training, checkpoint, debug, and evaluator controls are verified with targeted tests or CLI smoke checks unless they change solver evaluation output.
- Mixed-evidence controls remain explicit ablations.
- Removed controls keep historical evidence in docs, but no live usage instructions.

## Inventory

| Name | Kind | Owner | Default | Caller / Source | Evidence | Metric impact | Decision | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `src/arch_old/` | reference directory | legacy/reference | absent | no active imports expected | user removed; verify by exact search | not part of Production Solver Path | remove | accept deletion and remove README tree references |
| `src/architecture_v4_optimizer.py` | wrapper | evaluator wrapper | absent | current scripts use `src/architecture_v5_optimizer.py` | exact search and script check | v5 wrapper is current eval path | remove | accept deletion and update README/docs |
| `FLOORSET_ENABLE_RUNTIME_TAIL_CLAMP` | env flag | solver repair budget | removed | historical docs only | 2026-06-05 full validation over-clamped quality | no-runtime regressed to `2.2741`; total regressed to `3.0120` | remove | deleted hard clamp live path and tests |
| `FLOORSET_ENABLE_GROUPING_ADJACENCY_BIAS` | env flag | relative-order decoder | removed | historical docs and stale-flag guard only | 2026-06-05 full validation regressed; narrow grouping supersedes it | no-runtime regressed to `2.4989`; total regressed to `3.3102` | remove | deleted broad key blend, chaining path, and tests |
| `FLOORSET_ENABLE_NARROW_GROUPING_PAIR_BIAS` | env flag | relative-order decoder | on | `relative_order.py`, tests | 2026-06-10 full validation improved no-runtime and total | no-runtime `2.1538`, total `2.6993` | keep default | retain `=0` ablation switch |
| `FLOORSET_ENABLE_V10_SOFT_REPAIR` | env flag | repair | off | `budget_layer.py`, `repair.py`, tests | mixed full-validation evidence | no-runtime can improve, runtime tail regresses without budget | keep explicit ablation | document as ablation-only |
| `FLOORSET_ENABLE_CONDITIONAL_RUNTIME_BUDGET` | env flag | repair budget | off | `budget_layer.py`, `repair.py`, tests | improves soft-repair runtime-aware path but not no-runtime default | total/runtime useful, no-runtime not promoted | keep explicit ablation | document as ablation pair with v10 soft repair |
