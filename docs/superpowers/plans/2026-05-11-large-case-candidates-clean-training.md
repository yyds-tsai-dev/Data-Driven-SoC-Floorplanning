# Large-Case Candidates And Clean Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add large-case relative-order candidate search, clean-sample training skip, and visible `.env` defaults.

**Architecture:** Keep `ArchitectureV4Optimizer` as the Production Solver Path. Introduce small candidate-spec helpers that can later be executed in parallel, but keep the verified production default on the faster adaptive path unless the large-case matrix is explicitly enabled. Reuse existing parser, placement, and repair violation logic for training sample filtering.

**Tech Stack:** Python, PyTorch, pytest, shell scripts, repo-root `.env`.

---

## File Structure

- `src/floorset_arch/optimizer.py`: add candidate specs, opt-in large-case candidate matrix, opt-in repair profiles, and soft-count-first selection.
- `src/floorset_arch/training/losses.py`: add `fp_sol_soft_violations()` and `is_constraint_clean_training_sample()`.
- `src/floorset_arch/training/train.py`: skip soft-violating supervised samples and report skipped count.
- `tests/test_optimizer.py`: cover large-case candidate spec generation and soft-count-first selection.
- `tests/test_model.py`: cover clean/dirty `fp_sol` detection.
- `scripts/eval_total.sh`, `scripts/eval_single.sh`: source `.env` if present.
- `.env`: document every current `FLOORSET_*` environment knob with conservative defaults.
- `docs/optimization-notes.md`: record current large-case candidate policy.

## Tasks

- [ ] Add failing optimizer tests for large-case relative-order candidate specs and soft-count-first selection.
- [ ] Implement candidate specs, deduplicated large-case profiles, opt-in repair profiles, and selection in `optimizer.py`.
- [ ] Add failing training-loss tests for clean and dirty `fp_sol` samples.
- [ ] Implement clean-sample helpers and skip logic in `train.py`.
- [ ] Add `.env` and update eval scripts to source it.
- [ ] Update optimization notes.
- [ ] Run focused tests and full pytest.

## Self-Review

- Spec coverage: every design decision maps to a task.
- Placeholder scan: no TODO/TBD placeholders.
- Type consistency: uses existing `Instance`, `Placement`, `Rect`, and `SolverConfig` types.
