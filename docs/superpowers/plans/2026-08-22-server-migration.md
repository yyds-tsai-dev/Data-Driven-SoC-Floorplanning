# Repository Consolidation and Server Migration Plan

> **For agentic workers:** Execute inline in the current session. Do not restart the session or dispatch subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current research state, commit every human-authored worktree change in focused commits, consolidate the active solver branches into `main`, verify the consolidated tree, and push what Git can carry before moving to a new server.

**Architecture:** Treat `5.6-sol-reduce-time` as the integration trunk because it already contains `5.6-sol-flow-matching` and patch-equivalent copies of five of six `worktree-agent*` feature commits. Preserve unique branch history through explicit merges, preserve small local evidence in focused commits, and keep generated datasets/checkpoints outside ordinary Git with an explicit transfer manifest.

**Tech Stack:** Git worktrees, Python 3.12, `uv`, pytest, repository validation scripts, Git LFS for existing root checkpoint files, Graphify, and Markdown/JSON handoff artifacts.

## Global Constraints

- Do not restart or create another model session and do not dispatch subagents.
- Preserve unrelated edits and never use `git reset --hard`, force-push, or destructive worktree cleanup.
- Generated checkpoints, evaluator artifacts, W&B runs, and `FloorSet/floorset_lite` remain outside ordinary Git; record their locations and transfer them separately.
- Keep completed experiment decisions final unless new evidence explicitly authorizes a new gate; do not rerun killed/no-go branches as if they were unfinished implementation.
- Record every failed, skipped, or unexecuted verification honestly.
- Push only after checking branch ancestry, final status, and fresh verification output.

---

### Task 1: Inventory and baseline

**Files:**
- Read: `.superpowers/sdd/progress.md`
- Read: `docs/experiments/2026-08-14-g0-v2-transient-fp.md`
- Read: `docs/superpowers/specs/2026-08-13-tfdl-data-free-topology-prior-design.md`
- Create: `docs/project-status/2026-08-22-server-migration-handoff.md`

- [x] Confirm `get_goal` state: no active goal exists in the goal service.
- [x] Inventory all registered worktrees, branches, dirty paths, branch ancestry, patch-equivalent commits, local-only generated data, and remote configuration.
- [x] Run the pre-merge full suite and record the exact result: 2,232 passed, 5 skipped, 17 failed in `tests/test_icdc_topology_prior.py` because the frozen scorer module resolves to the FloorSet submodule copy rather than `scripts/iccad2026_evaluate.py`.
- [x] Run syntax/format checks on newly discovered scripts, JSON, tar, and xlsx inputs, plus the focused backend suite: 63 passed.

### Task 2: Preserve uncommitted human-authored work

**Files:**
- Add: `docs/official/beta_test/cadc1013.{tar.gz,xlsx}`
- Add: `docs/research/{前端報告,後端報告}.md`
- Add: `partner/icdc/causal_smoke.py`
- Add: `scratchpad/` experiment programs, harnesses, and small evidence files
- Modify: `.gitignore`

- [x] Commit official beta materials independently from reports.
- [x] Commit frontend/backend reports independently from solver and experiment code.
- [x] Commit topology causal-smoke/probe code independently from its JSON evidence.
- [x] Commit general runtime experiment harnesses independently.
- [x] Preserve small refine-kernel and anytime-ladder worktree evidence on their owning branches.
- [x] Ignore generated nested checkpoint run directories without changing existing tracked checkpoint behavior.

### Task 3: Consolidate branch history

**Branches:**
- Integration trunk: `5.6-sol-reduce-time`
- Already-contained ancestor: `5.6-sol-flow-matching`
- Unique experiment branches: `feat/column-split-merge-g0`, `feat/constructive-g0-g1`, `feat/frame-reinsert-g0`
- Agent branches: six `worktree-agent-*` branches
- Destination: `main`

- [ ] Merge the unique refine-kernel agent branch and its evidence into the integration trunk.
- [ ] Merge the five patch-equivalent agent branches so their branch ancestry and any newly preserved evidence are reachable.
- [ ] Merge all three unique `feat/*` experiment branches, preserving their no-go decisions and resolving conflicts in favor of the newest validated integration behavior.
- [ ] Confirm `5.6-sol-flow-matching` is still an ancestor; record it as integrated without manufacturing a redundant content change.
- [ ] Fast-forward `main` to the integration trunk, then verify every named worktree head is reachable from `main`.

### Task 4: Final handoff and verification

**Files:**
- Create: `.planning/HANDOFF.json`
- Create: `.planning/.continue-here.md`
- Update: `docs/project-status/2026-08-22-server-migration-handoff.md`
- Update: `graphify-out/` through `graphify update .` only

- [ ] Record final commit hashes, merges, known failures, unexecuted G1 work, generated-data transfer commands, and the exact first resume action.
- [ ] Run targeted tests for every newly merged experiment branch.
- [ ] Run `uv run pytest`, `bash scripts/validate.sh`, `git diff --check`, and `graphify update .`; distinguish pre-existing failures from merge regressions.
- [ ] Confirm no staged/untracked human-authored files remain in any worktree.
- [ ] Push `main`; if authentication is unavailable, preserve all commits locally and record the exact credential blocker and retry command.
