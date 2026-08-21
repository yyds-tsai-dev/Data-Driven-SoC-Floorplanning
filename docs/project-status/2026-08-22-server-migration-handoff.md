# Server migration handoff — 2026-08-22

## Executive state

The active development history is on `5.6-sol-reduce-time`, 466 commits ahead of the pre-consolidation `main` at `83f3067`. The goal service reports no active goal, so there is no API-owned goal to pause. This document is the durable pause/resume record for Codex, Claude Code, or a human on the new server.

The source tree contains substantial completed solver and experiment work. Do not recreate completed code or rerun killed experiment directions by default. The currently unfinished research gate is the topology-prior G1 execution, not its supporting implementation.

## Completed work — do not repeat

- Partner candidate foundation and retrieval infrastructure were implemented and reviewed. The R4 retrieval gate failed; CP-SAT continuation remains blocked. See `.superpowers/sdd/progress.md` and `docs/experiments/2026-07-23-retrieval-r4-gate.md`.
- Flow-matching, DPM-Solver++ sampling, runtime frontier work, fast setup, GPU arm, SA kernel, anytime ladder, and early-exit implementations are already included in `5.6-sol-reduce-time`. `5.6-sol-flow-matching` is an ancestor of that branch.
- Fast grouping bridge Tasks 1–4 completed. Task 5's fresh package gate was not run; it still requires an actual, non-projected average runtime at or below 0.300 seconds and 100/100 feasibility.
- LP-free changed-contact DAG bridge Track A was killed at G0. Do not run G1 or package work for it. See `docs/experiments/2026-08-13-lp-free-changed-contact-dag-bridge-gates.md`.
- Track B topology teacher/runtime/compiler/evidence infrastructure, SQLite streaming spool hardening, G0-v2 runner, training-label split, same-shape topology-prior training code, retained-gain audit, exact 3D/3F G1 runtime, and sealed G1 evidence tooling are implemented.
- Formal G0-v2 completed over 21,081 eligible heldout rows with `TARGET_GAIN_MET`; `Delta_H=0.18577722262568797`, about 7.11 times the target. Do not rerun G0 unless source identities change. See `docs/experiments/2026-08-14-g0-v2-transient-fp.md`.
- The frontend/backend external exchange reports are written and preserved in `docs/research/`.
- The column split-merge, constructive G0/G1, and frame-reinsert branches record no-go experiment outcomes; their source/probe history is being preserved for reproducibility, not promoted as production policy.

## Unfinished or untested work

### Topology-prior G1 execution

The implementation exists, but the following execution/evidence sequence is not complete:

1. Export the deterministic receipt-bound training-split sparse-label corpus using the already approved two-slot teacher semantics.
2. Train and heldout-select the same-shape Direct topology student using `partner/icdc/train_topology_prior.py`.
3. Run the retained-teacher-gain audit and require the frozen retention threshold.
4. Run the non-validation causal smoke and bind the exact candidate identity.
5. Freeze `g1_n100_3d3f` and run the one authorized blinded 3-Direct/3-Flow control/candidate pair.
6. Adjudicate the sealed pair. A pass may record `HIGH_TAIL_CAUSAL_PROOF` and must immediately stop as `STOP_REQUIRES_SEPARATE_APPROVAL`; it does not authorize G2 or packaging.
7. Create `docs/experiments/2026-08-13-tfdl-data-free-topology-prior-gates.md` only after G1 adjudication.

Read `docs/superpowers/specs/2026-08-13-tfdl-data-free-topology-prior-design.md` and `docs/superpowers/plans/2026-08-13-tfdl-data-free-topology-prior.md` before running any of these commands. Preserve the exact 3D/3F portfolio and all identity/receipt gates.

### Known failing verification

Pre-merge `uv run pytest` on 2026-08-22 collected 2,254 tests and ended with:

- 2,232 passed
- 5 skipped
- 17 failed
- elapsed 1,023.09 seconds

Every failure is in `tests/test_icdc_topology_prior.py`. The frozen scorer contract loads `FloorSet/iccad2026contest/iccad2026_evaluate.py`, while the tests and `_SCORER_PATH` require `scripts/iccad2026_evaluate.py`. The observable failure is `ValueError("scorer_sha256")`; one direct assertion shows the mismatched module paths. This is a pre-existing integration/environment import-order failure and must be fixed or explicitly re-baselined before claiming the full suite is green.

The focused external-report/backend contract suite passed 63 tests, and all newly found Python, shell, JSON, tar, and xlsx files passed syntax/container validation before commit.

## Local-only generated state that Git will not migrate

Do not add these directories to ordinary Git. They are generated, ignored by repository policy, or far above GitHub's practical limits.

- Main worktree `artifacts/`: about 57 GB. This includes the immutable formal G0 evidence (`artifacts/icdc_g0_v2_area_full/`, about 946 MB) and topology work (`artifacts/icdc_topology/`, about 821 MB).
- Main worktree `checkpoints/`: about 1.8 GB, excluding the separate flow worktree runs below.
- Main worktree `wandb/`: about 624 MB.
- `FloorSet/floorset_lite/`: 9,000 files, about 24 GB, untracked inside the FloorSet submodule.
- Flow-matching worktree nested checkpoint runs: about 246 GB total. The largest individual files are about 1.72 GB, and nested paths do not use the repository's existing root-only Git LFS rule.

Transfer required local state before retiring this server. From the destination server, use a resumable transport appropriate for the environment, for example:

```bash
rsync -aH --info=progress2 --partial \
  OLD_SERVER:/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/artifacts/ artifacts/
rsync -aH --info=progress2 --partial \
  OLD_SERVER:/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/FloorSet/floorset_lite/ FloorSet/floorset_lite/
rsync -aH --info=progress2 --partial \
  OLD_SERVER:/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/ checkpoints/flow-worktree-archive/
```

Only transfer the 246 GB flow archive if historical training snapshots are still required. The source and recipes are in Git; periodic checkpoints are generated evidence.

## Branch/worktree consolidation facts

- `5.6-sol-flow-matching`: zero unique commits relative to `5.6-sol-reduce-time`; already integrated by ancestry.
- `worktree-agent-a438…`, `a8b…`, `a940…`, `ae70…`, and `af25…`: their feature commits are patch-equivalent to commits already on `5.6-sol-reduce-time`.
- `worktree-agent-a412…`: the refine numeric-kernel feature commit is not patch-equivalent and must be merged, together with its small preserved reference inputs.
- `feat/column-split-merge-g0`: three unique commits.
- `feat/constructive-g0-g1`: five unique commits.
- `feat/frame-reinsert-g0`: four unique commits after excluding the already patch-equivalent tag-compress promotion.

## Remote/push state

The configured origin is `https://github.com/yyds-tsai-dev/Data-Driven-SoC-Floorplanning.git`. HTTPS fetch currently fails because no non-interactive GitHub credential is available. The `gh` CLI is not installed. SSH host verification was initialized, but the available key is not authorized by GitHub. Finish all local commits and merges, then retry after installing a credential with:

```bash
git fetch origin
git push origin main
```

Do not force-push. If the remote moved meanwhile, fetch and inspect `git log --left-right --graph main...origin/main` before integrating it.

## First resume action

Read `.planning/HANDOFF.json`, this document, `.superpowers/sdd/progress.md`, and the topology-prior design/plan in that order. Then verify `git status --short --branch` and whether `main` contains every branch head listed above. If GitHub credentials were the only remaining blocker, authenticate and run the non-force push; otherwise resolve the scorer-import test failure before making any green-suite claim.
