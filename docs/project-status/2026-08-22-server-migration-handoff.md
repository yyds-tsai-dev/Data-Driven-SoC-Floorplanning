# Server migration handoff — 2026-08-22

## Executive state

The consolidated development history is now on local `main`, 509 commits ahead of `origin/main` at the last locally visible remote state. The goal service reports no active goal, so there is no API-owned goal to pause. This document is the durable pause/resume record for Codex, Claude Code, or a human on the new server.

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
- Every requested worktree branch is reachable from `main`. The integration merge series is `29dcaab..798e6a4`; `9a0efc9` adds the flow-v1 checkpoint merge.
- `checkpoints/flow_matching_v1/final.pt` is LFS-tracked at commit `533a810`. It is 1,719,984,052 bytes and contains model, EMA, optimizer, scheduler, step, model configuration, and training arguments for continuation training.
- `submission/cadc1013_0812_groupbridge.tar.gz` is the newest submission package and is LFS-tracked at commit `5153e62`. It is 799,364,946 bytes, contains 32 archive entries, and passed gzip/tar integrity checks.

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

The final post-fix merged-main run collected 2,282 tests and ended with 2,260 passed, 5 skipped, and the same 17 topology scorer-contract failures in 1,034.76 seconds. No worktree/feature merge regression remained.

The focused external-report/backend contract suite passed 63 tests, and all newly found Python, shell, JSON, tar, and xlsx files passed syntax/container validation before commit. Merged worktree/feature coverage passed 289 tests. `scripts/validate.sh` passed against `ArchitectureV11Optimizer`, and Graphify rebuilt 16,635 nodes and 37,907 edges.

One additional merged-main full-suite failure exposed a wall-clock-dependent anytime-ladder test witness. Production code was unchanged; commit `efafe09` makes the test force the intended off-path traversal instead of relying on a one-second machine deadline. The previously failing prefix now passes 86 tests and the complete anytime-ladder file passes 30 tests.

## Local-only generated state that Git will not migrate

Do not add these directories to ordinary Git. They are generated, ignored by repository policy, or far above GitHub's practical limits.

- Main worktree `artifacts/`: about 57 GB. This includes the immutable formal G0 evidence (`artifacts/icdc_g0_v2_area_full/`, about 946 MB) and topology work (`artifacts/icdc_topology/`, about 821 MB).
- Main worktree `checkpoints/`: about 1.8 GB, excluding the separate flow worktree runs below.
- Main worktree `wandb/`: about 624 MB.
- `FloorSet/floorset_lite/`: 9,000 files, about 24 GB, untracked inside the FloorSet submodule.
- Flow-matching worktree nested checkpoint runs: about 246 GB total. The full v1 final continuation checkpoint is now represented through Git LFS; the remaining periodic snapshots are still generated local-only evidence.

Transfer required local state before retiring this server. From the destination server, use a resumable transport appropriate for the environment, for example:

```bash
rsync -aH --info=progress2 --partial \
  OLD_SERVER:/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/artifacts/ artifacts/
rsync -aH --info=progress2 --partial \
  OLD_SERVER:/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/FloorSet/floorset_lite/ FloorSet/floorset_lite/
rsync -aH --info=progress2 --partial \
  OLD_SERVER:/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/ checkpoints/flow-worktree-archive/
```

Only transfer the remaining flow archive if historical periodic snapshots are still required. The source, recipes, and full v1 final continuation state are represented in Git/LFS; periodic checkpoints remain generated evidence.

## Branch/worktree consolidation facts

- `5.6-sol-flow-matching`: original feature history was already integrated by ancestry; commit `533a810` adds the requested full v1 LFS checkpoint, merged by `9a0efc9`.
- `worktree-agent-a438…`, `a8b…`, `a940…`, `ae70…`, and `af25…`: patch-equivalent feature history and any new evidence are reachable through explicit merge commits.
- `worktree-agent-a412…`: refine numeric-kernel history and preserved reference inputs are reachable through `29dcaab`.
- `feat/column-split-merge-g0`, `feat/constructive-g0-g1`, and `feat/frame-reinsert-g0` are reachable through `16e0f49`, `42b3466`, and `798e6a4` respectively.
- A reachability audit returned `REACHABLE` for all 11 named worktree/feature branches.

## Remote/push state

The configured origin is `https://github.com/yyds-tsai-dev/Data-Driven-SoC-Floorplanning.git`. HTTPS fetch currently fails because no non-interactive GitHub credential is available. The `gh` CLI is not installed. SSH host verification was initialized, but the available key is not authorized by GitHub. Local `main` includes both requested LFS artifacts, but their LFS objects and local commits cannot reach GitHub until credentials are installed. Retry with:

A final non-force `git push origin main` was attempted after verification and failed before transfer with `fatal: could not read Username for 'https://github.com': No such device or address`. No remote ref or LFS object was changed by that attempt.

```bash
git fetch origin
git push origin main
```

Do not force-push. If the remote moved meanwhile, fetch and inspect `git log --left-right --graph main...origin/main` before integrating it.

## First resume action

Read `.planning/HANDOFF.json`, this document, `.superpowers/sdd/progress.md`, and the topology-prior design/plan in that order. Then verify `git status --short --branch`, authenticate GitHub, run `git fetch origin`, inspect any remote movement, and perform the non-force push. Resolve the scorer-import test failure before making any green-suite claim; resume topology-prior work only at the G1 sequence documented above.
