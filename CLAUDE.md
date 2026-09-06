# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team `cadc1013`'s solver for the **ICCAD 2026 FloorSet SoC floorplanning challenge** (Problem C). Given blocks with area targets, connectivity, pins and hard/soft constraints, produce non-overlapping `(x, y, w, h)` placements scored by the official evaluator. Python 3.12, managed with `uv`.

The contest closed 2026-08-31. On 2026-09-06 the repo was pruned to the **final-submission path only**: the shipped package source (`src/`), the tooling that produced its checkpoints and packages, and the full evidence trail in `docs/`. Mapping of everything that moved or was removed: [docs/project-status/2026-09-06-repo-cleanup.md](docs/project-status/2026-09-06-repo-cleanup.md). The official harness, loaders and evaluator live in the read-only `FloorSet/` submodule.

## Commands

Everything runs through `uv run` (never call `python` directly). Scripts are `#!/bin/bash`; invoke them with `bash scripts/...`.

```bash
bash scripts/install.sh                         # submodules, venv, deps, evaluator smoke test
uv run pytest                                   # full suite (pyproject: -s, testpaths=tests)
uv run pytest tests/test_solver_final_legal_guard.py -q

bash scripts/pack_cadc1013.sh <out_dir>         # build <out_dir>/cadc1013.tar.gz from src/solver + src/shipping
bash scripts/validate.sh                        # official --validate on the package (packs into artifacts/eval_package/ if needed)
bash scripts/eval_single.sh 95                  # one validation case
bash scripts/eval_total.sh                      # all 100 cases -> artifacts/eval_runs/total_<ts>.json + summary line
REPACK=1 bash scripts/eval_total.sh             # force a fresh pack first
bash scripts/eval_total.sh <package_dir>        # evaluate an existing unpacked package

bash scripts/release/rehearse_package.sh        # pack -> extract -> clean venv from the package's requirements -> official evaluator
bash scripts/release/apply_pack_variant.sh ft2 s16   # switch the working tree to the B' shipping variant (no args = 0828b)
bash scripts/gate/run_gate5.sh <tag> ENV=VAL ... # official + shadow v3/v5/v6 gate run
```

The eval scripts `cd` into `FloorSet/iccad2026contest` and set `PYTHONPATH`; run the evaluator through them.

## Architecture

### The shipped package

`scripts/pack_cadc1013.sh` copies the import closure of `src/solver/contest_optimizer.py` (26 modules, listed by `scripts/probes/package_closure.py`) flat into `cadc1013/`, renames the entry to `op_src.py`, adds `src/solver/synth_instances.py` (JIT warm-up), `src/shipping/op_wrapper.py` (the contest entry point: sets every promoted env knob via `os.environ.setdefault`, warms numba kernels and the worker pool in `__init__`), `src/shipping/requirements.txt`, and two checkpoints (`flow_matching_*.pt` from `submission/cadc1013/checkpoints/`, `direct_v2_student_s2.pt` from `artifacts/icdc_topology_prior_training/checkpoints_s2_20k/best.pt`).

**Module names inside `src/solver/` are the module names inside the tarball. Do not rename them, and do not edit shipped modules without gate evidence** — a repack from `src/` is expected to `diff -r` clean against the uploaded package.

### `solve()` pipeline (`src/solver/contest_optimizer.py`)

Per case, a wall-clock budget from `PARTNER_BUDGET_TABLE` (per-n seconds, `src/shipping/budget_table_mid.txt`):

1. **Candidates** — `candidate_supply.py` allocates slots to the flow prior (`flow_matching_model.py`, `FLOW_CKPT`, 8-step Euler antithetic, 16 slots), the direct topology-prior student (`direct_diffusion_model.py` / `direct_diffusion_train_v2.py`, `DIRECT_CKPT`, DPM++ 2 steps) and a deterministic heuristic seed. Retrieval (`retrieval_*.py`) is opt-in and not shipped.
2. **Column-slicing SA legalizer** — `column_sa_legalizer.py` + `sa_numeric_kernel.py` (+ `csa_coordinate_solver.py`, `column_lns.py`): overlap-free / exact-area / fixed-and-preplaced-preserving **by construction**, then a numba SA under the budget on HPWL, bbox area and soft violations, 24 parallel restart workers.
3. **Refine ladder** — `layout_refiner.py` + `refine_numeric_kernel.py`, `violation_killer.py` (grouping DAG bridge), `frame_repack.py`, `tag_compress.py`, `coord_polish.py`, `noise_optimization.py`, `physics_guidance.py`: edge seat, wall repair, tag compress, frame scale 1.02, final seat.
4. **Selection + guard** — candidates ranked on evaluator-form cost; `PARTNER_FINAL_LEGAL_GUARD=1` re-checks the final layout with evaluator-faithful predicates and falls back (area repair, then a verified-legal column layout) only on failure.

`src/icdc_engine/` is the IC/DC engine and the topology-prior trainer that produced the shipped direct checkpoint (`train_topology_prior.py`, `engine.py`, `g1_runtime.py` holds the canonical solver env for gate runs). `scripts/training/flow_finetune/` holds the round 1–4 fine-tune launchers behind the shipped flow prior (trainer = `src/solver/flow_matching_train.py`).

### Data

Blocks 21–120. Training set 1M samples (`FloorSet/LiteTensorData/`), validation 100 (`LiteTensorDataTest/`), hidden test 100 (same block-count distribution as validation per test_id). `fp_sol` golden layouts may violate soft constraints; treat as geometric references only. Shadow suites for gating (v3/v5/v6, alpha_1) live in `artifacts/shadow_hidden_suites/`.

## Project conventions that aren't obvious from the code

- **Promotion needs gate evidence, not local intuition.** A knob enters the package only after the complete candidate env vs the current package env runs on the same chain, 4 reps, both arm orders, across official + shadow v3/v5/v6, with the CI excluding 0. Same-chain rule and rejected-knob list: `docs/project-status/2026-08-28-final-sprint-handoff.md`, `docs/experiments/2026-08-21-post-beta-p0-execution.md` §17–18.
- **Primary metric is `total_score_no_runtime`** (weighted by `exp((n-120)/12)`); runtime-aware expectations are computed separately (`scripts/release/runtime_aware_*.py`) because the runtime factor depends on the field median.
- **Never key solver behaviour on validation `test_id`.** Use reusable instance statistics (block count, constraint density).
- **Runtime toggles use the `PARTNER_` prefix**, read via `os.environ`, set in `src/shipping/op_wrapper.py` for the package; `.env` is a decision record, not a consumed config.
- **Rehearse before any upload** (`scripts/release/rehearse_package.sh`): clean venv from the package's own `requirements.txt`, look for `[selfcheck] cuda_available=True`, `loaded flow model step 250000`, `legal-guard fires: 0`, 100/100 feasible. The beta 1.314 came from a 0-byte requirements file, not from the solver.
- Repacking changes the tarball md5 (mtimes); verify content with an extracted `diff -r`.
- `artifacts/`, `submission/`, `*.pt`, `*.log` are generated / evidence outputs — never hand-edit or commit them.
- Style: 4-space indent, `snake_case` functions/vars, `PascalCase` classes; imports grouped stdlib / third-party / local.

## Model orchestration (Fable 5.1 scheduler)

This project runs a multi-model team to spend the expensive scheduler budget
sparingly. **Fable 5.1 (max reasoning) is the scheduler**, set in
`.claude/settings.json` (`model: claude-fable-5-1`, `effortLevel: xhigh`) — takes
effect on the next session, not retroactively. The scheduler plans, decomposes,
delegates, and integrates results; it should keep its own context lean and push
the actual work down to the specialists rather than burning Fable budget on it.

Division of labour:

- **Scheduler — Fable 5.1 (this main thread).** Understand the request, break it
  into well-scoped units, route each to the cheapest capable executor below,
  then stitch the results together and verify. Do the thinking about *what* and
  *who*; delegate the *doing*.
- **Deep reasoning → `deep-reasoner` subagent (Opus).** Hard architecture,
  solver-policy design, scoring/ranking trade-offs, subtle root-cause debugging,
  "does this actually move the No-Runtime Quality Score?" judgements. Hand it a
  self-contained problem statement plus file paths. See
  [.claude/agents/deep-reasoner.md](.claude/agents/deep-reasoner.md).
- **Mechanical execution → `fast-worker` subagent (Sonnet).** Well-specified,
  low-ambiguity work: applying a decided edit, running scripts/tests and
  reporting, renames/moves, boilerplate, doc sync, fact-gathering. Give it an
  explicit instruction and exact commands. See
  [.claude/agents/fast-worker.md](.claude/agents/fast-worker.md).
- **Peer engineer → Codex (OpenAI), via the official `codex@openai-codex`
  plugin.** A same-level engineer with a *different vantage point*. Use
  `/codex:review` (read-only) or `/codex:adversarial-review` (steerable
  challenge that pressure-tests design choices and tradeoffs) for independent
  cross-checks and second opinions, and `/codex:rescue` / the
  `codex:codex-rescue` subagent to delegate a task or get an alternative
  implementation (`/codex:status` / `/codex:result` manage background jobs);
  then reconcile its take with `deep-reasoner`'s. The plugin (marketplace
  `openai-codex`, from `openai/codex-plugin-cc`) is enabled in
  `.claude/settings.json` and drives the already-authenticated `codex` CLI —
  run `/codex:setup` once per machine to confirm readiness. When Codex is
  unavailable, fall back to a second `deep-reasoner` pass with an adversarial
  framing.

Routing heuristic: if a task needs a *decision or a diagnosis*, send it to
`deep-reasoner`; if it needs *hands* on already-decided work, send it to
`fast-worker`; if you want a *dissenting second implementation or review*, ask
Codex. When in doubt about cost, prefer delegating over doing it in the
scheduler thread.

## Deeper references

- [docs/project-status/2026-08-31-final-handoff.md](docs/project-status/2026-08-31-final-handoff.md) — final state: B' (uploaded, md5 3f2cda42) vs C' candidate (md5 6d0ca94e); which is the final upload awaits the user's confirmation.
- [docs/project-status/2026-08-29-team-summary.md](docs/project-status/2026-08-29-team-summary.md) — team-facing shipping summary.
- [docs/experiments/2026-08-21-post-beta-p0-execution.md](docs/experiments/2026-08-21-post-beta-p0-execution.md) — every gate of the final sprint (§1–18w).
- [CONTEXT.md](CONTEXT.md) — glossary (includes retired floorset_arch vocabulary, kept for reading the 05–07 docs).
- [AGENTS.md](AGENTS.md) — repo guidelines plus `graphify` / `semble` tooling notes.
- [README.md](README.md) — 中文 walkthrough of the problem, the final package, scripts and layout.
