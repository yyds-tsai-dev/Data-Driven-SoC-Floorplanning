# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 `uv` project for the ICCAD 2026 FloorSet SoC floorplanning challenge. Active solver code lives in `src/floorset_arch/`; `src/architecture_v5_optimizer.py` is the evaluator-facing wrapper. Training code is under `src/floorset_arch/training/`, neural models under `src/floorset_arch/nn/`, and focused solver modules include `optimizer.py`, `repair.py`, `budget_layer.py`, `v10_proxy.py`, and `quality_portfolio.py`. `tests/` contains pytest coverage. `scripts/` holds install, train, validate, and evaluator workflows. `FloorSet/` is the contest submodule/data area. Treat `checkpoints/`, `artifacts/`, and `wandb/` as generated or evidence outputs unless a doc explicitly references a file.

## Build, Test, and Development Commands

- `bash scripts/install.sh`: initialize submodules, install `uv`, create the venv, install dependencies, and run a smoke evaluator check.
- `uv run pytest`: run the full test suite configured by `pyproject.toml`.
- `uv run pytest tests/test_budget_layer.py -q`: run a targeted test file during solver-policy changes.
- `bash scripts/validate.sh`: validate the submission interface for `src/architecture_v5_optimizer.py`.
- `bash scripts/eval_single.sh 95`: evaluate one validation case with diagnostics.
- `bash scripts/eval_total.sh`: run the full 100-case evaluator and report runtime plus no-runtime totals.

## Coding Style & Naming Conventions

Use 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and typed dataclasses where they clarify solver state. Keep imports grouped as standard library, third party, then local modules. Prefer small, explicit functions around geometry, scoring, and repair decisions. Environment toggles should use the `FLOORSET_` prefix and be documented near the workflow that consumes them.

## Testing Guidelines

Tests use `pytest`; new tests should be named `tests/test_<feature>.py` and should cover both normal placement behavior and edge cases around constraints, runtime gates, or v10 scoring. For solver changes, run targeted pytest first, then `uv run pytest`; use `scripts/validate.sh` or evaluator scripts when the submission path changes.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional-style subjects such as `docs: ...`, `refactor: ...`, and `chore: ...`. Keep commits focused and avoid bundling checkpoint noise with code or docs. PRs should summarize the changed solver path, list tests/evaluator commands run, and include score evidence when behavior affects v10 ranking or runtime.

## Agent-Specific Instructions

### Model Orchestration (GPT-5.6)

Activate this policy only when the user explicitly selects Subagent-Driven
execution. Otherwise, keep normal single-agent behaviour and do not spawn
subagents merely because the roles below are available. Project configuration
in `.codex/config.toml` pins the root task to `gpt-5.6-sol` at `xhigh`; a newly
started or reloaded Codex task is required for that setting and the custom-agent
files to take effect.

#### Roles and routing

- **Scheduler/Integrator — `gpt-5.6-sol` (`xhigh`).** The root task owns request
  interpretation, dependency-aware decomposition, routing, integration,
  conflict resolution, final verification, and acceptance. Keep the scheduler
  context focused on requirements and decisions; delegate noisy exploration,
  test output, and mechanical execution when delegation is worthwhile.
- **Deep reasoner — `deep-reasoner`, `gpt-5.6-terra` (`xhigh`).** Use for
  bounded architecture analysis, experiment design, integration reasoning,
  subtle diagnosis, task-scoped review, and decisions that require trade-off
  analysis. Terra returns evidence and a recommendation; Sol retains the final
  decision for highest-risk architecture, solver policy, unresolved ambiguity,
  and whole-branch acceptance. The project role is defined in
  `.codex/agents/deep-reasoner.toml` with a read-only default sandbox.
- **Fast worker — `fast-worker`, `gpt-5.6-luna` (`low`).** Use for
  well-specified, low-ambiguity implementation, commands and tests, renames and
  moves, boilerplate, documentation synchronization, and fact gathering. Luna
  executes decided work and stops at material design ambiguity instead of
  guessing. The project role is defined in `.codex/agents/fast-worker.toml`.

Route decisions, diagnoses, and non-mechanical review to Terra; route already
decided execution to Luna; keep cross-task integration, unresolved trade-offs,
highest-risk work, and final acceptance with Sol. The scheduler may perform a
trivial action directly when delegation overhead would exceed the work.

#### Dispatch contract

Every delegated task must state the objective, why the chosen role is
appropriate, exact file or subsystem ownership, relevant constraints and
already-made decisions, required output, and verification evidence. Always
specify the model and reasoning effort explicitly at dispatch time even when a
custom-agent file also pins them.

All agents share the same worktree. Parallelize only independent tasks with
disjoint write ownership; serialize dependencies and overlapping files. Tell
every writer that it is not alone in the codebase, must preserve unrelated
edits, and must not revert another agent's work.

#### Execution and review loop

1. Sol decomposes the request into bounded, dependency-aware tasks.
2. Terra resolves any undecided architecture or behaviour questions before
   implementation.
3. A fresh Luna implementer receives one task with explicit ownership and
   verification requirements.
4. A separate Terra pass reviews the task requirements and actual diff.
5. Important or Critical findings return for correction and re-review.
6. Sol integrates accepted tasks, resolves cross-task inconsistencies, runs or
   delegates final verification, and performs the broad final whole-branch
   review at high or xhigh reasoning.

Subagent reports are advisory until Sol checks the claimed files, evidence,
and integration state. Never report success based only on a subagent summary.

#### Fallbacks

- If the active interface has not loaded the project custom-agent names, use a
  built-in `default` or `worker` agent with the same explicit model, reasoning
  effort, and role instructions.
- If the active spawn interface rejects `gpt-5.6-luna`, invoke the verified
  fast-worker path through an ephemeral
  `codex exec -m gpt-5.6-luna` file-handoff workflow. Fall back to Terra at
  low or medium effort only when that CLI workflow is unsuitable.
- A failed subagent does not authorize discarding unrelated work or widening
  scope. Retry with a corrected bounded prompt, use the documented fallback,
  or report the blocker.

### graphify

This project has a knowledge graph at `graphify-out/` with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:

- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than `GRAPH_REPORT.md` or raw grep output.
- Dirty `graphify-out/` files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

### Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

```bash
semble search "authentication flow" .
semble search "save_pretrained" .
semble search "save model to disk" . --top-k 10
```

If you anticipate doing more than one search, use `semble index` to create an index:

```bash
semble index . -o .semble-index
semble search "save_pretrained" --index .semble-index
```

Use `--content docs` to search documentation and prose, `--content config` for config files, or `--content all` to search code, docs, and config:

```bash
semble search "deployment guide" . --content docs
semble search "database host port" . --content config
semble search "authentication" . --content all
```

Use `semble find-related` to discover code similar to a known location:

```bash
semble find-related src/auth.py 42 .
```

An index is not automatically updated, so reindex after significant code changes or if results look stale. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
