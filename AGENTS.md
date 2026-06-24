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
