# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A solver for the **ICCAD 2026 FloorSet SoC floorplanning challenge**. Given a set of blocks with area targets, connectivity, pins, and hard/soft constraints, the solver produces non-overlapping rectilinear block placements that are scored by an official evaluator. Python 3.12, managed with `uv`.

The contest harness, data loaders, cost functions, and official evaluator live in the `FloorSet/` git submodule (`https://github.com/IntelLabs/FloorSet.git`) — treat it as read-only vendored code. Active solver code lives in `src/floorset_arch/`.

## Commands

Everything runs through `uv run` (the venv is uv-managed; never call `python` directly). Scripts are `#!/bin/bash` — invoke them with `bash scripts/...` even though the interactive shell here is tcsh.

```bash
bash scripts/install.sh              # init submodules, create venv, install deps, smoke-test evaluator
uv run pytest                        # full test suite (pyproject sets -s, testpaths=tests)
uv run pytest tests/test_v10_proxy.py -q   # single test file (do this first for solver-policy changes)
uv run pytest tests/test_repair.py::test_name  # single test

bash scripts/validate.sh             # validate the submission interface (src/architecture_v11_optimizer.py)
bash scripts/eval_single.sh 95       # evaluate ONE validation case (id 95) with diagnostics
bash scripts/eval_total.sh           # full 100-case evaluator; reports runtime + no-runtime totals
```

**Evaluating checkpoints** (`eval_total.sh` argument forms):

```bash
bash scripts/eval_total.sh some_gnn_best.pt                 # GNN guidance checkpoint (checkpoints/ relative)
bash scripts/eval_total.sh --diffusion-checkpoint diffusion_latest.pt --diffusion-use-ema
bash scripts/eval_total.sh --best-since-0512                # rank all dated *best*.pt checkpoints
bash scripts/eval_total.sh <ckpt> --output eval.json        # write full result JSON
```

The eval/validate scripts `cd` into `FloorSet/iccad2026contest`, set `PYTHONPATH`, and source `.env`. Run the evaluator through these scripts — invoking `iccad2026_evaluate.py` by hand without that setup will fail on imports.

**Training** (each script exposes tunables as `UPPER_CASE` env overrides and tees a `train_arch_v11_*.log`):

```bash
bash scripts/train_diffusion.sh      # v11 graph-conditioned diffusion prior (current focus) -> floorset_arch.training.train_diffusion
bash scripts/train_hgt.sh            # Anchor-GNN, HGT encoder    -> floorset_arch.training.train
bash scripts/train.sh                # Anchor-GNN, MPNN encoder (default)
bash scripts/train_transformer.sh    # Anchor-GNN, graph-transformer encoder
bash scripts/train_diffusion_eval_probe.sh   # deliberate overfit probe on the 100 eval cases (diagnostic only)
bash scripts/promote_checkpoint.sh   # promote a checkpoint -> floorset_arch.training.promote_checkpoint
```

`scripts/update.sh` auto-`git add -A` + commit + push — do not run it casually.

## Architecture

### Submission interface

The contest calls a `FloorplanOptimizer` subclass's `solve(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions)` and expects a list of `(x, y, w, h)` rectangles. The contest-facing entrypoint is [src/architecture_v11_optimizer.py](src/architecture_v11_optimizer.py), a thin wrapper exposing `MyOptimizer`/`ContestOptimizer = ArchitectureV11Optimizer`. `src/architecture_v5_optimizer.py` is a compatibility alias. The real logic is `ArchitectureV11Optimizer` in [src/floorset_arch/optimizer.py](src/floorset_arch/optimizer.py).

### `solve()` — the Production Solver Path

`solve()` parses inputs into an `Instance` (`parser.py`), then chooses ONE of two priors — there is a single production path, not hidden fallbacks:

1. **v11 diffusion path** (active) — taken when `FLOORSET_DIFFUSION_CHECKPOINT` is set. Sample a graph-conditioned diffusion prior → concretize to placements → repair → rank → best. Lives in `src/floorset_arch/diffusion/` (`sampling.py`, `concretize.py`, `ranking.py`).
2. **v5 Anchor-GNN path** — fallback when no diffusion checkpoint. Load Anchor-GNN guidance (or a deterministic `surrogate_guidance` when no GNN checkpoint) → build candidates with the hetero-graph beam decoder over MER/skyline slots → select best via the v10 proxy.

Both priors are advisory; the decoder + repair + ranking produce the final legal layout. Hard legality (no overlaps/missing blocks/fixed-shape or preplaced violations) always precedes soft-constraint refinement.

### `src/floorset_arch/` module map

- `optimizer.py` — orchestrates `solve()`; reads all `FLOORSET_*` runtime toggles.
- `parser.py`, `hetero_graph.py`, `features.py` — build the `Instance` and the Heterogeneous Floorplan Graph (typed block/pin/cluster/MIB/boundary nodes).
- `constructive.py`, `geometry.py`, `relative_order.py` — beam decoder, MER/skyline slot candidates, geometry helpers.
- `repair.py` — hard-legality + soft-constraint repair (multiple repair profiles).
- `scoring.py`, `v10_proxy.py` — evaluator-style scoring, and the solver-internal V10 no-runtime acceptance proxy for candidate/repair selection.
- `budget_layer.py`, `risk_budget.py`, `quality_portfolio.py` — the V10 Evidence-Gated / Conditional Runtime budget layer and opt-in sample-local quality portfolio, gating extra per-sample search.
- `surrogate_guidance.py` — deterministic guidance when running with no GNN checkpoint.
- `models.py` — `SolverConfig`, `Placement`, `CandidateSpec`, and other dataclasses.
- `nn/model.py` — Anchor-GNN (selectable MPNN / graph-transformer / HGT encoder; shared anchor/priority/aspect/pairwise heads).
- `diffusion/` — v11: `contracts.py`, `graph_inputs.py`, `targets.py`, `model.py`, `sampling.py`, `concretize.py`, `ranking.py`, `layout_losses.py`, `training.py`.
- `training/` — `train.py` (Anchor-GNN), `train_diffusion.py` (v11), plus `checkpoint.py`, `promote_checkpoint.py`, `selection.py`, `eval_probe_dataset.py`.

### Data & datasets

Blocks range 21–120. Training set = 1M samples (`LiteTensorData/`), validation set = 100 samples (`LiteTensorDataTest/`), test set = 100 hidden cases (final ranking). `fp_sol` golden layouts are geometric references only — per contest QA they may themselves violate soft constraints, so treat them as imitation targets, not constraint oracles.

## Project conventions that aren't obvious from the code

- **Checkpoints are promoted on Evaluator Evidence, never on supervised validation loss.** The primary architecture-tuning metric is the full-validation **No-Runtime Quality Score** (`total_score_no_runtime`); supervised val loss is only a training-health signal. Report score evidence when a change affects ranking or runtime.
- **Never hard-code validation `test_id` behavior into the solver.** Trigger heavier per-sample search from reusable instance statistics (block count, boundary/group/MIB density, fixed/preplaced structure, net density), not from case IDs. Validation-tail cases may guide *design direction* only.
- **Extra runtime work is gated.** A no-runtime score win does not by itself justify enabling extra candidate/repair/portfolio work — it must clear the budget layer (score evidence + reusable risk signals + raw runtime tail).
- **Runtime toggles use the `FLOORSET_` prefix**, are read via `os.environ` (mostly in `optimizer.py`), and should be documented near the code that consumes them. `.env` holds the defaults the eval scripts source; presence of `FLOORSET_DIFFUSION_CHECKPOINT` is what switches on the v11 path.
- **Checkpoint/log filenames encode their config** (e.g. `diffusion_best_evaluator_0701_ns1000000_ep6_diffhgt_lite_h128_l2_steps1000_acc32_bs1.pt` = date `0701`, 1M samples, 6 epochs, `hgt_lite` variant, hidden 128, 2 layers, 1000 diffusion steps, accum 32, batch 1). Preserve this scheme when producing new artifacts.
- `checkpoints/`, `artifacts/`, `wandb/`, and `*.log` are generated/evidence outputs — don't hand-edit them or bundle them into code/doc commits.
- Style: 4-space indent, `snake_case` functions/vars, `PascalCase` classes, typed dataclasses for solver state; group imports stdlib / third-party / local.

## Deeper references

- [CONTEXT.md](CONTEXT.md) — the authoritative domain glossary (Production Solver Path, V10 proxy, budget layer, diffusion terms) and resolved ambiguities. Read it before proposing solver-policy changes; the precise vocabulary matters.
- [AGENTS.md](AGENTS.md) — repo guidelines plus the `graphify` (knowledge graph in `graphify-out/`) and `semble` code-search tooling. For codebase questions prefer `graphify query "<question>"` when `graphify-out/graph.json` exists; run `graphify update .` after code changes.
- [README.md](README.md) — detailed (bilingual EN/中文) walkthrough of the problem, scoring, scripts, and module architecture.
