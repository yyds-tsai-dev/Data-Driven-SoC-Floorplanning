# 2026-07-05 — Column-Backbone Pivot: Full Session Log

One-day architecture pivot: the partner column-slicing legalizer became the
production path, a slack refiner was built on top, and every ML-seed channel
was measured and closed. All numbers are 100-case validation
`total_score_no_runtime` (lower is better) from `scripts/eval_total.sh`.

## Final state

| Configuration | no-runtime | feasible |
|---|---|---|
| **Production (column backbone + refiner, now .env default)** | **1.238–1.246** (run band) | 100/100 every run |
| Previous production (v5 Anchor-GNN, gnn_transformer_best_0521) | 2.1158 | 100/100 |
| v11 diffusion path (diffusion_best_evaluator_0701, EMA) | 3.3894 (avg 18.6 s/case) | 100/100 |
| Ground truth fed to the evaluator | 1.1079 | — |

Run-to-run noise of the SA backbone is ±0.006 on the total; **gate small deltas
with same-run paired logs** (`FLOORSET_SLACK_REFINE_LOG` JSONL, before/after
HPWL per case), never with two independent full runs.

## What was built (commits on `fable5`)

1. `feat(legalizer)` — vendored `partner/legalizer_claude_v2.py` as
   `src/floorset_arch/legalizer/column_slicing.py` + `column_backbone.py`
   (heuristic seed, exp time budget `clamp(0.06·e^{n/20}, 0.8, 24)`, row
   fallback). Toggle `FLOORSET_COLUMN_BACKBONE=1` in `optimizer.py`.
   Includes two robustness fixes over the partner original (case 4, n=25,
   7.99 → 3.67): overflow-widen guard (absolute cap `w_base*1.6`, dead-space
   break at `overhead ≥ 0.9H`, non-monotone break) and cross-obstacle
   soft-unit stacking. Same pathology was latent in cases 89/94/95/99
   (preplaced obstacles), so the guard protects the hidden-test tail.
2. `feat(refine)` — slack refiner (`src/floorset_arch/refine/`): axis
   separation DAGs from the legal layout + weighted-median projected
   Gauss-Seidel (Phase 1), area-preserving aspect moves (Phase 2), boundary
   wall-snap with bounded local reorder (vsnap v1+v2). Pure function; any
   guard failure returns the input, so the backbone score is a hard floor.
   Spec: `docs/design/slack_refiner_spec.md`. 23 invariant tests in
   `tests/test_refine_invariants.py`.
3. `perf(refine)` — refiner budget carved out of the SA share
   (`min(2.0, 0.3+0.012n, 0.4·budget)`), runtime-neutral.
4. `feat(config)` — .env promotion of the four flags.
5. `chore(partner)` — partner sources, harness, real diffusion modules.

## Measured verdicts (do not re-litigate without new evidence)

- **Refiner paired gains are small**: Phase 1 ≈ 0.001, Phase 2 aspect ≈ 0.003
  (95/100 accepted). Fixed-topology refinement is capped: columns pack tight,
  translation/aspect slack is ~0.1–1 % HPWL on big cases.
- **Violation line is closed**: big-case (n≥100) mean v_rel is 0.0335,
  already **below ground truth's ~0.053**. vsnap v1+v2 (structurally correct,
  tested) captures almost nothing — remaining violations are structural
  (competing wall claims, LOCKED blockers, preplaced not touching its tagged
  boundary per official QA A5). A violation-averse SA portfolio boost
  (v_weight 5.0 on n≥100) was a wash and was reverted.
- **The seed channel of the column backbone is dead — triple-verified**:
  1. GT coordinates as seed: 1.2392 vs heuristic 1.2403 (Δ −0.001, inside
     noise; 51 better / 48 worse — SA basin noise fingerprint).
     Probe: `scripts/probes/gt_seed_optimizer.py` (matches cases by
     block_count, monkeypatches `_heuristic_init`).
  2. Partner README: GT-coordinate seeds through their diffusion refine still
     ~1.35× GT HPWL.
  3. Partner's real 14.8M diffusion ckpt (`partner/step_00080000.pt`, loads
     via `partner/harness/`): 1.2418 vs 1.2400 heuristic — inside noise.
  Root cause: the legalizer consumes seeds only as relative order
  (`seed_x/seed_y` sorting) and the time-budgeted SA washes that order out.
- **GNN checkpoint family sweep** (`artifacts/eval_v10/best_since_0512/`):
  best is `gnn_best_0514` = 2.1007; whole family 2.10–2.19 (0603 broken at
  4.00). Irrelevant to production; 0514 is the ordering-prior candidate if ML
  ever re-enters.
- **diffusion→guidance retraining: parked.** No pipeline stage consumes model
  output today. Re-entry condition: after a topology-changing refinement
  layer exists, re-run the GT-seed probe; integrate ML only if it moves
  ≥ 0.01.

## Next levers (priority order)

1. **Topology-changing refinement** — sequence-pair / order-swap local search
   over the legal layout, using the existing `refine/` DAG projection as the
   inner solver. The only path that attacks hpwl_gap 0.14–0.38 → ~0.08
   (GT floor 1.108, current plateau ~1.24: the whole remaining gap is HPWL).
   Also the gate that would let ML ordering priors matter again.
2. **PyInstaller CPU-only packaging** — mandatory before submission; the
   production path needs no torch (torch imports are lazy/legacy-gated);
   `op_wrapper.py` spawns the binary per case, so startup cost is charged
   per case. Test `multiprocessing` + `freeze_support` under the frozen
   binary early.
3. SA throughput vectorization (partner roadmap #2) — helps convergence
   variance; profile before compiling (object-heavy `_layout` resists Numba).
4. case 4 residual (3.67; FIX B incomplete) — negligible weight, idle-time
   work.

## Operational notes

- Evaluations: always through `scripts/eval_total.sh` / `eval_single.sh`;
  under concurrent load runtime numbers are inflated (no-runtime is the
  gating metric). Two trainings were running on the GPU all day; the
  backbone path is pure CPU.
- Do NOT let two agents/edits touch shared modules while an eval runs — a
  mid-edit import produced a fake 76/100-feasible run (step1b, discarded).
- `--best-since-0512` in eval_total.sh also sweeps `diffusion_best_*.pt` as
  GNN checkpoints (name glob) — use an explicit list instead.
- graphify knowledge graph (`graphify-out/graph.json`) predates this pivot —
  run `graphify update .` before relying on it for the new modules.
