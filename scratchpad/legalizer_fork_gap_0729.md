# Legalizer fork gap report: `partner/legalizer_claude.py` vs `src/floorset_arch/legalizer/column_slicing.py`

Date: 2026-07-29. Read-only inventory, no code changes made.

## File sizes

| File | Lines |
|---|---|
| `src/floorset_arch/legalizer/column_slicing.py` (production) | 3126 |
| `partner/legalizer_claude.py` (fork) | 3159 |

## Comparison table (function/method names)

Note: both files define a `_ColumnOptimizer` class plus module-level helpers; names below are as extracted from `def ` lines regardless of class scope.

| Function | In src? | In partner? | Notes |
|---|---|---|---|
| `_col`, `_parse_constraints`, `_target`, `rectangles_from_z` | yes | yes | shared input-parsing helpers, identical role |
| `_Unit`, `_DSU` (`find`/`union`/`flatten`) | yes | yes | shared data structures |
| `_ColumnOptimizer.__init__`, `_resolve_shapes`, `_build_hpwl_arrays`, `_build_soft_norm`, `_choose_frame`, `_build_units` | yes | yes | shared setup |
| `_col_of_x`, `_init_columns`, `_obstacles_in`, `_unit_height`, `_partition_subgroup`, `_refresh_unit` | yes | yes | shared column bootstrap |
| `_solve_band`, `_merge_band_once`, `_band_solutions`, `_dyn_split`, `_unit_h` | yes | yes | shared banding logic |
| `_place_chunk_up`, `_place_band_up`, `_place_band_down`, `_place_unit_up`, `_place_unit_down`, `_stack_column`, `_band_strip`, `_interval_free`, `_add_interval` | yes | yes | shared placement primitives |
| `_layout` | yes | yes | **diverged dispatcher** — see below |
| `_hpwl`, `_violations`, `_cost`, `_evaluate`, `_evaluate_bootstrap`, `_random_move`, `_snapshot`, `_restore` | yes | yes | shared SA scaffolding |
| `_anneal` | yes | yes | **diverged body** — see SA section |
| `locked_only`, `locked_positions`, `prepare`, `probe`, `finish`, `run` | yes | yes | shared restart-manager API |
| `_spread_tagged`, `_repair_boundary`, `_order_pass`, `_reduce_overflow`, `_greedy_polish` | yes | yes | shared post-anneal repair passes |
| `_ensure_no_overlap`, `_worker_init`, `_worker_ping`, `init_worker_pool`, `_worker_solve`, `_parallel_solve`, `_transpose_inputs`, `legalize_rectangles` | yes | yes | shared worker-pool / entrypoint plumbing |
| `_layout_fast` | **yes** | no | src-only single-layout-function perf path (memoized per-column cache; see section 3) |
| `_solve_column` | **yes** | no | src-only; partner inlines an equivalent solve inside `_layout_full`/`_layout_delta` instead of a standalone method |
| `_layout_widths`, `_finish_layout`, `_col_touches_obstacle` | **yes** | no | src-only helpers supporting the `_layout_fast` cache path |
| `_cumulative`, `_fastsa_temp` | **yes** | no | src-only: alternate ("fastsa") temperature schedule + weighted-choice cumulative-sum helper |
| `_adaptive_reset`, `_adaptive_reweight` | **yes** | no | src-only: adaptive move-mix reweighting inside `_anneal` (tracks proposal/accept/improvement stats per move type) |
| `_cfix_move` | **yes** | no | src-only: PARSAC-style constraint-fixing move fired probabilistically inside `_anneal` |
| `_count_components` | **yes** | no | src-only connectivity helper |
| `_width_optimize` | **yes** | no | src-only post-pass |
| `race_round1`, `race_round2`, `_build_worker_opt`, `_race_configs`, `_worker_race_r1`, `_worker_race_r2`, `_parallel_solve_racing`, `score1`, `score2`, `_log` | **yes** | no | src-only "racing" parallel-restart variant (multiple SA configs raced against each other across workers, with logging hooks via `FLOORSET_WIDTH_OPT_LOG`) |
| `_layout_full` | no | **yes** | partner-only: full (non-incremental) layout pass, analogous role to src's `_layout_fast` cold path |
| `_layout_delta` | no | **yes** | partner-only: incremental/delta layout recompute (`_dc_enabled`, `_dc_cache`, `_dc_prev_cols/_dc_prev_sig`) — a different caching strategy from src's per-column memo cache |
| `_perimeter_pack`, `_perimeter_seed_layouts` (+ inner helpers `along`, `corner_span`, `fill_members`, `line_need`, `locked_span`, `min_along`, `place_wall`, `seed_c`, `shape`, `shp`, `wall_items`) | no | **yes** | partner-only: explicit perimeter/wall-line packing for boundary-dense, wall-saturated instances |
| `_worker_refine` | no | **yes** | partner-only: worker-pool entrypoint that runs a full legalize+refine pass per candidate (calls into `refiner_claude.refine_prediction`) |

## Section 3 — perf/cache/worker-path functions in src

**`_layout_fast`** (`src/floorset_arch/legalizer/column_slicing.py:1577`): the single layout entrypoint used by every SA move in src (there is no separate "full" vs "delta" split — one function handles both cold and warm paths via a dict cache `self._col_cache`). For each column it builds a cache key from `(tuple(unit_ids), tuple(unit_versions))`; on a hit it reuses the previously solved relative geometry and just applies a rigid x-shift (`pos[ids] = rel; pos[ids,0] += x`); on a miss it calls `_solve_column` and stores the result. It carries a secondary keyed-by-`(key, x)` cache tier specifically to handle obstacle-touching columns, whose validity depends on absolute x (SA oscillations frequently revisit the same x), documented in the inline comments at lines 1564-1568 and 1600-1616. Comments at lines 1452-1453 and 2218 confirm this is "the FAST_EVAL cache" referenced elsewhere in the file (`u.hcache`, a one-slot per-unit band-solve cache used by `_band_solutions`, is the same idea at finer grain).

**Worker-path functions** (`_worker_init`, `init_worker_pool` at line 2626, `_worker_ping`, `_worker_solve`, `_race_configs`/`_worker_race_r1`/`_worker_race_r2`, `_parallel_solve`/`_parallel_solve_racing`): `init_worker_pool`'s docstring says it "create[s] the restart pool once and wait[s] until every worker has finished" its warmup; the surrounding comment explains the pool uses `fork` context specifically so workers skip re-importing `__main__` and per-worker `torch` import cost, and never touch CUDA so forking a CUDA-initialized parent is safe.

**Relation to the 0707 N3 note** ("graph build moved to worker path, 5.7x throughput"): this claim is about graph-construction cost being moved off the hot per-candidate path and into the worker/fork path, which is a different code layer (candidate/graph-build in `optimizer.py`/`hetero_graph.py`, not inside `column_slicing.py`). Nothing in `column_slicing.py`'s worker-pool code (fork-based `_POOL`, warmup ping) computes graph features — its job is purely legalization SA restarts. So the "5.7x" figure and this file's worker pool are related in *spirit* (both are "front-load one-time cost into worker init, amortize across restarts") but are not the same code path; this file's worker functions do not, by themselves, constitute the change the 0707 note describes. They are a plausible contributor to overall throughput but not verifiable as literally "the" 5.7x source from this file alone.

## Section 4 — SA main loop: structurally diverged

Both `_anneal` methods (src: line 1917; partner: line 1745) share the same skeleton (outer time-budget loop, exponential-schedule `T`, Metropolis accept via `_random_move`/`_cost`, `_snapshot`/`_restore` for best-so-far tracking) but have diverged in three concrete ways:

1. **Temperature schedule**: partner always uses `T = t0 * (t1/t0)**frac` (fixed exponential decay). src supports an additional `"fastsa"` schedule (`_fastsa_temp`, with parameters `_fastsa_k`, `_fastsa_c`, `_fastsa_steps`), selected via `self._sa_schedule == "fastsa"`, dispatched inside the loop — a schedule variant absent from partner entirely.
2. **Move set / extra move class**: src fires a probabilistic PARSAC-style "constraint-fixing" move (`_cfix_move`, gated by `self._cfix` / `self._cfix_p`) that is accepted unconditionally when it fires, bypassing the normal Metropolis test — not present in partner's loop at all.
3. **Adaptive move reweighting**: src tracks per-move-type proposal/accept/improvement counters (`_am_prop`, `_am_acc`, `_am_impr`) and periodically calls `_adaptive_reweight`/`_adaptive_reset` (`self._adaptive_moves`) to bias future move selection toward historically productive moves. Partner has no equivalent — its move selection is presumably static throughout the anneal.
4. **Early-stop / stall detection**: src has an "E2-adaptive early stop" block — tracks `stall_window`/`stall_eps` against `best_cost`, breaks the anneal loop early (`self._anneal_stalled = True`) once no improvement has been seen for `stall_window` seconds. Partner has no stall-detection break.
5. **Partner-only feature absent from src**: partner has a "late-phase violation-weight annealing" block (env-gated by `PARTNER_VW_ANNEAL` / `PARTNER_VW_ANNEAL_AT`) that runs most of the anneal at the true cost weight, then reprices violations upward near a configurable fraction (default 0.7) of the time budget to "squeeze survivors out," restoring the original weight and re-scoring `best` before returning so cross-restart comparisons stay weight-consistent. src's `_anneal` has no such phase-varying violation-weight mechanism.

Net: the two `_anneal` bodies are not a simple refactor of each other — they represent independently-evolved SA policies sharing only the outer loop shape and the Metropolis/temperature core arithmetic. A merge cannot be a mechanical text-diff; each divergent feature needs its own decision (keep/port/drop).

## Section 5 — Functionality that exists ONLY in the partner fork (must preserve on merge)

Grep across `partner/*.py` for `vkill`/`carve` shows these terms concentrated in a dedicated module, **not inlined in the legalizer file itself**:

- **`partner/vkill_claude.py`** (~1200+ lines) — a full violation-kill / repair pass distinct from anything in `column_slicing.py`: `kill_violations`/`_kill` entrypoints, boundary-violator detection (`_boundary_violators`), connected-component analysis (`_components`), exact violation counting (`_violations_exact`, `_grouping_count`, `_mib_count`), overlap checking (`_overlap_ok`), and a large set of *fix* passes — `_fix_boundary`, `_fix_grouping`, `_fix_mib`, `_fix_slivers` — plus geometric repair primitives: `_shove_rect`/`_shove_insert` (push blocks out of the way), `_weld_comp`/`_comp_attach_shove` (reattach disconnected components), `_wall_repack` (repack a wall line), `_free_gaps`, `_nudge_le`, and an LNS-style large-neighborhood-search pass `_lns_pass` (explicit comment: "net pull (weighted neighbor/pin centroid)... the move class the [SA] cannot reach"). Final-state guard check via `_final_guards_ok`. `vkill_claude.py` is imported by `legalizer_claude.py`, `my_opt_claude.py`, and `refiner_claude.py` — i.e. it is load-bearing infrastructure across the partner stack, not a standalone experiment.
- **`_perimeter_pack` / `_perimeter_seed_layouts`** (legalizer_claude.py:2267, 2560) — explicit wall-line packing for boundary-dense/wall-saturated instances, documented in its own docstring as targeting "the worst validation cases" where the normal column-slicing + SA move set "cannot recover" dropped tag pins. This is a structural (non-SA) placement strategy entirely absent from src.
- **`_layout_delta`** incremental-cache layout strategy (`_dc_enabled`/`_dc_cache`/`_dc_prev_cols`/`_dc_prev_sig`) — a different caching architecture from src's per-column memo (`_col_cache`), not present in src.
- **`_worker_refine`** — worker-pool entrypoint that runs a full legalize+refine pass per direct-model prediction, addressing (per its docstring) a specific starvation bug: "the in-process thread used to slice one budget across candidates, so most predictions never got refined at all." No equivalent in src (src's worker pool only does `_worker_solve`/racing, not a refine-specific worker call).
- **Late-phase violation-weight annealing** in `_anneal` (see section 4, item 5).

These five items are the concrete "must preserve on merge" list — none of them exist in `column_slicing.py`, and `vkill_claude.py` in particular is depended on outside the legalizer file, so it cannot be silently dropped in any merge that touches partner's stack.

## Porting-surface assessment

**Can likely port directly (mechanical, low interaction with partner-only features):**
- `_fastsa_temp`/`_cumulative` alternate temperature schedule — pure function, no dependency on partner's caching architecture.
- `_count_components`, `_width_optimize` — self-contained helpers.
- The FAST_EVAL per-column cache (`_layout_fast`/`_solve_column`/`_finish_layout`/`_col_touches_obstacle`) *could* replace partner's `_layout_full`/`_layout_delta` split, but this is a **caching-architecture swap, not an addition** — partner's `_dc_*` delta-cache and src's `_col_cache` are two different designs solving the same problem, so porting means choosing one, not merging both. Direct port risks silently changing partner's incremental-layout behavior (used elsewhere, e.g. by `_perimeter_pack`/`_worker_refine` callers) in ways that need re-verification.

**Need adaptation because they interact with partner-only features:**
- `_cfix_move`, adaptive move reweighting, and the stall-detection early-stop in `_anneal` all read/mutate state (`cur_cost`, `best`, `cols`) inside the same loop body that partner's late-phase violation-weight annealing also mutates (`self.v_weight`, `best_cost` rescoring on exit). Merging src's `_anneal` additions requires re-deriving the interaction with partner's `PARTNER_VW_ANNEAL` weight-repricing phase — e.g., does `_cfix_move`'s unconditional-accept semantics still make sense once `v_weight` is being annealed upward mid-run? This needs a design decision, not a text merge.
- The racing/parallel-restart machinery (`race_round1/2`, `_parallel_solve_racing`, `_race_configs`) would need to coexist with partner's `_worker_refine` racing/refine worker path — both are worker-pool orchestration layers with independent designs; merging risks duplicate/conflicting pool-management code (`init_worker_pool`, `_worker_init` are shared/common already, so this is specifically about the orchestration layer above them).
- `vkill_claude.py` is entirely orthogonal to `column_slicing.py`'s internals but is the single highest-value item to preserve; it is a repair layer that could in principle sit *after* whichever legalizer core is chosen (src's or a merged one), similar to how src's own `refine/` package sits after the column backbone. This is the item requiring the least code adaptation but the most decision-making about layering (should it become a repair stage of the production `refine/` pipeline in src, or stay a partner-only post-pass called by `my_opt_claude.py`/`refiner_claude.py`?).

## Effort estimate and risk assessment

Given `column_slicing.py` is the **production backbone** (per `CLAUDE.md`: evaluator scores 1.238–1.246 vs a 2.1158 no-backbone baseline, 100/100 feasible) — this is a "promoted on Evaluator Evidence" file, and any change requires a full-100 `bash scripts/eval_total.sh` A/B before promotion (never on assumption or supervised-loss-alone signal).

| Work item | Est. effort | Risk |
|---|---|---|
| Port `_fastsa_temp` schedule as an opt-in flag | 0.5 day | Low — additive, gated, easy to A/B in isolation |
| Port `_count_components`, `_width_optimize` | 0.5 day | Low — self-contained |
| Decide + port caching architecture (`_layout_fast`/`_col_cache` vs `_layout_delta`) | 2-3 days | **High** — this is a core hot-path rewrite; correctness bugs here (stale-cache reuse) directly threaten the 100/100 feasibility guarantee the production score depends on. Needs its own dedicated invariant tests before any eval run, in the spirit of `tests/test_refine_invariants.py`. |
| Reconcile `_anneal` divergence (cfix move, adaptive reweight, stall early-stop, vs partner's violation-weight annealing) | 3-4 days | **High** — five independently-evolved SA policy changes interacting in one loop; wrong interaction can silently degrade quality without throwing errors, only visible via full-100 eval, not unit tests. |
| Port `_perimeter_pack`/`_perimeter_seed_layouts` (wall-saturated instance handling) into src | 2-3 days | Medium — self-contained strategy but needs wiring into src's frame/obstacle model, which has diverged slightly (src's `_choose_frame`/obstacle handling has more machinery, e.g. `_col_touches_obstacle`, than partner's baseline). |
| Layer in `vkill_claude.py` as a post-backbone repair stage (or leave it in partner and just wire a call site) | 2-4 days depending on integration depth (call-site wiring only vs promoting to a `refine/`-style module) | Medium — orthogonal to the legalizer internals, but touches the production pipeline's repair/refine sequencing, which itself is Evaluator-Evidence gated. |
| Full-100 A/B evaluation cycles (one or more per merged chunk, per project convention of never promoting on assumption) | 0.5-1 day per eval cycle, likely 3-5 cycles across the above items | **High-stakes but mechanical** — required gate before any promotion; the real risk is *skipping* it, not the eval itself. |

**Total rough estimate**: 10-16 person-days for a full merge of every item above, dominated by the two "High" risk items (caching architecture + SA loop reconciliation) and multiple eval cycles. A narrower merge (e.g. only the low-risk schedule/helper functions, or only wiring in `vkill_claude.py` as an external post-pass without touching `column_slicing.py`'s internals) could be done in 2-4 days with correspondingly lower risk to the production 1.238-1.246 score.

**Key risk framing**: `column_slicing.py` is explicitly documented as "do not hand-edit without eval evidence" (CLAUDE.md module map). Every item above that touches this file's internals (the caching swap and the `_anneal` reconciliation especially) must clear a full-100 `bash scripts/eval_total.sh` A/B before promotion — no exceptions, since a regression here degrades the entire production score, not just one subsystem.
