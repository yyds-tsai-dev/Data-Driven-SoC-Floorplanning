# FloorSet Optimization Notes

## Current Baseline

- Historical v3 runtime-aware baselines: `2.6509` immediately after the v3 rename, `2.5256` after removing runtime calibration, and `2.3262` to `2.3991` after local-proxy overlap repair (`100/100` feasible).
- Those v3 totals used the local evaluator runtime factor, whose denominator is the solver's own validation-run median runtime. Treat them as runtime-aware risk signals, not as the primary architecture-quality metric.
- Current architecture comparisons should report the no-runtime total beside the local runtime-aware total before accepting or rejecting a candidate path.
- 2026-05-11 updated no-runtime promotion pass promotes `checkpoints/gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt`: `1.9560` no-runtime total versus `1.9848` for `gnn_best.pt`, `1.9953` for the 500k ep4 h192 checkpoints, and `2.2768` for `gnn_best_0510_ns200000_ep10_h192_l6_acc32.pt`.
- 2026-05-12 production default checkpoint is `checkpoints/gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt`; keep using evaluator evidence, especially `total_score_no_runtime`, before future default changes.
- Rechecking `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1` with the promoted checkpoint tied the no-runtime total at `1.9560` and had zero no-runtime delta on IDs 95-99, so the matrix remains opt-in.
- Score was dominated by validation IDs 99 and 98 because total score is exponentially weighted by block count.
- ID 99 contributed about `1.76 / 2.65`; ID 98 contributed about `0.59 / 2.65`.

## Local Score Policy

- Use this tuning order: feasible count -> large-case no-runtime score -> total no-runtime score -> soft violations -> HPWL/area -> raw runtime -> local runtime-aware score.
- Keep the official cost formula and runtime-aware total in the evaluator, but do not use the local runtime median artifact as the only decision-maker for beam, repair, or large-case candidate policy.
- `--score` saved-solution evaluation is effectively no-runtime unless stored runtimes are deliberately reintroduced, because saved positions are scored with neutral runtime.
- Checkpoint selection currently uses supervised validation loss, not no-runtime evaluator score. Treat it as a training health metric only; promote a checkpoint only after `--evaluate` or saved-solution scoring reports no-runtime improvements on dominant large cases.

## Ablation Results To Avoid Repeating

- Do not train on local WSL. Pairwise-head effectiveness must be judged from remote-server GPU training artifacts once available.
- Do not treat the temporary local pairwise fine-tune as conclusive against pairwise learning; it was CPU/small-run evidence only.
- If adding heavier deterministic repair, gate it by block count or constraint statistics for dominant large cases. Do not spend extra runtime on small cases.
- Extra repair acceptance should be soft-first: prefer moves that reduce boundary/group/MIB violations, guarded by proxy cost. If soft count does not improve, accept only clear bbox/HPWL proxy wins.
- Enabling beam candidates alone did not improve IDs 97, 98, or 99 and increased runtime.
- Evaluating both `soft` and `compact` profiles did not improve ID 98 or 99; it only increased runtime.
- `soft` helped ID 97, `compact` helped ID 98, and ID 99 stayed best with the default adaptive/soft path.
- No-guidance candidates did not improve IDs 96-99 and increased runtime.
- Naive external boundary moves reduced neither score nor violations; they expanded the bounding box and made other boundary constraints fail.
- Tall-layout pair-orientation bias worsened ID 99.
- Pairwise relation fine-tuning from `gnn_best.pt` reached high validation pair accuracy, but the temporary checkpoint worsened ID 99 (`3.1270`) and ID 98 (`2.6968`), so do not replace `checkpoints/gnn_best.pt` with that run.
- Preplaced-frame anchor clamp is available behind `FLOORSET_PREPLACED_FRAME_CLAMP=1`, but it worsened ID 99 in the current decoder path and must remain opt-in.
- Increasing `FLOORSET_BOUNDARY_AXIS_CAP` to `160` improved single-case ID 99 quality but added enough runtime risk that it should not become the default without a full-score win.
- Sparse large-case overlap cap `64` improved ID 98 single-case quality but worsened full-score runtime trade-off; keep it as `FLOORSET_OVERLAP_REPAIR_CANDIDATES=64` only.
- Large-case high-cap boundary pass/refine remains opt-in because default full evaluation regressed when the extra search ran on every large case.
- Opt-in knobs kept for future remote/server ablation: `FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP`, `FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS`, `FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_REFINE`, `FLOORSET_OVERLAP_REPAIR_CANDIDATES`.

## Current v4 Strategy

- Keep the active solver path in `floorset_arch`.
- Rename the contest wrapper and optimizer class from v3 to v4.
- Preserve the current Anchor-GNN guided relative-order decoder for large cases.
- Do not use runtime calibration or artificial sleep. Score improvements after v4 must come from placement quality, repair quality, or learned ordering/ranking.
- Keep pairwise-head plumbing checkpoint-compatible, but do not depend on it for local WSL optimization until remote training finishes.
- Use local-proxy overlap relocation instead of first legal frontier relocation; it improves dominant tail quality without the full-HPWL runtime blow-up.
- Keep `checkpoints/gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt` as the default checkpoint. Do not promote future training outputs by supervised validation loss alone; require evaluator evidence, especially `total_score_no_runtime`, before changing the production default.
- A sequential, sample-local relative-order candidate matrix is implemented behind `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1`: adaptive profile plus the opposite forced `soft`/`compact` profile when it is not a duplicate, with normal repair by default. Full validation regressed when this ran by default, so the production default keeps the faster adaptive single-candidate path.
- Large-boundary repair remains opt-in via `FLOORSET_LARGE_CASE_REPAIR_PROFILES=normal,large_boundary`; previous full validation showed defaulting it improved soft counts but regressed local runtime-aware score, so it needs no-runtime and raw-runtime review before becoming default.
- Under no-runtime tuning, the adaptive single-profile default is reasonable as the submission-safe path, but it should not be treated as settled architecture. Re-run the large-case matrix and large-boundary profiles using `total_score_no_runtime` before rejecting them.
- Boundary repair now skips already-satisfied boundary blocks only for `block_count >= 118` and widens the free cross-axis search for large edge-constrained boundary moves. This preserved the useful ID 99 boundary improvement without moving the whole default path to the expensive high-cap search.
- Keep beam and no-guidance candidates opt-in through environment variables; do not enable them by default without a full-score win.
- During supervised training, do not fully trust any `fp_sol` sample that violates boundary, grouping, or MIB constraints. The measured clean ratio is too low for strict skipping as the default, so current training uses weighted dirty geometry samples while suppressing dirty order/pairwise supervision; use `CLEAN_SAMPLE_POLICY=strict` only for clean-only experiments.
- Next deterministic optimization should target large-case boundary/group repair for `block_count >= 118` or similar constraint-stat triggers, not `test_id`-specific logic.
- Large-case repair should report soft-count deltas first, then bbox/HPWL deltas, because `V_rel` enters score through exponential penalty.

## Review Notes

- Runtime calibration was removed because it was a median-runtime exploit and would be fragile under official review or absolute-runtime scoring.
- Local full-evaluate runtime factor is also median-based, so every ablation should report no-runtime score and raw runtime separately.
- Evaluator runtime measurement uses monotonic `time.perf_counter()`; `time.time()` produced occasional negative runtimes in WSL and should not be used for solver timing.
- Repair tracing is opt-in via `FLOORSET_REPAIR_TRACE_JSONL` and records before/after overlap, boundary, group, MIB, HPWL proxy, bbox area, and movement deltas.
- Future score work should report both pre-repair and post-repair metrics so improvements are attributable to the model, decoder, or repair.
