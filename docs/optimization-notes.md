# FloorSet Optimization Notes

## Current Baseline

- Historical v3 runtime-aware baselines: `2.6509` immediately after the v3 rename, `2.5256` after removing runtime calibration, and `2.3262` to `2.3991` after local-proxy overlap repair (`100/100` feasible).
- Those v3 totals used the local evaluator runtime factor, whose denominator is the solver's own validation-run median runtime. Treat them as runtime-aware risk signals, not as the primary architecture-quality metric.
- Current architecture comparisons should report the no-runtime total beside the local runtime-aware total before accepting or rejecting a candidate path.
- 2026-05-11 updated no-runtime promotion pass promotes `checkpoints/gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt`: `1.9560` no-runtime total versus `1.9848` for `gnn_best.pt`, `1.9953` for the 500k ep4 h192 checkpoints, and `2.2768` for `gnn_best_0510_ns200000_ep10_h192_l6_acc32.pt`.
- 2026-05-12 historical production note used `checkpoints/gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt`; the 2026-06-05 v10 rerun below supersedes that default recommendation.
- Rechecking `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1` with the promoted checkpoint tied the no-runtime total at `1.9560` and had zero no-runtime delta on IDs 95-99, so the matrix remains opt-in.
- 2026-05-21 No-Checkpoint Guidance Mode baseline remains feasible (`100/100`) but much worse than the configured GNN path: `4.6289` no-runtime total versus `2.0326` for `.env` checkpoint `gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt`. Repair removes all traced overlaps and sharply reduces soft violations, but no-GNN final HPWL/area gaps are still too large, especially ID 98.
- 2026-05-21 no-guidance boundary-edge shrink improves No-Checkpoint Guidance Mode no-runtime total from `4.6289` to `4.5038`, mainly by improving ID 98 (`8.2411` to `7.7030`; area gap `6.3340` to `6.0228`). The local runtime-aware total worsened (`6.2174` to `6.5204`), so treat this as a no-runtime repair-ablation win, not a submission-safe default for runtime-aware scoring.
- 2026-05-21 configured-checkpoint high-risk portfolio changed its default repair profiles to normal-only. Full validation improved local runtime-aware total from `3.2697` to `2.3086` while keeping no-runtime quality effectively flat (`2.0326` to `2.0362`), because ID 99 runtime dropped from `10.75s` to `2.15s` with cost-no-runtime only moving from `1.9983` to `2.0040`.
- 2026-05-21 Sample-Local Quality Portfolio v1 is implemented but remains opt-in. It improved configured-checkpoint no-runtime total to `2.0311` by improving ID 99 HPWL, but local runtime-aware total regressed to `2.5615`; a lower-budget run still regressed total to `2.4574`.
- 2026-06-05 v10 scoring changes total weighting from `exp(n)` to `exp(n/12)`. Large cases still matter, but ID 99 alone no longer dominates the total score under the official formula.
- Under v10 weights, treat IDs 95-99 as diagnostics for the largest bucket, not as the whole promotion surface. Full-validation `total_score_no_runtime` remains the primary architecture metric.
- 2026-06-05 fresh v10 neutral-runtime rerun with `src/architecture_v5_optimizer.py`:
  `.env` checkpoint `gnn_best_0512_ns500000_ep4_h192_l6_acc32.pt` scores `2.2337`;
  `gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt` scores `2.3390`;
  `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1` ties `2.2337`;
  `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` slightly improves to `2.2299`;
  No-Checkpoint Guidance Mode is still much worse at `5.0509`.
- 2026-06-05 best-checkpoint sweep over dated `*best*.pt` checkpoints from 0512 onward promotes `checkpoints/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt` as the current global best: `2.1194` v10 no-runtime total, `2.6641` local runtime-aware total, `100/100` feasible. It beats `gnn_best_0514_ns800000_ep4_h192_l6_acc32.pt` (`2.1749` no-runtime), `gnn_hgt_best_val_loss_0603_ns800000_ep10_enchgt_h256_l4_acc32_bs8_heads4.pt` (`2.2144`), and the prior `.env` 0512 checkpoint (`2.2337`).
- 2026-06-05 follow-up `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` full run on the promoted Graph Transformer 0521 checkpoint regressed from `2.1194` to `2.2446` v10 no-runtime and from `2.6641` to `2.9168` total while staying `100/100` feasible. Keep quality portfolio opt-in only; the older 0512-checkpoint gain does not transfer to the current production default.
- 2026-06-05 opt-in `FLOORSET_ENABLE_V10_SOFT_REPAIR=1` full run on the promoted Graph Transformer 0521 checkpoint improved v10 no-runtime slightly from `2.1194` to `2.1082`, but worsened runtime-aware total from `2.6641` to `2.7881`. Runtime tail moved from avg/p90/max `1.23s`/`2.23s`/`6.80s` to `1.40s`/`2.54s`/`9.47s`; keep it opt-in and treat this as evidence that runtime-tail budget clamp must land before broader soft repair or grouping-biased decoding.
- In that rerun, ID 98 and ID 99 are not score-dominant singletons: their fixed v10 weights are `7.3580%` and `7.9975%`, and their default-run score contributions are `8.11%` and `7.53%`. The 95-99 bucket contributes `31.23%` of the default total, so use it as a diagnostic bucket rather than a promotion gate.

## Local Score Policy

- Use this tuning order: feasible count -> full v10 neutral/no-runtime total -> 95-99 diagnostic bucket -> soft violations -> HPWL/area -> raw runtime -> local runtime-aware score.
- Keep the official cost formula and runtime-aware total in the evaluator, but do not use the local runtime median artifact as the only decision-maker for beam, repair, or large-case candidate policy.
- `--score` saved-solution evaluation is effectively no-runtime unless stored runtimes are deliberately reintroduced, because saved positions are scored with neutral runtime.
- Checkpoint selection currently uses supervised validation loss, not no-runtime evaluator score. Treat it as a training health metric only; promote a checkpoint only after `--evaluate` or saved-solution scoring improves the full v10 neutral/no-runtime total. IDs 95-99 can explain the result, but ID 98/99 single-case wins are insufficient by themselves.
- No-Checkpoint Guidance Mode is the preferred ablation for isolating deterministic decoder and repair headroom. Run it with an intentionally missing checkpoint and repair tracing, then compare against the default checkpoint using the same evaluator settings.
- Repair-only work should be accepted only when it improves No-Checkpoint Guidance Mode without materially regressing the default Anchor-GNN guided path. Report before-repair and after-repair metrics, final no-runtime score, raw runtime, and large-case tail deltas.

## Ablation Results To Avoid Repeating

- Do not train on local WSL. Pairwise-head effectiveness must be judged from remote-server GPU training artifacts once available.
- Do not treat the temporary local pairwise fine-tune as conclusive against pairwise learning; it was CPU/small-run evidence only.
- If adding heavier deterministic repair, gate it through the shared v10 risk budget: score share, constraint density, and net density. Do not use ID 98/99 or a block-count-only threshold as the sole trigger.
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
- High-risk portfolio repair profiles are now normal-only by default. The heavier `boundary_first`, `grouping_first`, `quality_refine`, and `large_boundary` profiles remain available through `FLOORSET_HIGH_RISK_REPAIR_PROFILES`, but earlier full configured-checkpoint evaluation showed they were spending too much runtime on tail cases without improving no-runtime quality. Re-test them with v10 risk-budget gating before reviving them.
- Surrogate guidance is opt-in through `FLOORSET_ENABLE_SURROGATE_GUIDANCE=1`. It can help individual no-checkpoint cases, but the current surrogate worsened ID 99 and must not shadow the clean No-Checkpoint Guidance Mode baseline by default.
- Sample-Local Quality Portfolio is opt-in through `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto|1`. Under the 2026-06-05 v10 neutral-runtime rerun, `auto` slightly improved the older 0512 checkpoint from `2.2337` to `2.2299`, but the promoted Graph Transformer 0521 follow-up regressed to `2.2446` no-runtime and `2.9168` total. Do not enable it by default without new full-validation evidence against the current production checkpoint.
- V10 soft repair is opt-in through `FLOORSET_ENABLE_V10_SOFT_REPAIR=1`. The first full validation gained `-0.0112` no-runtime total but lost `+0.1240` runtime-aware total, with IDs 95-99 flat or worse on no-runtime cost and tail regressions concentrated around IDs 88, 89, 86, 91, and 95. Do not enable it by default; first add a hard runtime-tail budget clamp and then re-test a narrower gate.
- Opt-in knobs kept for future remote/server ablation: `FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP`, `FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS`, `FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_REFINE`, `FLOORSET_OVERLAP_REPAIR_CANDIDATES`.

## Current v4 Strategy

- Keep the active solver path in `floorset_arch`.
- Rename the contest wrapper and optimizer class from v3 to v4.
- Preserve the current Anchor-GNN guided relative-order decoder for large cases.
- Do not use runtime calibration or artificial sleep. Score improvements after v4 must come from placement quality, repair quality, or learned ordering/ranking.
- Keep pairwise-head plumbing checkpoint-compatible, but do not depend on it for local WSL optimization until remote training finishes.
- Use local-proxy overlap relocation instead of first legal frontier relocation; it improves large-case quality without the full-HPWL runtime blow-up.
- The current `.env` checkpoint is `checkpoints/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt`, selected by the 2026-06-05 v10 best-checkpoint sweep. Do not promote future training outputs by supervised validation loss alone; require evaluator evidence, especially full v10 neutral/no-runtime total, before changing the production default.
- A sequential, sample-local relative-order candidate matrix is implemented behind `FLOORSET_ENABLE_LARGE_CASE_CANDIDATES=1`: adaptive profile plus the opposite forced `soft`/`compact` profile when it is not a duplicate, with normal repair by default. Full validation regressed when this ran by default, so the production default keeps the faster adaptive single-candidate path.
- Large-boundary repair remains opt-in via `FLOORSET_LARGE_CASE_REPAIR_PROFILES=normal,large_boundary`; previous full validation showed defaulting it improved soft counts but regressed local runtime-aware score, so it needs no-runtime and raw-runtime review before becoming default.
- Under no-runtime tuning, the adaptive single-profile default is reasonable as the submission-safe path, but it should not be treated as settled architecture. The 2026-06-05 v10 rerun found the large-case matrix tied the current default at `2.2337`; re-run large-boundary profiles with v10 risk-budget gating before rejecting them.
- Boundary repair now skips already-satisfied boundary blocks only for `block_count >= 118` and widens the free cross-axis search for large edge-constrained boundary moves. This remains useful history, but future promotion must be justified by full v10 total and risk-budget diagnostics rather than an ID 99-only improvement.
- Keep beam and no-guidance candidates opt-in through environment variables; do not enable them by default without a full-score win.
- During supervised training, do not fully trust any `fp_sol` sample that violates boundary, grouping, or MIB constraints. The measured clean ratio is too low for strict skipping as the default, so current training uses weighted dirty geometry samples while suppressing dirty order/pairwise supervision; use `CLEAN_SAMPLE_POLICY=strict` only for clean-only experiments.
- Next deterministic optimization should target runtime-tail budget clamp before widening soft repair. Apply the clamp after repair acceptance and the v10 risk gate, with per-tier caps on extra passes/candidates and a hard stop for repeated no-acceptance loops. After that, revisit decoder-side grouping adjacency bias as a cheaper way to reduce group soft violations before repair has to spend runtime.
- Large-case repair should report soft-count deltas first, then bbox/HPWL deltas, because `V_rel` enters score through exponential penalty.

## v5 Encoder Experiment

- v5 keeps `floorset_arch` as the production package and adds `src/architecture_v5_optimizer.py` as the contest wrapper.
- Training now supports `--encoder mpnn|graph-transformer|hgt`. The default 1M full-data run remains the lower-risk MPNN baseline with `hidden_dim=256`, `layers=6`, and `epochs=3`.
- Graph Transformer checkpoints are still Anchor-GNN Guidance checkpoints. Promote them only after full evaluator evidence, especially `total_score_no_runtime`, because the decoder and repair path still determine final placement quality.
- 2026-05-26 Graph Transformer underperformed the then-current GNN checkpoint under the old scoring/evaluator setup, but the 2026-06-05 v10 best-checkpoint sweep supersedes that conclusion: `gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt` is now the global best by full v10 no-runtime total.
- Local HGT is implemented as relation-specific typed local attention over block, pin, cluster, MIB, and boundary nodes, with canonical relation specs saved in checkpoints and no global attention/refinement in v1. It preserves the AnchorGuidance output contract and uses low-risk decoder-aware training hooks: HGT-only high-risk order/pairwise loss multipliers are exposed through `--high-risk-order-multiplier` and `--high-risk-pairwise-multiplier`; optional GNN teacher distillation remains future work.
- Promote a Local HGT checkpoint only after full validation beats the current `.env` Graph Transformer baseline's `2.1194` v10 no-runtime total, then review runtime-aware total and raw runtime. Do not promote by supervised validation loss alone.

## Review Notes

- Runtime calibration was removed because it was a median-runtime exploit and would be fragile under official review or absolute-runtime scoring.
- Local full-evaluate runtime factor is also median-based, so every ablation should report no-runtime score and raw runtime separately.
- Evaluator runtime measurement uses monotonic `time.perf_counter()`; `time.time()` produced occasional negative runtimes in WSL and should not be used for solver timing.
- Repair tracing is opt-in via `FLOORSET_REPAIR_TRACE_JSONL` and records before/after overlap, boundary, group, MIB, HPWL proxy, bbox area, and movement deltas.
- Future score work should report both pre-repair and post-repair metrics so improvements are attributable to the model, decoder, or repair.
