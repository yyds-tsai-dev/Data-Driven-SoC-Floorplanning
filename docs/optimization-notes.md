# FloorSet Optimization Notes

## Current Baseline

- `iccad2026_evaluate.py --evaluate src/architecture_v3_optimizer.py` scored `2.6509` immediately after the v3 rename.
- After removing runtime calibration, the honest v3 score was `2.5256` in the first full run.
- After local-proxy overlap repair, honest full runs scored `2.3262` to `2.3991` (`100/100` feasible). The spread came from evaluator runtime noise, not artificial calibration.
- Later default full runs with large-case repair experiments left opt-in scored `2.3689` to `2.4774`; no new default path beat the local-proxy baseline reliably.
- Score was dominated by validation IDs 99 and 98 because total score is exponentially weighted by block count.
- ID 99 contributed about `1.76 / 2.65`; ID 98 contributed about `0.59 / 2.65`.

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

## Current v3 Strategy

- Keep the active solver path in `floorset_arch`.
- Rename the contest wrapper and optimizer class from v2 to v3.
- Preserve the current Anchor-GNN guided relative-order decoder for large cases.
- Do not use runtime calibration or artificial sleep. Score improvements after v3 must come from placement quality, repair quality, or learned ordering/ranking.
- Keep pairwise-head plumbing checkpoint-compatible, but do not depend on it for local WSL optimization until remote training finishes.
- Use local-proxy overlap relocation instead of first legal frontier relocation; it improves dominant tail quality without the full-HPWL runtime blow-up.
- Next deterministic optimization should target large-case boundary/group repair for `block_count >= 118` or similar constraint-stat triggers, not `test_id`-specific logic.
- Large-case repair should report soft-count deltas first, then bbox/HPWL deltas, because `V_rel` enters score through exponential penalty.

## Review Notes

- Runtime calibration was removed because it was a median-runtime exploit and would be fragile under official review or absolute-runtime scoring.
- Repair tracing is opt-in via `FLOORSET_REPAIR_TRACE_JSONL` and records before/after overlap, boundary, group, MIB, HPWL proxy, bbox area, and movement deltas.
- Future score work should report both pre-repair and post-repair metrics so improvements are attributable to the model, decoder, or repair.
