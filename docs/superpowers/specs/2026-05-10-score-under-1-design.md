# Score Under 1 Design

## Goal

Reduce `iccad2026_evaluate.py --evaluate` total score below 1 through real placement-quality improvements, not runtime calibration or artificial delay.

## Context

The v3 wrapper and `ArchitectureV3Optimizer` are active. The previous `1.8528` score depended on runtime calibration, which has been removed. The honest baseline returns to the placement quality of the Anchor-GNN guided relative-order decoder plus repair path.

Validation IDs 99 and 98 dominate the total score because the contest score uses exponential block-count weighting. Existing ablation notes show that enabling beam candidates, running both profiles, no-guidance candidates, naive external boundary moves, and tall-layout pair bias did not improve the dominant tail cases.

## Decisions

1. Runtime calibration is removed and must stay removed.
2. Add repair diagnostics before training or decoder changes, so improvements can be attributed to the model, decoder, or repair.
3. Prioritize relative-order learning before slot ranking.
4. Delay learned slot ranking until the pairwise relation path proves useful.

## Repair Diagnostics

The solver will support opt-in JSONL tracing through an environment variable such as `FLOORSET_REPAIR_TRACE_JSONL`. Each candidate placement should record:

- `before_repair`: overlap count, boundary violations, group violations, MIB violations, HPWL proxy, bbox area.
- `after_repair`: same metrics after `repair_placement`.
- `delta`: average moved Manhattan distance, bbox area delta, HPWL proxy delta.
- candidate metadata: profile name, candidate kind, block count, checkpoint presence.

Tracing must not affect default evaluator output or runtime when disabled.

## Pairwise Relative-Order Learning

Current `order_aux_loss()` only pressures predicted anchors to preserve pairwise order. The next model upgrade should add an explicit pairwise relation classifier:

- Input: encoded block embeddings for `(i, j)` plus geometric/constraint deltas.
- Labels from expert `fp_sol`: horizontal relation, vertical relation, and ambiguous masks.
- Output: logits consumed by `relative_order.py` to decide whether a pair becomes an x-precedence or y-precedence edge.

The decoder should blend pairwise logits with current heuristic scores at first, then allow a controlled switch once trace metrics improve.

## Slot Ranking

Slot ranking stays a later phase. It becomes worthwhile only after pairwise ordering reduces pre-repair overlap/soft counts. Candidate features can reuse `constructive.py` concepts: slot kind, anchor distance, bbox delta, HPWL proxy, group touch, boundary satisfied, MIB compatible, and overlap-free status.

## Success Criteria

- No artificial sleeping or runtime calibration remains in production solver code.
- Repair trace can explain before/after repair effects for IDs 97-99 and arbitrary cases.
- Pairwise relation model is trainable from `fp_sol` and checkpoint-compatible through a clear version path.
- Honest `--evaluate` score improves versus the non-calibrated v3 baseline.
- Target outcome remains total score `< 1`, but intermediate work must report real HPWL, area, soft-violation, and runtime trade-offs.

Execution note: the first implementation pass removed calibration and improved honest full score to the `2.3262`-`2.3991` range depending on runtime noise; target `< 1` remains open.

## Risks

- Existing anchor order accuracy is already high, so a pairwise head must affect decoder decisions, not only training loss.
- Repair can make layouts feasible while degrading HPWL/area. Trace metrics must be used before accepting decoder changes.
- Full slot imitation is larger than needed and risks overfitting before we understand relative-order failures.

## Self-Review

- Placeholder scan: no placeholders remain.
- Scope check: one focused project, split into diagnostics first and learned ordering second.
- Ambiguity check: runtime calibration is explicitly banned; slot ranking is explicitly later.
