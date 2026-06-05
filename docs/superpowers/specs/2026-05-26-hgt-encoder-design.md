# Local HGT Encoder Design

## Goal

Add a canonical HGT-style encoder option for Anchor-GNN Guidance that preserves the existing Production Solver Path while improving full-validation score beyond the current MPNN and graph-transformer checkpoints.

The first promotion target is to beat the current `.env` GNN baseline:

- Total Score: `2.3355`
- Total Score (No Runtime): `2.0864`
- Feasible: `100/100`

The stretch target remains a runtime-aware total near `1.5`, but this is not expected from the encoder alone. HGT v1 should first prove that relation-specific heterogeneous locality improves the no-runtime score without increasing runtime risk.

## Context

The current graph-transformer checkpoint underperforms the `.env` GNN checkpoint on full validation:

- Graph Transformer: `2.5819` total, `2.2024` no-runtime.
- GNN h192/500k: `2.3355` total, `2.0864` no-runtime.

The failure is not pure feasibility; both paths are `100/100` feasible. The transformer loses in weighted tail cases, especially where anchor/order/aspect guidance leads the relative-order decoder to worse area, HPWL, or soft-violation trade-offs. Its supervised validation loss can improve while evaluator score regresses, so HGT promotion must be based on evaluator evidence, not training loss alone.

## Decisions

1. Add a new selectable encoder type named `hgt`.
2. Implement HGT as local typed message passing over the explicit heterogeneous factor graph.
3. Do not add a global attention/refinement layer in HGT v1.
4. Preserve the existing AnchorGuidance output contract: anchor, priority, log-aspect, and pairwise-axis heads.
5. Preserve the existing decoder, repair, candidate policy, and evaluator path for the first HGT comparison.
6. Add low-risk decoder-aware training losses, but do not add evaluator-in-loop training or reinforcement learning.
7. Promote HGT only after full validation improves `total_score_no_runtime`, then review runtime-aware total and raw runtime.

## Architecture

HGT v1 consumes the existing heterogeneous floorplan factor graph:

- `block` nodes from block geometry, connectivity, and constraint features.
- `pin` nodes from terminal coordinates.
- `cluster` nodes from grouping constraints.
- `mib` nodes from same-shape constraints.
- `boundary` nodes from boundary code constraints.

The encoder uses only local typed edges:

- `block --connects--> block` for b2b connectivity.
- `pin --pin_connects--> block` and `block --pin_connects--> pin` for p2b connectivity.
- `block --member_of--> cluster` and `cluster --has_member--> block`.
- `block --same_shape_as--> mib` and `mib --has_member--> block`.
- `block --wants_boundary--> boundary` and `boundary --has_member--> block`.

Each HGT layer applies relation-specific attention or message projection keyed by the `(src_type, edge_type, dst_type)` meta-relation. b2b and p2b weights enter the attention score or message gate so strong connectivity has a stronger local influence than weak connectivity.

After the final HGT layer, only `block` embeddings are passed to the existing heads:

- `anchor_head`
- `priority_head`
- `aspect_head`
- `pair_head`

This keeps checkpoint inference compatible with `AnchorGuidance` and keeps the Production Solver Path unchanged.

## Locality Invariant

HGT v1 must preserve b2b/p2b local inductive bias. It must not become full block-to-block attention under a different name.

No message should flow between unrelated blocks unless it travels through:

- an actual b2b edge,
- a shared pin relation,
- a shared cluster factor,
- a shared MIB factor,
- or a shared boundary factor.

If global block attention is added later, it must be a separate HGT v2 design with a gated residual and its own evaluator evidence.

## Training

The base supervised losses remain:

- anchor center loss,
- log-aspect loss,
- priority loss,
- order auxiliary loss,
- edge delta loss,
- pairwise axis loss.

HGT v1 adds low-risk decoder-aware weighting:

- Tail-like or high-risk samples receive higher order and pairwise weights.
- The implemented low-risk hook is HGT-only high-risk order/pairwise loss multipliers, controlled by `--high-risk-order-multiplier`, `--high-risk-pairwise-multiplier`, `--high-risk-min-blocks`, and `--high-risk-min-constraints`.
- Dirty training samples continue to suppress unreliable order/pairwise supervision according to the existing clean-sample policy.
- Optional GNN teacher distillation may train HGT to match the currently stronger GNN checkpoint's anchor, order, and pairwise guidance on high-risk samples.

The distillation target is the guidance consumed by the decoder, not a complete placement. Soft-constraint satisfaction remains the responsibility of constraint-aware decoding, repair, and scoring.

## Evaluation

Promotion requires full validation, not supervised validation loss.

Report at minimum:

- total score,
- total score no-runtime,
- feasible count,
- average, median, p90, and max runtime,
- tail case costs for IDs 95-99,
- HPWL gap, area gap, and soft-violation ratio for weighted contributors.

The first acceptance gate is:

- HGT no-runtime total below `2.0864`.
- HGT runtime-aware total below `2.3355` or a clear no-runtime win with raw runtime no worse than the GNN baseline.

The stretch gate for later work is runtime-aware total near `1.5`, but that likely requires decoder/candidate/refinement improvements beyond HGT v1.

## Alternatives Considered

### Ablation-Lite HGT

Use typed edges but share most projection parameters. This is cheaper, but it risks failing to learn relation semantics and would not answer whether a real HGT encoder helps.

### HGT With Global Refinement

Add block-only global attention after local HGT. This may help long-range ordering and area shape, but it risks washing out b2b/p2b locality and makes the first comparison harder to interpret. It is deferred until local HGT has clear evaluator evidence.

### Current Graph Transformer Plus More Bias

Improve the existing block-only transformer with better attention bias. This is lower risk, but the project direction is to test a canonical heterogeneous encoder that keeps constraints as first-class graph factors.

## Documentation Updates

`CONTEXT.md` should define Local HGT Encoder as a Selectable Anchor-GNN Encoder variant that consumes the Heterogeneous Floorplan Graph while preserving the AnchorGuidance contract.

`docs/optimization-notes.md` should record that HGT v1 is local-only, relation-specific, and evaluated by full no-runtime score before runtime-aware promotion.

README updates should wait until implementation exists, because the current user-facing training script still supports only `mpnn` and `graph-transformer`.

## Success Criteria

- HGT can be selected as an encoder without changing solver output format.
- Existing MPNN and graph-transformer checkpoints remain loadable.
- HGT training uses explicit hetero graph inputs rather than clique-expanded block context.
- Full validation shows whether HGT improves the no-runtime score against the `.env` GNN baseline.
- Runtime-aware score and raw runtime are reported before any default checkpoint change.

## Self-Review

- Placeholder scan: no placeholders remain.
- Internal consistency: HGT v1 is local-only throughout; global refinement is explicitly deferred.
- Scope check: the design is focused on encoder, training loss, checkpoint compatibility, evaluation, and docs.
- Ambiguity check: "HGT" means relation-specific typed local message passing over the heterogeneous factor graph, not full block attention.
