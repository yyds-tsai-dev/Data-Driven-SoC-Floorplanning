# V11 Graph-Conditioned Diffusion Tree Decoder Design

## Goal

Replace the v5 AnchorGuidance-centered production path with a v11 diffusion-centered solver path that can sample floorplan candidates directly as block centers, aspect ratios, and pairwise/topology signals.

The first implementation must compare only two diffusion variants:

- A. Raw diffusion baseline: raw block, pair, mask, and global instance features condition the denoiser.
- B. HGT-lite conditioned diffusion: a shallow typed graph conditioner adds cached block context, while raw side channels stay visible to every denoising step.

Do not implement the deeper or overbuilt HGT variant in v11 milestone 1. The first question is whether lightweight heterogeneous graph conditioning improves evaluator-facing floorplan quality over raw diffusion, not whether a larger graph encoder can absorb every signal.

The promotion metric remains full-validation `total_score_no_runtime`, with raw runtime tails as a gate and runtime-aware total as submission sanity context.

## Research Basis

Graph-conditioned diffusion is aligned with recent chip placement work. `Chip Placement with Diffusion Models` formulates placement as sampling coordinates conditioned on a netlist graph and object sizes, and uses graph convolution plus attention inside an efficient denoising model.

The design should not become a deep stack of graph abstractions. GNN literature flags two relevant risks:

- Over-squashing: many long-range or high fan-in signals are compressed into fixed-width node embeddings.
- Over-smoothing: deeper message passing can make node states less distinguishable.

Heterogeneous Graph Transformer style relation-specific message passing is useful for FloorSet because pins, clusters, MIB groups, boundary constraints, and block connectivity are semantically different. In v11, this should be used as a shallow conditioner, not as a second complete placement model.

References:

- Chip Placement with Diffusion Models: https://arxiv.org/html/2407.12282v2
- Understanding over-squashing and bottlenecks on graphs via curvature: https://arxiv.org/abs/2111.14522
- Graph neural networks exponentially lose expressive power for node classification: https://arxiv.org/abs/1905.10947
- A Survey on Oversmoothing in Graph Neural Networks: https://arxiv.org/abs/2303.10993
- Heterogeneous Graph Transformer: https://arxiv.org/abs/2003.01332

## Decisions

1. Rename the production optimizer class to `ArchitectureV11Optimizer`.
2. Keep a v5 compatibility alias or shim so old imports and tests do not break immediately.
3. Add evaluator-facing `src/architecture_v11_optimizer.py`.
4. Update active validation and evaluation scripts to use the v11 wrapper.
5. Keep `parse_instance()` returning `Instance`; it must not return a heterogeneous graph or diffusion-specific object.
6. Add `build_diffusion_graph_inputs(inst)` as the deterministic bridge from `Instance` to model inputs.
7. Do not convert diffusion outputs into `AnchorGuidance`.
8. Keep `Placement` only as the exact candidate, repair, diagnostics, ranking, and evaluator boundary.
9. Introduce tensor-level diffusion contracts so most sampling and prefiltering happens before converting top candidates to `Placement`.
10. Use `tree_sol` and `metrics_sol` in training, but avoid train/test leakage by keeping solution-only fields out of inference inputs.
11. If `FLOORSET_DIFFUSION_CHECKPOINT` is missing or incompatible, v11 falls back to the preserved v5 path and marks the trace/eval metadata with an explicit fallback reason.
12. Full eval must emit PNGs for the ten highest-cost predicted floorplans. Single-case eval must emit that case's predicted floorplan PNG.

## Out Of Scope For Milestone 1

- Variant C or any deeper/full HGT ablation.
- Pure tree generation as the only model state.
- Reinforcement learning or evaluator-in-loop training.
- Replacing the official/evaluator-facing output format.
- Treating `metrics_sol` solution quality fields as inference conditioning.
- Validation-ID-specific production logic.

## Data Model Boundaries

The current system has four important layers:

```text
evaluator tensors
  -> Instance
  -> neural graph/features
  -> Placement / Rect
  -> evaluator position list
```

v11 preserves this separation:

- `Instance` remains the canonical parsed problem instance.
- `DiffusionGraphInputs` is the model input contract.
- `DiffusionPlacementPrior` and `PlacementTensorBatch` are diffusion output and candidate contracts.
- `Placement` is created only for shortlisted candidates that need exact repair, diagnostics, ranking, visualization, or evaluator output.

This avoids making `Placement` the only common data model. `Placement` is precise and evaluator-compatible, but inefficient for batch diffusion sampling and tensor-level prefiltering.

## Diffusion Graph Inputs

Add a new module:

```text
src/floorset_arch/diffusion/graph_inputs.py
```

The primary API is:

```python
def build_diffusion_graph_inputs(inst: Instance, device: torch.device | None = None) -> DiffusionGraphInputs:
    """Return deterministic diffusion inputs derived only from the parsed instance."""
```

The output should include:

```python
@dataclass(frozen=True)
class DiffusionGraphInputs:
    node_features: dict[str, torch.Tensor]
    edge_index: dict[Relation, torch.Tensor]
    edge_attr: dict[Relation, torch.Tensor]
    relation_specs: tuple[Relation, ...]

    raw_block_features: torch.Tensor
    raw_pair_features: torch.Tensor
    pair_index: torch.Tensor

    area: torch.Tensor
    scale: float
    fixed_mask: torch.Tensor
    preplaced_mask: torch.Tensor
    movable_mask: torch.Tensor

    boundary_codes: torch.Tensor
    cluster_ids: torch.Tensor
    mib_ids: torch.Tensor
    global_features: torch.Tensor
```

`Relation` is the same conceptual tuple used by the current HGT path:

```python
tuple[str, str, str]
```

The typed relations should cover:

- `block, connects, block`
- `pin, pin_connects, block`
- `block, pin_connects, pin`
- `block, member_of, cluster`
- `cluster, has_member, block`
- `block, same_shape_as, mib`
- `mib, has_member, block`
- `block, wants_boundary, boundary`
- `boundary, has_member, block`

`build_diffusion_graph_inputs()` may reuse the existing heterogeneous graph builder internally, but it must not reuse the anchor-specific return dataclasses as its public contract. The v11 model should not depend on `AnchorHGTGraphInputs`, `AnchorTransformerGraphInputs`, or `AnchorGuidance`.

## Raw Side Channels

Raw features must remain available to the denoiser, even in variant B. This is the main guard against information disappearance or semantic drift.

At minimum, every denoising step should have direct access to:

- block area and normalized sqrt area
- fixed, preplaced, movable masks
- boundary bit/code features
- cluster and MIB membership IDs or compact encodings
- weighted b2b and p2b degree features
- pin coordinate aggregates
- pair indices and pair raw features
- global instance statistics that are computable at inference time

The HGT-lite conditioner can enrich this state, but it must not be the only carrier of constraints.

## Model Variants

### Variant A: Raw Diffusion Baseline

Variant A uses no hetero message passing. It conditions the denoising network on:

- `raw_block_features`
- `raw_pair_features`
- masks
- area/scale
- timestep embedding
- current noisy diffusion state

This is the control condition. It answers whether a diffusion decoder alone, with direct problem features, can beat or match the current candidate-generation path.

### Variant B: HGT-Lite Conditioned Diffusion

Variant B adds a shallow typed graph conditioner:

```text
DiffusionGraphInputs
  -> HGTLiteConditioner
  -> block_context [N, H]
  -> global_context [H]
```

The conditioner should be 1 to 3 layers, relation-specific, and cached once per instance before denoising starts. It should not rerun a heavy graph encoder at every diffusion step.

Each denoising step consumes:

```text
x_t [S, N, 3]
timestep embedding
raw block side channels
raw pair side channels
block_context
global_context
masks
```

The diffusion state is:

```text
center_x, center_y, log_aspect
```

The model output is:

```text
eps_pred or x0_pred: [S, N, 3]
pairwise_axis_logits: [S, P, C]
quality_pred: [S]
uncertainty: [S, N] or [S]
```

`pairwise_axis_logits` should be produced from both graph context and predicted geometry:

```text
h_i, h_j, x0_i, x0_j, delta_x, delta_y, distance, raw_pair_features
```

This keeps pairwise/topology predictions consistent with the sampled candidate instead of becoming an independent stale graph prediction.

## Diffusion Output Contracts

Add:

```text
src/floorset_arch/diffusion/contracts.py
```

Recommended dataclasses:

```python
@dataclass
class DiffusionPlacementPrior:
    centers: torch.Tensor
    log_aspect: torch.Tensor
    pairwise_axis_logits: torch.Tensor
    pair_index: torch.Tensor
    quality_pred: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None
    scale: float = 1.0
    variant: str = ""

@dataclass
class PlacementTensorBatch:
    rect_xywh: torch.Tensor
    pairwise_axis_logits: torch.Tensor
    pair_index: torch.Tensor
    score_features: torch.Tensor | None = None
    source: str = ""
```

`DiffusionPlacementPrior` is the raw model output. `PlacementTensorBatch` is the sampled candidate batch after area/aspect concretization into `[x, y, w, h]` tensors. Neither is an `AnchorGuidance` substitute.

## Training Targets

The training path should add a diffusion-specific preparation flow rather than overloading the current anchor target code.

Use `fp_sol` for continuous diffusion supervision:

- normalized center targets
- log-aspect targets
- fixed/preplaced mask handling
- optional repaired pseudo target as the default quality-first target source

Use `tree_sol` for topology auxiliary supervision:

- parse or validate B*Tree rows as parent, child, side labels before relying on them
- derive tree adjacency pairs and side labels
- supervise pairwise/topology heads for compact relative structure
- optionally add a tree-consistency loss between predicted pairwise logits and tree-derived side labels

Use `metrics_sol` for quality auxiliary supervision:

- split inference-available instance stats from solution-only metrics
- inference-available stats may become `global_features`
- solution-only fields become training labels for `quality_pred`

Do not feed solution quality fields from `metrics_sol` into inference conditioning.

## Inference Data Flow

The intended v11 production path is:

```text
ArchitectureV11Optimizer.solve()
  -> parse_instance(...)
  -> _try_diffusion_prior(inst)
      -> build_diffusion_graph_inputs(inst)
      -> load FLOORSET_DIFFUSION_CHECKPOINT
      -> sample variant A or B
      -> DiffusionPlacementPrior
      -> PlacementTensorBatch
      -> tensor prefilter and diffusion-aware ranking
      -> top-k concretize to Placement
      -> diffusion-aware repair
      -> exact v11 ranking
  -> best Placement
  -> position list
```

If checkpoint loading or sampling fails, the path is:

```text
ArchitectureV11Optimizer.solve()
  -> parse_instance(...)
  -> fallback v5 solver path
  -> metadata fallback = v5_no_diffusion_checkpoint or v5_diffusion_error
```

The fallback exists to keep validation scripts usable during development. Fallback scores must be clearly labeled and must not be mistaken for diffusion evidence.

## Decoder, Repair, And Ranking

The decoder changes from anchor-guided relative ordering to diffusion candidate concretization:

```text
centers + log_aspect + area_targets
  -> rect tensors [S, N, 4]
  -> masks enforce fixed/preplaced blocks
  -> clamp nonfinite values and invalid dimensions
  -> top-k tensor shortlist
  -> Placement objects
```

Repair must use diffusion signals without converting them to `AnchorGuidance`:

- preserve high-confidence centers when resolving overlaps if they are not causing hard violations
- use pairwise logits as soft ordering hints during overlap and grouping repair
- let boundary, cluster, and MIB exact constraints override uncertain diffusion hints
- keep exact hard-legality checks in the existing repair/diagnostics path

Ranking should be two-stage:

1. Tensor prefilter before `Placement` conversion, using cheap area, bbox, overlap proxy, pairwise consistency, boundary/mask proxy, uncertainty, and `quality_pred`.
2. Exact ranking after repair, using current placement metrics and v10 proxy semantics.

This keeps diffusion sampling efficient while preserving evaluator-facing correctness.

## Evaluation Visualization

Full eval must save the ten highest-cost predicted floorplan PNGs after the run completes.

Recommended output shape:

```text
artifacts/legacy_floorset_arch_eval_floorplans/floorplans/<run-stem>/
  top_cost_rank_01_case_<id>_cost_<value>.png
  top_cost_rank_02_case_<id>_cost_<value>.png
  top_cost_rank_03_case_<id>_cost_<value>.png
  top_cost_rank_04_case_<id>_cost_<value>.png
  top_cost_rank_05_case_<id>_cost_<value>.png
  top_cost_rank_06_case_<id>_cost_<value>.png
  top_cost_rank_07_case_<id>_cost_<value>.png
  top_cost_rank_08_case_<id>_cost_<value>.png
  top_cost_rank_09_case_<id>_cost_<value>.png
  top_cost_rank_10_case_<id>_cost_<value>.png
```

Single-case eval should save:

```text
artifacts/legacy_floorset_arch_eval_floorplans/floorplans/<run-stem>/case_<id>_cost_<value>.png
```

The evaluator already records per-case positions and costs, so the visualization change should render predicted solution positions rather than only ground truth. PNG generation failures should be reported but must not hide the numeric eval result.

## Naming And Compatibility

New production names:

- `ArchitectureV11Optimizer`
- `src/architecture_v11_optimizer.py`
- `FLOORSET_DIFFUSION_CHECKPOINT`
- `FLOORSET_DIFFUSION_VARIANT=raw|hgt_lite`

Compatibility names:

- `ArchitectureV5Optimizer = ArchitectureV11Optimizer` remains available as an alias in the optimizer module.
- `src/architecture_v5_optimizer.py` remains importable and forwards `MyOptimizer` and `ContestOptimizer` to the v11 optimizer.
- Old tests that import v4/v5 should continue to run unless they are explicitly updated for v11 behavior.

## Error Handling

`build_diffusion_graph_inputs()` should fail fast for inconsistent tensor shapes in tests, but production solve should catch diffusion-path failures and fall back to v5 with trace metadata.

Diffusion sampling should sanitize:

- NaN or inf centers
- NaN or inf log-aspect values
- negative or zero widths/heights after concretization
- missing pair logits
- mismatched pair index length
- fixed/preplaced drift

Invalid environment variable values should fall back to conservative defaults and record a trace warning when tracing is enabled.

Visualization write failures should not change solver output.

## Testing

Unit tests:

- `build_diffusion_graph_inputs()` returns stable node types, relation specs, masks, pair indices, and raw side channels.
- `parse_instance()` still returns `Instance`, not a hetero or diffusion graph.
- `DiffusionPlacementPrior` and `PlacementTensorBatch` enforce expected tensor shapes.
- Concretization preserves area, fixed blocks, and preplaced blocks.
- Pairwise logits are consumed without converting to `AnchorGuidance`.
- Missing or incompatible diffusion checkpoint triggers explicit v5 fallback.
- V5 import shim remains compatible.
- Eval visualization writes predicted PNGs for single-case and top-10 full eval paths.

Training tests:

- `tree_sol` parser validates row semantics before using side labels.
- `tree_sol` auxiliary targets align with selected `pair_index`.
- `metrics_sol` solution-only fields are used as quality labels, not inference inputs.
- Variant A and B can run one tiny train/eval batch without shape drift.

Evaluator checks:

1. Targeted pytest for diffusion contracts and graph inputs.
2. Full `uv run pytest`.
3. `bash scripts/validate.sh`.
4. `bash scripts/eval_single.sh 95`, verifying the PNG output.
5. `bash scripts/eval_total.sh`, verifying top-10 high-cost PNG output.

After code changes, run `graphify update .`.

## Milestone Acceptance

Milestone 1 is accepted when:

- v11 wrapper validates through the evaluator interface.
- Variant A and variant B are both runnable behind explicit config.
- Missing checkpoint fallback is visible and unambiguous.
- With a diffusion checkpoint, the path uses diffusion outputs directly and does not create `AnchorGuidance`.
- Full eval emits top-10 high-cost PNGs.
- The A/B result table reports `total_score_no_runtime`, runtime summary, feasible count, soft violations, and top-cost cases.

Promotion from A to B requires B to improve full-validation `total_score_no_runtime` or show a clear soft-violation/top-cost improvement without unacceptable runtime regression.

## Self-Review

- Placeholder scan: no placeholders remain.
- Internal consistency: v11 uses `Instance -> DiffusionGraphInputs -> diffusion` and does not route diffusion outputs through `AnchorGuidance`.
- Scope check: milestone 1 compares only raw diffusion and HGT-lite conditioned diffusion; deeper HGT is explicitly out of scope.
- Ambiguity check: `tree_sol` and `metrics_sol` are training signals, while inference uses only fields computable from the test instance.
