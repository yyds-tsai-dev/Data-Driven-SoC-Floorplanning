# FloorSet Floorplanning

This context describes the solver architecture and domain language for the FloorSet SoC floorplanning optimizer.

## Language

**Production Solver Path**:
The single floorplan generation path used by `solve()` for official evaluation.
_Avoid_: Running multiple legacy strategies as hidden fallbacks.

**Legacy Strategy**:
The older hand-written constructive placement branch that can be mined for geometry helpers but is no longer an executable production strategy.
_Avoid_: legacy beam, default beam, model_first.

**Reference Architecture**:
An external or older solver path, such as `src/arch_new`, used only for comparison, checkpoint compatibility, or implementation guidance.
_Avoid_: treating reference wrappers as the active `floorset_arch` production entrypoint.

**Anchor-GNN Guidance**:
Block-level geometric priors produced by the trained Anchor-GNN checkpoint and consumed as guidance by the constructive decoder.
_Avoid_: model hints, pure hint placement.

**No-Checkpoint Guidance Mode**:
A deliberate ablation mode where no Anchor-GNN checkpoint is loaded, so `AnchorGuidance` is absent while the same decoder, repair path, and candidate ranking still run.
_Avoid_: calling this no-repair mode or treating it as a separate production solver.

**Selectable Anchor-GNN Encoder**:
The training-time and checkpoint-time choice between the legacy MPNN encoder, graph-transformer encoder, or Local HGT Encoder while preserving the same anchor, priority, aspect, and pairwise guidance heads.
_Avoid_: treating Graph Transformer as non-GNN, changing decoder behavior without evaluator evidence.

**Local HGT Encoder**:
A planned Selectable Anchor-GNN Encoder that consumes the Heterogeneous Floorplan Graph with relation-specific local attention over typed block, pin, cluster, MIB, and boundary nodes, then emits the normal Anchor-GNN Guidance heads.
_Avoid_: full block-to-block global attention, clique-expanding heterogeneous constraints, changing the Production Solver Path before evaluator evidence.

**Heterogeneous Floorplan Graph**:
A factor-style graph with block, pin, group, MIB, and boundary concepts represented separately so constraints are first-class solver inputs.
_Avoid_: flat block-only graph.

**MER/Skyline Slot**:
A legal candidate placement location derived from maximal empty rectangle candidates and skyline/frontier candidates.
_Avoid_: arbitrary greedy coordinate.

**Hard Legality Gate**:
The solver-internal first-pass acceptance boundary that keeps placements with missing blocks, overlaps, area violations, fixed-shape dimension changes, or preplaced position/dimension changes behind hard-legal alternatives.
_Avoid_: treating soft-constraint improvement as a reason to accept hard infeasibility.

**Beam Decoder**:
A constructive decoder that expands multiple partial placements by selecting a remaining block, shape, and legal slot at each step.
_Avoid_: evaluator-in-loop reranking, single greedy placement.

**Decoder-Side Grouping Adjacency Bias**:
A later decoder-ranking preference that encourages grouped blocks to become adjacent before repair runs.
_Avoid_: broad cluster key blending, chaining entire groups by default, or hard-coding validation IDs.

**Narrow Grouping Pair Bias**:
A constrained decoder-side grouping preference that applies only to ambiguous same-cluster block pairs under high group soft pressure, using a small pairwise axis bonus without global cluster key blending or default whole-group chaining.
_Avoid_: treating every grouped block as eligible for global reordering pressure.

**Cluster-Level Grouping Pressure**:
A local grouping-risk signal for one cluster, used to decide whether same-cluster ambiguous pairs should receive narrow decoder-side pressure.
_Avoid_: enabling grouping bias for every cluster because the overall instance is grouping-heavy.

**Architecture v4**:
The current production architecture generation that keeps `floorset_arch` as the active solver package while exposing a v4 contest wrapper.
_Avoid_: architecture_v2, treating wrapper version names as separate solver packages.

**Architecture v5**:
The current experimental architecture generation that keeps `floorset_arch` as the active solver package while exposing a v5 contest wrapper and selectable Anchor-GNN encoder.
_Avoid_: renaming the package, treating encoder experiments as a separate solver path.

**No-Runtime Quality Score**:
The local architecture-tuning score that uses official quality and soft-violation factors with runtime adjustment fixed to `1.0`.
_Avoid_: treating local runtime-aware score as the primary architecture metric.

**V10 No-Runtime Proxy**:
A solver-internal acceptance proxy that approximates **No-Runtime Quality Score** during placement selection by putting hard legality first, then balancing geometric quality with soft-violation pressure.
_Avoid_: soft-first acceptance that allows HPWL or bbox area to regress substantially because soft violations decreased.

**Local Runtime-Aware Score**:
The local evaluator score that uses each validation runtime divided by the solver's own validation-run median runtime.
_Avoid_: treating it as the official contest runtime factor.

**Conservative Runtime Budget**:
The submission-oriented policy that permits extra sample-local search only when reusable v10 risk signals justify the raw runtime cost.
_Avoid_: ignoring runtime after a no-runtime score win, enabling every opt-in portfolio by default.

**V10 Evidence-Gated Budget Layer**:
The decision surface that combines full-validation v10 score evidence, reusable instance risk signals, and raw runtime tail risk before allowing extra candidate, repair, or portfolio work.
_Avoid_: new solver branch, hand-tuned validation ID policy, enabling knobs from single-case wins.

**Runtime-Tail Budget Clamp**:
A follow-up policy that reduces expensive candidate or repair work on cases whose runtime tail is not buying v10 score improvement.
_Avoid_: treating hard clamp behavior as the preferred v10 runtime policy after evidence shows it can damage no-runtime quality.

**Conditional Runtime Budget**:
A softer runtime policy that preserves baseline repair while stopping only extra repair paths or profile expansion when repeated attempts are not accepted by the **V10 No-Runtime Proxy** or when tier-specific elapsed/attempt budgets are exhausted.
_Avoid_: global timeouts, fixed pass cuts, or disabling baseline repair before acceptance evidence is observed.

**Sample-Local Parallelism**:
Parallel work that happens inside one `solve()` call for a single contest sample, such as independent candidate generation or repair profiles.
_Avoid_: cross-sample batching, assuming validation or hidden test cases can be processed together.

**Sample-Local Quality Portfolio**:
An opt-in set of non-GNN post-processing candidates generated inside one `solve()` call for a high-impact sample, using the existing checkpoint guidance and deterministic decoder output as input.
_Avoid_: treating it as a new production solver branch, retraining, or cross-sample batching.

**Training Golden Answer**:
The provided `fp_sol` layout used as a supervised geometric reference. It is not a soft-constraint oracle because official QA confirms training golden answers may violate boundary, grouping, or MIB constraints.
_Avoid_: treating imitation learning as exact constraint satisfaction.

**Constraint-Clean Training Sample**:
A supervised training case whose `fp_sol` satisfies boundary, grouping, and MIB soft constraints, so it can safely teach geometry and constraint-sensitive layout priors. Clean-only training is available as `--clean-sample-policy strict`, but the default is weighted dirty-sample training because measured clean ratio is too low for reliable remote training.
_Avoid_: treating soft-violating `fp_sol` as reliable order/pairwise supervision.

**Validation Tail Diagnostics**:
Using the highest weighted validation cases to identify score drivers and design generalized solver triggers based on instance statistics.
_Avoid_: hard-coding validation `test_id` behavior into the Production Solver Path.

**High-Risk Case**:
A floorplanning instance whose general statistics predict outsized score risk, such as large block count, dense boundary/grouping constraints, difficult fixed/preplaced structure, or high net density.
_Avoid_: using validation case IDs as the risk definition.

## Relationships

- The **Production Solver Path** uses **Anchor-GNN Guidance** as a prior, not as a complete floorplan.
- **No-Checkpoint Guidance Mode** isolates deterministic decoder and repair behavior by removing learned guidance only; repair remains active.
- The **Beam Decoder** consumes **Heterogeneous Floorplan Graph** features and **MER/Skyline Slot** candidates.
- A **Legacy Strategy** may donate geometry helper logic, but it is not a selectable **Production Solver Path**.
- A **Reference Architecture** may be consulted while tuning `floorset_arch`, but it does not define the active solver architecture.
- A **MER/Skyline Slot** must respect hard placement legality before repair is allowed to refine soft constraints.
- The **Hard Legality Gate** precedes **V10 No-Runtime Proxy** decisions so soft repair, candidate ranking, and refinement do not trade hard feasibility for local quality.
- **Architecture v4** names the contest-facing wrapper generation; the **Production Solver Path** remains `floorset_arch`.
- **Architecture v5** extends **Architecture v4** with a **Selectable Anchor-GNN Encoder** while preserving `floorset_arch` as the **Production Solver Path**.
- A **Selectable Anchor-GNN Encoder** changes learned **Anchor-GNN Guidance** only; checkpoint promotion still requires evaluator evidence.
- A **Local HGT Encoder** is a **Selectable Anchor-GNN Encoder** variant that preserves b2b/p2b locality and heterogeneous constraint factors instead of flattening them into a block-only graph.
- **No-Runtime Quality Score** is the primary metric for local architecture comparison; **Local Runtime-Aware Score** is a runtime-risk signal.
- **V10 No-Runtime Proxy** is a solver-internal acceptance surface for candidate selection and repair/refine decisions; **No-Runtime Quality Score** remains the evaluator-facing validation metric.
- A **Conservative Runtime Budget** gates **Sample-Local Parallelism** and **Sample-Local Quality Portfolio** so no-runtime wins do not automatically become submission defaults.
- The **V10 Evidence-Gated Budget Layer** gathers **Validation Tail Diagnostics**, **High-Risk Case** signals, and raw runtime tails into one promotion decision surface.
- The **V10 Evidence-Gated Budget Layer** may allocate more budget to a medium-large dense instance than to a sparse largest instance when v10 score share, constraint density, and net density justify it.
- Extra **Sample-Local Quality Portfolio**, high-risk repair, **Runtime-Tail Budget Clamp**, or **Decoder-Side Grouping Adjacency Bias** work must pass the **V10 Evidence-Gated Budget Layer** before becoming a default submission behavior.
- **Conditional Runtime Budget** replaces hard **Runtime-Tail Budget Clamp** as the preferred follow-up when hard clamps reduce runtime at the cost of **No-Runtime Quality Score**.
- A runtime budget policy is considered only after **V10 No-Runtime Proxy** acceptance, so runtime reductions do not preempt quality-preserving repair.
- **Sample-Local Parallelism** may use multiprocessing or multithreading inside one sample, but contest samples remain sequential.
- A **Sample-Local Quality Portfolio** may refine and rank multiple placements for one sample, but it must preserve the **Production Solver Path** and remain gated by reusable instance statistics and full evaluator evidence.
- **Narrow Grouping Pair Bias** is the evidence-promoted default follow-up to broad **Decoder-Side Grouping Adjacency Bias** when full-cluster key blending or default chaining regresses HPWL/area.
- **Cluster-Level Grouping Pressure** gates **Narrow Grouping Pair Bias** so grouping-heavy instances do not globally reorder low-risk clusters.
- **Training Golden Answer** should teach geometric priors, while soft-constraint satisfaction remains the responsibility of constraint-aware decoding, repair, and scoring.
- A **Constraint-Clean Training Sample** is eligible for full imitation training; soft-violating training samples are low-weight geometry references by default, with dirty order/pairwise supervision suppressed.
- **Validation Tail Diagnostics** may guide optimization priorities, but production behavior must be triggered by reusable instance features such as block count, boundary/group density, fixed/preplaced structure, or net statistics.
- **High-Risk Case** instances may spend extra sample-local candidate and repair effort; low-risk cases should keep the fast default path.

## Example Dialogue

> **Dev:** "Can we keep default as a backup beam if the Anchor-GNN path looks weak?"
> **Domain expert:** "No. The Production Solver Path should be the Anchor-GNN guided hetero-graph beam decoder; default and legacy are only historical references."
>
> **Dev:** "Can we turn on v10 soft repair because it improved no-runtime score?"
> **Domain expert:** "Not by itself. The V10 Evidence-Gated Budget Layer also checks full-validation scope and raw runtime tail before promotion."

## Flagged Ambiguities

- "Remove greedy/hint architecture" was resolved to mean removing `legacy`, `default`, and `model_first` as production solver branches while preserving reusable geometry helpers where needed by the beam decoder.
- "Rename architecture_v2" was resolved to mean renaming the contest wrapper and active optimizer class to a numbered **Architecture vN**, not renaming the `floorset_arch` package.
- "Local score" was ambiguous between **No-Runtime Quality Score** and **Local Runtime-Aware Score**; resolved: tune architecture with no-runtime score first, then check raw runtime and local runtime-aware score before submission.
- "ICCAD submission score first" was resolved to require a **Conservative Runtime Budget**: keep the fast Graph Transformer default unless v10 risk signals and full-score evidence justify extra sample-local search.
- "Do runtime-tail and grouping-decoder work next" was resolved as ordering, not scope expansion: first v10 risk-gated repair acceptance, then **Runtime-Tail Budget Clamp**, then **Decoder-Side Grouping Adjacency Bias**.
- "Use validation tail diagnostics" was resolved to allow validation-tail evidence for design direction, while prohibiting `test_id`-specific production logic.
- "Only high-risk cases get heavier search" was resolved to mean generalized instance-stat triggers, not validation-tail IDs.
- "No GNN model ckpt mode" was resolved to mean **No-Checkpoint Guidance Mode**: use the normal production path with `AnchorGuidance=None` to measure how much deterministic decoding and repair can achieve without learned priors.
- "Canonical HGT hetero graph encoder" was resolved to mean **Local HGT Encoder**: relation-specific typed local message passing over the **Heterogeneous Floorplan Graph**, no global refinement in v1, and no decoder path change before full evaluator evidence.
- "Evidence & Budget Layer" was resolved to mean **V10 Evidence-Gated Budget Layer**: a shared decision surface for promotion and extra solver budget, not a new architecture version or executable solver branch.
- "Narrow grouping default" was resolved by the 2026-06-10 full-validation ablation: keep narrow same-cluster pair bias on by default, but preserve an environment switch for ablation.
