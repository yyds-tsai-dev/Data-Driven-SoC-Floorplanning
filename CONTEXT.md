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

**Heterogeneous Floorplan Graph**:
A factor-style graph with block, pin, group, MIB, and boundary concepts represented separately so constraints are first-class solver inputs.
_Avoid_: flat block-only graph.

**MER/Skyline Slot**:
A legal candidate placement location derived from maximal empty rectangle candidates and skyline/frontier candidates.
_Avoid_: arbitrary greedy coordinate.

**Beam Decoder**:
A constructive decoder that expands multiple partial placements by selecting a remaining block, shape, and legal slot at each step.
_Avoid_: evaluator-in-loop reranking, single greedy placement.

**Architecture v4**:
The current production architecture generation that keeps `floorset_arch` as the active solver package while exposing a v4 contest wrapper.
_Avoid_: architecture_v2, treating wrapper version names as separate solver packages.

**No-Runtime Quality Score**:
The local architecture-tuning score that uses official quality and soft-violation factors with runtime adjustment fixed to `1.0`.
_Avoid_: treating local runtime-aware score as the primary architecture metric.

**Local Runtime-Aware Score**:
The local evaluator score that uses each validation runtime divided by the solver's own validation-run median runtime.
_Avoid_: treating it as the official contest runtime factor.

**Sample-Local Parallelism**:
Parallel work that happens inside one `solve()` call for a single contest sample, such as independent candidate generation or repair profiles.
_Avoid_: cross-sample batching, assuming validation or hidden test cases can be processed together.

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
- The **Beam Decoder** consumes **Heterogeneous Floorplan Graph** features and **MER/Skyline Slot** candidates.
- A **Legacy Strategy** may donate geometry helper logic, but it is not a selectable **Production Solver Path**.
- A **Reference Architecture** may be consulted while tuning `floorset_arch`, but it does not define the active solver architecture.
- A **MER/Skyline Slot** must respect hard placement legality before repair is allowed to refine soft constraints.
- **Architecture v4** names the contest-facing wrapper generation; the **Production Solver Path** remains `floorset_arch`.
- **No-Runtime Quality Score** is the primary metric for local architecture comparison; **Local Runtime-Aware Score** is a runtime-risk signal.
- **Sample-Local Parallelism** may use multiprocessing or multithreading inside one sample, but contest samples remain sequential.
- **Training Golden Answer** should teach geometric priors, while soft-constraint satisfaction remains the responsibility of constraint-aware decoding, repair, and scoring.
- A **Constraint-Clean Training Sample** is eligible for full imitation training; soft-violating training samples are low-weight geometry references by default, with dirty order/pairwise supervision suppressed.
- **Validation Tail Diagnostics** may guide optimization priorities, but production behavior must be triggered by reusable instance features such as block count, boundary/group density, fixed/preplaced structure, or net statistics.
- **High-Risk Case** instances may spend extra sample-local candidate and repair effort; low-risk cases should keep the fast default path.

## Example Dialogue

> **Dev:** "Can we keep default as a backup beam if the Anchor-GNN path looks weak?"
> **Domain expert:** "No. The Production Solver Path should be the Anchor-GNN guided hetero-graph beam decoder; default and legacy are only historical references."

## Flagged Ambiguities

- "Remove greedy/hint architecture" was resolved to mean removing `legacy`, `default`, and `model_first` as production solver branches while preserving reusable geometry helpers where needed by the beam decoder.
- "Rename architecture_v2" was resolved to mean renaming the contest wrapper and active optimizer class to a numbered **Architecture vN**, not renaming the `floorset_arch` package.
- "Local score" was ambiguous between **No-Runtime Quality Score** and **Local Runtime-Aware Score**; resolved: tune architecture with no-runtime score first, then check raw runtime and local runtime-aware score before submission.
- "Use validation tail diagnostics" was resolved to allow validation-tail evidence for design direction, while prohibiting `test_id`-specific production logic.
- "Only high-risk cases get heavier search" was resolved to mean generalized instance-stat triggers, not validation-tail IDs.
