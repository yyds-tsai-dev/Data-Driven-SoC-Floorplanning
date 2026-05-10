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

**Architecture v3**:
The current production architecture generation that keeps `floorset_arch` as the active solver package while exposing a v3 contest wrapper.
_Avoid_: architecture_v2, treating wrapper version names as separate solver packages.

## Relationships

- The **Production Solver Path** uses **Anchor-GNN Guidance** as a prior, not as a complete floorplan.
- The **Beam Decoder** consumes **Heterogeneous Floorplan Graph** features and **MER/Skyline Slot** candidates.
- A **Legacy Strategy** may donate geometry helper logic, but it is not a selectable **Production Solver Path**.
- A **Reference Architecture** may be consulted while tuning `floorset_arch`, but it does not define the active solver architecture.
- A **MER/Skyline Slot** must respect hard placement legality before repair is allowed to refine soft constraints.
- **Architecture v3** names the contest-facing wrapper generation; the **Production Solver Path** remains `floorset_arch`.

## Example Dialogue

> **Dev:** "Can we keep default as a backup beam if the Anchor-GNN path looks weak?"
> **Domain expert:** "No. The Production Solver Path should be the Anchor-GNN guided hetero-graph beam decoder; default and legacy are only historical references."

## Flagged Ambiguities

- "Remove greedy/hint architecture" was resolved to mean removing `legacy`, `default`, and `model_first` as production solver branches while preserving reusable geometry helpers where needed by the beam decoder.
- "Rename architecture_v2" was resolved to mean renaming the contest wrapper and active optimizer class to **Architecture v3**, not renaming the `floorset_arch` package.
