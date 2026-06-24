from floorset_arch.diffusion.contracts import (
    DiffusionGraphInputs,
    DiffusionPlacementPrior,
    PlacementTensorBatch,
    Relation,
)

__all__ = [
    "DiffusionGraphInputs",
    "DiffusionPlacementPrior",
    "PlacementTensorBatch",
    "Relation",
    "GraphConditionedPlacementDiffusion",
    "build_diffusion_graph_inputs",
    "concretize_diffusion_prior",
    "load_diffusion_checkpoint",
    "placement_from_tensor_candidate",
    "rank_repaired_placement",
    "sample_diffusion_prior",
    "select_tensor_shortlist",
]


def __getattr__(name: str):
    if name == "build_diffusion_graph_inputs":
        from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs

        return build_diffusion_graph_inputs
    if name == "GraphConditionedPlacementDiffusion":
        from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion

        return GraphConditionedPlacementDiffusion
    if name in {"concretize_diffusion_prior", "placement_from_tensor_candidate"}:
        from floorset_arch.diffusion.concretize import (
            concretize_diffusion_prior,
            placement_from_tensor_candidate,
        )

        return {
            "concretize_diffusion_prior": concretize_diffusion_prior,
            "placement_from_tensor_candidate": placement_from_tensor_candidate,
        }[name]
    if name in {"rank_repaired_placement", "select_tensor_shortlist"}:
        from floorset_arch.diffusion.ranking import (
            rank_repaired_placement,
            select_tensor_shortlist,
        )

        return {
            "rank_repaired_placement": rank_repaired_placement,
            "select_tensor_shortlist": select_tensor_shortlist,
        }[name]
    if name in {"load_diffusion_checkpoint", "sample_diffusion_prior"}:
        from floorset_arch.diffusion.sampling import (
            load_diffusion_checkpoint,
            sample_diffusion_prior,
        )

        return {
            "load_diffusion_checkpoint": load_diffusion_checkpoint,
            "sample_diffusion_prior": sample_diffusion_prior,
        }[name]
    raise AttributeError(name)
