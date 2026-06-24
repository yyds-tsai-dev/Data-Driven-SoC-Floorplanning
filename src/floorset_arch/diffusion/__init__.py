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
    "load_diffusion_checkpoint",
    "sample_diffusion_prior",
]


def __getattr__(name: str):
    if name == "build_diffusion_graph_inputs":
        from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs

        return build_diffusion_graph_inputs
    if name == "GraphConditionedPlacementDiffusion":
        from floorset_arch.diffusion.model import GraphConditionedPlacementDiffusion

        return GraphConditionedPlacementDiffusion
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
