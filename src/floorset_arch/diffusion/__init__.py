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
    "build_diffusion_graph_inputs",
]


def __getattr__(name: str):
    if name == "build_diffusion_graph_inputs":
        from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs

        return build_diffusion_graph_inputs
    raise AttributeError(name)
