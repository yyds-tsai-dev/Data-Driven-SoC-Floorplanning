from __future__ import annotations

from dataclasses import dataclass

import torch


Relation = tuple[str, str, str]


def _require_shape(condition: bool, name: str, expected: str, actual: tuple[int, ...]) -> None:
    if not condition:
        raise ValueError(f"{name} must have shape {expected}; got {actual}")


def _require_dtype(tensor: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        dtype_name = "torch.long" if dtype is torch.long else str(dtype)
        raise ValueError(f"{name} must be {dtype_name}")


def _require_same_device(named_tensors: tuple[tuple[str, torch.Tensor | None], ...]) -> None:
    devices = {tensor.device for _name, tensor in named_tensors if tensor is not None}
    if len(devices) > 1:
        raise ValueError("all tensors must be on the same device")


def _validate_pair_index(pair_index: torch.Tensor, block_count: int) -> int:
    _require_shape(
        pair_index.dim() == 2 and pair_index.shape[1] == 2,
        "pair_index",
        "[P,2]",
        tuple(pair_index.shape),
    )
    _require_dtype(pair_index, "pair_index", torch.long)
    pair_count = int(pair_index.shape[0])
    if pair_count > 0 and pair_index.device.type != "meta":
        valid_min = bool((pair_index >= 0).all().item())
        valid_max = bool((pair_index < block_count).all().item())
        if not (valid_min and valid_max):
            raise ValueError("pair_index values must reference valid blocks")
    return pair_count


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

    def __post_init__(self) -> None:
        block_count = int(self.area.shape[0]) if self.area.dim() == 1 else -1
        _require_shape(self.area.dim() == 1, "area", "[N]", tuple(self.area.shape))
        for name, value, dtype in (
            ("fixed_mask", self.fixed_mask, torch.bool),
            ("preplaced_mask", self.preplaced_mask, torch.bool),
            ("movable_mask", self.movable_mask, torch.bool),
            ("boundary_codes", self.boundary_codes, torch.long),
            ("cluster_ids", self.cluster_ids, torch.long),
            ("mib_ids", self.mib_ids, torch.long),
        ):
            _require_dtype(value, name, dtype)
        for name, value in (
            ("fixed_mask", self.fixed_mask),
            ("preplaced_mask", self.preplaced_mask),
            ("movable_mask", self.movable_mask),
            ("boundary_codes", self.boundary_codes),
            ("cluster_ids", self.cluster_ids),
            ("mib_ids", self.mib_ids),
        ):
            _require_shape(value.shape == (block_count,), name, "[N]", tuple(value.shape))
        _require_shape(
            self.raw_block_features.dim() == 2 and self.raw_block_features.shape[0] == block_count,
            "raw_block_features",
            "[N,F]",
            tuple(self.raw_block_features.shape),
        )
        pair_count = _validate_pair_index(self.pair_index, block_count)
        _require_shape(
            self.raw_pair_features.dim() == 2 and self.raw_pair_features.shape[0] == pair_count,
            "raw_pair_features",
            "[P,F]",
            tuple(self.raw_pair_features.shape),
        )
        _require_same_device(
            (
                ("area", self.area),
                ("raw_block_features", self.raw_block_features),
                ("raw_pair_features", self.raw_pair_features),
                ("pair_index", self.pair_index),
                ("fixed_mask", self.fixed_mask),
                ("preplaced_mask", self.preplaced_mask),
                ("movable_mask", self.movable_mask),
                ("boundary_codes", self.boundary_codes),
                ("cluster_ids", self.cluster_ids),
                ("mib_ids", self.mib_ids),
                ("global_features", self.global_features),
            )
        )


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

    def __post_init__(self) -> None:
        _require_shape(
            self.centers.dim() == 3 and self.centers.shape[2] == 2,
            "centers",
            "[S,N,2]",
            tuple(self.centers.shape),
        )
        sample_count, block_count = int(self.centers.shape[0]), int(self.centers.shape[1])
        _require_shape(
            self.log_aspect.shape == (sample_count, block_count),
            "log_aspect",
            "[S,N]",
            tuple(self.log_aspect.shape),
        )
        pair_count = _validate_pair_index(self.pair_index, block_count)
        _require_shape(
            self.pairwise_axis_logits.dim() == 3
            and self.pairwise_axis_logits.shape[0] == sample_count
            and self.pairwise_axis_logits.shape[1] == pair_count,
            "pairwise_axis_logits",
            "[S,P,C]",
            tuple(self.pairwise_axis_logits.shape),
        )
        if self.quality_pred is not None:
            _require_shape(
                self.quality_pred.shape == (sample_count,),
                "quality_pred",
                "[S]",
                tuple(self.quality_pred.shape),
            )
        if self.uncertainty is not None:
            _require_shape(
                self.uncertainty.shape[0] == sample_count,
                "uncertainty",
                "[S,...]",
                tuple(self.uncertainty.shape),
            )
        _require_same_device(
            (
                ("centers", self.centers),
                ("log_aspect", self.log_aspect),
                ("pairwise_axis_logits", self.pairwise_axis_logits),
                ("pair_index", self.pair_index),
                ("quality_pred", self.quality_pred),
                ("uncertainty", self.uncertainty),
            )
        )

    @property
    def sample_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def block_count(self) -> int:
        return int(self.centers.shape[1])

    @property
    def pair_count(self) -> int:
        return int(self.pair_index.shape[0])


@dataclass(frozen=True)
class PlacementTensorBatch:
    rect_xywh: torch.Tensor
    pairwise_axis_logits: torch.Tensor
    pair_index: torch.Tensor
    score_features: torch.Tensor | None = None
    source: str = ""

    def __post_init__(self) -> None:
        _require_shape(
            self.rect_xywh.dim() == 3 and self.rect_xywh.shape[2] == 4,
            "rect_xywh",
            "[S,N,4]",
            tuple(self.rect_xywh.shape),
        )
        sample_count = int(self.rect_xywh.shape[0])
        block_count = int(self.rect_xywh.shape[1])
        pair_count = _validate_pair_index(self.pair_index, block_count)
        _require_shape(
            self.pairwise_axis_logits.dim() == 3
            and self.pairwise_axis_logits.shape[0] == sample_count
            and self.pairwise_axis_logits.shape[1] == pair_count,
            "pairwise_axis_logits",
            "[S,P,C]",
            tuple(self.pairwise_axis_logits.shape),
        )
        if self.score_features is not None:
            _require_shape(
                self.score_features.shape[0] == sample_count,
                "score_features",
                "[S,...]",
                tuple(self.score_features.shape),
            )
        _require_same_device(
            (
                ("rect_xywh", self.rect_xywh),
                ("pairwise_axis_logits", self.pairwise_axis_logits),
                ("pair_index", self.pair_index),
                ("score_features", self.score_features),
            )
        )

    def select(self, indices: torch.Tensor | list[int]) -> "PlacementTensorBatch":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.rect_xywh.device)
        score_features = self.score_features[index] if self.score_features is not None else None
        return PlacementTensorBatch(
            rect_xywh=self.rect_xywh[index],
            pairwise_axis_logits=self.pairwise_axis_logits[index],
            pair_index=self.pair_index,
            score_features=score_features,
            source=self.source,
        )
