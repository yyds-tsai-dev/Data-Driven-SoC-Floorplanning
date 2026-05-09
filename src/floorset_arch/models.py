from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    def moved(self, x: float, y: float) -> "Rect":
        return Rect(float(x), float(y), self.width, self.height)

    def resized(self, width: float, height: float) -> "Rect":
        return Rect(self.x, self.y, float(width), float(height))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (float(self.x), float(self.y), float(self.width), float(self.height))


@dataclass
class Placement:
    rects: Dict[int, Rect] = field(default_factory=dict)

    def copy(self) -> "Placement":
        return Placement(dict(self.rects))

    def to_position_list(self, block_count: int) -> List[tuple[float, float, float, float]]:
        return [self.rects[i].as_tuple() for i in range(block_count)]


@dataclass
class SolverConfig:
    max_repair_passes: int = 8
    max_candidates_per_block: int = 20
    max_start_candidates: int = 2
    max_push_iters: int = 32
    coordinate_eps: float = 1e-6
    checkpoint_env: str = "FLOORSET_GNN_CHECKPOINT"
    default_checkpoint: str = "checkpoints/gnn_best.pt"
    beam_width: int = 1
    shape_variant_count: int = 2
    local_search_moves: int = 64
    quality_mode: str = "score_first"
    hpwl_weight: float = 1.0
    bbox_weight: float = 0.015
    boundary_penalty: float = 350.0
    group_penalty: float = 240.0
    mib_penalty: float = 300.0
    anchor_weight: float = 0.35
    raw_weight: float = 0.30
    boundary_order_bias: float = 0.72
    anchor_translation_strength: float = 0.58
    guarded_repair_slack: float = 0.025
    soft_proxy_slack: float = 0.42
    equal_soft_proxy_slack: float = 0.015
    max_boundary_component_snaps: int = 30
    max_cluster_component_moves: int = 26
    max_pair_candidates_per_component: int = 40
    checkpoint_repo_relative: bool = True


@dataclass
class AnchorGuidance:
    rect_priors: Dict[int, Rect] = field(default_factory=dict)
    priority: Dict[int, float] = field(default_factory=dict)
    log_aspect: Dict[int, float] = field(default_factory=dict)
    scale: float = 1.0


@dataclass
class Instance:
    block_count: int
    area_targets: torch.Tensor
    b2b_connectivity: torch.Tensor
    p2b_connectivity: torch.Tensor
    pins_pos: torch.Tensor
    constraints: torch.Tensor
    fixed: Set[int]
    preplaced: Set[int]
    mib_groups: Dict[int, List[int]]
    cluster_groups: Dict[int, List[int]]
    boundary: Dict[int, int]
    target_rects: Dict[int, Rect]
    valid_b2b: torch.Tensor
    valid_p2b: torch.Tensor
    b2b_by_block: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    p2b_by_block: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    anchor_guidance: Optional[AnchorGuidance] = None
