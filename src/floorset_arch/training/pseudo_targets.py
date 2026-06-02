from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from floorset_arch.models import Instance, Placement, SolverConfig
from floorset_arch.repair import repair_placement, soft_violation_counts
from floorset_arch.training.losses import _placement_from_fp_sol


class TrainingTargetSource(StrEnum):
    CLEAN_FP_SOL = "clean_fp_sol"
    DIRTY_ORIGINAL = "dirty_original"
    DIRTY_REPAIRED = "dirty_repaired"
    DIRTY_REPAIRED_CLEAN_ENOUGH = "dirty_repaired_clean_enough"


@dataclass(frozen=True)
class PseudoTargetConfig:
    enabled: bool = False
    clean_enough_soft_violations: int = 0
    dirty_pseudo_order_weight: float = 0.20
    dirty_pseudo_clean_enough_order_weight: float = 0.35


@dataclass(frozen=True)
class TrainingTargetRecord:
    target_fp_sol: torch.Tensor
    source: TrainingTargetSource
    original_soft_violations: tuple[int, int, int]
    repaired_soft_violations: tuple[int, int, int] | None


def placement_to_fp_sol(placement: Placement, block_count: int) -> torch.Tensor:
    rows = []
    for block in range(block_count):
        rect = placement.rects[block]
        rows.append([rect.width, rect.height, rect.x, rect.y])
    return torch.tensor(rows, dtype=torch.float32)


def fp_sol_to_target_positions(fp_sol: torch.Tensor, block_count: int) -> torch.Tensor:
    gt = fp_sol[:block_count].float()
    return torch.stack([gt[:, 2], gt[:, 3], gt[:, 0], gt[:, 1]], dim=1)


def build_training_target_record(
    inst: Instance,
    fp_sol: torch.Tensor,
    config: PseudoTargetConfig,
) -> TrainingTargetRecord:
    block_count = inst.block_count
    original_placement = _placement_from_fp_sol(fp_sol, block_count)
    original_soft = soft_violation_counts(inst, original_placement)
    if sum(original_soft) == 0:
        return TrainingTargetRecord(
            target_fp_sol=fp_sol[:block_count].detach().cpu().float(),
            source=TrainingTargetSource.CLEAN_FP_SOL,
            original_soft_violations=original_soft,
            repaired_soft_violations=None,
        )
    if not config.enabled:
        return TrainingTargetRecord(
            target_fp_sol=fp_sol[:block_count].detach().cpu().float(),
            source=TrainingTargetSource.DIRTY_ORIGINAL,
            original_soft_violations=original_soft,
            repaired_soft_violations=None,
        )
    repaired = repair_placement(inst, original_placement.copy(), SolverConfig())
    repaired_soft = soft_violation_counts(inst, repaired)
    source = (
        TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH
        if sum(repaired_soft) <= config.clean_enough_soft_violations
        else TrainingTargetSource.DIRTY_REPAIRED
    )
    return TrainingTargetRecord(
        target_fp_sol=placement_to_fp_sol(repaired, block_count),
        source=source,
        original_soft_violations=original_soft,
        repaired_soft_violations=repaired_soft,
    )
