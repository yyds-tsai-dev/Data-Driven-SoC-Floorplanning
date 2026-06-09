from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

import torch

from floorset_arch.diagnostics import placement_metrics
from floorset_arch.geometry import bbox
from floorset_arch.models import Instance, Placement, Rect
from floorset_arch.repair import soft_violation_counts

MetricMap = Mapping[str, float | int]

_DEFAULT_TIE_TOLERANCE = 0.001
_DIMENSION_EPS = 1e-6
_POSITION_EPS = 1e-6


@dataclass(frozen=True)
class HardLegality:
    missing_blocks: int
    overlap_count: int
    area_violations: int
    fixed_violations: int
    preplaced_violations: int

    @property
    def rank(self) -> tuple[int, int, int, int, int]:
        return (
            self.missing_blocks,
            self.overlap_count,
            self.area_violations,
            self.fixed_violations,
            self.preplaced_violations,
        )

    @property
    def legal(self) -> bool:
        return all(value == 0 for value in self.rank)


def hard_legality(
    inst: Instance, placement: Placement, metrics: MetricMap | None = None
) -> HardLegality:
    metrics = metrics if metrics is not None else placement_metrics(inst, placement)
    rects = placement.rects
    missing_blocks = sum(1 for block in range(inst.block_count) if block not in rects)
    fixed_violations = 0
    preplaced_violations = 0
    area_violations = 0

    for block in range(inst.block_count):
        rect = rects.get(block)
        if rect is None:
            continue
        target = inst.target_rects.get(block)
        if block in inst.preplaced:
            if target is not None and not _same_rect(rect, target):
                preplaced_violations += 1
            continue
        if block in inst.fixed:
            if target is not None and not _same_dimensions(rect, target):
                fixed_violations += 1
            continue
        target_area = _target_area(inst, block)
        if target_area > 0.0 and abs(rect.area - target_area) > _DIMENSION_EPS:
            area_violations += 1

    return HardLegality(
        missing_blocks=missing_blocks,
        overlap_count=int(metrics["overlap_count"]),
        area_violations=area_violations,
        fixed_violations=fixed_violations,
        preplaced_violations=preplaced_violations,
    )


def hard_legality_rank(
    inst: Instance, placement: Placement, metrics: MetricMap | None = None
) -> tuple[int, int, int, int, int]:
    return hard_legality(inst, placement, metrics=metrics).rank


def v10_proxy_cost(
    inst: Instance, placement: Placement, metrics: MetricMap | None = None
) -> float:
    metrics = metrics if metrics is not None else placement_metrics(inst, placement)
    bounds = bbox(list(placement.rects.values()))
    total_area = _total_area(inst)
    edge_weight = _edge_weight(inst)
    hpwl_scale = max(1.0, edge_weight * max(1.0, total_area**0.5))
    area_score = bounds.area / max(total_area, 1.0)
    hpwl_score = float(metrics["hpwl_proxy"]) / hpwl_scale
    soft_total = _soft_total(inst, placement, metrics)
    v_rel = soft_total / _soft_capacity(inst)
    return (1.0 + 0.5 * (hpwl_score + area_score)) * math.exp(2.0 * v_rel)


def v10_proxy_rank(
    inst: Instance, placement: Placement, metrics: MetricMap | None = None
) -> tuple[tuple[int, int, int, int, int], float]:
    metrics = metrics if metrics is not None else placement_metrics(inst, placement)
    return (
        hard_legality_rank(inst, placement, metrics=metrics),
        v10_proxy_cost(inst, placement, metrics=metrics),
    )


def v10_proxy_better(
    inst: Instance,
    candidate: Placement,
    current: Placement,
    *,
    candidate_metrics: MetricMap | None = None,
    current_metrics: MetricMap | None = None,
    allow_tie_soft: bool = True,
    allow_proxy_regression: bool = False,
) -> bool:
    candidate_metrics = (
        candidate_metrics
        if candidate_metrics is not None
        else placement_metrics(inst, candidate)
    )
    current_metrics = (
        current_metrics
        if current_metrics is not None
        else placement_metrics(inst, current)
    )
    candidate_hard = hard_legality_rank(inst, candidate, metrics=candidate_metrics)
    current_hard = hard_legality_rank(inst, current, metrics=current_metrics)
    if candidate_hard > current_hard:
        return False
    if candidate_hard < current_hard:
        return True

    candidate_proxy = v10_proxy_cost(inst, candidate, metrics=candidate_metrics)
    current_proxy = v10_proxy_cost(inst, current, metrics=current_metrics)
    tolerance = _tie_tolerance()
    if candidate_proxy < current_proxy * (1.0 - tolerance):
        return True
    if candidate_proxy > current_proxy * (1.0 + tolerance):
        return bool(allow_proxy_regression)

    if not allow_tie_soft:
        return False

    candidate_soft = _soft_key(inst, candidate, candidate_metrics)
    current_soft = _soft_key(inst, current, current_metrics)
    if candidate_soft < current_soft:
        return True
    if candidate_soft > current_soft:
        return False
    return _quality_key(candidate_metrics) < _quality_key(current_metrics)


def _same_dimensions(rect: Rect, target: Rect) -> bool:
    return (
        abs(rect.width - target.width) <= _DIMENSION_EPS
        and abs(rect.height - target.height) <= _DIMENSION_EPS
    )


def _same_rect(rect: Rect, target: Rect) -> bool:
    return (
        abs(rect.x - target.x) <= _POSITION_EPS
        and abs(rect.y - target.y) <= _POSITION_EPS
        and _same_dimensions(rect, target)
    )


def _target_area(inst: Instance, block: int) -> float:
    if block >= inst.area_targets.shape[0]:
        return 0.0
    return float(inst.area_targets[block].item())


def _total_area(inst: Instance) -> float:
    return float(torch.clamp(inst.area_targets[: inst.block_count], min=1.0).sum().item())


def _edge_weight(inst: Instance) -> float:
    return sum(float(w) for *_ij, w in inst.valid_b2b.tolist()) + sum(
        float(w) for *_ij, w in inst.valid_p2b.tolist()
    )


def _soft_capacity(inst: Instance) -> int:
    capacity = max(1, len(inst.boundary))
    capacity += sum(max(0, len(members) - 1) for members in inst.cluster_groups.values())
    capacity += sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
    return capacity


def _soft_total(
    inst: Instance, placement: Placement, metrics: MetricMap | None = None
) -> int:
    if metrics is not None:
        return (
            int(metrics["boundary_violations"])
            + int(metrics["group_violations"])
            + int(metrics["mib_violations"])
        )
    return sum(soft_violation_counts(inst, placement))


def _soft_key(
    inst: Instance, placement: Placement, metrics: MetricMap | None = None
) -> tuple[int, int, int, int]:
    if metrics is not None:
        boundary = int(metrics["boundary_violations"])
        group = int(metrics["group_violations"])
        mib = int(metrics["mib_violations"])
    else:
        boundary, group, mib = soft_violation_counts(inst, placement)
    return (boundary + group + mib, boundary, group, mib)


def _quality_key(metrics: MetricMap) -> tuple[float, float]:
    return (float(metrics["hpwl_proxy"]), float(metrics["bbox_area"]))


def _tie_tolerance() -> float:
    raw = os.environ.get("FLOORSET_V10_PROXY_TIE_TOLERANCE")
    if raw is None:
        return _DEFAULT_TIE_TOLERANCE
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIE_TOLERANCE
    if value < 0.0 or not math.isfinite(value):
        return _DEFAULT_TIE_TOLERANCE
    return value
