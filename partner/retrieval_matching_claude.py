"""Deterministic equal-size block matching for retrieved layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_HARD_FEATURE_COLUMNS = (5, 6, 7, 8)


@dataclass(frozen=True)
class MatchResult:
    target_to_source: np.ndarray
    total_cost: float
    confidence: float
    accepted: bool


def hungarian_min_cost(cost: np.ndarray) -> np.ndarray:
    """Return the minimum-cost source column selected for each target row."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Hungarian input must be finite square cost matrix")
    if not np.isfinite(cost).all():
        raise ValueError("Hungarian cost must be finite")

    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for row in range(1, n + 1):
        p[0] = row
        min_value = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = p[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, n + 1):
                if used[candidate]:
                    continue
                reduced = cost[active_row - 1, candidate - 1] - u[active_row] - v[candidate]
                if reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    way[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(n + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break

    assignment = np.empty(n, dtype=np.int64)
    for column in range(1, n + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


def _standardized_soft_l1(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return target-to-source L1 costs without overflowing finite inputs."""
    soft_columns = np.setdiff1d(np.arange(source.shape[1]), _HARD_FEATURE_COLUMNS)
    pooled = np.concatenate((source[:, soft_columns], target[:, soft_columns]), axis=0)
    input_scale = np.maximum(np.max(np.abs(pooled), axis=0), 1.0)
    scaled = pooled / input_scale
    center = scaled.mean(axis=0)
    spread = np.sqrt(np.mean(np.square(scaled - center), axis=0))
    standardized = (scaled - center) / np.maximum(spread, 1e-12)
    source_standardized = standardized[: len(source)]
    target_standardized = standardized[len(source) :]
    return np.abs(target_standardized[:, None, :] - source_standardized[None, :, :]).mean(axis=-1)


def _forbidden_cost(soft_cost: np.ndarray) -> float:
    """Return a per-edge penalty larger than any all-soft assignment sum."""
    maximum_soft_edge = float(np.max(soft_cost))
    return float(np.nextafter(len(soft_cost) * maximum_soft_edge, np.inf))


def match_blocks(source_nodes, target_nodes, max_cost) -> MatchResult:
    """Match each target node to one compatible source node.

    The returned vector is indexed by target block and contains source block
    indices.  Fixed/preplaced/group membership features are non-negotiable;
    boundary-direction features remain part of the soft distance only.
    """
    source = np.asarray(source_nodes, dtype=np.float64)
    target = np.asarray(target_nodes, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("first retrieval version requires equal N and feature width")
    if source.shape[0] == 0 or source.shape[1] <= max(_HARD_FEATURE_COLUMNS):
        raise ValueError("retrieval node features must contain at least one compatible block")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("retrieval node features must be finite")
    if not np.isfinite(max_cost):
        raise ValueError("maximum matching cost must be finite")

    soft_cost = _standardized_soft_l1(source, target)
    forbidden_cost = _forbidden_cost(soft_cost)
    hard = np.any(
        target[:, None, _HARD_FEATURE_COLUMNS] != source[None, :, _HARD_FEATURE_COLUMNS], axis=-1
    )
    cost = np.where(hard, forbidden_cost, soft_cost)
    assignment = hungarian_min_cost(cost)
    chosen = cost[np.arange(len(assignment)), assignment]
    second = np.partition(cost, 1, axis=1)[:, 1] if len(assignment) > 1 else chosen + 1.0
    confidence = float(np.mean(second - chosen))
    total = float(chosen.mean())
    return MatchResult(
        target_to_source=assignment,
        total_cost=total,
        confidence=confidence,
        accepted=bool(total <= max_cost and chosen.max() < forbidden_cost),
    )
