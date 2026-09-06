"""Deterministic equal-size block matching for retrieved layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
            candidate_columns = np.flatnonzero(~used[1:]) + 1
            reduced = (
                cost[active_row - 1, candidate_columns - 1]
                - u[active_row]
                - v[candidate_columns]
            )
            improved = reduced < min_value[candidate_columns]
            improved_columns = candidate_columns[improved]
            min_value[improved_columns] = reduced[improved]
            way[improved_columns] = column

            next_column = int(candidate_columns[np.argmin(min_value[candidate_columns])])
            delta = min_value[next_column]
            used_columns = np.flatnonzero(used)
            u[p[used_columns]] += delta
            v[used_columns] -= delta
            min_value[~used] -= delta
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
    pooled = np.concatenate((source, target), axis=0)
    input_scale = np.maximum(np.max(np.abs(pooled), axis=0), 1.0)
    scaled = pooled / input_scale
    center = scaled.mean(axis=0)
    spread = np.sqrt(np.mean(np.square(scaled - center), axis=0))
    standardized = (scaled - center) / np.maximum(spread, 1e-12)
    source_standardized = standardized[: len(source)]
    target_standardized = standardized[len(source) :]
    return np.abs(target_standardized[:, None, :] - source_standardized[None, :, :]).mean(axis=-1)


def _assignment_confidence(cost: np.ndarray, assignment: np.ndarray) -> float:
    """Return a bounded diagnostic margin for an exact target-to-source assignment."""
    if len(assignment) == 1:
        return 1.0

    rows = np.arange(len(assignment))
    chosen = cost[rows, assignment]
    row_minimum = cost.min(axis=1)
    best_alternative = np.partition(cost, 1, axis=1)[:, 1]
    unique_row_minimum = (chosen == row_minimum) & (best_alternative > chosen)
    margins = np.zeros(len(assignment), dtype=np.float64)
    margins[unique_row_minimum] = (
        (best_alternative[unique_row_minimum] - chosen[unique_row_minimum])
        / best_alternative[unique_row_minimum]
    )
    return float(np.clip(margins, 0.0, 1.0).mean())


def match_blocks(source_nodes, target_nodes, max_cost) -> MatchResult:
    """Match each target node to one source node using all-soft feature costs."""
    source = np.asarray(source_nodes, dtype=np.float64)
    target = np.asarray(target_nodes, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("first retrieval version requires equal N and feature width")
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("retrieval node features must have nonzero block count and feature width")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("retrieval node features must be finite")
    if not np.isfinite(max_cost):
        raise ValueError("maximum matching cost must be finite")

    cost = _standardized_soft_l1(source, target)
    assignment = hungarian_min_cost(cost)
    chosen = cost[np.arange(len(assignment)), assignment]
    confidence = _assignment_confidence(cost, assignment)
    total = float(chosen.mean())
    return MatchResult(
        target_to_source=assignment,
        total_cost=total,
        confidence=confidence,
        accepted=bool(total <= max_cost),
    )
