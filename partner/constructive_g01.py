"""Isolated G0/G1 constructive floorplanning prototype.

This module deliberately has no production entry-point integration.  It works
on the raw partner tensors and returns an evaluator-shaped ``[x, y, w, h]``
array so the architecture can be killed or promoted using paired evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ConstructivePolicy:
    name: str
    order_mode: str
    bbox_weight: float
    hpwl_weight: float
    anchor_weight: float
    constraint_weight: float


@dataclass(frozen=True)
class ConstructiveResult:
    policy: str
    rects: np.ndarray
    order: tuple[int, ...]
    elapsed_s: float
    hard_legal: bool
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def _constraint_column(constraints: np.ndarray, column: int, n: int) -> np.ndarray:
    if constraints.ndim != 2 or constraints.shape[1] <= column:
        return np.zeros(n, dtype=np.int64)
    return constraints[:n, column].astype(np.int64, copy=False)


def _shape_array(
    area: np.ndarray,
    constraints: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    n = len(area)
    side = np.sqrt(np.maximum(area, 1e-12))
    shapes = np.column_stack((side, side))
    fixed = _constraint_column(constraints, 0, n) != 0
    preplaced = _constraint_column(constraints, 1, n) != 0
    hard = fixed | preplaced
    if hard.any():
        shapes[hard] = target_positions[hard, 2:4]
    mib = _constraint_column(constraints, 2, n)
    for group in sorted(set(mib) - {0}):
        members = np.flatnonzero(mib == group)
        hard_members = members[hard[members]]
        if len(hard_members):
            reference = shapes[hard_members[0]].copy()
            reference_area = float(reference[0] * reference[1])
            compatible = (
                np.abs(area[members] - reference_area) / np.maximum(area[members], 1e-12)
                <= 0.005
            )
            shapes[members[compatible]] = reference
        elif np.ptp(area[members]) / max(float(area[members].max()), 1e-12) <= 0.005:
            side = math.sqrt(float(np.mean(area[members])))
            shapes[members] = (side, side)
    return shapes


def _valid_edges(edges: np.ndarray, n: int) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    if edges.ndim != 2 or edges.shape[1] < 3:
        return out
    for row in edges:
        a, b, weight = int(row[0]), int(row[1]), float(row[2])
        if 0 <= a < n and 0 <= b < n and a != b and weight > 0:
            out.append((a, b, weight))
    return out


def _node_degree(n: int, b2b: list[tuple[int, int, float]]) -> np.ndarray:
    degree = np.zeros(n, dtype=np.float64)
    for left, right, weight in b2b:
        degree[left] += weight
        degree[right] += weight
    return degree


def _block_order(
    area: np.ndarray,
    constraints: np.ndarray,
    b2b: list[tuple[int, int, float]],
    preplaced: np.ndarray,
    mode: str,
) -> list[int]:
    n = len(area)
    degree = _node_degree(n, b2b)
    constrained = (
        (_constraint_column(constraints, 2, n) != 0).astype(np.float64)
        + (_constraint_column(constraints, 3, n) != 0).astype(np.float64)
        + (_constraint_column(constraints, 4, n) != 0).astype(np.float64)
    )
    free = [block for block in range(n) if not preplaced[block]]
    if mode == "large_first":
        free.sort(key=lambda block: (-area[block], -degree[block], block))
    elif mode == "constraint_first":
        free.sort(key=lambda block: (-constrained[block], -degree[block], -area[block], block))
    else:
        free.sort(key=lambda block: (-degree[block], -area[block], block))
    cluster = _constraint_column(constraints, 3, n)
    grouped: list[int] = []
    emitted: set[int] = set()
    for block in free:
        if block in emitted:
            continue
        group = cluster[block]
        if group == 0:
            grouped.append(block)
            emitted.add(block)
            continue
        members = [member for member in free if cluster[member] == group and member not in emitted]
        grouped.extend(members)
        emitted.update(members)
    return list(np.flatnonzero(preplaced)) + grouped


def _overlaps_any(rect: np.ndarray, placed: np.ndarray) -> bool:
    if len(placed) == 0:
        return False
    overlap_x = np.minimum(rect[0] + rect[2], placed[:, 0] + placed[:, 2]) - np.maximum(
        rect[0], placed[:, 0]
    )
    overlap_y = np.minimum(rect[1] + rect[3], placed[:, 1] + placed[:, 3]) - np.maximum(
        rect[1], placed[:, 1]
    )
    return bool(((overlap_x > 1e-7) & (overlap_y > 1e-7)).any())


def _edge_touches_any(rect: np.ndarray, others: np.ndarray, tol: float = 1e-7) -> bool:
    if len(others) == 0:
        return False
    overlap_x = np.minimum(rect[0] + rect[2], others[:, 0] + others[:, 2]) - np.maximum(
        rect[0], others[:, 0]
    )
    overlap_y = np.minimum(rect[1] + rect[3], others[:, 1] + others[:, 3]) - np.maximum(
        rect[1], others[:, 1]
    )
    vertical = (np.abs(overlap_x) <= tol) & (overlap_y > tol)
    horizontal = (np.abs(overlap_y) <= tol) & (overlap_x > tol)
    return bool((vertical | horizontal).any())


def _bbox_area(rects: np.ndarray) -> float:
    if len(rects) == 0:
        return 0.0
    return float(
        ((rects[:, 0] + rects[:, 2]).max() - rects[:, 0].min())
        * ((rects[:, 1] + rects[:, 3]).max() - rects[:, 1].min())
    )


def _candidate_points(width: float, height: float, placed: np.ndarray) -> list[tuple[float, float]]:
    if len(placed) == 0:
        return [(0.0, 0.0)]
    points: set[tuple[float, float]] = set()
    for x, y, other_w, other_h in placed:
        points.update(
            {
                (x + other_w, y),
                (x - width, y),
                (x, y + other_h),
                (x, y - height),
            }
        )
    x_min = float(placed[:, 0].min())
    y_min = float(placed[:, 1].min())
    x_max = float((placed[:, 0] + placed[:, 2]).max())
    y_max = float((placed[:, 1] + placed[:, 3]).max())
    points.update({(x_max, y_min), (x_min - width, y_min), (x_min, y_max), (x_min, y_min - height)})
    return sorted(points)


def _incremental_hpwl(
    block: int,
    rect: np.ndarray,
    rects: np.ndarray,
    placed_mask: np.ndarray,
    b2b: list[tuple[int, int, float]],
    p2b: list[tuple[int, int, float]],
    pins: np.ndarray,
) -> float:
    cx = rect[0] + 0.5 * rect[2]
    cy = rect[1] + 0.5 * rect[3]
    score = 0.0
    for left, right, weight in b2b:
        if left == block and placed_mask[right]:
            other = rects[right]
        elif right == block and placed_mask[left]:
            other = rects[left]
        else:
            continue
        score += weight * (
            abs(cx - (other[0] + 0.5 * other[2]))
            + abs(cy - (other[1] + 0.5 * other[3]))
        )
    for pin, owner, weight in p2b:
        if owner == block and 0 <= pin < len(pins):
            score += weight * (abs(cx - pins[pin, 0]) + abs(cy - pins[pin, 1]))
    return score


def _hard_legal(
    rects: np.ndarray,
    area: np.ndarray,
    constraints: np.ndarray,
    target_positions: np.ndarray,
) -> bool:
    if rects.shape != (len(area), 4) or not np.isfinite(rects).all():
        return False
    fixed = _constraint_column(constraints, 0, len(area)) != 0
    preplaced = _constraint_column(constraints, 1, len(area)) != 0
    soft = ~(fixed | preplaced)
    if soft.any() and (
        np.abs(rects[soft, 2] * rects[soft, 3] - area[soft])
        / np.maximum(area[soft], 1e-12)
        > 0.005
    ).any():
        return False
    hard = fixed | preplaced
    if hard.any() and (np.abs(rects[hard, 2:4] - target_positions[hard, 2:4]) > 1e-5).any():
        return False
    if preplaced.any() and (np.abs(rects[preplaced, :2] - target_positions[preplaced, :2]) > 1e-5).any():
        return False
    for block in range(len(rects)):
        if _overlaps_any(rects[block], rects[:block]):
            return False
    return True


def construct_candidate(
    area: np.ndarray,
    constraints: np.ndarray,
    target_positions: np.ndarray,
    b2b: np.ndarray,
    p2b: np.ndarray,
    pins: np.ndarray,
    policy: ConstructivePolicy,
    *,
    region_codes: np.ndarray | None = None,
    aspect_codes: np.ndarray | None = None,
) -> ConstructiveResult:
    """Build one deterministic, hard-legal candidate for a G0/G1 policy."""
    del region_codes, aspect_codes  # introduced by the later oracle-code task
    started = time.perf_counter()
    area = np.asarray(area, dtype=np.float64).reshape(-1)
    constraints = np.asarray(constraints, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    b2b_edges = _valid_edges(np.asarray(b2b, dtype=np.float64), len(area))
    p2b_edges = _valid_edges(np.asarray(p2b, dtype=np.float64), max(len(area), len(pins)))
    pins = np.asarray(pins, dtype=np.float64)
    shapes = _shape_array(area, constraints, target_positions)
    preplaced = _constraint_column(constraints, 1, len(area)) != 0
    cluster = _constraint_column(constraints, 3, len(area))
    order = _block_order(area, constraints, b2b_edges, preplaced, policy.order_mode)

    rects = np.zeros((len(area), 4), dtype=np.float64)
    placed_mask = np.zeros(len(area), dtype=bool)
    for block in np.flatnonzero(preplaced):
        rects[block] = target_positions[block]
        placed_mask[block] = True

    scale = math.sqrt(max(float(area.sum()), 1.0))
    disconnected_cluster_fallbacks = 0
    fallback_count = 0
    for block in order:
        if placed_mask[block]:
            continue
        width, height = shapes[block]
        placed = rects[placed_mask]
        group_mask = placed_mask & (cluster == cluster[block]) if cluster[block] else np.zeros(len(area), dtype=bool)
        placed_group = rects[group_mask]
        choices: list[tuple[float, float, float, np.ndarray]] = []
        for x, y in _candidate_points(width, height, placed):
            candidate = np.array([x, y, width, height], dtype=np.float64)
            if _overlaps_any(candidate, placed):
                continue
            if len(placed_group) and not _edge_touches_any(candidate, placed_group):
                continue
            trial = np.vstack((placed, candidate))
            score = policy.bbox_weight * _bbox_area(trial) / max(float(area.sum()), 1.0)
            score += policy.hpwl_weight * _incremental_hpwl(
                block, candidate, rects, placed_mask, b2b_edges, p2b_edges, pins
            ) / scale
            choices.append((score, x, y, candidate))
        if choices:
            choices.sort(key=lambda item: (item[0], item[1], item[2]))
            rects[block] = choices[0][3]
        else:
            fallback_count += 1
            fallback = None
            if len(placed_group):
                for x, y in _candidate_points(width, height, placed_group):
                    candidate = np.array([x, y, width, height], dtype=np.float64)
                    if not _overlaps_any(candidate, placed) and _edge_touches_any(candidate, placed_group):
                        fallback = candidate
                        break
            if fallback is None:
                x = float((placed[:, 0] + placed[:, 2]).max()) if len(placed) else 0.0
                y = float(placed[:, 1].min()) if len(placed) else 0.0
                fallback = np.array([x, y, width, height], dtype=np.float64)
                if len(placed_group):
                    disconnected_cluster_fallbacks += 1
            rects[block] = fallback
        placed_mask[block] = True

    elapsed = time.perf_counter() - started
    legal = _hard_legal(rects, area, constraints, target_positions)
    return ConstructiveResult(
        policy=policy.name,
        rects=rects,
        order=tuple(order),
        elapsed_s=elapsed,
        hard_legal=legal,
        diagnostics={
            "fallbacks": fallback_count,
            "disconnected_cluster_fallbacks": disconnected_cluster_fallbacks,
        },
    )
