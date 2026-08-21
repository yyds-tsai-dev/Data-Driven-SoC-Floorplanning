"""Isolated G0/G1 constructive floorplanning prototype.

This module deliberately has no production entry-point integration.  It works
on the raw partner tensors and returns an evaluator-shaped ``[x, y, w, h]``
array so the architecture can be killed or promoted using paired evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class OracleCodes:
    order: tuple[int, ...]
    log_aspect: np.ndarray
    regions: np.ndarray
    region_bins: int
    frame_aspect_code: int
    utilization_code: int


def default_policies() -> tuple[ConstructivePolicy, ...]:
    """Four fixed-budget arms with genuinely different next-block priorities."""
    return (
        ConstructivePolicy("net_closure", "net_closure", 0.75, 1.20, 0.35, 0.50),
        ConstructivePolicy("constraint_first", "constraint_first", 0.90, 0.85, 0.25, 1.50),
        ConstructivePolicy("large_first", "large_first", 1.20, 0.70, 0.20, 0.50),
        ConstructivePolicy("pin_gravity", "pin_gravity", 0.70, 1.00, 0.80, 0.40),
    )


def encode_oracle_codes(rects: np.ndarray, *, region_bins: int = 16) -> OracleCodes:
    """Encode geometry as order, log-aspect, and coarse normalized regions.

    Source coordinates and the source frame are intentionally not retained.
    This makes the G0 capacity test stricter than copying or snapping the
    repaired-golden rectangles.
    """
    rects = np.asarray(rects, dtype=np.float64)
    if rects.ndim != 2 or rects.shape[1] != 4 or len(rects) == 0:
        raise ValueError("oracle rectangles must have shape [N,4]")
    if region_bins < 2:
        raise ValueError("region_bins must be at least 2")
    lo = rects[:, :2].min(axis=0)
    hi = (rects[:, :2] + rects[:, 2:4]).max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    centres = rects[:, :2] + 0.5 * rects[:, 2:4]
    normalized = np.clip((centres - lo) / span, 0.0, 1.0 - np.finfo(np.float64).eps)
    regions = np.floor(normalized * region_bins).astype(np.int16)
    log_aspect = np.log(np.maximum(rects[:, 2], 1e-12) / np.maximum(rects[:, 3], 1e-12))
    # Morton-like coarse traversal: only region cells, never raw coordinates,
    # decide the encoded priority. Area breaks same-cell ties deterministically.
    area = rects[:, 2] * rects[:, 3]
    frame_aspect_code = int(
        np.clip(np.floor((np.clip(math.log(span[0] / span[1]), -2.0, 2.0) + 2.0) / 4.0 * 32), 0, 31)
    )
    utilization = float(np.clip(area.sum() / (span[0] * span[1]), 0.2, 1.0))
    utilization_code = int(np.clip(np.floor((utilization - 0.2) / 0.8 * 32), 0, 31))
    order = tuple(
        sorted(
            range(len(rects)),
            key=lambda block: (
                int(regions[block, 0]) + int(regions[block, 1]),
                int(regions[block, 1]),
                int(regions[block, 0]),
                -float(area[block]),
                block,
            ),
        )
    )
    return OracleCodes(
        order,
        log_aspect,
        regions,
        region_bins,
        frame_aspect_code,
        utilization_code,
    )


def _constraint_column(constraints: np.ndarray, column: int, n: int) -> np.ndarray:
    if constraints.ndim != 2 or constraints.shape[1] <= column:
        return np.zeros(n, dtype=np.int64)
    return constraints[:n, column].astype(np.int64, copy=False)


def _shape_array(
    area: np.ndarray,
    constraints: np.ndarray,
    target_positions: np.ndarray,
    aspect_codes: np.ndarray | None = None,
) -> np.ndarray:
    n = len(area)
    side = np.sqrt(np.maximum(area, 1e-12))
    shapes = np.column_stack((side, side))
    fixed = _constraint_column(constraints, 0, n) != 0
    preplaced = _constraint_column(constraints, 1, n) != 0
    hard = fixed | preplaced
    if aspect_codes is not None:
        log_aspect = np.asarray(aspect_codes, dtype=np.float64).reshape(n)
        aspect = np.exp(np.clip(log_aspect, -4.0, 4.0))
        shapes[:, 0] = np.sqrt(np.maximum(area, 1e-12) * aspect)
        shapes[:, 1] = np.sqrt(np.maximum(area, 1e-12) / aspect)
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
            reference_area = float(np.mean(area[members]))
            if aspect_codes is None:
                aspect = 1.0
            else:
                aspect = math.exp(float(np.median(np.asarray(aspect_codes)[members])))
            shapes[members] = (math.sqrt(reference_area * aspect), math.sqrt(reference_area / aspect))
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


def _pin_degree(n: int, p2b: list[tuple[int, int, float]]) -> np.ndarray:
    degree = np.zeros(n, dtype=np.float64)
    for _pin, block, weight in p2b:
        if 0 <= block < n:
            degree[block] += weight
    return degree


def _placed_neighbor_weight(
    block: int,
    placed: np.ndarray,
    b2b: list[tuple[int, int, float]],
) -> float:
    total = 0.0
    for left, right, weight in b2b:
        if left == block and placed[right]:
            total += weight
        elif right == block and placed[left]:
            total += weight
    return total


def _choose_next_block(
    remaining: set[int],
    placed: np.ndarray,
    area: np.ndarray,
    constraints: np.ndarray,
    b2b: list[tuple[int, int, float]],
    p2b: list[tuple[int, int, float]],
    mode: str,
    oracle_priority: dict[int, int] | None,
) -> int:
    n = len(area)
    degree = _node_degree(n, b2b)
    pin_degree = _pin_degree(n, p2b)
    constrained = (
        (_constraint_column(constraints, 2, n) != 0).astype(np.float64)
        + (_constraint_column(constraints, 3, n) != 0).astype(np.float64)
        + (_constraint_column(constraints, 4, n) != 0).astype(np.float64)
    )
    boundary = _constraint_column(constraints, 4, n) != 0

    def rank(block: int) -> tuple[float, ...]:
        closure = _placed_neighbor_weight(block, placed, b2b)
        if mode == "oracle" and oracle_priority is not None:
            return (0.0 if boundary[block] else 1.0, float(oracle_priority.get(block, n)), block)
        if mode == "large_first":
            return (-area[block], -degree[block], -constrained[block], block)
        if mode == "constraint_first":
            return (-constrained[block], -closure, -degree[block], -area[block], block)
        if mode == "pin_gravity":
            return (-pin_degree[block], -closure, -degree[block], -area[block], block)
        return (-closure, -degree[block], -constrained[block], -area[block], block)

    return min(remaining, key=rank)


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


def _candidate_points(
    width: float,
    height: float,
    placed: np.ndarray,
    anchor: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    if len(placed) == 0:
        return [anchor] if anchor is not None else [(0.0, 0.0)]
    points: set[tuple[float, float]] = set()
    if anchor is not None:
        points.add(anchor)
    for x, y, other_w, other_h in placed:
        points.update(
            {
                (x + other_w, y),
                (x - width, y),
                (x, y + other_h),
                (x, y - height),
            }
        )
        if anchor is not None:
            points.update(
                {
                    (x + other_w, anchor[1]),
                    (x - width, anchor[1]),
                    (anchor[0], y + other_h),
                    (anchor[0], y - height),
                }
            )
    x_min = float(placed[:, 0].min())
    y_min = float(placed[:, 1].min())
    x_max = float((placed[:, 0] + placed[:, 2]).max())
    y_max = float((placed[:, 1] + placed[:, 3]).max())
    points.update({(x_max, y_min), (x_min - width, y_min), (x_min, y_max), (x_min, y_min - height)})
    if anchor is not None:
        return sorted(points, key=lambda point: (abs(point[0] - anchor[0]) + abs(point[1] - anchor[1]), point))[:48]
    return sorted(points)[:48]


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


def _oracle_frame(
    total_area: float,
    frame_codes: tuple[int, int] | None,
    preplaced_rects: np.ndarray,
) -> tuple[float, float, float, float]:
    if frame_codes is None:
        aspect = 1.0
        utilization = 0.72
    else:
        aspect_code, utilization_code = frame_codes
        aspect = math.exp(-2.0 + (float(aspect_code) + 0.5) / 32.0 * 4.0)
        utilization = 0.2 + (float(utilization_code) + 0.5) / 32.0 * 0.8
    frame_area = max(total_area, 1.0) / max(utilization, 0.2)
    width = math.sqrt(frame_area * aspect)
    height = math.sqrt(frame_area / aspect)
    x0 = y0 = 0.0
    if len(preplaced_rects):
        x0 = min(x0, float(preplaced_rects[:, 0].min()))
        y0 = min(y0, float(preplaced_rects[:, 1].min()))
        width = max(width, float((preplaced_rects[:, 0] + preplaced_rects[:, 2]).max()) - x0)
        height = max(height, float((preplaced_rects[:, 1] + preplaced_rects[:, 3]).max()) - y0)
    return x0, y0, x0 + width, y0 + height


def _region_anchor(
    block: int,
    width: float,
    height: float,
    region_codes: np.ndarray | None,
    region_bins: int,
    frame: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    if region_codes is None:
        return None
    regions = np.asarray(region_codes)
    if regions.shape != (len(regions), 2) or block >= len(regions):
        raise ValueError("region_codes must have shape [N,2]")
    x0, y0, x1, y1 = frame
    normalized = (regions[block].astype(np.float64) + 0.5) / region_bins
    centre = np.array([x0, y0]) + normalized * np.array([x1 - x0, y1 - y0])
    return float(centre[0] - width / 2.0), float(centre[1] - height / 2.0)


def _seat_in_frame(
    point: tuple[float, float],
    width: float,
    height: float,
    boundary_code: int,
    frame: tuple[float, float, float, float],
) -> tuple[float, float]:
    x0, y0, x1, y1 = frame
    x = min(max(point[0], x0), x1 - width)
    y = min(max(point[1], y0), y1 - height)
    if boundary_code & 1:
        x = x0
    if boundary_code & 2:
        x = x1 - width
    if boundary_code & 4:
        y = y1 - height
    if boundary_code & 8:
        y = y0
    return x, y


def _boundary_violations(rects: np.ndarray, boundary: np.ndarray, tol: float = 1e-6) -> int:
    if not len(rects):
        return 0
    x0 = float(rects[:, 0].min())
    y0 = float(rects[:, 1].min())
    x1 = float((rects[:, 0] + rects[:, 2]).max())
    y1 = float((rects[:, 1] + rects[:, 3]).max())
    violations = 0
    for block, code in enumerate(boundary):
        if code == 0:
            continue
        x, y, width, height = rects[block]
        satisfied = (
            (not code & 1 or abs(x - x0) < tol)
            and (not code & 2 or abs(x + width - x1) < tol)
            and (not code & 4 or abs(y + height - y1) < tol)
            and (not code & 8 or abs(y - y0) < tol)
        )
        violations += int(not satisfied)
    return violations


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
    oracle_order: tuple[int, ...] | None = None,
    region_bins: int = 16,
    frame_codes: tuple[int, int] | None = None,
) -> ConstructiveResult:
    """Build one deterministic, hard-legal candidate for a G0/G1 policy."""
    started = time.perf_counter()
    area = np.asarray(area, dtype=np.float64).reshape(-1)
    constraints = np.asarray(constraints, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    b2b_edges = _valid_edges(np.asarray(b2b, dtype=np.float64), len(area))
    p2b_edges = _valid_edges(np.asarray(p2b, dtype=np.float64), max(len(area), len(pins)))
    pins = np.asarray(pins, dtype=np.float64)
    shapes = _shape_array(area, constraints, target_positions, aspect_codes)
    preplaced = _constraint_column(constraints, 1, len(area)) != 0
    cluster = _constraint_column(constraints, 3, len(area))
    boundary = _constraint_column(constraints, 4, len(area))
    oracle_priority = (
        {block: priority for priority, block in enumerate(oracle_order)}
        if oracle_order is not None
        else None
    )

    rects = np.zeros((len(area), 4), dtype=np.float64)
    placed_mask = np.zeros(len(area), dtype=bool)
    order: list[int] = []
    for block in np.flatnonzero(preplaced):
        rects[block] = target_positions[block]
        placed_mask[block] = True
        order.append(int(block))

    frame = _oracle_frame(float(area.sum()), frame_codes, rects[placed_mask])

    scale = math.sqrt(max(float(area.sum()), 1.0))
    disconnected_cluster_fallbacks = 0
    fallback_count = 0
    remaining = set(range(len(area))) - set(order)
    active_cluster = 0
    while remaining:
        eligible = remaining
        if active_cluster:
            group_remaining = {block for block in remaining if cluster[block] == active_cluster}
            if group_remaining:
                eligible = group_remaining
            else:
                active_cluster = 0
        block = _choose_next_block(
            eligible,
            placed_mask,
            area,
            constraints,
            b2b_edges,
            p2b_edges,
            policy.order_mode,
            oracle_priority,
        )
        if cluster[block] and not active_cluster:
            active_cluster = int(cluster[block])
        order.append(block)
        width, height = shapes[block]
        placed = rects[placed_mask]
        group_mask = placed_mask & (cluster == cluster[block]) if cluster[block] else np.zeros(len(area), dtype=bool)
        placed_group = rects[group_mask]
        anchor = _region_anchor(
            block, width, height, region_codes, region_bins, frame
        )
        choices: list[tuple[float, float, float, np.ndarray]] = []
        for x, y in _candidate_points(width, height, placed, anchor):
            if region_codes is not None:
                x, y = _seat_in_frame((x, y), width, height, int(boundary[block]), frame)
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
            if anchor is not None:
                score += policy.anchor_weight * (
                    abs(x - anchor[0]) + abs(y - anchor[1])
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
        remaining.remove(block)

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
            "boundary_violations": _boundary_violations(rects, boundary),
        },
    )


def construct_portfolio(
    area: np.ndarray,
    constraints: np.ndarray,
    target_positions: np.ndarray,
    b2b: np.ndarray,
    p2b: np.ndarray,
    pins: np.ndarray,
    *,
    policies: tuple[ConstructivePolicy, ...] | None = None,
    oracle_codes: OracleCodes | None = None,
) -> list[ConstructiveResult]:
    selected = policies or default_policies()
    return [
        construct_candidate(
            area,
            constraints,
            target_positions,
            b2b,
            p2b,
            pins,
            policy,
            region_codes=oracle_codes.regions if oracle_codes is not None else None,
            aspect_codes=oracle_codes.log_aspect if oracle_codes is not None else None,
            oracle_order=oracle_codes.order if oracle_codes is not None else None,
            region_bins=oracle_codes.region_bins if oracle_codes is not None else 16,
            frame_codes=(oracle_codes.frame_aspect_code, oracle_codes.utilization_code)
            if oracle_codes is not None
            else None,
        )
        for policy in selected
    ]


def weighted_score(costs: Sequence[float], block_counts: Sequence[int]) -> float:
    """Contest aggregation: weighted mean with ``lambda_i = exp(n_i / 12)``."""
    if len(costs) != len(block_counts) or not costs:
        raise ValueError("costs and block_counts must have the same non-zero length")
    counts = np.asarray(block_counts, dtype=np.float64)
    weights = np.exp((counts - counts.max()) / 12.0)
    return float(np.average(np.asarray(costs, dtype=np.float64), weights=weights))


def summarize_g1(
    rows: Sequence[Mapping[str, float | int]],
    all_block_counts: Sequence[int],
) -> dict[str, float | int | bool]:
    """Aggregate the 21-case portfolio using the full-100 score denominator."""
    if not rows or not all_block_counts:
        raise ValueError("G1 rows and full validation block counts are required")
    counts = [int(row["n"]) for row in rows]
    baseline = [float(row["baseline_cost"]) for row in rows]
    candidate = [float(row["candidate_cost"]) for row in rows]
    wins = sum(cand < base - 1e-12 for base, cand in zip(baseline, candidate))
    legal = sum(int(row["legal_candidates"]) for row in rows)
    attempted = sum(int(row["candidate_count"]) for row in rows)
    legal_coverage = legal / max(attempted, 1)

    maximum = max(int(n) for n in all_block_counts)
    full_weights = np.exp((np.asarray(all_block_counts, dtype=np.float64) - maximum) / 12.0)
    row_weights = np.exp((np.asarray(counts, dtype=np.float64) - maximum) / 12.0)
    gain = float(
        np.dot(row_weights, np.asarray(baseline) - np.asarray(candidate))
        / full_weights.sum()
    )
    return {
        "wins": wins,
        "legal_candidates": legal,
        "candidate_count": attempted,
        "legal_coverage": legal_coverage,
        "tail_baseline_score": weighted_score(baseline, counts),
        "tail_candidate_score": weighted_score(candidate, counts),
        "full_total_equivalent_gain": gain,
        "passed": wins >= 4 and gain >= 0.005 and legal_coverage >= 0.90,
    }
