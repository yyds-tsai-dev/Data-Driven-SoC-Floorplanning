"""Weighted-median 1-D target and evaluator-exact centroid HPWL.

See docs/design/slack_refiner_spec.md section 2 and the evaluator reference
FloorSet/iccad2026contest/iccad2026_evaluate.py (HPWL, lines ~153-194).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Rect = Tuple[float, float, float, float]


def weighted_median_target(anchors: Sequence[float], weights: Sequence[float]) -> float:
    """Exact 1-D weighted-median minimizer of sum_k w_k * |t - a_k|.

    Returns 0.0 if there are no anchors (caller should treat this as "no
    pull" and leave the block wherever its bounds/clamp puts it).
    """
    if not anchors:
        return 0.0
    pairs = sorted(zip(anchors, weights), key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        # No informative weight: fall back to the plain median of anchors.
        mid = len(pairs) // 2
        if len(pairs) % 2 == 1:
            return pairs[mid][0]
        return 0.5 * (pairs[mid - 1][0] + pairs[mid][0])

    half = total / 2.0
    cum = 0.0
    for a, w in pairs:
        cum += w
        if cum >= half:
            return a
    return pairs[-1][0]


def hpwl(
    positions: List[Rect],
    b2b: Sequence[Tuple[int, int, float]],
    p2b: Sequence[Tuple[int, int, float]],
    pins: Sequence[Tuple[float, float]],
) -> float:
    """Evaluator-exact centroid HPWL: sum over b2b of weight*|dcx|+|dcy| plus
    sum over p2b of weight*|dcx|+|dcy| against the pin position.

    b2b entries: (i, j, weight); i == -1 marks padding, skipped.
    p2b entries: (pin_idx, block_idx, weight); pin_idx == -1 marks padding.
    """
    n = len(positions)
    total = 0.0
    for edge in b2b:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1:
            continue
        if not (0 <= i < n and 0 <= j < n):
            continue
        x1, y1, w1, h1 = positions[i]
        x2, y2, w2, h2 = positions[j]
        cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
        cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
        total += w * (abs(cx2 - cx1) + abs(cy2 - cy1))

    n_pins = len(pins)
    for edge in p2b:
        p, b, w = int(edge[0]), int(edge[1]), float(edge[2])
        if p == -1:
            continue
        if not (0 <= b < n and 0 <= p < n_pins):
            continue
        px, py = float(pins[p][0]), float(pins[p][1])
        bx, by, bw, bh = positions[b]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        total += w * (abs(px - cx) + abs(py - cy))
    return total
