from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

from floorset_arch.models import Rect

EPS = 1e-6


def overlaps(a: Rect, b: Rect, eps: float = EPS) -> bool:
    overlap_x = min(a.right, b.right) - max(a.x, b.x)
    overlap_y = min(a.top, b.top) - max(a.y, b.y)
    return overlap_x > eps and overlap_y > eps


def has_overlaps(rects: Sequence[Rect], eps: float = EPS) -> bool:
    return any(overlaps(a, b, eps=eps) for a, b in combinations(rects, 2))


def bbox(rects: Sequence[Rect]) -> Rect:
    if not rects:
        return Rect(0.0, 0.0, 0.0, 0.0)
    x_min = min(r.x for r in rects)
    y_min = min(r.y for r in rects)
    x_max = max(r.right for r in rects)
    y_max = max(r.top for r in rects)
    return Rect(x_min, y_min, x_max - x_min, y_max - y_min)


def edge_touch_length(a: Rect, b: Rect, eps: float = EPS) -> float:
    if abs(a.right - b.x) <= eps or abs(b.right - a.x) <= eps:
        return max(0.0, min(a.top, b.top) - max(a.y, b.y))
    if abs(a.top - b.y) <= eps or abs(b.top - a.y) <= eps:
        return max(0.0, min(a.right, b.right) - max(a.x, b.x))
    return 0.0


def boundary_satisfied(rect: Rect, bounds: Rect, code: int, eps: float = EPS) -> bool:
    touches = {
        1: abs(rect.x - bounds.x) <= eps,
        2: abs(rect.right - bounds.right) <= eps,
        4: abs(rect.top - bounds.top) <= eps,
        8: abs(rect.y - bounds.y) <= eps,
    }
    return all(touches[bit] for bit in (1, 2, 4, 8) if code & bit)


def candidate_frontier_points(placed: Iterable[Rect]) -> list[tuple[float, float]]:
    rects = list(placed)
    points: set[tuple[float, float]] = {(0.0, 0.0)}
    if not rects:
        return [(0.0, 0.0)]

    xs = {0.0}
    ys = {0.0}
    for rect in rects:
        xs.update({rect.x, rect.right})
        ys.update({rect.y, rect.top})
        points.add((rect.right, rect.y))
        points.add((rect.x, rect.top))
        points.add((rect.right, rect.top))

    bounds = bbox(rects)
    points.update({
        (bounds.right, bounds.y),
        (bounds.x, bounds.top),
        (bounds.right, 0.0),
        (0.0, bounds.top),
    })
    for x in xs:
        points.add((x, bounds.top))
    for y in ys:
        points.add((bounds.right, y))
    return sorted(points, key=lambda p: (p[0] + p[1], p[0], p[1]))


def first_non_overlapping(rect: Rect, placed: Sequence[Rect]) -> bool:
    return not any(overlaps(rect, other) for other in placed)

