"""Reinsert movable outliers inside preplaced boundary-tag frame lines.

This is the G0 mechanism core.  It does not know evaluator baselines and does
not accept a move by itself; the guarded wrapper added separately owns that
decision.  Shapes and locked rectangles are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np


EPS = 1e-6


@dataclass(frozen=True, order=True)
class FrameTarget:
    """One attainable bbox wall fixed by a tagged, locked rectangle."""

    axis: int
    side: int
    line: float
    owner: int


def frame_targets(
    rects: np.ndarray,
    locked: np.ndarray,
    boundary: Sequence[int],
) -> list[FrameTarget]:
    """Return overshot tag lines that no other locked rectangle crosses."""
    p = np.asarray(rects, dtype=np.float64)
    lk = np.asarray(locked, dtype=bool)
    bnd = np.asarray(boundary, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 4 or len(lk) != len(p):
        return []
    ids = np.flatnonzero(lk)
    if not len(ids):
        return []

    lo = (float(p[:, 0].min()), float(p[:, 1].min()))
    hi = (float((p[:, 0] + p[:, 2]).max()),
          float((p[:, 1] + p[:, 3]).max()))
    found: dict[tuple[int, int, float], FrameTarget] = {}
    for i in ids:
        code = int(bnd[i]) if i < len(bnd) else 0
        for bit, axis, side in ((1, 0, 0), (2, 0, 1),
                                (4, 1, 1), (8, 1, 0)):
            if not code & bit:
                continue
            line = float(p[i, axis] + (p[i, axis + 2] if side else 0.0))
            wall = hi[axis] if side else lo[axis]
            overshot = wall > line + EPS if side else wall < line - EPS
            if not overshot:
                continue
            locked_edge = p[ids, axis] + (p[ids, axis + 2] if side else 0.0)
            crosses = locked_edge > line + EPS if side else locked_edge < line - EPS
            if bool(crosses.any()):
                continue
            key = (axis, side, line)
            found.setdefault(key, FrameTarget(axis, side, line, int(i)))
    return sorted(found.values())


def _overlaps(rect: np.ndarray, obstacles: np.ndarray) -> bool:
    if not len(obstacles):
        return False
    ox = np.minimum(rect[0] + rect[2], obstacles[:, 0] + obstacles[:, 2]) \
        - np.maximum(rect[0], obstacles[:, 0])
    oy = np.minimum(rect[1] + rect[3], obstacles[:, 1] + obstacles[:, 3]) \
        - np.maximum(rect[1], obstacles[:, 1])
    return bool(((ox > EPS) & (oy > EPS)).any())


def _axis_slots(
    low: float,
    high: float,
    size: float,
    preferred: float,
    obstacles: np.ndarray,
    axis: int,
) -> list[float]:
    values = [low, high - size, min(max(preferred, low), high - size)]
    if len(obstacles):
        values.extend(float(v) for v in obstacles[:, axis] - size)
        values.extend(float(v) for v in obstacles[:, axis] + obstacles[:, axis + 2])
    legal = {float(v) for v in values
             if v >= low - EPS and v + size <= high + EPS}
    return sorted(legal, key=lambda v: (abs(v - preferred), v))


def reinsert_to_target(
    rects: np.ndarray,
    target: FrameTarget,
    locked: np.ndarray,
    cluster: Sequence[int],
    hpwl_fn: Callable[[np.ndarray], float],
) -> np.ndarray | None:
    """Move every movable outlier into an obstacle-edge slot in the frame."""
    p = np.asarray(rects, dtype=np.float64)
    lk = np.asarray(locked, dtype=bool)
    clu = np.asarray(cluster, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 4 or len(lk) != len(p):
        return None
    if not np.isfinite(p).all() or (p[:, 2:] <= 0.0).any():
        return None

    xlo = float(p[:, 0].min())
    xhi = float((p[:, 0] + p[:, 2]).max())
    ylo = float(p[:, 1].min())
    yhi = float((p[:, 1] + p[:, 3]).max())
    if target.axis == 0:
        if target.side:
            xhi = target.line
        else:
            xlo = target.line
    else:
        if target.side:
            yhi = target.line
        else:
            ylo = target.line
    if xhi <= xlo + EPS or yhi <= ylo + EPS:
        return None

    if target.side:
        outlier = p[:, target.axis] + p[:, target.axis + 2] > target.line + EPS
    else:
        outlier = p[:, target.axis] < target.line - EPS
    ids = np.flatnonzero(outlier)
    if not len(ids):
        return None
    if bool(lk[ids].any()) or bool((clu[ids] != 0).any()):
        return None

    q = p.copy()
    retained = [int(i) for i in range(len(p)) if not outlier[i]]
    placed = list(retained)
    order = sorted((int(i) for i in ids),
                   key=lambda i: (-float(p[i, 2] * p[i, 3]), i))
    for i in order:
        obstacles = q[placed] if placed else np.empty((0, 4), dtype=np.float64)
        xs = _axis_slots(xlo, xhi, float(p[i, 2]), float(p[i, 0]), obstacles, 0)
        ys = _axis_slots(ylo, yhi, float(p[i, 3]), float(p[i, 1]), obstacles, 1)
        best = None
        for x, y in product(xs, ys):
            rect = np.asarray((x, y, p[i, 2], p[i, 3]), dtype=np.float64)
            if _overlaps(rect, obstacles):
                continue
            trial = q.copy()
            trial[i] = rect
            try:
                hp = float(hpwl_fn(trial))
            except Exception:
                return None
            if not np.isfinite(hp):
                continue
            displacement = abs(x - p[i, 0]) + abs(y - p[i, 1])
            key = (hp, displacement, x, y)
            if best is None or key < best[0]:
                best = (key, rect)
        if best is None:
            return None
        q[i] = best[1]
        placed.append(i)

    for i in range(len(q)):
        if _overlaps(q[i], np.delete(q, i, axis=0)):
            return None
    q[lk] = p[lk]
    return q
