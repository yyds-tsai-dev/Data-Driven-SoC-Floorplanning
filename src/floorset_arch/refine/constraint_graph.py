"""Axis-separation DAG extraction from a legal layout.

See docs/design/slack_refiner_spec.md section 1. Reuses
`floorset_arch.legalizer.column_slicing._parse_constraints` for the
constraint bit semantics (fixed/preplaced/mib/cluster/boundary columns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from floorset_arch.legalizer.column_slicing import _parse_constraints, _target

Rect = Tuple[float, float, float, float]

SEP_TOL = 1e-6

# Boundary bitmask (matches evaluator: 1=left, 2=right, 4=top, 8=bottom).
BOUND_LEFT = 1
BOUND_RIGHT = 2
BOUND_TOP = 4
BOUND_BOTTOM = 8

# Sentinel node ids for the frozen bbox wall literals.
WALL_X0 = "WALL_X0"
WALL_X1 = "WALL_X1"
WALL_Y0 = "WALL_Y0"
WALL_Y1 = "WALL_Y1"


@dataclass
class AxisGraph:
    """One axis's separation DAG.

    nodes: block indices 0..n-1 that participate (all of them).
    edges: list of (u, v, gap) meaning coordinate_v >= coordinate_u + gap.
    pinned: node index -> pinned coordinate value (lo == hi == value).
    group_of: node index -> group id for movable-cluster rigid sub-assemblies
        (Phase-1 mitigation: all members of a cluster group translate
        together on this axis). None if not grouped.
    """

    n: int
    edges: List[Tuple[int, int, float]] = field(default_factory=list)
    pinned: Dict[int, float] = field(default_factory=dict)
    group_of: Dict[int, int] = field(default_factory=dict)
    wall_lo: float = float("-inf")
    wall_hi: float = float("inf")
    dim_extent: Dict[int, float] = field(default_factory=dict)


def _bbox(rects: Sequence[Rect]) -> Tuple[float, float, float, float]:
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return x_min, x_max, y_min, y_max


def build_axis_dags(
    rects: Sequence[Rect],
    constraints,
    target_positions,
    b2b=None,
    p2b=None,
    pins=None,
) -> Tuple[AxisGraph, AxisGraph]:
    """Build the x-axis and y-axis separation DAGs from a legal layout.

    Pair-to-axis assignment (spec section 1): for each pair, compute the
    current-geometry y-overlap and x-overlap. If the y-intervals overlap by
    more than SEP_TOL, add an x-edge (smaller-x block -> larger-x block,
    gap = width of the left block). Else if the x-intervals overlap, add a
    y-edge (smaller-y -> larger-y, gap = height of the bottom block). Corner
    -only neighbours (neither overlaps) get no edge. Ties broken by index
    to guarantee acyclicity.
    """
    n = len(rects)
    fixed, preplaced, mib, cluster, boundary = _parse_constraints(constraints, n)

    x0, x1, y0, y1 = _bbox(rects)

    gx = AxisGraph(n=n, wall_lo=x0, wall_hi=x1)
    gy = AxisGraph(n=n, wall_lo=y0, wall_hi=y1)
    for i in range(n):
        gx.dim_extent[i] = rects[i][2]
        gy.dim_extent[i] = rects[i][3]

    # --- pinned nodes: LOCKED (preplaced with valid target) ---
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
            gx.pinned[i] = tx
            gy.pinned[i] = ty

    # --- pinned nodes: boundary-tagged blocks pinned to frozen wall literals ---
    #
    # Only pin a boundary-tagged block to its wall coordinate if it is
    # ALREADY touching that wall (within SEP_TOL) in the incoming layout.
    # The incoming backbone layout can itself carry a pre-existing boundary
    # soft-violation (a boundary-tagged block that is NOT at the wall); in
    # that case forcing gx.pinned[i]/gy.pinned[i] to the wall coordinate
    # would silently relocate the block to a position inconsistent with its
    # separation edges (which were extracted from its ACTUAL, non-wall
    # position), corrupting the DAG and producing overlaps after
    # projection. An already-violating boundary block is instead left
    # unpinned on that axis (free to move within its normal DAG bounds);
    # the soft_violations guard in api.py still forbids the candidate from
    # making boundary worse, so this cannot regress the baseline.
    for i in range(n):
        code = boundary[i]
        if code == 0:
            continue
        bx, by, bw, bh = rects[i]
        if code & BOUND_LEFT and abs(bx - x0) <= SEP_TOL:
            gx.pinned.setdefault(i, x0)
        if code & BOUND_RIGHT and abs(bx + bw - x1) <= SEP_TOL:
            gx.pinned.setdefault(i, x1 - bw)
        if code & BOUND_BOTTOM and abs(by - y0) <= SEP_TOL:
            gy.pinned.setdefault(i, y0)
        if code & BOUND_TOP and abs(by + bh - y1) <= SEP_TOL:
            gy.pinned.setdefault(i, y1 - bh)

    # --- movable cluster groups: rigid sub-assembly (Phase-1 mitigation) ---
    for i in range(n):
        g = cluster[i]
        if g > 0:
            gx.group_of[i] = g
            gy.group_of[i] = g

    # --- pair-to-axis edges from current geometry ---
    for i in range(n):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = rects[j]

            oy = min(yi + hi, yj + hj) - max(yi, yj)
            ox = min(xi + wi, xj + wj) - max(xi, xj)

            if oy > SEP_TOL:
                # x-edge, smaller-x -> larger-x (ties by index)
                if xi < xj or (xi == xj and i < j):
                    u, v = i, j
                else:
                    u, v = j, i
                uw = rects[u][2]
                gx.edges.append((u, v, uw))
            elif ox > SEP_TOL:
                # y-edge, smaller-y -> larger-y (ties by index)
                if yi < yj or (yi == yj and i < j):
                    u, v = i, j
                else:
                    u, v = j, i
                uh = rects[u][3]
                gy.edges.append((u, v, uh))
            # else: corner-only, no edge.

    return gx, gy
