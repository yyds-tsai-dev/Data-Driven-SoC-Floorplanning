"""Synthetic FloorSet-like instance builder for partner SA tests/benchmarks.

Deliberately self-contained: no dataset access, no evaluator, single process.
Produces the exact tensor shapes `_ColumnOptimizer` consumes, with the
structural features the layout path branches on -- fixed-shape blocks,
preplaced obstacles, boundary tags, cluster groups, soft-MIB groups, b2b/p2b
connectivity and perimeter pins.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SynthInstance:
    rects: List[Tuple[float, float, float, float]]
    area_targets: torch.Tensor
    constraints: torch.Tensor
    target_positions: torch.Tensor
    b2b: torch.Tensor
    p2b: torch.Tensor
    pins: torch.Tensor


def build_instance(n: int = 100, seed: int = 0,
                   frac_fixed: float = 0.10,
                   n_preplaced: int = 2,
                   frac_boundary: float = 0.16,
                   n_clusters: int = 6,
                   n_mib: int = 3,
                   edge_mult: float = 3.0,
                   n_pins: int = 40,
                   anchor_clusters: int = 0) -> SynthInstance:
    """`anchor_clusters`: how many cluster groups also contain a preplaced
    block. Those groups become "anchored units", which is the only way to
    reach `_stack_column`'s anchor-gluing branch and `_place_unit_down`."""
    rng = random.Random(seed)

    areas = [math.exp(rng.gauss(4.2, 0.55)) for _ in range(n)]
    total = sum(areas)
    side = math.sqrt(total / 0.96)

    fixed = [0.0] * n
    preplaced = [0.0] * n
    mib = [0.0] * n
    cluster = [0.0] * n
    boundary = [0.0] * n
    tpos = [[-1.0, -1.0, -1.0, -1.0] for _ in range(n)]

    idx = list(range(n))
    rng.shuffle(idx)
    cursor = 0

    # fixed-shape blocks: exact (w, h), free position
    n_fixed = int(round(frac_fixed * n))
    for i in idx[cursor:cursor + n_fixed]:
        a = areas[i]
        ar = math.exp(rng.gauss(0.0, 0.45))
        w = math.sqrt(a * ar)
        h = a / w
        fixed[i] = 1.0
        tpos[i] = [-1.0, -1.0, w, h]
    cursor += n_fixed

    # preplaced blocks: exact (x, y, w, h) obstacles
    for i in idx[cursor:cursor + n_preplaced]:
        a = areas[i]
        w = math.sqrt(a)
        h = a / w
        px = rng.uniform(0.05, 0.6) * side
        py = rng.uniform(0.05, 0.6) * side
        preplaced[i] = 1.0
        tpos[i] = [px, py, w, h]
    cursor += n_preplaced

    # boundary tags
    n_bnd = int(round(frac_boundary * n))
    for i in idx[cursor:cursor + n_bnd]:
        boundary[i] = float(rng.choice([1, 2, 4, 8, 8, 4]))
    cursor += n_bnd

    # cluster groups
    pool = [i for i in range(n) if preplaced[i] == 0.0]
    rng.shuffle(pool)
    p = 0
    pre_ids = [i for i in range(n) if preplaced[i] != 0.0]
    for g in range(1, n_clusters + 1):
        m = rng.randint(2, 5)
        members = pool[p:p + m]
        p += m
        if len(members) < 2:
            break
        for i in members:
            cluster[i] = float(g)
        # glue a preplaced block into the first few groups -> anchored units
        if g <= anchor_clusters and g - 1 < len(pre_ids):
            cluster[pre_ids[g - 1]] = float(g)

    # MIB groups (equal areas within a group)
    for g in range(1, n_mib + 1):
        m = rng.randint(2, 3)
        members = pool[p:p + m]
        p += m
        if len(members) < 2:
            break
        a = areas[members[0]]
        for i in members:
            areas[i] = a
            mib[i] = float(g)

    # b2b connectivity
    n_edges = int(edge_mult * n)
    edges = set()
    while len(edges) < n_edges:
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        edges.add((min(a, b), max(a, b)))
    b2b = torch.tensor([[float(a), float(b), float(rng.randint(1, 8))]
                        for (a, b) in sorted(edges)], dtype=torch.float32)

    # pins on the frame perimeter + p2b
    pins = []
    for _ in range(n_pins):
        t = rng.random()
        if t < 0.25:
            pins.append([0.0, rng.uniform(0, side)])
        elif t < 0.5:
            pins.append([side, rng.uniform(0, side)])
        elif t < 0.75:
            pins.append([rng.uniform(0, side), 0.0])
        else:
            pins.append([rng.uniform(0, side), side])
    pins_t = torch.tensor(pins, dtype=torch.float32)
    p2b = torch.tensor([[float(rng.randrange(n_pins)), float(rng.randrange(n)),
                         float(rng.randint(1, 5))]
                        for _ in range(2 * n_pins)], dtype=torch.float32)

    # seed rectangles: rough centroid scatter (what the backbone hands in)
    rects = []
    for i in range(n):
        a = areas[i]
        if preplaced[i]:
            rects.append(tuple(tpos[i]))
            continue
        if fixed[i]:
            w, h = tpos[i][2], tpos[i][3]
        else:
            w = math.sqrt(a)
            h = a / w
        rects.append((rng.uniform(0, max(side - w, 1e-3)),
                      rng.uniform(0, max(side - h, 1e-3)), w, h))

    constraints = torch.tensor(
        [[fixed[i], preplaced[i], mib[i], cluster[i], boundary[i]]
         for i in range(n)], dtype=torch.float32)

    return SynthInstance(
        rects=rects,
        area_targets=torch.tensor(areas, dtype=torch.float32),
        constraints=constraints,
        target_positions=torch.tensor(tpos, dtype=torch.float32),
        b2b=b2b,
        p2b=p2b,
        pins=pins_t,
    )


def make_optimizer(inst: SynthInstance, seed: int = 0, deadline: Optional[float] = None):
    import column_sa_legalizer as csl
    return csl._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, deadline, seed=seed,
    )
