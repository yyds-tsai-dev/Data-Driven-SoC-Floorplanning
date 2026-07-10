"""Track-B generative-ML gate: order-faithful exact decoder probe.

Hypothesis under test: a model's value is its TOPOLOGY (pairwise relative
order), and the right consumer is an order-faithful EXACT decoder:

    hint (cx,cy,w,h) per block
      -> pairwise separation orders (which axis, which side)
      -> shapes: exact-area soft sizing with hint aspect (MIB shared)
      -> constraint DAG per axis
      -> longest-path compaction (overlap-free BY CONSTRUCTION)
      -> constraint pass (preplaced pin + boundary snap + cluster rigidity)
      -> slack-refine polish (weighted-median projection, NOT the SA backbone)
      -> official offline score.

E1 (the gate) feeds GOLDEN layouts as hints and measures how much of the
golden score (1.1079) survives order-extract -> compact -> constrain. This
isolates DECODER FIDELITY from model quality: if golden hints decode to a
near-GT score, model quality is the only remaining lever; if not, the decoder
concept is the bottleneck and we learn which constraint kills it.

Self-contained: imports only refine.{guards,slack_solve,constraint_graph,
wirelength}, legalizer.column_slicing._parse_constraints/_target, and the
official evaluator's evaluate_solution/compute_total_score. Does NOT import
refine.api or floorset_arch.optimizer; legalizer.column_backbone is imported
ONLY lazily by the production hint provider (golden/gnn/diffusion paths never
touch it). No src/ edits.

Usage (from repo root, tcsh):
  cd FloorSet/iccad2026contest
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
      <repo>/scripts/probes/gen_decoder_probe.py --hints golden [--cases N]

Hint sources:
  --hints golden        E1: golden bbox layout as hints (the gate)
  --hints gnn           E2: Anchor-GNN anchor+aspect heads as hints
  --hints diffusion     E3: v11 diffusion samples as hints
  --hints production    D1: production column-backbone layout as hints
                        (decoder-as-polish; reports paired baseline/decoded/
                        portfolio(min) totals -- portfolio(min) is the
                        strict-better deployment semantics)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# --- make src/ and this probe dir importable (mirrors gt_seed_optimizer.py) ---
_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))  # so `import _probe_hints` works
_REPO = None
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        _REPO = _p
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
if _REPO is None:
    _REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
    sys.path.insert(0, str(_REPO / "src"))

# Official evaluator scoring (source of truth for offline score).
from iccad2026_evaluate import (  # noqa: E402
    evaluate_solution,
    compute_total_score,
)

# Allowed src imports (no api / optimizer / column_backbone).
from floorset_arch.legalizer.column_slicing import (  # noqa: E402
    _parse_constraints,
    _target,
)
from floorset_arch.refine.constraint_graph import (  # noqa: E402
    build_axis_dags,
    AxisGraph,
)
from floorset_arch.refine.slack_solve import project_axis  # noqa: E402
from floorset_arch.refine.wirelength import hpwl  # noqa: E402
from floorset_arch.refine import guards as refine_guards  # noqa: E402

Rect = Tuple[float, float, float, float]

# Aspect clamp: matches column_slicing convention (log-aspect in [-3, 3]).
LOG_ASPECT_CLAMP = 3.0
SEP_TOL = 1e-6

BOUND_LEFT = 1
BOUND_RIGHT = 2
BOUND_TOP = 4
BOUND_BOTTOM = 8


# =============================================================================
# Data extraction (evaluator-faithful)
# =============================================================================
def _n_of(sample) -> int:
    at = sample["input"][0]
    return int((at != -1).sum().item())


def _golden_rects(sample, n: int) -> List[Rect]:
    """Golden layout as per-block bbox rects (evaluator _extract_baseline)."""
    polys = sample["label"][0]
    out: List[Rect] = []
    for i in range(n):
        block = polys[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            mn = valid.min(dim=0).values
            mx = valid.max(dim=0).values
            out.append((float(mn[0]), float(mn[1]),
                        float(mx[0] - mn[0]), float(mx[1] - mn[1])))
        else:
            out.append((0.0, 0.0, 1.0, 1.0))
    return out


def _baseline(sample, n: int, b2b, p2b, pins) -> Dict[str, float]:
    """Replicate ContestEvaluator._extract_baseline baseline metrics."""
    from iccad2026_evaluate import (
        calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area,
    )
    polys, metrics = sample["label"][0], sample["label"][1]
    positions = _golden_rects(sample, n)
    hpwl_b2b = calculate_hpwl_b2b(positions, b2b)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b, pins)
    area = calculate_bbox_area(positions)
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hpwl_b2b = float(metrics[-2])
        if metrics[-1] >= 0:
            hpwl_p2b = float(metrics[-1])
    return {"hpwl_baseline": hpwl_b2b + hpwl_p2b, "area_baseline": area}


def _opt_target_positions(sample, n: int, golden: List[Rect]) -> torch.Tensor:
    """Build the opt_target_pos tensor exactly as ContestEvaluator.evaluate:
    fixed blocks get (w,h); preplaced blocks get (x,y,w,h); else -1."""
    cons = sample["input"][4]
    tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and cons[i, 0] != 0
        is_preplaced = nc > 1 and cons[i, 1] != 0
        gx, gy, gw, gh = golden[i]
        if is_preplaced:
            tpos[i] = torch.tensor([gx, gy, gw, gh])
        elif is_fixed:
            tpos[i, 2] = gw
            tpos[i, 3] = gh
    return tpos


def _edge_lists(b2b, p2b, pins):
    b2b_e = [(int(r[0]), int(r[1]), float(r[2])) for r in b2b.tolist()]
    p2b_e = [(int(r[0]), int(r[1]), float(r[2])) for r in p2b.tolist()]
    pin_l = [(float(r[0]), float(r[1])) for r in pins.tolist()]
    return b2b_e, p2b_e, pin_l


# =============================================================================
# Decoder v0
# =============================================================================
def _centroids(hints: List[Rect]) -> List[Tuple[float, float]]:
    return [(x + w / 2.0, y + h / 2.0) for x, y, w, h in hints]


def decode_shapes(
    hints: List[Rect],
    area_targets: Sequence[float],
    fixed: List[bool],
    preplaced: List[bool],
    mib: List[int],
    target_positions: torch.Tensor,
    n: int,
    shape_hints: Optional[List[Rect]] = None,
    keep_valid_hint_shapes: bool = False,
) -> Tuple[List[float], List[float]]:
    """Step 3: exact-area soft sizing with hint aspect; fixed/preplaced exact;

    Gate: ``shape_hints`` decouples the SHAPE source from the (order-channel)
    hint source. When provided, the per-block aspect used to derive ws/hs is
    read from ``shape_hints`` instead of ``hints`` (default None = unchanged
    behavior, aspect from ``hints``). Orthogonal to ``order_hints`` in
    ``build_order_dags``.
    MIB groups share one aspect (area-weighted mean log-aspect).

    ``keep_valid_hint_shapes`` (faithful-realize mode, default False = unchanged):
    when the hint's OWN (w,h) is already a valid exact-area shape we reproduce it
    verbatim instead of re-deriving ws/hs from a clamped/MIB-shared aspect. A
    block qualifies only when its hint area matches the target within tol, its
    log-aspect is within the clamp, and (for MIB members) the whole group shares
    one identical shape in the hint. This is what makes a production/golden
    round-trip land on identity shapes; blocks that fail any test fall back to
    the exact-area recompute so legality is never sacrificed."""
    ws = [0.0] * n
    hs = [0.0] * n

    # Per-block preferred log-aspect from hint (or shape_hints when given).
    shape_src = shape_hints if shape_hints is not None else hints
    log_a = [0.0] * n
    for i in range(n):
        hw, hh = shape_src[i][2], shape_src[i][3]
        if hw > 1e-9 and hh > 1e-9:
            la = math.log(hw / hh)
        else:
            la = 0.0
        log_a[i] = max(-LOG_ASPECT_CLAMP, min(LOG_ASPECT_CLAMP, la))

    # MIB groups: shared aspect (area-weighted mean of member log-aspects),
    # so members with equal area land on identical (w,h) -> 0 MIB violations.
    mib_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        if mib[i] > 0:
            mib_members[mib[i]].append(i)
    shared_la: Dict[int, float] = {}
    for g, members in mib_members.items():
        num = 0.0
        den = 0.0
        for i in members:
            a = float(area_targets[i])
            num += a * log_a[i]
            den += a
        shared_la[g] = num / den if den > 0 else 0.0

    # Faithful mode: decide per free block whether the hint's own (w,h) is a
    # valid exact-area shape we can keep verbatim (identity round-trip).
    keep = [False] * n
    if keep_valid_hint_shapes:
        # MIB group hint-consistency: all members share one shape within tol.
        mib_group_ok: Dict[int, bool] = {}
        for g, members in mib_members.items():
            w0, h0 = shape_src[members[0]][2], shape_src[members[0]][3]
            mib_group_ok[g] = all(
                abs(shape_src[m][2] - w0) <= 1e-6
                and abs(shape_src[m][3] - h0) <= 1e-6
                for m in members
            )
        for i in range(n):
            if fixed[i] or preplaced[i]:
                continue
            hw, hh = shape_src[i][2], shape_src[i][3]
            if hw <= 1e-9 or hh <= 1e-9:
                continue
            a = float(area_targets[i])
            if abs(hw * hh - a) > 1e-6 * max(1.0, a):
                continue
            # No aspect-clamp gate here: the clamp only bounds the RECOMPUTE
            # path; a hint's own shape is already legal in the (feasible) hint,
            # so reproducing it verbatim is safe at any aspect. Clamping-reject
            # here would force a recompute mismatch on legitimately-thin blocks.
            if mib[i] > 0 and not mib_group_ok.get(mib[i], False):
                continue
            keep[i] = True

    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            ws[i] = float(tw)
            hs[i] = float(th)
            continue
        if keep[i]:
            ws[i] = float(shape_src[i][2])
            hs[i] = float(shape_src[i][3])
            continue
        a = float(area_targets[i])
        la = shared_la[mib[i]] if mib[i] > 0 else log_a[i]
        e = math.exp(la)
        ws[i] = math.sqrt(a * e)
        hs[i] = math.sqrt(a / e)
    return ws, hs


def build_order_dags(
    hints: List[Rect],
    ws: List[float],
    hs: List[float],
    n: int,
    order_hints: Optional[List[Rect]] = None,
    geo_faithful: bool = False,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Steps 2+4 setup: for every pair, pick the SEPARATING AXIS as the one
    with larger normalized centroid separation, direction from the hint. Emit
    a DAG edge on that axis: coord[v] >= coord[u] + dim[u].

    Normalized separation on an axis = |d_axis| / (half_u + half_v) so the axis
    whose overlap is 'more resolved' wins. Deterministic tie-break by index.
    Every pair separated on exactly ONE axis => overlap-free by construction.

    Gate 0 (order-channel upper-bound probe): ``order_hints`` decouples the
    ORDER source from the SHAPE source. Pair CENTROIDS (dx, dy -> which axis
    separates, which side) come from ``order_hints`` when provided, else from
    ``hints`` (unchanged default). The normalized-separation DENOMINATOR
    (half_u + half_v) and the emitted edge GAPS (ws[u]/hs[u]) always come from
    the production ws/hs -- i.e. golden centroids are scored against PRODUCTION
    shapes. Semantic consequence: with golden centroids x production shapes the
    axis-selection can differ from a pure-golden run wherever a pair's golden
    separation is small relative to the production half-extents; the *direction*
    (sign of dx/dy) is still taken from the golden geometry, so the extracted
    pairwise relative order is golden-faithful while the overlap-resolution
    threshold is set by the shapes the compaction will actually realize.
    """
    src = order_hints if order_hints is not None else hints
    cx = [src[i][0] + src[i][2] / 2.0 for i in range(n)]
    cy = [src[i][1] + src[i][3] / 2.0 for i in range(n)]
    x_edges: List[Tuple[int, int, float]] = []
    y_edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[j] - cx[i]
            dy = cy[j] - cy[i]
            hxi, hxj = ws[i] / 2.0, ws[j] / 2.0
            hyi, hyj = hs[i] / 2.0, hs[j] / 2.0
            # Normalized separation per axis (scaled by summed half-extents).
            sep_x = abs(dx) / (hxi + hxj) if (hxi + hxj) > 0 else 0.0
            sep_y = abs(dy) / (hyi + hyj) if (hyi + hyj) > 0 else 0.0
            use_x = sep_x >= sep_y  # tie -> x-axis (deterministic)
            if geo_faithful:
                # Faithful mode: emit the edge on the axis the pair is ACTUALLY
                # separated on in the (order) hint's own boxes, so the hint
                # coordinates already satisfy the edge (=> hint-floored longest
                # path reproduces the hint). Only overlapping (noisy) or
                # corner-separated pairs fall back to the centroid heuristic.
                xov = (min(src[i][0] + src[i][2], src[j][0] + src[j][2])
                       - max(src[i][0], src[j][0]))
                yov = (min(src[i][1] + src[i][3], src[j][1] + src[j][3])
                       - max(src[i][1], src[j][1]))
                xsep = xov <= SEP_TOL
                ysep = yov <= SEP_TOL
                if xsep and not ysep:
                    use_x = True
                elif ysep and not xsep:
                    use_x = False
                # else (both separated = corner, or neither = overlap): keep the
                # centroid-heuristic use_x computed above.
            if use_x:
                if dx > 0 or (dx == 0 and i < j):
                    u, v = i, j          # block i is left of j
                else:
                    u, v = j, i
                x_edges.append((u, v, ws[u]))
            else:
                if dy > 0 or (dy == 0 and i < j):
                    u, v = i, j          # block i is below j
                else:
                    u, v = j, i
                y_edges.append((u, v, hs[u]))
    return x_edges, y_edges


def _topo(n: int, edges: List[Tuple[int, int, float]]) -> Optional[List[int]]:
    from collections import deque
    succ: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    indeg = [0] * n
    for u, v, g in edges:
        succ[u].append((v, g))
        indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v, g in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None


def longest_path_coords(
    n: int,
    edges: List[Tuple[int, int, float]],
    pins: Optional[Dict[int, float]] = None,
    floor: Optional[List[float]] = None,
) -> Optional[List[float]]:
    """Step 4/5: pin-aware longest-path compaction. Separation constraint is
    coord[v] >= coord[u] + gap; a subset of nodes is pinned to EXACT values.

    Without pins this is plain ASAP longest path (overlap-free by construction).
    With pins (preplaced/fixed-position blocks) we run:
      (a) ASAP forward pass with each pinned node floored at its pin,
      (b) an ALAP backward pass that pushes pin-UPSTREAM chains left so a
          predecessor never collides with a pin that sits at a smaller
          coordinate than the free ASAP schedule would place it,
      (c) a final forward realization: pinned -> exact; free -> max over its
          predecessors' realized coords + gap (still >= its ASAP lower bound),
          clamped down toward its ALAP upper bound only when that stays >= the
          predecessor lower bound (feasible because golden is a valid schedule
          for this exact order + these exact pins).
    Returns None on a cycle. Pin coordinates are honored exactly for feasible
    (golden-consistent) orders.
    """
    pins = pins or {}
    order = _topo(n, edges)
    if order is None:
        return None
    pred: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    succ: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for u, v, g in edges:
        pred[v].append((u, g))
        succ[u].append((v, g))

    if floor is not None:
        # Faithful realization (default None => byte-identical ASAP/ALAP below):
        # seed each free node at its hint-coordinate floor and relax forward over
        # the DAG. Still overlap-free by construction (every edge coord[v] >=
        # coord[u] + gap holds after the pass); pins stay exact. When the hint
        # already satisfies every edge (geometry-faithful extraction on a legal
        # hint) this returns the hint coordinates unchanged.
        out = [0.0] * n
        for u in order:
            if u in pins:
                out[u] = pins[u]
                continue
            c = floor[u]
            for p, g in pred[u]:
                c = max(c, out[p] + g)
            out[u] = c
        return out

    INF = float("inf")
    # (a) ASAP lower bounds with pin floors.
    asap = [0.0] * n
    for u in order:
        base = pins[u] if u in pins else 0.0
        c = base
        for p, g in pred[u]:
            c = max(c, asap[p] + g)
        asap[u] = c

    # (b) ALAP upper bounds: pinned nodes force exact; propagate left.
    hi = [INF] * n
    for u in reversed(order):
        h = pins[u] if u in pins else INF
        for v, g in succ[u]:
            hv = hi[v] if hi[v] < INF else asap[v]
            h = min(h, hv - g)
        hi[u] = h

    # (c) Final realization.
    out = [0.0] * n
    for u in order:
        if u in pins:
            out[u] = pins[u]
            continue
        c = 0.0
        for p, g in pred[u]:
            c = max(c, out[p] + g)
        if hi[u] < INF and hi[u] >= c:
            c = hi[u]  # slide toward the pin-consistent ALAP position
        out[u] = c
    return out


def _overlaps_any(rects: List[Rect], i: int, nx: float, ny: float,
                  n: int, tol: float = SEP_TOL) -> bool:
    wi, hi = rects[i][2], rects[i][3]
    for j in range(n):
        if j == i:
            continue
        xj, yj, wj, hj = rects[j]
        ox = max(0.0, min(nx + wi, xj + wj) - max(nx, xj))
        oy = max(0.0, min(ny + hi, yj + hj) - max(ny, yj))
        if ox > tol and oy > tol:
            return True
    return False


def boundary_snap(
    rects: List[Rect],
    boundary: List[int],
    preplaced: List[bool],
    n: int,
) -> List[Rect]:
    """Step 5b: move each boundary-tagged block so its tagged edge touches the
    current bbox wall, greedily and only when it introduces no overlap. Wall
    coordinates are the layout's own min/max (the bbox is defined by the
    realized hull, matching the evaluator). Preplaced blocks are never moved
    (their position is a hard pin). A block that cannot reach its wall without
    an overlap is left in place -- the guard in decode() ensures no regression.
    """
    r = [list(x) for x in rects]
    x0 = min(b[0] for b in r)
    x1 = max(b[0] + b[2] for b in r)
    y0 = min(b[1] for b in r)
    y1 = max(b[1] + b[3] for b in r)
    rt = [tuple(x) for x in r]
    for i in range(n):
        code = boundary[i]
        if code == 0 or preplaced[i]:
            continue
        nx, ny = r[i][0], r[i][1]
        if code & BOUND_LEFT:
            nx = x0
        if code & BOUND_RIGHT:
            nx = x1 - r[i][2]
        if code & BOUND_BOTTOM:
            ny = y0
        if code & BOUND_TOP:
            ny = y1 - r[i][3]
        if not _overlaps_any(rt, i, nx, ny, n):
            r[i][0] = nx
            r[i][1] = ny
            rt[i] = (nx, ny, r[i][2], r[i][3])
    return [tuple(x) for x in r]


def _make_axis_graph(
    n: int,
    edges: List[Tuple[int, int, float]],
    dims: List[float],
    coords: List[float],
) -> AxisGraph:
    g = AxisGraph(n=n)
    g.edges = list(edges)
    for i in range(n):
        g.dim_extent[i] = dims[i]
    lo = min(coords)
    hi = max(coords[i] + dims[i] for i in range(n))
    g.wall_lo = lo
    g.wall_hi = hi
    return g


def _anchors_by_block(
    n: int,
    rects: List[Rect],
    b2b_e, p2b_e, pin_l,
    axis: int,
) -> Dict[int, Tuple[List[float], List[float]]]:
    """Weighted-median anchors for one axis: each block's connected partners'
    centroid coordinate on this axis, with net weights."""
    cen = []
    for x, y, w, h in rects:
        cen.append((x + w / 2.0, y + h / 2.0))
    anchors: Dict[int, Tuple[List[float], List[float]]] = {
        i: ([], []) for i in range(n)
    }
    for i, j, w in b2b_e:
        if i == -1 or not (0 <= i < n and 0 <= j < n):
            continue
        anchors[i][0].append(cen[j][axis]); anchors[i][1].append(w)
        anchors[j][0].append(cen[i][axis]); anchors[j][1].append(w)
    npins = len(pin_l)
    for p, b, w in p2b_e:
        if p == -1 or not (0 <= b < n and 0 <= p < npins):
            continue
        anchors[b][0].append(pin_l[p][axis]); anchors[b][1].append(w)
    return anchors


def _rescale_centroid(hint_c: List[float], skel_c: List[float]) -> List[float]:
    """Linearly map hint centroids onto the skeleton centroid range (per axis)
    so the hint's RELATIVE structure is expressed in the compact (skeleton)
    frame. Without this, a hint frame larger than the ASAP-compact skeleton would
    pull most blocks past the frozen wall and pile them up at the boundary."""
    hlo, hhi = min(hint_c), max(hint_c)
    slo, shi = min(skel_c), max(skel_c)
    hspan = hhi - hlo
    if hspan <= 1e-9:
        return list(skel_c)
    scale = (shi - slo) / hspan
    return [slo + (h - hlo) * scale for h in hint_c]


def _anchored_project(
    skel: List[Rect],
    hints: List[Rect],
    ws: List[float],
    hs: List[float],
    cons,
    tpos,
    at,
    boundary: List[int],
    b2b_e, p2b_e, pin_l,
    n: int,
    hint_weight: float,
    wall_weight: float,
) -> Tuple[List[Rect], Dict[str, object]]:
    """Anchored (compaction-capable) realization. On the ASAP-compact skeleton,
    run the refine weighted-median projected sweep (project_axis) inside the
    refine constraint graph (build_axis_dags: boundary wall-pins + cluster rigid
    groups + preplaced pins + the frozen skeleton bbox as universal walls),
    pulling each block toward BOTH its net partners (HPWL anchors) AND its
    rescaled hint coordinate (weight ``hint_weight``).

    The skeleton fixes a COMPACT bbox (ASAP is the pointwise minimum of the
    geo-faithful order DAG, so bbox <= the hint's); project_axis then redistributes
    blocks toward the hint structure inside that compact frame -- two-sided, unlike
    faithful's monotone-up ratchet. Returns (rects, trace); returns the skeleton
    unchanged on any guard failure (illegal skeleton, project exception, or an
    illegal projected candidate) so it never regresses below the skeleton."""
    tr: Dict[str, object] = {}
    if not refine_guards.hard_legal(skel, at, cons, tpos):
        tr["anchored_applied"] = False
        tr["anchored_reject"] = "skeleton_illegal"
        return skel, tr
    try:
        gx, gy = build_axis_dags(skel, cons, tpos, b2b_e, p2b_e, pin_l)
        xs = [skel[i][0] for i in range(n)]
        ys = [skel[i][1] for i in range(n)]
        ax = _anchors_by_block(n, skel, b2b_e, p2b_e, pin_l, axis=0)
        ay = _anchors_by_block(n, skel, b2b_e, p2b_e, pin_l, axis=1)
        if hint_weight > 0:
            rx = _rescale_centroid(
                [hints[i][0] + ws[i] / 2.0 for i in range(n)],
                [xs[i] + ws[i] / 2.0 for i in range(n)])
            ry = _rescale_centroid(
                [hints[i][1] + hs[i] / 2.0 for i in range(n)],
                [ys[i] + hs[i] / 2.0 for i in range(n)])
            for i in range(n):
                ax[i][0].append(rx[i])
                ax[i][1].append(hint_weight)
                ay[i][0].append(ry[i])
                ay[i][1].append(hint_weight)
        # Explicit wall-target anchors for boundary-tagged blocks. build_axis_dags
        # only wall-PINS blocks already touching the wall, so on a bottom-left
        # ASAP skeleton it cannot CREATE wall adhesion -- we add a high-weight
        # anchor at the (compact) wall coordinate so the projected sweep pulls the
        # tagged edge to the wall wherever slack allows (faithful has no snap; the
        # wall adhesion must be supplied on the hint/anchor side).
        if wall_weight > 0:
            x0 = min(xs)
            x1 = max(xs[i] + ws[i] for i in range(n))
            y0 = min(ys)
            y1 = max(ys[i] + hs[i] for i in range(n))
            for i in range(n):
                code = boundary[i]
                if code == 0:
                    continue
                if code & BOUND_LEFT:
                    ax[i][0].append(x0 + ws[i] / 2.0)
                    ax[i][1].append(wall_weight)
                if code & BOUND_RIGHT:
                    ax[i][0].append(x1 - ws[i] / 2.0)
                    ax[i][1].append(wall_weight)
                if code & BOUND_BOTTOM:
                    ay[i][0].append(y0 + hs[i] / 2.0)
                    ay[i][1].append(wall_weight)
                if code & BOUND_TOP:
                    ay[i][0].append(y1 - hs[i] / 2.0)
                    ay[i][1].append(wall_weight)
        nx, _ = project_axis(gx, xs, ws, ax)
        ny, _ = project_axis(gy, ys, hs, ay)
        cand = [(nx[i], ny[i], ws[i], hs[i]) for i in range(n)]
        if refine_guards.hard_legal(cand, at, cons, tpos):
            tr["anchored_applied"] = True
            return cand, tr
        tr["anchored_applied"] = False
        tr["anchored_reject"] = "not_hard_legal"
    except Exception as e:  # noqa: BLE001
        tr["anchored_applied"] = False
        tr["anchored_error"] = repr(e)
    return skel, tr


def decode(
    hints: List[Rect],
    sample,
    n: int,
    b2b_e, p2b_e, pin_l,
    do_polish: bool = True,
    order_source: str = "self",
    shape_source: str = "self",
    do_repair: bool = False,
    do_refine: bool = False,
    refine_deadline: float = 2.0,
    realize: str = "compact",
    anchor_hint_weight: float = 2.0,
    anchor_wall_weight: float = 8.0,
) -> Tuple[Optional[List[Rect]], Dict[str, object]]:
    """Full order-faithful exact decoder. Returns (rects, per-step trace).
    rects is None only on a hard structural failure (cycle).

    realize: "compact" (default, byte-identical to the legacy path), "faithful",
    or "anchored". Anchored is the compaction-capable soft-anchor realizer: it
    keeps the faithful verified parts (keep-valid-hint-shapes + geo-faithful axis
    extraction) but realizes coordinates TWO-SIDED -- an ASAP longest-path (floor
    0) gives a COMPACT skeleton (restores packing), then a weighted-median
    projected sweep (_anchored_project) pulls each block toward its net partners
    AND its rescaled hint coordinate inside the refine constraint graph
    (boundary wall-pins + cluster groups + frozen skeleton bbox). Unlike faithful
    (a monotone-up ratchet), anchored can compact below the hint. Boundary/
    grouping come from build_axis_dags; pin-order illegality falls back to the
    repair + evict floors (enabled implicitly in anchored). ``anchor_hint_weight``
    sets the hint-anchor weight relative to the unit net-anchor weights (0 =>
    pure HPWL polish inside the compact frame).

    Faithful mode makes the decoder a COORD-RESPECTING realizer for
    already-good hints (production / golden / analytic): it keeps the hint's own
    valid exact-area shapes, extracts the separating axis from the hint geometry,
    and floors the longest-path at the hint coordinates -- so a legal, exact-area
    hint round-trips to itself (measured tax +0.000 on the production cache). It
    also SKIPS boundary_snap / polish / repair / refine (all no-ops or
    net-harmful on a coord-faithful layout) but keeps the final legality
    fallback, so a noisy hint (overlapping / wrong-area) degrades gracefully via
    per-block/per-pair fallbacks + the shelf floor rather than crashing.

    order_source: "self" -> pair order extracted from ``hints`` centroids
    (default, zero behavior change). "golden" -> pair order extracted from the
    golden bbox centroids while shapes/decode/snap/polish all stay on ``hints``
    (Gate 0 order-channel upper-bound probe). See build_order_dags docstring
    for the golden-centroids x production-shapes semantics.

    shape_source: "self" -> ws/hs aspect extracted from ``hints`` (default,
    zero behavior change). "golden" -> ws/hs aspect extracted from the golden
    bbox instead, while order/decode/snap/polish stay driven by ``hints``
    (independent from order_source; both may be set simultaneously).

    ML-realization end (both default OFF => byte-identical to the original
    decode path):
      do_repair -- when the decoded/polished layout is hard-ILLEGAL, rebuild a
        legal layout from the SAME order edges via the v2 pin-order + aspect
        repair (repair_pin_shapes) before resorting to the shelf floor, and
        insert the non-destructive eviction floor ahead of the shelf pack.
        Converts the pin-overlap fallback cases (41-66% on GNN/diffusion hints)
        into real layouts.
      do_refine -- run the production refine stack (slack projection + aspect +
        vsnap, evaluator-faithful guards) on the hard-legal layout, accepted
        only when it does not raise the evaluator cost (realization-faithful
        non-regression). Cuts the boundary/grouping/MIB violation tax (the D1
        mechanism)."""
    at = sample["input"][0][:n]
    cons = sample["input"][4][:n]
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)

    fixed, preplaced, mib, cluster, boundary = _parse_constraints(cons, n)
    area_targets = [float(at[i]) for i in range(n)]

    faithful = realize == "faithful"
    anchored = realize == "anchored"
    trace: Dict[str, object] = {}
    trace["shape_source"] = shape_source
    trace["realize"] = realize

    # Step 3: shapes. Faithful and anchored both keep the hint's valid exact-area
    # shapes (anchored's ASAP skeleton + hint anchoring rely on the same frame).
    shape_hints = golden if shape_source == "golden" else None
    ws, hs = decode_shapes(hints, area_targets, fixed, preplaced, mib, tpos, n,
                           shape_hints=shape_hints,
                           keep_valid_hint_shapes=(faithful or anchored))

    # Step 5a: preplaced blocks are pinned to EXACT golden coordinates and act
    # as immovable anchors INSIDE the compaction (not snapped afterward, which
    # reintroduces overlaps -- see the E0 diagnosis). Fixed-shape blocks keep
    # exact dims (handled in decode_shapes) but are NOT position-pinned.
    x_pins: Dict[int, float] = {}
    y_pins: Dict[int, float] = {}
    for i in range(n):
        if preplaced[i]:
            tx, ty, tw, th = _target(tpos, i)
            if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                x_pins[i] = float(tx)
                y_pins[i] = float(ty)

    # Steps 2+4: order DAGs + pin-aware longest-path compaction. Gate 0:
    # optionally take pair CENTROIDS from golden geometry (order channel) while
    # ws/hs (shapes) stay on the --hints source.
    order_hints = golden if order_source == "golden" else None
    trace["order_source"] = order_source
    x_edges, y_edges = build_order_dags(hints, ws, hs, n,
                                        order_hints=order_hints,
                                        geo_faithful=(faithful or anchored))
    # Faithful mode floors the longest path at the hint coordinates (the order
    # source's own boxes) so a legal hint round-trips to itself; compact AND
    # anchored keep floor=None (plain ASAP) -- anchored wants the compact skeleton
    # as the pointwise-minimum base for its two-sided hint pull.
    order_src = golden if order_source == "golden" else hints
    xfloor = [order_src[i][0] for i in range(n)] if faithful else None
    yfloor = [order_src[i][1] for i in range(n)] if faithful else None
    xc = longest_path_coords(n, x_edges, x_pins, floor=xfloor)
    yc = longest_path_coords(n, y_edges, y_pins, floor=yfloor)
    if xc is None or yc is None:
        trace["fail"] = "cycle"
        return None, trace
    rects = [(xc[i], yc[i], ws[i], hs[i]) for i in range(n)]
    trace["hpwl_after_compact"] = hpwl(rects, b2b_e, p2b_e, pin_l)

    # Step 5b: boundary wall-snap. Move boundary-tagged blocks to their wall
    # where slack allows (greedy, overlap-guarded). Kept only if still hard
    # legal (it always is: snap never creates overlaps by construction, and
    # never touches preplaced/fixed dims). Skipped in faithful mode: on a
    # coord-faithful layout the boundary tags are already satisfied and snapping
    # only trades HPWL (measured +0.027 regression).
    if faithful or anchored:
        # faithful: walls already satisfied (identity). anchored: the projected
        # sweep re-pins boundary blocks to the (compact) walls via build_axis_dags,
        # so a greedy pre-snap would only fight the projection.
        trace["snap_applied"] = False
    else:
        snapped = boundary_snap(rects, boundary, preplaced, n)
        if refine_guards.hard_legal(snapped, at, cons, tpos):
            rects = snapped
            trace["snap_applied"] = True
        else:
            trace["snap_applied"] = False
    trace["hpwl_after_snap"] = hpwl(rects, b2b_e, p2b_e, pin_l)

    # Step 6 (opt-in): weighted-median projection HPWL polish. This is the
    # existing refine.slack_solve.project_axis on a geometry-extracted DAG. It
    # can trade boundary satisfaction for HPWL, so it is default-OFF; enabled
    # only via do_polish and kept only if hard legal (v_rel regression is
    # measured, not blocked, so we can attribute it).
    projected = rects
    if anchored:
        # Skeleton choice: when the hint (with keep-shapes) is ALREADY a legal
        # layout (the SA seed, or a legal move), use it directly so the projection
        # PRESERVES its boundary adhesion + cluster grouping. Re-ASAP-ing a good
        # layout scrambles boundary blocks off their walls (measured: idx95 0->21
        # boundary violations), which the projection only partially restores. Only
        # an overlapping hint needs the ASAP-compact skeleton.
        rects_hint = [(hints[i][0], hints[i][1], ws[i], hs[i]) for i in range(n)]
        if refine_guards.hard_legal(rects_hint, at, cons, tpos):
            skel = rects_hint
            trace["anchored_skeleton"] = "hint"
        else:
            skel = rects
            trace["anchored_skeleton"] = "asap"
        projected, _antr = _anchored_project(
            skel, hints, ws, hs, cons, tpos, at, boundary,
            b2b_e, p2b_e, pin_l, n, anchor_hint_weight, anchor_wall_weight)
        trace.update(_antr)
    elif do_polish and not faithful:
        try:
            gx, gy = build_axis_dags(rects, cons, tpos, b2b_e, p2b_e, pin_l)
            dimx = [rects[i][2] for i in range(n)]
            dimy = [rects[i][3] for i in range(n)]
            ax = _anchors_by_block(n, rects, b2b_e, p2b_e, pin_l, axis=0)
            ay = _anchors_by_block(n, rects, b2b_e, p2b_e, pin_l, axis=1)
            xcoord = [rects[i][0] for i in range(n)]
            ycoord = [rects[i][1] for i in range(n)]
            nx, _ = project_axis(gx, xcoord, dimx, ax)
            ny, _ = project_axis(gy, ycoord, dimy, ay)
            cand = [(nx[i], ny[i], dimx[i], dimy[i]) for i in range(n)]
            if refine_guards.hard_legal(cand, at, cons, tpos):
                projected = cand
                trace["polish_applied"] = True
            else:
                trace["polish_applied"] = False
                trace["polish_reject"] = "not_hard_legal"
        except Exception as e:  # noqa: BLE001
            trace["polish_applied"] = False
            trace["polish_error"] = repr(e)

    # (A) Repair floor (do_repair): when the polished layout is hard-illegal,
    # rebuild a legal layout from the SAME order edges via the v2 pin-order +
    # aspect repair (repair_pin_shapes), then re-snap. This is the primary
    # fallback-conversion lever: the residual overlaps on noisy hints are
    # preplaced-pin penetrations that the order/shape repair resolves, turning a
    # 10.0 shelf case into a real layout. Lazy import avoids the _probe_repair
    # <-> gen_decoder_probe import cycle. Default OFF => block skipped entirely.
    if ((do_repair or anchored) and not faithful
            and not refine_guards.hard_legal(projected, at, cons, tpos)):
        try:
            from _probe_repair import repair_pin_shapes  # noqa: PLC0415
            rep, rtr = repair_pin_shapes(
                hints, x_edges, y_edges, ws, hs, x_pins, y_pins,
                preplaced, fixed, mib, area_targets, n,
                hpwl_fn=hpwl, hpwl_args=(b2b_e, p2b_e, pin_l),
            )
            trace["repair_status"] = rtr.get("status")
            trace["repair_reshaped"] = rtr.get("reshaped")
            if rep is not None:
                rep_snap = boundary_snap(rep, boundary, preplaced, n)
                if refine_guards.hard_legal(rep_snap, at, cons, tpos):
                    rep = rep_snap
                if refine_guards.hard_legal(rep, at, cons, tpos):
                    projected = rep
                    trace["repair_applied"] = True
        except Exception as e:  # noqa: BLE001
            trace["repair_error"] = repr(e)

    # (B) Refine (do_refine): production refine stack (slack projection + aspect
    # + boundary/grouping vsnap) on the legal layout, evaluator-gated. Factored
    # into apply_refine() so the "selected" mode can defer it to the multi-sample
    # winner. In "on" mode do_refine is True here and refine runs per sample.
    if do_refine and not faithful and not anchored:
        projected, _rtr = apply_refine(
            projected, sample, n, b2b_e, p2b_e, pin_l, refine_deadline
        )
        trace.update(_rtr)

    # (C) Final legality guarantee. Noisy hints (GNN/diffusion) can produce a
    # preplaced-pin-vs-order infeasibility that the order-DAG cannot resolve
    # (the hint puts a soft block on the wrong side of a hard pin). Rather than
    # let one overlap cost the whole case the M-penalty (which would mask hint
    # quality), fall back to a guaranteed-legal shelf pack in hint-centroid
    # order that keeps preplaced exact. This is a floor, not the decoder's
    # quality result; trace records whether it fired. With do_repair, the
    # non-destructive eviction floor (park only the still-overlapping soft
    # blocks; keeps the good ~95%) is tried BEFORE the whole-layout shelf pack
    # (v2 ordering: evict scores far better than shelf's flat 10.0). Default OFF
    # => only the original shelf pack runs, byte-identical to before.
    if not refine_guards.hard_legal(projected, at, cons, tpos):
        if do_repair or anchored:
            try:
                from _probe_repair import evict_overlaps  # noqa: PLC0415
                ev, n_ev = evict_overlaps(projected, preplaced, n)
                if refine_guards.hard_legal(ev, at, cons, tpos):
                    projected = ev
                    trace["legal_fallback"] = "evict"
                    trace["evicted"] = n_ev
            except Exception as e:  # noqa: BLE001
                trace["evict_error"] = repr(e)
        if not refine_guards.hard_legal(projected, at, cons, tpos):
            fb = _legal_shelf_fallback(hints, at, cons, tpos, preplaced, fixed,
                                       mib, area_targets, n)
            if fb is not None and refine_guards.hard_legal(fb, at, cons, tpos):
                projected = fb
                trace["legal_fallback"] = True
            else:
                trace["legal_fallback"] = "failed"
    trace["hpwl_final"] = hpwl(projected, b2b_e, p2b_e, pin_l)
    return projected, trace


def apply_refine(
    projected: List[Rect],
    sample,
    n: int,
    b2b_e, p2b_e, pin_l,
    refine_deadline: float,
) -> Tuple[List[Rect], Dict[str, object]]:
    """Evaluator-gated production refine on a hard-legal layout.

    Returns (layout, trace_updates). No-op (returns its input, empty-ish trace)
    when the input is not hard-legal. refine_layout is a pure, guard-gated
    function (returns its input on any doubt); we additionally accept ONLY when
    the official evaluator cost does not increase, so a guard/evaluator mismatch
    can never regress the realized score. Shared by decode()'s "on" path and the
    multi-sample "selected" path (refine only the selected winner)."""
    at = sample["input"][0][:n]
    cons = sample["input"][4][:n]
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)
    tr: Dict[str, object] = {}
    if not refine_guards.hard_legal(projected, at, cons, tpos):
        return projected, tr
    try:
        from floorset_arch.refine.api import refine_layout  # noqa: PLC0415
        b2b_t = sample["input"][1]
        p2b_t = sample["input"][2]
        pins_t = sample["input"][3]
        dl = time.time() + refine_deadline
        refined = refine_layout(
            projected, at, cons, tpos, b2b_t, p2b_t, pins_t,
            deadline=dl, enable_aspect=True,
        )
        if (refined is not projected
                and refine_guards.hard_legal(refined, at, cons, tpos)):
            base_cost = score_case(sample, projected, n)["cost"]
            ref_cost = score_case(sample, refined, n)["cost"]
            tr["refine_cost_delta"] = ref_cost - base_cost
            if ref_cost <= base_cost + 1e-12:
                tr["refine_applied"] = True
                return refined, tr
            tr["refine_applied"] = False
        else:
            tr["refine_applied"] = False
    except Exception as e:  # noqa: BLE001
        tr["refine_error"] = repr(e)
    return projected, tr


def _legal_shelf_fallback(hints, area_targets_t, cons, tpos, preplaced, fixed,
                          mib, area_targets, n) -> Optional[List[Rect]]:
    """Guaranteed-overlap-free shelf pack. Preplaced blocks are placed at their
    exact pins as fixed obstacles; the rest are shelf-packed (row by row) in
    hint-centroid reading order into the half-plane ABOVE the tallest preplaced
    block, so nothing can collide with a pin. Exact-area shapes preserved."""
    ws, hs = decode_shapes(hints, area_targets, fixed, preplaced, mib, tpos, n)
    rects: List[Optional[Rect]] = [None] * n
    top = 0.0
    right_max = 0.0
    for i in range(n):
        if preplaced[i]:
            tx, ty, tw, th = _target(tpos, i)
            rects[i] = (float(tx), float(ty), float(tw), float(th))
            top = max(top, float(ty) + float(th))
            right_max = max(right_max, float(tx) + float(tw))
    # Shelf-pack the rest above `top`, in hint reading order (y then x).
    order = sorted((i for i in range(n) if not preplaced[i]),
                   key=lambda i: (hints[i][1] + hints[i][3] / 2.0,
                                  hints[i][0] + hints[i][2] / 2.0))
    x = 0.0
    y = top
    row_h = 0.0
    width_cap = max(right_max, math.sqrt(sum(area_targets)))
    for i in order:
        w, h = ws[i], hs[i]
        if x > 0.0 and x + w > width_cap:
            x = 0.0
            y += row_h
            row_h = 0.0
        rects[i] = (x, y, w, h)
        x += w
        row_h = max(row_h, h)
    return [r if r is not None else (0.0, 0.0, 1.0, 1.0) for r in rects]


# =============================================================================
# Scoring (official evaluator, no-runtime)
# =============================================================================
def score_case(sample, positions: List[Rect], n: int) -> Dict[str, float]:
    at, b2b, p2b, pins, cons = sample["input"]
    at = at[:n]
    b2b_e_t, p2b_e_t, pin_l = _edge_lists(b2b, p2b, pins)  # noqa: F841
    baseline = _baseline(sample, n, b2b, p2b, pins)
    golden = _golden_rects(sample, n)  # target_positions (all-block golden bbox)
    m = evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        baseline,
        cons,
        b2b,
        p2b,
        pins,
        at,
        golden,
        median_runtime=1.0,
    )
    return {
        "cost": m.cost,
        "feasible": m.is_feasible,
        "hpwl_gap": m.hpwl_gap,
        "area_gap": m.area_gap,
        "v_rel": m.violations_relative,
        "overlap": m.overlap_violations,
        "area_viol": m.area_violations,
        "dim_viol": m.dimension_violations,
        "boundary_v": m.boundary_violations,
        "grouping_v": m.grouping_violations,
        "mib_v": m.mib_violations,
        "n_soft": m.max_possible_violations,
    }


# =============================================================================
# Hint sources
# =============================================================================
def golden_hints(sample, n: int) -> List[Rect]:
    return _golden_rects(sample, n)


# =============================================================================
# Driver
# =============================================================================
def _band(n: int) -> str:
    if n < 60:
        return "n<60"
    if n < 100:
        return "60-99"
    return ">=100"


def run(args) -> None:
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    if args.refine != "off":
        # Enable the boundary/grouping violation snap inside refine_layout (the
        # slack projection + aspect stages run unconditionally). Kept OFF unless
        # --refine is requested, so the default path's env is untouched.
        os.environ.setdefault("FLOORSET_SLACK_REFINE_VSNAP", "1")
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))

    # Pick case indices (stratified subsample if requested).
    all_idx = list(range(len(ds)))
    if args.cases and args.cases < len(all_idx):
        step = max(1, len(all_idx) // args.cases)
        idxs = all_idx[::step][: args.cases]
    else:
        idxs = all_idx

    # Optional hint provider (GNN/diffusion) built once.
    hint_provider = None
    if args.hints == "gnn":
        from _probe_hints import GnnHintProvider  # local helper
        hint_provider = GnnHintProvider(args.gnn_checkpoint)
    elif args.hints == "diffusion":
        from _probe_hints import DiffusionHintProvider
        hint_provider = DiffusionHintProvider(
            args.diffusion_checkpoint,
            use_ema=not args.diffusion_raw,
            steps=args.diffusion_steps,
            n_samples=args.diffusion_samples,
        )
    elif args.hints == "production":
        from _probe_hints import ProductionHintProvider
        hint_provider = ProductionHintProvider(args.production_cache)

    rows = []
    costs = []
    bcs = []
    t0 = time.time()
    for idx in idxs:
        sample = ds[idx]
        n = _n_of(sample)
        b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
        b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)

        infer_ms = 0.0
        if args.hints == "golden":
            hints = golden_hints(sample, n)
        elif args.hints == "production":
            ti = time.time()
            hints = hint_provider.hints(sample, n, idx=idx)
            infer_ms = 1000.0 * (time.time() - ti)
        else:
            ti = time.time()
            hints = hint_provider.hints(sample, n)
            infer_ms = 1000.0 * (time.time() - ti)

        # D1 paired baseline: the production hints ARE a legal layout; score
        # them as-is so baseline vs decoded is a same-layout comparison (no
        # cross-run SA noise).
        base_sc = None
        if args.hints == "production":
            base_sc = score_case(sample, hints, n)

        dt0 = time.time()
        # diffusion: multiple samples -> keep best by --select criterion:
        # "hpwl" (default, legacy proxy) or "cost" (evaluator-faithful
        # score_case cost; E-lever best-of-N selection upgrade).
        if isinstance(hints, list) and len(hints) and isinstance(hints[0], list):
            best = None
            for h in hints:
                # "selected" mode defers refine to the winner: per-sample decode
                # runs repair (if on) but NOT refine, so selection sees the
                # repaired-but-unrefined layout. "on" runs refine per sample.
                pos, tr = decode(h, sample, n, b2b_e, p2b_e, pin_l,
                                 do_polish=not args.no_polish,
                                 order_source=args.order_hints,
                                 shape_source=args.shape_hints,
                                 do_repair=args.repair == "on",
                                 do_refine=args.refine == "on",
                                 refine_deadline=args.refine_deadline,
                                 realize=args.realize,
                                 anchor_hint_weight=args.anchor_hint_weight,
                                 anchor_wall_weight=args.anchor_wall_weight)
                if pos is None:
                    continue
                if args.select == "cost":
                    proxy = score_case(sample, pos, n)["cost"]
                else:
                    proxy = hpwl(pos, b2b_e, p2b_e, pin_l)
                if best is None or proxy < best[2]:
                    best = (pos, tr, proxy)
            pos, tr = (best[0], best[1]) if best else (None, {"fail": "all_cycle"})
            # "selected": refine ONCE on the chosen winner (evaluator-gated, so
            # the final score is min(refined, selected) by construction).
            if args.refine == "selected" and pos is not None:
                pos, _rtr = apply_refine(pos, sample, n, b2b_e, p2b_e, pin_l,
                                         args.refine_deadline)
                tr.update(_rtr)
        else:
            # Single-sample path: "selected" is equivalent to "on".
            pos, tr = decode(hints, sample, n, b2b_e, p2b_e, pin_l,
                             do_polish=not args.no_polish,
                             order_source=args.order_hints,
                             shape_source=args.shape_hints,
                             do_repair=args.repair == "on",
                             do_refine=args.refine in ("on", "selected"),
                             refine_deadline=args.refine_deadline,
                             realize=args.realize,
                             anchor_hint_weight=args.anchor_hint_weight,
                             anchor_wall_weight=args.anchor_wall_weight)
        dec_ms = 1000.0 * (time.time() - dt0)

        if pos is None:
            sc = {"cost": 10.0, "feasible": False, "hpwl_gap": 0, "area_gap": 0,
                  "v_rel": 1.0, "overlap": -1, "area_viol": -1, "dim_viol": -1,
                  "boundary_v": -1, "grouping_v": -1, "mib_v": -1, "n_soft": 0}
        else:
            sc = score_case(sample, pos, n)

        row = {"idx": idx, "n": n, "band": _band(n),
               "dec_ms": round(dec_ms, 1), "infer_ms": round(infer_ms, 1),
               **sc, **{f"tr_{k}": v for k, v in tr.items()}}
        if base_sc is not None:
            row["base_cost"] = base_sc["cost"]
            row["base_hpwl_gap"] = base_sc["hpwl_gap"]
            row["base_area_gap"] = base_sc["area_gap"]
            row["base_v_rel"] = base_sc["v_rel"]
        rows.append(row)
        costs.append(sc["cost"])
        bcs.append(n)
        if args.verbose:
            extra = (f" base={base_sc['cost']:.4f}" if base_sc is not None
                     else "")
            print(f"  idx={idx:3d} n={n:3d} cost={sc['cost']:.4f} "
                  f"feas={int(sc['feasible'])} hpwl_gap={sc['hpwl_gap']:+.3f} "
                  f"area_gap={sc['area_gap']:+.3f} v_rel={sc['v_rel']:.3f} "
                  f"ovl={sc['overlap']} dec_ms={dec_ms:.0f}{extra}",
                  flush=True)

    total = compute_total_score(costs, bcs)
    elapsed = time.time() - t0

    # Aggregates.
    feas = sum(1 for r in rows if r["feasible"])
    band_stats = defaultdict(lambda: {"costs": [], "ns": [], "feas": 0})
    for r in rows:
        b = band_stats[r["band"]]
        b["costs"].append(r["cost"]); b["ns"].append(r["n"])
        b["feas"] += int(r["feasible"])

    fb_fired = sum(1 for r in rows if r.get("tr_legal_fallback") is True)
    feas_rows = [r for r in rows if r["feasible"]]
    feas_mean = (sum(r["cost"] for r in feas_rows) / len(feas_rows)
                 if feas_rows else float("nan"))
    mean_hgap = (sum(r["hpwl_gap"] for r in feas_rows) / len(feas_rows)
                 if feas_rows else float("nan"))
    mean_agap = (sum(r["area_gap"] for r in feas_rows) / len(feas_rows)
                 if feas_rows else float("nan"))
    mean_dec = sum(r["dec_ms"] for r in rows) / max(1, len(rows))
    mean_inf = sum(r["infer_ms"] for r in rows) / max(1, len(rows))

    print("\n" + "=" * 78)
    print(f"HINTS={args.hints}  ORDER={args.order_hints}  "
          f"SHAPES={args.shape_hints}  SELECT={args.select}  "
          f"cases={len(rows)}  polish={not args.no_polish}")
    print(f"WEIGHTED TOTAL (exp(n/12), no-runtime) = {total:.4f}   "
          f"feasible={feas}/{len(rows)}   wall={elapsed:.1f}s")
    print(f"  feasible-only mean cost = {feas_mean:.4f}   "
          f"mean hpwl_gap = {mean_hgap:+.3f}   mean area_gap = {mean_agap:+.3f}")
    print(f"  legal-fallback fired = {fb_fired}/{len(rows)}   "
          f"decode {mean_dec:.0f} ms/case   inference {mean_inf:.0f} ms/case")
    if args.repair == "on" or args.refine != "off":
        rep_applied = sum(1 for r in rows if r.get("tr_repair_applied") is True)
        evicted = sum(1 for r in rows if r.get("tr_legal_fallback") == "evict")
        shelf = sum(1 for r in rows if r.get("tr_legal_fallback") is True)
        ref_applied = sum(1 for r in rows if r.get("tr_refine_applied") is True)
        deltas = [r["tr_refine_cost_delta"] for r in rows
                  if isinstance(r.get("tr_refine_cost_delta"), (int, float))]
        mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
        print(f"  REALIZE: repair->legal = {rep_applied}/{len(rows)}   "
              f"evict floor = {evicted}   shelf floor = {shelf}   "
              f"(fallback conversion = repair+evict = {rep_applied + evicted})")
        print(f"  REALIZE: refine applied = {ref_applied}/{len(rows)}   "
              f"mean refine cost delta (attempted) = {mean_delta:+.4f}")
    if args.hints == "production" and rows and "base_cost" in rows[0]:
        base_costs = [r["base_cost"] for r in rows]
        port_costs = [min(r["cost"], r["base_cost"]) for r in rows]
        base_total = compute_total_score(base_costs, bcs)
        port_total = compute_total_score(port_costs, bcs)
        wins = sum(1 for r in rows if r["cost"] < r["base_cost"] - 1e-9)
        losses = sum(1 for r in rows if r["cost"] > r["base_cost"] + 1e-9)
        print(f"  D1 PAIRED: baseline(production)={base_total:.4f}   "
              f"decoded={total:.4f}   portfolio(min)={port_total:.4f}")
        print(f"  D1 PAIRED: decoder wins {wins} / loses {losses} / "
              f"ties {len(rows) - wins - losses} of {len(rows)}   "
              f"(deploy delta = {port_total - base_total:+.4f})")
    print("-" * 78)
    for band in ("n<60", "60-99", ">=100"):
        if band not in band_stats:
            continue
        st = band_stats[band]
        cs = st["costs"]
        sub_total = compute_total_score(cs, st["ns"])
        print(f"  {band:7s}: cases={len(cs):3d}  feas={st['feas']:3d}  "
              f"mean_cost={sum(cs)/len(cs):.4f}  weighted={sub_total:.4f}  "
              f"max_cost={max(cs):.4f}")
    # Worst 5 by weighted contribution.
    max_n = max(bcs)
    contrib = [(r["cost"] * math.exp((r["n"] - max_n) / 12.0), r) for r in rows]
    contrib.sort(key=lambda t: -t[0])
    print("-" * 78)
    print("  TOP-5 worst (by weighted contribution):")
    for c, r in contrib[:5]:
        print(f"    idx={r['idx']:3d} n={r['n']:3d} cost={r['cost']:.4f} "
              f"w_contrib={c:.4f} feas={int(r['feasible'])} "
              f"hpwl_gap={r['hpwl_gap']:+.3f} area_gap={r['area_gap']:+.3f} "
              f"v_rel={r['v_rel']:.3f} ovl={r['overlap']} "
              f"bnd={r['boundary_v']} grp={r['grouping_v']} mib={r['mib_v']} "
              f"polish={r.get('tr_polish_applied')}")
    print("=" * 78)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"hints": args.hints, "total": total,
                       "feasible": feas, "rows": rows}, f, indent=2, default=str)
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hints",
                    choices=["golden", "gnn", "diffusion", "production"],
                    default="golden")
    ap.add_argument("--order-hints", dest="order_hints",
                    choices=["self", "golden"], default="self",
                    help="Gate 0: source of the PAIR ORDER (centroids). 'self' "
                         "(default, zero behavior change) = order from --hints; "
                         "'golden' = order from golden geometry while shapes / "
                         "decode / snap / polish stay on --hints. Measures the "
                         "order-channel upper bound (golden centroids x "
                         "production shapes).")
    ap.add_argument("--shape-hints", dest="shape_hints",
                    choices=["self", "golden"], default="self",
                    help="Source of the SHAPE (ws/hs aspect). 'self' "
                         "(default, zero behavior change) = shapes from "
                         "--hints; 'golden' = aspect from golden geometry "
                         "while order/decode/snap/polish stay on --hints. "
                         "Orthogonal to --order-hints.")
    ap.add_argument("--select", choices=["hpwl", "cost"], default="hpwl",
                    help="Multi-sample (diffusion) selection criterion: "
                         "'hpwl' (default, legacy proxy) or 'cost' "
                         "(evaluator-faithful score_case cost per sample).")
    ap.add_argument("--repair", choices=["off", "on"], default="off",
                    help="ML-realization end: when a decoded/polished layout is "
                         "hard-illegal, rebuild it legal via the v2 pin-order + "
                         "aspect repair (repair_pin_shapes) and add the "
                         "non-destructive eviction floor before the shelf pack. "
                         "Default off => byte-identical to the legacy path. "
                         "Composes per-sample with multi-sample + --select.")
    ap.add_argument("--refine", choices=["off", "on", "selected"], default="off",
                    help="ML-realization end: run the production refine stack "
                         "(slack projection + aspect + vsnap) on the hard-legal "
                         "layout, accepted only when the evaluator cost does not "
                         "increase. 'on' refines EVERY sample; 'selected' refines "
                         "ONLY the multi-sample winner (chosen by --select on the "
                         "repaired-but-unrefined layout) -- ~1/S the refine cost "
                         "for S samples, evaluator-gated so the final score is "
                         "min(refined, selected). Single-sample: 'selected'=='on'. "
                         "Default off => byte-identical.")
    ap.add_argument("--refine-deadline", dest="refine_deadline", type=float,
                    default=2.0,
                    help="per-case wall-clock budget (s) for the --refine pass")
    ap.add_argument("--anchor-hint-weight", dest="anchor_hint_weight",
                    type=float, default=2.0,
                    help="anchored mode: weight of the (rescaled) hint anchor "
                         "relative to unit net anchors in the projected sweep. "
                         "0 => pure HPWL polish inside the compact frame.")
    ap.add_argument("--anchor-wall-weight", dest="anchor_wall_weight",
                    type=float, default=8.0,
                    help="anchored mode: weight of the wall-target anchor for "
                         "boundary-tagged blocks (0 => no explicit wall pull).")
    ap.add_argument("--realize", choices=["compact", "faithful", "anchored"],
                    default="compact",
                    help="realization mode. 'compact' (default, byte-identical "
                         "to the legacy path) = ASAP longest-path recompaction. "
                         "'faithful' = coord-respecting realizer for already-good "
                         "hints (production / golden / analytic): keeps the "
                         "hint's valid exact-area shapes, extracts the separating "
                         "axis from the hint geometry, floors the longest path at "
                         "the hint coordinates, and skips snap/polish/repair/"
                         "refine so a legal exact-area hint round-trips to itself. "
                         "Noisy hints degrade gracefully via per-block fallbacks "
                         "+ the shelf legality floor.")
    ap.add_argument("--production-cache", default=None,
                    help="JSON cache of production layouts keyed by case idx "
                         "(D1 probe: retests reuse identical layouts)")
    ap.add_argument("--cases", type=int, default=0,
                    help="0 = all 100; else stratified subsample of size N")
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--no-polish", action="store_true",
                    help="skip the slack-refine projection polish (step 6)")
    ap.add_argument("--gnn-checkpoint", default=None)
    ap.add_argument("--diffusion-checkpoint", default=None)
    ap.add_argument("--diffusion-raw", action="store_true")
    ap.add_argument("--diffusion-steps", type=int, default=0,
                    help="0 = model default")
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
