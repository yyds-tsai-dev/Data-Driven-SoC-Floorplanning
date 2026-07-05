"""Windowed re-pack: LNS-style local class escape (M5).

The column-slicing backbone commits every layout to the column-slicing
representation class, whose HPWL ceiling is ~1.35x GT. The slack refiner
(`slack_solve.project_axis`) can only translate blocks inside a FIXED
separation topology; the M2 topology-search stage (`topo_search`) could edit
the DAG but could not *displace* incumbents to realize a genuinely different
packing (measured-killed: 0-2 accepts/case, 91% hard_legal rejects on
re-insertion). This stage takes the honest large-neighborhood destruction/
repair route instead:

  1. Freeze the entire exterior.
  2. Pick a small window (6-10 movable blocks) drawn from 2-3 adjacent columns
     around the highest weighted-net tension.
  3. Carve the window blocks OUT, leaving a clean rectangular hole, and re-pack
     them from scratch with a NON-column packer (randomized bottom-left-fill /
     skyline, plus exhaustive order enumeration for tiny k) that minimizes
     anchored HPWL (internal window nets + external nets pinned to frozen-block
     / pin centroids) and honors the window bbox and exact areas.
  4. Accept the re-pack only if the FULL-layout guard chain passes and
     full-layout HPWL strictly improves. Several windows per case, greedily
     over the tension ranking, until the deadline.

Because the exterior is frozen and the window bbox is a clean hole, a strict
window-local HPWL improvement maps one-to-one onto a full-layout HPWL
improvement (frozen-frozen nets are invariant). Soft blocks have EXACT area
with free aspect (the evaluator only enforces the 1% area tolerance), so the
packer may resize them; MIB members are shape-group-tied and are treated as
fixed-shape (position-only) inside the window; cluster-group members that fit
move as one rigid super-rect (or the group is excluded); LOCKED/preplaced and
boundary blocks off the window's walls are never selected.

Called from `refine/api.py::refine_layout` at the same point as topo_search
(before the slack projection, reusing its `stage_input` fallback), gated by
FLOORSET_WINDOW_REPACK (default OFF) and sharing the `topo_deadline` reserve
carved out in `legalizer/column_backbone.py::solve_with_column_backbone`. Pure
function: on any exception, guard failure, or deadline hit it returns the stage
input rects unchanged. The backbone+refiner 1.2337 baseline is a hard floor.

See docs/design/slack_refiner_spec.md (guard/constraint machinery this reuses)
and topo_search.py (tension ranking, JSONL logging, api.py hookup pattern).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from floorset_arch.legalizer.column_slicing import _parse_constraints, _target

from .guards import hard_legal, soft_violations
from .wirelength import hpwl

Rect = Tuple[float, float, float, float]

EPS = 1e-6
SEP_TOL = 1e-6
AREA_GUARD = 0.0095  # stay strictly under the 1% hard area tolerance

# Boundary bitmask (matches evaluator: 1=left, 2=right, 4=top, 8=bottom).
BOUND_LEFT = 1
BOUND_RIGHT = 2
BOUND_TOP = 4
BOUND_BOTTOM = 8

# Tunables (kept modest -- the per-case budget is only ~3s at n=120 and the
# machine is under load). All overridable via FLOORSET_WINDOW_* env vars.
DEFAULT_TENSION_TOPK = 24         # rank this many highest-tension b2b pairs
DEFAULT_MAX_WINDOWS = 40          # hard cap on windows attempted per stage call
DEFAULT_MAX_K = 9                 # max movable blocks per window
DEFAULT_MIN_K = 2                 # skip windows smaller than this
DEFAULT_INFLATE = 1.6            # bbox inflation factor around the endpoint pair
DEFAULT_RESTARTS = 220            # BLF restarts for medium k
DEFAULT_ENUM_K = 4               # exhaustive order enumeration for k <= this
DEFAULT_SEED = 12345
DEFAULT_ASPECT_CAP = 4.0         # max soft-block aspect (bound the resize)
DEFAULT_MAX_ROUNDS = 3           # re-rank + re-sweep rounds while budget remains


def _envf(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        return int(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------


def _bbox(rects: Sequence[Rect]) -> Tuple[float, float, float, float]:
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return x_min, y_min, x_max, y_max


def _bbox_area(rects: Sequence[Rect]) -> float:
    x0, y0, x1, y1 = _bbox(rects)
    return (x1 - x0) * (y1 - y0)


def _overlaps(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> bool:
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    return ox > SEP_TOL and oy > SEP_TOL


def _boxes_overlap_free(boxes: Sequence[Rect]) -> bool:
    """True iff no two rects in `boxes` overlap (SEP_TOL tolerance). O(k^2),
    but k <= max_k (<= ~9) so this is cheap."""
    m = len(boxes)
    for a in range(m):
        ax0, ay0, aw, ah = boxes[a]
        ax1, ay1 = ax0 + aw, ay0 + ah
        for b in range(a + 1, m):
            bx0, by0, bw, bh = boxes[b]
            if _overlaps(ax0, ay0, ax1, ay1, bx0, by0, bx0 + bw, by0 + bh):
                return False
    return True


# ---------------------------------------------------------------------------
# Per-stage classification context.
# ---------------------------------------------------------------------------


class _Ctx:
    """Immutable per-stage block classification."""

    __slots__ = ("n", "locked", "boundary", "group_of", "group_members",
                 "mib_of", "areas")

    def __init__(self, n, locked, boundary, group_of, group_members, mib_of, areas):
        self.n = n
        self.locked = locked                 # set of preplaced/fixed indices (never move/resize)
        self.boundary = boundary             # index -> boundary bitmask (0 if none)
        self.group_of = group_of             # index -> cluster group id (0 if none)
        self.group_members = group_members   # cluster group id -> [member indices]
        self.mib_of = mib_of                 # index -> mib group id (0 if none)
        self.areas = areas                   # index -> area target (0 if unknown)


def _build_ctx(n: int, area_targets, constraints, target_positions) -> _Ctx:
    fixed, preplaced, mib, cluster, boundary = _parse_constraints(constraints, n)
    locked = set()
    for i in range(n):
        if fixed[i] or preplaced[i]:
            locked.add(i)
    boundary_map = {i: int(boundary[i]) for i in range(n) if boundary[i]}
    group_of = {i: cluster[i] for i in range(n) if cluster[i] > 0}
    group_members: Dict[int, List[int]] = {}
    for i, g in group_of.items():
        group_members.setdefault(g, []).append(i)
    mib_of = {i: mib[i] for i in range(n) if mib[i] > 0}
    areas: Dict[int, float] = {}
    if area_targets is not None:
        for i in range(n):
            if i < len(area_targets):
                a = float(area_targets[i])
                if a > 0:
                    areas[i] = a
    return _Ctx(n, locked, boundary_map, group_of, group_members, mib_of, areas)


# ---------------------------------------------------------------------------
# Tension ranking (reused from topo_search): b2b pairs by w*(|dcx|+|dcy|).
# ---------------------------------------------------------------------------


def _tension_ranked_pairs(
    rects: Sequence[Rect],
    b2b_edges: Sequence[Tuple[float, float, float]],
    topk: int,
) -> List[Tuple[int, int, float]]:
    n = len(rects)
    scored: Dict[Tuple[int, int], float] = {}
    for edge in b2b_edges:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1 or not (0 <= i < n and 0 <= j < n) or w <= 0 or i == j:
            continue
        xi, yi, wi, hi = rects[i]
        xj, yj, wj, hj = rects[j]
        cxi, cyi = xi + wi / 2.0, yi + hi / 2.0
        cxj, cyj = xj + wj / 2.0, yj + hj / 2.0
        tension = w * (abs(cxi - cxj) + abs(cyi - cyj))
        key = (i, j) if i < j else (j, i)
        if tension > scored.get(key, -1.0):
            scored[key] = tension
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [(pair[0], pair[1], t) for pair, t in ordered[:topk]]


# ---------------------------------------------------------------------------
# Window selection.
# ---------------------------------------------------------------------------


def _select_window(
    rects: Sequence[Rect],
    i: int,
    j: int,
    ctx: _Ctx,
    inflate: float,
    max_k: int,
    used: set,
) -> Optional[Tuple[List[int], Tuple[float, float, float, float]]]:
    """Form a window around the endpoint pair (i, j): the bbox of the two
    endpoints inflated by `inflate`, then pick up to `max_k` movable blocks
    whose centers fall inside the inflated box. Returns (member indices, clean
    window bbox) or None if the window is unusable.

    Movable-block filters (a block is REJECTED from the window if):
      - it is LOCKED (fixed/preplaced): its position is immovable.
      - it is boundary-tagged on a wall the window does NOT own: relocating it
        would break the boundary constraint (we cannot guarantee the repack
        keeps it on a wall the window does not span).
      - it belongs to a cluster group not fully contained in the window: rigid
        super-rects are only movable as a whole; a partial group is excluded.

    A block that is NOT selected stays frozen. The returned window bbox is a
    CLEAN HOLE: it is the union bbox of the selected blocks (so every selected
    block starts inside it), and any FROZEN block overlapping that bbox forces
    the window to be rejected (we require a clean rectangular hole so the
    packer has the whole box to work with). This keeps the accepted repack a
    pure local perturbation.
    """
    n = len(rects)
    xi, yi, wi, hi = rects[i]
    xj, yj, wj, hj = rects[j]
    bx0 = min(xi, xj)
    by0 = min(yi, yj)
    bx1 = max(xi + wi, xj + wj)
    by1 = max(yi + hi, yj + hj)
    cx = (bx0 + bx1) / 2.0
    cy = (by0 + by1) / 2.0
    hw = max(bx1 - bx0, 1e-9) * inflate / 2.0
    hh = max(by1 - by0, 1e-9) * inflate / 2.0
    qx0, qy0, qx1, qy1 = cx - hw, cy - hh, cx + hw, cy + hh

    def _boundary_ok(b: int, box) -> bool:
        code = ctx.boundary.get(b, 0)
        if code == 0:
            return True
        # A boundary-tagged block can move only if the window owns the wall it
        # is tagged to AND it is currently on that wall. We conservatively
        # require the block to already touch its tagged wall(s) and the window
        # box to reach that wall so the packer can keep it there.
        bx, by, bw, bh = rects[b]
        wx0, wy0, wx1, wy1 = box
        if code & BOUND_LEFT and not (abs(bx - wx0) <= 1e-3):
            return False
        if code & BOUND_RIGHT and not (abs((bx + bw) - wx1) <= 1e-3):
            return False
        if code & BOUND_BOTTOM and not (abs(by - wy0) <= 1e-3):
            return False
        if code & BOUND_TOP and not (abs((by + bh) - wy1) <= 1e-3):
            return False
        return True

    # Seed: the endpoint pair plus the nearest blocks whose centers fall inside
    # the query box, ordered nearest-first. The seed is then grown to bbox-
    # closure below so the final window bbox is a CLEAN hole by construction.
    cand: List[Tuple[float, int]] = []
    for b in range(n):
        bx, by, bw, bh = rects[b]
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        if qx0 - SEP_TOL <= bcx <= qx1 + SEP_TOL and qy0 - SEP_TOL <= bcy <= qy1 + SEP_TOL:
            d = abs(bcx - cx) + abs(bcy - cy)
            cand.append((d, b))
    cand.sort()
    seed_order = [b for b in (i, j)] + [b for _d, b in cand if b not in (i, j)]

    def _group_of(b: int) -> int:
        return ctx.group_of.get(b, 0)

    def _closure(seed_members: List[int]) -> Optional[Tuple[List[int], Tuple[float, float, float, float]]]:
        """Grow `seed_members` to a bbox-closed, cluster-closed set: repeatedly
        take the member bbox, pull in every OTHER block that geometrically
        overlaps that bbox (so no frozen block is left inside the hole), plus
        every remaining member of any admitted cluster group. Return
        (members, bbox) once a fixpoint under max_k is reached, or None if the
        window cannot be made clean (an admitted block is LOCKED, or a
        boundary block is off a wall the window does not own, or the closure
        exceeds max_k)."""
        members = list(dict.fromkeys(seed_members))  # dedupe, keep order
        for _iter in range(n + 2):
            # cluster closure
            changed = False
            member_set = set(members)
            for b in list(members):
                g = _group_of(b)
                if g:
                    for m in ctx.group_members.get(g, []):
                        if m not in member_set:
                            members.append(m)
                            member_set.add(m)
                            changed = True
            if len(members) > max_k:
                return None
            # bbox closure
            mx0, my0, mx1, my1 = _bbox([rects[b] for b in members])
            for b in range(n):
                if b in member_set:
                    continue
                bx, by, bw, bh = rects[b]
                if _overlaps(mx0, my0, mx1, my1, bx, by, bx + bw, by + bh):
                    members.append(b)
                    member_set.add(b)
                    changed = True
            if len(members) > max_k:
                return None
            if not changed:
                break
        # Validate: no LOCKED block may be inside the hole (it cannot move);
        # boundary blocks must be keepable on their walls within the window.
        box = _bbox([rects[b] for b in members])
        for b in members:
            if b in ctx.locked:
                return None
            if not _boundary_ok(b, box):
                return None
        return members, box

    # Grow the seed incrementally: start from the pair, add nearest candidates
    # one at a time, taking the bbox-closure after each addition; keep the
    # LARGEST clean closure that still fits max_k. This favors a compact
    # contiguous window (2-3 adjacent columns) over a scattered set.
    best_members: Optional[List[int]] = None
    best_box: Optional[Tuple[float, float, float, float]] = None
    current_seed: List[int] = []
    for b in seed_order:
        if b in ctx.locked:
            continue
        trial_seed = current_seed + [b]
        clo = _closure(trial_seed)
        if clo is None:
            continue
        members, box = clo
        current_seed = trial_seed
        if len(members) >= DEFAULT_MIN_K:
            best_members, best_box = members, box
        if len(members) >= max_k:
            break

    if best_members is None or best_box is None:
        return None

    # Skip windows that duplicate an already-improved region.
    key = frozenset(best_members)
    if key in used:
        return None

    return best_members, best_box


# ---------------------------------------------------------------------------
# External anchors for the window blocks (weighted-median target of the
# partner centroids that are FROZEN -- internal partners are handled live).
# ---------------------------------------------------------------------------


def _window_nets(
    selected: List[int],
    rects: Sequence[Rect],
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
) -> Tuple[
    List[Tuple[int, int, float]],
    Dict[int, List[Tuple[float, float, float]]],
]:
    """Split nets touching the window into:
      - internal: (a, b, w) with both a, b in the window (local index space).
      - external: per window-block b -> list of (ax, ay, w) anchor points from
        frozen partners / pins (a fixed target center).
    Returns (internal_edges, external_anchors) both keyed by LOCAL window index.
    """
    n = len(rects)
    local = {b: k for k, b in enumerate(selected)}
    sel_set = set(selected)
    n_pins = len(pins)

    internal: List[Tuple[int, int, float]] = []
    external: Dict[int, List[Tuple[float, float, float]]] = {k: [] for k in range(len(selected))}

    for edge in b2b_edges:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i == -1 or not (0 <= i < n and 0 <= j < n) or w <= 0 or i == j:
            continue
        i_in = i in sel_set
        j_in = j in sel_set
        if i_in and j_in:
            internal.append((local[i], local[j], w))
        elif i_in:
            xj, yj, wj, hj = rects[j]
            external[local[i]].append((xj + wj / 2.0, yj + hj / 2.0, w))
        elif j_in:
            xi, yi, wi, hi = rects[i]
            external[local[j]].append((xi + wi / 2.0, yi + hi / 2.0, w))

    for edge in p2b_edges:
        p, b, w = int(edge[0]), int(edge[1]), float(edge[2])
        if p == -1 or not (0 <= b < n and 0 <= p < n_pins) or w <= 0:
            continue
        if b in sel_set:
            px, py = pins[p]
            if px == -1.0 and py == -1.0:
                continue
            external[local[b]].append((px, py, w))

    return internal, external


def _window_hpwl(
    boxes: List[Rect],
    internal: List[Tuple[int, int, float]],
    external: Dict[int, List[Tuple[float, float, float]]],
) -> float:
    """Anchored HPWL of a candidate window packing (local index space)."""
    total = 0.0
    cx = [b[0] + b[2] / 2.0 for b in boxes]
    cy = [b[1] + b[3] / 2.0 for b in boxes]
    for a, b, w in internal:
        total += w * (abs(cx[a] - cx[b]) + abs(cy[a] - cy[b]))
    for k, anchors in external.items():
        ckx, cky = cx[k], cy[k]
        for ax, ay, w in anchors:
            total += w * (abs(ckx - ax) + abs(cky - ay))
    return total


# ---------------------------------------------------------------------------
# Local repack: shape resolution + skyline bottom-left-fill packer.
# ---------------------------------------------------------------------------


def _resolve_shapes(
    selected: List[int],
    rects: Sequence[Rect],
    ctx: _Ctx,
    aspect_cap: float,
) -> Dict[int, List[Tuple[float, float]]]:
    """For each window block (local index), the list of (w, h) shape options
    the packer may use. Fixed-shape blocks (LOCKED never selected; MIB members;
    RIGID/fixed already excluded via ctx.locked when fixed) get exactly one.
    Soft blocks get a small aspect ladder honoring exact area within the
    guard's tolerance, plus both orientations of the incoming shape."""
    shapes: Dict[int, List[Tuple[float, float]]] = {}
    # MIB group -> the single shared (w, h) literal (from any member; the
    # incoming layout already satisfies MIB shape equality, so all members
    # share one shape). Position may change but shape is group-tied.
    mib_shape: Dict[int, Tuple[float, float]] = {}
    for b in selected:
        g = ctx.mib_of.get(b, 0)
        if g and g not in mib_shape:
            mib_shape[g] = (rects[b][2], rects[b][3])

    for k, b in enumerate(selected):
        bw, bh = rects[b][2], rects[b][3]
        mibg = ctx.mib_of.get(b, 0)
        if mibg:
            # Fixed shape (group-tied). Position only.
            shapes[k] = [mib_shape[mibg]]
            continue
        area = ctx.areas.get(b, bw * bh)
        # Soft block: aspect ladder + orientations, area-exact.
        opts: List[Tuple[float, float]] = []
        seen = set()

        def _add(w, h):
            if w <= 1e-9 or h <= 1e-9:
                return
            ar = max(w / h, h / w)
            if ar > aspect_cap + 1e-9:
                return
            key = (round(w, 6), round(h, 6))
            if key not in seen:
                seen.add(key)
                opts.append((w, h))

        # Incoming shape and its transpose.
        _add(bw, bh)
        _add(bh, bw)
        # Aspect ladder around a square, area-preserving.
        for rho in (0.5, 0.7, 0.85, 1.0, 1.18, 1.4, 2.0):
            w = math.sqrt(area * rho)
            h = area / w
            _add(w, h)
        if not opts:
            opts = [(bw, bh)]
        shapes[k] = opts
    return shapes


def _block_area(k: int, areas_local: Dict[int, float]) -> float:
    return areas_local.get(k, 1.0)


def _place_leaf(
    k: int,
    box: Tuple[float, float, float, float],
    fixed_shape: Dict[int, Tuple[float, float]],
    areas_local: Dict[int, float],
    aspect_cap: float,
) -> Optional[Rect]:
    """Place a single block (local index k) inside its slice `box`.

    Soft blocks flex their aspect to fill the slice as much as possible while
    preserving EXACT area (w = slice_w, h = area / slice_w, clamped to the
    slice height and the aspect cap; the block is bottom-left justified within
    the slice, so a slack gap is legal). Fixed-shape blocks (MIB) keep their
    literal (w, h) and are placed bottom-left inside the slice iff they fit.
    Returns the placed Rect or None if it cannot fit the slice."""
    bx0, by0, bx1, by1 = box
    W = bx1 - bx0
    H = by1 - by0
    if W <= 1e-9 or H <= 1e-9:
        return None
    if k in fixed_shape:
        fw, fh = fixed_shape[k]
        if fw <= W + 1e-6 and fh <= H + 1e-6:
            return (bx0, by0, fw, fh)
        # try transpose
        if fh <= W + 1e-6 and fw <= H + 1e-6:
            return (bx0, by0, fh, fw)
        return None
    area = areas_local.get(k, W * H)
    # Prefer to fill the slice width; derive height from exact area.
    w = W
    h = area / w
    if h > H + 1e-9:
        # Too tall: fill height instead, derive width.
        h = H
        w = area / h
        if w > W + 1e-9:
            # Slice cannot hold this block's area at all.
            return None
    # Aspect cap: if flexing to the slice exceeds the cap, clamp toward a
    # squarer shape (area preserved) and bottom-left justify.
    ar = max(w / h, h / w)
    if ar > aspect_cap:
        if w >= h:
            w = math.sqrt(area * aspect_cap)
            h = area / w
        else:
            h = math.sqrt(area * aspect_cap)
            w = area / h
        if w > W + 1e-6 or h > H + 1e-6:
            return None
    return (bx0, by0, w, h)


def _pack_slicing(
    order: List[int],
    box: Tuple[float, float, float, float],
    cut_seq: List[int],
    depth: int,
    fixed_shape: Dict[int, Tuple[float, float]],
    areas_local: Dict[int, float],
    aspect_cap: float,
    placed: Dict[int, Rect],
) -> bool:
    """Recursive slicing packer: bipartition `order` into two area-balanced
    halves, split `box` proportional to area along the depth's cut direction
    (cut_seq[depth] chooses vertical=0 / horizontal=1), and recurse. Leaves
    flex/place a single block. Slices are disjoint by construction, so the
    result is always overlap-free within the window. Returns True on success
    (fills `placed` local-index -> Rect), False if any leaf cannot fit."""
    if not order:
        return True
    if len(order) == 1:
        rect = _place_leaf(order[0], box, fixed_shape, areas_local, aspect_cap)
        if rect is None:
            return False
        placed[order[0]] = rect
        return True

    total = sum(_block_area(k, areas_local) for k in order)
    if total <= 1e-12:
        return False
    # Balanced split: accumulate until ~half the area.
    half = total / 2.0
    acc = 0.0
    split = 1
    for idx in range(len(order) - 1):
        acc += _block_area(order[idx], areas_local)
        split = idx + 1
        if acc >= half:
            break
    left = order[:split]
    right = order[split:]
    frac = sum(_block_area(k, areas_local) for k in left) / total

    bx0, by0, bx1, by1 = box
    direction = cut_seq[depth % len(cut_seq)] if cut_seq else (depth % 2)
    if direction == 0:
        # vertical cut: left group gets the left frac of the width.
        cut = bx0 + (bx1 - bx0) * frac
        box_l = (bx0, by0, cut, by1)
        box_r = (cut, by0, bx1, by1)
    else:
        # horizontal cut: left group gets the bottom frac of the height.
        cut = by0 + (by1 - by0) * frac
        box_l = (bx0, by0, bx1, cut)
        box_r = (bx0, cut, bx1, by1)

    if not _pack_slicing(left, box_l, cut_seq, depth + 1, fixed_shape,
                         areas_local, aspect_cap, placed):
        return False
    if not _pack_slicing(right, box_r, cut_seq, depth + 1, fixed_shape,
                         areas_local, aspect_cap, placed):
        return False
    return True


def _pack_strip(
    order: List[int],
    box: Tuple[float, float, float, float],
    n_rows: int,
    fixed_shape: Dict[int, Tuple[float, float]],
    areas_local: Dict[int, float],
    horizontal: bool,
) -> Optional[Dict[int, Rect]]:
    """Guaranteed-valid shelf/strip packer with soft-block aspect flex.

    Splits `box` into `n_rows` equal-thickness shelves (rows if horizontal,
    columns if vertical), distributes the ordered blocks across shelves so each
    shelf holds a contiguous run whose total area is ~1/n_rows of the total,
    and within a shelf gives each SOFT block width = area / shelf_thickness (so
    the shelf fills exactly to the blocks' combined area) placed side by side.
    Fixed-shape blocks keep their literal (w, h). Because block areas came from
    this same box, the packed extent per shelf is <= box extent, so the pack
    always fits (returns None only if a fixed-shape block is taller than the
    shelf, which the caller treats as this-order-infeasible). This is the
    fallback that guarantees large-k windows produce at least one valid
    candidate."""
    bx0, by0, bx1, by1 = box
    W = bx1 - bx0
    H = by1 - by0
    if W <= 1e-9 or H <= 1e-9 or n_rows < 1:
        return None
    k = len(order)
    if k == 0:
        return {}
    n_rows = min(n_rows, k)

    total_area = sum(_block_area(kk, areas_local) for kk in order)
    if total_area <= 1e-12:
        return None
    per_row = total_area / n_rows
    # Partition `order` into n_rows contiguous runs by cumulative area.
    rows: List[List[int]] = []
    cur: List[int] = []
    acc = 0.0
    for idx, kk in enumerate(order):
        cur.append(kk)
        acc += _block_area(kk, areas_local)
        remaining_rows = n_rows - len(rows)
        if remaining_rows > 1 and acc >= per_row and (k - idx - 1) >= (remaining_rows - 1):
            rows.append(cur)
            cur = []
            acc = 0.0
    if cur:
        rows.append(cur)
    # Pad to exactly len(rows) shelves (some may be empty if k < n_rows).
    thickness = (H if horizontal else W) / max(1, len(rows))

    placed: Dict[int, Rect] = {}
    for r, row in enumerate(rows):
        if not row:
            continue
        base = (by0 + r * thickness) if horizontal else (bx0 + r * thickness)
        pos = bx0 if horizontal else by0
        for kk in row:
            if kk in fixed_shape:
                fw, fh = fixed_shape[kk]
                if horizontal:
                    if fh > thickness + 1e-6:
                        # transpose or fail
                        if fw <= thickness + 1e-6:
                            fw, fh = fh, fw
                        else:
                            return None
                    placed[kk] = (pos, base, fw, fh)
                    pos += fw
                else:
                    if fw > thickness + 1e-6:
                        if fh <= thickness + 1e-6:
                            fw, fh = fh, fw
                        else:
                            return None
                    placed[kk] = (base, pos, fw, fh)
                    pos += fh
                continue
            area = areas_local.get(kk, thickness * thickness)
            if horizontal:
                h = thickness
                w = area / h
                placed[kk] = (pos, base, w, h)
                pos += w
            else:
                w = thickness
                h = area / w
                placed[kk] = (base, pos, w, h)
                pos += h
    if len(placed) != k:
        return None
    return placed


def _wall_snap(
    boxes: List[Rect],
    win: Tuple[float, float, float, float],
    bound_local: Dict[int, int],
) -> Optional[List[Rect]]:
    """Translate every boundary-tagged window block flush to its owned window
    wall (LEFT->win.x0, RIGHT->win.x1, BOTTOM->win.y0, TOP->win.y1), preserving
    each block's shape. Because `_boundary_ok` only admits a boundary block when
    its tagged wall coincides with the window box edge, snapping to the window
    wall is exactly the evaluator's boundary condition for that block.

    Returns the wall-consistent boxes iff they stay inside the window AND remain
    overlap-free among the k window blocks; otherwise None. When the input is
    ALREADY wall-consistent (every boundary block flush to its wall) the boxes
    are returned unchanged -- an already-flush raw packing is itself a valid
    snapped candidate, so the caller must not discard it. Pure translation --
    never resizes, so exact-area / MIB-shape invariants are untouched."""
    if not bound_local:
        return None
    wx0, wy0, wx1, wy1 = win
    out = list(boxes)
    for k, code in bound_local.items():
        bx, by, bw, bh = out[k]
        nx, ny = bx, by
        if code & BOUND_LEFT:
            nx = wx0
        if code & BOUND_RIGHT:
            nx = wx1 - bw
        if code & BOUND_BOTTOM:
            ny = wy0
        if code & BOUND_TOP:
            ny = wy1 - bh
        # Containment: a snapped block must still fit the window on both axes.
        if nx < wx0 - 1e-6 or ny < wy0 - 1e-6 or nx + bw > wx1 + 1e-6 or ny + bh > wy1 + 1e-6:
            return None
        out[k] = (nx, ny, bw, bh)
    if not _boxes_overlap_free(out):
        return None
    return out


def _repack_window(
    selected: List[int],
    rects: Sequence[Rect],
    win: Tuple[float, float, float, float],
    ctx: _Ctx,
    internal: List[Tuple[int, int, float]],
    external: Dict[int, List[Tuple[float, float, float]]],
    rng: random.Random,
    restarts: int,
    enum_k: int,
    aspect_cap: float,
) -> Optional[Tuple[List[Rect], float, bool]]:
    """Repack the window blocks inside `win`, minimizing anchored HPWL.

    Uses a randomized recursive SLICING packer that flexes soft-block aspects
    to fill the window: a bottom-left-fill packer cannot re-tile a ~100%-full
    window (the incoming column-slicing sub-layout is space-tight), so the real
    geometric freedom comes from re-slicing the box with different block orders,
    cut directions, and per-block aspect flex (exact area preserved). MIB /
    fixed-shape blocks keep their literal shape.

    Boundary-tagged window blocks are kept flush to their owned window wall via
    a `_wall_snap` projection applied to every candidate BEFORE scoring: the
    packer is wall-agnostic, so a raw slicing/strip pack usually drops a
    boundary block off its wall and the full-layout `soft_regress` guard then
    rejects the whole window. Snapping the boundary blocks flush to the window
    wall (a pure translation, overlap-revalidated) recovers those windows.

    Returns (new_boxes_local_order, window_hpwl, used_wall_snap) for the best
    packing found, or None.
    """
    k = len(selected)
    if k < 2:
        return None
    shapes = _resolve_shapes(selected, rects, ctx, aspect_cap)
    # Fixed-shape blocks: single-option in `shapes` (MIB group-tied).
    fixed_shape: Dict[int, Tuple[float, float]] = {}
    areas_local: Dict[int, float] = {}
    for kk in range(k):
        b = selected[kk]
        if len(shapes[kk]) == 1 and ctx.mib_of.get(b, 0):
            fixed_shape[kk] = shapes[kk][0]
        areas_local[kk] = ctx.areas.get(b, rects[b][2] * rects[b][3])

    # Boundary code per LOCAL window index (0 if none). Drives the wall snap
    # so boundary blocks stay glued to their owned window wall after packing.
    bound_local: Dict[int, int] = {}
    for kk in range(k):
        code = ctx.boundary.get(selected[kk], 0)
        if code:
            bound_local[kk] = code
    # Kill-switch for A/B measurement only (default ON). Never set in .env.
    wall_snap_on = os.environ.get("FLOORSET_WINDOW_WALL_SNAP", "1") != "0"

    # Baseline: the incoming window layout -- the packer must strictly beat it.
    base_boxes = [rects[b] for b in selected]
    base_hpwl = _window_hpwl(base_boxes, internal, external)

    # Two separate optima:
    #   best_raw     -- lowest-HPWL packing ignoring boundary walls.
    #   best_snap    -- lowest-HPWL packing with all boundary blocks snapped
    #                   flush to their owned window wall (overlap-revalidated).
    # For a window with boundary blocks, the RAW optimum almost always drops a
    # boundary block off its wall and the full-layout `soft_regress` guard then
    # rejects it -- so we must PREFER the snapped optimum for such windows, even
    # if its window-HPWL is slightly higher, because it is the only candidate
    # that can survive the guard. Non-boundary windows have no snapped variant
    # and fall back to best_raw. Both must strictly beat the incoming baseline.
    best_raw: Optional[List[Rect]] = None
    best_raw_hpwl = base_hpwl
    best_snap: Optional[List[Rect]] = None
    best_snap_hpwl = base_hpwl

    # Anchor-sorted orders (x-pull, y-pull) bias the pack toward the netlist;
    # random restarts explore the slicing space.
    def _anchor_sorted(coord_idx: int) -> List[int]:
        scored: List[Tuple[float, int]] = []
        for kk in range(k):
            anc = external.get(kk, [])
            if anc:
                tw = sum(a[2] for a in anc)
                if tw > 0:
                    m = sum(a[coord_idx] * a[2] for a in anc) / tw
                else:
                    m = rects[selected[kk]][coord_idx]
            else:
                m = rects[selected[kk]][coord_idx]
            scored.append((m, kk))
        scored.sort()
        return [kk for _m, kk in scored]

    base_orders: List[List[int]] = [_anchor_sorted(0), _anchor_sorted(1)]
    if k <= enum_k:
        from itertools import permutations
        base_orders += [list(p) for p in permutations(range(k))]

    # Cut-direction sequences to try per order (alternating, all-vertical,
    # all-horizontal). More sequences -> more distinct tilings.
    cut_seqs: List[List[int]] = [[0, 1], [1, 0], [0], [1]]

    wx0, wy0, wx1, wy1 = win

    def _in_box(boxes: List[Rect]) -> bool:
        for (bx, by, bw, bh) in boxes:
            if (bx < wx0 - 1e-6 or by < wy0 - 1e-6
                    or bx + bw > wx1 + 1e-6 or by + bh > wy1 + 1e-6):
                return False
        return True

    def _consider(placed_map: Optional[Dict[int, Rect]]) -> None:
        nonlocal best_raw, best_raw_hpwl, best_snap, best_snap_hpwl
        if not placed_map or len(placed_map) != k:
            return
        boxes = [placed_map[kk] for kk in range(k)]
        if not _in_box(boxes):
            return
        # Raw optimum (boundary-agnostic).
        h = _window_hpwl(boxes, internal, external)
        if h < best_raw_hpwl - EPS:
            best_raw_hpwl = h
            best_raw = boxes
        # Snapped optimum: glue boundary blocks flush to their owned wall, then
        # re-score. Overlap-revalidated inside _wall_snap.
        if bound_local and wall_snap_on:
            snapped = _wall_snap(boxes, win, bound_local)
            if snapped is not None and _in_box(snapped):
                hs = _window_hpwl(snapped, internal, external)
                if hs < best_snap_hpwl - EPS:
                    best_snap_hpwl = hs
                    best_snap = snapped

    n_restarts = max(1, restarts)
    orders_pool = list(base_orders)
    # Pad with random orders up to the restart budget.
    while len(orders_pool) < n_restarts:
        perm = list(range(k))
        rng.shuffle(perm)
        orders_pool.append(perm)

    # Row counts to try for the guaranteed strip packer (always produces a
    # valid candidate even when the recursive slicer's leaves cannot fit).
    max_rows = min(k, 5)
    row_counts = list(range(1, max_rows + 1))

    for order in orders_pool:
        # Recursive slicing (rich tilings; may fail for some orders).
        for cut_seq in cut_seqs:
            placed: Dict[int, Rect] = {}
            if _pack_slicing(order, win, cut_seq, 0, fixed_shape,
                             areas_local, aspect_cap, placed):
                _consider(placed)
        # Guaranteed strip packer (both orientations, several row counts).
        for nrows in row_counts:
            _consider(_pack_strip(order, win, nrows, fixed_shape,
                                  areas_local, horizontal=True))
            _consider(_pack_strip(order, win, nrows, fixed_shape,
                                  areas_local, horizontal=False))

    # Selection: for a boundary window, return the snapped optimum when one
    # exists (it is the only wall-consistent packing that can pass the soft
    # guard); otherwise the raw optimum. Non-boundary windows have no snapped
    # variant and always take the raw optimum.
    if bound_local and best_snap is not None:
        return list(best_snap), best_snap_hpwl, True
    if best_raw is not None:
        return list(best_raw), best_raw_hpwl, False
    return None


# ---------------------------------------------------------------------------
# Main stage.
# ---------------------------------------------------------------------------


def refine_window(
    rects: List[Rect],
    area_targets,
    constraints,
    target_positions,
    b2b_edges: Sequence[Tuple[float, float, float]],
    p2b_edges: Sequence[Tuple[float, float, float]],
    pins: Sequence[Tuple[float, float]],
    deadline: Optional[float] = None,
    log_path: Optional[str] = None,
) -> Tuple[List[Rect], Dict[str, object]]:
    """Windowed re-pack stage. Returns (out_rects, detail).

    Guarantees: out_rects is either the input with one-or-more accepted window
    re-packs (each of which passed the FULL guard chain and strictly lowered
    full-layout HPWL) or the input rects unchanged. Pure function -- never
    raises.
    """
    stage_start = time.time()
    detail: Dict[str, object] = {
        "stage": "window",
        "pre_hpwl": None,
        "post_hpwl": None,
        "elapsed": 0.0,
        "n_windows_tried": 0,
        "n_windows_accepted": 0,
        "mean_k": 0.0,
        "guard_result": "no_change",
    }
    original = list(rects)

    try:
        n = len(rects)
        if n < 3:
            detail["elapsed"] = time.time() - stage_start
            return original, detail

        topk = _envi("FLOORSET_WINDOW_TENSION_TOPK", DEFAULT_TENSION_TOPK)
        max_windows = _envi("FLOORSET_WINDOW_MAX_WINDOWS", DEFAULT_MAX_WINDOWS)
        max_k = _envi("FLOORSET_WINDOW_MAX_K", DEFAULT_MAX_K)
        inflate = _envf("FLOORSET_WINDOW_INFLATE", DEFAULT_INFLATE)
        restarts = _envi("FLOORSET_WINDOW_RESTARTS", DEFAULT_RESTARTS)
        enum_k = _envi("FLOORSET_WINDOW_ENUM_K", DEFAULT_ENUM_K)
        seed = _envi("FLOORSET_WINDOW_SEED", DEFAULT_SEED)
        aspect_cap = _envf("FLOORSET_WINDOW_ASPECT_CAP", DEFAULT_ASPECT_CAP)
        max_rounds = max(1, _envi("FLOORSET_WINDOW_MAX_ROUNDS", DEFAULT_MAX_ROUNDS))
        rng = random.Random(seed)

        pre_hpwl = hpwl(original, b2b_edges, p2b_edges, pins)
        pre_bbox = _bbox_area(original)
        base_soft = soft_violations(original, constraints)
        detail["pre_hpwl"] = pre_hpwl

        if deadline is not None and time.time() >= deadline:
            detail["guard_result"] = "deadline"
            detail["elapsed"] = time.time() - stage_start
            _log(log_path, detail)
            return original, detail

        ctx = _build_ctx(n, area_targets, constraints, target_positions)

        cur = list(original)
        cur_hpwl = pre_hpwl
        n_tried = 0
        n_accepted = 0
        k_sum = 0
        used_regions: set = set()

        n_rounds = 0
        # Multi-round sweep: each round re-ranks tension on the CURRENT
        # (improved) layout and re-sweeps the pairs. `used_regions` carries
        # across rounds so an already-improved window is never re-attempted; a
        # round that accepts nothing new ends the loop (fixpoint). All rounds
        # share the SAME reserve (the outer deadline is unchanged) -- the stage
        # runs well under budget, so re-ranking harvests the slack instead of
        # growing it. Round 1 alone is byte-identical to the single-pass stage.
        for round_idx in range(max_rounds):
            if n_tried >= max_windows:
                break
            if deadline is not None and time.time() >= deadline:
                break
            round_accepts_before = n_accepted
            n_rounds = round_idx + 1
            pairs = _tension_ranked_pairs(cur, b2b_edges, topk)

            for (i, j, _t) in pairs:
                if n_tried >= max_windows:
                    break
                if deadline is not None and time.time() >= deadline:
                    break
                if not (0 <= i < n and 0 <= j < n) or i == j:
                    continue

                sel = _select_window(cur, i, j, ctx, inflate, max_k, used_regions)
                if sel is None:
                    continue
                selected, win = sel
                n_tried += 1
                k_sum += len(selected)

                internal, external = _window_nets(selected, cur, b2b_edges, p2b_edges, pins)
                repacked = _repack_window(
                    selected, cur, win, ctx, internal, external,
                    rng, restarts, enum_k, aspect_cap,
                )
                win_pre = _window_hpwl([cur[b] for b in selected], internal, external)
                win_rec: Dict[str, object] = {
                    "stage": "window",
                    "round": round_idx,
                    "window_blocks": list(selected),
                    "k": len(selected),
                    "win_pre_hpwl": win_pre,
                    "full_pre_hpwl": cur_hpwl,
                    "elapsed": None,
                    "guard_result": "no_repack",
                }
                if repacked is None:
                    _log(log_path, win_rec)
                    continue

                new_boxes, win_post, used_snap = repacked
                win_rec["win_post_hpwl"] = win_post
                win_rec["wall_snap"] = bool(used_snap)

                # Build the candidate full layout: splice re-packed boxes back in.
                cand = list(cur)
                in_bbox_ok = True
                for k_local, b in enumerate(selected):
                    nb = new_boxes[k_local]
                    # Belt-and-braces: every re-packed block must lie inside the
                    # window bbox (with tolerance) BEFORE the full guard check.
                    if not (nb[0] >= win[0] - 1e-6 and nb[1] >= win[1] - 1e-6
                            and nb[0] + nb[2] <= win[2] + 1e-6
                            and nb[1] + nb[3] <= win[3] + 1e-6):
                        in_bbox_ok = False
                        break
                    cand[b] = nb
                if not in_bbox_ok:
                    win_rec["guard_result"] = "out_of_bbox"
                    _log(log_path, win_rec)
                    continue

                cand_hpwl = hpwl(cand, b2b_edges, p2b_edges, pins)
                if not (cand_hpwl < cur_hpwl - EPS):
                    win_rec["guard_result"] = "hpwl_not_lower"
                    _log(log_path, win_rec)
                    continue
                if not (_bbox_area(cand) <= pre_bbox + 1e-9):
                    win_rec["guard_result"] = "bbox_grew"
                    _log(log_path, win_rec)
                    continue
                cand_soft = soft_violations(cand, constraints)
                if not all(c <= b for c, b in zip(cand_soft, base_soft)):
                    win_rec["guard_result"] = "soft_regress"
                    _log(log_path, win_rec)
                    continue
                if not hard_legal(cand, area_targets, constraints, target_positions):
                    win_rec["guard_result"] = "hard_legal"
                    _log(log_path, win_rec)
                    continue

                # Accept.
                cur = cand
                cur_hpwl = cand_hpwl
                n_accepted += 1
                used_regions.add(frozenset(selected))
                win_rec["guard_result"] = "accepted"
                win_rec["full_post_hpwl"] = cand_hpwl
                _log(log_path, win_rec)

            # Round fixpoint: a full pass over the re-ranked pairs that accepted
            # nothing new means further rounds cannot help (the ranking depends
            # only on `cur`, which is unchanged) -- stop and keep the budget.
            if n_accepted == round_accepts_before:
                break

        detail["n_rounds"] = n_rounds
        detail["n_windows_tried"] = n_tried
        detail["n_windows_accepted"] = n_accepted
        detail["mean_k"] = (k_sum / n_tried) if n_tried else 0.0
        detail["post_hpwl"] = cur_hpwl
        detail["elapsed"] = time.time() - stage_start

        if n_accepted > 0 and cur_hpwl < pre_hpwl - EPS and cur != original:
            # Final defense-in-depth re-check on the survivor.
            dims_note = True  # dims may legitimately change (soft resize)
            post_bbox = _bbox_area(cur)
            post_soft = soft_violations(cur, constraints)
            post_hpwl = hpwl(cur, b2b_edges, p2b_edges, pins)
            if (
                dims_note
                and post_hpwl < pre_hpwl - EPS
                and post_bbox <= pre_bbox + 1e-9
                and all(c <= b for c, b in zip(post_soft, base_soft))
                and hard_legal(cur, area_targets, constraints, target_positions)
            ):
                detail["guard_result"] = "accepted"
                _log(log_path, detail)
                return cur, detail

        detail["post_hpwl"] = pre_hpwl
        detail["guard_result"] = "rejected" if n_tried else "no_change"
        _log(log_path, detail)
        return original, detail

    except Exception:
        detail["guard_result"] = "exception"
        detail["elapsed"] = time.time() - stage_start
        try:
            _log(log_path, detail)
        except Exception:
            pass
        return original, detail


def _log(log_path: Optional[str], detail: Dict[str, object]) -> None:
    if not log_path:
        return
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(detail) + "\n")
    except Exception:
        pass
