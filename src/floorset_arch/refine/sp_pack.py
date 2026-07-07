"""Exhaustive sequence-pair packer for tiny windows (M5 v3).

For k <= FLOORSET_WINDOW_SP_K (default 5) movable blocks with their INCOMING
shapes (no resize -- exact-area / MIB-shape / dims invariants hold by
construction), enumerate ALL (k!)^2 sequence pairs, derive each packing by
longest-path compaction on both axes, keep the packings that fit the window
box, and return the top-N by anchored window HPWL as local-index -> Rect maps
for `window_repack._repack_window`'s `_consider()` path.

Why this exists: the slicing/strip packers in `window_repack` own the
aspect-FLEX dimension of the search but can only realize slicing tilings; a
sequence pair realizes every overlap-free topology (including pinwheels and
other non-slicing packings). The two generators are complementary and feed
the same strict-better selection, so enabling this stage can only add
candidates, never remove them.

Sequence-pair semantics (Murata et al. 1996): with positive locus Gp and
negative locus Gn, block a is LEFT-OF b iff a precedes b in BOTH Gp and Gn;
a is BELOW b iff a follows b in Gp and precedes b in Gn. Every ordered pair
is x- or y-related exactly once, so the derived packing is overlap-free by
construction.

Per sequence pair we emit up to four justification variants -- ASAP/ALAP per
axis (compact toward the left/right and bottom/top window walls). ALAP is the
per-block latest schedule (blocks with slack slide independently, chains stay
tight), which matters for anchored HPWL when the net pull is toward the far
wall. Everything is vectorized in numpy across the whole (k!)^2 enumeration;
a k=5 window costs single-digit milliseconds. numpy-only, deterministic,
PyInstaller-safe.

Gated by FLOORSET_WINDOW_SP_PACK (default OFF -- promote only on Evaluator
Evidence). Knobs read in `window_repack`: FLOORSET_WINDOW_SP_K (default 5),
FLOORSET_WINDOW_SP_TOPN (default 24). `_ABS_K_CAP` = 6 bounds the numpy
working set (~200 MB at k=6) regardless of the env override; k=6 is opt-in
via FLOORSET_WINDOW_SP_K=6 and should be paired with a runtime measurement.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]

_ABS_K_CAP = 6      # hard cap: (6!)^2 = 518k combos is the numpy memory limit
_FIT_TOL = 1e-6
_DEDUP_DECIMALS = 6
_SCAN_FACTOR = 32   # scan up to 32*top_n sorted combos while deduping


def _longest_path(mask: np.ndarray, dim: np.ndarray) -> np.ndarray:
    """Vectorized DAG longest path. `mask[..., a, b]` means a precedes b
    (coord_b >= coord_a + dim_a). Returns coords (...,k) with sources at 0.
    Bellman-Ford style: k-1 rounds propagate the longest chain."""
    k = dim.shape[0]
    coord = np.zeros(mask.shape[:-2] + (k,), dtype=np.float64)
    for _ in range(max(1, k - 1)):
        reach = coord + dim                      # (..., a)
        cand = np.where(mask, reach[..., :, None], -np.inf)  # (..., a, b)
        coord = np.maximum(coord, cand.max(axis=-2))         # max over a
    return coord


def sp_enumerate(
    shapes: Sequence[Tuple[float, float]],
    win: Tuple[float, float, float, float],
    internal: List[Tuple[int, int, float]],
    external: Dict[int, List[Tuple[float, float, float]]],
    top_n: int = 24,
    max_k: int = 5,
) -> List[Dict[int, Rect]]:
    """Enumerate sequence-pair packings of `shapes` inside `win`.

    Returns up to `top_n` coordinate-deduped candidates as local-index -> Rect
    maps, sorted by anchored window HPWL (same semantics as
    `window_repack._window_hpwl`), best first. Empty list when k is out of
    range or no sequence-pair packing of the UNCHANGED shapes fits the window.
    """
    k = len(shapes)
    if k < 2 or k > min(max_k, _ABS_K_CAP) or top_n < 1:
        return []
    wx0, wy0, wx1, wy1 = win
    win_w = wx1 - wx0
    win_h = wy1 - wy0
    if win_w <= 0.0 or win_h <= 0.0:
        return []

    w = np.asarray([s[0] for s in shapes], dtype=np.float64)
    h = np.asarray([s[1] for s in shapes], dtype=np.float64)
    if (w > win_w + _FIT_TOL).any() or (h > win_h + _FIT_TOL).any():
        return []

    # --- precedence masks over all (k!)^2 sequence pairs -------------------
    perms = np.asarray(list(itertools.permutations(range(k))), dtype=np.int64)
    n_p = perms.shape[0]
    rank = np.empty((n_p, k), dtype=np.int64)
    rank[np.arange(n_p)[:, None], perms] = np.arange(k)[None, :]
    prec = rank[:, :, None] < rank[:, None, :]          # (P, a, b): a before b
    prec_p = prec[:, None, :, :]                        # Gp axis
    prec_q = prec[None, :, :, :]                        # Gn axis
    left = prec_p & prec_q                              # a left-of b
    below = (~prec_p) & prec_q                          # a below b (diag False)

    # --- coordinates: ASAP + per-block ALAP on both axes -------------------
    x_asap = _longest_path(left, w)
    y_asap = _longest_path(below, h)
    span_x = (x_asap + w).max(axis=-1)                  # critical-path extent
    span_y = (y_asap + h).max(axis=-1)
    fit = (span_x <= win_w + _FIT_TOL) & (span_y <= win_h + _FIT_TOL)
    if not fit.any():
        return []

    x_tail = _longest_path(np.swapaxes(left, -1, -2), w)   # widths after a
    y_tail = _longest_path(np.swapaxes(below, -1, -2), h)
    x_alap = win_w - w - x_tail
    y_alap = win_h - h - y_tail

    # --- anchored HPWL per justification variant ----------------------------
    ext_idx: List[int] = []
    ext_x: List[float] = []
    ext_y: List[float] = []
    ext_w: List[float] = []
    for kk, anchors in external.items():
        for ax, ay, aw in anchors:
            ext_idx.append(kk)
            ext_x.append(ax)
            ext_y.append(ay)
            ext_w.append(aw)
    e_idx = np.asarray(ext_idx, dtype=np.int64)
    e_x = np.asarray(ext_x, dtype=np.float64)
    e_y = np.asarray(ext_y, dtype=np.float64)
    e_w = np.asarray(ext_w, dtype=np.float64)

    def _hpwl(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
        cx = bx + w / 2.0
        cy = by + h / 2.0
        total = np.zeros(bx.shape[:-1], dtype=np.float64)
        for a, b, wgt in internal:
            total += wgt * (np.abs(cx[..., a] - cx[..., b])
                            + np.abs(cy[..., a] - cy[..., b]))
        if e_idx.size:
            dx = np.abs(cx[..., e_idx] - e_x)
            dy = np.abs(cy[..., e_idx] - e_y)
            total += ((dx + dy) * e_w).sum(axis=-1)
        return total

    variants: List[Tuple[np.ndarray, np.ndarray]] = [
        (x_asap, y_asap),
        (x_asap, y_alap),
        (x_alap, y_asap),
        (x_alap, y_alap),
    ]
    scores = np.empty((len(variants),) + fit.shape, dtype=np.float64)
    for vi, (vx, vy) in enumerate(variants):
        s = _hpwl(wx0 + vx, wy0 + vy)
        scores[vi] = np.where(fit, s, np.inf)

    # --- rank, dedupe identical geometries, emit top-N ---------------------
    flat = scores.reshape(-1)
    order = np.argsort(flat, kind="stable")
    combos = fit.shape[0] * fit.shape[1]
    out: List[Dict[int, Rect]] = []
    seen: set = set()
    for fi in order[: _SCAN_FACTOR * top_n]:
        if not np.isfinite(flat[fi]):
            break
        vi, rem = divmod(int(fi), combos)
        p, q = divmod(rem, fit.shape[1])
        vx, vy = variants[vi]
        bx = wx0 + vx[p, q]
        by = wy0 + vy[p, q]
        key = tuple(
            (round(float(bx[i]), _DEDUP_DECIMALS),
             round(float(by[i]), _DEDUP_DECIMALS))
            for i in range(k)
        )
        if key in seen:
            continue
        seen.add(key)
        out.append({
            i: (float(bx[i]), float(by[i]), float(w[i]), float(h[i]))
            for i in range(k)
        })
        if len(out) >= top_n:
            break
    return out
