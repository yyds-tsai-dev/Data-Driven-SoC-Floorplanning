"""CSF analytical floorplanning prototype probe (arXiv:2504.03796).

Minimal, standalone re-implementation of the CSF pipeline adapted to the
FloorSet ICCAD-2026 objective, used to measure whether a nonsmooth analytical
channel has any headroom against the column-slicing backbone on the n>100 tail.

Pipeline (see docs/design/2026-07-29-csf-analytical-prototype.md):

  A. Global analytical stage -- conjugate subgradient (CSA, paper Alg. 1) on
     f_g = alpha*W + lambda*D + mu*B  (paper Eq. 8), with lambda escalated by
     q=1.3 per outer round (paper Alg. 4) until the total overlap area drops
     below A/v.  Exact-area is enforced BY CONSTRUCTION: shapes (w_i, h_i) are
     frozen with w_i*h_i == a_i, and only block centers move.  MIB is satisfied
     by construction whenever a MIB group has equal area targets (square seed
     shapes -> identical (w,h)).
  B. Legalization -- constraint-graph pair (HCG/VCG) extracted from the
     analytical layout by the paper's Sec. 3.4 rules, then longest-path
     packing.  Overlap-free and exact-area by construction.  This replaces the
     paper's ILA-CG / LA-CSAQ, which we do not need at this stage.

Deliberate deviations from the paper, and why:
  * HPWL here is the EVALUATOR's pairwise edge cost (weighted L1 between block
    centers, plus pin anchors), not the paper's per-net max-min HPWL.  It is
    convex piecewise-linear, so the subgradient is exact and cheap.
  * The paper's I/O pad assignment is not needed: FloorSet pins are given at
    fixed coordinates.
  * The paper has no fixed outline problem to synthesize: FloorSet scores
    Area_bbox_gap against the golden bbox, so W*/H* are DERIVED (see
    `_outline`) from sum(area) * (1 + gamma) and the pin-box aspect ratio.
  * Q-learning step-size scheduling (paper Sec. 3.3 / Alg. 3) is implemented as
    an optional flag; it schedules c only and does not change the model.
  * Optional soft-constraint terms (boundary pull, grouping attraction) are
    added because the paper models none of FloorSet's three soft constraints.

Usage (from the repo root):

    uv run scripts/probes/csf_analytical_probe.py --ids 81 96
    uv run scripts/probes/csf_analytical_probe.py --tail --budget 3.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Import plumbing: the evaluator + loader live in the FloorSet submodule and
# are normally reached through scripts/eval_*.sh.  Make the probe standalone.
# --------------------------------------------------------------------------
def _bootstrap_paths() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "FloorSet" / "iccad2026contest").is_dir():
            root = parent
            break
    else:
        raise RuntimeError("could not locate repo root containing FloorSet/")
    for extra in (root / "FloorSet" / "iccad2026contest", root / "FloorSet", root / "src"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    return root


REPO_ROOT = _bootstrap_paths()

from iccad2026_evaluate import evaluate_solution  # noqa: E402
from litetestLoader import FloorplanDatasetLiteTest  # noqa: E402

Rect = Tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Instance view: dense numpy arrays for the analytical stage.
# --------------------------------------------------------------------------
@dataclass
class AnalyticInstance:
    n: int
    area: np.ndarray                 # (n,) target areas
    w: np.ndarray                    # (n,) frozen widths  (w*h == area)
    h: np.ndarray                    # (n,) frozen heights
    b2b_i: np.ndarray                # (m,) int
    b2b_j: np.ndarray                # (m,) int
    b2b_w: np.ndarray                # (m,) float
    p2b_b: np.ndarray                # (k,) int block index
    p2b_px: np.ndarray               # (k,) pin x
    p2b_py: np.ndarray               # (k,) pin y
    p2b_w: np.ndarray                # (k,) float
    frozen: np.ndarray               # (n,) bool -- preplaced blocks, centers pinned
    frozen_cx: np.ndarray            # (n,) target center x for frozen blocks
    frozen_cy: np.ndarray
    boundary_code: np.ndarray        # (n,) int bitmask 1=L 2=R 4=T 8=B
    cluster_pairs: np.ndarray        # (g,2) int -- same-cluster block pairs
    outline_w: float
    outline_h: float
    target_positions: List[Rect] = field(default_factory=list)


def build_analytic_instance(
    block_count: int,
    area_targets: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
    gamma: float,
) -> AnalyticInstance:
    n = block_count
    area = area_targets[:n].detach().cpu().numpy().astype(np.float64)
    cons = constraints[:n].detach().cpu().numpy().astype(np.float64)
    tpos = target_positions[:n].detach().cpu().numpy().astype(np.float64)

    is_fixed = cons[:, 0] != 0
    is_pre = cons[:, 1] != 0
    mib = cons[:, 2].astype(np.int64)
    clust = cons[:, 3].astype(np.int64)
    bnd = cons[:, 4].astype(np.int64)

    # --- shapes: exact area by construction -------------------------------
    # Soft blocks get a square (w = h = sqrt(a)).  That keeps w*h == a exactly
    # AND makes every MIB group with equal area targets automatically uniform,
    # so V_mib == 0 for free.  Fixed / preplaced blocks keep their spec dims.
    w = np.sqrt(np.maximum(area, 1e-12))
    h = area / np.maximum(w, 1e-12)
    spec = is_fixed | is_pre
    has_spec = spec & (tpos[:, 2] > 0) & (tpos[:, 3] > 0)
    w[has_spec] = tpos[has_spec, 2]
    h[has_spec] = tpos[has_spec, 3]

    # Force MIB groups to a single shape (they may contain non-uniform areas;
    # exact-area is hard and wins, so we only unify when areas already agree).
    for g in range(1, int(mib.max()) + 1 if mib.size else 1):
        idx = np.flatnonzero(mib == g)
        if idx.size < 2:
            continue
        if np.allclose(area[idx], area[idx[0]], rtol=1e-9) and not spec[idx].any():
            side = math.sqrt(float(area[idx[0]]))
            w[idx] = side
            h[idx] = side

    # --- nets -------------------------------------------------------------
    b2b_np = b2b.detach().cpu().numpy()
    mask = (b2b_np[:, 0] >= 0) & (b2b_np[:, 1] >= 0)
    bi = b2b_np[mask, 0].astype(np.int64)
    bj = b2b_np[mask, 1].astype(np.int64)
    bw = b2b_np[mask, 2].astype(np.float64)
    keep = (bi < n) & (bj < n) & (bi != bj)
    bi, bj, bw = bi[keep], bj[keep], bw[keep]

    p2b_np = p2b.detach().cpu().numpy()
    pins_np = pins.detach().cpu().numpy()
    pmask = (p2b_np[:, 0] >= 0) & (p2b_np[:, 1] >= 0)
    pidx = p2b_np[pmask, 0].astype(np.int64)
    pblk = p2b_np[pmask, 1].astype(np.int64)
    pw = p2b_np[pmask, 2].astype(np.float64)
    pkeep = (pblk < n) & (pidx < pins_np.shape[0])
    pidx, pblk, pw = pidx[pkeep], pblk[pkeep], pw[pkeep]
    px = pins_np[pidx, 0].astype(np.float64)
    py = pins_np[pidx, 1].astype(np.float64)
    valid_pin = (px >= 0) & (py >= 0)
    pblk, pw, px, py = pblk[valid_pin], pw[valid_pin], px[valid_pin], py[valid_pin]

    # --- fixed outline (paper Eq. 6, R taken from the pin bounding box) ----
    total_area = float(area.sum())
    finite_pins = pins_np[(pins_np[:, 0] >= 0) & (pins_np[:, 1] >= 0)]
    if finite_pins.shape[0] >= 2:
        pw_span = float(finite_pins[:, 0].max() - finite_pins[:, 0].min())
        ph_span = float(finite_pins[:, 1].max() - finite_pins[:, 1].min())
        ratio = ph_span / max(pw_span, 1e-9) if pw_span > 0 and ph_span > 0 else 1.0
    else:
        ratio = 1.0
    ratio = min(max(ratio, 0.25), 4.0)
    ow = math.sqrt((1.0 + gamma) * total_area / ratio)
    oh = math.sqrt((1.0 + gamma) * total_area * ratio)

    # --- same-cluster pairs (soft grouping attraction) ---------------------
    pairs: List[Tuple[int, int]] = []
    for g in range(1, int(clust.max()) + 1 if clust.size else 1):
        idx = np.flatnonzero(clust == g)
        for a_i in range(idx.size):
            for b_i in range(a_i + 1, idx.size):
                pairs.append((int(idx[a_i]), int(idx[b_i])))
    cluster_pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    frozen = is_pre & (tpos[:, 0] >= 0) & (tpos[:, 1] >= 0)
    return AnalyticInstance(
        n=n, area=area, w=w, h=h,
        b2b_i=bi, b2b_j=bj, b2b_w=bw,
        p2b_b=pblk, p2b_px=px, p2b_py=py, p2b_w=pw,
        frozen=frozen,
        frozen_cx=tpos[:, 0] + 0.5 * tpos[:, 2],
        frozen_cy=tpos[:, 1] + 0.5 * tpos[:, 3],
        boundary_code=bnd,
        cluster_pairs=cluster_pairs,
        outline_w=ow, outline_h=oh,
        target_positions=[tuple(float(v) for v in row) for row in tpos],
    )


# --------------------------------------------------------------------------
# Objective + subgradient (paper Eq. 2-8, adapted).
# --------------------------------------------------------------------------
@dataclass
class Weights:
    alpha: float = 1.0
    lam: float = 20.0
    mu: float = 100.0
    w_boundary: float = 0.0
    w_group: float = 0.0


def objective(
    inst: AnalyticInstance, cx: np.ndarray, cy: np.ndarray, wt: Weights
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Return (f, grad_cx, grad_cy, total_overlap_area).

    All terms are the paper's, with the HPWL swapped for the evaluator's
    pairwise weighted-L1 form.  Everything is a subgradient at kinks (sign(0)
    == 0), matching the paper's "random scheme" only in the degenerate sense
    that we take the zero element of the subdifferential.
    """
    n = inst.n
    gx = np.zeros(n)
    gy = np.zeros(n)

    # --- W: pairwise wirelength (evaluator-faithful) ----------------------
    dxe = cx[inst.b2b_i] - cx[inst.b2b_j]
    dye = cy[inst.b2b_i] - cy[inst.b2b_j]
    wl = float(np.sum(inst.b2b_w * (np.abs(dxe) + np.abs(dye))))
    sx = inst.b2b_w * np.sign(dxe)
    sy = inst.b2b_w * np.sign(dye)
    np.add.at(gx, inst.b2b_i, wt.alpha * sx)
    np.add.at(gx, inst.b2b_j, -wt.alpha * sx)
    np.add.at(gy, inst.b2b_i, wt.alpha * sy)
    np.add.at(gy, inst.b2b_j, -wt.alpha * sy)

    dxp = cx[inst.p2b_b] - inst.p2b_px
    dyp = cy[inst.p2b_b] - inst.p2b_py
    wl += float(np.sum(inst.p2b_w * (np.abs(dxp) + np.abs(dyp))))
    np.add.at(gx, inst.p2b_b, wt.alpha * inst.p2b_w * np.sign(dxp))
    np.add.at(gy, inst.p2b_b, wt.alpha * inst.p2b_w * np.sign(dyp))

    # --- D: pairwise overlap area (paper Eq. 3-5), dense n x n ------------
    ddx = cx[:, None] - cx[None, :]
    ddy = cy[:, None] - cy[None, :]
    hw = 0.5 * (inst.w[:, None] + inst.w[None, :])
    hh = 0.5 * (inst.h[:, None] + inst.h[None, :])
    mw = np.minimum(inst.w[:, None], inst.w[None, :])
    mh = np.minimum(inst.h[:, None], inst.h[None, :])
    raw_x = hw - np.abs(ddx)
    raw_y = hh - np.abs(ddy)
    ox = np.clip(raw_x, 0.0, mw)
    oy = np.clip(raw_y, 0.0, mh)
    np.fill_diagonal(ox, 0.0)
    np.fill_diagonal(oy, 0.0)
    overlap = ox * oy
    total_overlap = 0.5 * float(overlap.sum())

    # d/dcx_i of O_x_ij is -sign(dx) only in the partially-overlapping regime.
    act_x = ((raw_x > 0.0) & (raw_x < mw)).astype(np.float64)
    act_y = ((raw_y > 0.0) & (raw_y < mh)).astype(np.float64)
    np.fill_diagonal(act_x, 0.0)
    np.fill_diagonal(act_y, 0.0)
    gx += wt.lam * np.sum(-np.sign(ddx) * act_x * oy, axis=1)
    gy += wt.lam * np.sum(-np.sign(ddy) * act_y * ox, axis=1)

    # --- B: fixed-outline violation (paper Eq. 7) -------------------------
    lo_x = np.maximum(0.0, 0.5 * inst.w - cx)
    hi_x = np.maximum(0.0, 0.5 * inst.w + cx - inst.outline_w)
    lo_y = np.maximum(0.0, 0.5 * inst.h - cy)
    hi_y = np.maximum(0.0, 0.5 * inst.h + cy - inst.outline_h)
    bviol = float((lo_x + hi_x + lo_y + hi_y).sum())
    gx += wt.mu * (-(lo_x > 0).astype(np.float64) + (hi_x > 0).astype(np.float64))
    gy += wt.mu * (-(lo_y > 0).astype(np.float64) + (hi_y > 0).astype(np.float64))

    # --- optional soft-constraint terms (NOT in the paper) ----------------
    if wt.w_boundary > 0.0:
        code = inst.boundary_code
        for bit, (arr, coord, half, limit) in (
            (1, (gx, cx, 0.5 * inst.w, None)),
            (2, (gx, cx, 0.5 * inst.w, inst.outline_w)),
            (8, (gy, cy, 0.5 * inst.h, None)),
            (4, (gy, cy, 0.5 * inst.h, inst.outline_h)),
        ):
            sel = (code & bit) != 0
            if not sel.any():
                continue
            if limit is None:                       # pull low edge to 0
                resid = coord[sel] - half[sel]
            else:                                   # pull high edge to limit
                resid = limit - (coord[sel] + half[sel])
            arr[sel] += wt.w_boundary * (np.sign(resid) if limit is None
                                         else -np.sign(resid))

    if wt.w_group > 0.0 and inst.cluster_pairs.size:
        gi = inst.cluster_pairs[:, 0]
        gj = inst.cluster_pairs[:, 1]
        gdx = cx[gi] - cx[gj]
        gdy = cy[gi] - cy[gj]
        np.add.at(gx, gi, wt.w_group * np.sign(gdx))
        np.add.at(gx, gj, -wt.w_group * np.sign(gdx))
        np.add.at(gy, gi, wt.w_group * np.sign(gdy))
        np.add.at(gy, gj, -wt.w_group * np.sign(gdy))

    # preplaced blocks never move
    gx[inst.frozen] = 0.0
    gy[inst.frozen] = 0.0

    f = wt.alpha * wl + wt.lam * total_overlap + wt.mu * bviol
    return f, gx, gy, total_overlap


# --------------------------------------------------------------------------
# CSA (paper Algorithm 1) + Q-learning step-size scheduling (Alg. 3).
# --------------------------------------------------------------------------
def csa(
    inst: AnalyticInstance,
    cx: np.ndarray,
    cy: np.ndarray,
    wt: Weights,
    c: float,
    k_max: int,
    deadline: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Conjugate subgradient on u = (cx, cy).  Returns the BEST-f iterate."""
    u = np.concatenate([cx, cy])
    f, gx, gy, ov = objective(inst, cx, cy, wt)
    g = np.concatenate([gx, gy])
    d = np.zeros_like(u)
    best_u, best_f, best_ov = u.copy(), f, ov

    for _ in range(k_max):
        if time.time() > deadline:
            break
        g_prev = g
        f, gx, gy, ov = objective(inst, u[: inst.n], u[inst.n:], wt)
        g = np.concatenate([gx, gy])
        if f < best_f:
            best_u, best_f, best_ov = u.copy(), f, ov
        denom = float(g_prev @ g_prev)
        eta = float(g @ (g - g_prev)) / denom if denom > 1e-18 else 0.0
        d = -g + eta * d
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            break
        u = u + (c / nrm) * d

    return best_u[: inst.n], best_u[inst.n:], best_f, best_ov


def global_floorplan(
    inst: AnalyticInstance,
    cx: np.ndarray,
    cy: np.ndarray,
    wt: Weights,
    c_actions: Sequence[float],
    pop: int,
    k_max: int,
    q: float,
    v: float,
    rounds: int,
    deadline: float,
    rng: np.random.Generator,
    use_qlearn: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Paper Alg. 4 (lambda escalation) around Alg. 3 (population CSAQ).

    The population is `pop` LHS-perturbed copies of the warm-start centers.
    Q-learning (Eq. 10-13) selects the scaling factor c per individual.
    """
    n = inst.n
    m = len(c_actions)
    qtab = np.ones((pop, m))
    pop_cx = np.tile(cx, (pop, 1))
    pop_cy = np.tile(cy, (pop, 1))
    if pop > 1:
        jitter = 0.05 * math.sqrt(inst.outline_w * inst.outline_h)
        pop_cx[1:] += rng.normal(0.0, jitter, size=(pop - 1, n))
        pop_cy[1:] += rng.normal(0.0, jitter, size=(pop - 1, n))
        pop_cx[:, inst.frozen] = inst.frozen_cx[inst.frozen]
        pop_cy[:, inst.frozen] = inst.frozen_cy[inst.frozen]

    total_area = float(inst.area.sum())
    stats = {"rounds": 0.0, "final_overlap": float("inf"), "final_f": float("inf")}
    c_idx = [rng.integers(0, m) for _ in range(pop)]

    for r in range(rounds):
        if time.time() > deadline:
            break
        best_ov = float("inf")
        for p in range(pop):
            if time.time() > deadline:
                break
            f_pre, _, _, _ = objective(inst, pop_cx[p], pop_cy[p], wt)
            c = float(c_actions[c_idx[p]])
            pop_cx[p], pop_cy[p], f_post, ov = csa(
                inst, pop_cx[p], pop_cy[p], wt, c, k_max,
                min(deadline, time.time() + 1e9),
            )
            best_ov = min(best_ov, ov)
            if use_qlearn:
                reward = (f_pre - f_post) / 100.0
                a = c_idx[p]
                qtab[p, a] = 0.6 * qtab[p, a] + 0.4 * (reward + 0.8 * qtab[p].max())
                probs = np.maximum(qtab[p], 1e-6)
                c_idx[p] = int(rng.choice(m, p=probs / probs.sum()))
        stats["rounds"] = float(r + 1)
        stats["final_overlap"] = best_ov
        # Paper Alg. 4 stops here.  We instead stop ESCALATING lambda but keep
        # spending the per-case budget on CSA at the converged lambda, since
        # the contest charges a fixed per-case wall clock either way.
        if best_ov >= total_area / v:
            wt.lam *= q

    fs = [objective(inst, pop_cx[p], pop_cy[p], wt)[0] for p in range(pop)]
    b = int(np.argmin(fs))
    stats["final_f"] = float(fs[b])
    return pop_cx[b], pop_cy[b], stats


# --------------------------------------------------------------------------
# Legalization: CG pair (paper Sec. 3.4 rules) + longest-path packing.
# --------------------------------------------------------------------------
def legalize_cg(inst: AnalyticInstance, cx: np.ndarray, cy: np.ndarray) -> List[Rect]:
    """Overlap-free, exact-area layout from the analytical centers.

    Arcs always point from the smaller lower-left coordinate to the larger
    (index breaks ties), so both graphs are acyclic by construction and the
    longest-path pack terminates.
    """
    n = inst.n
    xl = cx - 0.5 * inst.w
    yl = cy - 0.5 * inst.h

    h_succ: List[List[int]] = [[] for _ in range(n)]
    v_succ: List[List[int]] = [[] for _ in range(n)]
    h_indeg = np.zeros(n, dtype=np.int64)
    v_indeg = np.zeros(n, dtype=np.int64)

    def order_key(i: int, j: int, axis_lo: np.ndarray) -> Tuple[int, int]:
        if axis_lo[i] < axis_lo[j] or (axis_lo[i] == axis_lo[j] and i < j):
            return i, j
        return j, i

    for i in range(n):
        for j in range(i + 1, n):
            if inst.frozen[i] and inst.frozen[j]:
                continue                           # preplaced pair: disjoint by spec
            ox = min(xl[i] + inst.w[i], xl[j] + inst.w[j]) - max(xl[i], xl[j])
            oy = min(yl[i] + inst.h[i], yl[j] + inst.h[j]) - max(yl[i], yl[j])
            use_h = False
            use_v = False
            if ox <= 0.0 and oy <= 0.0:
                use_h = True                       # rule 1: both directions
                use_v = True
            elif ox <= 0.0:
                use_h = True                       # rule 2
            elif oy <= 0.0:
                use_v = True
            else:
                if ox > oy:                        # rule 3: split the smaller
                    use_v = True
                else:
                    use_h = True
            # Preplaced blocks are sources: every incident arc points OUT of
            # them, so nothing can push them off their spec position.  Both
            # graphs stay acyclic (frozen nodes have in-degree 0; the rest are
            # ordered by lower-left coordinate).
            if use_h:
                a, b = order_key(i, j, xl)
                if inst.frozen[b] and not inst.frozen[a]:
                    a, b = b, a
                h_succ[a].append(b)
                h_indeg[b] += 1
            if use_v:
                a, b = order_key(i, j, yl)
                if inst.frozen[b] and not inst.frozen[a]:
                    a, b = b, a
                v_succ[a].append(b)
                v_indeg[b] += 1

    def longest_path(succ: List[List[int]], indeg: np.ndarray, size: np.ndarray,
                     lower: Optional[np.ndarray]) -> np.ndarray:
        pos = np.zeros(n) if lower is None else lower.copy()
        deg = indeg.copy()
        stack = [i for i in range(n) if deg[i] == 0]
        seen = 0
        while stack:
            u = stack.pop()
            seen += 1
            for w_ in succ[u]:
                pos[w_] = max(pos[w_], pos[u] + size[u])
                deg[w_] -= 1
                if deg[w_] == 0:
                    stack.append(w_)
        if seen != n:                              # defensive: cycle -> fall back
            order = np.argsort(np.where(lower is None, 0.0, lower) if lower is not None
                               else np.zeros(n))
            cursor = 0.0
            for i in order:
                pos[i] = cursor
                cursor += size[i]
        return pos

    # Preplaced blocks must land exactly at their spec position: seed the
    # longest-path with their target as a lower bound.  (v1 does not GUARANTEE
    # it -- successors can push them; the probe reports the resulting
    # dimension/position violations honestly.)
    lo_x = np.zeros(n)
    lo_y = np.zeros(n)
    for i in range(n):
        if inst.frozen[i]:
            lo_x[i] = inst.target_positions[i][0]
            lo_y[i] = inst.target_positions[i][1]

    x = longest_path(h_succ, h_indeg, inst.w, lo_x)
    y = longest_path(v_succ, v_indeg, inst.h, lo_y)
    for i in range(n):
        if inst.frozen[i]:
            x[i] = inst.target_positions[i][0]
            y[i] = inst.target_positions[i][1]
    return [(float(x[i]), float(y[i]), float(inst.w[i]), float(inst.h[i]))
            for i in range(n)]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _bbox_area(rects: List[Rect]) -> float:
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    return (x1 - x0) * (y1 - y0)


def solve_case(
    inst: AnalyticInstance,
    args: argparse.Namespace,
    warm: Optional[Tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> Tuple[List[Rect], Dict[str, float], float]:
    start = time.time()
    deadline = start + args.budget

    if warm is not None:
        cx, cy = warm[0].copy(), warm[1].copy()
    else:
        # centroid warm start: pin-weighted, mirrors legalizer/_heuristic_init
        cx = np.full(inst.n, 0.5 * inst.outline_w)
        cy = np.full(inst.n, 0.5 * inst.outline_h)
        acc_x = np.zeros(inst.n)
        acc_y = np.zeros(inst.n)
        acc_w = np.zeros(inst.n)
        np.add.at(acc_x, inst.p2b_b, inst.p2b_w * inst.p2b_px)
        np.add.at(acc_y, inst.p2b_b, inst.p2b_w * inst.p2b_py)
        np.add.at(acc_w, inst.p2b_b, inst.p2b_w)
        has = acc_w > 1e-9
        cx[has] = acc_x[has] / acc_w[has]
        cy[has] = acc_y[has] / acc_w[has]
        cx += rng.normal(0.0, 0.02 * inst.outline_w, size=inst.n)
        cy += rng.normal(0.0, 0.02 * inst.outline_h, size=inst.n)
    cx[inst.frozen] = inst.frozen_cx[inst.frozen]
    cy[inst.frozen] = inst.frozen_cy[inst.frozen]

    wt = Weights(alpha=args.alpha, lam=args.lam, mu=args.mu,
                 w_boundary=args.w_boundary, w_group=args.w_group)
    c_scale = math.sqrt(inst.outline_w * inst.outline_h) / math.sqrt(max(inst.n, 1))
    c_actions = [f * c_scale for f in (0.05, 0.1, 0.2, 0.4, 0.8)]

    cx, cy, stats = global_floorplan(
        inst, cx, cy, wt, c_actions, args.pop, args.kmax, args.q, args.v,
        args.rounds, deadline, rng, use_qlearn=not args.no_qlearn,
    )
    t_global = time.time() - start
    rects = legalize_cg(inst, cx, cy)
    # Paper Fig. 1: re-extracting the CG pair from the packed layout and
    # re-packing compresses the floorplan.  Keep only strict bbox improvements.
    for _ in range(max(0, args.cg_iters - 1)):
        cx2 = np.array([r[0] + 0.5 * r[2] for r in rects])
        cy2 = np.array([r[1] + 0.5 * r[3] for r in rects])
        cand = legalize_cg(inst, cx2, cy2)
        if _bbox_area(cand) < _bbox_area(rects) - 1e-9:
            rects = cand
        else:
            break
    stats["t_global"] = t_global
    stats["t_legal"] = time.time() - start - t_global
    return rects, stats, time.time() - start


def golden_baseline(sample, n: int) -> Tuple[Dict[str, float], List[Rect]]:
    from iccad2026_evaluate import calculate_bbox_area, calculate_hpwl_b2b, calculate_hpwl_p2b

    polys = sample["label"][0]
    positions: List[Rect] = []
    for i in range(n):
        blk = polys[i]
        valid = blk[blk[:, 0] != -1]
        if len(valid) > 0:
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            positions.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
    hb = calculate_hpwl_b2b(positions, b2b)
    hp = calculate_hpwl_p2b(positions, p2b, pins)
    metrics = sample["label"][1]
    if metrics is not None and len(metrics) >= 2 and float(metrics[-2]) > 0:
        hb = float(metrics[-2])
        hp = float(metrics[-1])
    return ({"hpwl_baseline": hb + hp,
             "area_baseline": calculate_bbox_area(positions)}, positions)


def load_column_reference(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(t["test_id"]): t for t in data.get("test_results", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--tail", action="store_true", help="all validation cases with n>100")
    ap.add_argument("--budget", type=float, default=3.5, help="per-case wall clock (s)")
    ap.add_argument("--gamma", type=float, default=0.095, help="fixed-outline whitespace")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--mu", type=float, default=100.0)
    ap.add_argument("--w-boundary", type=float, default=0.0)
    ap.add_argument("--w-group", type=float, default=0.0)
    ap.add_argument("--pop", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--cg-iters", type=int, default=3)
    ap.add_argument("--q", type=float, default=1.3)
    ap.add_argument("--v", type=float, default=100.0)
    ap.add_argument("--no-qlearn", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-path", type=str,
                    default=str(REPO_ROOT / "FloorSet" / "iccad2026contest" / ".."))
    ap.add_argument("--reference", type=str,
                    default=str(REPO_ROOT / "artifacts" / "partner_eval"
                                / "budget35_dm2_df650k_s10.json"))
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    ds = FloorplanDatasetLiteTest(args.data_path)
    ref = load_column_reference(Path(args.reference))

    ids = args.ids
    if args.tail or not ids:
        ids = []
        for idx in range(len(ds)):
            n = int((ds[idx]["input"][0] != -1).sum().item())
            if n > 100:
                ids.append(idx)
        if args.ids:
            ids = [i for i in ids if i in set(args.ids)]

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"{'id':>4} {'n':>4} {'CSF':>8} {'COL':>8} {'delta':>8} "
          f"{'hpwlg':>8} {'areag':>8} {'Vrel':>7} {'feas':>5} {'ovl':>5} "
          f"{'dim':>4} {'tg':>5} {'tl':>5}")
    for idx in ids:
        sample = ds[idx]
        area_t = sample["input"][0]
        n = int((area_t != -1).sum().item())
        b2b, p2b, pins, cons = (sample["input"][1], sample["input"][2],
                                sample["input"][3], sample["input"][4])
        base, gold_positions = golden_baseline(sample, n)
        tpos = torch.tensor(gold_positions, dtype=torch.float32)

        inst = build_analytic_instance(n, area_t, b2b, p2b, pins, cons, tpos, args.gamma)
        rects, stats, elapsed = solve_case(inst, args, None, rng)

        metrics = evaluate_solution(
            {"positions": rects, "runtime": 1.0}, base, cons[:n], b2b, p2b, pins,
            area_t[:n], target_positions=gold_positions, median_runtime=1.0,
        )
        cost_nr = metrics.cost if metrics.is_feasible else 10.0
        col = ref.get(idx)
        col_cost = col["cost_no_runtime"] if col else float("nan")
        print(f"{idx:4d} {n:4d} {cost_nr:8.4f} {col_cost:8.4f} "
              f"{cost_nr - col_cost:+8.4f} {metrics.hpwl_gap:8.4f} "
              f"{metrics.area_gap:8.4f} {metrics.violations_relative:7.4f} "
              f"{str(metrics.is_feasible):>5} {metrics.overlap_violations:5d} "
              f"{metrics.dimension_violations:4d} "
              f"{stats['t_global']:5.2f} {stats['t_legal']:5.2f}")
        rows.append({
            "test_id": idx, "block_count": n, "cost_no_runtime": cost_nr,
            "column_cost_no_runtime": col_cost,
            "hpwl_gap": metrics.hpwl_gap, "area_gap": metrics.area_gap,
            "violations_relative": metrics.violations_relative,
            "is_feasible": metrics.is_feasible,
            "overlap_violations": metrics.overlap_violations,
            "area_violations": metrics.area_violations,
            "dimension_violations": metrics.dimension_violations,
            "boundary_violations": metrics.boundary_violations,
            "grouping_violations": metrics.grouping_violations,
            "mib_violations": metrics.mib_violations,
            "runtime": elapsed, **{k: float(v) for k, v in stats.items()},
        })

    if rows:
        wgt = [math.exp(r["block_count"] / 12.0) for r in rows]
        tot = sum(wgt)
        csf = sum(w * r["cost_no_runtime"] for w, r in zip(wgt, rows)) / tot
        have_ref = all(not math.isnan(r["column_cost_no_runtime"]) for r in rows)
        colw = (sum(w * r["column_cost_no_runtime"] for w, r in zip(wgt, rows)) / tot
                if have_ref else float("nan"))
        print(f"\nsubset weighted no-runtime cost:  CSF {csf:.4f}   "
              f"COLUMN {colw:.4f}   delta {csf - colw:+.4f}")
        print("GATE: continue only if delta <= -0.0100 (weighted, n>100 subset)")
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
