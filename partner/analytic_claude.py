"""Analytical global polish: simultaneous gradient optimization of every
movable block's position (and soft blocks' shapes).

The column SA moves one unit at a time and the refiner applies local
operations; nothing in the pipeline ever moves all blocks in a coordinated
direction.  This stage does: a differentiable objective (smoothed HPWL +
pairwise overlap + soft bounding-box area + wall pull for boundary-coded
blocks) is minimized with Adam over all coordinates jointly, then the
result is re-legalized by the existing min-displacement machinery and
accepted only if the true cost improves.

Fully instance-generic (constraint codes and connectivity only, no case
logic).  CPU torch; ~0.2-0.4 s for n<=120 at 300 steps.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

Rect = Tuple[float, float, float, float]


def _edges_from_b2b(b2b: torch.Tensor, n: int):
    """(i, j, w) arrays from the padded b2b connectivity tensor."""
    if b2b is None or b2b.numel() == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.float64))
    eb = b2b.detach().cpu().numpy()
    ii, jj, ww = [], [], []
    for row in eb:
        i, j, w = int(row[0]), int(row[1]), float(row[2])
        if i < 0 or j < 0 or i >= n or j >= n or w <= 0:
            continue
        ii.append(i)
        jj.append(j)
        ww.append(w)
    return (np.asarray(ii, np.int64), np.asarray(jj, np.int64),
            np.asarray(ww, np.float64))


def _pin_edges(p2b: torch.Tensor, pins: torch.Tensor, n: int):
    """(block, pin_xy, w) arrays from the p2b connectivity tensor."""
    if p2b is None or p2b.numel() == 0 or pins is None:
        return (np.zeros(0, np.int64), np.zeros((0, 2), np.float64),
                np.zeros(0, np.float64))
    ep = p2b.detach().cpu().numpy()
    pn = pins.detach().cpu().numpy()
    bb, xy, ww = [], [], []
    for row in ep:
        p, b, w = int(row[0]), int(row[1]), float(row[2])
        if p < 0 or b < 0 or b >= n or p >= len(pn) or w <= 0:
            continue
        px, py = float(pn[p, 0]), float(pn[p, 1])
        if px == -1.0 and py == -1.0:
            continue
        bb.append(b)
        xy.append((px, py))
        ww.append(w)
    return (np.asarray(bb, np.int64),
            np.asarray(xy, np.float64).reshape(-1, 2),
            np.asarray(ww, np.float64))


def analytic_polish(opt, P0: np.ndarray, b2b, p2b, pins,
                    deadline: float,
                    steps: int = 120,
                    shape_opt: bool = False,
                    trust: float = 0.04,
                    seed: int = 0) -> Optional[np.ndarray]:
    """Return a globally-polished (still overlapping) layout, or None.

    opt: a _ColumnOptimizer built on the same instance (kind/areas/
    boundary/hp_ref/area_ref are read from it).  P0: [n,4] (x,y,w,h).
    The output is meant to be fed into refine_prediction for
    re-legalization; acceptance stays with the caller.
    """
    n = len(P0)
    kind = np.asarray(opt.kind, np.int64)
    areas = np.asarray(opt.areas, np.float64)
    movable = kind != 2

    ei, ej, ew = _edges_from_b2b(b2b, n)
    pb, pxy, pw = _pin_edges(p2b, pins, n)
    dev = torch.device("cpu")
    T = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=dev)

    x = T(P0[:, 0]).clone()
    y = T(P0[:, 1]).clone()
    w0 = T(P0[:, 2]).clone()
    h0 = T(P0[:, 3]).clone()
    A = T(areas)
    mv = T(movable, torch.bool)
    soft = T((kind == 0) & movable, torch.bool)

    x.requires_grad_(True)
    y.requires_grad_(True)
    # soft-block shape via log-width (h = area / w stays exact)
    logw = torch.log(torch.clamp(w0, min=1e-6)).clone()
    logw.requires_grad_(bool(shape_opt))

    eiT, ejT, ewT = T(ei, torch.long), T(ej, torch.long), T(ew)
    pbT, pwT = T(pb, torch.long), T(pw)
    pxT, pyT = T(pxy[:, 0]), T(pxy[:, 1])

    bnd_idx = [int(i) for i in opt._bnd_idx]
    bnd_code = [int(c) for c in opt._bnd_codes]
    bl = T([i for i, c in zip(bnd_idx, bnd_code) if c & 1], torch.long)
    br = T([i for i, c in zip(bnd_idx, bnd_code) if c & 2], torch.long)
    bt = T([i for i, c in zip(bnd_idx, bnd_code) if c & 4], torch.long)
    bb_ = T([i for i, c in zip(bnd_idx, bnd_code) if c & 8], torch.long)

    hp_ref = float(getattr(opt, "hp_ref", 0.0) or 0.0)
    if hp_ref <= 1e-5:
        try:
            hp_ref = float(opt._hpwl(np.asarray(P0, np.float64)))
        except Exception:
            hp_ref = 0.0
    hp_ref = max(hp_ref, 1e-6)
    area_ref = max(float(getattr(opt, "area_ref", 0.0) or 0.0), 1e-6)
    scale = math.sqrt(float(A.sum()))
    beta = 24.0 / scale        # softmax sharpness for the soft bbox

    params = [x, y] + ([logw] if shape_opt else [])
    opt_t = torch.optim.Adam(params, lr=0.0015 * scale)
    x_fix = T(P0[:, 0])
    y_fix = T(P0[:, 1])
    r_tr = trust * scale  # projected trust region around the incumbent

    def sizes():
        if shape_opt:
            w = torch.where(soft, torch.exp(logw), w0)
            # aspect clamp for soft blocks
            s_min = torch.sqrt(A / 10.0)
            s_max = torch.sqrt(A * 10.0)
            w = torch.where(soft, torch.clamp(w, s_min, s_max), w)
            h = torch.where(soft, A / torch.clamp(w, min=1e-6), h0)
            return w, h
        return w0, h0

    last = None
    for it in range(steps):
        if (it & 15) == 0 and time.time() >= deadline:
            break
        opt_t.zero_grad()
        w, h = sizes()
        xe = torch.where(mv, x, x_fix)
        ye = torch.where(mv, y, y_fix)
        cx = xe + 0.5 * w
        cy = ye + 0.5 * h

        loss = torch.zeros((), dtype=torch.float64)
        if len(eiT):
            dx = torch.abs(cx[eiT] - cx[ejT])
            dy = torch.abs(cy[eiT] - cy[ejT])
            loss = loss + (ewT * (dx + dy)).sum() / hp_ref
        if len(pbT):
            loss = loss + (pwT * (torch.abs(cx[pbT] - pxT)
                                  + torch.abs(cy[pbT] - pyT))).sum() / hp_ref

        # soft bounding box (log-sum-exp extremes)
        x0e, y0e = xe, ye
        x1e, y1e = xe + w, ye + h
        Xmin = -torch.logsumexp(-beta * x0e, 0) / beta
        Xmax = torch.logsumexp(beta * x1e, 0) / beta
        Ymin = -torch.logsumexp(-beta * y0e, 0) / beta
        Ymax = torch.logsumexp(beta * y1e, 0) / beta
        loss = loss + 0.9 * (Xmax - Xmin) * (Ymax - Ymin) / area_ref

        # boundary-coded blocks: pull flush to the soft walls
        wall = torch.zeros((), dtype=torch.float64)
        if len(bl):
            wall = wall + ((x0e[bl] - Xmin) ** 2).sum()
        if len(br):
            wall = wall + ((Xmax - x1e[br]) ** 2).sum()
        if len(bt):
            wall = wall + ((Ymax - y1e[bt]) ** 2).sum()
        if len(bb_):
            wall = wall + ((y0e[bb_] - Ymin) ** 2).sum()
        loss = loss + 4.0 * wall / (scale * scale)

        # pairwise overlap, ramped up over the schedule
        ox = (torch.minimum(x1e[:, None], x1e[None, :])
              - torch.maximum(x0e[:, None], x0e[None, :])).clamp(min=0.0)
        oy = (torch.minimum(y1e[:, None], y1e[None, :])
              - torch.maximum(y0e[:, None], y0e[None, :])).clamp(min=0.0)
        ov = ox * oy
        ov = ov - torch.diag(torch.diag(ov))
        ramp = 1.0 + 9.0 * (it / max(steps - 1, 1)) ** 2
        loss = loss + ramp * ov.sum() / (2.0 * float(A.sum()))

        loss.backward()
        opt_t.step()
        with torch.no_grad():
            x.clamp_(x_fix - r_tr, x_fix + r_tr)
            y.clamp_(y_fix - r_tr, y_fix + r_tr)
        last = float(loss.detach())

    with torch.no_grad():
        w, h = sizes()
        xe = torch.where(mv, x, x_fix)
        ye = torch.where(mv, y, y_fix)
        Q = np.stack([xe.numpy(), ye.numpy(), w.numpy(), h.numpy()], axis=1)
    if not np.isfinite(Q).all():
        return None
    # exact-area restore for soft blocks (float drift from clamps)
    for i in range(n):
        if kind[i] == 0 and movable[i]:
            Q[i, 3] = areas[i] / max(Q[i, 2], 1e-9)
    if os.environ.get("ANALYTIC_DEBUG"):
        print(f"[analytic] steps done, last loss {last}", flush=True)
    return Q


def cg_legalize(P0: np.ndarray, Q: np.ndarray,
                kind: Sequence[int]) -> Optional[np.ndarray]:
    """Constraint-graph legalization: project the polished coordinates Q
    onto the no-overlap polytope defined by the ORIGINAL legal layout
    P0's pairwise orders (the ported core of our production slack
    refiner).  For every pair the separating axis and direction are taken
    from P0, so the result is overlap-free by construction and stays as
    close to Q as the order allows.  Locked blocks never move; None when
    a locked bound is violated (caller rejects)."""
    n = len(P0)
    x0p, y0p = P0[:, 0], P0[:, 1]
    x1p, y1p = x0p + P0[:, 2], y0p + P0[:, 3]
    w, h = Q[:, 2], Q[:, 3]
    locked = np.asarray([k == 2 for k in kind], dtype=bool)

    # pair -> separating axis from the legal reference layout
    gx = np.maximum(x0p[:, None] - x1p[None, :],
                    x0p[None, :] - x1p[:, None])   # x-gap (>=0: disjoint)
    gy = np.maximum(y0p[:, None] - y1p[None, :],
                    y0p[None, :] - y1p[:, None])
    use_x = gx >= gy   # separate along the axis with the larger P0 gap

    out = Q.copy()
    for axis in (0, 1):
        size = w if axis == 0 else h
        ref0 = x0p if axis == 0 else y0p
        des = out[:, axis].copy()
        des[locked] = P0[locked, axis]
        order = np.argsort(ref0, kind="stable")
        sel = use_x if axis == 0 else ~use_x
        pos = des.copy()
        for oi, i in enumerate(order):
            best = des[i]
            for oj in range(oi):
                j = order[oj]
                if not sel[i, j]:
                    continue
                # j precedes i along this axis in P0
                best = max(best, pos[j] + size[j])
            if locked[i]:
                if best > P0[i, axis] + 1e-6:
                    return None    # a locked bound cannot be pushed
                pos[i] = P0[i, axis]
            else:
                pos[i] = best
        out[:, axis] = pos
    # exact final check
    x0, y0 = out[:, 0], out[:, 1]
    x1, y1 = x0 + out[:, 2], y0 + out[:, 3]
    oxm = (np.minimum(x1[:, None], x1[None, :])
           - np.maximum(x0[:, None], x0[None, :]))
    oym = (np.minimum(y1[:, None], y1[None, :])
           - np.maximum(y0[:, None], y0[None, :]))
    bad = (oxm > 1e-7) & (oym > 1e-7)
    np.fill_diagonal(bad, False)
    if bad.any():
        return None
    return out
