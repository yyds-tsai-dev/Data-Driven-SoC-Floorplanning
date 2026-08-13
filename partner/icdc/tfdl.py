"""TFDL -- Topology-Frozen Differentiable Legalization layer.

The engine's model head emits a raw layout `x_hat0` that overlaps.  TFDL turns
it into a layout that is hard-legal *by construction* while staying
differentiable in the model's outputs:

    x_hat0 -> [detached, discrete] per-pair separation axis + direction
           -> [differentiable] bounded longest-path compaction on each axis
           -> legal rectangles

Why the split matters (0810 design memo Sec. 4.1): the raw prediction's
hpwl/area are already *better* than golden precisely because everything is
piled on top of everything, so any energy evaluated before legalization is
minimised by the degenerate blob the model already sits in.  Freezing the
topology and projecting first means the energy only ever sees legal layouts,
and the gradient it produces is about the part the downstream refiner cannot
fix (hpwl topology optimality and soft violations) rather than raw overlap.

Guarantees of `tfdl()` output, unconditionally:
  * no pair overlaps            -- every pair is separated on one axis by an
                                   inequality the compaction enforces exactly;
  * soft-block area exact       -- (w, h) come from the exact-area
                                   parametrisation and are never touched here;
  * fixed shapes exact          -- ditto (the decoder overwrites them).

The one invariant that is *not* unconditional is the preplaced origin: a chain
of blocks the frozen topology puts between two preplaced blocks can be longer
than the gap the contest left there, and something has to give.  Pins are
therefore imposed as lower bounds only -- so the solve never has an infeasible
branch -- and the residual is returned as a per-block, non-negative, fully
differentiable `drift`.  `drift == 0` everywhere is exactly hard legality; a
positive drift is a gradient that says "stop inflating", which is what the
fine-tune has to learn, and is far more useful than a dead sample.  The dump
path treats any sample with drift > 0 as inadmissible.

Axis mechanics
--------------
Each axis is a difference-constraint system  c_u + s_u <= c_v  over a DAG.
With per-node box bounds [lo, hi] the system is feasible iff the longest-path
lower bound never exceeds the shortest-path upper bound, and

    mx_v = min(hi_v, min_u (hi_u - P[v, u]))      (latest feasible)
    mn_v = max(lo_v, max_u (lo_u + P[u, v]))      (earliest feasible)

where P is the max-plus transitive closure of the edge weights.  Setting
`lo_v <- min(c0_v, mx_v)` for every unpinned node makes `mn <= mx` provable by
induction, so the *only* residual infeasibility is a pin whose own upper bound
`mx_p` already dropped below the pinned value.  That is exactly the check
`tfdl()` reports.

`mn` is the output: the most compact realisation of the frozen topology that
still respects every block's predicted coordinate as a lower bound.  It stays
differentiable in both the predicted coordinates (through `min(c0, mx)`) and
the block sizes (through the closure), which is what carries gradient back to
the denoiser.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

NEG = -1.0e30
POS = 1.0e30
# a pin counts as honoured below this residual.  Matched to
# `layout_refiner._GUARD_DIM_TOL` (1e-5), which is the tolerance the rung-(-1)
# admission gate actually applies to a preplaced origin, itself 10x inside the
# evaluator's own 1e-4 hard check.
PIN_TOL = 1e-6


# ---------------------------------------------------------------------------
# discrete stage (detached): pair -> axis + direction
# ---------------------------------------------------------------------------
def extract_topology(rects: torch.Tensor, mask: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign every valid pair to the horizontal or the vertical axis.

    The pair goes on whichever axis already separates it more in the raw
    layout (the classic min-displacement rule); the direction is then fixed by
    centre order on that axis, which is a strict total order once ties are
    broken by block index, so each axis graph is acyclic by construction.

    Returns ``(use_h, use_v)``, both ``[B, N, N]`` symmetric boolean masks with
    a false diagonal and false rows/columns for padded blocks.
    """
    with torch.no_grad():
        x, y, w, h = rects.unbind(dim=-1)
        x1, y1 = x + w, y + h
        gx = torch.maximum(x.unsqueeze(1) - x1.unsqueeze(2),
                           x.unsqueeze(2) - x1.unsqueeze(1))
        gy = torch.maximum(y.unsqueeze(1) - y1.unsqueeze(2),
                           y.unsqueeze(2) - y1.unsqueeze(1))
        pair = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(mask.shape[1], dtype=torch.bool,
                        device=mask.device).unsqueeze(0)
        pair = pair & ~eye
        use_h = (gx >= gy) & pair
        use_v = (~(gx >= gy)) & pair
    return use_h, use_v


def _perm_for_axis(centre: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Stable sort by centre with padded blocks pushed to the tail."""
    key = torch.where(mask, centre, torch.full_like(centre, POS))
    return torch.argsort(key, dim=1, stable=True)


# ---------------------------------------------------------------------------
# differentiable stage: bounded longest path on a frozen DAG
# ---------------------------------------------------------------------------
def _closure(adj: torch.Tensor, size: torch.Tensor,
             chunk: int = 32) -> torch.Tensor:
    """Max-plus transitive closure of the DAG.

    ``adj[b, u, v]`` true means ``c_u + size_u <= c_v``.  Returns ``P`` with
    ``P[b, u, v]`` = the heaviest path weight from u to v (``NEG`` if no path),
    computed by repeated squaring so the sequential depth is log2(N) instead of
    N.  Differentiable in ``size``.
    """
    B, N, _ = adj.shape
    P = torch.where(adj, size.unsqueeze(2).expand(B, N, N),
                    torch.full((), NEG, dtype=size.dtype, device=size.device))
    steps = max(1, int(N - 1).bit_length())
    for _ in range(steps):
        parts = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            # (P[u,k] + P[k,v]) maxed over k, for u in [s, e)
            parts.append((P[:, s:e, :].unsqueeze(3)
                          + P.unsqueeze(1)).max(dim=2).values)
        P = torch.maximum(P, torch.cat(parts, dim=1))
    return P


def _closure_sequential(adj: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Reference implementation of :func:`_closure` (O(N) sequential passes).

    Used by the unit tests to pin the fast doubling version; the adjacency is
    strictly upper triangular so a single sweep in index order suffices.
    """
    B, N, _ = adj.shape
    P = torch.where(adj, size.unsqueeze(2).expand(B, N, N),
                    torch.full((), NEG, dtype=size.dtype, device=size.device))
    for k in range(N):
        P = torch.maximum(P, P[:, :, k:k + 1] + P[:, k:k + 1, :])
    return P


def _exact_pass_rev(hi: torch.Tensor, size: torch.Tensor,
                    adj: torch.Tensor) -> torch.Tensor:
    """Reverse twin of :func:`_exact_pass` for the latest schedule, so a block
    slid to the far wall abuts its successor at a bit-identical float."""
    with torch.no_grad():
        B, N = hi.shape
        C = hi.clone()
        for v in range(N - 1, -1, -1):
            succ = adj[:, v, :]
            if not bool(succ.any()):
                continue
            cand = torch.where(succ, C - size[:, v:v + 1],
                               torch.full_like(C, POS))
            C[:, v] = torch.minimum(hi[:, v], cand.min(dim=1).values)
    return C


def _seat(mn: torch.Tensor, size: torch.Tensor, P: torch.Tensor,
          pinned: torch.Tensor, seat0: torch.Tensor,
          mask: torch.Tensor, adj: Optional[torch.Tensor] = None,
          exact: bool = False) -> torch.Tensor:
    """Slide max-side boundary-tagged blocks to the far wall, legally.

    The earliest schedule is biased: it packs everything toward the low wall,
    so a block tagged "must touch the right edge" is left stranded in the
    middle and the evaluator charges a boundary bit for it.  Measured, that
    bias is the dominant violation term the projection itself introduces.

    The fix uses the fact that the feasible set of a fixed topology is a
    *convex polyhedron*: with `mn` the earliest schedule and `mx` the latest
    one inside the same frame, `c = (1-t)*mn + t*mx` satisfies every separation
    constraint for any `t` that is non-decreasing along the DAG.  (For an edge
    u->v the slack condition reduces to `(t_v - t_u)*(mx_v - mn_v) >= 0`.)
    So: set `t = 1` on the tagged blocks, propagate it forward to every
    descendant through the closure's reachability, and blend.  Blocks that are
    not tagged and not downstream of one do not move at all.
    """
    big = torch.full_like(mn, POS)
    ext = torch.where(mask, mn + size, torch.full_like(mn, NEG)).max(dim=1, keepdim=True).values
    hi = torch.where(pinned, mn, torch.minimum(big, ext - size))
    cand = torch.where(P > NEG / 2, hi.unsqueeze(1) - P, big.unsqueeze(1))
    mx = torch.minimum(hi, cand.min(dim=2).values)
    if exact and adj is not None:
        seq = _exact_pass_rev(hi.detach(), size.detach(), adj)
        mx = mx + (seq - mx).detach()
    mx = torch.maximum(mx, mn)              # guards the drifted-pin corner case
    # only tagged blocks that are NOT already against the wall trigger a slide;
    # otherwise the propagation would drag their descendants around for nothing
    # (measured: sliding unconditionally costs +0.004 V_rel on a production
    # layout that was already seated)
    with torch.no_grad():
        need = seat0 * ((ext - (mn + size)) > PIN_TOL).to(seat0.dtype)
    reach = (P > NEG / 2)
    t = torch.maximum(need, torch.where(reach, need.unsqueeze(2),
                                        torch.zeros_like(P)).max(dim=1).values)
    return mn + t * (mx - mn)


def _axis_solve(c0: torch.Tensor, size: torch.Tensor, adj: torch.Tensor,
                pinned: torch.Tensor, pin_val: torch.Tensor,
                closure=_closure, seat0: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                exact: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """One axis.  All tensors are in the axis' own permuted order.

    Returns ``(c, drift)``.  `c` satisfies every separation edge exactly, by
    construction, for *every* input -- there is no infeasible branch.  What can
    fail is only the pin: `drift = c_p - pin_p >= 0` is the distance a chain of
    blocks pushed a preplaced block past the position the contest nailed it to.
    `drift == 0` on every pin is exactly the condition for the sample to be
    hard-legal; a non-zero drift is a differentiable quantity the trainer
    penalises rather than a dead sample it has to throw away.

    Keeping the pins as *lower* bounds (never upper) is what makes the solve
    unconditionally feasible; the upper bounds still enter through `mx`, which
    tightens the unpinned lower bounds and therefore reduces drift wherever the
    system does admit a pin-exact solution.
    """
    P = closure(adj, size)
    hi = torch.where(pinned, pin_val, torch.full_like(c0, POS))
    # mx_v = min(hi_v, min_u (hi_u - P[v, u])): the latest position that could
    # still respect every pin downstream of v
    cand_hi = torch.where(P > NEG / 2, hi.unsqueeze(1) - P,
                          torch.full_like(P, POS))
    mx = torch.minimum(hi, cand_hi.min(dim=2).values)
    lo = torch.where(pinned, pin_val, torch.minimum(c0, mx))
    # mn_v = max(lo_v, max_u (lo_u + P[u, v]))
    cand_lo = torch.where(P > NEG / 2, lo.unsqueeze(2) + P,
                          torch.full_like(P, NEG))
    mn = torch.maximum(lo, cand_lo.max(dim=1).values)
    if exact:
        seq = _exact_pass(lo.detach(), size.detach(), adj)
        mn = mn + (seq - mn).detach()          # straight-through
        # Sequential exact abutment can move a mathematically fixed pin by a
        # handful of ulps because it rebuilds coordinates through a different
        # addition chain.  Snap only sub-contract residuals back to the
        # authorized origin; a genuinely infeasible pin remains displaced and
        # is still rejected by the drift/hard-legality gates.
        near_pin = pinned & ((mn - pin_val).abs() <= PIN_TOL)
        mn = torch.where(near_pin, pin_val, mn)
    drift = torch.where(pinned, mn - pin_val, torch.zeros_like(mn))
    if seat0 is not None and bool((seat0 > 0).any()):
        m = mask if mask is not None else torch.ones_like(pinned)
        mn = _seat(mn, size, P, pinned, seat0, m, adj=adj, exact=exact)
    return mn, drift


# ---------------------------------------------------------------------------
# guaranteed-legal fallback
# ---------------------------------------------------------------------------
def _exact_pass(lo: torch.Tensor, size: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
    """Recompute the compaction sequentially so abutments are bit-exact.

    Why this exists, measured: the official grouping check unions the group's
    boxes with shapely and counts the pieces, so two blocks that are meant to
    touch must have block j's left edge be *the same float* as block i's right
    edge.  The closure computes ``c_j`` as ``lo_w + P[w, j]`` -- an accumulated
    sum -- which lands 1-2 ulp off, and a 2e-13 gap is enough for shapely to
    report two components.  Injecting an already-legal 0.3 s production layout
    through the closure alone moved nothing (max |dx| = 2e-13) yet pushed the
    official cost from 1.1051 to 1.2449 on the n>=100 band, entirely through
    V_grouping.

    One sequential sweep in topological order, taking each binding coordinate
    as the literal float ``c_i + size_i``, removes that failure mode.  It runs
    under no_grad and is wired straight-through, so it changes the value the
    evaluator sees without touching the gradient the closure produced.
    """
    with torch.no_grad():
        B, N = lo.shape
        C = lo.clone()
        for u in range(N):
            pred = adj[:, :, u]
            if not bool(pred.any()):
                continue
            cand = torch.where(pred, C + size, torch.full_like(C, NEG))
            C[:, u] = torch.maximum(lo[:, u], cand.max(dim=1).values)
    return C


def shelf_fallback(rects: torch.Tensor, mask: torch.Tensor,
                   pinned: torch.Tensor) -> torch.Tensor:
    """Detached, always-legal layout: preplaced stay put, everything else is
    shelf-packed in a strip strictly above every preplaced block.

    Quality is bad on purpose -- this exists so the layer's *legality* contract
    holds for 100% of samples even when the frozen topology cannot host the
    pins, not to be competitive.
    """
    with torch.no_grad():
        B, N, _ = rects.shape
        x, y, w, h = rects.unbind(dim=-1)
        out = rects.clone()
        top = torch.where(pinned & mask, y + h,
                          torch.full_like(y, NEG)).max(dim=1).values
        left = torch.where(pinned & mask, x,
                           torch.full_like(x, POS)).min(dim=1).values
        left = torch.where(left > POS / 2, torch.zeros_like(left), left)
        top = torch.where(top < NEG / 2, torch.zeros_like(top), top)
        span = torch.sqrt((w * h * mask).sum(dim=1).clamp_min(1.0)) * 1.05
        span = torch.maximum(span, w.max(dim=1).values)
        for b in range(B):
            cx, cy, row_h = float(left[b]), float(top[b]), 0.0
            lim = float(left[b]) + float(span[b])
            for i in range(N):
                if not bool(mask[b, i]) or bool(pinned[b, i]):
                    continue
                wi, hi_ = float(w[b, i]), float(h[b, i])
                if cx > float(left[b]) and cx + wi > lim:
                    cx, cy, row_h = float(left[b]), cy + row_h, 0.0
                out[b, i, 0] = cx
                out[b, i, 1] = cy
                cx += wi
                row_h = max(row_h, hi_)
    return out


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def tfdl(rects: torch.Tensor, mask: torch.Tensor,
         pinned: Optional[torch.Tensor] = None,
         pin_xy: Optional[torch.Tensor] = None,
         boundary_code: Optional[torch.Tensor] = None,
         exact: bool = False,
         closure=_closure) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a raw layout onto the overlap-free set with its topology frozen.

    Args:
        rects:   ``[B, N, 4]`` as ``(x, y, w, h)``.  Sizes are taken as given
                 (the decoder already made them exact-area / fixed-shape /
                 MIB-uniform) and are never modified here.
        mask:    ``[B, N]`` valid-block mask.
        pinned:  ``[B, N]`` preplaced mask.
        pin_xy:  ``[B, N, 2]`` the origins those blocks must keep; defaults to
                 the origins already in `rects` (the decoder writes them).

    Returns ``(legal_rects, drift)``:
      * `legal_rects` is overlap-free for every sample, unconditionally;
      * `drift` is ``[B, N, 2]``, non-negative, and zero exactly where the pin
        survived.  ``drift.amax((1,2)) == 0`` is the hard-legality predicate.
    """
    B, N, _ = rects.shape
    if pinned is None:
        pinned = torch.zeros_like(mask)
    pinned = pinned & mask
    x, y, w, h = rects.unbind(dim=-1)
    if pin_xy is None:
        pin_xy = rects[..., :2]
    use_h, use_v = extract_topology(rects, mask)
    zero = torch.zeros_like(x)
    w_eff = torch.where(mask, w, zero)
    h_eff = torch.where(mask, h, zero)

    if boundary_code is None:
        seat_x = seat_y = None
    else:                                   # bit 2 = right wall, 4 = top wall
        seat_x = (((boundary_code & 2) != 0) & mask).to(rects.dtype)
        seat_y = (((boundary_code & 4) != 0) & mask).to(rects.dtype)

    out, drifts = [], []
    for cen, c0, size, adj_full, pv, s0 in (
            (x + 0.5 * w, x, w_eff, use_h, pin_xy[..., 0], seat_x),
            (y + 0.5 * h, y, h_eff, use_v, pin_xy[..., 1], seat_y)):
        perm = _perm_for_axis(cen.detach(), mask)
        gi = perm.unsqueeze(2).expand(B, N, N)
        gj = perm.unsqueeze(1).expand(B, N, N)
        adj = adj_full.gather(1, gi).gather(2, gj)
        adj = adj & torch.triu(torch.ones(N, N, dtype=torch.bool,
                                          device=rects.device),
                               diagonal=1).unsqueeze(0)
        c, d = _axis_solve(
            torch.where(mask, c0, zero).gather(1, perm),
            size.gather(1, perm),
            adj,
            pinned.gather(1, perm),
            torch.where(pinned, pv, zero).gather(1, perm),
            closure=closure, exact=exact,
            seat0=None if s0 is None else s0.gather(1, perm),
            mask=mask.gather(1, perm),
        )
        inv = torch.argsort(perm, dim=1)
        out.append(c.gather(1, inv))
        drifts.append(d.gather(1, inv))

    legal = torch.stack([out[0], out[1], w, h], dim=-1)
    legal = torch.where(mask.unsqueeze(-1), legal, rects)
    return legal, torch.stack(drifts, dim=-1).clamp_min(0.0)
