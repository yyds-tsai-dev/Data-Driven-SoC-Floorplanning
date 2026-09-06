#!/usr/bin/env python3
"""Conjugate-subgradient (CSA) coordinate solver for the stage-2 refiner.

Surviving descendant of the CSF gate (see
`docs/design/2026-07-29-csf-analytical-prototype.md` sec. 7): the "analytical
replacement for the column backbone" line is closed, but ONE narrow bet
remains -- CSA as the *coordinate* solver on a topology the backbone has
already legalized.

Why this can beat the refiner's median sweeps.  `_Refiner._axis_pass` is a
coordinate descent over groups: it walks the separation DAG in topological
order and gives each group its own 1-D weighted-median optimum, clipped to
the interval left by the groups already placed.  A chain CAN translate
together (the backward `dmax` pass prices downstream slack), but the shift is
still chosen by the *upstream* group alone -- the HPWL that the dragged
downstream groups pay is never in the objective.  Coordinate descent on a
nonsmooth convex objective is exactly where that asymmetry bites: it can sit
at a point where no single group can improve alone yet a joint move does.

CSA optimizes the whole shift vector at once against the true weighted-L1
axis HPWL, so it prices the drag.  Nothing else changes: shapes are frozen
(`w*h` untouched -> area_gap inherited), the feasible set is the refiner's own
separation polytope (-> zero overlap by construction, `V_rel` inherited), and
the caller reverts the pass unless the full proxy strictly improves.

The objective, the analytic subgradient and the Polak-Ribiere loop are the
ones measured in `scripts/probes/csf_analytical_probe.py:objective/csa`;
what is new here is the polytope projection.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np


class AxisHpwlObjective:
    """Weighted-L1 HPWL restricted to one axis, as a function of the
    per-group rigid shift vector `d` (length G).

    A block in group `g` has center `cen_i + d[g]`; blocks with group id -1
    (preplaced / locked) are constants and are mapped to a sentinel slot G
    that is always 0.  Edges whose endpoints share a group are dropped --
    a rigid shift cannot change them, so they only add a constant.

    `value` and `subgrad` are one vectorized pass over the edge/pin arrays
    (n <= 120, |E| <= ~7k -> tens of microseconds).
    """

    __slots__ = ("G", "eu", "ev", "eb", "ew", "pu", "pb", "pw", "_pad")

    def __init__(self, G: int,
                 eu: np.ndarray, ev: np.ndarray,
                 eb: np.ndarray, ew: np.ndarray,
                 pu: np.ndarray, pb: np.ndarray, pw: np.ndarray) -> None:
        self.G = int(G)
        self.eu = np.ascontiguousarray(eu, dtype=np.int64)
        self.ev = np.ascontiguousarray(ev, dtype=np.int64)
        self.eb = np.ascontiguousarray(eb, dtype=np.float64)
        self.ew = np.ascontiguousarray(ew, dtype=np.float64)
        self.pu = np.ascontiguousarray(pu, dtype=np.int64)
        self.pb = np.ascontiguousarray(pb, dtype=np.float64)
        self.pw = np.ascontiguousarray(pw, dtype=np.float64)
        self._pad = np.zeros(self.G + 1, dtype=np.float64)

    @property
    def terms(self) -> int:
        return int(len(self.eb) + len(self.pb))

    def _dp(self, d: np.ndarray) -> np.ndarray:
        dp = self._pad
        dp[:self.G] = d
        return dp

    def value(self, d: np.ndarray) -> float:
        dp = self._dp(d)
        f = 0.0
        if len(self.eb):
            f += float(self.ew @ np.abs(self.eb + dp[self.eu] - dp[self.ev]))
        if len(self.pb):
            f += float(self.pw @ np.abs(self.pb + dp[self.pu]))
        return f

    def subgrad(self, d: np.ndarray) -> np.ndarray:
        """A subgradient of `value` at `d` (exact gradient a.e.)."""
        dp = self._dp(d)
        G = self.G
        g = np.zeros(G + 1, dtype=np.float64)
        if len(self.eb):
            s = self.ew * np.sign(self.eb + dp[self.eu] - dp[self.ev])
            g += np.bincount(self.eu, weights=s, minlength=G + 1)
            g -= np.bincount(self.ev, weights=s, minlength=G + 1)
        if len(self.pb):
            s = self.pw * np.sign(self.pb + dp[self.pu])
            g += np.bincount(self.pu, weights=s, minlength=G + 1)
        return g[:G]


class ShiftPolytope:
    """The refiner's own feasible set for a per-group rigid shift vector.

    `d[g] in [lo[g], dmax[g]]` plus the separation difference constraints
    `d[j] - d[i] >= c` collected by `_Refiner._axis_constraints`.

    `project` is a THREE-stage operator, and the middle stage is the whole
    reason this class exists:

      1. clamp to the box, tightened by a forward longest-path pass so the
         lower bound also prices the upstream chain;
      2. `sweeps` averaged-projection (Cimmino) passes over the violated
         difference constraints: each violated pair is pushed apart by half
         the violation, corrections are averaged per variable, over-relaxed
         by `omega`.  This is what turns an infeasible subgradient step into
         a COORDINATED one -- the mechanism the pass is built to exploit;
      3. `_feasible_clip`, the refiner's own topological clip, as the
         guarantee: it always returns a feasible point (that is the existing
         overlap-free-by-construction argument, unchanged) and it is the
         identity on feasible points, so a converged stage 2 passes through
         untouched.

    Stage 3 alone is NOT a usable projection: it walks the groups in
    topological order and gives each one its own bound, so a step that wants
    the head of a packed chain to go one way and its tail the other collapses
    to zero -- measured, on the `tests/test_partner_csa_refine.py` drag case
    it pins CSA at the coordinate-descent fixed point.  Stage 2 is what lets
    the chain reach consensus first.
    """

    __slots__ = ("G", "lo", "dmax", "edges_in", "edges_out", "order",
                 "pinned", "dmin", "ci", "cj", "cc", "deg", "sweeps",
                 "omega", "_rorder")

    def __init__(self, G: int, lo: np.ndarray, dmax: np.ndarray,
                 edges_in: List[List[Tuple[int, float]]],
                 edges_out: List[List[Tuple[int, float]]],
                 order: Sequence[int], pinned: np.ndarray,
                 sweeps: int = 12, omega: float = 1.6) -> None:
        self.G = int(G)
        self.lo = lo
        self.dmax = dmax
        self.edges_in = edges_in
        self.edges_out = edges_out
        self.order = order
        self._rorder = list(order)[::-1]
        self.pinned = pinned
        self.sweeps = int(sweeps)
        self.omega = float(omega)

        ci: List[int] = []
        cj: List[int] = []
        cc: List[float] = []
        for gj, lst in enumerate(edges_in):
            for gi, c in lst:
                ci.append(gi)
                cj.append(gj)
                cc.append(c)
        self.ci = np.array(ci, dtype=np.int64)
        self.cj = np.array(cj, dtype=np.int64)
        self.cc = np.array(cc, dtype=np.float64)
        if len(cc):
            deg = (np.bincount(self.ci, minlength=G)
                   + np.bincount(self.cj, minlength=G)).astype(np.float64)
            self.deg = np.maximum(deg, 1.0)
        else:
            self.deg = np.ones(G, dtype=np.float64)

        # forward longest path: the tightest lower bound implied by upstream
        # groups (the mirror of the refiner's backward `dmax` pass).  Purely a
        # bound tightening -- stage 3 remains the legality authority.
        dmin = np.array(lo, dtype=np.float64, copy=True)
        for gi in order:
            for gp, c in edges_in[gi]:
                v = dmin[gp] + c
                if v > dmin[gi]:
                    dmin[gi] = v
        self.dmin = dmin

    def _feasible_clip(self, t: np.ndarray) -> np.ndarray:
        d = np.zeros(self.G, dtype=np.float64)
        assigned = np.zeros(self.G, dtype=bool)
        lo_fixed = self.lo
        dmax = self.dmax
        pinned = self.pinned
        for gi in self.order:
            if pinned[gi]:
                assigned[gi] = True
                continue
            lo = lo_fixed[gi]
            for gp, c in self.edges_in[gi]:
                base = d[gp] if assigned[gp] else 0.0
                if base + c > lo:
                    lo = base + c
            hi = dmax[gi]
            for gj, c in self.edges_out[gi]:
                if assigned[gj] and d[gj] - c < hi:
                    hi = d[gj] - c
            if lo > hi:
                assigned[gi] = True
                continue
            v = t[gi]
            d[gi] = lo if v < lo else (hi if v > hi else v)
            assigned[gi] = True
        return d

    def _down_repair(self, d: np.ndarray) -> np.ndarray:
        """Make `d` satisfy every difference constraint by only LOWERING
        values: walk the order backwards and cap each group at
        `min_succ(d[j] - c)`.  A packed chain therefore adopts its most
        negative member -- the coordinated LEFT/DOWN move."""
        d = d.copy()
        for gi in self._rorder:
            v = d[gi]
            for gj, c in self.edges_out[gi]:
                u = d[gj] - c
                if u < v:
                    v = u
            d[gi] = v
        return d

    def _up_repair(self, d: np.ndarray) -> np.ndarray:
        """Mirror of `_down_repair`, raising instead: a packed chain adopts
        its most positive member -- the coordinated RIGHT/UP move."""
        d = d.copy()
        for gi in self.order:
            di = d[gi]
            for gj, c in self.edges_out[gi]:
                u = di + c
                if u > d[gj]:
                    d[gj] = u
        return d

    def retract(self, t: np.ndarray) -> List[np.ndarray]:
        """Feasible candidates for the raw (generally infeasible) point `t`.

        A single retraction is not enough.  `_feasible_clip` alone walks the
        groups head-first and hands each one its own bound, so a step that
        wants the head of a packed chain to go one way and its tail the other
        collapses to zero -- measured on the drag case in
        `tests/test_partner_csa_refine.py`, that pins CSA exactly at the
        coordinate-descent fixed point it is supposed to escape.  Nor can the
        step simply be shortened until feasible: on a saturated packing the
        separation constraints are TIGHT, so any direction that shrinks a
        contact gap admits only lambda = 0.

        So: smooth with a few averaged-projection (Cimmino) sweeps to spread
        each group's pull along its chain, then take BOTH one-sided repairs --
        `_down_repair` (the chain adopts its most negative target) and
        `_up_repair` (its most positive).  Each is exactly feasible after
        `_feasible_clip`, and the caller keeps whichever scores better.  The
        two together are what make a coordinated chain translation reachable
        in one step.
        """
        d = np.clip(t, self.dmin, self.dmax)
        d[self.pinned] = 0.0
        if len(self.cc):
            half = 0.5 * self.omega
            for _ in range(self.sweeps):
                viol = self.cc - (d[self.cj] - d[self.ci])
                if float(viol.max()) <= 1e-12:
                    break
                np.maximum(viol, 0.0, out=viol)
                viol *= half
                d = d + (np.bincount(self.cj, weights=viol, minlength=self.G)
                         - np.bincount(self.ci, weights=viol,
                                       minlength=self.G)) / self.deg
                np.clip(d, self.dmin, self.dmax, out=d)
                d[self.pinned] = 0.0
        if not len(self.cc):
            return [self._feasible_clip(d)]
        out = [self._feasible_clip(self._down_repair(d)),
               self._feasible_clip(self._up_repair(d))]
        return out

    def project(self, t: np.ndarray) -> np.ndarray:
        """Single-candidate form (the down repair); kept for callers that do
        not want to score alternatives."""
        return self.retract(t)[0]


def csa_shifts(obj: AxisHpwlObjective, poly: ShiftPolytope,
               step: float, iters: int, decay: float,
               deadline: Optional[float] = None
               ) -> Tuple[np.ndarray, float, float, int]:
    """Projected conjugate-subgradient descent (paper Alg. 1 + a projection).

        g_k  in  d f(u_{k-1})
        eta  =  g_k.(g_k - g_{k-1}) / ||g_{k-1}||^2        (Polak-Ribiere)
        p_k  =  -g_k + eta * p_{k-1}
        u_k  =  Proj( u_{k-1} + (c_k / ||p_k||) * p_k )

    The step is scale-free (`c / ||p||`), exactly as in the paper -- `c` is
    literally "how far a group may move this iteration".  The paper schedules
    `c` with tabular Q-learning; that layer is deliberately NOT ported (the
    paper itself reports it significant on 1 of 5 circuits, and a per-pass
    Q-table has nowhere to learn across 60 iterations).  A geometric decay
    takes its place, which is what makes the sweep converge.

    Returns `(best_d, best_f, f_at_zero, iterations_run)`.  `best_d` is
    feasible and `best_f` is its true objective value, so the caller can gate
    on `best_f < f_at_zero`.
    """
    G = obj.G
    d = np.zeros(G, dtype=np.float64)
    f0 = obj.value(d)
    best_d, best_f = d.copy(), f0
    p = np.zeros(G, dtype=np.float64)
    g_prev: Optional[np.ndarray] = None
    c = float(step)
    frozen = poly.pinned
    k = 0
    while k < iters:
        if deadline is not None and (k & 3) == 3 and time.time() >= deadline:
            break
        g = obj.subgrad(d)
        g[frozen] = 0.0
        if g_prev is None:
            eta = 0.0
        else:
            den = float(g_prev @ g_prev)
            eta = float(g @ (g - g_prev)) / den if den > 1e-18 else 0.0
        p = -g + eta * p
        nrm = float(np.sqrt(p @ p))
        if nrm < 1e-12:
            break
        cands = poly.retract(d + (c / nrm) * p)
        d = min(cands, key=obj.value)
        f = obj.value(d)
        if f < best_f - 1e-12:
            best_f = f
            best_d = d.copy()
        g_prev = g
        c *= decay
        k += 1
    return best_d, best_f, f0, k
