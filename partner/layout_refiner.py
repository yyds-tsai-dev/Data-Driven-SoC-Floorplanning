#!/usr/bin/env python3
"""Stage-2 continuous refiner (constraint-graph / compaction style).

Takes a legal column-slicing layout and improves HPWL by moving blocks
continuously, freed from the column model's x-quantization and stack-order
y-placement.  All moves keep the layout legal by construction:

  * Per axis pass, a block (or rigid cluster group) may only slide inside
    the interval bounded by its projection-overlapping neighbors, so no
    overlap can ever be created (the frozen axis does not change, hence the
    set of potentially-colliding pairs does not change).
  * The target coordinate is the weighted median of connected neighbor
    centers and pin positions (the exact 1-D HPWL optimum for the block),
    clipped to the feasible interval — a Gauss-Seidel sweep over blocks.
  * Cluster groups move rigidly and their internal touch contacts are
    re-snapped to exact float equality after every move (the evaluator's
    grouping check via shapely requires exact abutment).
  * Soft blocks reshape with w*h == target area exactly; MIB groups
    reshape in sync so all members keep identical (w, h).
  * Blocks whose boundary tag is currently satisfied are pinned on that
    axis; all blocks stay inside the initial bounding box, so neither the
    bbox area nor any satisfied tag can regress.

Every round is scored with the same HPWL / area / violation proxy the
stage-1 search uses; the best snapshot is returned, so the refiner can
never return something worse than its input under that proxy.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import List, Optional

import numpy as np

import csa_coordinate_solver as _csa

_DEBUG = bool(os.environ.get("REFINER_DEBUG"))
# PARTNER_SEAT_DEBUG=1 (default off, read once at import): per-call rung-0
# anatomy of `refine_prediction` -- the slice the pool worker was handed, the
# `_Refiner` build cost (the instance's own O(n^2) cost unit) and whether the
# fixed-frame rung closed.  Calibration instrument for the direct-channel seat
# gate (`PARTNER_DIRECT_SEAT_FIX` in contest_optimizer.py); costs nothing when
# off and never touches a numeric path.
_SEAT_DBG = bool(os.environ.get("PARTNER_SEAT_DEBUG"))
# PARTNER_FRAME_SCALE_DEBUG=1 (default off, read once at import): per-attempt
# trace of the rung-0 frame-scale ladder (`frame_scale_set`), plus an
# in-process `_FS_STATS` log the offline replay harnesses read.  Off it is a
# falsy module constant, so every capture site is one `if` on a dead branch.
_FS_DBG = bool(os.environ.get("PARTNER_FRAME_SCALE_DEBUG"))
_FS_STATS: List[tuple] = []

EDGE_EPS = 1e-6     # evaluator boundary-touch / overlap tolerance
SEP_TOL = 5e-7      # projection overlap beyond this forces a separation constraint
CONTACT_TOL = 1e-9  # tolerance when recording exact cluster contacts
TOUCH_TOL = 1e-7    # touching test for grouping checks (matches legalizer)


_TRUE = ("1", "true", "True", "on", "ON")


def _flag_on(name: str) -> bool:
    return os.environ.get(name, "0") in _TRUE


def early_exit_on() -> bool:
    """PARTNER_EARLY_EXIT=1 (default off).  Mirror of the canonical helper in
    column_sa_legalizer.py -- duplicated (12 lines) so this module keeps its
    stand-alone import graph; see
    docs/design/2026-08-04-early-exit-true-time-reduction.md."""
    return _flag_on("PARTNER_EARLY_EXIT")


def edge_seat_v2_on() -> bool:
    """PARTNER_EDGE_SEAT_V2=1 (default off): widen `_edge_seat`'s reach and
    make its acceptance test evaluator-faithful.

    Four coupled changes, all inside `_edge_seat` (see its docstring for the
    per-change rationale):

      (i)   GAP scales with the layout -- max(2.0, 0.08 * min(bbox_w, bbox_h))
            instead of a flat 2.0.  A flat window is a different fraction of
            the frame on a 21-block and a 120-block case, so on large frames
            it declared every real hover "a real misplace" and declined.
      (ii)  the (c) outlier-pull cap goes 3 -> 8 outliers.
      (iii) a corner-seat pass runs BEFORE the per-edge loop: a block tagged
            for two walls needs a corner, and seating it one axis at a time
            makes the intermediate state worse on the other axis, so the
            per-edge loop's strict-improvement gate rejects step one and the
            corner is never reached.  The pass moves both axes at once.
      (iv)  acceptance switches from the solver-internal `full_violations`
            proxy to the evaluator's own boundary+grouping+MIB total.  The
            proxy nets moves across categories with different semantics
            (TOUCH_TOL vs exact touch), so a net-negative proxy delta can be
            a net-POSITIVE official delta -- measured on the offline probe.

    Off: `_edge_seat` runs the historical proxy/2.0/3 path unchanged."""
    return _flag_on("PARTNER_EDGE_SEAT_V2")


def early_exit_window(specific: str, default: float = 0.25) -> float:
    """Stall-window fraction: explicit phase env > PARTNER_EARLY_EXIT_WINDOW
    (when EARLY_EXIT is on) > historical default."""
    raw = os.environ.get(specific)
    if raw is None and early_exit_on():
        raw = os.environ.get("PARTNER_EARLY_EXIT_WINDOW")
        default = 0.15
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if 0.0 < val < 1.0 else default


def early_exit_min_window() -> float:
    """Absolute floor (seconds) under the fractional stall window."""
    try:
        val = float(os.environ.get("PARTNER_EARLY_EXIT_MIN_WINDOW", "0.05"))
    except ValueError:
        return 0.05
    return val if val >= 0.0 else 0.05


# ---------------------------------------------------------------------------
# PARTNER_REFINE_PROF: wall-clock anatomy of `_Refiner.run` (default off)
#
# The gate question for kernelizing more of the refiner is "what fraction of a
# run does each section actually cost", and that is not answerable from the
# rung anatomy in refine_numeric_kernel.py: a rung is `legalize_soft`
# (hold=True), whereas `run` is the HPWL half -- the median sweep, the reshape
# pass, the discrete batch, the tag snap.  This measures exactly those.
#
# Off (the default) every site is `self._prof is not None`, so not even
# `time.time()` is called; the conditional-expression form below matters,
# because `run` is deadline-bounded and a timer per pass would BE a policy
# change.  On, one JSON line per `run()` call goes to `PARTNER_REFINE_PROF_FILE`
# (append; pass a per-process path if you fan out) or, failing that, stderr.
# ---------------------------------------------------------------------------
_PROF_KEYS = ("axis_soft", "reshape", "tag_snap", "discrete", "squeeze",
              "overlap", "key", "csa", "perturb", "deflate", "build_swap")

# -- PARTNER_REFINE_PROF_DISC: second-tier anatomy of the `discrete` bucket --
#
# `discrete` is one bucket in `_PROF_KEYS` but four different move classes
# under the hood, and the kernelization question ("what inside it is worth
# moving to numba") needs the split.  These keys live in a SEPARATE namespace
# (`d_` prefix, `_DPROF_KEYS`) precisely so that `sum(_PROF_KEYS) + other ==
# span` -- the invariant the PROF test asserts -- keeps holding: the sub-
# buckets partition `discrete`, they do not add to it.
#
# Requires PARTNER_REFINE_PROF (there is nothing to attribute to otherwise).
# Off, `self._dprof is None` and no timer is read; the flag exists for the
# micro harness, never for a scored run -- it adds ~2 `time.time()` per peer
# probe, which on a deadline-bounded batch IS a policy perturbation.
_DPROF_KEYS = (
    "enum_optpt",    # `_optimal_point` (per swappable block)
    "enum_gain",     # the two `_block_hp` calls that score the block
    "enum_rest",     # candidate list build + sort
    "screen_np",     # per-candidate dist/ratio/argsort over `idxs`
    "screen_delta",  # the 4 `_block_hp` calls of the pair delta
    "swap_edit",     # `_try_swap` geometry (snapshot + reshape), pre-legalize
    "insert_edit",   # `_try_insert` geometry, pre-legalize
    "lc_axis",       # `_legal_check`: `_axis_pass(hold=True)` pairs
    "lc_ovl",        # `_legal_check`: `_has_overlap`
    "lc_key",        # `_legal_check`: `_key` (hpwl + violations)
    "lc_rest",       # `_legal_check`: snapshot restore / control
    "book",          # accept bookkeeping (centroid recompute)
    "m_enum",        # `_matching_batch` gain enumeration
    "m_cost",        # `_matching_batch` m*m cost matrix (`_block_hp`)
    "m_dp",          # `_matching_batch` bitmask assignment DP
    "m_apply",       # `_matching_batch` permutation apply + legal check
    "m_rest",        # `_matching_batch` pool build / control
)


def _prof_new() -> dict:
    d = {}
    for k in _PROF_KEYS:
        d["t_" + k] = 0.0
        d["c_" + k] = 0
    return d


def _dprof_new() -> dict:
    d = {}
    for k in _DPROF_KEYS:
        d["t_" + k] = 0.0
        d["c_" + k] = 0
    return d


def _prof_emit(ref, span: float, rounds: int, batches: int,
               swaps: int = -1) -> None:
    import json
    import sys
    p = ref._prof
    # `swaps` is the discrete phase's ACCEPT count.  It is in the record
    # because the accept rate decides which half of `_legal_check` the wall
    # clock is in: a rejected attempt stops at the overlap test, an accepted
    # one goes on to `_key` (`_hpwl` + `_violations`, ~50 us at n=118).  A
    # profile without it cannot tell those two regimes apart.
    rec = {"pid": os.getpid(), "n": int(ref.n), "G": len(ref.groups),
           "span": round(span, 6), "rounds": int(rounds),
           "batches": int(batches), "swaps": int(swaps)}
    acc = 0.0
    for k in _PROF_KEYS:
        t = p["t_" + k]
        acc += t
        rec[k] = round(t, 6)
        rec["c_" + k] = p["c_" + k]
    rec["other"] = round(max(span - acc, 0.0), 6)
    for k in _PROF_KEYS + ("other",):
        rec["f_" + k] = round(rec[k] / span, 4) if span > 0 else 0.0
    dp = ref._dprof
    if dp is not None:
        # sub-buckets are a partition OF `discrete`, so they are reported as
        # fractions of `discrete`, not of `span`, and never enter `acc`.
        den = rec["discrete"]
        dacc = 0.0
        for k in _DPROF_KEYS:
            t = dp["t_" + k]
            dacc += t
            rec["d_" + k] = round(t, 6)
            rec["dc_" + k] = dp["c_" + k]
            rec["df_" + k] = round(t / den, 4) if den > 0 else 0.0
        rec["d_unattributed"] = round(max(den - dacc, 0.0), 6)
        rec["df_unattributed"] = (round(rec["d_unattributed"] / den, 4)
                                  if den > 0 else 0.0)
    line = json.dumps(rec, sort_keys=True)
    path = os.environ.get("PARTNER_REFINE_PROF_FILE", "")
    if path:
        # one file per process: `run` executes in forked pool workers, and
        # O_APPEND interleaving is only atomic on local filesystems -- this
        # repo lives on a network mount.  Aggregate with a glob.
        try:
            with open("%s.%d" % (path, os.getpid()), "a") as fh:
                fh.write(line + "\n")
            return
        except OSError:
            pass
    print("[refine-prof] " + line, file=sys.stderr, flush=True)


def _env_pos(name: str, default: float, hi: float) -> float:
    """Positive float env override; malformed or out of (0, hi] -> default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if 0.0 < v <= hi else default


def anytime_ladder_on() -> bool:
    """PARTNER_ANYTIME_LADDER=1 (default off).

    Makes `refine_prediction`'s legalization ladder anytime: the tight rungs
    get a bounded share of the span, a guaranteed-legal rung is always
    reachable, and the quality tail (frame anneal + refiner + repair) is
    funded by fractions of the span instead of absolute second offsets.  See
    docs/design/2026-08-04-anytime-ladder.md."""
    return _flag_on("PARTNER_ANYTIME_LADDER")


def anytime_frac(name: str, default: float) -> float:
    """Fractional anytime tunable in (0, 1); malformed -> default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if 0.0 < v < 1.0 else default


def frame_scale_set() -> tuple:
    """PARTNER_FRAME_SCALE_LADDER=1 (default off): rung-0 frame-scale ladder.

    `refine_prediction`'s rung 0 legalizes into a frame whose area is
    `area_ref * 1.08`, and `_seed_tags` then seats every boundary-tagged
    block flush against that frame's walls -- so a rung-0 success returns a
    bbox of EXACTLY `1.08 * area_ref` (verified per-case on the tail).  The
    evaluator's area baseline is the golden bbox, which sits at ~1.01 *
    area_ref there, so every direct candidate pays a structural
    `area_gap` of +0.05..0.08 (~+0.025..0.04 of no-runtime cost) that
    nothing downstream recovers at the sub-second tiers: the step-7
    recompression is gated on `bbox > 1.08 * area_ref` (strict, so a rung-0
    bbox never trips it) behind an absolute `t_hard - 2.5` gate that never
    opens there.

    ON: try `PARTNER_FRAME_SCALE_SET` (default "1.02,1.05,1.08") in
    ascending order and keep the first scale whose fixed-frame rung
    legalizes.  A SINGLE-element set is a pure constant shift -- one
    attempt, no extra time, only a different frame.  Multi-element sets buy
    the tighter frame with rung-0 attempts (see
    `PARTNER_FRAME_SCALE_TIGHT_FRAC` / `PARTNER_FRAME_SCALE_MIN_N`).

    OFF -> `(1.08,)`: one attempt at the shipped scale, bit for bit."""
    if not _flag_on("PARTNER_FRAME_SCALE_LADDER"):
        return (1.08,)
    raw = os.environ.get("PARTNER_FRAME_SCALE_SET", "1.02,1.05,1.08")
    out: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        # a frame below the area reference cannot hold the blocks, and a
        # scale that is not strictly tighter than its successor is a wasted
        # attempt -- drop both rather than fail
        if 1.0 <= v <= 2.0 and (not out or v > out[-1] + 1e-9):
            out.append(v)
    return tuple(out) if out else (1.08,)


class _AxisCons:
    """Separation polytope for one axis: per-group rigid-shift box
    [lo, dmax] plus the difference constraints d[j] - d[i] >= c."""
    __slots__ = ("G", "lo", "dmax", "edges_in", "edges_out", "order",
                 "pinned")

    def __init__(self, G, lo, dmax, edges_in, edges_out, order, pinned):
        self.G = G
        self.lo = lo
        self.dmax = dmax
        self.edges_in = edges_in
        self.edges_out = edges_out
        self.order = order
        self.pinned = pinned


class _Group:
    __slots__ = ("members", "pin_x", "pin_y", "cH", "cV",
                 "eM", "eJ", "eW", "pM", "pX", "pY", "pW")

    def __init__(self, members):
        self.members = np.asarray(members, dtype=np.int64)
        self.pin_x = False
        self.pin_y = False
        self.cH: List = []   # (a, b) with x0[b] == x0[a] + w[a]
        self.cV: List = []   # (a, b) with y0[b] == y0[a] + h[a]


class _Refiner:
    def __init__(self, opt, pos: np.ndarray, seed: int = 0):
        self.opt = opt
        self.n = opt.n
        self.rng = random.Random(seed)
        self.P = np.array(pos, dtype=np.float64, copy=True)

        P = self.P
        self.xmin = float(P[:, 0].min())
        self.ymin = float(P[:, 1].min())
        self.xmax = float((P[:, 0] + P[:, 2]).max())
        self.ymax = float((P[:, 1] + P[:, 3]).max())
        # frame edges pinned to tagged-preplaced block edges by
        # _anchor_frame_to_tags; squeeze/tighten must not cross them
        self.lock_xmin = self.lock_xmax = None
        self.lock_ymin = self.lock_ymax = None

        self.kind = np.array(opt.kind, dtype=np.int64)
        self.in_cluster = np.zeros(self.n, dtype=bool)
        # deflate only pays off on prediction-shaped layouts; on the
        # saturated column packings every attempt fails slowly, and
        # pool workers must never burn their deadline on it
        self.enable_deflate = False

        # -- PARTNER_REFINE_KERNEL companions, set before the builds --------
        # `_nk` must exist before any dispatch site can run; `_fast_build`
        # selects the O(|E| + G) bucketing in `_build_edges` (see there).
        # `PARTNER_REFINE_FASTBUILD=0` disables the latter on its own, so the
        # two mechanisms stay separately attributable in an A/B.
        self._nk = None
        self._fast_build = (
            os.environ.get("PARTNER_REFINE_KERNEL", "") == "numba"
            and os.environ.get("PARTNER_REFINE_FASTBUILD", "1") in _TRUE)
        # PARTNER_REFINE_KERNEL_DISC dispatch flags; both stay False unless
        # `try_attach` actually produced a kernel with the DISC CSR built, so
        # a numba-less box or a failed attach silently keeps the Python path.
        self._disc_fuse = False
        self._disc_hp = False

        self._build_groups()
        self._build_pins()
        self._build_edges()
        self._build_reshape_units()
        self._tagged_set = {int(i) for i in opt._bnd_idx}

        self.hp0 = max(opt._hpwl(P), 1e-9)

        # -- refine stall early stop (PARTNER_REFINE_STALL_STOP=1, off) ----
        # Companion to `PARTNER_SA_STALL_STOP` in column_sa_legalizer.py: `run`
        # breaks out of a search phase once `_key()` (the same proxy the
        # phase already computes each pass) has not improved by >= stall_eps
        # (relative) within one stall window.  The window is a fraction of
        # the CURRENT phase's own remaining span -- self-scaling, like the SA
        # port, because a refiner span ranges from ~0.5s (a rung tighten) to
        # ~20s (a tail-case worker).
        #
        # This is the terminal time sink of BOTH pool worker classes:
        # `_worker_refine` ends in `run` (refine_prediction step 5) and
        # `_worker_solve` ends in `run` too (finish() -> refine_positions,
        # the stage-2 slice that is what actually pins an SA worker to
        # worker_deadline).  So the stop can convert convergence into a
        # shorter per-case wall clock, which the SA-side stop alone could
        # not do (it only re-spent the time inside the same worker).
        # Default off => decision logic unchanged, rng stream untouched
        # (the added predicates consume no randomness).
        #
        # PARTNER_EARLY_EXIT=1 additionally changes what happens to the
        # reclaimed time: `run`'s discrete phase stops re-scaling its window
        # to the (now large) leftover span, and `refine_prediction` claws the
        # unspent tail back out of its own hard deadline instead of feeding
        # it to the recompression retry.  EARLY_EXIT implies this stop.
        self.stall_eps = 0.002
        self._stall_frac = 0.0
        self._run_stalled = False
        self._early_exit = early_exit_on()
        if self._early_exit or _flag_on("PARTNER_REFINE_STALL_STOP"):
            self._stall_frac = early_exit_window(
                "PARTNER_REFINE_STALL_WINDOW", 0.25)
            try:
                self.stall_eps = float(os.environ.get(
                    "PARTNER_REFINE_STALL_EPS", "0.002"))
            except ValueError:
                self.stall_eps = 0.002

        # -- CSA coordinate polish (PARTNER_CSA_REFINE=1, off) -------------
        # Surviving descendant of the CSF gate (docs/design/
        # 2026-07-29-csf-analytical-prototype.md sec. 7): a joint projected
        # conjugate-subgradient sweep over the per-group shift vector, run at
        # the median sweep's fixed point.  `_axis_pass` is a coordinate
        # descent -- each group takes its OWN 1-D weighted-median optimum and
        # never pays for the downstream chain it drags -- so it can stall at
        # a point where only a joint move improves.  CSA prices the drag.
        #
        # Shapes are untouched (area_gap inherited), the feasible set is
        # `_axis_constraints` (overlap-free by construction, satisfied
        # boundary tags pinned, clusters rigid -> V_rel inherited), and
        # `_csa_pass` reverts unless the full `_key()` proxy improves.  The
        # pass consumes no `self.rng` draws, so with the flag off the
        # decision logic AND the rng stream are bit-identical to the
        # pre-port fork (csa_share == 0.0 makes every predicate dead).
        self.csa_iters = 60
        self.csa_step = 0.05      # first step, as a fraction of the axis span
        self.csa_decay = 0.95     # geometric schedule replacing the paper's Q-table
        self.csa_ms = 15.0        # per-pass wall-clock cap
        self.csa_share = 0.0      # >0 enables; share of the run span CSA may use
        self.csa_where = "stall"  # "stall" | "end" | "both" (see `run`)
        self._csa_spent = 0.0
        self._csa_calls = 0
        self._csa_applied = 0
        if os.environ.get("PARTNER_CSA_REFINE", "0") in (
                "1", "true", "True", "on", "ON"):
            self.csa_share = _env_pos("PARTNER_CSA_SHARE", 0.08, 1.0)
            self.csa_iters = int(_env_pos("PARTNER_CSA_ITERS", 60.0, 100000.0))
            self.csa_step = _env_pos("PARTNER_CSA_STEP", 0.05, 1.0)
            self.csa_decay = _env_pos("PARTNER_CSA_DECAY", 0.95, 1.0)
            self.csa_ms = _env_pos("PARTNER_CSA_MS", 15.0, 1e6)
            w = os.environ.get("PARTNER_CSA_WHERE", "stall")
            self.csa_where = w if w in ("stall", "end", "both") else "stall"

        # -- flat-array numba rung kernel attach ----------------------------
        # Bit-exact transcription of the rung's numeric hot loops -- the
        # `hold=True` axis pass, the `_evict` anchor scan and the overlap
        # tests -- over structure-of-arrays buffers; see
        # partner/refine_numeric_kernel.py.  Default off: `_nk is None` makes
        # every dispatch site a single identity test, so the decision logic
        # AND the rng stream are identical to the pre-port fork.
        if os.environ.get("PARTNER_REFINE_KERNEL", "") == "numba":
            try:
                from refine_numeric_kernel import try_attach as _rk_attach
                from refine_numeric_kernel import disc_parts as _rk_parts
                self._nk = _rk_attach(self)
                if self._nk is not None and self._nk.disc:
                    _parts = _rk_parts()
                    self._disc_fuse = "fuse" in _parts
                    self._disc_hp = "hp" in _parts
            except Exception:
                self._nk = None
                self._disc_fuse = False
                self._disc_hp = False

        # -- PARTNER_REFINE_PROF: `run()` section timers (default off) ------
        # Diagnostic only; `None` makes every site in `run` a dead branch.
        self._prof = _prof_new() if _flag_on("PARTNER_REFINE_PROF") else None
        # second tier: splits the `discrete` bucket (see `_DPROF_KEYS`).
        # Gated under the parent flag -- there is no record to attach to
        # otherwise -- and `None` makes every site in the discrete family a
        # dead branch.
        self._dprof = (_dprof_new()
                       if (self._prof is not None
                           and _flag_on("PARTNER_REFINE_PROF_DISC"))
                       else None)

    # ------------------------------------------------------------------
    def _contacts(self, idxs):
        """Exact touch contacts among idxs on the current layout."""
        P = self.P
        cH, cV, adj = [], [], {i: [] for i in idxs}
        for ai in range(len(idxs)):
            a = idxs[ai]
            ax0, ay0, aw, ah = P[a]
            ax1, ay1 = ax0 + aw, ay0 + ah
            for bi in range(len(idxs)):
                if ai == bi:
                    continue
                b = idxs[bi]
                bx0, by0, bw, bh = P[b]
                oy = min(ay1, by0 + bh) - max(ay0, by0)
                ox = min(ax1, bx0 + bw) - max(ax0, bx0)
                if abs(ax1 - bx0) <= CONTACT_TOL and oy > SEP_TOL:
                    cH.append((a, b))
                    adj[a].append(b)
                    adj[b].append(a)
                elif abs(ay1 - by0) <= CONTACT_TOL and ox > SEP_TOL:
                    cV.append((a, b))
                    adj[a].append(b)
                    adj[b].append(a)
        return cH, cV, adj

    def _build_groups(self):
        n = self.n
        opt = self.opt
        self.group_of = np.full(n, -1, dtype=np.int64)
        self.groups: List[_Group] = []

        for idxs in opt.cluster_groups.values():
            movable = [i for i in idxs if self.kind[i] != 2]
            for i in idxs:
                self.in_cluster[i] = True
            if not movable:
                continue
            g = _Group(movable)
            cH, cV, adj = self._contacts(list(idxs))
            # snap recorded contacts to exact equality right away
            g.cH = [(a, b) for (a, b) in cH if self.kind[b] != 2]
            g.cV = [(a, b) for (a, b) in cV if self.kind[b] != 2]
            self._snap(g)
            locked = [i for i in idxs if self.kind[i] == 2]
            if locked:
                # if the group currently touches its preplaced anchor(s) as
                # one connected component, moving it would break grouping:
                # pin it entirely (conservative but always safe)
                seen = {idxs[0]}
                stack = [idxs[0]]
                while stack:
                    for m in adj[stack.pop()]:
                        if m not in seen:
                            seen.add(m)
                            stack.append(m)
                if len(seen) == len(idxs):
                    g.pin_x = g.pin_y = True
            for i in movable:
                self.group_of[i] = len(self.groups)
            self.groups.append(g)

        for i in range(n):
            if self.kind[i] != 2 and self.group_of[i] < 0:
                self.group_of[i] = len(self.groups)
                self.groups.append(_Group([i]))

    def _build_pins(self):
        """Pin groups whose boundary tags are currently satisfied."""
        P = self.P
        opt = self.opt
        n = self.n
        self.satL = np.zeros(n, dtype=bool)
        self.satR = np.zeros(n, dtype=bool)
        self.satB = np.zeros(n, dtype=bool)
        self.satT = np.zeros(n, dtype=bool)
        self.tag_wanted = []   # (block, bit) unsatisfied tags worth snapping
        for i, code in zip(opt._bnd_idx, opt._bnd_codes):
            i = int(i)
            code = int(code)
            if self.kind[i] == 2:
                continue
            gi = self.group_of[i]
            g = self.groups[gi] if gi >= 0 else None
            x0, y0, w, h = P[i]
            if code & 1:
                if abs(x0 - self.xmin) < EDGE_EPS:
                    self.satL[i] = True
                    if g:
                        g.pin_x = True
                else:
                    self.tag_wanted.append((i, 1))
            if code & 2:
                if abs(x0 + w - self.xmax) < EDGE_EPS:
                    self.satR[i] = True
                    if g:
                        g.pin_x = True
                else:
                    self.tag_wanted.append((i, 2))
            if code & 4:
                if abs(y0 + h - self.ymax) < EDGE_EPS:
                    self.satT[i] = True
                    if g:
                        g.pin_y = True
                else:
                    self.tag_wanted.append((i, 4))
            if code & 8:
                if abs(y0 - self.ymin) < EDGE_EPS:
                    self.satB[i] = True
                    if g:
                        g.pin_y = True
                else:
                    self.tag_wanted.append((i, 8))

    def _build_edges(self):
        """Per-group external connectivity for the weighted-median targets."""
        opt = self.opt
        gof = self.group_of
        if self._fast_build:
            # PARTNER_REFINE_KERNEL: same lists, one pass.  The shipped form
            # re-scans every edge once PER GROUP -- O(G*|E|), measured 0.137 s
            # of a 0.140 s `_Refiner.__init__` at n=116, which is ~3x a whole
            # kernel rung.  Bucketing by group is O(|E| + |P| + G) and appends
            # in the same edge order, so every `g.eM/eJ/eW/pM/pX/pY/pW` array
            # is element-for-element what the loop below produces.
            G = len(self.groups)
            eMs = [[] for _ in range(G)]
            eJs = [[] for _ in range(G)]
            eWs = [[] for _ in range(G)]
            pMs = [[] for _ in range(G)]
            pXs = [[] for _ in range(G)]
            pYs = [[] for _ in range(G)]
            pWs = [[] for _ in range(G)]
            for a, b, w in zip(opt.eI, opt.eJ, opt.eW):
                a, b = int(a), int(b)
                ga, gb = gof[a], gof[b]
                if ga == gb:
                    continue
                if ga >= 0:
                    eMs[ga].append(a); eJs[ga].append(b); eWs[ga].append(w)
                if gb >= 0:
                    eMs[gb].append(b); eJs[gb].append(a); eWs[gb].append(w)
            for b, w, px, py in zip(opt.pB, opt.pW, opt.pX, opt.pY):
                b = int(b)
                gb = gof[b]
                if gb >= 0:
                    pMs[gb].append(b); pXs[gb].append(px)
                    pYs[gb].append(py); pWs[gb].append(w)
            for gi, g in enumerate(self.groups):
                g.eM = np.array(eMs[gi], dtype=np.int64)
                g.eJ = np.array(eJs[gi], dtype=np.int64)
                g.eW = np.array(eWs[gi], dtype=np.float64)
                g.pM = np.array(pMs[gi], dtype=np.int64)
                g.pX = np.array(pXs[gi], dtype=np.float64)
                g.pY = np.array(pYs[gi], dtype=np.float64)
                g.pW = np.array(pWs[gi], dtype=np.float64)
            self._build_badj()
            return
        for gi, g in enumerate(self.groups):
            eM, eJ, eW = [], [], []
            for a, b, w in zip(opt.eI, opt.eJ, opt.eW):
                a, b = int(a), int(b)
                if gof[a] == gi and gof[b] != gi:
                    eM.append(a); eJ.append(b); eW.append(w)
                elif gof[b] == gi and gof[a] != gi:
                    eM.append(b); eJ.append(a); eW.append(w)
            pM, pX, pY, pW = [], [], [], []
            for b, w, px, py in zip(opt.pB, opt.pW, opt.pX, opt.pY):
                if gof[int(b)] == gi:
                    pM.append(int(b)); pX.append(px); pY.append(py); pW.append(w)
            g.eM = np.array(eM, dtype=np.int64)
            g.eJ = np.array(eJ, dtype=np.int64)
            g.eW = np.array(eW, dtype=np.float64)
            g.pM = np.array(pM, dtype=np.int64)
            g.pX = np.array(pX, dtype=np.float64)
            g.pY = np.array(pY, dtype=np.float64)
            g.pW = np.array(pW, dtype=np.float64)

        self._build_badj()

    def _build_badj(self):
        """Per-block incident edges / pins (for reshape deltas).  Shared
        verbatim by both `_build_edges` variants -- already one pass."""
        opt = self.opt
        self.badj = [[] for _ in range(self.n)]
        for a, b, w in zip(opt.eI, opt.eJ, opt.eW):
            a, b = int(a), int(b)
            self.badj[a].append((b, float(w)))
            self.badj[b].append((a, float(w)))
        self.bpin = [[] for _ in range(self.n)]
        for b, w, px, py in zip(opt.pB, opt.pW, opt.pX, opt.pY):
            self.bpin[int(b)].append((float(px), float(py), float(w)))

    def _build_reshape_units(self):
        """Soft blocks that may reshape; MIB groups reshape in sync."""
        opt = self.opt
        used = np.zeros(self.n, dtype=bool)
        self.reshape_units: List[List[int]] = []
        for idxs in opt.mib_groups.values():
            if any(self.kind[i] != 0 or self.in_cluster[i] for i in idxs):
                for i in idxs:
                    used[i] = True
                continue
            self.reshape_units.append(list(idxs))
            for i in idxs:
                used[i] = True
        for i in range(self.n):
            if not used[i] and self.kind[i] == 0 and not self.in_cluster[i]:
                self.reshape_units.append([i])

    # ------------------------------------------------------------------
    def _snap(self, g: _Group):
        """Restore exact float abutment of recorded cluster contacts."""
        P = self.P
        if g.cH:
            for a, b in sorted(g.cH, key=lambda ab: P[ab[0], 0]):
                P[b, 0] = P[a, 0] + P[a, 2]
        if g.cV:
            for a, b in sorted(g.cV, key=lambda ab: P[ab[0], 1]):
                P[b, 1] = P[a, 1] + P[a, 3]

    def _interval(self, g: _Group, axis: int):
        """Feasible rigid-shift interval [lo, hi] for the group along axis,
        holding every other block fixed at its current position."""
        P = self.P
        mem = g.members
        a = axis          # moving axis: 0 = x, 1 = y
        o = 1 - axis      # frozen axis
        c0 = P[:, a]
        c1 = c0 + P[:, a + 2]
        f0 = P[:, o]
        f1 = f0 + P[:, o + 2]
        if axis == 0:
            lo = self.xmin - float(c0[mem].min())
            hi = self.xmax - float(c1[mem].max())
        else:
            lo = self.ymin - float(c0[mem].min())
            hi = self.ymax - float(c1[mem].max())
        inmem = np.zeros(self.n, dtype=bool)
        inmem[mem] = True
        for m in mem:
            ov = np.minimum(f1[m], f1) - np.maximum(f0[m], f0)
            cand = (ov > SEP_TOL) & ~inmem
            if not cand.any():
                continue
            idx = np.nonzero(cand)[0]
            cm = c0[m] + 0.5 * P[m, a + 2]
            oc = c0[idx] + 0.5 * P[idx, a + 2]
            pred = idx[oc <= cm]
            succ = idx[oc > cm]
            if len(pred):
                lo = max(lo, float(c1[pred].max()) - float(c0[m]))
            if len(succ):
                hi = min(hi, float(c0[succ].min()) - float(c1[m]))
        return lo, hi

    def _move(self, g: _Group, d: float, axis: int):
        P = self.P
        P[g.members, axis] += d
        if axis == 0:
            if g.cH:
                for a, b in sorted(g.cH, key=lambda ab: P[ab[0], 0]):
                    P[b, 0] = P[a, 0] + P[a, 2]
        else:
            if g.cV:
                for a, b in sorted(g.cV, key=lambda ab: P[ab[0], 1]):
                    P[b, 1] = P[a, 1] + P[a, 3]

    @staticmethod
    def _wmedian(vals: np.ndarray, wts: np.ndarray) -> float:
        order = np.argsort(vals)
        v = vals[order]
        c = np.cumsum(wts[order])
        half = 0.5 * c[-1]
        return float(v[np.searchsorted(c, half)])

    def _median_shift(self, g: _Group, axis: int) -> Optional[float]:
        """Desired rigid shift of the group along axis: weighted median of
        the per-edge/per-pin optimal offsets (exact 1-D HPWL optimum)."""
        if not len(g.eM) and not len(g.pM):
            return None
        P = self.P
        cm = P[g.eM, axis] + 0.5 * P[g.eM, axis + 2]
        cj = P[g.eJ, axis] + 0.5 * P[g.eJ, axis + 2]
        des = cj - cm
        if len(g.pM):
            pc = P[g.pM, axis] + 0.5 * P[g.pM, axis + 2]
            pt = (g.pX if axis == 0 else g.pY) - pc
            des = np.concatenate([des, pt])
            wts = np.concatenate([g.eW, g.pW])
        else:
            wts = g.eW
        return self._wmedian(des, wts)

    def _axis_constraints(self, axis: int, invert: bool = False):
        """Build the separation polytope for one axis (extracted verbatim from
        `_axis_pass` so the CSA pass optimizes over the SAME feasible set --
        any divergence here would be a legality hole)."""
        P = self.P
        n = self.n
        o = 1 - axis
        c0 = P[:, axis]
        c1 = c0 + P[:, axis + 2]
        f0 = P[:, o]
        f1 = f0 + P[:, o + 2]

        # block pairs that must stay separated along `axis`: the frozen-axis
        # projections overlap, and (for pairs currently overlapping in both
        # axes, which only happens mid-legalization) this axis is picked by
        # movability first — an axis where both endpoints are hard-pinned
        # can never separate the pair — then by smaller penetration
        # (`invert` flips that preference to shake deadlocks)
        ovf = (np.minimum(f1[:, None], f1[None, :])
               - np.maximum(f0[:, None], f0[None, :]))
        ovm = (np.minimum(c1[:, None], c1[None, :])
               - np.maximum(c0[:, None], c0[None, :]))
        both = (ovf > SEP_TOL) & (ovm > SEP_TOL)
        pin_this = self.kind == 2
        pin_oth = self.kind == 2
        pt = np.zeros(n, dtype=bool)
        po = np.zeros(n, dtype=bool)
        for g in self.groups:
            if (g.pin_x if axis == 0 else g.pin_y):
                pt[g.members] = True
            if (g.pin_y if axis == 0 else g.pin_x):
                po[g.members] = True
        pin_this = pin_this | pt
        pin_oth = pin_oth | po
        can_this = ~(pin_this[:, None] & pin_this[None, :])
        can_oth = ~(pin_oth[:, None] & pin_oth[None, :])
        pref = (ovm >= ovf) if invert else (ovm <= ovf)
        ov = ((ovf > SEP_TOL) & (ovm <= SEP_TOL)) \
            | (both & can_this & (~can_oth | pref))
        np.fill_diagonal(ov, False)
        cen = c0 + 0.5 * P[:, axis + 2]
        ridx = np.arange(n)
        left = (cen[:, None] < cen[None, :]) \
            | ((cen[:, None] == cen[None, :]) & (ridx[:, None] < ridx[None, :]))
        pairs = np.nonzero(ov & left)

        G = len(self.groups)
        gof = self.group_of
        NEG = -1e18
        POS = 1e18
        lo_fixed = np.full(G, NEG)   # bounds from frame + locked blocks
        hi_fixed = np.full(G, POS)
        edges_in: List[List] = [[] for _ in range(G)]   # (pred_group, c)
        edges_out: List[List] = [[] for _ in range(G)]  # (succ_group, c)
        tight: dict = {}
        for i, j in zip(*pairs):
            gi_, gj_ = gof[i], gof[j]
            if gi_ == gj_ and gi_ >= 0:
                continue
            c = float(c1[i] - c0[j])        # need d_j - d_i >= c  (c <= ~0)
            if gi_ >= 0 and gj_ >= 0:
                key = (gi_, gj_)
                if key not in tight or c > tight[key]:
                    tight[key] = c
            elif gj_ >= 0:                  # locked i: d_j >= c
                if c > lo_fixed[gj_]:
                    lo_fixed[gj_] = c
            elif gi_ >= 0:                  # locked j: d_i <= -c
                if -c < hi_fixed[gi_]:
                    hi_fixed[gi_] = -c
        for (gi_, gj_), c in tight.items():
            edges_in[gj_].append((gi_, c))
            edges_out[gi_].append((gj_, c))

        lim_lo = self.xmin if axis == 0 else self.ymin
        lim_hi = self.xmax if axis == 0 else self.ymax
        key_of = np.empty(G)
        for gi_, g in enumerate(self.groups):
            mem = g.members
            lo_fixed[gi_] = max(lo_fixed[gi_], lim_lo - float(c0[mem].min()))
            hi_fixed[gi_] = min(hi_fixed[gi_], lim_hi - float(c1[mem].max()))
            key_of[gi_] = float(c0[mem].min())
        pinned = np.array([(g.pin_x if axis == 0 else g.pin_y)
                           for g in self.groups])
        lo_fixed[pinned] = 0.0
        hi_fixed[pinned] = 0.0

        order = np.argsort(key_of, kind="stable")

        # backward: dmax_g accounting for downstream groups also shifting
        dmax = hi_fixed.copy()
        for gi_ in order[::-1]:
            for gj_, c in edges_out[gi_]:
                v = dmax[gj_] - c
                if v < dmax[gi_]:
                    dmax[gi_] = v
        return _AxisCons(len(self.groups), lo_fixed, dmax, edges_in,
                         edges_out, order, pinned)

    def _axis_pass(self, axis: int, max_step: Optional[float] = None,
                   hold: bool = False, invert: bool = False):
        """Compaction-style global sweep: separation constraints between
        groups become difference constraints on rigid shifts d_g.  A
        backward longest-path pass yields each group's maximum feasible
        shift given that everything downstream may also move; the forward
        pass then assigns d_g = clip(median target, assigned-predecessor
        bound, dmax_g).  Whole zero-gap chains can therefore translate
        together — the freedom a purely local sweep never sees.

        hold=True turns the sweep into a minimum-displacement legalizer:
        every target is "stay put" and overlapping pairs are separated
        along their axis of smaller penetration."""
        if self._nk is not None:
            # PARTNER_REFINE_KERNEL: constraint build + forward assignment +
            # rigid moves fused into one njit call.
            if hold:
                self._nk.axis_pass_hold(axis, invert)
                return
            if self._nk.qsweep:
                # PARTNER_REFINE_KERNEL_QSWEEP (sub-switch, default off): the
                # same fusion for the HPWL median sweep.  Off -> this is one
                # extra attribute test and the sweep stays in Python, exactly
                # as it shipped.
                self._nk.axis_pass_soft(axis, max_step, invert)
                return
        cons = self._axis_constraints(axis, invert)
        G = cons.G
        lo_fixed = cons.lo
        dmax = cons.dmax
        edges_in = cons.edges_in
        edges_out = cons.edges_out
        order = cons.order
        pinned = cons.pinned

        # forward: assign shifts toward each group's weighted-median target
        d = np.zeros(G)
        assigned = np.zeros(G, dtype=bool)
        for gi_ in order:
            g = self.groups[gi_]
            if pinned[gi_]:
                assigned[gi_] = True
                continue
            lo = lo_fixed[gi_]
            for gp_, c in edges_in[gi_]:
                base = d[gp_] if assigned[gp_] else 0.0
                if base + c > lo:
                    lo = base + c
            hi = dmax[gi_]
            for gj_, c in edges_out[gi_]:
                if assigned[gj_] and d[gj_] - c < hi:
                    hi = d[gj_] - c
            if lo > hi:
                assigned[gi_] = True
                continue
            if hold:
                t = 0.0
            else:
                t = self._median_shift(g, axis)
                if t is None:
                    t = 0.0
                if max_step is not None:
                    t = min(max(t, -max_step), max_step)
            d[gi_] = min(max(t, lo), hi)
            assigned[gi_] = True
            if abs(d[gi_]) > 1e-12:
                self._move(g, d[gi_], axis)

    def _csa_problem(self, axis: int, cons: "_AxisCons"):
        """Axis HPWL as a function of the group shift vector + its polytope.

        Same-group edges are dropped (a rigid shift leaves them invariant),
        preplaced blocks map to a sentinel slot that is pinned at 0, so the
        objective is exactly `opt._hpwl`'s `axis` half up to a constant."""
        opt = self.opt
        P = self.P
        G = len(self.groups)
        gof = self.group_of
        cen = P[:, axis] + 0.5 * P[:, axis + 2]
        ei = np.asarray(opt.eI, dtype=np.int64)
        ej = np.asarray(opt.eJ, dtype=np.int64)
        if len(ei):
            gu = gof[ei]
            gv = gof[ej]
            keep = gu != gv
            eu = np.where(gu >= 0, gu, G)[keep]
            ev = np.where(gv >= 0, gv, G)[keep]
            eb = (cen[ei] - cen[ej])[keep]
            ew = np.asarray(opt.eW, dtype=np.float64)[keep]
        else:
            eu = ev = np.zeros(0, dtype=np.int64)
            eb = ew = np.zeros(0)
        pb_i = np.asarray(opt.pB, dtype=np.int64)
        if len(pb_i):
            gp = gof[pb_i]
            pu = np.where(gp >= 0, gp, G)
            tgt = np.asarray(opt.pX if axis == 0 else opt.pY, dtype=np.float64)
            pb = cen[pb_i] - tgt
            pw = np.asarray(opt.pW, dtype=np.float64)
        else:
            pu = np.zeros(0, dtype=np.int64)
            pb = pw = np.zeros(0)
        obj = _csa.AxisHpwlObjective(G, eu, ev, eb, ew, pu, pb, pw)
        poly = _csa.ShiftPolytope(G, cons.lo, cons.dmax, cons.edges_in,
                                  cons.edges_out, cons.order, cons.pinned)
        return obj, poly

    def _csa_pass(self, axis: int, key0: Optional[float] = None) -> bool:
        """Joint conjugate-subgradient polish of the group shifts on `axis`.

        Monotone by construction: the shift vector is a point of the same
        polytope the median sweep walks, and the whole pass is reverted
        unless the layout is still overlap-free AND the full proxy strictly
        improves.  Returns True iff the move was kept."""
        t_start = time.time()
        self._csa_calls += 1
        try:
            cons = self._axis_constraints(axis)
            obj, poly = self._csa_problem(axis, cons)
            if obj.terms == 0 or cons.G == 0 or bool(cons.pinned.all()):
                return False
            span = ((self.xmax - self.xmin) if axis == 0
                    else (self.ymax - self.ymin))
            step = self.csa_step * span
            if not (step > 0.0):
                return False
            d, f, f0, _it = _csa.csa_shifts(
                obj, poly, step, self.csa_iters, self.csa_decay,
                t_start + self.csa_ms * 1e-3)
            if f >= f0 - 1e-12:
                return False
            if key0 is None:
                key0 = self._key()
            snap = self.P.copy()
            for gi_ in cons.order:
                dv = float(d[gi_])
                if abs(dv) > 1e-12:
                    self._move(self.groups[gi_], dv, axis)
            if self._has_overlap() or self._key() >= key0 - 1e-12:
                self.P[...] = snap
                return False
            self._csa_applied += 1
            return True
        except Exception:
            if _DEBUG:
                import traceback
                traceback.print_exc()
            return False
        finally:
            self._csa_spent += time.time() - t_start

    def _overlap_count(self) -> int:
        if self._nk is not None:
            return self._nk.overlap_count()
        P = self.P
        x0 = P[:, 0]
        x1 = x0 + P[:, 2]
        y0 = P[:, 1]
        y1 = y0 + P[:, 3]
        ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
        oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
        m = (ox > 9e-7) & (oy > 9e-7)
        np.fill_diagonal(m, False)
        return int(m.sum()) // 2

    def _has_overlap(self) -> bool:
        if self._nk is not None:
            return self._nk.has_overlap()
        P = self.P
        x0 = P[:, 0]
        x1 = x0 + P[:, 2]
        y0 = P[:, 1]
        y1 = y0 + P[:, 3]
        ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
        oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
        m = (ox > 9e-7) & (oy > 9e-7)
        np.fill_diagonal(m, False)
        return bool(m.any())

    # ------------------------------------------------------------------
    def _block_hp(self, i: int, cx: float, cy: float, exclude=()) -> float:
        """HPWL of edges incident to block i with its center at (cx, cy)."""
        if self._disc_hp:
            # PARTNER_REFINE_KERNEL_DISC (part `hp`): same CSR order, same
            # accumulation order, so the float sum is bit-identical.
            return self._nk.block_hp(i, cx, cy, exclude)
        P = self.P
        s = 0.0
        for j, w in self.badj[i]:
            if j in exclude:
                continue
            s += w * (abs(cx - (P[j, 0] + 0.5 * P[j, 2]))
                      + abs(cy - (P[j, 1] + 0.5 * P[j, 3])))
        for px, py, w in self.bpin[i]:
            s += w * (abs(cx - px) + abs(cy - py))
        return s

    def _reshape_pass(self, deadline: float):
        P = self.P
        opt = self.opt
        for unit in self.reshape_units:
            if time.time() >= deadline:
                return
            i0 = unit[0]
            w0 = float(P[i0, 2])
            if any(abs(float(P[i, 2]) - w0) > 1e-9 for i in unit[1:]):
                continue
            area = float(opt.areas[i0])
            best = None
            for s in (0.82, 0.90, 1.11, 1.22):
                nw = w0 * s
                nh = area / nw
                cand = []
                ok = True
                for i in unit:
                    x0, y0, w, h = P[i]
                    if self.satL[i] and self.satR[i]:
                        ok = False
                        break
                    if self.satL[i]:
                        nx = x0
                    elif self.satR[i]:
                        nx = x0 + w - nw
                    else:
                        nx = x0 + 0.5 * (w - nw)
                    nx = min(max(nx, self.xmin), self.xmax - nw)
                    if self.satB[i] and self.satT[i]:
                        ok = False
                        break
                    if self.satB[i]:
                        ny = y0
                    elif self.satT[i]:
                        ny = y0 + h - nh
                    else:
                        ny = y0 + 0.5 * (h - nh)
                    ny = min(max(ny, self.ymin), self.ymax - nh)
                    if nx < self.xmin - 1e-9 or nx + nw > self.xmax + 1e-9 \
                            or ny < self.ymin - 1e-9 or ny + nh > self.ymax + 1e-9:
                        ok = False
                        break
                    cand.append((i, nx, ny))
                if not ok:
                    continue
                # overlap check: candidate rects vs everything else
                for ci, (i, nx, ny) in enumerate(cand):
                    ox = np.minimum(nx + nw, P[:, 0] + P[:, 2]) - np.maximum(nx, P[:, 0])
                    oy = np.minimum(ny + nh, P[:, 1] + P[:, 3]) - np.maximum(ny, P[:, 1])
                    clash = (ox > SEP_TOL) & (oy > SEP_TOL)
                    clash[i] = False
                    for (i2, nx2, ny2) in cand[:ci]:
                        clash[i2] = (min(nx + nw, nx2 + nw) - max(nx, nx2) > SEP_TOL
                                     and min(ny + nh, ny2 + nh) - max(ny, ny2) > SEP_TOL)
                    if clash.any():
                        ok = False
                        break
                if not ok:
                    continue
                excl = set(unit)
                delta = 0.0
                for (i, nx, ny) in cand:
                    delta += self._block_hp(i, nx + 0.5 * nw, ny + 0.5 * nh, excl)
                    delta -= self._block_hp(i, P[i, 0] + 0.5 * P[i, 2],
                                            P[i, 1] + 0.5 * P[i, 3], excl)
                # member-to-member edges (both endpoints move)
                for ai in range(len(cand)):
                    i, nx, ny = cand[ai]
                    for j, w in self.badj[i]:
                        if j not in excl:
                            continue
                        for (j2, nx2, ny2) in cand:
                            if j2 == j:
                                delta += w * (abs((nx + 0.5 * nw) - (nx2 + 0.5 * nw))
                                              + abs((ny + 0.5 * nh) - (ny2 + 0.5 * nh)))
                                delta -= w * (abs((P[i, 0] + 0.5 * P[i, 2]) - (P[j, 0] + 0.5 * P[j, 2]))
                                              + abs((P[i, 1] + 0.5 * P[i, 3]) - (P[j, 1] + 0.5 * P[j, 3])))
                                break
                if delta < -1e-9 and (best is None or delta < best[0]):
                    best = (delta, nw, nh, list(cand))
            if best is not None:
                _d, nw, nh, cand = best
                for (i, nx, ny) in cand:
                    P[i, 0] = nx
                    P[i, 1] = ny
                    P[i, 2] = nw
                    P[i, 3] = nh

    # ------------------------------------------------------------------
    def _tag_snap(self):
        """Try to earn unsatisfied boundary tags by sliding singleton blocks
        flush to the required edge (kept only if the full proxy improves)."""
        P = self.P
        cur = self._key()
        for (i, bit) in self.tag_wanted:
            gi = self.group_of[i]
            if gi < 0:
                continue
            g = self.groups[gi]
            if len(g.members) != 1:
                continue
            axis = 0 if bit in (1, 2) else 1
            if (g.pin_x if axis == 0 else g.pin_y):
                continue
            if bit == 1:
                d = self.xmin - P[i, 0]
            elif bit == 2:
                d = self.xmax - (P[i, 0] + P[i, 2])
            elif bit == 4:
                d = self.ymax - (P[i, 1] + P[i, 3])
            else:
                d = self.ymin - P[i, 1]
            if abs(d) < EDGE_EPS:
                continue
            lo, hi = self._interval(g, axis)
            if not (lo - 1e-12 <= d <= hi + 1e-12):
                continue
            old = P[i, axis]
            P[i, axis] += d
            k = self._key()
            if k < cur - 1e-12:
                cur = k
                if bit == 1:
                    self.satL[i] = True
                    g.pin_x = True
                elif bit == 2:
                    self.satR[i] = True
                    g.pin_x = True
                elif bit == 4:
                    self.satT[i] = True
                    g.pin_y = True
                else:
                    self.satB[i] = True
                    g.pin_y = True
            else:
                P[i, axis] = old

    def _perturb(self):
        """Kick a few random groups to random feasible offsets to escape
        the deterministic fixed point of the median sweeps."""
        cands = [g for g in self.groups
                 if (not g.pin_x or not g.pin_y) and (len(g.eM) or len(g.pM))]
        if not cands:
            return
        for _ in range(min(3, len(cands))):
            g = self.rng.choice(cands)
            axis = 0 if (not g.pin_x and (g.pin_y or self.rng.random() < 0.5)) else 1
            if axis == 1 and g.pin_y:
                continue
            lo, hi = self._interval(g, axis)
            if hi - lo < 1e-9:
                continue
            self._move(g, self.rng.uniform(lo, hi), axis)

    # ------------------------------------------------------------------
    def _key(self):
        P = self.P
        opt = self.opt
        hp = opt._hpwl(P)
        area = ((P[:, 0] + P[:, 2]).max() - P[:, 0].min()) \
            * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min())
        V = opt._violations(P)
        return (1.0 + 0.5 * ((hp / self.hp0 - 1.0)
                             + max(0.0, area / opt.area_ref - 1.0))) \
            * math.exp(2.0 * opt.v_weight * V / opt.n_soft_den)

    # ------------------------------------------------------------------
    # Discrete topology moves.  Continuous sweeps cannot reorder a
    # saturated packing (every maximal chain sums exactly to the frame), so
    # real HPWL gains need discrete slot exchanges: two soft blocks swap
    # rectangles (each reshapes to the other's width, area exact), or a
    # block is re-inserted beside its strongest-pull neighbor.  Each move is
    # followed by minimum-displacement legalization and kept only if the
    # full proxy cost improves — the classic global-swap step of detailed
    # placement, adapted to free-form rects with exact-area reshaping.
    # ------------------------------------------------------------------
    def _build_swappable(self):
        """Blocks free to change slot: movable, not cluster-bound (exact
        contacts), not MIB (shape sync), and not pinned by a satisfied tag."""
        n = self.n
        mib = np.array(self.opt.mib[:n], dtype=np.int64) > 0
        pinned = np.zeros(n, dtype=bool)
        for i in range(n):
            gi = self.group_of[i]
            if gi >= 0 and (self.groups[gi].pin_x or self.groups[gi].pin_y):
                pinned[i] = True
        self.swappable = (self.kind != 2) & ~self.in_cluster & ~mib & ~pinned

    def _optimal_point(self, i: int):
        """Weighted-median optimum of block i's incident edges/pins."""
        if self._disc_hp:
            # PARTNER_REFINE_KERNEL_DISC (part `hp`).  Inherits `_wmedian_k`'s
            # stable-vs-introsort caveat -- the same one QSWEEP already ships
            # under, argued in refine_numeric_kernel.py's docstring and
            # measured in tests/test_partner_refine_disc.py.
            return self._nk.optimal_point(i)
        P = self.P
        xs, ys, ws = [], [], []
        for j, w in self.badj[i]:
            xs.append(P[j, 0] + 0.5 * P[j, 2])
            ys.append(P[j, 1] + 0.5 * P[j, 3])
            ws.append(w)
        for px, py, w in self.bpin[i]:
            xs.append(px)
            ys.append(py)
            ws.append(w)
        if not xs:
            return None
        wa = np.asarray(ws)
        return (self._wmedian(np.asarray(xs), wa),
                self._wmedian(np.asarray(ys), wa))

    def _legal_check(self, snap: np.ndarray, cur_key: float):
        """Legalize after a discrete edit; accept iff the proxy improves."""
        dp = self._dprof          # PARTNER_REFINE_PROF_DISC; None -> dead
        if self._disc_fuse:
            # PARTNER_REFINE_KERNEL_DISC (part `fuse`): the loop below plus
            # its trailing re-test, in one njit call.  Same functions, same
            # order -- what it removes is 6 numba dispatches and 6 O(G) Python
            # pin re-syncs per attempt, which the sub-profile prices at ~26%
            # of the whole `discrete` bucket.
            _t = time.time() if dp is not None else 0.0
            _ov = self._nk.legal_sweeps()
            if dp is not None:
                dp["t_lc_axis"] += time.time() - _t
                dp["c_lc_axis"] += 6
            if _ov:
                self.P[...] = snap
                return False, cur_key
            _t = time.time() if dp is not None else 0.0
            k = self._key()
            if dp is not None:
                dp["t_lc_key"] += time.time() - _t
                dp["c_lc_key"] += 1
            if k < cur_key - 1e-12:
                return True, k
            self.P[...] = snap
            return False, cur_key
        if dp is None:
            # Uninstrumented path, verbatim as it shipped.  This IS a second
            # copy of the loop below, and that is deliberate: `_legal_check`
            # runs ~7 k times in a 0.23 s `run`, so even the ~10 `dp is not
            # None` tests the instrumented copy pays would be ~1% of the
            # span -- and on a deadline-bounded search a 1% tax is not a
            # measurement, it is a policy change.
            for _ in range(3):
                self._axis_pass(0, hold=True)
                self._axis_pass(1, hold=True)
                if not self._has_overlap():
                    break
            if self._has_overlap():
                self.P[...] = snap
                return False, cur_key
            k = self._key()
            if k < cur_key - 1e-12:
                return True, k
            self.P[...] = snap
            return False, cur_key
        for _ in range(3):
            _t = time.time() if dp is not None else 0.0
            self._axis_pass(0, hold=True)
            self._axis_pass(1, hold=True)
            if dp is not None:
                dp["t_lc_axis"] += time.time() - _t
                dp["c_lc_axis"] += 2
                _t = time.time()
            _ov = self._has_overlap()
            if dp is not None:
                dp["t_lc_ovl"] += time.time() - _t
                dp["c_lc_ovl"] += 1
            if not _ov:
                break
        _t = time.time() if dp is not None else 0.0
        _ov = self._has_overlap()
        if dp is not None:
            dp["t_lc_ovl"] += time.time() - _t
            dp["c_lc_ovl"] += 1
        if _ov:
            _t = time.time() if dp is not None else 0.0
            self.P[...] = snap
            if dp is not None:
                dp["t_lc_rest"] += time.time() - _t
                dp["c_lc_rest"] += 1
            return False, cur_key
        _t = time.time() if dp is not None else 0.0
        k = self._key()
        if dp is not None:
            dp["t_lc_key"] += time.time() - _t
            dp["c_lc_key"] += 1
        if k < cur_key - 1e-12:
            return True, k
        _t = time.time() if dp is not None else 0.0
        self.P[...] = snap
        if dp is not None:
            dp["t_lc_rest"] += time.time() - _t
            dp["c_lc_rest"] += 1
        return False, cur_key

    def _try_swap(self, i: int, j: int, cur_key: float):
        dp = self._dprof          # PARTNER_REFINE_PROF_DISC; None -> dead
        _t = time.time() if dp is not None else 0.0
        P = self.P
        snap = P.copy()
        xi, yi, wi, hi = P[i]
        xj, yj, wj, hj = P[j]
        ai = float(self.opt.areas[i])
        aj = float(self.opt.areas[j])
        if self.kind[i] == 0:
            nh = ai / wj
            P[i] = (xj, min(yj, self.ymax - nh), wj, nh)
        else:
            P[i] = (min(max(xj + 0.5 * (wj - wi), self.xmin), self.xmax - wi),
                    min(max(yj + 0.5 * (hj - hi), self.ymin), self.ymax - hi),
                    wi, hi)
        if self.kind[j] == 0:
            nh = aj / wi
            P[j] = (xi, min(yi, self.ymax - nh), wi, nh)
        else:
            P[j] = (min(max(xi + 0.5 * (wi - wj), self.xmin), self.xmax - wj),
                    min(max(yi + 0.5 * (hi - hj), self.ymin), self.ymax - hj),
                    wj, hj)
        if dp is not None:
            dp["t_swap_edit"] += time.time() - _t
            dp["c_swap_edit"] += 1
        return self._legal_check(snap, cur_key)

    def _try_insert(self, i: int, cur_key: float):
        """Re-insert block i directly above/below its strongest neighbor."""
        if not self.badj[i]:
            return False, cur_key
        dp = self._dprof          # PARTNER_REFINE_PROF_DISC; None -> dead
        _t = time.time() if dp is not None else 0.0
        P = self.P
        j, _w = max(self.badj[i], key=lambda t: t[1])
        xj, yj, wj, hj = P[j]
        ai = float(self.opt.areas[i])
        if self.kind[i] == 0:
            nw = wj
            nh = ai / nw
            nx = xj
        else:
            nw, nh = P[i, 2], P[i, 3]
            nx = min(max(xj + 0.5 * (wj - nw), self.xmin), self.xmax - nw)
        for ny in (yj + hj, yj - nh):
            ny = min(max(ny, self.ymin), self.ymax - nh)
            snap = P.copy()
            P[i] = (nx, ny, nw, nh)
            if dp is not None:
                dp["t_insert_edit"] += time.time() - _t
                dp["c_insert_edit"] += 1
            ok, cur_key = self._legal_check(snap, cur_key)
            if ok:
                return True, cur_key
            _t = time.time() if dp is not None else 0.0
        return False, cur_key

    def _discrete_batch(self, cur_key: float, deadline: float,
                        max_cands: int = 48):
        """One sweep of HPWL-driven swap/insert attempts over the most
        displaced blocks.  Returns (accepted_count, new_key)."""
        P = self.P
        dp = self._dprof          # PARTNER_REFINE_PROF_DISC; None -> dead
        idxs = np.nonzero(self.swappable)[0]
        if len(idxs) < 2:
            return 0, cur_key
        cx = P[:, 0] + 0.5 * P[:, 2]
        cy = P[:, 1] + 0.5 * P[:, 3]
        areas = np.array(self.opt.areas)
        cands = []
        _t = time.time() if dp is not None else 0.0
        if self._disc_hp:
            # PARTNER_REFINE_KERNEL_DISC (part `hp`): the whole per-block
            # `_optimal_point` + two `_block_hp` + `gain > 1e-9` scan in one
            # njit call.  Entry-for-entry identical to the loop below, so the
            # `sort(reverse=True)` that follows sees the same list.
            keep, gains, opx, opy = self._nk.discrete_gains(idxs)
            for t in range(len(idxs)):
                if keep[t]:
                    cands.append((float(gains[t]), int(idxs[t]),
                                  float(opx[t]), float(opy[t])))
            if dp is not None:
                _e = time.time()
                dp["t_enum_gain"] += _e - _t
                dp["c_enum_gain"] += 2 * len(idxs)
                _t = _e
            cands.sort(reverse=True)
            if dp is not None:
                _e = time.time()
                dp["t_enum_rest"] += _e - _t
                dp["c_enum_rest"] += 1
                _t = _e
            return self._discrete_apply(cands, idxs, cx, cy, areas,
                                        cur_key, deadline, max_cands)
        for i in idxs:
            o = self._optimal_point(int(i))
            if dp is not None:
                _e = time.time()
                dp["t_enum_optpt"] += _e - _t
                dp["c_enum_optpt"] += 1
                _t = _e
            if o is None:
                continue
            gain = self._block_hp(int(i), cx[i], cy[i]) \
                - self._block_hp(int(i), o[0], o[1])
            if dp is not None:
                _e = time.time()
                dp["t_enum_gain"] += _e - _t
                dp["c_enum_gain"] += 2
                _t = _e
            if gain > 1e-9:
                cands.append((gain, int(i), o[0], o[1]))
        cands.sort(reverse=True)
        if dp is not None:
            _e = time.time()
            dp["t_enum_rest"] += _e - _t
            dp["c_enum_rest"] += 1
            _t = _e
        return self._discrete_apply(cands, idxs, cx, cy, areas,
                                    cur_key, deadline, max_cands)

    def _discrete_apply(self, cands, idxs, cx, cy, areas,
                        cur_key: float, deadline: float, max_cands: int):
        """The accept half of `_discrete_batch`, extracted verbatim so the
        Python and the DISC-kernel enumerations share ONE copy of it (a second
        copy is exactly how a `==` contract rots)."""
        P = self.P
        dp = self._dprof          # PARTNER_REFINE_PROF_DISC; None -> dead
        accepted = 0
        for gain, i, ox, oy in cands[:max_cands]:
            if time.time() >= deadline:
                break
            _t = time.time() if dp is not None else 0.0
            dist = np.abs(cx[idxs] - ox) + np.abs(cy[idxs] - oy)
            ratio = areas[idxs] / max(areas[i], 1e-9)
            bad = (idxs == i) | (ratio < 0.45) | (ratio > 2.2)
            order = np.argsort(dist + 1e18 * bad)
            if dp is not None:
                dp["t_screen_np"] += time.time() - _t
                dp["c_screen_np"] += 1
            moved = False
            for t in order[:6]:
                if bad[t] or time.time() >= deadline:
                    break
                _t = time.time() if dp is not None else 0.0
                j = int(idxs[t])
                if self._disc_hp:
                    # DISC part `hp`: the same four terms in the same order,
                    # one njit call instead of four Python ones.
                    delta = self._nk.swap_delta(i, j, cx[i], cy[i],
                                                cx[j], cy[j])
                else:
                    excl = (i, j)
                    delta = (self._block_hp(i, cx[j], cy[j], excl)
                             + self._block_hp(j, cx[i], cy[i], excl)
                             - self._block_hp(i, cx[i], cy[i], excl)
                             - self._block_hp(j, cx[j], cy[j], excl))
                if dp is not None:
                    dp["t_screen_delta"] += time.time() - _t
                    dp["c_screen_delta"] += 4
                if delta > -1e-9:
                    continue
                ok, cur_key = self._try_swap(i, j, cur_key)
                if ok:
                    moved = True
                    break
            if not moved and gain > 1e-6 and time.time() < deadline:
                moved, cur_key = self._try_insert(i, cur_key)
            if moved:
                accepted += 1
                _t = time.time() if dp is not None else 0.0
                cx = P[:, 0] + 0.5 * P[:, 2]
                cy = P[:, 1] + 0.5 * P[:, 3]
                if dp is not None:
                    dp["t_book"] += time.time() - _t
                    dp["c_book"] += 1
        return accepted, cur_key

    def _matching_batch(self, cur_key: float, deadline: float,
                        pools: int = 3, K: int = 8):
        """Independent-set matching — the half of FastPlace-DP that the
        pairwise batch lacks.  Build a small pool of size-compatible,
        displaced blocks, solve the EXACT assignment of blocks to the
        pool's position slots (bitmask DP over <=8 blocks), and apply the
        best permutation under the usual _legal_check.  Unlocks the 3+-
        cycles that ratio-capped pairwise swaps cannot express."""
        P = self.P
        dpf = self._dprof         # PARTNER_REFINE_PROF_DISC; None -> dead
        idxs = np.nonzero(self.swappable)[0]
        if len(idxs) < 3:
            return 0, cur_key
        cx = P[:, 0] + 0.5 * P[:, 2]
        cy = P[:, 1] + 0.5 * P[:, 3]
        areas = np.asarray(self.opt.areas)
        gains = []
        _t = time.time() if dpf is not None else 0.0
        for i in idxs:
            o = self._optimal_point(int(i))
            if o is None:
                continue
            g = self._block_hp(int(i), cx[i], cy[i]) \
                - self._block_hp(int(i), o[0], o[1])
            if g > 1e-9:
                gains.append((g, int(i)))
        gains.sort(reverse=True)
        gmap = {i: g for g, i in gains}
        if dpf is not None:
            _e = time.time()
            dpf["t_m_enum"] += _e - _t
            dpf["c_m_enum"] += 1
            _t = _e
        used: set = set()
        accepted = 0
        for _g0, seedb in gains[:pools * 3]:
            if time.time() >= deadline:
                break
            if seedb in used or accepted >= pools:
                continue
            _t = time.time() if dpf is not None else 0.0
            ratio = areas[idxs] / max(areas[seedb], 1e-9)
            peers = [int(j) for j, rr in zip(idxs, ratio)
                     if 0.4 <= rr <= 2.5 and int(j) not in used]
            peers.sort(key=lambda j: -gmap.get(j, 0.0))
            pool = peers[:K]
            if seedb not in pool:
                pool = [seedb] + pool[:K - 1]
            if len(pool) < 3:
                continue
            m = len(pool)
            slots = [(float(P[j, 0]), float(P[j, 1]),
                      float(P[j, 2]), float(P[j, 3])) for j in pool]
            excl = tuple(pool)
            if dpf is not None:
                _e = time.time()
                dpf["t_m_rest"] += _e - _t
                dpf["c_m_rest"] += 1
                _t = _e
            C = np.zeros((m, m))
            for a, i in enumerate(pool):
                for b in range(m):
                    sx, sy, sw, sh = slots[b]
                    if self.kind[i] == 0:
                        ww = sw
                        hh = float(self.opt.areas[i]) / max(sw, 1e-9)
                    else:
                        ww, hh = float(P[i, 2]), float(P[i, 3])
                    C[a, b] = self._block_hp(i, sx + 0.5 * ww,
                                             sy + 0.5 * hh, excl)
            if dpf is not None:
                _e = time.time()
                dpf["t_m_cost"] += _e - _t
                dpf["c_m_cost"] += m * m
                _t = _e
            INF = 1e18
            size = 1 << m
            dp = np.full(size, INF)
            par = np.full(size, -1, dtype=np.int64)
            choice = np.full(size, -1, dtype=np.int64)
            dp[0] = 0.0
            for mask in range(size):
                if dp[mask] >= INF:
                    continue
                a = bin(mask).count("1")
                if a >= m:
                    continue
                base = dp[mask]
                for b in range(m):
                    if mask & (1 << b):
                        continue
                    nm = mask | (1 << b)
                    v = base + C[a, b]
                    if v < dp[nm]:
                        dp[nm] = v
                        par[nm] = mask
                        choice[nm] = b
            ident = float(sum(C[a, a] for a in range(m)))
            if dpf is not None:
                _e = time.time()
                dpf["t_m_dp"] += _e - _t
                dpf["c_m_dp"] += 1
                _t = _e
            if dp[size - 1] >= ident - 1e-9:
                continue
            assign = [-1] * m
            mask = size - 1
            while mask > 0:
                b = int(choice[mask])
                pm = int(par[mask])
                assign[bin(pm).count("1")] = b
                mask = pm
            if all(assign[a] == a for a in range(m)):
                continue
            snap = P.copy()
            for a, i in enumerate(pool):
                sx, sy, sw, sh = slots[assign[a]]
                if self.kind[i] == 0:
                    ww = sw
                    hh = float(self.opt.areas[i]) / max(sw, 1e-9)
                    P[i] = (sx, min(sy, self.ymax - hh), ww, hh)
                else:
                    ww, hh = float(P[i, 2]), float(P[i, 3])
                    P[i] = (min(max(sx + 0.5 * (sw - ww), self.xmin),
                                self.xmax - ww),
                            min(max(sy + 0.5 * (sh - hh), self.ymin),
                                self.ymax - hh), ww, hh)
            if dpf is not None:
                _e = time.time()
                dpf["t_m_apply"] += _e - _t
                dpf["c_m_apply"] += 1
                _t = _e
            ok, cur_key = self._legal_check(snap, cur_key)
            if ok:
                accepted += 1
                used.update(pool)
                _t = time.time() if dpf is not None else 0.0
                cx = P[:, 0] + 0.5 * P[:, 2]
                cy = P[:, 1] + 0.5 * P[:, 3]
                if dpf is not None:
                    dpf["t_book"] += time.time() - _t
                    dpf["c_book"] += 1
        return accepted, cur_key

    def _evict(self, i: int, target=None, max_anchors: int = 40) -> bool:
        """Relocate block i to any overlap-free slot (anchored beside some
        other block, reshaped to match it when soft).  Used when monotone
        push legalization cannot free a block trapped between locked
        neighbors, and by _deflate to pull blocks out of saturated chains
        (then `target` points into the emptiest region)."""
        P = self.P
        if self.kind[i] == 2:
            return False
        o = target if target is not None else self._optimal_point(i)
        cx = P[:, 0] + 0.5 * P[:, 2]
        cy = P[:, 1] + 0.5 * P[:, 3]
        if o is None:
            o = (float(cx[i]), float(cy[i]))
        d = np.abs(cx - o[0]) + np.abs(cy - o[1])
        pc = getattr(self, "_pred_c", None)
        if target is None and pc is not None and not self.in_cluster[i]:
            # the HPWL median points at the congested core where no
            # slot exists, making the walk-out landing arbitrary — a
            # slot near the model's predicted spot ranks just as high.
            # Cluster members are exempt: their predicted spot lies
            # inside the cluster body, and pulling one member toward
            # it breaks the assembly's exact contacts.
            d = np.minimum(d, np.abs(cx - pc[i, 0])
                           + np.abs(cy - pc[i, 1]))
        order = np.argsort(d)
        area = float(self.opt.areas[i])
        if self._nk is not None:
            # PARTNER_REFINE_KERNEL: the anchor x variant scan is ~90% of
            # `_evict`.  The prologue above stays in Python because
            # `_optimal_point` ends in `_wmedian`'s non-stable argsort.
            return self._nk.evict_scan(i, order, max_anchors,
                                       bool(self.kind[i] == 0), area)
        x0a = P[:, 0]
        y0a = P[:, 1]
        x1a = x0a + P[:, 2]
        y1a = y0a + P[:, 3]
        for j in order[:max_anchors]:
            j = int(j)
            if j == i:
                continue
            xj, yj, wj, hj = P[j]
            variants = []
            if self.kind[i] == 0:
                nw, nh = wj, area / wj
                variants += [(xj, yj + hj, nw, nh), (xj, yj - nh, nw, nh)]
                nh2, nw2 = hj, area / hj
                variants += [(xj + wj, yj, nw2, nh2), (xj - nw2, yj, nw2, nh2)]
            else:
                nw, nh = float(P[i, 2]), float(P[i, 3])
                variants += [(xj, yj + hj, nw, nh), (xj, yj - nh, nw, nh),
                             (xj + wj, yj, nw, nh), (xj - nw, yj, nw, nh)]
            for (nx, ny, nw_, nh_) in variants:
                nx = min(max(nx, self.xmin), self.xmax - nw_)
                ny = min(max(ny, self.ymin), self.ymax - nh_)
                ox = np.minimum(nx + nw_, x1a) - np.maximum(nx, x0a)
                oy = np.minimum(ny + nh_, y1a) - np.maximum(ny, y0a)
                clash = (ox > SEP_TOL) & (oy > SEP_TOL)
                clash[i] = False
                if not clash.any():
                    P[i] = (nx, ny, nw_, nh_)
                    return True
        return False

    def legalize(self, max_sweeps: int = 40,
                 deadline: Optional[float] = None) -> bool:
        """Full legalization of an arbitrarily overlapping input: monotone
        min-displacement sweeps, deadlock kicks (inverted axis preference),
        and eviction of blocks trapped between locked neighbors."""
        for it in range(max_sweeps):
            if deadline is not None and time.time() >= deadline:
                break
            inv = (it % 5 == 4)
            self._axis_pass(0, hold=True, invert=inv)
            self._axis_pass(1, hold=True, invert=inv)
            if not self._has_overlap():
                return True
            if it >= 6 and it % 3 == 2:
                P = self.P
                x0 = P[:, 0]
                x1 = x0 + P[:, 2]
                y0 = P[:, 1]
                y1 = y0 + P[:, 3]
                ox = np.minimum(x1[:, None], x1[None, :]) \
                    - np.maximum(x0[:, None], x0[None, :])
                oy = np.minimum(y1[:, None], y1[None, :]) \
                    - np.maximum(y0[:, None], y0[None, :])
                m = (ox > 9e-7) & (oy > 9e-7)
                np.fill_diagonal(m, False)
                for a, b in zip(*np.nonzero(np.triu(m))):
                    a, b = int(a), int(b)
                    # evict the lighter, movable, non-cluster, unpinned one
                    cands = sorted(
                        (c for c in (a, b)
                         if self.kind[c] != 2 and not self.in_cluster[c]
                         and not (self.groups[self.group_of[c]].pin_x
                                  or self.groups[self.group_of[c]].pin_y)),
                        key=lambda c: self.opt.areas[c])
                    for c in cands:
                        if self._evict(c):
                            break
        return not self._has_overlap()

    # ------------------------------------------------------------------
    # Direct-prediction glue: turn a raw model prediction into a legal,
    # constraint-respecting layout ready for refinement.
    # ------------------------------------------------------------------
    def _anchor_frame_to_tags(self):
        """A preplaced block carrying an edge tag can only satisfy it if
        the layout ends exactly at its edge (the evaluator measures tags
        against the layout extremes and the block cannot move).  Pin the
        frame to the preplaced extent on each tagged side so legalization
        pulls every stray block inside it, and record the lock so
        squeeze/tighten never cross it."""
        P = self.P
        opt = self.opt
        k2 = np.nonzero(self.kind == 2)[0]
        if not len(k2):
            return
        for i, code in zip(opt._bnd_idx, opt._bnd_codes):
            if self.kind[int(i)] != 2:
                continue
            code = int(code)
            if code & 1:
                self.lock_xmin = float(P[k2, 0].min())
            if code & 2:
                self.lock_xmax = float((P[k2, 0] + P[k2, 2]).max())
            if code & 4:
                self.lock_ymax = float((P[k2, 1] + P[k2, 3]).max())
            if code & 8:
                self.lock_ymin = float(P[k2, 1].min())

    def _seed_tags(self):
        """Snap movable tagged blocks flush to the frame edge and pin that
        axis, so legalization keeps the (model-suggested) tag satisfied."""
        P = self.P
        opt = self.opt
        # PARTNER_TAG_ANCHOR=1: correct the snap targets with the
        # preplaced-implied frame BEFORE seeding.  Without this the
        # targets are the raw prediction's extents — a single stray block
        # outside the intended frame drags xmin/ymax off (e.g. -0.048 /
        # +14 units) and every tagged block gets seeded flush to the
        # WRONG line; the late _lock_compact stage then has to evacuate
        # whole walls and often gives up.  (The main refine path never
        # called _anchor_frame_to_tags; only _lock_compact did.)
        if os.environ.get("PARTNER_TAG_ANCHOR") \
                or getattr(opt, "_tag_anchor", False):
            self._anchor_frame_to_tags()
            if self.lock_xmin is not None:
                self.xmin = max(self.xmin, self.lock_xmin)
            if self.lock_xmax is not None:
                self.xmax = min(self.xmax, self.lock_xmax)
            if self.lock_ymin is not None:
                self.ymin = max(self.ymin, self.lock_ymin)
            if self.lock_ymax is not None:
                self.ymax = min(self.ymax, self.lock_ymax)
        for i, code in zip(opt._bnd_idx, opt._bnd_codes):
            i = int(i)
            code = int(code)
            if self.kind[i] == 2 or self.in_cluster[i]:
                continue
            gi = self.group_of[i]
            if gi < 0:
                continue
            g = self.groups[gi]
            if (code & 1) and (code & 2) or (code & 4) and (code & 8):
                continue                      # opposite-edge tags: hopeless
            if code & 1:
                P[i, 0] = self.xmin
                g.pin_x = True
                self.satL[i] = True
            elif code & 2:
                P[i, 0] = self.xmax - P[i, 2]
                g.pin_x = True
                self.satR[i] = True
            if code & 8:
                P[i, 1] = self.ymin
                g.pin_y = True
                self.satB[i] = True
            elif code & 4:
                P[i, 1] = self.ymax - P[i, 3]
                g.pin_y = True
                self.satT[i] = True

        # tagged cluster members: translate the whole rigid group so the
        # member sits flush, then pin that axis of the group — moving the
        # member alone would tear the cluster apart.  (No sat* marks here:
        # _reimpose_seeded_edges moves single blocks, which would also
        # tear the group; the pin itself keeps legalize from drifting it.)
        by_group: dict = {}
        for i, code in zip(opt._bnd_idx, opt._bnd_codes):
            i = int(i)
            code = int(code)
            if self.kind[i] == 2 or not self.in_cluster[i]:
                continue
            gi = self.group_of[i]
            if gi >= 0:
                by_group.setdefault(gi, []).append((i, code))
        for gi, mem in by_group.items():
            g = self.groups[gi]
            wx = {c & 3 for _, c in mem if c & 3}
            wy = {c & 12 for _, c in mem if c & 12}
            if not g.pin_x and len(wx) == 1 and wx != {3}:
                if wx == {1}:
                    i = min((i for i, c in mem if c & 1),
                            key=lambda i: P[i, 0])
                    self._move(g, self.xmin - P[i, 0], 0)
                else:
                    i = max((i for i, c in mem if c & 2),
                            key=lambda i: P[i, 0] + P[i, 2])
                    self._move(g, self.xmax - (P[i, 0] + P[i, 2]), 0)
                g.pin_x = True
            if not g.pin_y and len(wy) == 1 and wy != {12}:
                if wy == {8}:
                    i = min((i for i, c in mem if c & 8),
                            key=lambda i: P[i, 1])
                    self._move(g, self.ymin - P[i, 1], 1)
                else:
                    i = max((i for i, c in mem if c & 4),
                            key=lambda i: P[i, 1] + P[i, 3])
                    self._move(g, self.ymax - (P[i, 1] + P[i, 3]), 1)
                g.pin_y = True

        if os.environ.get("PARTNER_TAG_PACK"):
            self._pack_wall_lines()

    def _pack_wall_lines(self):
        """Joint de-overlap of the seeded wall lines (PARTNER_TAG_PACK=1).

        _seed_tags snaps every tagged block flush at the model's
        along-coordinate, piling blocks on top of each other exactly on
        the wall lines; legalization must then untangle the pile and on
        saturated walls the retry ladder ends up dropping the pins.
        Treat each wall line as a 1-D interval problem instead:
        order-preserving, minimal displacement, locked blocks as
        barriers.  Lines that genuinely do not fit are left untouched
        (the ladder keeps its old behavior there)."""
        P = self.P
        n = len(P)
        sat = {'L': self.satL, 'R': self.satR, 'B': self.satB,
               'T': self.satT}
        for wall in ('L', 'R', 'B', 'T'):
            vertical = wall in ('L', 'R')
            axis = 1 if vertical else 0
            if wall == 'L':
                on = np.abs(P[:, 0] - self.xmin) < 1e-9
            elif wall == 'R':
                on = np.abs(P[:, 0] + P[:, 2] - self.xmax) < 1e-9
            elif wall == 'B':
                on = np.abs(P[:, 1] - self.ymin) < 1e-9
            else:
                on = np.abs(P[:, 1] + P[:, 3] - self.ymax) < 1e-9
            lo, hi = ((self.ymin, self.ymax) if vertical
                      else (self.xmin, self.xmax))
            bars = []
            items = []  # (tag, ref, a0, size)
            seen_g = set()
            for i in range(n):
                if not on[i]:
                    continue
                if self.kind[i] == 2:
                    bars.append((float(P[i, axis]),
                                 float(P[i, axis] + P[i, 2 + axis])))
                    continue
                gi = self.group_of[i]
                if gi >= 0 and self.in_cluster[i]:
                    if gi in seen_g:
                        continue
                    g = self.groups[gi]
                    if not (g.pin_x if vertical else g.pin_y):
                        continue
                    seen_g.add(gi)
                    mem = g.members
                    a0 = float(P[mem, axis].min())
                    a1 = float((P[mem, axis] + P[mem, 2 + axis]).max())
                    items.append(('g', gi, a0, a1 - a0))
                elif sat[wall][i]:
                    items.append(('s', i, float(P[i, axis]),
                                  float(P[i, 2 + axis])))
            if len(items) <= 1:
                continue
            bars.sort()
            segs = []
            cur = lo
            for b0, b1 in bars:
                if b0 > cur + 1e-12:
                    segs.append((cur, min(b0, hi)))
                cur = max(cur, b1)
            if cur < hi - 1e-12:
                segs.append((cur, hi))
            if not segs:
                continue
            if sum(sz for _t, _r, _a, sz in items) > \
                    sum(e - s for s, e in segs) + 1e-9:
                continue  # saturated line: leave the ladder's old behavior
            items.sort(key=lambda it: it[2] + 0.5 * it[3])
            si = 0
            cursor = segs[0][0]
            placed = []
            ok = True
            for tag, ref, a0, sz in items:
                while si < len(segs) and cursor + sz > segs[si][1] + 1e-9:
                    si += 1
                    if si < len(segs):
                        cursor = segs[si][0]
                if si >= len(segs):
                    ok = False
                    break
                placed.append([tag, ref, a0, sz, si, cursor])
                cursor += sz
            if not ok:
                continue  # fragmentation defeat: leave untouched
            by_seg: dict = {}
            for rec in placed:
                by_seg.setdefault(rec[4], []).append(rec)
            for sj, recs in by_seg.items():
                recs.sort(key=lambda r: r[5])
                limit = segs[sj][1]
                for rec in reversed(recs):
                    pos = min(max(rec[2], rec[5]), limit - rec[3])
                    rec[5] = pos
                    limit = pos
            for tag, ref, a0, _sz, _sj, pos in placed:
                d = pos - a0
                if abs(d) < 1e-12:
                    continue
                if tag == 's':
                    P[ref, axis] = P[ref, axis] + d
                else:
                    self._move(self.groups[ref], d, axis)

    def _free_rects(self, max_rects: int = 10,
                    min_area: float = 0.0, region=None, skip=None):
        """Decompose the free space inside the current frame into up to
        `max_rects` large empty rectangles (greedy largest-first on the
        edge-coordinate grid).  Soft blocks have no aspect limit, so any
        free rectangle with enough area can host any smaller soft block —
        this map is what eviction/squeeze never had.  `region` restricts
        the scan to (x0, y0, x1, y1); blocks in `skip` are treated as
        absent (the remove-and-repack primitive)."""
        P = self.P
        if region is None:
            rx0, ry0, rx1, ry1 = self.xmin, self.ymin, self.xmax, self.ymax
        else:
            rx0, ry0, rx1, ry1 = region
        xs = np.unique(np.clip(np.concatenate(
            [[rx0, rx1], P[:, 0], P[:, 0] + P[:, 2]]), rx0, rx1))
        ys = np.unique(np.clip(np.concatenate(
            [[ry0, ry1], P[:, 1], P[:, 1] + P[:, 3]]), ry0, ry1))
        if len(xs) < 2 or len(ys) < 2:
            return []
        wx = np.diff(xs)
        wy = np.diff(ys)
        occ = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
        for i in range(self.n):
            if skip is not None and skip[i]:
                continue
            c0 = np.searchsorted(xs, P[i, 0] + 1e-9)
            c1 = np.searchsorted(xs, P[i, 0] + P[i, 2] - 1e-9)
            r0 = np.searchsorted(ys, P[i, 1] + 1e-9)
            r1 = np.searchsorted(ys, P[i, 1] + P[i, 3] - 1e-9)
            occ[max(r0 - 1, 0):r1, max(c0 - 1, 0):c1] = True
        out = []
        for _ in range(max_rects):
            # largest empty rectangle via weighted-histogram stack scan
            best = None          # (area, c0, c1, height, y_bottom)
            hist = np.zeros(len(wx))
            for r in range(occ.shape[0]):
                hist = np.where(~occ[r], hist + wy[r], 0.0)
                y_bot = float(ys[r + 1])
                stack = []
                for c in range(len(wx) + 1):
                    h = float(hist[c]) if c < len(wx) else -1.0
                    start = c
                    while stack and stack[-1][1] > h + 1e-12:
                        s, sh = stack.pop()
                        area = float(xs[c] - xs[s]) * sh
                        if best is None or area > best[0]:
                            best = (area, s, c, sh, y_bot)
                        start = s
                    if not stack or stack[-1][1] < h - 1e-12:
                        stack.append((start, h))
            if best is None or best[0] <= max(min_area, 1e-9):
                break
            area, c0, c1, hh, y_bot = best
            x0 = float(xs[c0])
            y0 = y_bot - hh
            out.append((x0, y0, float(xs[c1] - xs[c0]), float(hh)))
            rt = max(int(np.searchsorted(ys, y0 + 1e-9)) - 1, 0)
            r1 = int(np.searchsorted(ys, y_bot - 1e-9))
            occ[rt:r1, c0:c1] = True
        return out

    def _movable_free(self, i: int, allow_fixed: bool = False) -> bool:
        """Unconstrained block: no cluster, no MIB, no boundary tag, group
        not pinned — safe to relocate anywhere.  kind-1 (fixed SHAPE)
        blocks may relocate too when allow_fixed (their position is free,
        only the dims are hard)."""
        if self.kind[i] == 2 or self.in_cluster[i]:
            return False
        if self.kind[i] == 1 and not allow_fixed:
            return False
        if hasattr(self.opt, "mib") and self.opt.mib[i] > 0:
            return False
        if (i in getattr(self, "_tagged_set", ())
                and not getattr(self, "_ignore_tags", False)):
            return False
        gi = int(self.group_of[i])
        if gi < 0:
            return False
        g = self.groups[gi]
        return not (g.pin_x or g.pin_y) and len(g.members) == 1

    def _relocate_to_free(self, i: int, rects) -> bool:
        """Place soft block i inside the closest-fitting free rectangle,
        reshaped to a healthy aspect that fits; updates the rect list."""
        P = self.P
        area = float(self.opt.areas[i])
        rigid = self.kind[i] != 0
        o = self._optimal_point(i)
        if o is None:
            o = (float(P[i, 0] + 0.5 * P[i, 2]),
                 float(P[i, 1] + 0.5 * P[i, 3]))
        anchors = [o]
        pc = getattr(self, "_pred_c", None)
        if pc is not None and not self.in_cluster[i]:
            # a rect near the model's predicted spot beats a slightly
            # nearer one at the HPWL median when the median's region
            # is packed solid (the landing there was arbitrary anyway);
            # cluster members are exempt — their predicted spot is
            # inside the cluster body and pulls them off the assembly
            anchors.append((float(pc[i, 0]), float(pc[i, 1])))
        best = None
        for k, (rx, ry, rw, rh) in enumerate(rects):
            if rigid:
                # fixed shape: dims are hard, only fit as-is
                w, h = float(P[i, 2]), float(P[i, 3])
                if w > rw + 1e-9 or h > rh + 1e-9:
                    continue
            else:
                if rw * rh < area * 1.02:
                    continue
                # width range that fits: h = area/w <= rh and w <= rw
                w_lo = max(area / rh, 1e-6)
                w_hi = rw
                if w_lo > w_hi:
                    continue
                # prefer the square-ish shape inside the feasible range
                w = min(max(math.sqrt(area), w_lo), w_hi)
                h = area / w
            for ax_, ay_ in anchors:
                nx = min(max(ax_ - 0.5 * w, rx), rx + rw - w)
                ny = min(max(ay_ - 0.5 * h, ry), ry + rh - h)
                d = abs(nx + 0.5 * w - ax_) + abs(ny + 0.5 * h - ay_)
                if best is None or d < best[0]:
                    best = (d, k, nx, ny, w, h)
        if best is None:
            return False
        _d, k, nx, ny, w, h = best
        P[i] = (nx, ny, w, h)
        # conservatively retire the used rect (no re-split bookkeeping)
        rects.pop(k)
        return True

    def _rect_clash(self, i: int, x0: float, y0: float,
                    w: float, h: float, exclude=None) -> bool:
        """Would rect (x0,y0,w,h) for block i overlap anything else?"""
        P = self.P
        cx = np.minimum(x0 + w, P[:, 0] + P[:, 2]) - np.maximum(x0, P[:, 0])
        cy = np.minimum(y0 + h, P[:, 1] + P[:, 3]) - np.maximum(y0, P[:, 1])
        hit = (cx > SEP_TOL) & (cy > SEP_TOL)
        hit[i] = False
        if exclude is not None:
            hit[exclude] = False
        return bool(hit.any())

    def _compact_reloc(self, i: int, rects, rigid: bool,
                       avoid=None, flush=None) -> bool:
        """Relocate block i into the free-rect map (softs reshape to fit,
        rigid dims placed as-is), preferring the landing closest to the
        block's HPWL optimum.  flush=(axis, side, lock) restricts
        landings to rects on the lock line and seats the block's far
        edge exactly on it — a block TAGGED on the compacted side may
        only relocate to where the new layout edge will be.  The used
        rect is guillotine-split and the remainders go back to the map,
        so one coalesced channel can host several strip blocks.  Rects
        can be stale after earlier commits, so every landing is
        clash-checked before it is taken."""
        P = self.P
        area = float(self.opt.areas[i])
        anchors = []
        o = self._optimal_point(i)
        if o is not None:
            anchors.append(o)
        pc = getattr(self, "_pred_c", None)
        if pc is not None:
            anchors.append((float(pc[i, 0]), float(pc[i, 1])))
        if not anchors:
            anchors.append((float(P[i, 0] + 0.5 * P[i, 2]),
                            float(P[i, 1] + 0.5 * P[i, 3])))
        cands = []
        for k, (rx, ry, rw, rh) in enumerate(rects):
            if avoid is not None and any(
                    min(rx + rw, tx + tw) - max(rx, tx) > SEP_TOL
                    and min(ry + rh, ty + th) - max(ry, ty) > SEP_TOL
                    for (tx, ty, tw, th) in avoid):
                continue
            if flush is not None:
                fa, fs, flk = flush
                redge = (ry + rh if fa else rx + rw) if fs \
                    else (ry if fa else rx)
                if (redge < flk - 1e-6) if fs else (redge > flk + 1e-6):
                    continue
            if rigid:
                w, h = float(P[i, 2]), float(P[i, 3])
                if w > rw + 1e-9 or h > rh + 1e-9:
                    continue
            else:
                if rw * rh < area * 1.02:
                    continue
                w_lo = max(area / rh, 1e-6)
                if w_lo > rw:
                    continue
                w = min(max(math.sqrt(area), w_lo), rw)
                h = area / w
            for o in anchors:
                nx = min(max(o[0] - 0.5 * w, rx), rx + rw - w)
                ny = min(max(o[1] - 0.5 * h, ry), ry + rh - h)
                if flush is not None:
                    fa, fs, flk = flush
                    if fa:
                        ny = (flk - h) if fs else flk
                    else:
                        nx = (flk - w) if fs else flk
                d = abs(nx + 0.5 * w - o[0]) + abs(ny + 0.5 * h - o[1])
                cands.append((d, k, nx, ny, w, h))
        cands.sort()
        for d, k, nx, ny, w, h in cands[:4]:
            if not self._rect_clash(i, nx, ny, w, h):
                P[i] = (nx, ny, w, h)
                rx, ry, rw, rh = rects.pop(k)
                # guillotine remainders: left/right full-height slices,
                # top/bottom slices above/below the block
                for (sx, sy, sw, sh) in (
                        (rx, ry, nx - rx, rh),
                        (nx + w, ry, rx + rw - (nx + w), rh),
                        (nx, ry, w, ny - ry),
                        (nx, ny + h, w, ry + rh - (ny + h))):
                    if sw > 0.5 and sh > 0.5 and sw * sh > 4.0:
                        rects.append((sx, sy, sw, sh))
                return True
        return False

    def _dig_drop(self, idxs, d: float, axis: int, rects) -> bool:
        """Drop blocks `idxs` (a singleton or a whole rigid cluster
        group) by d along axis into currently occupied territory: every
        occupant of the landing footprint must be freely movable, and
        each is relocated into the free-rect map before the drop.  This
        is the op the strip shelves need — the tagged block belongs just
        inside the lock, but untagged material sits there; swap them.
        Reverts cleanly when any occupant has nowhere to go."""
        P = self.P
        ex = np.asarray(idxs, dtype=np.int64)
        fr_lo = (self.xmin, self.ymin)[axis]
        fr_hi = (self.xmax, self.ymax)[axis]
        tg = []
        for j in idxs:
            r = [float(P[j, 0]), float(P[j, 1]),
                 float(P[j, 2]), float(P[j, 3])]
            r[axis] += d
            if r[axis] < fr_lo - 1e-6 or r[axis] + r[axis + 2] > fr_hi + 1e-6:
                return False
            tg.append(tuple(r))
        x0 = P[:, 0]
        y0 = P[:, 1]
        x1 = x0 + P[:, 2]
        y1 = y0 + P[:, 3]
        occ = set()
        for (rx, ry, rw, rh) in tg:
            cx = np.minimum(rx + rw, x1) - np.maximum(rx, x0)
            cy = np.minimum(ry + rh, y1) - np.maximum(ry, y0)
            hit = (cx > SEP_TOL) & (cy > SEP_TOL)
            hit[ex] = False
            occ.update(int(v) for v in np.nonzero(hit)[0])
        if not occ or len(occ) > 8:
            return False
        occ = sorted(occ, key=lambda j: -float(P[j, 2] * P[j, 3]))
        if not all(self._movable_free(j, allow_fixed=True) for j in occ):
            return False
        snap = P.copy()
        for j in occ:
            if not self._compact_reloc(j, rects, self.kind[j] != 0,
                                       avoid=tg):
                P[...] = snap
                return False
        if len(idxs) > 1:
            gi = int(self.group_of[idxs[0]])
            self._move(self.groups[gi], d, axis)
        else:
            P[idxs[0], axis] += d
        for j in idxs:
            if self._rect_clash(int(j), float(P[j, 0]), float(P[j, 1]),
                                float(P[j, 2]), float(P[j, 3]), ex):
                P[...] = snap
                return False
        return True

    def _gravity(self, axis: int, side: int, tags,
                 skip_bit: int) -> bool:
        """Monotone rigid drop of every movable unit toward the frame
        edge opposite `side` along `axis`.  The free slivers scattered
        through the packing are individually too small to host any
        strip block — gravity coalesces them into a channel where the
        strip needs them, without breaking a single contact or seat:
        cluster groups drop rigidly, the cross coordinate never
        changes, groups pinned on the axis (satisfied seats) stay, and
        blocks tagged on the COMPACTED side (skip_bit) stay — they
        belong at the new edge, not in a mid-layout hole."""
        P = self.P
        # a block tagged on the edge gravity moves AWAY from must stay:
        # dragging it off its (possibly not-yet-seated) edge only breaks
        # the seat the judge would then veto
        opp_bit = ((8 if side else 4) if axis
                   else (1 if side else 2))
        units = []
        for g in self.groups:
            if g.pin_y if axis else g.pin_x:
                continue
            if any(tags.get(int(j), 0) & (skip_bit | opp_bit)
                   for j in g.members):
                continue
            key = float(P[g.members, axis].min())
            units.append((key, g))
        # blocks nearest the gravity sink drop first so the ones behind
        # can follow into the vacated room
        units.sort(key=lambda kg: kg[0], reverse=bool(side))
        moved = False
        for _k, g in units:
            lo, hi = self._interval(g, axis)
            d = hi if side else lo
            if (side and d > 1e-9) or (not side and d < -1e-9):
                self._move(g, d, axis)
                moved = True
        return moved

    def _member_reland(self, i: int, axis: int, side: int,
                       lock: float) -> bool:
        """Cluster member i sticks beyond the lock but its group cannot
        translate (it holds the very preplaced anchor whose tag defines
        the lock, or the body is jammed).  Detach the single member and
        re-land it flush against another member of the same cluster
        wholly inside the lock — the landing coordinate on the contact
        axis is EXACT edge arithmetic (the evaluator's grouping test
        needs exact abutment) and the position ALONG the fellow's edge
        slides continuously to the nearest clash-free interval.
        Skipped when i is an articulation block of the cluster's touch
        graph."""
        P = self.P
        opt = self.opt
        cl = int(opt.cluster[i])
        others = [j for j in range(self.n)
                  if j != i and int(opt.cluster[j]) == cl]
        if not others:
            return False
        comps = _touch_components(P, others)
        if len(comps) > 1:
            return False
        w, h = float(P[i, 2]), float(P[i, 3])
        dims = (w, h)
        fr = (self.xmin, self.ymin, self.xmax, self.ymax)
        x0a = P[:, 0]
        y0a = P[:, 1]
        x1a = x0a + P[:, 2]
        y1a = y0a + P[:, 3]
        best = None
        for j in others:
            for ca in (0, 1):        # contact axis: 0 = land left/right
                co_ = 1 - ca
                for sd in (0, 1):    # 1 = land on j's max side
                    if sd:
                        base = float(x1a[j] if ca == 0 else y1a[j])
                    else:
                        base = float((x0a[j] if ca == 0 else y0a[j])
                                     - dims[ca])
                    # frame + lock containment on the contact axis
                    if base < fr[ca] - 1e-9 \
                            or base + dims[ca] > fr[ca + 2] + 1e-9:
                        continue
                    if ca == axis:
                        far = base + dims[ca] if side else base
                        if (far > lock + 1e-9) if side \
                                else (far < lock - 1e-9):
                            continue
                    # slide range along j's edge (must keep overlap)
                    s_lo = float((x0a[j] if ca == 1 else y0a[j])
                                 - dims[co_]) + 1e-3
                    s_hi = float((x1a[j] if ca == 1 else y1a[j])) - 1e-3
                    s_lo = max(s_lo, fr[co_])
                    s_hi = min(s_hi, fr[co_ + 2] - dims[co_])
                    if co_ == axis:
                        if side:
                            s_hi = min(s_hi, lock - dims[co_])
                        else:
                            s_lo = max(s_lo, lock)
                    if s_lo > s_hi:
                        continue
                    # obstacles overlapping the landing lane
                    if ca == 0:
                        lane = (np.minimum(x1a, base + w)
                                - np.maximum(x0a, base)) > SEP_TOL
                        obs0, obs1 = y0a, y1a
                    else:
                        lane = (np.minimum(y1a, base + h)
                                - np.maximum(y0a, base)) > SEP_TOL
                        obs0, obs1 = x0a, x1a
                    lane[i] = False
                    tgt = float(P[i, co_])
                    cand = min(max(tgt, s_lo), s_hi)
                    ivs = [(float(obs0[k2]) - dims[co_], float(obs1[k2]))
                           for k2 in np.nonzero(lane)[0]]
                    ivs.sort()
                    # nearest point of [s_lo, s_hi] minus the blocked
                    # intervals to the block's current coordinate
                    pts = []
                    cur = s_lo
                    for a, b in ivs + [(s_hi + 1.0, s_hi + 2.0)]:
                        if a - cur > 1e-9:
                            seg_lo, seg_hi = cur, min(a, s_hi)
                            if seg_lo <= seg_hi:
                                pts.append(min(max(tgt, seg_lo), seg_hi))
                        cur = max(cur, b)
                        if cur > s_hi:
                            break
                    for t in pts:
                        d = abs(t - tgt) + abs(base - float(P[i, ca]))
                        if best is None or d < best[0]:
                            nx, ny = (base, t) if ca == 0 else (t, base)
                            best = (d, nx, ny)
        if best is None:
            return False
        _d, nx, ny = best
        if self._rect_clash(i, nx, ny, w, h):
            return False
        P[i, 0], P[i, 1] = nx, ny
        return True

    def _piece_reland(self, i: int, axis: int, side: int,
                      lock: float) -> bool:
        """Cluster chains make every interior member an articulation
        block, so single-member relanding can never fix a protruding
        chain END of two or more blocks.  Detach the whole connected
        protruding sub-piece containing i and re-land it rigidly — the
        intra-piece exact contacts are recorded first and re-imposed
        down a spanning tree after the translate — against the
        cluster's remaining body, wholly inside the lock."""
        P = self.P
        opt = self.opt
        cl = int(opt.cluster[i])
        mem = [j for j in range(self.n) if int(opt.cluster[j]) == cl]
        if len(mem) < 2:
            return False
        lo = P[:, axis]
        far = lo + P[:, axis + 2]
        if side:
            bad = [j for j in mem if far[j] > lock + 1e-9]
        else:
            bad = [j for j in mem if lo[j] < lock - 1e-9]
        if i not in bad:
            return False
        piece = next((c for c in _touch_components(P, bad) if i in c),
                     None)
        if piece is None or any(self.kind[j] == 2 for j in piece):
            return False
        rest = [j for j in mem if j not in piece]
        if not rest or len(_touch_components(P, rest)) > 1:
            return False
        pset = np.zeros(self.n, dtype=bool)
        pset[piece] = True
        # intra-piece exact contacts, BFS tree from i
        tree = []
        if len(piece) > 1:
            seen = {i}
            frontier = [i]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in piece:
                        if v in seen:
                            continue
                        ovy = min(P[u, 1] + P[u, 3], P[v, 1] + P[v, 3]) \
                            - max(P[u, 1], P[v, 1])
                        ovx = min(P[u, 0] + P[u, 2], P[v, 0] + P[v, 2]) \
                            - max(P[u, 0], P[v, 0])
                        rel = None
                        if ovy > 1e-9 and abs(P[u, 0] + P[u, 2]
                                              - P[v, 0]) <= TOUCH_TOL:
                            rel = (0, 1)
                        elif ovy > 1e-9 and abs(P[v, 0] + P[v, 2]
                                                - P[u, 0]) <= TOUCH_TOL:
                            rel = (0, -1)
                        elif ovx > 1e-9 and abs(P[u, 1] + P[u, 3]
                                                - P[v, 1]) <= TOUCH_TOL:
                            rel = (1, 1)
                        elif ovx > 1e-9 and abs(P[v, 1] + P[v, 3]
                                                - P[u, 1]) <= TOUCH_TOL:
                            rel = (1, -1)
                        if rel is not None:
                            tree.append((u, v, rel))
                            seen.add(v)
                            nxt.append(v)
                frontier = nxt
            if len(seen) != len(piece):
                return False
        fr = (self.xmin, self.ymin, self.xmax, self.ymax)
        # landing candidates: piece block a corner-snapped onto rest
        # block b (the abutting coordinate is exact edge arithmetic)
        cands = []
        for a in piece:
            aw, ah = float(P[a, 2]), float(P[a, 3])
            ax, ay = float(P[a, 0]), float(P[a, 1])
            for b in rest:
                bx0, by0 = float(P[b, 0]), float(P[b, 1])
                bx1 = bx0 + float(P[b, 2])
                by1 = by0 + float(P[b, 3])
                for nx, ny in ((bx1, by0), (bx1, by1 - ah),
                               (bx0 - aw, by0), (bx0 - aw, by1 - ah),
                               (bx0, by1), (bx1 - aw, by1),
                               (bx0, by0 - ah), (bx1 - aw, by0 - ah)):
                    cands.append((abs(nx - ax) + abs(ny - ay),
                                  a, nx - ax, ny - ay))
        cands.sort(key=lambda c: c[0])
        for _d, a, dx, dy in cands[:24]:
            ok = True
            for j in piece:
                jx = float(P[j, 0]) + dx
                jy = float(P[j, 1]) + dy
                jw, jh = float(P[j, 2]), float(P[j, 3])
                jfar = (jy + jh if axis else jx + jw) if side \
                    else (jy if axis else jx)
                if jx < fr[0] - 1e-9 or jy < fr[1] - 1e-9 \
                        or jx + jw > fr[2] + 1e-9 \
                        or jy + jh > fr[3] + 1e-9 \
                        or ((jfar > lock + 1e-9) if side
                            else (jfar < lock - 1e-9)) \
                        or self._rect_clash(j, jx, jy, jw, jh, pset):
                    ok = False
                    break
            if not ok:
                continue
            snap = P[piece].copy()
            for j in piece:
                P[j, 0] += dx
                P[j, 1] += dy
            for u, v, (rax, sgn) in tree:
                if sgn > 0:
                    P[v, rax] = P[u, rax] + P[u, rax + 2]
                else:
                    P[v, rax] = P[u, rax] - P[v, rax + 2]
            # the tree re-imposition nudges floats: re-verify
            bad2 = False
            for j in piece:
                if self._rect_clash(j, float(P[j, 0]), float(P[j, 1]),
                                    float(P[j, 2]), float(P[j, 3]),
                                    pset):
                    bad2 = True
                    break
            if bad2:
                P[piece] = snap
                continue
            return True
        return False

    def _split_rect(self, rects, k, nx, ny, w, h):
        """Take rect k for a placement at (nx,ny,w,h); return the
        guillotine remainders to the map."""
        rx, ry, rw, rh = rects.pop(k)
        for (sx, sy, sw, sh) in (
                (rx, ry, nx - rx, rh),
                (nx + w, ry, rx + rw - (nx + w), rh),
                (nx, ry, w, ny - ry),
                (nx, ny + h, w, ry + rh - (ny + h))):
            if sw > 0.5 and sh > 0.5 and sw * sh > 4.0:
                rects.append((sx, sy, sw, sh))

    def _skyline_flush(self, j: int, axis: int, side: int, lock: float,
                       c0: float, c1: float, far_bound: float,
                       skip, anchors, pin=None) -> bool:
        """Composite flush landing on the lock line.  The rectangular
        free-space decomposition splits an L-shaped pocket into stacked
        rects, so single-rect flush placement under-reads the true
        depth — the first big tagged block then eats the whole lock
        line as a full-width pancake and starves the rest of the
        shelf.  Scan the actual depth profile (distance from the lock
        line to the nearest non-evacuated obstacle) along the cross
        axis instead: a deep pocket hosts the same area in LESS
        lock-line width (w = area/depth), which is what lets a whole
        tagged shelf seat on a crowded line."""
        P = self.P
        o = 1 - axis
        area = float(self.opt.areas[j])
        base = (lock - far_bound) if side else (far_bound - lock)
        if base <= 1e-9:
            return False
        bps = [c0, c1]
        obs = []
        for k in range(self.n):
            if skip[k] or k == j:
                continue
            a0 = float(P[k, o])
            a1 = a0 + float(P[k, o + 2])
            if min(a1, c1) - max(a0, c0) <= SEP_TOL:
                continue
            if side:
                if float(P[k, axis]) >= lock - 1e-9:
                    continue
                dk = lock - float(P[k, axis] + P[k, axis + 2])
            else:
                if float(P[k, axis] + P[k, axis + 2]) <= lock + 1e-9:
                    continue
                dk = float(P[k, axis]) - lock
            if dk >= base - 1e-9:
                continue
            obs.append((max(a0, c0), min(a1, c1), dk))
            bps.append(max(a0, c0))
            bps.append(min(a1, c1))
        xs = np.unique(np.asarray(bps, dtype=np.float64))
        m = len(xs) - 1
        if m < 1:
            return False
        dep = np.full(m, base)
        for a0, a1, dk in obs:
            s0 = max(int(np.searchsorted(xs, a0 + 1e-9)) - 1, 0)
            s1 = int(np.searchsorted(xs, a1 - 1e-9))
            dep[s0:s1] = np.minimum(dep[s0:s1], dk)
        h_cap = 8.0 * math.sqrt(area)
        best = None
        if pin is not None:
            # corner landing: the cross position is FORCED to the
            # pinned frame edge (a dual-tagged block keeps its other
            # seat only there) — scan the single run anchored at that
            # edge, deepest fit first
            dmin = float('inf')
            rng = range(m) if pin == 0 else range(m - 1, -1, -1)
            for se in rng:
                if dep[se] <= 1e-9:
                    break
                dmin = min(dmin, float(dep[se]))
                du = min(dmin, h_cap)
                w = area / du
                span = (float(xs[se + 1]) - c0) if pin == 0 \
                    else (c1 - float(xs[se]))
                if w <= span:
                    t = c0 if pin == 0 else c1 - w
                    land = (lock - du) if side else lock
                    best = (0.0, t, land, w, du)
                    break
        for si in (range(m) if pin is None else ()):
            if dep[si] <= 1e-9:
                continue
            dmin = float('inf')
            xl = float(xs[si])
            for se in range(si, m):
                if dep[se] <= 1e-9:
                    break
                dmin = min(dmin, float(dep[se]))
                xr = float(xs[se + 1])
                du = min(dmin, h_cap)
                w = area / du
                if w <= xr - xl:
                    land = (lock - du) if side else lock
                    for a in anchors:
                        ac, al = a[o], a[axis]
                        t = min(max(ac - 0.5 * w, xl), xr - w)
                        dd = abs(t + 0.5 * w - ac) \
                            + abs(land + 0.5 * du - al)
                        if best is None or dd < best[0]:
                            best = (dd, t, land, w, du)
                    break       # deepest fit per run start
        if best is None:
            return False
        _dd, t, land, w, du = best
        rect = [0.0, 0.0, 0.0, 0.0]
        rect[o] = t
        rect[axis] = land
        rect[o + 2] = w
        rect[axis + 2] = du
        ex = np.nonzero(skip)[0]
        if self._rect_clash(j, rect[0], rect[1], rect[2], rect[3], ex):
            return False
        P[j] = rect
        return True

    def _band_restack(self, i: int, axis: int, side: int, lock: float,
                      rects, tags, widen: float = 1.0,
                      depth: float = None) -> bool:
        """Remove-and-repack of the band under strip block i.  The
        sliver seams between columns can never coalesce by sliding
        (saturated packings jam laterally) and single free rects are
        individually too small — but soft blocks have NO aspect limit,
        so pulling every free soft block of the band OUT and re-placing
        them into the band's own free-rect map (exact-width fill,
        h = area/w, landings as close to the original spot as the map
        allows) repacks the whole column staircase in one shot.  Blocks
        tagged on the compacted side land flush on the lock (the future
        layout edge).  Members that no longer fit are expelled into the
        global free-rect map; any failure reverts the band.  `widen`
        grows the band beyond i's own span — a big tagged block may
        need the free area of two neighbouring columns.  `depth` caps
        how far from the lock the band reaches: a full-height full-width
        evacuation repacks half the layout and the HPWL bill dwarfs the
        bbox prize — a shallow strip disturbs only the shelf that must
        move."""
        P = self.P
        opt = self.opt
        o = 1 - axis
        bit = (4 if side else 8) if axis else (2 if side else 1)
        W0 = float(P[i, o + 2])
        x0 = float(P[i, o]) - 0.5 * (widen - 1.0) * W0
        x1 = x0 + widen * W0
        x0 = max(x0, (self.xmin, self.ymin)[o])
        x1 = min(x1, (self.xmax, self.ymax)[o])
        W = x1 - x0
        if W <= 1e-6:
            return False
        fr_min = (self.xmin, self.ymin)
        fr_max = (self.xmax, self.ymax)
        lo_b = fr_min[axis] if side else lock
        hi_b = lock if side else fr_max[axis]
        if depth is not None:
            if side:
                lo_b = max(lo_b, lock - depth)
            else:
                hi_b = min(hi_b, lock + depth)
        if hi_b - lo_b <= 1e-6:
            return False
        members, tagged = [], []
        skip = np.zeros(self.n, dtype=bool)
        for j in range(self.n):
            jx0 = float(P[j, o])
            jx1 = jx0 + float(P[j, o + 2])
            if min(jx1, x1) - max(jx0, x0) <= SEP_TOL:
                continue
            # a depth-capped band only evacuates blocks reaching into
            # the strip (or beyond the lock); the rest stay put as
            # obstacles in the profile
            if side:
                if float(P[j, axis] + P[j, axis + 2]) <= lo_b + 1e-7:
                    continue
            elif float(P[j, axis]) >= hi_b - 1e-7:
                continue
            code = tags.get(j, 0)
            if jx0 >= x0 - 1e-7 and jx1 <= x1 + 1e-7 \
                    and self.kind[j] == 0 and opt.mib[j] <= 0 \
                    and not self.in_cluster[j] and code in (0, bit):
                (tagged if code else members).append(j)
                skip[j] = True
        if i not in members and i not in tagged:
            return False
        region = (x0, lo_b, x1, hi_b) if axis else (lo_b, x0, hi_b, x1)
        band_rects = self._free_rects(24, region=region, skip=skip)
        need = sum(float(opt.areas[j]) for j in members + tagged)
        if sum(rw * rh for (_a, _b, rw, rh) in band_rects) < 0.7 * need:
            return False        # hopeless band, save the placement work
        snap = P.copy()

        pc = getattr(self, "_pred_c", None)

        def _place(j, flush_only):
            area = float(opt.areas[j])
            anchors = [(float(snap[j, 0] + 0.5 * snap[j, 2]),
                        float(snap[j, 1] + 0.5 * snap[j, 3]))]
            if pc is not None:
                anchors.append((float(pc[j, 0]), float(pc[j, 1])))
            best = None
            for k, (rx, ry, rw, rh) in enumerate(band_rects):
                if rw * rh < area * (1.0 + 1e-9):
                    continue
                if flush_only:
                    redge = (ry + rh if axis else rx + rw) if side \
                        else (ry if axis else rx)
                    if abs(redge - lock) > 1e-7:
                        continue
                # exact fill across the rect, spill into the stack axis
                cw = rw if axis else rh
                hh = area / cw
                if hh > (rh if axis else rw) + 1e-9:
                    hh = rh if axis else rw
                    cw = area / hh
                for ox, oy in anchors:
                    if axis:
                        nx, w2, h2 = rx, cw, hh
                        if flush_only:
                            ny = (lock - hh) if side else lock
                        else:
                            # nearest rect edge to the anchor — mid-rect
                            # landings fragment the map for no gain
                            ny = ry if abs(ry + 0.5 * hh - oy) \
                                <= abs(ry + rh - 0.5 * hh - oy) \
                                else ry + rh - hh
                    else:
                        ny, w2, h2 = ry, hh, cw
                        if flush_only:
                            nx = (lock - hh) if side else lock
                        else:
                            nx = rx if abs(rx + 0.5 * hh - ox) \
                                <= abs(rx + rw - 0.5 * hh - ox) \
                                else rx + rw - hh
                    d = abs(nx + 0.5 * w2 - ox) + abs(ny + 0.5 * h2 - oy)
                    if best is None or d < best[0]:
                        best = (d, k, nx, ny, w2, h2)
            if best is None:
                return False
            _d, k, nx, ny, w2, h2 = best
            self._split_rect(band_rects, k, nx, ny, w2, h2)
            P[j] = (nx, ny, w2, h2)
            return True

        expel = []
        for j in sorted(tagged, key=lambda j: -float(opt.areas[j])):
            # flush landings always go through the depth profile: the
            # single-rect exact-fill would spread the first big block
            # across the rect's whole width and starve the rest of the
            # shelf of lock-line length
            anch = [(float(snap[j, 0] + 0.5 * snap[j, 2]),
                     float(snap[j, 1] + 0.5 * snap[j, 3]))]
            if pc is not None:
                anch.append((float(pc[j, 0]), float(pc[j, 1])))
            if not self._skyline_flush(j, axis, side, lock, x0, x1,
                                       lo_b if side else hi_b,
                                       skip, anch):
                P[...] = snap
                return False
            skip[j] = False
        if tagged:
            band_rects = self._free_rects(24, region=region, skip=skip)
        sky_m = False
        for j in sorted(members, key=lambda j: -float(opt.areas[j])):
            ok = (not sky_m) and _place(j, False)
            if not ok:
                # no single rect fits: bottom-fill the strip from its
                # far bound with a composite landing (same profile
                # scan, lock flipped to the band floor)
                anch = [(float(snap[j, 0] + 0.5 * snap[j, 2]),
                         float(snap[j, 1] + 0.5 * snap[j, 3]))]
                if pc is not None:
                    anch.append((float(pc[j, 0]), float(pc[j, 1])))
                ok = self._skyline_flush(j, axis, 1 - side,
                                         lo_b if side else hi_b,
                                         x0, x1,
                                         lock, skip, anch)
                if ok:
                    sky_m = True
            if ok:
                skip[j] = False
            else:
                expel.append(j)
        band = [(x0, lo_b, W, hi_b - lo_b) if axis
                else (lo_b, x0, hi_b - lo_b, W)]
        for j in expel:
            if not self._compact_reloc(j, rects, False, avoid=band):
                P[...] = snap
                return False
        # exact-fill construction cannot clash inside the band; verify
        # the strip block itself landed inside the lock
        far = float(P[i, axis] + P[i, axis + 2]) if side \
            else float(P[i, axis])
        if (far > lock + 1e-7) if side else (far < lock - 1e-7):
            P[...] = snap
            return False
        return True

    def _compact_side(self, axis: int, side: int, lock: float,
                      rects, tags, reloc_left: int) -> int:
        """One evacuation round of the strip beyond a tag-locked frame
        edge.  Per over-block, in order: straight inward translate;
        rigid whole-cluster translate; flatten-and-widen reshape (soft,
        area-exact); dig-drop (relocate the movable occupants of the
        landing spot, then drop the block/cluster flush to the lock);
        plain relocation for untagged blocks.  Returns the number of
        relocations spent (-1 when the round made no progress)."""
        P = self.P
        opt = self.opt
        eps = 1e-6
        o = 1 - axis
        n = self.n
        fr_lo = (self.xmin, self.ymin)
        fr_hi = (self.xmax, self.ymax)
        lo = P[:, axis]
        far = lo + P[:, axis + 2]
        if side:
            over = [i for i in range(n) if far[i] > lock + eps]
        else:
            over = [i for i in range(n) if lo[i] < lock - eps]
        if not over:
            return -1
        bit = (4 if side else 8) if axis else (2 if side else 1)
        # side-tagged blocks first and LARGEST first (they must grab
        # the lock-flush landings before anything else fragments the
        # lock line — a small tagged block seated early once starved a
        # big one of its only contiguous flush rect); untagged small
        # next: their vacated slots become widening room
        over.sort(key=lambda i: (
            bool(self.in_cluster[i]),
            0 if tags.get(i, 0) & bit else 1,
            (-1.0 if tags.get(i, 0) & bit else 1.0)
            * float(P[i, 2] * P[i, 3])))
        prog = False
        spent = 0
        for i in over:
            # coordinates go stale after cluster translates — recheck
            lo_i = float(P[i, axis])
            far_i = lo_i + float(P[i, axis + 2])
            if (far_i <= lock + eps) if side else (lo_i >= lock - eps):
                continue
            w, h = float(P[i, 2]), float(P[i, 3])
            dims = (w, h)
            code = tags.get(i, 0)
            d = (lock - far_i) if side else (lock - lo_i)
            # 1. straight inward translate: cross coordinate untouched,
            #    far edge exactly on the lock — tag-safe on every side
            if self.kind[i] != 2 and not self.in_cluster[i]:
                pos = [float(P[i, 0]), float(P[i, 1])]
                pos[axis] = (lock - dims[axis]) if side else lock
                if pos[axis] >= fr_lo[axis] - eps \
                        and pos[axis] + dims[axis] <= fr_hi[axis] + eps \
                        and not self._rect_clash(i, pos[0], pos[1], w, h):
                    P[i, 0], P[i, 1] = pos
                    prog = True
                    continue
            # 2. rigid whole-cluster translate (contacts preserved by
            #    construction; groups holding a preplaced anchor are
            #    pinned by _build_groups and skipped here), with a
            #    dig-drop fallback that clears the landing first
            if self.in_cluster[i]:
                gi = int(self.group_of[i])
                if gi < 0:
                    continue
                g = self.groups[gi]
                if g.pin_y if axis else g.pin_x:
                    # the group holds the lock's own anchor (or a seated
                    # edge): the member must leave the strip alone and
                    # re-attach to the cluster body inside the lock —
                    # or, when the chain makes it an articulation block,
                    # the whole protruding sub-piece moves rigidly
                    if self._member_reland(i, axis, side, lock) \
                            or self._piece_reland(i, axis, side, lock):
                        prog = True
                    continue
                mem = list(g.members)
                if side:
                    dg = lock - max(float(P[j, axis] + P[j, axis + 2])
                                    for j in mem)
                else:
                    dg = lock - min(float(P[j, axis]) for j in mem)
                if abs(dg) <= eps:
                    continue
                ok = True
                ex = np.asarray(mem, dtype=np.int64)
                for j in mem:
                    p2 = [float(P[j, 0]), float(P[j, 1])]
                    p2[axis] += dg
                    if p2[axis] < fr_lo[axis] - eps \
                            or p2[axis] + float(P[j, axis + 2]) \
                            > fr_hi[axis] + eps \
                            or self._rect_clash(j, p2[0], p2[1],
                                                float(P[j, 2]),
                                                float(P[j, 3]), ex):
                        ok = False
                        break
                if ok:
                    self._move(g, dg, axis)
                    prog = True
                elif spent + 2 <= reloc_left \
                        and self._dig_drop(mem, dg, axis, rects):
                    prog = True
                    spent += 2
                elif self._member_reland(i, axis, side, lock) \
                        or self._piece_reland(i, axis, side, lock):
                    # translating the whole group is jammed — detach
                    # the protruding member (or its whole sub-piece);
                    # freeze the group after: its recorded contacts no
                    # longer match the layout
                    g.pin_x = g.pin_y = True
                    prog = True
                continue
            # 3. flatten-and-widen reshape in place: the lock-side edge
            #    lands exactly on the lock, the kept edge stays, and the
            #    exact area moves into the cross dimension.  A tagged
            #    cross edge must not move: grow away from it only.
            if self.kind[i] == 0 and opt.mib[i] <= 0:
                dim = (lock - lo_i) if side else (far_i - lock)
                if dim > eps:
                    area = float(opt.areas[i])
                    od = area / dim
                    cur = float(P[i, o])
                    grow = od - dims[o]
                    hi_bit, lo_bit = (2, 1) if o == 0 else (4, 8)
                    if code & hi_bit:
                        cands = [cur - grow]
                    elif code & lo_bit:
                        cands = [cur]
                    else:
                        cands = [cur, cur - grow, cur - 0.5 * grow]
                        # neighbour-edge candidates: the widened block
                        # must line up with whatever seam exists in the
                        # shelf strip — centre offsets almost never do
                        s_lo = lock - dim if side else lock
                        s_hi = lock if side else lock + dim
                        for j2 in range(n):
                            if j2 == i:
                                continue
                            j0 = float(P[j2, axis])
                            j1 = j0 + float(P[j2, axis + 2])
                            if min(j1, s_hi) - max(j0, s_lo) <= SEP_TOL:
                                continue
                            for t2 in (float(P[j2, o] + P[j2, o + 2]),
                                       float(P[j2, o]) - od):
                                if abs(t2 - cur) <= 3.0 * grow + 2.0:
                                    cands.append(t2)
                        cands = cands[:12]
                    done = False
                    for t in cands:
                        t = min(max(t, fr_lo[o]), fr_hi[o] - od)
                        if t < fr_lo[o] - eps:
                            continue
                        rect = [0.0, 0.0, 0.0, 0.0]
                        rect[axis] = lo_i if side else lock
                        rect[o] = t
                        rect[axis + 2] = dim
                        rect[o + 2] = od
                        if not self._rect_clash(i, rect[0], rect[1],
                                                rect[2], rect[3]):
                            P[i] = rect
                            prog = True
                            done = True
                            break
                    if done:
                        continue
            # 4. dig-drop: the shelf block belongs just inside the lock
            #    but untagged material occupies the spot — relocate the
            #    occupants and drop the block flush.  The only move that
            #    seats a TAGGED shelf block sitting wholly beyond the
            #    lock (relocation would forfeit its seat).
            if self.kind[i] != 2 and spent + 2 <= reloc_left \
                    and self._dig_drop([i], d, axis, rects):
                prog = True
                spent += 2
                continue
            # 4.5 band restack: rebuild the whole column under the
            #     strip block with exact-fill slices ending at the lock;
            #     big tagged blocks may need two neighbouring columns'
            #     worth of free area — widen the band before giving up
            if self.kind[i] == 0 and opt.mib[i] <= 0 \
                    and not self.in_cluster[i] and code in (0, bit) \
                    and spent + 3 <= reloc_left:
                done = False
                for widen in (1.0, 2.0, 3.5):
                    if self._band_restack(i, axis, side, lock,
                                          rects, tags, widen):
                        done = True
                        break
                    if not code:
                        break
                if done:
                    prog = True
                    spent += 3
                    continue
            # 5. relocation.  Untagged blocks land anywhere in the
            #    map; a block tagged ONLY on the compacted side lands
            #    flush on the lock line — exactly where the new layout
            #    edge will be, so its seat survives the move.  Any
            #    other tag forfeits its seat mid-layout: skip.
            if spent < reloc_left and self.kind[i] != 2:
                rigid = self.kind[i] != 0 or opt.mib[i] > 0
                if code == 0:
                    if self._compact_reloc(i, rects, rigid):
                        prog = True
                        spent += 1
                elif code == bit:
                    if self._compact_reloc(i, rects, rigid,
                                           flush=(axis, side, lock)):
                        prog = True
                        spent += 1
        return spent if prog else -1

    def _joint_restack(self, axis: int, side: int, lock: float,
                       tags, bit: int, ov: float, full: bool) -> bool:
        """Joint full-width restack of the over-strip, shallow depths
        first: seat every tagged shelf block together via composite
        flush landings while disturbing only the strip that must move.
        `full` adds the unbounded-depth attempt as a last resort."""
        P = self.P
        lo = P[:, axis]
        far = lo + P[:, axis + 2]
        cand = [j for j in range(self.n)
                if ((far[j] > lock + 1e-6) if side
                    else (lo[j] < lock - 1e-6))
                and self.kind[j] == 0 and self.opt.mib[j] <= 0
                and not self.in_cluster[j]
                and tags.get(j, 0) in (0, bit)]
        if not cand:
            return False
        jj = max(cand, key=lambda j: float(P[j, 2] * P[j, 3]))
        o2 = 1 - axis
        wb = (self.xmax - self.xmin) if o2 == 0 \
            else (self.ymax - self.ymin)
        t_area = sum(float(self.opt.areas[j]) for j in range(self.n)
                     if tags.get(j, 0) == bit and self.kind[j] == 0
                     and self.opt.mib[j] <= 0
                     and not self.in_cluster[j])
        d1 = max(2.5 * t_area / max(wb, 1e-6), 2.0 * ov)
        for dp in ((d1, 2.5 * d1, None) if full else (d1, 2.5 * d1)):
            rects = self._free_rects(14)
            if self._band_restack(jj, axis, side, lock, rects, tags,
                                  widen=1e9, depth=dp):
                return True
        return False

    def _shelf_tuck(self, axis: int, side: int, lock: float,
                    tags, bit: int, ov: float) -> bool:
        """Re-land the whole over-shelf into the pockets right beneath
        it, moving nothing else.  On dense winners the judge's prize
        is only the shaved bbox strip (<1% area), so the only
        affordable transaction is one whose HPWL bill is near zero:
        members stay put and act as depth-profile obstacles; small
        loose softs lodged in the shelf's pockets may be dug out and
        re-homed nearby.  All-or-nothing per attempt — a partial tuck
        can overlap the unseated remainder's original spots.  A greedy
        strict landing can starve a later shelf block (a fat shallow
        landing over a diggable rock eats the lock line), so a failed
        pass is retried with digging enabled from the first block."""
        P = self.P
        eps = 1e-6
        lo0 = P[:, axis]
        far0 = lo0 + P[:, axis + 2]
        over = [j for j in range(self.n)
                if ((far0[j] > lock + eps) if side
                    else (lo0[j] < lock - eps))]
        # cross-axis tags are corner blocks: land flush at the lock
        # with the cross position pinned to their tagged frame edge.
        # The OPPOSITE-axis tag would forfeit its seat — excluded.
        c_lo, c_hi = (1, 2) if axis else (8, 4)
        pin_of = {}
        cand = []
        for j in over:
            if self.kind[j] != 0 or self.opt.mib[j] > 0 \
                    or self.in_cluster[j]:
                continue
            code = tags.get(j, 0)
            if code & ~(bit | c_lo | c_hi):
                continue
            rem = code & (c_lo | c_hi)
            if rem == c_lo | c_hi:
                continue
            cand.append(j)
            pin_of[j] = 0 if rem == c_lo else (1 if rem == c_hi
                                               else None)
        # a shelf block that cannot tuck (cluster/kind1/corner jam)
        # keeps the bbox edge inflated: seating the rest at the lock
        # then pays HPWL for zero area prize and pulls tagged blocks
        # off the REAL layout edge — only tuck when the seal is total
        if not cand or len(cand) != len(over):
            return False
        snap00 = P.copy()
        pc = getattr(self, "_pred_c", None)
        o2 = 1 - axis
        c0f = (self.xmin, self.ymin)[o2]
        c1f = (self.xmax, self.ymax)[o2]
        fb = (self.xmin, self.ymin)[axis] if side \
            else (self.xmax, self.ymax)[axis]
        t_area = sum(float(self.opt.areas[j]) for j in cand)
        dstrip = max(2.5 * t_area / max(c1f - c0f, 1e-6), 2.0 * ov)
        s_lo = (lock - dstrip) if side else lock
        s_hi = lock if side else (lock + dstrip)
        # corners first (their cross position is forced — the greedy
        # mid-line landings must not eat their only spot)
        order = sorted(cand, key=lambda j: (pin_of[j] is None,
                                            -float(self.opt.areas[j])))

        def _anchors(j):
            a = [(float(snap00[j, 0] + 0.5 * snap00[j, 2]),
                  float(snap00[j, 1] + 0.5 * snap00[j, 3]))]
            if pc is not None:
                a.append((float(pc[j, 0]), float(pc[j, 1])))
            return a

        for dig_all in (False, True):
            P[...] = snap00
            skipm = np.zeros(self.n, dtype=bool)
            skipm[cand] = True
            displaced = []
            ok = True
            for j in order:
                anch = _anchors(j)
                if not dig_all and self._skyline_flush(
                        j, axis, side, lock, c0f, c1f, fb,
                        skipm, anch, pin=pin_of[j]):
                    skipm[j] = False
                    continue
                aj = float(self.opt.areas[j])
                digm = skipm.copy()
                for k in range(self.n):
                    if skipm[k] or self.kind[k] != 0 \
                            or self.opt.mib[k] > 0 \
                            or self.in_cluster[k] \
                            or tags.get(k, 0) != 0 \
                            or float(self.opt.areas[k]) > 0.75 * aj:
                        continue
                    if float(P[k, axis]) >= s_lo - 1e-9 \
                            and float(P[k, axis] + P[k, axis + 2]) \
                            <= s_hi + 1e-9:
                        digm[k] = True
                if not self._skyline_flush(j, axis, side, lock,
                                           c0f, c1f, fb, digm, anch,
                                           pin=pin_of[j]):
                    ok = False
                    break
                jx0, jy0, jw, jh = [float(v) for v in P[j]]
                for k in np.nonzero(digm & ~skipm)[0]:
                    if min(jx0 + jw, float(P[k, 0] + P[k, 2])) \
                            - max(jx0, float(P[k, 0])) > SEP_TOL \
                            and min(jy0 + jh,
                                    float(P[k, 1] + P[k, 3])) \
                            - max(jy0, float(P[k, 1])) > SEP_TOL:
                        skipm[k] = True
                        displaced.append(int(k))
                skipm[j] = False
            if ok and displaced:
                # re-home the dug-out blocks near their old spots
                # (or the prediction's)
                rects = self._free_rects(16, skip=skipm)
                for k in sorted(displaced,
                                key=lambda k: -float(
                                    self.opt.areas[k])):
                    ka = float(self.opt.areas[k])
                    bestk = None
                    for ri, (rx, ry, rw, rh) in enumerate(rects):
                        if rw * rh < ka * (1.0 + 1e-9):
                            continue
                        w_lo = max(ka / rh, 1e-6)
                        if w_lo > rw:
                            continue
                        for ax_, ay_ in _anchors(k):
                            w2 = min(max(math.sqrt(ka), w_lo), rw)
                            h2 = ka / w2
                            nx = min(max(ax_ - 0.5 * w2, rx),
                                     rx + rw - w2)
                            ny = min(max(ay_ - 0.5 * h2, ry),
                                     ry + rh - h2)
                            dd = abs(nx + 0.5 * w2 - ax_) \
                                + abs(ny + 0.5 * h2 - ay_)
                            if bestk is None or dd < bestk[0]:
                                bestk = (dd, ri, nx, ny, w2, h2)
                    if bestk is None:
                        ok = False
                        break
                    _dd, ri, nx, ny, w2, h2 = bestk
                    self._split_rect(rects, ri, nx, ny, w2, h2)
                    P[k] = (nx, ny, w2, h2)
                    skipm[k] = False
            if ok:
                if _DEBUG:
                    print(f"[compact] tuck a{axis}s{side} "
                          f"{len(cand)} blocks +{len(displaced)} dug"
                          f"{' dig-all' if dig_all else ''}",
                          flush=True)
                return True
            displaced = []
        P[...] = snap00
        return False

    def compact_to_locks(self, deadline: float, judge=None) -> bool:
        """Discrete area compaction toward the tag-locked frame.  The
        residual failure shape on saturated packings: the layout edge
        overshoots a preplaced block's tagged edge by units — a whole
        shelf of blocks sits flush past the lock with no gap, so no
        seat/trim/translate can fix it, while the matching free area
        lies scattered in slivers elsewhere.  Evacuate the strip block
        by block, recomputing the free-rect map every round so vacated
        slots become the next block's widening room.  Each side is one
        transaction judged by the caller's proxy: a partial evacuation
        leaves tags unseated at a still-inflated edge, so mid-states
        must never leak out."""
        P = self.P
        eps = 1e-6
        if self.lock_xmin is not None:
            self.xmin = max(self.xmin, self.lock_xmin)
        if self.lock_xmax is not None:
            self.xmax = min(self.xmax, self.lock_xmax)
        if self.lock_ymin is not None:
            self.ymin = max(self.ymin, self.lock_ymin)
        if self.lock_ymax is not None:
            self.ymax = min(self.ymax, self.lock_ymax)
        tags = {int(i): int(c)
                for i, c in zip(self.opt._bnd_idx, self.opt._bnd_codes)}
        changed = False
        for _sweep in range(2):
            sides = []
            for axis, side, lock in ((1, 1, self.lock_ymax),
                                     (0, 1, self.lock_xmax),
                                     (1, 0, self.lock_ymin),
                                     (0, 0, self.lock_xmin)):
                if lock is None:
                    continue
                ov = (float((P[:, axis] + P[:, axis + 2]).max()) - lock) \
                    if side else (lock - float(P[:, axis].min()))
                # sub-0.25 hovers are _edge_seat territory (translate/
                # dilate/trim); restructuring rounds there only burn
                # the recompression reserve
                if ov > 0.25:
                    sides.append((ov, axis, side, lock))
            if not sides:
                break
            sides.sort(reverse=True)
            any_ok = False
            # phase 1: tucks — near-zero-HPWL seals for EVERY side
            # before the expensive rounds machinery can eat the
            # deadline on the first side alone
            for _ov, axis, side, lock in sides:
                if time.time() >= deadline:
                    return changed
                snap = P.copy()
                base = judge(P) if judge is not None else None
                bit = (4 if side else 8) if axis else (2 if side else 1)
                if not self._shelf_tuck(axis, side, lock, tags, bit,
                                        _ov):
                    continue
                if self._has_overlap() or (judge is not None
                                           and judge(P)
                                           >= base - 1e-12):
                    P[...] = snap
                    if _DEBUG:
                        print(f"[compact] tuck a{axis}s{side} "
                              f"reverted", flush=True)
                else:
                    changed = True
                    any_ok = True
            # phase 2: full machinery on the sides still open
            for _ov, axis, side, lock in sides:
                if time.time() >= deadline:
                    return changed
                _ov = (float((P[:, axis] + P[:, axis + 2]).max())
                       - lock) if side \
                    else (lock - float(P[:, axis].min()))
                if _ov <= 0.25:
                    continue
                snap = P.copy()
                base = judge(P) if judge is not None else None
                bit = (4 if side else 8) if axis else (2 if side else 1)
                prog = False
                grav = 0
                # stage 00: retry the tuck bundled with the rounds —
                # a tuck whose HPWL bill alone loses to the judge can
                # still win once the rounds clean up behind it
                if self._shelf_tuck(axis, side, lock, tags, bit, _ov):
                    prog = True
                # stage 0: joint shallow restack — the rounds'
                # translate/dig/gravity churn costs HPWL the judge
                # must amortize, but on the saturated-shelf pattern a
                # near-in-place reseat of the whole tagged shelf seals
                # the side for almost nothing
                lo0 = P[:, axis]
                far0 = lo0 + P[:, axis + 2]
                sealed = (float(far0.max()) <= lock + eps) if side \
                    else (float(lo0.min()) >= lock - eps)
                if not sealed and self._joint_restack(axis, side, lock,
                                                      tags, bit, _ov,
                                                      full=False):
                    prog = True
                for _round in range(12):
                    if time.time() >= deadline:
                        break
                    # side sealed: stop before gravity drift burns
                    # HPWL for no further gain
                    lo1 = P[:, axis]
                    far1 = lo1 + P[:, axis + 2]
                    if ((float(far1.max()) <= lock + eps) if side
                            else (float(lo1.min()) >= lock - eps)):
                        break
                    rects = self._free_rects(14)
                    if self._compact_side(axis, side, lock,
                                          rects, tags, 8) >= 0:
                        prog = True
                        continue
                    # jammed: gravity coalesces the scattered slivers
                    # into a channel, then retry the per-block ops.
                    # Stage 0 pulls away from the lock (row-gap slack);
                    # stages 1/2 sweep the CROSS axis (column-gap slack
                    # — the common case: tall sliver seams between
                    # columns that same-axis gravity can never close).
                    moved = False
                    while grav < 3 and not moved:
                        if grav == 0:
                            moved = self._gravity(axis, 1 - side,
                                                  tags, bit)
                        elif grav == 1:
                            moved = self._gravity(1 - axis, 0,
                                                  tags, bit)
                        else:
                            moved = self._gravity(1 - axis, 1,
                                                  tags, bit)
                        grav += 1
                    if moved:
                        continue
                    # very last resort: ONE joint full-width restack.
                    # Per-block bands fail on order effects — the first
                    # seated tagged blocks fragment the lock line for
                    # the rest; the joint repack seats every tagged
                    # block together, largest first.
                    if grav < 4:
                        grav = 4
                        if self._joint_restack(axis, side, lock, tags,
                                               bit, _ov, full=True):
                            prog = True
                            continue
                    break
                if not prog:
                    P[...] = snap      # undo pure-gravity drift
                    continue
                if self._has_overlap() or (judge is not None
                                           and judge(P) >= base - 1e-12):
                    P[...] = snap
                    if _DEBUG:
                        print(f"[compact] side a{axis}s{side} reverted",
                              flush=True)
                else:
                    changed = True
                    any_ok = True
                    if _DEBUG:
                        print(f"[compact] side a{axis}s{side} ov={_ov:.2f} "
                              f"kept", flush=True)
            if not any_ok:
                break
        return changed

    def _rock_burst(self, cur_key: float, deadline: float) -> float:
        """Tag-pinned singletons are immovable rocks that block every
        squeeze, and per-step keyed accepts can never pay the tag cost.
        Un-pin everything once, squeeze hard (with overflow relocation
        allowed even for tagged blocks), then re-seat the tags — accept
        only if the TOTAL key improves, so the temporary V cost is
        amortized across all the squeezes it unlocks."""
        P = self.P
        snap = P.copy()
        frame = (self.xmin, self.xmax, self.ymin, self.ymax)
        pins = [(g.pin_x, g.pin_y) for g in self.groups]
        sat = (self.satL.copy(), self.satR.copy(),
               self.satB.copy(), self.satT.copy())
        # lift the pins but KEEP the sat marks: satR/satT groups follow
        # the shrinking edge inside _try_squeeze, so their seats survive
        # the burst; the pins were what froze the cascade
        for g in self.groups:
            if len(g.members) == 1:
                g.pin_x = g.pin_y = False
        self._squeeze_phase(self._key(),
                            min(deadline, time.time() + 1.2))
        self._reimpose_seeded_edges()
        ok = self.legalize(12, deadline=min(deadline, time.time() + 1.0))
        if ok and not self._has_overlap():
            k3 = self._key()
            if k3 < cur_key - 1e-12:
                if _DEBUG:
                    print(f"[burst] key {cur_key:.4f}->{k3:.4f} "
                          f"frame {self.xmax - self.xmin:.1f}x"
                          f"{self.ymax - self.ymin:.1f}", flush=True)
                return k3
            if _DEBUG:
                print(f"[burst] no gain {cur_key:.4f}->{k3:.4f}",
                      flush=True)
        elif _DEBUG:
            print(f"[burst] reseat failed ok={ok} "
                  f"ovl={self._overlap_count()}", flush=True)
        P[...] = snap
        self.xmin, self.xmax, self.ymin, self.ymax = frame
        for g, (px, py) in zip(self.groups, pins):
            g.pin_x, g.pin_y = px, py
        self.satL[:], self.satR[:], self.satB[:], self.satT[:] = sat
        return cur_key

    def _reshape_clear(self, m: int, a: int) -> bool:
        """Soft block m overlaps immovable a with a shallow penetration:
        shrink m's dim on the shallow axis so its edge lands exactly on
        a's edge, grow the other dim to preserve area, and accept only a
        collision-free result.  The escape for blocks too big for any
        free rectangle."""
        P = self.P
        if self.kind[m] != 0 or self.in_cluster[m]:
            return False
        if hasattr(self.opt, "mib") and self.opt.mib[m] > 0:
            return False
        area = float(self.opt.areas[m])
        x0, y0 = P[:, 0], P[:, 1]
        x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
        ox = min(x1[m], x1[a]) - max(x0[m], x0[a])
        oy = min(y1[m], y1[a]) - max(y0[m], y0[a])
        if ox <= SEP_TOL or oy <= SEP_TOL:
            return True
        axis = 1 if oy <= ox else 0          # shrink the shallow axis
        o = 1 - axis
        lo_m, hi_m = float(P[m, axis]), float(P[m, axis] + P[m, axis + 2])
        lo_a, hi_a = float(P[a, axis]), float(P[a, axis] + P[a, axis + 2])
        snap = P[m].copy()
        for new_lo, new_hi in ((lo_m, lo_a), (hi_a, hi_m)):
            dim = new_hi - new_lo - 1e-7
            if dim < 0.04 * (hi_m - lo_m):
                continue
            other = area / dim
            grow = other - float(P[m, o + 2])
            no = float(P[m, o]) - 0.5 * grow
            lo_f = self.xmin if o == 0 else self.ymin
            hi_f = self.xmax if o == 0 else self.ymax
            no = min(max(no, lo_f), hi_f - other)
            if no < lo_f - 1e-9:
                continue
            P[m, axis] = new_lo + 1e-7 * 0.5
            P[m, axis + 2] = dim
            P[m, o] = no
            P[m, o + 2] = other
            nx0, ny0 = P[m, 0], P[m, 1]
            nx1, ny1 = nx0 + P[m, 2], ny0 + P[m, 3]
            cx = np.minimum(nx1, x1) - np.maximum(nx0, x0)
            cy = np.minimum(ny1, y1) - np.maximum(ny0, y0)
            clash = (cx > SEP_TOL) & (cy > SEP_TOL)
            clash[m] = False
            if not clash.any():
                return True
            P[m] = snap
        return False

    def _chain_ok(self, j: int) -> bool:
        """Eligible as a link in a reshape chain: free soft, no cluster,
        no MIB, and not tag-pinned (unless tags are being ignored)."""
        if self.kind[j] != 0 or self.in_cluster[j]:
            return False
        if hasattr(self.opt, "mib") and self.opt.mib[j] > 0:
            return False
        if getattr(self, "_ignore_tags", False):
            return True
        if self.satL[j] or self.satR[j] or self.satB[j] or self.satT[j]:
            return False
        gi = int(self.group_of[j])
        return gi < 0 or not (self.groups[gi].pin_x
                              or self.groups[gi].pin_y)

    def _reshape_chain(self, m: int, a: int, depth: int = 2,
                       _visited=None) -> bool:
        """Multi-block extension of _reshape_clear for saturated wedges:
        shrinking m's penetrated axis must grow the other axis, which on
        a saturated packing lands on a soft neighbor — reshape that
        neighbor too, recursively, and accept only a chain that ends
        with zero new overlap.  Tries the shallow axis first (least
        material moved), then the deep axis.  Restores every touched
        block on failure."""
        P = self.P
        if not self._chain_ok(m):
            return False
        if _visited is None:
            _visited = {a}
        if m in _visited or depth < 0:
            return False
        _visited = _visited | {m}
        area = float(self.opt.areas[m])
        x0, y0 = P[:, 0], P[:, 1]
        x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
        ox = min(x1[m], x1[a]) - max(x0[m], x0[a])
        oy = min(y1[m], y1[a]) - max(y0[m], y0[a])
        if ox <= SEP_TOL or oy <= SEP_TOL:
            return True
        ax_shallow = 1 if oy <= ox else 0
        snap_all = P.copy()
        for axis in (ax_shallow, 1 - ax_shallow):
            o = 1 - axis
            lo_m = float(P[m, axis])
            hi_m = lo_m + float(P[m, axis + 2])
            lo_a = float(P[a, axis])
            hi_a = lo_a + float(P[a, axis + 2])
            for new_lo, new_hi in ((lo_m, lo_a), (hi_a, hi_m)):
                dim = new_hi - new_lo - 1e-7
                if dim < 0.03 * (hi_m - lo_m):
                    continue
                other = area / dim
                grow = other - float(P[m, o + 2])
                no = float(P[m, o]) - 0.5 * grow
                lo_f = self.xmin if o == 0 else self.ymin
                hi_f = self.xmax if o == 0 else self.ymax
                no = min(max(no, lo_f), hi_f - other)
                if no < lo_f - 1e-9 or no + other > hi_f + 1e-9:
                    continue
                P[m, axis] = new_lo + 1e-7 * 0.5
                P[m, axis + 2] = dim
                P[m, o] = no
                P[m, o + 2] = other
                cx0, cy0 = P[:, 0], P[:, 1]
                cx1 = cx0 + P[:, 2]
                cy1 = cy0 + P[:, 3]
                cx = (np.minimum(cx1[m], cx1) - np.maximum(cx0[m], cx0))
                cy = (np.minimum(cy1[m], cy1) - np.maximum(cy0[m], cy0))
                clash = (cx > SEP_TOL) & (cy > SEP_TOL)
                clash[m] = False
                idxs = np.nonzero(clash)[0]
                if not len(idxs):
                    self.satL[m] = self.satR[m] = False
                    self.satB[m] = self.satT[m] = False
                    return True
                if depth > 0 and len(idxs) <= 2 \
                        and all(self._chain_ok(int(j))
                                and int(j) not in _visited
                                for j in idxs):
                    if all(self._reshape_chain(int(j), m, depth - 1,
                                               _visited)
                           for j in idxs):
                        self.satL[m] = self.satR[m] = False
                        self.satB[m] = self.satT[m] = False
                        return True
                P[...] = snap_all
        return False

    def _overlap_pairs(self, limit: int = 12):
        """List up to `limit` currently overlapping pairs."""
        P = self.P
        x0, y0 = P[:, 0], P[:, 1]
        x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
        out = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (min(x1[i], x1[j]) - max(x0[i], x0[j]) > SEP_TOL
                        and min(y1[i], y1[j]) - max(y0[i], y0[j]) > SEP_TOL):
                    out.append((i, j))
                    if len(out) >= limit:
                        return out
        return out

    def _pull_inside_frame(self):
        """Evict free blocks that stick out past the current frame (used
        after clamping the frame onto tag locks).  _evict only accepts
        overlap-free slots inside the frame, so this never creates new
        overlap; blocks it cannot place go to recorded free rectangles,
        and only then are left for legalize to push."""
        P = self.P
        rects = None
        for i in range(self.n):
            if self.kind[i] == 2 or self.in_cluster[i]:
                continue
            if (P[i, 0] < self.xmin - 1e-9
                    or P[i, 0] + P[i, 2] > self.xmax + 1e-9
                    or P[i, 1] < self.ymin - 1e-9
                    or P[i, 1] + P[i, 3] > self.ymax + 1e-9):
                if self._evict(i):
                    continue
                if self._movable_free(i):
                    if rects is None:
                        rects = self._free_rects(10)
                    self._relocate_to_free(i, rects)

    def _reimpose_seeded_edges(self):
        P = self.P
        for i in np.nonzero(self.satL)[0]:
            P[i, 0] = self.xmin
        for i in np.nonzero(self.satR)[0]:
            P[i, 0] = self.xmax - P[i, 2]
        for i in np.nonzero(self.satB)[0]:
            P[i, 1] = self.ymin
        for i in np.nonzero(self.satT)[0]:
            P[i, 1] = self.ymax - P[i, 3]

    def legalize_soft(self, max_sweeps: int = 14,
                      deadline: Optional[float] = None,
                      fine: bool = False) -> bool:
        """Robust legalization for scattered predictions: temporarily
        shrink every free soft block (density drops ~25%, so the push
        sweeps almost always succeed), then grow back stepwise while
        re-legalizing.  Unlike the saturated column layouts (where regrow
        provably cannot fit), a prediction-shaped layout has its slack
        distributed everywhere, so the growth is absorbed locally."""
        P = self.P
        soft = (self.kind == 0) & ~self.in_cluster
        idx = np.nonzero(soft)[0]
        if not len(idx):
            return self.legalize(max_sweeps * 3, deadline=deadline)
        areas = np.array([self.opt.areas[i] for i in idx])
        for s in ((0.86, 0.93, 0.965, 0.985, 1.0) if fine
                  else (0.88, 0.94, 0.97, 1.0)):
            # re-derive dims from target area (scaled s^2) and the CURRENT
            # aspect — evictions inside legalize() reshape blocks, and any
            # absolute restore would divorce dims from positions
            aspect = P[idx, 2] / np.maximum(P[idx, 3], 1e-9)
            na = areas * (s * s)
            nw = np.sqrt(na * aspect)
            nh = na / nw
            P[idx, 0] += 0.5 * (P[idx, 2] - nw)
            P[idx, 1] += 0.5 * (P[idx, 3] - nh)
            P[idx, 2] = nw
            P[idx, 3] = nh
            self._reimpose_seeded_edges()
            ok = self.legalize(max_sweeps if s < 1.0 else 3 * max_sweeps,
                               deadline=deadline)
            if not ok and s == 1.0 and self._overlap_count() <= 10 \
                    and (deadline is None or time.time() < deadline):
                # the regrow usually dies within a pair or two of closing —
                # one more sweep block with fresh kick phases often finishes
                ok = self.legalize(2 * max_sweeps, deadline=deadline)
            if not ok and s == 1.0 \
                    and (deadline is None or time.time() < deadline):
                # last resort: the residual overlaps are almost always a
                # tag-pinned soft wedged against a preplaced block with
                # sub-unit penetration — the pin blocks every escape and
                # _evict skips pinned groups.  Un-pin the involved
                # singleton groups (a lost tag seat costs V+=1 and the
                # repair pass may re-seat it; a failed candidate costs
                # everything) and push again, evicting as a final kick.
                pairs = self._overlap_pairs(12)
                if 0 < len(pairs) <= 6:
                    for i, j in pairs:
                        for b in (i, j):
                            gi = int(self.group_of[b])
                            if gi < 0 or self.in_cluster[b]:
                                continue
                            g = self.groups[gi]
                            if len(g.members) == 1 and (g.pin_x or g.pin_y):
                                g.pin_x = g.pin_y = False
                                self.satL[b] = self.satR[b] = False
                                self.satB[b] = self.satT[b] = False
                    ok = self.legalize(2 * max_sweeps, deadline=deadline)
                    if not ok:
                        rects = self._free_rects(8)
                        for i, j in self._overlap_pairs(12):
                            if self.kind[i] == 2 \
                                    and (self._reshape_clear(j, i)
                                         or self._reshape_chain(j, i)):
                                continue
                            if self.kind[j] == 2 \
                                    and (self._reshape_clear(i, j)
                                         or self._reshape_chain(i, j)):
                                continue
                            b = j if (P[j, 2] * P[j, 3] <= P[i, 2] * P[i, 3]
                                      or self.kind[i] == 2) else i
                            if self.kind[b] == 2 or self.in_cluster[b]:
                                continue
                            # recorded free space first, local slot second
                            if not (self._movable_free(b, allow_fixed=True)
                                    and self._relocate_to_free(b, rects)):
                                if not self._evict(b, max_anchors=self.n):
                                    # saturated wedge between softs: a
                                    # multi-block reshape chain is the
                                    # only tool that restructures in place
                                    a = i if b == j else j
                                    self._reshape_chain(b, a)
                        ok = self.legalize(max_sweeps, deadline=deadline)
            if _DEBUG:
                print(f"[lsoft] s={s} ok={ok} ovl={self._overlap_count()}",
                      flush=True)
                if not ok and s == 1.0 and self._overlap_count() <= 6:
                    x0, y0 = P[:, 0], P[:, 1]
                    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
                    for i in range(self.n):
                        for j in range(i + 1, self.n):
                            ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
                            oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
                            if ox > SEP_TOL and oy > SEP_TOL:
                                gi = int(self.group_of[i])
                                gj = int(self.group_of[j])
                                pi = (f"{int(self.groups[gi].pin_x)}"
                                      f"{int(self.groups[gi].pin_y)}"
                                      if gi >= 0 else "--")
                                pj = (f"{int(self.groups[gj].pin_x)}"
                                      f"{int(self.groups[gj].pin_y)}"
                                      if gj >= 0 else "--")
                                print(f"[lsoft-ovl] {i}(k{self.kind[i]} "
                                      f"clu{int(self.in_cluster[i])} p{pi}) "
                                      f"x {j}(k{self.kind[j]} "
                                      f"clu{int(self.in_cluster[j])} p{pj}) "
                                      f"ox={ox:.2f} oy={oy:.2f}", flush=True)
            if not ok and s == 1.0:
                return False
        return not self._has_overlap()

    def _tighten(self, deadline: float, rounds: int = 10):
        """Anneal the frame down after a successful legalization: shrink
        both max edges a few percent, move the edge-pinned tag blocks onto
        the new edges, and re-legalize (evictions included).  Stops at the
        tightest frame that still legalizes — far stronger than local
        squeezing because every round may restructure the packing."""
        P = self.P
        for _ in range(rounds):
            if time.time() >= deadline:
                break
            bx1 = float((P[:, 0] + P[:, 2]).max())
            by1 = float((P[:, 1] + P[:, 3]).max())
            done = False
            for f in (0.97, 0.988):
                snap = P.copy()
                frame = (self.xmin, self.xmax, self.ymin, self.ymax)
                nx = self.xmin + (bx1 - self.xmin) * f
                ny = self.ymin + (by1 - self.ymin) * f
                if self.lock_xmax is not None:
                    nx = max(nx, self.lock_xmax)
                if self.lock_ymax is not None:
                    ny = max(ny, self.lock_ymax)
                if nx >= self.xmax - 1e-9 and ny >= self.ymax - 1e-9:
                    break
                moved = set()
                for i in np.nonzero(self.satR)[0]:
                    gi = int(self.group_of[i])
                    if gi >= 0 and gi not in moved:
                        moved.add(gi)
                        self._move(self.groups[gi],
                                   nx - (float(P[i, 0]) + float(P[i, 2])), 0)
                moved = set()
                for i in np.nonzero(self.satT)[0]:
                    gi = int(self.group_of[i])
                    if gi >= 0 and gi not in moved:
                        moved.add(gi)
                        self._move(self.groups[gi],
                                   ny - (float(P[i, 1]) + float(P[i, 3])), 1)
                self.xmax = nx
                self.ymax = ny
                if self.legalize_soft(10, deadline=deadline):
                    done = True
                    break
                P[...] = snap
                self.xmin, self.xmax, self.ymin, self.ymax = frame
            if not done:
                break
        if _DEBUG:
            print(f"[tighten] frame {self.xmax - self.xmin:.1f}"
                  f"x{self.ymax - self.ymin:.1f}", flush=True)

    def _assemble_clusters(self, cluster_groups):
        """After legalization, cluster members sit near each other but not
        exactly abutted (the evaluator's shapely union needs exact touch).
        Greedy assembly: grow each group from its locked anchor / largest
        member, moving one member at a time onto an exact-contact position
        beside the nearest already-placed member (skipping any move that
        would overlap outsiders)."""
        P = self.P
        for idxs in cluster_groups.values():
            mem = [int(i) for i in idxs]
            if len(mem) < 2:
                continue
            placed = [i for i in mem if self.kind[i] == 2]
            free = sorted((i for i in mem if self.kind[i] != 2),
                          key=lambda i: -(P[i, 2] * P[i, 3]))
            if not placed:
                if not free:
                    continue
                placed = [free.pop(0)]
            while free:
                bd = None
                for i in free:
                    ci = (P[i, 0] + 0.5 * P[i, 2], P[i, 1] + 0.5 * P[i, 3])
                    for j in placed:
                        d = max(0.0, abs(ci[0] - (P[j, 0] + 0.5 * P[j, 2]))
                                - 0.5 * (P[i, 2] + P[j, 2])) \
                            + max(0.0, abs(ci[1] - (P[j, 1] + 0.5 * P[j, 3]))
                                  - 0.5 * (P[i, 3] + P[j, 3]))
                        if bd is None or d < bd[0]:
                            bd = (d, i, j)
                _d, i, j = bd
                free.remove(i)
                self._snap_member(i, j, placed)
                placed.append(i)

    def _snap_member(self, i: int, j: int, placed) -> bool:
        """Place block i in exact edge contact with block j (or another
        placed member), choosing the least-displacement collision-free
        side.  Coordinates are set exactly equal so the contact survives
        the evaluator's exact-touch test."""
        P = self.P
        anchors = [j] + [a for a in placed if a != j]
        x0a = P[:, 0]
        y0a = P[:, 1]
        x1a = x0a + P[:, 2]
        y1a = y0a + P[:, 3]
        wi, hi = float(P[i, 2]), float(P[i, 3])
        best = None
        for a in anchors[:4]:
            xa, ya, wa, ha = [float(v) for v in P[a]]
            span_x = (max(self.xmin, xa - wi + min(wi, wa) * 0.5),
                      min(self.xmax - wi, xa + wa - min(wi, wa) * 0.5))
            span_y = (max(self.ymin, ya - hi + min(hi, ha) * 0.5),
                      min(self.ymax - hi, ya + ha - min(hi, ha) * 0.5))
            cands = []
            cy = min(max(P[i, 1], span_y[0]), span_y[1])
            # least-displacement alignment first, then edge-aligned
            # fallbacks — on dense layouts the centered slot is often
            # blocked by a corner while a flush-aligned one is free
            for yy in (cy, ya, ya + ha - hi):
                yy = min(max(yy, span_y[0]), span_y[1])
                cands.append((xa + wa, yy, wi, hi))    # right of a
                cands.append((xa - wi, yy, wi, hi))    # left of a
            cx = min(max(P[i, 0], span_x[0]), span_x[1])
            for xx in (cx, xa, xa + wa - wi):
                xx = min(max(xx, span_x[0]), span_x[1])
                cands.append((xx, ya + ha, wi, hi))    # above a
                cands.append((xx, ya - hi, wi, hi))    # below a
            # reshape fallbacks for soft non-MIB members: match the
            # anchor's edge exactly (soft area is only checked to 1%, and
            # contact needs exact-length abutment anyway)
            mib_i = (self.opt.mib[i] if hasattr(self.opt, "mib") else 0)
            if self.kind[i] == 0 and mib_i <= 0:
                area_i = float(self.opt.areas[i])
                nh = ha
                nw = area_i / nh
                cands.append((xa + wa, ya, nw, nh))    # right, h-matched
                cands.append((xa - nw, ya, nw, nh))    # left, h-matched
                nw2 = wa
                nh2 = area_i / nw2
                cands.append((xa, ya + ha, nw2, nh2))  # above, w-matched
                cands.append((xa, ya - nh2, nw2, nh2))  # below, w-matched
            for (nx, ny, nw_, nh_) in cands:
                if nx < self.xmin - 1e-9 or nx + nw_ > self.xmax + 1e-9 \
                        or ny < self.ymin - 1e-9 or ny + nh_ > self.ymax + 1e-9:
                    continue
                ox = np.minimum(nx + nw_, x1a) - np.maximum(nx, x0a)
                oy = np.minimum(ny + nh_, y1a) - np.maximum(ny, y0a)
                clash = (ox > SEP_TOL) & (oy > SEP_TOL)
                clash[i] = False
                if clash.any():
                    continue
                disp = abs(nx - float(P[i, 0])) + abs(ny - float(P[i, 1])) \
                    + (2.0 if nw_ != wi else 0.0)
                if best is None or disp < best[0]:
                    best = (disp, nx, ny, nw_, nh_)
        if best is None:
            return False
        P[i] = (best[1], best[2], best[3], best[4])
        return True

    def _try_squeeze(self, axis: int, delta: float, cur_key: float,
                     deadline: Optional[float] = None):
        """Shrink the frame's max edge by delta: blocks pinned to that edge
        move with it (so their tags stay satisfied at the new, smaller
        bbox), everything else is re-legalized inside the tighter frame.
        Pays off directly in the area term; reverts when the packing does
        not fit."""
        P = self.P
        lock = self.lock_xmax if axis == 0 else self.lock_ymax
        if lock is not None and (self.xmax if axis == 0
                                 else self.ymax) - delta < lock - 1e-9:
            # squeeze may land exactly on the tagged-preplaced edge (that
            # satisfies its tag) but never cross it
            delta = (self.xmax if axis == 0 else self.ymax) - lock
            if delta <= 1e-9:
                return False, cur_key
        snap = P.copy()
        sat = self.satR if axis == 0 else self.satT
        old_max = self.xmax if axis == 0 else self.ymax
        lim_lo = self.xmin if axis == 0 else self.ymin
        if old_max - delta <= lim_lo + 1.0:
            return False, cur_key
        moved = set()
        for i in np.nonzero(sat)[0]:
            gi = int(self.group_of[i])
            if gi >= 0 and gi not in moved:
                moved.add(gi)
                self._move(self.groups[gi], -delta, axis)
        if axis == 0:
            self.xmax = old_max - delta
        else:
            self.ymax = old_max - delta
        # corner-tagged singletons pinned on both axes are immovable rocks
        # mid-field; let them (and everyone) evade along the orthogonal
        # axis during the squeeze — a drifted tag shows up as V in the key
        # and the attempt reverts, so this is safe.  Direct-pipeline
        # layouts only (enable_deflate): on saturated column packings the
        # squeeze almost never succeeds, and the extra sweeps just burn
        # budget the discrete moves need.
        if self.enable_deflate:
            saved_pins = [(g.pin_x, g.pin_y) for g in self.groups]
            for g in self.groups:
                if len(g.members) == 1:
                    if axis == 0:
                        g.pin_y = False
                    else:
                        g.pin_x = False
            self.legalize(max_sweeps=12, deadline=deadline)
            if self._has_overlap():
                # squeeze-with-overflow: relocate the still-overlapping
                # unconstrained blocks into recorded free rectangles
                # (inside the already-shrunken frame) instead of giving
                # the whole attempt up
                rects = None
                moved = 0
                for a, b in self._overlap_pairs(8):
                    mb = self._movable_free(b, allow_fixed=True)
                    ma = self._movable_free(a, allow_fixed=True)
                    m = b if (mb and (not ma or P[b, 2] * P[b, 3]
                                      <= P[a, 2] * P[a, 3])) else a
                    if not self._movable_free(m, allow_fixed=True):
                        continue
                    if rects is None:
                        rects = self._free_rects(10)
                    if self._relocate_to_free(int(m), rects):
                        moved += 1
                if moved:
                    self.legalize(max_sweeps=6, deadline=deadline)
                if _DEBUG and os.environ.get("SQUEEZE_DEBUG"):
                    print(f"[sqz-ovf] ax={axis} d={delta:.2f} moved={moved} "
                          f"ovl={self._overlap_count()}", flush=True)
            for g, (px, py) in zip(self.groups, saved_pins):
                g.pin_x, g.pin_y = px, py
        else:
            self.legalize(max_sweeps=8, deadline=deadline)
        if not self._has_overlap():
            k = self._key()
            if k < cur_key - 1e-12:
                return True, k
        elif _DEBUG and os.environ.get("SQUEEZE_DEBUG"):
            for a, b in self._overlap_pairs(4):
                ga, gb = int(self.group_of[a]), int(self.group_of[b])
                pa = (f"{int(self.groups[ga].pin_x)}"
                      f"{int(self.groups[ga].pin_y)}" if ga >= 0 else "--")
                pb = (f"{int(self.groups[gb].pin_x)}"
                      f"{int(self.groups[gb].pin_y)}" if gb >= 0 else "--")
                print(f"[sqz-ovl] ax={axis} d={delta:.2f} "
                      f"{a}(k{self.kind[a]} p{pa}) x {b}(k{self.kind[b]} "
                      f"p{pb})", flush=True)
        self.P[...] = snap
        if axis == 0:
            self.xmax = old_max
        else:
            self.ymax = old_max
        return False, cur_key

    def _deflate(self, cur_key: float, deadline: float, rounds: int = 8):
        """Break saturated chains that block the squeeze: find the band
        with the highest width-coverage, move its smallest free block into
        the emptiest band, and re-squeeze.  Only kept when the proxy
        improves.  On layouts that are saturated by construction (column
        packing) every eviction fails fast, so this is a no-op there."""
        P = self.P
        n = self.n
        free = np.array([
            self.kind[i] != 2 and not self.in_cluster[i]
            and not (self.groups[self.group_of[i]].pin_x
                     or self.groups[self.group_of[i]].pin_y)
            for i in range(n)])
        # choice -2: rock burst — un-pin everything, squeeze hard, re-seat
        # tags, keep only on total-key improvement
        if time.time() < deadline:
            cur_key = self._rock_burst(cur_key, deadline)

        # choice -1: sliver rescue — blocks squashed into extreme aspects
        # (by evictions) wedge between their neighbors and jam every
        # squeeze; relocate them into recorded free rectangles with a
        # healthy shape, then re-squeeze
        if time.time() < deadline:
            slivers = [i for i in range(n) if free[i]
                       and self._movable_free(i)
                       and (P[i, 2] > 5.0 * P[i, 3]
                            or P[i, 3] > 5.0 * P[i, 2])]
            if slivers:
                snap = P.copy()
                frame = (self.xmin, self.xmax, self.ymin, self.ymax)
                rects = self._free_rects(10)
                moved = 0
                for i in slivers[:6]:
                    if self._relocate_to_free(int(i), rects):
                        moved += 1
                if moved and not self._has_overlap():
                    k2 = self._squeeze_phase(self._key(),
                                             min(deadline,
                                                 time.time() + 0.6))
                    if _DEBUG:
                        print(f"[deflate-fs] slivers moved={moved} "
                              f"k2={k2:.4f} cur={cur_key:.4f}", flush=True)
                    if k2 < cur_key - 1e-12:
                        cur_key = k2
                    else:
                        P[...] = snap
                        (self.xmin, self.xmax,
                         self.ymin, self.ymax) = frame
                else:
                    P[...] = snap
                    self.xmin, self.xmax, self.ymin, self.ymax = frame

        for _ in range(rounds):
            if time.time() >= deadline:
                break
            improved = False
            for axis in (0, 1):
                o = 1 - axis
                span = (self.xmax - self.xmin) if axis == 0 \
                    else (self.ymax - self.ymin)
                lines = P[:, o] + 0.5 * P[:, o + 2]
                f0 = P[:, o]
                f1 = f0 + P[:, o + 2]
                hit = (f0[None, :] <= lines[:, None]) & (f1[None, :] > lines[:, None])
                cover = (hit * P[None, :, axis + 2]).sum(axis=1)
                worst = int(np.argmax(cover))
                if _DEBUG:
                    print(f"[deflate] axis={axis} max_cover="
                          f"{float(cover[worst]) / span:.3f}", flush=True)
                if cover[worst] < 0.94 * span:
                    continue
                band = [i for i in np.nonzero(hit[worst])[0] if free[i]]

                mib = np.array(self.opt.mib[:n]) if hasattr(self.opt, "mib") \
                    else np.zeros(n)

                # choice 0: slack-aware precise narrowing — only narrow
                # band softs whose cross-axis free room provably absorbs
                # the height growth, and only by as much as the squeeze
                # actually needs.  Unlike the blind 15% cut below, this
                # never depends on the legalize lottery.
                need = 0.015 * span
                cand = []
                a0 = P[:, axis]
                a1 = a0 + P[:, axis + 2]
                b0 = P[:, o]
                b1 = b0 + P[:, o + 2]
                lo_o = self.xmin if o == 0 else self.ymin
                hi_o = self.xmax if o == 0 else self.ymax
                for i in band:
                    if self.kind[i] != 0 or mib[i] > 0:
                        continue
                    ov_a = ((np.minimum(a1[i], a1) - np.maximum(a0[i], a0))
                            > SEP_TOL)
                    ov_a[i] = False
                    up_n = ov_a & (b0 >= b1[i] - SEP_TOL)
                    dn_n = ov_a & (b1 <= b0[i] + SEP_TOL)
                    up = (float(b0[up_n].min()) if up_n.any() else hi_o) \
                        - float(b1[i])
                    dn = float(b0[i]) \
                        - (float(b1[dn_n].max()) if dn_n.any() else lo_o)
                    room = max(0.0, up) + max(0.0, dn)
                    if room <= SEP_TOL:
                        continue
                    w_i = float(P[i, axis + 2])
                    h_i = float(P[i, o + 2])
                    area_i = float(self.opt.areas[i])
                    dw = min(w_i - area_i / (h_i + 0.9 * room), 0.35 * w_i)
                    if dw > 1e-6:
                        cand.append((dw, i, max(0.0, up), max(0.0, dn)))
                if cand:
                    cand.sort(reverse=True)
                    snap = P.copy()
                    frame = (self.xmin, self.xmax, self.ymin, self.ymax)
                    removed = 0.0
                    for dw, i, up, dn in cand[:6]:
                        dw = min(dw, max(need - removed, 0.25 * need))
                        w_i = float(P[i, axis + 2])
                        area_i = float(self.opt.areas[i])
                        nw = w_i - dw
                        nh = area_i / nw
                        grow = nh - float(P[i, o + 2])
                        if grow > up + dn - 1e-9:
                            continue
                        # grow upward into the free gap first, then shift
                        # down for the remainder — guaranteed collision-free
                        P[i, o] -= max(0.0, grow - up)
                        P[i, axis + 2] = nw
                        P[i, o + 2] = nh
                        removed += dw
                        if removed >= need:
                            break
                    if removed > 1e-9 and not self._has_overlap():
                        k2 = self._squeeze_phase(self._key(),
                                                 min(deadline,
                                                     time.time() + 0.6))
                        if _DEBUG:
                            print(f"[deflate-sl] axis={axis} "
                                  f"removed={removed:.2f} k2={k2:.4f} "
                                  f"cur={cur_key:.4f}", flush=True)
                        if k2 < cur_key - 1e-12:
                            cur_key = k2
                            improved = True
                            break
                    P[...] = snap
                    self.xmin, self.xmax, self.ymin, self.ymax = frame

                # first choice: narrow the band's soft blocks in place —
                # aspect is free, the block stays put (no HPWL cost), and
                # the chain shortens by the width taken out
                softs = sorted((i for i in band
                                if self.kind[i] == 0 and mib[i] <= 0),
                               key=lambda i: -P[i, axis + 2])
                if softs:
                    snap = P.copy()
                    frame = (self.xmin, self.xmax, self.ymin, self.ymax)
                    for i in softs[:6]:
                        area_i = float(self.opt.areas[i])
                        nw = 0.85 * float(P[i, axis + 2])
                        nh = area_i / nw
                        nc = [0.0, 0.0]
                        nc[axis] = P[i, axis] + 0.5 * (P[i, axis + 2] - nw)
                        nc[o] = P[i, o] + 0.5 * (P[i, o + 2] - nh)
                        lo_a = self.xmin if axis == 0 else self.ymin
                        hi_a = (self.xmax if axis == 0 else self.ymax)
                        lo_o = self.xmin if o == 0 else self.ymin
                        hi_o = (self.xmax if o == 0 else self.ymax)
                        nc[axis] = min(max(nc[axis], lo_a), hi_a - nw)
                        nc[o] = min(max(nc[o], lo_o), hi_o - nh)
                        P[i, axis] = nc[axis]
                        P[i, o] = nc[o]
                        P[i, axis + 2] = nw
                        P[i, o + 2] = nh
                    # let min-displacement legalization (with deadlock
                    # kicks and evictions) absorb the height growth
                    self.legalize(max_sweeps=12, deadline=deadline)
                    if not self._has_overlap():
                        k_mid = self._key()
                        k2 = self._squeeze_phase(k_mid,
                                                 min(deadline, time.time() + 0.6))
                        if _DEBUG:
                            hp_ = self.opt._hpwl(P)
                            V_ = self.opt._violations(P)
                            ar_ = ((P[:, 0] + P[:, 2]).max() - P[:, 0].min()) \
                                * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min())
                            print(f"[deflate-rs] cur={cur_key:.4f} "
                                  f"mid={k_mid:.4f} k2={k2:.4f} "
                                  f"hp={hp_:.1f} V={V_} bbox={ar_:.0f} "
                                  f"aref={self.opt.area_ref:.0f}", flush=True)
                        if k2 < cur_key - 1e-12:
                            cur_key = k2
                            improved = True
                            break
                    elif _DEBUG:
                        print("[deflate-rs] hold-legalize failed", flush=True)
                    P[...] = snap
                    self.xmin, self.xmax, self.ymin, self.ymax = frame

                # fallback: move the smallest band member into the
                # emptiest band
                empty_line = float(lines[int(np.argmin(cover))])
                mid = 0.5 * ((self.xmin + self.xmax) if axis == 0
                             else (self.ymin + self.ymax))
                target = (mid, empty_line) if axis == 0 else (empty_line, mid)
                band.sort(key=lambda i: P[i, 2] * P[i, 3])
                for i in band[:3]:
                    snap = P.copy()
                    frame = (self.xmin, self.xmax, self.ymin, self.ymax)
                    if not self._evict(int(i), target=target):
                        continue
                    if self._has_overlap():
                        P[...] = snap
                        continue
                    k2 = self._squeeze_phase(self._key(),
                                             min(deadline, time.time() + 0.6))
                    if k2 < cur_key - 1e-12:
                        cur_key = k2
                        improved = True
                        break
                    P[...] = snap
                    self.xmin, self.xmax, self.ymin, self.ymax = frame
                if improved:
                    break
            if not improved:
                break
        return cur_key

    def _squeeze_phase(self, cur_key: float, deadline: float):
        """Repeatedly try to shrink the bbox on both axes."""
        improved = True
        wins = fails = 0
        while improved and time.time() < deadline:
            improved = False
            for axis in (0, 1):
                span = (self.xmax - self.xmin) if axis == 0 \
                    else (self.ymax - self.ymin)
                for frac in (0.08, 0.04, 0.015, 0.006):
                    ok, cur_key = self._try_squeeze(axis, frac * span,
                                                    cur_key, deadline=deadline)
                    if ok:
                        wins += 1
                        improved = True
                        break
                    fails += 1
        if _DEBUG and (wins or fails):
            print(f"[squeeze] {wins} wins {fails} fails "
                  f"frame {self.xmax - self.xmin:.1f}x{self.ymax - self.ymin:.1f}",
                  flush=True)
        return cur_key

    def run(self, deadline: float) -> np.ndarray:
        # "end"/"both" placement CARVES its slice out of the caller's span
        # rather than adding to it: the fork is deadline-bounded, so an
        # overrun would show up as raw runtime, not as free search.
        hard_deadline = deadline
        prof = self._prof          # PARTNER_REFINE_PROF; None -> dead branches
        t_prof0 = time.time() if prof is not None else 0.0
        csa_end = (self.csa_share > 0.0
                   and self.csa_where in ("end", "both"))
        if csa_end:
            deadline = deadline - self.csa_share * max(
                deadline - time.time(), 0.0)
        best_key = self._key()
        best = self.P.copy()
        stall = 0
        kicks = 0
        rounds = 0
        # stall early stop (see __init__): disabled -> stall_win is None and
        # every predicate below is a dead branch.
        self._run_stalled = False
        stall_eps = self.stall_eps
        stall_win = (self._stall_frac * max(deadline - time.time(), 1e-6)
                     if self._stall_frac > 0.0 else None)
        if stall_win is not None and self._early_exit:
            stall_win = max(stall_win, early_exit_min_window())
        t_run0 = time.time()
        ref_key = best_key
        ref_time = time.time()
        # CSA carve-out: a bounded slice of THIS phase's span, spent only at
        # the median sweep's fixed point (stall >= 1, i.e. the previous round
        # bought nothing).  csa_share == 0.0 (default) -> dead branch.
        csa_cap = (self.csa_share * max(deadline - time.time(), 0.0)
                   if self.csa_share > 0.0 else 0.0)
        while time.time() < deadline and rounds < 400:
            if (csa_cap > 0.0 and self.csa_where in ("stall", "both")
                    and stall >= 1 and self._csa_spent < csa_cap):
                # gate on the CURRENT point's proxy (not best_key): the round
                # that stalled left `self.P` at its own, possibly worse,
                # result, and the job here is to escape THAT fixed point.
                # Global acceptance is still the round's `best` snapshot.
                _t = time.time() if prof is not None else 0.0
                self._csa_pass(0)
                self._csa_pass(1)
                if prof is not None:
                    prof["t_csa"] += time.time() - _t
                    prof["c_csa"] += 2
            if stall_win is not None:
                # sign-safe relative test: improvement is measured from the
                # last window reset, not from the previous pass, so a run of
                # micro-accepts cannot keep a converged phase alive forever.
                now = time.time()
                if ref_key - best_key >= stall_eps * abs(ref_key):
                    ref_key, ref_time = best_key, now
                elif now - ref_time >= stall_win:
                    self._run_stalled = True
                    break
            _t = time.time() if prof is not None else 0.0
            self._axis_pass(0)
            self._axis_pass(1)
            if prof is not None:
                prof["t_axis_soft"] += time.time() - _t
                prof["c_axis_soft"] += 2
            if rounds >= 1 and rounds % 2 == 1:
                _t = time.time() if prof is not None else 0.0
                self._reshape_pass(deadline)
                if prof is not None:
                    prof["t_reshape"] += time.time() - _t
                    prof["c_reshape"] += 1
            _t = time.time() if prof is not None else 0.0
            self._tag_snap()
            if prof is not None:
                prof["t_tag_snap"] += time.time() - _t
                prof["c_tag_snap"] += 1
                _t = time.time()
            ovl = self._has_overlap()
            if prof is not None:
                prof["t_overlap"] += time.time() - _t
                prof["c_overlap"] += 1
            if ovl:
                # a sweep produced an illegal state (only possible via the
                # rare unresolved-cycle fallback) — discard the whole round
                self.P[...] = best
                rounds += 1
                stall += 1
                if stall >= 2:
                    if kicks >= 6:
                        break
                    _t = time.time() if prof is not None else 0.0
                    self._perturb()
                    if prof is not None:
                        prof["t_perturb"] += time.time() - _t
                        prof["c_perturb"] += 1
                    kicks += 1
                    stall = 0
                continue
            _t = time.time() if prof is not None else 0.0
            k = self._key()
            if prof is not None:
                prof["t_key"] += time.time() - _t
                prof["c_key"] += 1
            if k < best_key - 1e-12:
                best_key = k
                best = self.P.copy()
                stall = 0
            else:
                stall += 1
            if stall >= 2:
                if kicks >= 6:
                    break
                self.P[...] = best
                _t = time.time() if prof is not None else 0.0
                self._perturb()
                if prof is not None:
                    prof["t_perturb"] += time.time() - _t
                    prof["c_perturb"] += 1
                kicks += 1
                stall = 0
            rounds += 1

        phase1_span = max(time.time() - t_run0, 1e-6)

        # phase 2: discrete slot exchanges (global swap / re-insert), the
        # only move class that can change topology on a saturated packing;
        # each accepted move already improved the proxy, and a continuous
        # touch-up harvests the slack the reorder opened up.  When a sweep
        # finds nothing, a random kick re-seeds the search.
        self.P[...] = best
        cur_key = best_key
        _t = time.time() if prof is not None else 0.0
        cur_key = self._squeeze_phase(cur_key, deadline)
        if prof is not None:
            prof["t_squeeze"] += time.time() - _t
            prof["c_squeeze"] += 1
        if self.enable_deflate:
            _t = time.time() if prof is not None else 0.0
            cur_key = self._deflate(cur_key, deadline)
            if prof is not None:
                prof["t_deflate"] += time.time() - _t
                prof["c_deflate"] += 1
        if cur_key < best_key - 1e-12:
            best_key = cur_key
            best = self.P.copy()
        batches = swaps = 0
        stall2 = 0
        # phase boundary: the discrete phase is a DIFFERENT move class (the
        # only one that can reorder a saturated packing), so it gets a fresh
        # anchor and a window scaled to what is left of the budget -- a
        # converged continuous phase must not veto it, and the time
        # squeeze/deflate spent must not be charged to it.
        if stall_win is not None:
            now = time.time()
            remaining = max(deadline - now, 1e-6)
            if self._early_exit:
                # The historical rescale ties the discrete window to the
                # LEFTOVER span -- so a continuous phase that converged after
                # 0.2 s of a 3 s slice hands the discrete phase a nearly
                # full-size window, i.e. it re-spends exactly the wall clock
                # this flag exists to give back.  Under EARLY_EXIT the window
                # is tied to how long real progress actually took
                # (`phase1_span`), never longer than the historical value.
                stall_win = max(
                    self._stall_frac * min(phase1_span, remaining),
                    early_exit_min_window())
            else:
                stall_win = self._stall_frac * remaining
            ref_key, ref_time = best_key, now
        while time.time() < deadline and stall2 < 4:
            if stall_win is not None:
                now = time.time()
                if ref_key - best_key >= stall_eps * abs(ref_key):
                    ref_key, ref_time = best_key, now
                elif now - ref_time >= stall_win:
                    self._run_stalled = True
                    break
            _t = time.time() if prof is not None else 0.0
            self._build_swappable()
            if prof is not None:
                prof["t_build_swap"] += time.time() - _t
                prof["c_build_swap"] += 1
                _t = time.time()
            acc, cur_key = self._discrete_batch(cur_key, deadline)
            if prof is not None:
                prof["t_discrete"] += time.time() - _t
                prof["c_discrete"] += 1
            if acc == 0 and os.environ.get("PARTNER_MATCH"):
                # escalation ladder: exact cycle matching ONLY when the
                # pairwise batch is exhausted — free on non-stalled
                # sweeps, and a smarter kick than the random perturb
                # (a successful cycle also re-opens pairwise wins)
                _t = time.time() if prof is not None else 0.0
                acc, cur_key = self._matching_batch(cur_key, deadline)
                if prof is not None:
                    prof["t_discrete"] += time.time() - _t
            batches += 1
            swaps += acc
            if cur_key < best_key - 1e-12:
                best_key = cur_key
                best = self.P.copy()
            if acc == 0:
                stall2 += 1
                _t = time.time() if prof is not None else 0.0
                self._perturb()
                if prof is not None:
                    prof["t_perturb"] += time.time() - _t
                    prof["c_perturb"] += 1
                    _t = time.time()
                self._axis_pass(0)
                self._axis_pass(1)
                if prof is not None:
                    prof["t_axis_soft"] += time.time() - _t
                    prof["c_axis_soft"] += 2
                    _t = time.time()
                self._tag_snap()
                if prof is not None:
                    prof["t_tag_snap"] += time.time() - _t
                    prof["c_tag_snap"] += 1
                    _t = time.time()
                ovl = self._has_overlap()
                if prof is not None:
                    prof["t_overlap"] += time.time() - _t
                    prof["c_overlap"] += 1
                if ovl:
                    self.P[...] = best
                    cur_key = best_key
                else:
                    _t = time.time() if prof is not None else 0.0
                    cur_key = self._key()
                    if prof is not None:
                        prof["t_key"] += time.time() - _t
                        prof["c_key"] += 1
                    if cur_key < best_key - 1e-12:
                        best_key = cur_key
                        best = self.P.copy()
                continue
            stall2 = 0
            _t = time.time() if prof is not None else 0.0
            self._axis_pass(0)
            self._axis_pass(1)
            if prof is not None:
                prof["t_axis_soft"] += time.time() - _t
                prof["c_axis_soft"] += 2
                _t = time.time()
            self._reshape_pass(deadline)
            if prof is not None:
                prof["t_reshape"] += time.time() - _t
                prof["c_reshape"] += 1
                _t = time.time()
            self._tag_snap()
            if prof is not None:
                prof["t_tag_snap"] += time.time() - _t
                prof["c_tag_snap"] += 1
                _t = time.time()
            ovl = self._has_overlap()
            if prof is not None:
                prof["t_overlap"] += time.time() - _t
                prof["c_overlap"] += 1
            if ovl:
                self.P[...] = best
                cur_key = best_key
                continue
            _t = time.time() if prof is not None else 0.0
            cur_key = self._key()
            if prof is not None:
                prof["t_key"] += time.time() - _t
                prof["c_key"] += 1
                _t = time.time()
            cur_key = self._squeeze_phase(cur_key, deadline)
            if prof is not None:
                prof["t_squeeze"] += time.time() - _t
                prof["c_squeeze"] += 1
            if cur_key < best_key - 1e-12:
                best_key = cur_key
                best = self.P.copy()
        # terminal CSA polish: everything else has run, so the squeeze phase
        # has already banked whatever bbox area it could and the slack CSA
        # spends on HPWL is slack nothing else was going to use.  (The
        # in-loop "stall" placement competes with the squeeze for that same
        # slack -- which is exactly what the A/B is there to separate.)
        if csa_end:
            self.P[...] = best
            _t = time.time() if prof is not None else 0.0
            while time.time() < hard_deadline:
                if not (self._csa_pass(0) | self._csa_pass(1)):
                    break
            if prof is not None:
                prof["t_csa"] += time.time() - _t
                prof["c_csa"] += 1
            k = self._key()
            if k < best_key - 1e-12:
                best_key = k
                best = self.P.copy()

        if prof is not None:
            _prof_emit(self, time.time() - t_prof0, rounds, batches, swaps)

        if _DEBUG:
            print(f"[refiner] rounds={rounds} kicks={kicks} "
                  f"discrete: {batches} batches, {swaps} moves accepted"
                  + (f" | csa {self._csa_applied}/{self._csa_calls} kept "
                     f"in {self._csa_spent:.3f}s" if self.csa_share > 0.0
                     else ""),
                  flush=True)
        self.P[...] = best
        return best


def full_violations(opt, pos: np.ndarray) -> int:
    """opt._violations plus grouping checks for pure-movable clusters.

    The column construction keeps movable clusters contiguous, so the
    stage-1 proxy skips them; direct-prediction candidates can break them,
    and cross-pipeline selection must see that."""
    V = opt._violations(pos)
    checked = {tuple(g.tolist()) for g in opt._clu_arrays}
    x0 = pos[:, 0]
    y0 = pos[:, 1]
    x1 = x0 + pos[:, 2]
    y1 = y0 + pos[:, 3]
    for idxs in opt.cluster_groups.values():
        g = np.array(sorted(int(i) for i in idxs), dtype=np.int64)
        if len(g) < 2 or tuple(g.tolist()) in checked:
            continue
        ox = np.minimum(x1[g][:, None], x1[g][None, :]) \
            - np.maximum(x0[g][:, None], x0[g][None, :])
        oy = np.minimum(y1[g][:, None], y1[g][None, :]) \
            - np.maximum(y0[g][:, None], y0[g][None, :])
        adj = ((ox > TOUCH_TOL) & (oy >= -TOUCH_TOL)) \
            | ((oy > TOUCH_TOL) & (ox >= -TOUCH_TOL))
        m = len(g)
        seen = np.zeros(m, dtype=bool)
        comps = 0
        for s in range(m):
            if seen[s]:
                continue
            comps += 1
            frontier = np.zeros(m, dtype=bool)
            frontier[s] = True
            seen[s] = True
            while frontier.any():
                nxt = adj[frontier].any(axis=0) & ~seen
                seen |= nxt
                frontier = nxt
        V += comps - 1
    return V


def _touch_components(Q, g):
    """Connected components of blocks g under the proxy touch test."""
    x0 = Q[g, 0]
    y0 = Q[g, 1]
    x1 = x0 + Q[g, 2]
    y1 = y0 + Q[g, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) \
        - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) \
        - np.maximum(y0[:, None], y0[None, :])
    adj = ((ox > TOUCH_TOL) & (oy >= -TOUCH_TOL)) \
        | ((oy > TOUCH_TOL) & (ox >= -TOUCH_TOL))
    m = len(g)
    seen = np.zeros(m, dtype=bool)
    comps = []
    for s in range(m):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in range(m):
                    if adj[u, v] and not seen[v]:
                        seen[v] = True
                        nxt.append(v)
            comp += nxt
            frontier = nxt
        comps.append([g[k] for k in comp])
    return comps


def _cluster_seat(opt, out, deadline=None):
    """Rigid re-snap for broken clusters: translate each detached piece
    onto exact contact with the anchor piece (the one holding a locked
    member, else the largest), trying the smallest collision-free
    landing.  After the rigid move the piece's internal contacts are
    re-imposed exactly along a spanning tree — the evaluator's shapely
    union needs bit-equal edges and a float translate drifts them.
    Kept only when the proxy violation count strictly drops."""
    Q = np.asarray(out, dtype=np.float64).copy()
    V0 = full_violations(opt, Q)
    if V0 <= 0:
        return out
    for idxs in opt.cluster_groups.values():
        if deadline is not None and time.time() >= deadline:
            break
        g = sorted(int(v) for v in idxs)
        if not 2 <= len(g) <= 16:
            continue
        merges = 0
        while merges < 4:
            pieces = _touch_components(Q, g)
            if len(pieces) < 2:
                break
            # merging ANY two pieces pays the same — try every movable
            # piece against every block of every other piece, nearest
            # landing first.  Pieces too big or anchored by a preplaced
            # member cannot translate, but a soft block can still grow
            # across the gap, so they are offered reshape-only.
            progress = False
            pieces.sort(key=lambda p: sum(Q[i, 2] * Q[i, 3] for i in p))
            for piece in pieces:
                rigid_ok = (len(piece) <= 6
                            and not any(opt.kind[i] == 2 for i in piece))
                rest = [b for p in pieces for b in p if p is not piece]
                got = _piece_land(opt, Q, piece, rest,
                                  rigid_ok=rigid_ok, deadline=deadline)
                if got:
                    progress = True
                    merges += 1
                    break
            if not progress:
                break
    V1 = full_violations(opt, Q)
    if V1 < V0:
        if _DEBUG:
            print(f"[cseat] V {V0}->{V1}", flush=True)
        return Q
    return out


def _piece_land(opt, Q, piece, rest, rigid_ok=True, deadline=None):
    """Merge `piece` onto some block in `rest`; returns True when a
    landing stuck (V-improving).  Three tools, cheapest first:
    (1) rigid translate onto exact contact; (2) reshape-to-contact —
    one soft block on either side grows along the gap axis to land
    exactly on the other side's edge (its far edge stays put, so an
    edge tag on the far side survives — measured on case 99, the
    detached R-tagged pair is 5 units left of the body while the whole
    right wall pins the bbox edge); (3) dig-retry translate — a landing
    blocked only by free movables relocates them via _evict first.
    The landing radius is generous — measured broken pieces sit 4-8
    units off the body — because every landing is collision-checked
    and V-guarded, so a far transaction that hurts anything reverts."""
    Vcur = full_violations(opt, Q)
    hp_cap = opt._hpwl(Q) * 1.02
    GAP = 12.0
    # candidate landings: some piece block a abuts outside block b
    cands = []
    for a in piece:
        for b in rest:
            oy = min(Q[a, 1] + Q[a, 3], Q[b, 1] + Q[b, 3]) \
                - max(Q[a, 1], Q[b, 1])
            if oy > 1e-9:
                for tx in (float(Q[b, 0] + Q[b, 2]),
                           float(Q[b, 0]) - float(Q[a, 2])):
                    d = tx - float(Q[a, 0])
                    if abs(d) <= GAP:
                        cands.append((abs(d), a, 0, tx))
            ox = min(Q[a, 0] + Q[a, 2], Q[b, 0] + Q[b, 2]) \
                - max(Q[a, 0], Q[b, 0])
            if ox > 1e-9:
                for ty in (float(Q[b, 1] + Q[b, 3]),
                           float(Q[b, 1]) - float(Q[a, 3])):
                    d = ty - float(Q[a, 1])
                    if abs(d) <= GAP:
                        cands.append((abs(d), a, 1, ty))
    cands.sort(key=lambda c: c[0])
    others = np.ones(opt.n, dtype=bool)
    for i in piece:
        others[i] = False

    def _accept():
        V = full_violations(opt, Q)
        return V < Vcur and opt._hpwl(Q) <= hp_cap

    def _record_tree(a):
        # intra-piece exact-contact spanning tree rooted at a
        tree = []
        if len(piece) > 1:
            comps = _touch_components(Q, piece)
            if len(comps) == 1:
                seen = {a}
                frontier = [a]
                while frontier:
                    nxt = []
                    for u in frontier:
                        for v in piece:
                            if v in seen:
                                continue
                            rel = None
                            ovy = min(Q[u, 1] + Q[u, 3],
                                      Q[v, 1] + Q[v, 3]) \
                                - max(Q[u, 1], Q[v, 1])
                            ovx = min(Q[u, 0] + Q[u, 2],
                                      Q[v, 0] + Q[v, 2]) \
                                - max(Q[u, 0], Q[v, 0])
                            if ovy > 1e-9 and abs(
                                    Q[u, 0] + Q[u, 2]
                                    - Q[v, 0]) <= TOUCH_TOL:
                                rel = (0, 1)   # v.x0 = u.x1
                            elif ovy > 1e-9 and abs(
                                    Q[v, 0] + Q[v, 2]
                                    - Q[u, 0]) <= TOUCH_TOL:
                                rel = (0, -1)  # v.x1 = u.x0
                            elif ovx > 1e-9 and abs(
                                    Q[u, 1] + Q[u, 3]
                                    - Q[v, 1]) <= TOUCH_TOL:
                                rel = (1, 1)
                            elif ovx > 1e-9 and abs(
                                    Q[v, 1] + Q[v, 3]
                                    - Q[u, 1]) <= TOUCH_TOL:
                                rel = (1, -1)
                            if rel is not None:
                                tree.append((u, v, rel))
                                seen.add(v)
                                nxt.append(v)
                    frontier = nxt
                if len(seen) != len(piece):
                    tree = []
        return tree

    def _try_translate(a, axis, tgt, dig):
        snap = Q.copy()
        d = tgt - float(Q[a, axis])
        tree = _record_tree(a)
        # rigid translate, root landed exactly
        for i in piece:
            Q[i, axis] += d
        Q[a, axis] = tgt
        # re-impose recorded contacts exactly down the tree
        for u, v, (rax, sgn) in tree:
            if sgn > 0:
                Q[v, rax] = Q[u, rax] + Q[u, rax + 2]
            else:
                Q[v, rax] = Q[u, rax] - Q[v, rax + 2]
        # collision check piece vs rest
        x0, y0 = Q[:, 0], Q[:, 1]
        x1, y1 = x0 + Q[:, 2], y0 + Q[:, 3]
        blockers = set()
        for i in piece:
            cx = np.minimum(x1[i], x1) - np.maximum(x0[i], x0)
            cy = np.minimum(y1[i], y1) - np.maximum(y0[i], y0)
            hit = (cx > SEP_TOL) & (cy > SEP_TOL) & others
            blockers.update(int(v) for v in np.nonzero(hit)[0])
        if blockers:
            ok = False
            if dig and len(blockers) <= 4 and all(
                    opt.kind[dd] != 2 and opt.cluster[dd] <= 0
                    and opt.mib[dd] <= 0 and opt.boundary[dd] <= 0
                    for dd in blockers):
                # every blocker is a free movable: pull them out to a
                # slot of their own (anchor-adjacent slot first, then
                # the free-rect map — dense layouts keep their slack in
                # edge slivers _evict's anchor-shaped landings miss),
                # then the landing stands or falls as one transaction
                r = _Refiner(opt, Q, 0)
                pending = [dd for dd in sorted(blockers)
                           if not r._evict(dd)]
                ok = True
                while pending:
                    # the map must be refreshed per landing: spots just
                    # freed by evicted/relocated blockers are exactly
                    # where the big leftover block fits
                    sk = np.zeros(opt.n, dtype=bool)
                    sk[pending] = True
                    rects = r._free_rects(max_rects=12, skip=sk)
                    dd = pending.pop()
                    if not r._relocate_to_free(dd, rects):
                        ok = False
                        break
                if ok and not r._has_overlap():
                    Q[...] = r.P
                else:
                    ok = False
        else:
            ok = True
        if ok and _accept():
            return True
        Q[...] = snap
        return False

    def _try_reshape():
        # one soft block grows across the gap onto the other side's
        # edge; the contact edge is ASSIGNED (grow left/down) or
        # drift-corrected (grow right/up) so the evaluator's exact-
        # abutment union sees the touch
        pairs = []
        for a in piece:
            for b in rest:
                pairs.append((a, b))
                pairs.append((b, a))
        trials = []
        for g_, t_ in pairs:
            if opt.kind[g_] != 0 or opt.mib[g_] > 0:
                continue
            gx0, gy0, gw, gh = (float(v) for v in Q[g_])
            tx0, ty0, tw, th = (float(v) for v in Q[t_])
            oy = min(gy0 + gh, ty0 + th) - max(gy0, ty0)
            ox = min(gx0 + gw, tx0 + tw) - max(gx0, tx0)
            if oy > 1e-9:
                gap = tx0 - (gx0 + gw)
                if 1e-9 < gap <= GAP:            # grow right
                    trials.append((gap, g_, t_, 0, +1))
                gap = gx0 - (tx0 + tw)
                if 1e-9 < gap <= GAP:            # grow left
                    trials.append((gap, g_, t_, 0, -1))
            if ox > 1e-9:
                gap = ty0 - (gy0 + gh)
                if 1e-9 < gap <= GAP:            # grow up
                    trials.append((gap, g_, t_, 1, +1))
                gap = gy0 - (ty0 + th)
                if 1e-9 < gap <= GAP:            # grow down
                    trials.append((gap, g_, t_, 1, -1))
        trials.sort(key=lambda t: t[0])
        for _gap, g_, t_, axis, sgn in trials[:12]:
            area = float(opt.areas[g_])
            gx0, gy0, gw, gh = (float(v) for v in Q[g_])
            tx0, ty0, tw, th = (float(v) for v in Q[t_])
            if axis == 0:
                if sgn > 0:
                    nx0, nw = gx0, tx0 - gx0
                    for _ in range(3):           # contact edge bit-equal
                        drift = (nx0 + nw) - tx0
                        if drift == 0.0:
                            break
                        nw -= drift
                else:
                    nx0 = tx0 + tw
                    nw = (gx0 + gw) - nx0
                nh = area / nw
                # cross anchor: keep low edge, else keep high edge
                cross = [(nx0, gy0, nw, nh),
                         (nx0, gy0 + gh - nh, nw, nh)]
                tlo, thi = ty0, ty0 + th
            else:
                if sgn > 0:
                    ny0, nh = gy0, ty0 - gy0
                    for _ in range(3):
                        drift = (ny0 + nh) - ty0
                        if drift == 0.0:
                            break
                        nh -= drift
                else:
                    ny0 = ty0 + th
                    nh = (gy0 + gh) - ny0
                nw = area / nh
                cross = [(gx0, ny0, nw, nh),
                         (gx0 + gw - nw, ny0, nw, nh)]
                tlo, thi = tx0, tx0 + tw
            for rect in cross:
                cl = rect[1 - axis]
                ch = cl + rect[3 - axis]
                if min(ch, thi) - max(cl, tlo) <= TOUCH_TOL:
                    continue                     # shrunk out of overlap
                if rect[2] < 0.5 or rect[3] < 0.5:
                    continue
                snap = Q[g_].copy()
                Q[g_] = rect
                x0, y0 = Q[:, 0], Q[:, 1]
                x1, y1 = x0 + Q[:, 2], y0 + Q[:, 3]
                cx = np.minimum(x1[g_], x1) - np.maximum(x0[g_], x0)
                cy = np.minimum(y1[g_], y1) - np.maximum(y0[g_], y0)
                hit = (cx > SEP_TOL) & (cy > SEP_TOL)
                hit[g_] = False
                if not hit.any() and _accept():
                    return True
                Q[g_] = snap
        return False

    if rigid_ok:
        for _ad, a, axis, tgt in cands[:16]:
            if _try_translate(a, axis, tgt, dig=False):
                return True
    if _try_reshape():
        return True
    if rigid_ok:
        digs = 0
        for _ad, a, axis, tgt in cands[:12]:
            if deadline is not None and time.time() >= deadline:
                break
            if _try_translate(a, axis, tgt, dig=True):
                return True
            digs += 1
            if digs >= 5:
                break
    return False


def _edge_seat(opt, out):
    """Surgical last pass for residual boundary-tag violations.  The push
    sweeps leave tagged blocks hovering sub-unit distances off the
    layout's extreme edges — and a single untagged outlier can DEFINE an
    edge slightly past the tagged shelf, unseating every tag on that
    side at once.  Per edge, in order: translate the tagged block onto
    the edge; reshape it (soft, area-preserving) to extend to the edge;
    pull the few outliers that overshoot the tagged shelf back level
    with it.  Kept only if the full violation count strictly drops.

    PARTNER_EDGE_SEAT_V2=1 (see `edge_seat_v2_on`) widens the reach --
    layout-scaled GAP, an 8-outlier pull cap, a joint two-axis corner
    pass -- and swaps the acceptance test for the evaluator's own
    boundary+grouping+MIB total.  Default off: byte-identical."""
    Q = np.asarray(out, dtype=np.float64).copy()
    n = opt.n
    bnd = [(i, int(opt.boundary[i])) for i in range(n)
           if opt.boundary[i] > 0]
    if not bnd:
        return out

    v2 = edge_seat_v2_on()
    _exact = None
    if v2:
        # (iv) evaluator-faithful acceptance.  Lazily imported: the module
        # pulls in column_sa_legalizer, and the off path must not pay for
        # it.  If it is unavailable we silently keep the proxy -- a wider
        # search under the old test, never a crash.
        try:
            from violation_killer import _violations_exact as _exact_fn
            _exact = _exact_fn
        except Exception:
            _exact = None

    def _viol(P):
        if _exact is not None:
            return _exact(opt, P)
        return full_violations(opt, P)

    V0 = _viol(Q)
    if V0 <= 0:
        return out
    eps = 1e-6
    GAP = 2.0        # only chase small hovers; big gaps are real misplaces
    OVER_CAP = 3     # (c) outlier-pull cap
    if v2:
        # (i) the hover window is a property of the FRAME, not an absolute
        # length: 2.0 units is a big gap on a 21-block frame and rounding
        # noise on a 120-block one.
        _bw = float((Q[:, 0] + Q[:, 2]).max() - Q[:, 0].min())
        _bh = float((Q[:, 1] + Q[:, 3]).max() - Q[:, 1].min())
        GAP = max(2.0, 0.08 * min(_bw, _bh))
        OVER_CAP = 8                                  # (ii)

    def _clash(i, rect):
        x0, y0, w, h = rect
        cx = np.minimum(x0 + w, Q[:, 0] + Q[:, 2]) - np.maximum(x0, Q[:, 0])
        cy = np.minimum(y0 + h, Q[:, 1] + Q[:, 3]) - np.maximum(y0, Q[:, 1])
        hit = (cx > SEP_TOL) & (cy > SEP_TOL)
        hit[i] = False
        return bool(hit.any())

    Vcur = V0

    def _commit(changes):
        """Apply the rect changes; keep only if V strictly drops."""
        nonlocal Vcur
        snap = [(i, Q[i].copy()) for i, _ in changes]
        for i, rect in changes:
            Q[i] = rect
        V = _viol(Q)
        if V < Vcur:
            Vcur = V
            return True
        for i, old in snap:
            Q[i] = old
        return False

    def _can_translate(i):
        return opt.kind[i] != 2

    def _can_reshape(i):
        return (opt.kind[i] == 0 and opt.cluster[i] <= 0
                and opt.mib[i] <= 0)

    # (bit, axis, side): side 0 = min edge, side 1 = max edge
    edges = ((1, 0, 0), (8, 1, 0), (2, 0, 1), (4, 1, 1))

    if v2:
        # (iii) corner seat.  A two-bit tag (e.g. left AND bottom) asks for
        # a CORNER.  The per-edge loop below can only offer one axis at a
        # time, and the intermediate -- seated on x, still hovering on y --
        # scores no better than the start, so `_commit`'s strict gate
        # rejects step one and step two never happens.  Move both axes
        # together and let the same gate arbitrate the finished move.
        # Locked blocks are excluded (kind 2 may not translate); the
        # clash test and the global V test do the rest, so cluster/MIB
        # members are allowed in and simply lose when they break something.
        for i, code in bnd:
            if bin(code).count("1") < 2 or opt.kind[i] == 2:
                continue
            x_lo = float(Q[:, 0].min())
            y_lo = float(Q[:, 1].min())
            x_hi = float((Q[:, 0] + Q[:, 2]).max())
            y_hi = float((Q[:, 1] + Q[:, 3]).max())
            rect = Q[i].copy()
            if code & 1:
                rect[0] = x_lo
            elif code & 2:
                rect[0] = x_hi - rect[2]
            if code & 8:
                rect[1] = y_lo
            elif code & 4:
                rect[1] = y_hi - rect[3]
            if abs(rect[0] - Q[i, 0]) < 1e-12 \
                    and abs(rect[1] - Q[i, 1]) < 1e-12:
                continue                      # already seated
            if _clash(i, tuple(rect)):
                continue
            _commit([(i, rect)])

    for _round in range(2):
        for bit, axis, side in edges:
            lo = Q[:, axis]
            hi = lo + Q[:, axis + 2]
            edge = float(lo.min()) if side == 0 else float(hi.max())
            viol = []
            for i, code in bnd:
                if not (code & bit):
                    continue
                g = (lo[i] - edge) if side == 0 else (edge - hi[i])
                if g > eps:
                    viol.append((g, i))
            if not viol:
                continue
            viol.sort()
            o = 1 - axis
            blo = float(Q[:, o].min())
            bhi = float((Q[:, o] + Q[:, o + 2]).max())
            for g, i in viol:
                if g > GAP:
                    # (b2) relocate onto the edge: blocks placed FAR off
                    # their tagged edge (kind-1 shapes and free softs) —
                    # carve a clash-free slot touching the edge, landing
                    # positions aligned to the blocks already in the
                    # edge strip
                    if opt.kind[i] == 2 or opt.cluster[i] > 0 \
                            or opt.mib[i] > 0:
                        continue
                    depth = float(Q[i, axis + 2])
                    o_dim = float(Q[i, o + 2])
                    strip_lo = edge if side == 0 else edge - depth
                    cands_o = [float(Q[i, o]), blo, bhi - o_dim]
                    for j in range(n):
                        if j == i:
                            continue
                        jlo = float(Q[j, axis])
                        jhi = jlo + float(Q[j, axis + 2])
                        if jhi > strip_lo + 1e-9 \
                                and jlo < strip_lo + depth - 1e-9:
                            cands_o.append(float(Q[j, o] + Q[j, o + 2]))
                            cands_o.append(float(Q[j, o]) - o_dim)
                    rect = Q[i].copy()
                    rect[axis] = strip_lo
                    for t in cands_o:
                        if t < blo - 1e-9 or t + o_dim > bhi + 1e-9:
                            continue
                        rect[o] = t
                        if not _clash(i, tuple(rect)) \
                                and _commit([(i, rect)]):
                            break
                    continue
                if _can_translate(i):
                    # (a) translate onto the edge — for cluster members
                    # also try landing spots along the other axis that
                    # restore exact contact with a fellow member (the
                    # hover and the broken abutment usually go
                    # together).  Absolute targets, never deltas: the
                    # evaluator's contact test needs bit-equal edges.
                    cur_o = float(Q[i, o])
                    targets = [cur_o]
                    if opt.cluster[i] > 0:
                        for j in range(n):
                            if j == i or opt.cluster[j] != opt.cluster[i]:
                                continue
                            for t in (float(Q[j, o] + Q[j, o + 2]),
                                      float(Q[j, o]) - float(Q[i, o + 2])):
                                if 0 < abs(t - cur_o) <= GAP:
                                    targets.append(t)
                        targets.sort(key=lambda t: (t == cur_o,
                                                    abs(t - cur_o)))
                    done = False
                    for t in targets:
                        rect = Q[i].copy()
                        rect[axis] = edge if side == 0 \
                            else edge - rect[axis + 2]
                        rect[o] = t
                        if not _clash(i, tuple(rect)) \
                                and _commit([(i, rect)]):
                            done = True
                            break
                    if done:
                        continue
                # (a2) dilate to the edge: grow ONLY the tagged edge
                # outward — every other edge (and so every cluster
                # contact) stays put.  Soft areas have a 1% HARD
                # tolerance; sub-unit hovers on 10+-unit blocks cost
                # well under it (0.9% cap leaves float margin).
                if opt.kind[i] == 0 and opt.mib[i] <= 0:
                    rect = Q[i].copy()
                    if side == 0:
                        rect[axis] = edge
                        rect[axis + 2] = float(hi[i]) - edge
                    else:
                        rect[axis + 2] = edge - float(lo[i])
                    if rect[2] * rect[3] <= float(opt.areas[i]) * 1.009 \
                            and not _clash(i, tuple(rect)) \
                            and _commit([(i, rect)]):
                        continue
                # (b) area-preserving reshape extending to the edge
                if not _can_reshape(i):
                    continue
                area = float(opt.areas[i])
                if side == 0:
                    new_lo = edge
                    dim = float(hi[i]) - edge
                else:
                    new_lo = float(lo[i])
                    dim = edge - float(lo[i])
                if dim <= eps or area / dim > (bhi - blo):
                    continue
                other = area / dim
                no = float(Q[i, o]) + 0.5 * (float(Q[i, o + 2]) - other)
                no = min(max(no, blo), bhi - other)
                rect2 = Q[i].copy()
                rect2[axis] = new_lo
                rect2[axis + 2] = dim
                rect2[o] = no
                rect2[o + 2] = other
                if not _clash(i, tuple(rect2)):
                    _commit([(i, rect2)])
            # (c) outlier pull: blocks overshooting the tagged shelf
            lo = Q[:, axis]
            hi = lo + Q[:, axis + 2]
            edge = float(lo.min()) if side == 0 else float(hi.max())
            still = []
            for i, code in bnd:
                if not (code & bit):
                    continue
                g = (lo[i] - edge) if side == 0 else (edge - hi[i])
                if g > eps:
                    still.append((g, i))
            # long reach: pulling a handful of outliers level with the
            # tagged shelf both seats the tags AND shrinks the bbox —
            # a loose-rung inflated top edge is exactly this shape.
            # Accept on V drop, or on equal V with a smaller bbox.
            if not still or min(g for g, _i in still) > 12.0:
                continue
            if side == 0:
                tgt = min(lo[i] for _g, i in still)
                over = [j for j in range(n) if lo[j] < tgt - eps]
            else:
                tgt = max(hi[i] for _g, i in still)
                over = [j for j in range(n) if hi[j] > tgt + eps]
            if 0 < len(over) <= OVER_CAP \
                    and all(_can_translate(j)
                            and opt.cluster[j] <= 0 for j in over):
                def _bbox_area(P):
                    return float(((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                                 * ((P[:, 1] + P[:, 3]).max()
                                    - P[:, 1].min()))
                ar0 = _bbox_area(Q)
                snap = Q.copy()
                bad = False
                for j in over:
                    rect = Q[j].copy()
                    rect[axis] = tgt if side == 0 else tgt - rect[axis + 2]
                    if _clash(j, tuple(rect)):
                        bad = True
                        break
                    Q[j] = rect
                if bad:
                    Q[...] = snap
                else:
                    V = _viol(Q)
                    if V < Vcur or (V == Vcur
                                    and _bbox_area(Q) < ar0 - 1e-9):
                        Vcur = V
                    else:
                        Q[...] = snap
            # (c2) trim the overshoot: shrink every block protruding
            # past the tagged shelf back level with it — a shrink
            # cannot clash, and the trimmed edge faces the bbox rim so
            # contacts survive; whole shelves pushed sub-unit past a
            # preplaced tag land here, where translation never can
            lo = Q[:, axis]
            hi = lo + Q[:, axis + 2]
            edge = float(lo.min()) if side == 0 else float(hi.max())
            still = [(g, i) for g, i in still
                     if ((lo[i] - edge) if side == 0
                         else (edge - hi[i])) > eps]
            if not still or min(g for g, _i in still) > GAP:
                continue
            if side == 0:
                tgt = min(float(lo[i]) for _g, i in still)
                over = [j for j in range(n) if lo[j] < tgt - eps]
            else:
                tgt = max(float(hi[i]) for _g, i in still)
                over = [j for j in range(n) if hi[j] > tgt + eps]
            if not 0 < len(over) <= 16:
                continue
            snap = Q.copy()
            ok = True
            for j in over:
                beta = (tgt - float(lo[j])) if side == 0 \
                    else (float(hi[j]) - tgt)
                if beta > GAP:
                    ok = False
                    break
                # translate level first — keeps area and MIB shapes
                rect = Q[j].copy()
                rect[axis] = tgt if side == 0 else tgt - rect[axis + 2]
                if opt.kind[j] != 2 and opt.cluster[j] <= 0 \
                        and not _clash(j, tuple(rect)):
                    Q[j] = rect
                    continue
                # else trim; cluster members allowed — the global V
                # check arbitrates the trade.  MIB members trim in
                # sync with their whole group (identical dims, so the
                # same cut keeps the shapes matched).
                if opt.kind[j] != 0:
                    ok = False
                    break
                dim = (float(hi[j]) - tgt) if side == 0 \
                    else (tgt - float(lo[j]))
                area2 = dim * float(Q[j, o + 2])
                if dim <= eps or area2 < float(opt.areas[j]) * 0.992:
                    ok = False
                    break
                group = ()
                if opt.mib[j] > 0:
                    group = tuple(
                        int(k)
                        for idxs in opt.mib_groups.values()
                        if any(int(v) == j for v in idxs)
                        for k in idxs if int(k) != j)
                    if any(opt.kind[k] != 0
                           or abs(Q[k, axis + 2] - Q[j, axis + 2]) > 1e-6
                           for k in group):
                        ok = False
                        break
                if side == 0:
                    Q[j, axis] = tgt
                Q[j, axis + 2] = dim
                for k in group:
                    Q[k, axis + 2] = dim
            if ok:
                V = _viol(Q)
                if V < Vcur:
                    Vcur = V
                else:
                    Q[...] = snap
            else:
                Q[...] = snap
    if Vcur < V0:
        if _DEBUG:
            print(f"[seat] V {V0}->{Vcur}", flush=True)
        return Q
    return out


def _lock_compact(opt, out, deadline: float, seed: int = 0,
                  pred=None):
    """Discrete area-compaction toward the tag-locked frame, accepted on
    the candidate-selection proxy: seating a preplaced tag (V drop) and
    shedding the overshoot bbox area both pay, and a small HPWL cost from
    relocations is arbitrated instead of vetoed.  `pred` (the model's
    raw prediction) anchors relocation landings: a block evicted far
    from its predicted spot during legalization lands back near the
    model's intent instead of wherever the HPWL median points."""
    try:
        if time.time() >= deadline - 0.15:
            return out
        Q = np.asarray(out, dtype=np.float64)
        r = _Refiner(opt, Q, seed + 13)
        if pred is not None:
            Pp = np.asarray(pred, dtype=np.float64)
            r._pred_c = np.stack([Pp[:, 0] + 0.5 * Pp[:, 2],
                                  Pp[:, 1] + 0.5 * Pp[:, 3]], axis=1)
        r._anchor_frame_to_tags()
        if r.lock_xmin is None and r.lock_xmax is None \
                and r.lock_ymin is None and r.lock_ymax is None:
            return out
        hp0 = opt._hpwl(Q)
        V0 = full_violations(opt, Q)
        den = max(getattr(opt, "n_soft_den", 1), 1)
        ref = max(hp0, 1e-9)

        def judge(P):
            ar = float(((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                       * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))
            return (1.0 + 0.5 * ((opt._hpwl(P) - ref) / ref
                                 + max(0.0, ar / opt.area_ref - 1.0))) \
                * math.exp(2.0 * full_violations(opt, P) / den)

        if not r.compact_to_locks(deadline, judge):
            return out
        if r._has_overlap():
            return out
        P1 = r.P
        if _DEBUG:
            ar0 = float(((Q[:, 0] + Q[:, 2]).max() - Q[:, 0].min())
                        * ((Q[:, 1] + Q[:, 3]).max() - Q[:, 1].min()))
            ar1 = float(((P1[:, 0] + P1[:, 2]).max() - P1[:, 0].min())
                        * ((P1[:, 1] + P1[:, 3]).max() - P1[:, 1].min()))
            print(f"[compact] V {V0}->{full_violations(opt, P1)} "
                  f"area {ar0:.0f}->{ar1:.0f} "
                  f"hp {hp0:.1f}->{opt._hpwl(P1):.1f}", flush=True)
        return P1.copy()
    except Exception:
        if _DEBUG:
            import traceback
            traceback.print_exc()
        return out


# ---------------------------------------------------------------------------
# PARTNER_REFINE_GUARD (default off) -- no-degradation guard on the pipeline's
# own output.
#
# `refine_prediction` is a PIPELINE, not a search: every stage carries its own
# acceptance gate, and several of those gates are blind to terms the official
# cost charges for.  `_edge_seat` / `_cluster_seat` keep a result whenever the
# violation count strictly drops (bbox area is not consulted); the step-6
# repair loop arbitrates on (V, hpwl) only; and the ladder itself re-legalizes
# into a frame sized from `area_ref * 1.08`, which `_seed_tags` then fills
# exactly.  Nothing anywhere asks whether what comes out is better than what
# went in, or than an earlier stage's own output.
#
# Measured on the G-T3-1 oracle replay (a repaired golden layout injected as
# the prediction, so the input is feasible at cost 1.0000): tid 87 comes back
# at 1.1328 -- bbox exactly `1.08 * area_ref` (+5.2% over the evaluator's area
# baseline, +0.026) plus 2 soft violations the input did not have (x1.0975).
#
# The guard re-scores the pipeline's stage snapshots -- and the INPUT itself,
# when the input passes a hard-legality re-check -- under the same
# evaluator-style cost `_lock_compact.judge` already uses, and returns the
# best.  It is a SELECTION, never a new layout: everything it can return was
# produced (and overlap-checked) by the pipeline or handed in by the caller.
#
# Off (the default) `_GUARD_ON` is a falsy module constant, so `_snaps` stays
# None, every capture site collapses to one `is not None` test, and the
# returned layout is bit-identical to the shipped path.
_GUARD_ON = _flag_on("PARTNER_REFINE_GUARD")
_GUARD_DBG = _flag_on("PARTNER_REFINE_GUARD_DEBUG")
# The guard costs ~1 hpwl + 1 `full_violations` per snapshot.  Restricting it
# to the tail (n >= 95) is the same reusable instance statistic the direct
# channel's own slots use (`PARTNER_NREF_MIN_N`), not a case list.
try:
    _GUARD_MIN_N = int(float(os.environ.get("PARTNER_REFINE_GUARD_MIN_N", "95")))
except ValueError:
    _GUARD_MIN_N = 95
# accept a substitution only on a strict, non-noise improvement
_GUARD_EPS = 1e-9
# half the evaluator's 1% soft-area tolerance: the guard may only hand back an
# input it is sure the evaluator will call feasible
_GUARD_AREA_TOL = 0.005
# dimension / preplaced-origin tolerance.  The evaluator's own hard check runs
# at 1e-4; 1e-5 keeps a 10x margin while staying above the float drift the
# pipeline itself leaves on a legal layout (measured ~2e-6 on a preplaced
# origin after a full `refine_prediction`), so a legitimate layout is not
# refused by a tolerance tighter than the rule it is protecting.
_GUARD_DIM_TOL = 1e-5


# --- rung-(-1): "legal is admissible" --------------------------------------
# PARTNER_LEGAL_ADMIT=1 (default off, read once at import).
#
# The direct channel's coverage boundary is not the sampler, it is the LADDER:
# below n ~ 101 the per-case budget (0.05-0.33 s) cannot pay for rung 0's
# `_Refiner` build + fixed-frame legalization, so `refine_prediction` returns
# None and the reserved pool worker contributes nothing -- even when the
# prediction handed to it is *already* a legal floorplan.  Rung (-1) sits below
# rung 0: if the ladder produced nothing, but the caller's raw input passes the
# same evaluator-faithful hard-legality re-check the guard uses
# (`_guard_hard_ok`: overlap 1e-7, soft area +/-0.5%, fixed/MIB dims, preplaced
# origin), hand the input back as the candidate instead of None.
#
# Two properties make this cheap rather than a new search stage:
#   * it only ever fires where the shipped path returns None, so it cannot
#     displace a ladder result and adds no wall clock to a case that succeeds;
#   * the check itself is one O(n^2) numpy overlap matrix (n <= 120), paid once
#     per failed refine worker, after that worker's slice is already spent.
# The second call site is the guard's INPUT arm at any block count: the shipped
# guard is restricted to n >= PARTNER_REFINE_GUARD_MIN_N (95), so a small case
# whose ladder returned a legal-but-worse layout still discards a legal input.
#
# Off -> `_LEGAL_ADMIT` is a falsy module constant, `_admit_legal_input`
# returns None before touching numpy, and every call site collapses to the
# shipped expression bit for bit.
_LEGAL_ADMIT = _flag_on("PARTNER_LEGAL_ADMIT")
_LEGAL_ADMIT_DBG = _flag_on("PARTNER_LEGAL_ADMIT_DEBUG")


def _admit_legal_input(opt, pred, why: str):
    """Rung (-1).  Return `pred` iff rung (-1) is on and `pred` is hard-legal.

    Returns None in every other case -- including the flag being off, which is
    tested first so the shipped path never builds an array or imports a thing.
    """
    if not _LEGAL_ADMIT:
        return None
    ok = False
    P = None
    try:
        P = np.asarray(pred, dtype=np.float64)
        ok = _guard_hard_ok(opt, P)
    except Exception:
        ok = False
    if _LEGAL_ADMIT_DBG:
        import sys as _sys
        print(f"[admit] n={getattr(opt, 'n', -1)} why={why} ok={int(ok)}",
              file=_sys.stderr, flush=True)
    return P.copy() if ok else None


def _guard_bbox(P: np.ndarray) -> float:
    return float(((P[:, 0] + P[:, 2]).max() - P[:, 0].min())
                 * ((P[:, 1] + P[:, 3]).max() - P[:, 1].min()))


def _guard_cost(opt, P: np.ndarray, hp_ref: float, den: int) -> float:
    """Evaluator-style no-runtime cost of one layout.

    Same functional form as the official cost and as `_lock_compact.judge`:
    `(1 + 0.5*(hpwl_gap + max(0, area_gap))) * exp(2*V/N_soft)`, with
    `full_violations` (boundary + grouping + MIB, the official soft set) and
    `opt.area_ref` standing in for the evaluator's area baseline.

    Two known deviations from the official number, both harmless for a
    same-case COMPARISON: the hpwl term is referenced to `hp_ref` instead of
    the (unknowable at solve time) golden hpwl, which rescales that term by
    `hp_ref / hpwl_baseline` (~1.01-1.08 on the tail, i.e. the guard slightly
    under-weights hpwl); and the official `max(0, hpwl_gap)` clip is dropped,
    which is inert here because every direct-channel candidate measured at
    n >= 95 sits above the golden hpwl.  `area_ref = total_area / 0.97` is
    within ~1.1% of the golden bbox on the measured tail cases (golden
    utilization 0.959-0.970), so the area clip is meaningful and kept.
    """
    return (1.0 + 0.5 * ((float(opt._hpwl(P)) - hp_ref) / hp_ref
                         + max(0.0, _guard_bbox(P) / opt.area_ref - 1.0))) \
        * math.exp(2.0 * full_violations(opt, P) / den)


def _guard_overlap(P: np.ndarray) -> bool:
    """True iff some pair overlaps by more than 1e-7 on BOTH axes (10x
    stricter than the evaluator's 1e-6, matching `violation_killer`)."""
    x0 = P[:, 0]
    y0 = P[:, 1]
    x1 = x0 + P[:, 2]
    y1 = y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > 1e-7) & (oy > 1e-7)
    np.fill_diagonal(bad, False)
    return bool(bad.any())


def _guard_hard_ok(opt, P: np.ndarray) -> bool:
    """Hard legality of a candidate the pipeline did NOT produce.

    Only the caller's raw prediction goes through here; the stage snapshots
    are legal by construction.  Every hard constraint the evaluator can call
    infeasible on is re-checked: overlap, soft-block exact area, fixed-shape
    dimensions, preplaced dimensions AND origin.
    """
    if P.ndim != 2 or P.shape[1] != 4 or P.shape[0] != opt.n:
        return False
    if not np.isfinite(P).all() or (P[:, 2] <= 0).any() or (P[:, 3] <= 0).any():
        return False
    kind = np.asarray(opt.kind)
    areas = np.asarray(opt.areas, dtype=np.float64)
    rw = np.asarray(opt.rw, dtype=np.float64)
    rh = np.asarray(opt.rh, dtype=np.float64)
    soft = kind == 0
    if soft.any():
        a = P[soft, 2] * P[soft, 3]
        if (np.abs(a - areas[soft]) / np.maximum(areas[soft], 1e-9)
                > _GUARD_AREA_TOL).any():
            return False
    hard = kind != 0
    if hard.any():
        if (np.abs(P[hard, 2] - rw[hard]) > _GUARD_DIM_TOL).any() \
                or (np.abs(P[hard, 3] - rh[hard]) > _GUARD_DIM_TOL).any():
            return False
    lock = kind == 2
    if lock.any():
        lx = np.asarray(getattr(opt, "lx", None), dtype=np.float64)
        ly = np.asarray(getattr(opt, "ly", None), dtype=np.float64)
        if lx.shape != (opt.n,) or ly.shape != (opt.n,):
            return False
        if (np.abs(P[lock, 0] - lx[lock]) > _GUARD_DIM_TOL).any() \
                or (np.abs(P[lock, 1] - ly[lock]) > _GUARD_DIM_TOL).any():
            return False
    return not _guard_overlap(P)


def _guard_pick(opt, out, snaps, pred):
    """Return whichever of `out` / the stage snapshots / the input scores
    best under `_guard_cost`.  Returns `out` unchanged on any failure."""
    try:
        P = np.asarray(out, dtype=np.float64)
        den = max(getattr(opt, "n_soft_den", 1), 1)
        hp_ref = max(float(opt._hpwl(P)), 1e-9)
        best = None
        best_c = _guard_cost(opt, P, hp_ref, den)
        base_c = best_c
        cands = list(snaps)
        if pred is not None:
            cands.append(("input", np.asarray(pred, dtype=np.float64)))
        for tag, Q in cands:
            if Q.shape != P.shape:
                continue
            c = _guard_cost(opt, Q, hp_ref, den)
            if _GUARD_DBG:
                import sys as _sys
                print(f"[guard] n={opt.n} {tag:<8} cost={c:.6f} "
                      f"hp={opt._hpwl(Q):.1f} area={_guard_bbox(Q):.0f} "
                      f"V={full_violations(opt, Q)} den={den} "
                      f"aref={opt.area_ref:.0f}", file=_sys.stderr, flush=True)
            if c >= best_c - _GUARD_EPS:
                continue
            ok = (_guard_hard_ok(opt, Q) if tag == "input"
                  else not _guard_overlap(Q))
            if ok:
                best = (tag, Q)
                best_c = c
        if _GUARD_DBG:
            import sys as _sys
            print(f"[guard] n={opt.n} out cost={base_c:.6f} -> "
                  f"{best[0] if best else 'out'} {best_c:.6f}",
                  file=_sys.stderr, flush=True)
        if best is None:
            return out
        return best[1].copy()
    except Exception:
        if _DEBUG or _GUARD_DBG:
            import traceback
            traceback.print_exc()
        return out


def refine_prediction(opt, pred: np.ndarray, deadline: float,
                      seed: int = 0, _depth: int = 0) -> Optional[np.ndarray]:
    """Full direct-prediction pipeline glue:

      1. MIB shape unification (identical (w,h) floats per group).
      2. Tag seeding: tagged blocks snapped to the frame edge and pinned.
      3. Min-displacement legalization (retry ladder: growing frame
         expansion, finally dropping the tag pins).
      4. Cluster reassembly into exact-contact groups.
      5. Fresh refiner on the now-legal layout (rigid cluster groups with
         recorded contacts, pins re-derived) -> continuous + discrete
         refinement + squeeze.

    Returns refined positions or None if no attempt legalized."""
    try:
        if opt.n < 2 or time.time() >= deadline:
            # rung (-1): a worker handed no budget at all still costs a pool
            # slot; a legal input is a free candidate.  Off -> None, as shipped.
            return (_admit_legal_input(opt, pred, "nobudget")
                    if _depth == 0 else None)
        # PARTNER_REFINE_GUARD stage snapshots; None (the default) makes every
        # capture site below one `is not None` test and changes nothing.
        _snaps = ([] if (_GUARD_ON and _depth == 0 and opt.n >= _GUARD_MIN_N)
                  else None)
        # reserve the violation-repair slice up front — the legalization
        # rungs and the refiner will consume every second they are given,
        # and an unrepaired candidate (drifted tags, broken clusters)
        # loses the cross-pipeline pick no matter how good its HPWL is
        t_hard = deadline
        slice_ = max(0.0, deadline - time.time())
        res = min(3.5, 0.3 * slice_)
        if _depth == 0 and slice_ > 8.0:
            # the recompression retry (step 7) reruns the whole pipeline
            # on the legal layout — give it a real share of the slice,
            # but only when the slice can afford three stages at all
            res += 0.35 * slice_
        deadline = deadline - res

        # -- anytime budget plan (PARTNER_ANYTIME_LADDER=1, default off) ---
        # Shipped ladder semantics are "try tight frames in order until one
        # legalizes"; below ~3 s of worker deadline the tight rungs consume
        # the whole span, so the pipeline returns None (the pool slot is
        # wasted) or a loose-frame layout whose quality tail never runs.
        # ANYTIME bounds the tight rungs, keeps a guaranteed-legal rung
        # reachable, and pays the tail out of fractions of the carved
        # reserve instead of absolute second offsets.
        _any = anytime_ladder_on()
        t_lad0 = time.time()
        span_lad = max(0.0, deadline - t_lad0)
        # end of the TIGHT (fixed-frame + small-expand) rungs.  Its real
        # value is set once the first `_Refiner` build has been timed; until
        # then it is non-binding, exactly as on the shipped path.
        t_tight = deadline
        t_build = 0.0

        # captured, not read from `res`: step 6 below rebinds the name `res`
        # to a repair attempt's result
        _reserve = res

        def _tg(off_val: float, frac: float) -> float:
            """Tail budget term: the shipped absolute constant, capped by a
            share of the carved reserve when ANYTIME is on (off -> the
            shipped constant, unchanged)."""
            return off_val if not _any else min(off_val, frac * _reserve)
        P0 = np.array(pred, dtype=np.float64, copy=True)
        # the raw prediction's centers anchor relocation landings all
        # through legalization: the HPWL median points at the congested
        # core where no free slot exists, so an evicted block's landing
        # was effectively arbitrary — the prediction re-anchors it to
        # the model's intent (same idea as _lock_compact's landings)
        pred_c = np.stack([P0[:, 0] + 0.5 * P0[:, 2],
                           P0[:, 1] + 0.5 * P0[:, 3]], axis=1)

        # -- 1. MIB shape unification ----------------------------------
        # Reference dims come from a true fixed/preplaced member if any
        # (their shapes are hard); every other member — including soft
        # members that _resolve_shapes promoted to kind 1 as MIB copies —
        # gets the exact same floats, so round(w/h, 4) matches everywhere.
        for idxs in opt.mib_groups.values():
            if len(idxs) < 2:
                continue
            ref = next((i for i in idxs
                        if opt.fixed[i] or opt.preplaced[i]), None)
            if ref is not None:
                w, h = float(opt.rw[ref]), float(opt.rh[ref])
            else:
                la = float(np.mean([math.log(P0[i, 2] / P0[i, 3]) for i in idxs]))
                area = float(opt.areas[idxs[0]])
                w = math.sqrt(area * math.exp(la))
                h = area / w
            for i in idxs:
                # fixed/preplaced dims are hard constraints — never touch
                if opt.kind[i] != 2 and not (opt.fixed[i] or opt.preplaced[i]):
                    cx = P0[i, 0] + 0.5 * P0[i, 2]
                    cy = P0[i, 1] + 0.5 * P0[i, 3]
                    P0[i] = (cx - 0.5 * w, cy - 0.5 * h, w, h)

        saved_cg = opt.cluster_groups
        legal = None

        # -- rung 0: fixed-frame legalization ---------------------------
        # The tagged-preplaced edges LEAK the true frame (user insight);
        # legalize straight into it instead of expanding and squeezing
        # back — every stage of expand/squeeze/repair fights the previous
        # one and destroys the model's structure.  Missing sides come
        # from the area reference.  This only became viable once the
        # wedge tools existed (unpin retry, free-rect relocation, global
        # evict) — a bare clamp used to kill every legalization.
        if time.time() < deadline:
            # -- rung-0 frame-scale ladder (PARTNER_FRAME_SCALE_LADDER) ---
            # Off -> `_fs_set == (1.08,)`: one attempt at the shipped scale,
            # `_fs_last` True on it, `_a_end is _r0_end` -- bit for bit the
            # shipped rung.  See `frame_scale_set`.
            _fs_set = frame_scale_set()
            _fs_min_n = int(_env_pos("PARTNER_FRAME_SCALE_MIN_N", 0.0, 1e6))
            if len(_fs_set) > 1 and opt.n < _fs_min_n:
                # budget band (a reusable instance statistic, never a case
                # id): below it only the widest scale is affordable, which is
                # the shipped attempt.
                _fs_set = _fs_set[-1:]
            # the shipped rung-0 window, measured once so the tight probes
            # are sized against it and not against each other's leftovers
            _fs_win = 0.35 * max(0.0, deadline - time.time())
            _fs_frac = _env_pos("PARTNER_FRAME_SCALE_TIGHT_FRAC", 0.5, 1.0)
            _fs_live = False
            for _fs_i, _fs in enumerate(_fs_set):
                _fs_last = (_fs_i == len(_fs_set) - 1)
                _fs_t0 = time.time()
                if legal is not None or _fs_t0 >= deadline:
                    break
                _t_b0 = time.time()
                try:
                    opt.cluster_groups = {}
                    r = _Refiner(opt, P0, seed + 9)
                finally:
                    opt.cluster_groups = saved_cg
                _dbg_build = (time.time() - _t_b0) if _SEAT_DBG else 0.0
                if _any and _fs_i == 0:
                    # The `_Refiner` build is O(n^2) and machine-relative, so it
                    # is the instance's own cost unit: every cap below is sized
                    # in build-times, never in absolute seconds.  A rung costs
                    # roughly 4-7 builds, so keeping `BUILD_MULT` builds back
                    # guarantees the secure rung is still affordable when the
                    # tight rungs fail.
                    t_build = max(time.time() - _t_b0, 1e-6)
                    keep = max(anytime_frac("PARTNER_ANYTIME_SECURE_MIN", 0.25)
                               * span_lad,
                               _env_pos("PARTNER_ANYTIME_BUILD_MULT", 6.0, 100.0)
                               * t_build)
                    # ... but never more than SECURE_MAX of the span: with the
                    # 0.65 default the tight window keeps >= 0.35 of the span,
                    # which is exactly the fixed-frame rung's own shipped share,
                    # so ANYTIME never shortens that rung -- only the
                    # intermediate rungs and the unbounded calls.
                    keep = min(keep, anytime_frac("PARTNER_ANYTIME_SECURE_MAX",
                                                  0.65) * span_lad)
                    # the reserve is the ONLY thing that shortens the tight
                    # rungs: on a cheap instance `keep` is small and the fixed
                    # frame rung keeps its shipped window (its own 0.35 share
                    # below is never widened), on an expensive one the reserve
                    # binds and the secure rung stays affordable.
                    t_tight = max(deadline - keep, t_lad0 + 0.10 * span_lad)
                    if _DEBUG:
                        print(f"[any] n={opt.n} span={span_lad:.3f} "
                              f"build={t_build:.3f} keep={keep:.3f} "
                              f"tight={t_tight - t_lad0:.3f} res={res:.3f}",
                              flush=True)
                _r0_end = min(deadline, t_tight) if _any else deadline
                # per-attempt hard end: the LAST scale keeps the shipped
                # rung-0 window (`_a_end is _r0_end`, so a single-scale set is
                # bit-exact); the tighter probes before it are capped by a
                # share of that window.
                _a_end = _r0_end if _fs_last else min(
                    _r0_end, time.time() + _fs_frac * _fs_win)
                r._pred_c = pred_c
                r._anchor_frame_to_tags()
                W = (r.lock_xmax - r.xmin) if r.lock_xmax is not None else None
                H = (r.lock_ymax - r.ymin) if r.lock_ymax is not None else None
                # rung-0 frame area.  `_fs` is the shipped 1.08 unless the
                # frame-scale ladder is on; it is only ever consulted for a
                # side that is NOT leaked by a tagged preplaced block, so
                # `_fs_live` records whether this instance can feel it at all.
                _fs_live = (W is None) or (H is None)
                aref = opt.area_ref * _fs
                if W is None and H is not None:
                    W = aref / H
                elif H is None and W is not None:
                    H = aref / W
                elif W is None and H is None:
                    # no tagged preplaced to leak the frame: take the area
                    # reference at the PREDICTION's aspect ratio — the model
                    # sees the pin/frame anchors and its bbox shape is close
                    bw = float((r.P[:, 0] + r.P[:, 2]).max() - r.P[:, 0].min())
                    bh = float((r.P[:, 1] + r.P[:, 3]).max() - r.P[:, 1].min())
                    if bw > 1e-6 and bh > 1e-6:
                        H = math.sqrt(aref * bh / bw)
                        W = aref / H
                if W is not None:
                    k2i = np.nonzero(r.kind == 2)[0]
                    if len(k2i):
                        W = max(W, float((r.P[k2i, 0] + r.P[k2i, 2]).max())
                                - r.xmin)
                        H = max(H, float((r.P[k2i, 1] + r.P[k2i, 3]).max())
                                - r.ymin)
                    r.xmax = r.xmin + W
                    r.ymax = r.ymin + H
                    for g in r.groups:
                        g.pin_x = g.pin_y = False
                    r.satL[:] = False
                    r.satR[:] = False
                    r.satB[:] = False
                    r.satT[:] = False
                    r._pull_inside_frame()
                    r._seed_tags()
                    sub = min(_a_end,
                              time.time() + 0.35 * (deadline - time.time()))
                    ok0 = r.legalize_soft(12, deadline=sub, fine=True)
                    if not ok0 and r._overlap_count() <= 6 \
                            and time.time() < _a_end:
                        # inches from closing: give the frame 2% and finish —
                        # still far tighter than the loose-ladder path, and
                        # the tag seats survive intact
                        r.xmax += 0.02 * W
                        r.ymax += 0.02 * H
                        # the salvage is the fixed-frame rung's last chance and
                        # the highest-value branch in the ladder: under ANYTIME
                        # it gets the whole remaining tight window (the reserve
                        # is the guardrail), not a share of it
                        _sal = _a_end if _any else (time.time() + 1.5)
                        ok0 = r.legalize_soft(10, deadline=min(_a_end, _sal))
                    if not ok0 and r._overlap_count() <= 2 \
                            and time.time() < _a_end:
                        # final pair(s): a wedge where every participant is
                        # tagged/fixed — sacrifice the tag seat (V+1) for the
                        # tight frame; the cross-candidate selection judges
                        r._ignore_tags = True
                        try:
                            rects = r._free_rects(8)
                            for a, b in r._overlap_pairs(4):
                                if r.kind[a] == 2 and (r._reshape_clear(b, a)
                                                       or r._reshape_chain(b, a)):
                                    continue
                                if r.kind[b] == 2 and (r._reshape_clear(a, b)
                                                       or r._reshape_chain(a, b)):
                                    continue
                                if r._reshape_chain(b, a) \
                                        or r._reshape_chain(a, b):
                                    continue
                                m = b if (r._movable_free(b, allow_fixed=True)
                                          and (not r._movable_free(
                                              a, allow_fixed=True)
                                              or r.P[b, 2] * r.P[b, 3]
                                              <= r.P[a, 2] * r.P[a, 3])) else a
                                if r._movable_free(m, allow_fixed=True):
                                    gi = int(r.group_of[m])
                                    if gi >= 0:
                                        r.groups[gi].pin_x = False
                                        r.groups[gi].pin_y = False
                                    r.satL[m] = r.satR[m] = False
                                    r.satB[m] = r.satT[m] = False
                                    r._relocate_to_free(int(m), rects)
                            ok0 = r.legalize(12, deadline=_a_end)
                        finally:
                            r._ignore_tags = False
                    if ok0 and not r._has_overlap():
                        if _DEBUG:
                            bb = ((r.P[:, 0] + r.P[:, 2]).max()
                                  - r.P[:, 0].min()) * \
                                ((r.P[:, 1] + r.P[:, 3]).max()
                                 - r.P[:, 1].min())
                            print(f"[rp] rung FIXED legal bbox={bb:.0f} "
                                  f"aref={opt.area_ref:.0f}", flush=True)
                        r._assemble_clusters(saved_cg)
                        if not r._has_overlap():
                            legal = r.P.copy()
                    elif os.environ.get("PARTNER_RUNG05"):
                        # rung 0.5: fixed-frame salvage.  The min-displacement
                        # toolkit preserves structure but stalls 20-35 overlaps
                        # short at ~96% frame utilization; a from-scratch
                        # repack fits but destroys structure.  Hybrid: keep the
                        # legalized majority, extract only the residual
                        # overlappers (prefer soft / untagged / ungrouped) and
                        # re-place them into the frame's free space (MaxRects,
                        # reshape allowed, landing near the model's intent).
                        try:
                            from frame_repack import _split_free, _prune
                            P = r.P
                            nn = len(P)
                            x0 = P[:, 0]
                            y0 = P[:, 1]
                            x1 = x0 + P[:, 2]
                            y1 = y0 + P[:, 3]
                            oxm = (np.minimum(x1[:, None], x1[None, :])
                                   - np.maximum(x0[:, None], x0[None, :]))
                            oym = (np.minimum(y1[:, None], y1[None, :])
                                   - np.maximum(y0[:, None], y0[None, :]))
                            badm = (oxm > 1e-7) & (oym > 1e-7)
                            np.fill_diagonal(badm, False)
                            codes = np.zeros(nn, dtype=np.int64)
                            for bi_, bc_ in zip(opt._bnd_idx, opt._bnd_codes):
                                codes[int(bi_)] = int(bc_)
                            offenders: list = []
                            salv_ok = True
                            for _ in range(40):
                                cnt = badm.sum(1)
                                if cnt.max() == 0:
                                    break
                                cand = np.nonzero(cnt > 0)[0]
                                j = min(cand, key=lambda t: (
                                    (r.kind[t] == 2) * 1000
                                    + (codes[t] != 0) * 100
                                    + int(r.in_cluster[t]) * 10
                                    + (r.kind[t] == 1) * 5
                                    - int(cnt[t])))
                                if r.kind[j] == 2:
                                    salv_ok = False
                                    break
                                offenders.append(int(j))
                                badm[j, :] = False
                                badm[:, j] = False
                            if salv_ok and offenders and badm.sum() == 0:
                                free = [(float(r.xmin), float(r.ymin),
                                         float(r.xmax - r.xmin),
                                         float(r.ymax - r.ymin))]
                                offs = set(offenders)
                                for j in range(nn):
                                    if j in offs:
                                        continue
                                    used = (float(x0[j]), float(y0[j]),
                                            float(P[j, 2]), float(P[j, 3]))
                                    nxt = []
                                    for fr in free:
                                        nxt.extend(_split_free(fr, used))
                                    free = _prune(nxt)
                                for j in sorted(offenders,
                                                key=lambda t:
                                                -float(opt.areas[t])):
                                    a_j = float(opt.areas[j])
                                    if r._pred_c is not None:
                                        px_ = float(r._pred_c[j, 0])
                                        py_ = float(r._pred_c[j, 1])
                                    else:
                                        px_, py_ = float(x0[j]), float(y0[j])
                                    bestp = None
                                    for fx, fy, fw, fh in free:
                                        cands2 = []
                                        if r.kind[j] != 0:
                                            cands2.append((float(P[j, 2]),
                                                           float(P[j, 3])))
                                        else:
                                            for wf in (float(P[j, 2]), fw,
                                                       (a_j / fh) if fh > 0
                                                       else 0.0):
                                                if wf <= 0:
                                                    continue
                                                hf = a_j / wf
                                                if (wf <= fw + 1e-9
                                                        and hf <= fh + 1e-9
                                                        and max(wf / hf,
                                                                hf / wf)
                                                        <= 12.0):
                                                    cands2.append((wf, hf))
                                        for w2, h2 in cands2:
                                            if (w2 > fw + 1e-9
                                                    or h2 > fh + 1e-9):
                                                continue
                                            x2 = min(max(px_ - 0.5 * w2, fx),
                                                     fx + fw - w2)
                                            y2 = min(max(py_ - 0.5 * h2, fy),
                                                     fy + fh - h2)
                                            d = (abs(x2 + 0.5 * w2 - px_)
                                                 + abs(y2 + 0.5 * h2 - py_))
                                            if bestp is None or d < bestp[0]:
                                                bestp = (d, x2, y2, w2, h2)
                                    if bestp is None:
                                        salv_ok = False
                                        break
                                    _d, x2, y2, w2, h2 = bestp
                                    P[j] = (x2, y2, w2, h2)
                                    nxt = []
                                    for fr in free:
                                        nxt.extend(_split_free(
                                            fr, (x2, y2, w2, h2)))
                                    free = _prune(nxt)
                                if salv_ok and not r._has_overlap():
                                    r._assemble_clusters(saved_cg)
                                    if not r._has_overlap():
                                        legal = r.P.copy()
                                        if _DEBUG:
                                            print(f"[rp] rung 0.5 salvage legal"
                                                  f" ({len(offenders)} moved)",
                                                  flush=True)
                            if legal is None and _DEBUG:
                                print(f"[rp] rung 0.5 salvage failed "
                                      f"(off={len(offenders)})", flush=True)
                        except Exception:
                            if _DEBUG:
                                import traceback as _tb
                                _tb.print_exc()
                    elif _DEBUG:
                        print(f"[rp] rung FIXED failed "
                              f"ovl={r._overlap_count()}", flush=True)
                if _FS_DBG:
                    import sys as _sys
                    _FS_STATS.append((int(opt.n), float(_fs),
                                      int(legal is not None),
                                      int(_fs_live),
                                      round(time.time() - _fs_t0, 4)))
                    print(f"[fs] n={opt.n} scale={_fs:.4f} "
                          f"ok={int(legal is not None)} "
                          f"live={int(_fs_live)} "
                          f"t={time.time() - _fs_t0:.4f}",
                          file=_sys.stderr, flush=True)
                if not _fs_live or W is None:
                    # `aref` never entered the geometry -- both frame sides
                    # are leaked by tagged preplaced blocks, or the frame
                    # could not be sized at all.  Every other scale would
                    # replay this exact attempt; stop paying for it.
                    break
            if _SEAT_DBG:
                import sys as _sys
                print(f"[rp0] n={opt.n} slice={slice_:.4f} "
                      f"span={span_lad:.4f} build={_dbg_build:.4f} "
                      f"r0={int(legal is not None)} "
                      f"t={time.time() - t_lad0:.4f}",
                      file=_sys.stderr, flush=True)

        rungs = ((0.02, True), (0.05, True), (0.08, True), (0.12, True),
                 (0.18, False), (0.28, False))
        if _any:
            # escape rung: reached only when the secure rung above failed and
            # time is left.  A legal-but-loose candidate loses the
            # cross-pipeline selection at worst; a missing candidate wastes
            # the pool slot outright.
            rungs = rungs + ((0.50, False),)
        # index of the first SECURE rung: from here on the rung is the one
        # that (almost) always legalizes, so it is budgeted as a share of
        # what is left rather than out of the tight-rung window.
        i_secure = len(rungs) - (2 if _any else 1)
        # cap the ladder at 45% of the slice: the last (loose) rung almost
        # always legalizes, and the time saved goes to squeeze/refine which
        # recovers the bbox inflation the loose rung causes
        rung_cap = time.time() + 0.45 * (deadline - time.time())
        if _any:
            # ... and, under ANYTIME, never past the tight-rung budget: the
            # `continue` below then jumps straight to the secure rung.
            rung_cap = min(rung_cap, t_tight)
        for ridx, (expand, use_pins) in enumerate(rungs):
            if legal is not None or time.time() >= deadline:
                break
            if time.time() > rung_cap and ridx < i_secure:
                continue
            try:
                opt.cluster_groups = {}
                r = _Refiner(opt, P0, seed)
            finally:
                opt.cluster_groups = saved_cg
            r._pred_c = pred_c
            r.xmax += expand * (r.xmax - r.xmin)
            r.ymax += expand * (r.ymax - r.ymin)
            for g in r.groups:
                g.pin_x = g.pin_y = False
            r.satL[:] = False
            r.satR[:] = False
            r.satB[:] = False
            r.satT[:] = False
            r._anchor_frame_to_tags()
            if use_pins:
                r._seed_tags()
            # ANYTIME: the shipped call is unbounded (`deadline=None`), so a
            # single rung can and does overrun the whole worker deadline.
            # Tight rungs are bounded by the tight budget; the last (secure)
            # rung keeps a share back for the frame anneal and the refiner.
            _rdl = None
            if _any:
                if ridx >= i_secure:
                    _rdl = min(deadline, time.time() + anytime_frac(
                        "PARTNER_ANYTIME_SECURE", 0.55)
                        * max(0.0, deadline - time.time()))
                else:
                    _rdl = min(deadline, t_tight)
            if r.legalize_soft(deadline=_rdl):
                if not use_pins:
                    # tag recovery: the pin-less rung legalized but every
                    # boundary tag is loose — try to re-seat them; revert
                    # if the layout cannot absorb it
                    snap_t = r.P.copy()
                    r._seed_tags()
                    if not r.legalize_soft(10, deadline=_rdl):
                        r.P[...] = snap_t
                        for g in r.groups:
                            g.pin_x = g.pin_y = False
                        r.satL[:] = False
                        r.satR[:] = False
                        r.satB[:] = False
                        r.satT[:] = False
                if _DEBUG:
                    bb = ((r.P[:, 0] + r.P[:, 2]).max() - r.P[:, 0].min()) * \
                        ((r.P[:, 1] + r.P[:, 3]).max() - r.P[:, 1].min())
                    print(f"[rp] rung expand={expand} legal bbox={bb:.0f}",
                          flush=True)
                # frame anneal: the shipped 3.0 s cap is an absolute-time
                # assumption (it never binds below a ~4 s worker deadline,
                # where the anneal is instead starved to ~0).  ANYTIME gives
                # it a share of what is left, so it scales with the tier.
                _tg_end = (time.time() + anytime_frac(
                    "PARTNER_ANYTIME_TIGHTEN", 0.45)
                    * max(0.0, deadline - time.time())) if _any \
                    else (time.time() + 3.0)
                r._tighten(min(deadline, _tg_end))
                r._assemble_clusters(saved_cg)
                if not r._has_overlap():
                    legal = r.P.copy()
                    break
                elif _DEBUG:
                    print("[rp] assembly created overlap", flush=True)
        if legal is None:
            # rung (-1): the whole ladder failed to legalize.  Off -> None,
            # the shipped return, bit for bit.
            return (_admit_legal_input(opt, pred, "ladder")
                    if _depth == 0 else None)
        if _snaps is not None:
            _snaps.append(("legal", legal.copy()))

        # -- 5. fresh refiner on the legal layout ----------------------
        r2 = _Refiner(opt, legal, seed + 1)
        r2._pred_c = pred_c
        r2.enable_deflate = True
        r2._anchor_frame_to_tags()
        run_end = deadline - 0.02
        out = r2.run(run_end)
        if _snaps is not None:
            _snaps.append(("run", np.asarray(out, dtype=np.float64).copy()))
        # EARLY_EXIT clawback: `run` is the search stage of this pipeline;
        # steps 6/6.5/7 below are its repair + retry tail, sized off
        # `t_hard`.  Without the clawback a converged `run` silently donates
        # its unspent share to the step-7 recompression rerun (which reruns
        # the WHOLE pipeline) -- that is the promoted stall-stop's quality
        # mechanism, and the exact opposite of returning wall clock.  Shrink
        # `t_hard` by the unspent share so the tail keeps its own reserve and
        # nothing else, and the step-7 `t_hard - 2.5` gate self-disables when
        # the retry no longer fits inside the planned span.
        if r2._early_exit:
            t_hard = max(time.time(), t_hard - max(0.0, run_end - time.time()))

        # -- 6. post-refine violation repair ---------------------------
        # Refinement can drift tagged blocks off the frame edges, break
        # cluster contact, and evictions can reshape MIB members.  The
        # violations enter the cost through exp(2*V/n_soft), so a repaired
        # layout wins even at a small HPWL cost — but keep it only when
        # the violation count actually drops.
        if _any:
            # value-per-second ordering inside the reserve: the seats cost
            # ~1 ms and reliably drop V by 1-3 (each keeps its result only
            # when the violation count strictly drops), while one repair
            # attempt below costs a build plus a legalization and often
            # fails.  Banking the cheap win first means a failed repair can
            # never cost it, and a V driven to 0 skips the repair outright.
            out = _cluster_seat(opt, _edge_seat(opt, out),
                                deadline=t_hard - _tg(0.05, 0.02))

        # The tail gates below are absolute second offsets, i.e. they assume
        # a reserve of several seconds.  At `res = 0.3 * slice` a 1.7 s
        # worker deadline reserves 0.51 s, so `t_hard - 0.9` is already in
        # the past before the reserve starts and every repair/compaction
        # stage self-disables -- the reserve is carved and then thrown away.
        # `_tg` caps each constant by a share of the actual reserve.
        if time.time() < t_hard - _tg(0.4, 0.25):
            V0 = full_violations(opt, out)
            if V0 > 0:
                Q = np.array(out, dtype=np.float64, copy=True)
                for idxs in opt.mib_groups.values():
                    if len(idxs) < 2:
                        continue
                    ref = next((i for i in idxs
                                if opt.fixed[i] or opt.preplaced[i]), idxs[0])
                    w, h = float(Q[ref, 2]), float(Q[ref, 3])
                    for i in idxs:
                        if opt.kind[i] != 2 and not (opt.fixed[i]
                                                     or opt.preplaced[i]):
                            cx = Q[i, 0] + 0.5 * Q[i, 2]
                            cy = Q[i, 1] + 0.5 * Q[i, 3]
                            Q[i] = (cx - 0.5 * w, cy - 0.5 * h, w, h)
                def _attempt(clamp):
                    r3 = _Refiner(opt, Q, seed + 2)
                    r3._pred_c = pred_c
                    r3._anchor_frame_to_tags()
                    if clamp:
                        # clamp the frame straight onto the tag locks and
                        # individually evict the blocks sticking out past
                        # them — the evaluator only cares about layout
                        # extremes, so this satisfies a tagged-preplaced
                        # edge without a (usually jammed) global squeeze
                        if r3.lock_xmin is not None:
                            r3.xmin = r3.lock_xmin
                        if r3.lock_xmax is not None:
                            r3.xmax = r3.lock_xmax
                        if r3.lock_ymin is not None:
                            r3.ymin = r3.lock_ymin
                        if r3.lock_ymax is not None:
                            r3.ymax = r3.lock_ymax
                        r3._pull_inside_frame()
                    r3._seed_tags()
                    if not r3.legalize_soft(
                            10, deadline=min(t_hard, time.time()
                                             + _tg(1.6, 0.5))):
                        return None
                    r3._assemble_clusters(saved_cg)
                    if r3._has_overlap():
                        return None
                    return r3.P

                best_V = V0
                best_hp = opt._hpwl(out)
                for clamp in (True, False):
                    if time.time() >= t_hard - _tg(0.3, 0.2):
                        break
                    res = _attempt(clamp)
                    if res is None:
                        if _DEBUG:
                            print(f"[repair] clamp={clamp} failed",
                                  flush=True)
                        continue
                    V1 = full_violations(opt, res)
                    hp1 = opt._hpwl(res)
                    if _DEBUG:
                        print(f"[repair] clamp={clamp} V {V0}->{V1} "
                              f"hp {best_hp:.1f}->{hp1:.1f}", flush=True)
                    if V1 < best_V or (V1 == best_V and hp1 < best_hp):
                        best_V = V1
                        best_hp = hp1
                        out = res.copy()

        out = _cluster_seat(opt, _edge_seat(opt, out),
                            deadline=t_hard - _tg(0.9, 0.1))
        if _snaps is not None:
            _snaps.append(("repair", np.asarray(out, dtype=np.float64).copy()))

        # -- 6.5 discrete area compaction toward the tag locks ---------
        # The residual failure shape the seats cannot fix: a whole shelf
        # sits flush past a tagged-preplaced edge while the matching free
        # area lies scattered in slivers.  Success here also deflates the
        # bbox below the recompression trigger, saving that whole rerun.
        if time.time() < t_hard - _tg(0.9, 0.35):
            out = _lock_compact(opt, out,
                                min(t_hard - _tg(0.5, 0.15),
                                    time.time() + _tg(2.2, 0.5)),
                                seed=seed, pred=P0)

        if _snaps is not None:
            _snaps.append(("compact", np.asarray(out, dtype=np.float64).copy()))

        # -- 7. area recompression retry --------------------------------
        # A candidate that came through a loose rung carries an inflated
        # bbox that squeeze cannot recover (pinned rocks jam the cascade).
        # But the LEGAL layout itself is a far better "prediction" than
        # the raw model output — scale it toward the origin so its bbox
        # matches the area reference and push it through the pipeline
        # again with the tight rungs; keep whichever scores better.
        pos0 = np.asarray(out, dtype=np.float64)
        bbox = float(((pos0[:, 0] + pos0[:, 2]).max() - pos0[:, 0].min())
                     * ((pos0[:, 1] + pos0[:, 3]).max() - pos0[:, 1].min()))
        # NOTE: the 2.5 s gate is deliberately NOT relaxed under ANYTIME.
        # Step 7 reruns the whole pipeline (3+ `_Refiner` builds), so a
        # reserve-sized share of a 1.7 s worker deadline cannot fund it; the
        # absolute gate self-disables at exactly the tiers where it would
        # only steal the repair/compaction stages' reserve.
        if (_depth == 0 and time.time() < t_hard - 2.5
                and bbox > 1.08 * opt.area_ref):
            sc = max(0.88, math.sqrt(1.02 * opt.area_ref / bbox))
            Q2 = pos0.copy()
            x0f = float(Q2[:, 0].min())
            y0f = float(Q2[:, 1].min())
            Q2[:, 0] = x0f + (Q2[:, 0] - x0f) * sc
            Q2[:, 1] = y0f + (Q2[:, 1] - y0f) * sc
            for i in range(opt.n):
                if opt.kind[i] == 2:
                    Q2[i] = pos0[i]
            out2 = refine_prediction(opt, Q2, t_hard - 0.05,
                                     seed=seed + 5, _depth=1)
            if out2 is not None:
                pos2 = np.asarray(out2, dtype=np.float64)
                hp0 = opt._hpwl(pos0)
                hp2 = opt._hpwl(pos2)
                hp_ref = max(min(hp0, hp2), 1e-9)
                den = max(getattr(opt, "n_soft_den", 1), 1)

                def _proxy(pos, hp):
                    ar = float(((pos[:, 0] + pos[:, 2]).max()
                                - pos[:, 0].min())
                               * ((pos[:, 1] + pos[:, 3]).max()
                                  - pos[:, 1].min()))
                    V = full_violations(opt, pos)
                    return (1.0 + 0.5 * ((hp - hp_ref) / hp_ref
                                         + max(0.0, ar / opt.area_ref - 1.0))) \
                        * math.exp(2.0 * V / den)

                if _proxy(pos2, hp2) < _proxy(pos0, hp0):
                    if _DEBUG:
                        print(f"[recompress] kept sc={sc:.3f}", flush=True)
                    out = out2

        if time.time() < t_hard - _tg(0.6, 0.25):
            out = _lock_compact(opt, out,
                                min(t_hard - _tg(0.2, 0.08),
                                    time.time() + _tg(2.2, 0.5)),
                                seed=seed + 7, pred=P0)
        out = _cluster_seat(opt, _edge_seat(opt, out),
                            deadline=t_hard - _tg(0.05, 0.02))
        if _DEBUG:
            print(f"[refine_prediction] n={opt.n} hp={opt._hpwl(out):.1f} "
                  f"V={opt._violations(out)}", flush=True)
        if _snaps is not None:
            out = _guard_pick(opt, out, _snaps, pred)
        elif _LEGAL_ADMIT and _depth == 0:
            # rung (-1), second arm: the guard's INPUT comparison without its
            # n >= 95 restriction and without any stage snapshots.  The
            # legality re-check runs FIRST, so on the production path (where
            # the raw prediction overlaps) this arm costs one O(n^2) overlap
            # matrix and never pays for a cost evaluation at all.
            if _admit_legal_input(opt, pred, "cmp") is not None:
                out = _guard_pick(opt, out, (), pred)
        return out
    except Exception:
        if _DEBUG:
            import traceback
            traceback.print_exc()
        return None


def refine_positions(opt, pos: np.ndarray, deadline: float,
                     seed: int = 0) -> Optional[np.ndarray]:
    """Refine a legal layout; returns the improved positions or None."""
    try:
        if opt.n < 2 or time.time() >= deadline:
            return None
        t0 = time.time()
        r = _Refiner(opt, pos, seed)
        hp_in = r.hp0
        out = r.run(deadline - 0.02)
        if _DEBUG:
            hp_out = opt._hpwl(out)
            print(f"[refiner] n={opt.n} groups={len(r.groups)} "
                  f"hp {hp_in:.1f} -> {hp_out:.1f} "
                  f"({hp_out / hp_in:.3f}) V {opt._violations(pos)} -> "
                  f"{opt._violations(out)} in {time.time() - t0:.2f}s", flush=True)
        return out
    except Exception:
        if _DEBUG:
            import traceback
            traceback.print_exc()
        return None
