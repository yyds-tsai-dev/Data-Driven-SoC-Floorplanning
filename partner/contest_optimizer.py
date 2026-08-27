#!/usr/bin/env python3
"""Contest optimizer: diffusion-seeded column-slicing floorplanner.

Pipeline per test case:
  1. Deterministic heuristic seed layout (pin/graph-weighted centroids).
  2. Optional graph-conditioned diffusion refinement of the seed (the seed
     only provides relative-position hints; a few DDIM steps suffice).
  3. column_sa_legalizer.legalize_rectangles: column-slicing layout with
     hard-constraint guarantees plus a time-budgeted simulated-annealing
     search that minimizes HPWL / bbox area / soft violations.

The per-case time budget grows exponentially with block count (the contest
total score weights cases by exp(n/12), so large cases deserve nearly all
of the runtime), and averages roughly 5 s over the 100 validation cases.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import os

from candidate_supply import CandidateBatch, allocate_quotas, rank_predictions
from iccad2026_evaluate import FloorplanOptimizer
from diffusion_data import build_condition, fp_sol_to_z0, layout_scale, z_to_rectangles
from diffusion_model import DiffusionSchedule, GraphDiffusionDenoiser, ModelConfig, ddim_refine
from column_sa_legalizer import (ORACLE_PRED_ON, _ColumnOptimizer,
                              _b2b_smooth_np,
                              _ensure_no_overlap, _parse_constraints,
                              _pin_centroids_np, _target, fast_setup_on,
                              init_worker_pool, legalize_rectangles,
                              oracle_pred_override, rectangles_from_z)
from layout_refiner import (edge_seat_v2_on, full_violations,
                            refine_prediction, wall_repair_on)

# The retrieval channel (retrieval_*) is opt-in: it only runs when both
# PARTNER_RETRIEVAL_INDEX and PARTNER_RETRIEVAL_SLOTS are set.  Its modules are
# imported lazily inside _sample_retrieval_preds so a deployment that ships
# only the active channels (Direct + flow + column) needs no retrieval sources.

def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# Default keeps the historical half-core cap; PARTNER_POOL overrides.  On a
# 48-core box the eval leaves ~22 cores idle per case — a wider pool is
# more independent restart/refine draws per case at the SAME wall-clock
# (the same mechanism the doubled-budget oracle measured, minus the time).
N_RESTART_WORKERS = _env_int(
    "PARTNER_POOL", max(2, min(24, (os.cpu_count() or 4) // 2)))

Rect = Tuple[float, float, float, float]

DEFAULT_CHECKPOINT = (
    Path(__file__).parent / "checkpoints" / "diffusion_stable_xywh_order_2day" / "step_00080000.pt"
)
DIRECT_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "direct_v2"
# First R4 is intentionally fixed-capacity: retrieval may replace, but never
# add to, more than two Direct refinement slots.
FIRST_R4_RETRIEVAL_SLOTS = 2

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _print_dag_bridge_diag(scorer, before_array: np.ndarray,
                           after_array: np.ndarray, elapsed_ms: float,
                           committed: bool) -> None:
    """Emit observational DAG-bridge accounting without affecting results."""
    try:
        from violation_killer import _bbox_area, _grouping_count, _violations_exact

        grouping0 = _grouping_count(scorer, before_array)
        grouping1 = _grouping_count(scorer, after_array)
        violations0 = _violations_exact(scorer, before_array)
        violations1 = _violations_exact(scorer, after_array)
        hpwl0 = float(scorer._hpwl(before_array))
        hpwl1 = float(scorer._hpwl(after_array))
        bbox0 = _bbox_area(before_array)
        bbox1 = _bbox_area(after_array)
        print(f"[gdag] n={len(before_array)} ms={elapsed_ms:.3f} "
              f"grouping={grouping0}->{grouping1} V={violations0}->{violations1} "
              f"hpwl={hpwl0:.6f}->{hpwl1:.6f} "
              f"bbox={bbox0:.6f}->{bbox1:.6f} "
              f"committed={int(bool(committed))}",
              file=sys.stderr, flush=True)
    except Exception:
        return


# =============================================================================
# PARTNER_VAUDIT_JSONL=<path> (default unset = off): per-case violation
# audit, one JSON line per case, buffered in memory and flushed once at
# interpreter exit (atexit) -- same discipline as column_sa_legalizer's
# _PSEL_DUMP_PATH: no I/O inside the timed solve().  Off: the env var is
# never read outside `solve()`'s own guard, no import, no buffer growth.
# =============================================================================
_VAUDIT_BUF: List[dict] = []
_VAUDIT_HOOKED = False


def _vaudit_flush() -> None:
    """Append the buffered records to PARTNER_VAUDIT_JSONL and clear them.

    Exposed as a module function (not inlined in the atexit lambda) so tests
    can call it directly instead of relying on interpreter exit."""
    global _VAUDIT_BUF
    path = os.environ.get("PARTNER_VAUDIT_JSONL", "").strip()
    if not path or not _VAUDIT_BUF:
        return
    import json
    with open(path, "a") as f:
        for rec in _VAUDIT_BUF:
            f.write(json.dumps(rec) + "\n")
    _VAUDIT_BUF = []


def _vaudit_record(rec: dict) -> None:
    global _VAUDIT_HOOKED
    if not _VAUDIT_HOOKED:
        import atexit
        atexit.register(_vaudit_flush)
        _VAUDIT_HOOKED = True
    _VAUDIT_BUF.append(rec)


# Historical defaults; env-overridable so experiments can rescale the
# per-case budget without editing code (defaults reproduce old behavior).
BUDGET_SCALE = _env_float("PARTNER_BUDGET_SCALE", 0.06)
BUDGET_TAU = _env_float("PARTNER_BUDGET_TAU", 20.0)
BUDGET_MIN = _env_float("PARTNER_BUDGET_MIN", 0.8)
BUDGET_MAX = _env_float("PARTNER_BUDGET_MAX", 24.0)
# Direct/flow candidate must beat the column champion's proxy score by this
# factor to be selected (0.985 = historical 1.5% dead zone).  PARTNER_PICK_MARGIN
# makes it tunable; 1.0 disables the dead zone.
PICK_MARGIN = _env_float("PARTNER_PICK_MARGIN", 0.985)


def _parse_budget_table() -> Optional[List[float]]:
    """PARTNER_BUDGET_TABLE: comma-separated per-n budget seconds for
    n = 21..120 (100 entries).  Keys the budget on block_count -- a reusable
    instance statistic -- with values derived offline from the published
    per-testcase median runtimes.  Unset or malformed -> None (exponential
    curve below is used, byte-identical to historical behavior)."""
    raw = os.environ.get("PARTNER_BUDGET_TABLE", "").strip()
    if not raw:
        return None
    try:
        vals = [float(v) for v in raw.split(",") if v.strip()]
    except ValueError:
        return None
    if len(vals) != 100 or any((not math.isfinite(v)) or v <= 0 for v in vals):
        return None
    return vals


_BUDGET_TABLE = _parse_budget_table()


def _time_budget(block_count: int) -> float:
    """Exponential per-case budget: ~0.8 s for the smallest cases up to
    ~24 s for n=120; averages about 5 s over the validation set.
    PARTNER_BUDGET_TABLE (per-n seconds, n=21..120) overrides the curve."""
    if _BUDGET_TABLE is not None:
        return _BUDGET_TABLE[max(0, min(99, int(block_count) - 21))]
    b = BUDGET_SCALE * math.exp(block_count / BUDGET_TAU)
    return max(BUDGET_MIN, min(BUDGET_MAX, b))


def _area_ok(rects, area_targets, n: int, tol: float = 0.01) -> bool:
    """Evaluator-form per-block area check (|w*h - a| / a <= tol)."""
    try:
        at = np.asarray(area_targets, dtype=np.float64).reshape(-1)[:n]
        P = np.asarray([list(map(float, r)) for r in rects],
                       dtype=np.float64)
        if P.shape[0] != n or at.shape[0] != n:
            return False
        area = P[:, 2] * P[:, 3]
        return bool(np.all(np.abs(area - at) <= tol * np.abs(at)))
    except Exception:
        return False


def _final_area_guard(result, fallbacks, area_targets, n: int):
    """Return `result` unless a block violates the evaluator's 1% area
    tolerance; then the first fallback layout that passes, else `result`.

    PARTNER_FINAL_AREA_GUARD=0 disables the check (shipped behaviour before
    2026-08-26).  On the common path this is a read-only O(n) check and the
    SAME object is returned."""
    if os.environ.get("PARTNER_FINAL_AREA_GUARD", "1") in ("0", "false",
                                                           "False", "off"):
        return result
    if _area_ok(result, area_targets, n):
        return result
    for fb in fallbacks:
        if fb is not None and fb is not result and _area_ok(fb, area_targets, n):
            if os.environ.get("PARTNER_FINAL_AREA_GUARD_DEBUG"):
                print(f"[area_guard] n={n} final layout failed the 1% area "
                      f"check; shipping fallback", file=sys.stderr, flush=True)
            return [tuple(map(float, r)) for r in fb]
    return result


def _cond_p2b(p2b):
    """PARTNER_COND_P2B (default unset = shipped): what the MODEL conditioning
    sees as pin-to-block connectivity.  `off` invalidates every p2b row for
    the conditioning only (pin index -1 -> `fast_condition` drops it: zero
    degree/weight/pin-centroid features), so the Flow/Direct prior ignores
    pins while the legalizer, refiner and proxy still optimise the true p2b
    HPWL.  Insurance against a pin-distribution shift in the hidden set
    (alpha_1-type: P2B distance rank 0.37 vs 0.04 on official/v3, see
    docs/experiments/2026-08-21-post-beta-p0-execution.md §17c), where the
    models fall back to column layouts.  Off path returns the same tensor."""
    mode = os.environ.get("PARTNER_COND_P2B", "")
    if mode != "off" or p2b is None or p2b.numel() == 0:
        return p2b
    q = p2b.clone()
    q[..., 0] = -1.0
    return q


def _direct_rung0_projection(block_count: int, remaining: float):
    """Project what a RESERVED refine worker will actually get, and what
    `refine_prediction`'s rung 0 (fixed-frame legalization) costs.

    Every term is either read off the code path it models or a per-machine
    constant times a *reusable instance statistic* (block count and the
    case's own remaining budget) -- never a case id.

      1. `column_sa_legalizer._parallel_solve` hands the pool
         `worker_deadline = deadline_A - min(0.30, 0.15 * spanA)`; phase B
         is inactive at these budgets, so `deadline_A == deadline`.
      2. The reserved workers sit idle until the wave-1 sampler returns:
         a batched forward, i.e. roughly linear in `n`.  The same term
         absorbs the legalizer's own pre-dispatch setup (the gate is read a
         little before `t_pb`), which is why it is calibrated, not derived.
      3. `refine_prediction` carves its violation-repair reserve
         (`min(3.5, 0.3*slice)`, plus 0.35*slice for the recompression retry
         when the slice exceeds 8 s) off the top before the ladder starts.
      4. Rung 0 costs one O(n^2) `_Refiner` build plus the fixed-frame
         legalization it feeds; the ladder's own ANYTIME notes size a rung
         at 4-7 builds, so the whole rung is modelled as c * (n/100)^2.

    Returns `(span, need)`: the projected ladder span and the rung-0 cost.

    Defaults calibrated on the shipped 0.3 s operating point with
    `PARTNER_SEAT_DEBUG=1` (`[seat]` / `[rp0]` instruments, 2026-08-10):
    warm wave-1 sampler latency 0.145 s @ n=100 -> 0.158 s @ n=120 (hence
    the linear TS term), `_Refiner` build 3-11 ms, and a rung-0 completion
    boundary of 6/6 workers failing at n=100, 4/6 at n=101, 0/6 at n>=102.
    BOTH constants are machine- and sampler-relative (they move with
    `PARTNER_DDIM_STEPS` / `PARTNER_FLOW_*` and with the accelerator), which
    is why they are env-overridable; the shipped gate they refine
    (`PARTNER_DIRECT_MIN`) is an absolute-seconds constant with the same
    exposure.
    """
    n = max(1, int(block_count))
    wd = remaining - min(0.30, 0.15 * max(0.0, remaining))
    slice_ = wd - _env_float("PARTNER_DIRECT_SEAT_TS", 0.148) * (n / 100.0)
    if slice_ <= 0.0:
        return 0.0, float("inf")
    span = slice_ - min(3.5, 0.3 * slice_)
    if slice_ > 8.0:
        span -= 0.35 * slice_
    need = _env_float("PARTNER_DIRECT_SEAT_R0", 0.125) * (n / 100.0) ** 2
    return span, need


def _direct_seat_ok(block_count: int, remaining: float) -> bool:
    """Should this case reserve pool seats for the direct channel?

    Shipped gate: `remaining > PARTNER_DIRECT_MIN`.  That tests whether
    there is time to SAMPLE, not whether there is time to REFINE -- but the
    seats are taken from the column-restart portfolio either way
    (`PARTNER_NREF=6` of a 24-worker pool = 25% of restart breadth at the
    0.3 s operating point).  In the band where the budget covers sampling
    but not rung 0, `refine_prediction` returns None for every reserved
    worker and the case pays a quarter of its restarts for nothing.

    `PARTNER_DIRECT_SEAT_FIX=1` (default off) adds the missing condition.
    It can only ever CLOSE the channel (`DIRECT_MIN` stays as a floor), so
    it adds no runtime and cannot open the channel anywhere it is shut
    today.  Off -> the shipped expression, bit for bit.
    """
    if not (remaining > _env_float("PARTNER_DIRECT_MIN", 4.5)):
        return False
    fix_on = os.environ.get("PARTNER_DIRECT_SEAT_FIX", "0") in (
        "1", "true", "True", "on", "ON")
    dbg = bool(os.environ.get("PARTNER_SEAT_DEBUG"))
    if not (fix_on or dbg):
        return True
    span, need = _direct_rung0_projection(block_count, remaining)
    ok = span >= need
    if dbg:
        print(f"[seatgate] n={block_count} rem={remaining:.4f} "
              f"span={span:.4f} need={need:.4f} ok={int(ok)} "
              f"fix={int(fix_on)}", file=sys.stderr, flush=True)
    return ok if fix_on else True


def _select_ranked_source_quota(
    predictions: List[np.ndarray], sources: List[str], order: List[int],
    total: int, retrieval_quota: int,
) -> List[np.ndarray]:
    """Keep ranked candidates under fixed Direct/retrieval capacity quotas."""
    total = max(0, int(total))
    retrieval_quota = min(total, max(0, int(retrieval_quota)))
    direct_quota = total - retrieval_quota
    selected: List[np.ndarray] = []
    selected_indexes: set[int] = set()
    direct_count = 0
    retrieval_count = 0
    for index in order:
        source = sources[index]
        if source == "retrieval":
            if retrieval_count >= retrieval_quota:
                continue
            retrieval_count += 1
        elif source == "direct":
            if direct_count >= direct_quota:
                continue
            direct_count += 1
        else:
            continue
        selected.append(predictions[index])
        selected_indexes.add(index)

    # A missing retrieval source must not leave direct refinement capacity idle.
    if len(selected) < total and retrieval_count < retrieval_quota:
        for index in order:
            if sources[index] == "direct" and index not in selected_indexes:
                selected.append(predictions[index])
                selected_indexes.add(index)
                if len(selected) == total:
                    break
    return selected


class MyOptimizer(FloorplanOptimizer):
    def __init__(
        self,
        verbose: bool = False,
        checkpoint_path: Optional[str] = None,
        steps: int = 4,
        refine_start_t: int = 250,
        device: Optional[str] = None,
    ):
        super().__init__(verbose=verbose)
        self.steps = steps
        self.refine_start_t = refine_start_t
        self.diffusion_seed = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[GraphDiffusionDenoiser] = None
        self.schedule: Optional[DiffusionSchedule] = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self._load_model()
        self.direct_model = None
        self._load_direct_model()
        self.flow_model = None
        self._load_flow_model()
        self.retrieval_index = None
        self.retrieval_slots = 0
        # PARTNER_VAUDIT_JSONL bookkeeping (see module-level _vaudit_* below)
        self._last_pick_channel = "column"
        self._vaudit_seq = 0
        configured_max_cost = _env_float("PARTNER_RETRIEVAL_MAX_COST", 2.0)
        self.retrieval_max_cost = configured_max_cost if math.isfinite(configured_max_cost) else 2.0
        retrieval_path = os.environ.get("PARTNER_RETRIEVAL_INDEX", "").strip()
        requested_slots = _env_int("PARTNER_RETRIEVAL_SLOTS", 0)
        if retrieval_path and requested_slots > 0:
            try:
                from retrieval_index import RetrievalIndex
                self.retrieval_index = RetrievalIndex.load(Path(retrieval_path))
                self.retrieval_slots = min(requested_slots, FIRST_R4_RETRIEVAL_SLOTS)
            except Exception as exc:
                self.retrieval_index = None
                self.retrieval_slots = 0
                if self.verbose:
                    print(f"retrieval index unavailable: {exc}")
        # spawn the parallel-restart pool now so worker startup cost is not
        # charged to any test case
        init_worker_pool(N_RESTART_WORKERS)
        self._warm_tag_compress_dependencies()
        self._warm_direct_sampler()
        self._warm_flow_sampler()

    def _warm_flow_sampler(self) -> None:
        """Optionally pay Flow's first-call CUDA cost outside ``solve()``.

        Opt-in via PARTNER_FLOW_WARM=1 (default off; keeps other wrappers'
        historical constructor behavior unchanged).  Ported verbatim from the
        handover build."""
        self.flow_warm_succeeded = False
        self.flow_warm_latency = None
        if os.environ.get("PARTNER_FLOW_WARM", "0") not in (
                "1", "true", "True", "on", "ON"):
            return
        if self.flow_model is None:
            return
        flow_slots = _env_int("PARTNER_FLOW_SLOTS", 0)
        if flow_slots <= 0:
            return
        try:
            n = _env_int("PARTNER_FLOW_WARM_N", 100)
            g = torch.Generator().manual_seed(23456)
            at = torch.rand(n, generator=g) * 40.0 + 10.0
            cons = torch.zeros(n, 5)
            tpos = torch.full((n, 4), -1.0)
            # Exercise the same anchored antithetic path as real FloorSet
            # cases.  Passing a known-mask with no known nodes makes
            # sample_flow_diff reject the warm draw because no known noise is
            # supplied, which used to turn this warm-up into a silent no-op.
            cons[0, 1] = 1.0
            anchor_side = float(torch.sqrt(at[0]))
            tpos[0] = torch.tensor([0.0, 0.0, anchor_side, anchor_side])
            e = torch.randint(0, n, (3 * n, 2), generator=g).float()
            b2b = torch.cat([e, torch.ones(3 * n, 1)], dim=1)
            npin = 16
            pe = torch.stack([
                torch.randint(0, npin, (n,), generator=g).float(),
                torch.arange(n, dtype=torch.float32),
            ], dim=1)
            p2b = torch.cat([pe, torch.ones(n, 1)], dim=1)
            side = float(torch.sqrt(at.sum() / 0.96))
            pins = torch.rand(npin, 2, generator=g) * side
            samples = min(flow_slots, _env_int("PARTNER_NREF", 6) or 6)
            produced = self._sample_flow_preds(
                n, at, cons, tpos, b2b, p2b, pins, samples,
            )
            if len(produced) < samples:
                raise RuntimeError(
                    f"Flow warm-up returned {len(produced)}/{samples} samples"
                )
            self.flow_warm_succeeded = True
            # Deployment self-check + adaptive seat gate (2026-08-27).  The
            # official beta per-case results (docs/official/beta_test/
            # cadc1013.tar.gz) show the model channel never delivered on the
            # contest box: from the first n whose budget opens the arm
            # (n=99) every case overran its budget by ~1.0 s and shipped the
            # column fallback (tail hpwl_gap 0.14-0.69), i.e. the Flow
            # sampler took ~1 s there (CPU torch / missing deps) against the
            # 0.148 s @ n=100 that `PARTNER_DIRECT_SEAT_TS` assumes.  A
            # CPU-only local run reproduces it (official 1.298 vs 1.09).
            # Measure the WARM latency with a second draw (the first call
            # carries the CUDA/JIT cold start) and, when it exceeds the
            # assumed constant, raise the gate's sampler term so arms that
            # cannot finish are not launched (no quality gain, but the
            # ~1 s/case runtime blow-up and its rt_adj penalty disappear).
            # PARTNER_SEAT_TS_ADAPT=0 disables the adaptation; the
            # self-check line is always printed under PARTNER_FLOW_WARM=1.
            _t0 = time.time()
            self._sample_flow_preds(n, at, cons, tpos, b2b, p2b, pins, samples)
            warm_lat = time.time() - _t0
            self.flow_warm_latency = warm_lat
            ts_assumed = _env_float("PARTNER_DIRECT_SEAT_TS", 0.148)
            ts_measured = warm_lat / max(n / 100.0, 1e-6)
            adapted = False
            # CPU-speed self-calibration for the gate's refine term.  The
            # rung-0 cost constant (PARTNER_DIRECT_SEAT_R0, 0.125 s @ n=100)
            # was measured on this dev box (Xeon Silver 4510, 4.1 GHz turbo);
            # the contest box is an Icelake 48-core.  A fixed single-thread
            # numpy micro-benchmark (reference 1.0 = dev box, quiet) scales
            # R0 up when the host is slower, so seats are not reserved for
            # refines that cannot finish there.  PARTNER_SEAT_R0_ADAPT=0
            # disables; only ratios > 1.25 change anything.
            cpu_ratio = float("nan")
            r0_adapted = False
            try:
                import numpy as _np
                # element-wise + sort only: no BLAS, so the number reflects
                # ONE core's speed, not the thread pool
                _rng = _np.random.default_rng(0)
                _x = _rng.random(200_000)
                _t1 = time.time()
                for _ in range(30):
                    _x = _np.sin(_x) * 1.1 + _np.sqrt(_np.abs(_x))
                    _np.sort(_x[:60_000])
                cpu_ms = (time.time() - _t1) * 1000.0
                cpu_ratio = cpu_ms / _env_float("PARTNER_CPU_CALIB_REF_MS", 95.0)
                # OPT-IN (default off, 2026-08-27): on a loaded dev box the
                # benchmark read 1.91x and the raised R0 closed the model
                # arms (official 1.10 -> 1.13); on a 1.3-1.45x slower CPU the
                # arms still pay for themselves (emulation: 1.137 with arms
                # vs 1.155 without), so closing them is the wrong trade.
                # The ratio is printed for diagnostics only.
                if (os.environ.get("PARTNER_SEAT_R0_ADAPT", "0") in
                        ("1", "true", "True", "on") and cpu_ratio > 1.25):
                    r0 = _env_float("PARTNER_DIRECT_SEAT_R0", 0.125) * cpu_ratio
                    os.environ["PARTNER_DIRECT_SEAT_R0"] = f"{r0:.4f}"
                    r0_adapted = True
            except Exception:
                pass
            if (os.environ.get("PARTNER_SEAT_TS_ADAPT", "1") not in
                    ("0", "false", "False", "off")
                    and ts_measured > ts_assumed * 1.25):
                os.environ["PARTNER_DIRECT_SEAT_TS"] = f"{ts_measured:.4f}"
                adapted = True
            print(f"[selfcheck] cuda_available={torch.cuda.is_available()} "
                  f"device={self.device} flow_warm_n={n} "
                  f"flow_warm_latency={warm_lat:.3f}s "
                  f"seat_ts_assumed={ts_assumed:.3f} "
                  f"seat_ts={'adapted->' if adapted else 'kept '}"
                  f"{ts_measured if adapted else ts_assumed:.3f} "
                  f"cpu_ratio={cpu_ratio:.2f} "
                  f"seat_r0={'adapted->' + os.environ.get('PARTNER_DIRECT_SEAT_R0', '?') if r0_adapted else 'kept'}",
                  file=sys.stderr, flush=True)
            if self.verbose:
                print("flow sampler warmed")
        except Exception as exc:
            if self.verbose:
                print(f"flow sampler warm-up skipped: {exc}")

    def _warm_tag_compress_dependencies(self) -> None:
        """Pay the opt-in tag-compression import cost outside `solve()`.

        The evaluator constructs the optimizer before starting its per-case
        timer.  Keeping this behind the existing default-off experiment flag
        preserves the production import path exactly.
        """
        if not (os.environ.get("PARTNER_TAG_COMPRESS")
                or os.environ.get("PARTNER_GROUP_BRIDGE")
                or os.environ.get("PARTNER_GROUP_DAG_BRIDGE") == "1"):
            return
        try:
            from tag_compress import warm_dependencies

            warm_dependencies()
        except Exception as exc:
            if self.verbose:
                print(f"tag-compress warm-up skipped: {exc}")

    def _warm_direct_sampler(self) -> None:
        """PARTNER_DIRECT_WARM=1 (default off): pay the sampler's first-call
        cost HERE, in the untimed constructor, instead of inside the first
        case that opens the direct channel.

        Measured at the 0.3 s operating point (`PARTNER_SEAT_DEBUG`,
        2026-08-10): wave-1 latency is 0.520 s on the first gated case and
        0.145-0.158 s on every one after it.  That first case is n=99, whose
        whole budget is 0.325 s -- the cold draw alone overruns the deadline,
        so all `PARTNER_NREF` reserved workers start past `worker_deadline`
        and return nothing, AND the case's runtime blows out to 0.89 s
        (2.7x budget) against a per-case official RuntimeFactor.  Same class
        of defect as the numba cold-JIT cliff caught by the 2026-08-06
        repack verification, and the same fix: move it into module/ctor
        time.  Fully contained -- any failure leaves the shipped lazy path.

        MEASURED VERDICT (2026-08-10, 4 paired reps on top of the seat fix):
        HOLD, do not ship.  The warm draw does remove the cold latency
        (0.520 s -> 0.151 s at the first gated case), but the case is
        deadline-bounded and the sampler overlaps the already-dispatched
        column restarts, so a cold draw mostly burns time the parent was
        going to spend blocked anyway: warming the first gated case made it
        SLOWER (0.758 s vs 0.701 s -- the reserved workers now have a live
        window and use it), and the no-runtime total moved +0.0046 +-
        0.0027 the wrong way.  `PARTNER_DIRECT_SEAT_FIX` collects the same
        first-case runtime (0.902 s -> 0.668 s) by not opening the channel
        at all.  Kept as a documented, default-off probe.
        """
        if os.environ.get("PARTNER_DIRECT_WARM", "0") not in ("1", "true",
                                                              "True", "on",
                                                              "ON"):
            return
        if self.direct_model is None:
            return
        try:
            # synthetic instance shaped like the band that opens the gate
            # (block count only -- no validation data is read here)
            n = _env_int("PARTNER_DIRECT_WARM_N", 100)
            g = torch.Generator().manual_seed(12345)
            at = torch.rand(n, generator=g) * 40.0 + 10.0
            cons = torch.zeros(n, 5)
            tpos = torch.full((n, 4), -1.0)
            e = torch.randint(0, n, (3 * n, 2), generator=g).float()
            b2b = torch.cat([e, torch.ones(3 * n, 1)], dim=1)
            npin = 16
            pe = torch.stack([
                torch.randint(0, npin, (n,), generator=g).float(),
                torch.arange(n, dtype=torch.float32)], dim=1)
            p2b = torch.cat([pe, torch.ones(n, 1)], dim=1)
            side = float(torch.sqrt(at.sum() / 0.96))
            pins = torch.rand(npin, 2, generator=g) * side
            self._sample_direct_preds(
                n, at, cons, tpos, b2b, p2b, pins,
                _env_int("PARTNER_NREF", 6) or 6, oversample=False)
            if self.verbose:
                print("direct sampler warmed")
        except Exception as exc:
            if self.verbose:
                print(f"direct sampler warm-up skipped: {exc}")

    def _load_direct_model(self) -> None:
        """Load the direct-prediction denoiser (EMA weights) if a trained
        checkpoint exists.  Missing/broken checkpoints are silently ignored
        — the column pipeline works standalone."""
        if os.environ.get("DIRECT_OFF"):
            return
        try:
            from direct_diffusion_model import DirectDenoiser, DirectModelConfig
            env_path = os.environ.get("DIRECT_CKPT")
            if env_path:
                path = Path(env_path)
                if not path.exists():
                    return
            else:
                path = DIRECT_CHECKPOINT_DIR / "latest.pt"
                if not path.exists():
                    cands = sorted(DIRECT_CHECKPOINT_DIR.glob("step_*.pt"))
                    if not cands:
                        return
                    path = cands[-1]
            ck = torch.load(path, map_location=self.device, weights_only=False)
            cfg = DirectModelConfig(**{k: v for k, v in ck["model_config"].items()
                                       if k in DirectModelConfig.__dataclass_fields__})
            m = DirectDenoiser(cfg).to(self.device)
            m.load_state_dict(ck["model"])
            if "ema" in ck:
                sd = m.state_dict()
                for k in sd:
                    sd[k].copy_(ck["ema"][k].to(sd[k].dtype))
            m.eval()
            self.direct_model = m
            self.direct_cfg = cfg
            self.direct_schedule = DiffusionSchedule(cfg.timesteps, device=self.device)
            if self.verbose:
                print(f"loaded direct model step {ck.get('step')} from {path}")
        except Exception as exc:
            self.direct_model = None
            if self.verbose:
                print(f"direct model unavailable: {exc}")

    def _load_flow_model(self) -> None:
        """Load the opt-in flow-matching candidate source (off by default:
        requires both FLOW_CKPT and PARTNER_FLOW_SLOTS>0).  Any failure
        (missing file, wrong checkpoint tag, load error) silently disables
        the flow channel — it never harms the Direct/column pipeline."""
        self.flow_model = None
        path = os.environ.get("FLOW_CKPT", "").strip()
        slots = _env_int("PARTNER_FLOW_SLOTS", 0)
        if not path or slots <= 0 or not Path(path).exists():
            return
        try:
            from flow_matching_train import checkpoint_method
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            checkpoint_method(ckpt)
            from direct_diffusion_model import DirectDenoiser, DirectModelConfig
            cfg = DirectModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                                       if k in DirectModelConfig.__dataclass_fields__})
            model = DirectDenoiser(cfg).to(self.device)
            model.load_state_dict(ckpt.get("ema") or ckpt["model"])
            model.eval()
            self.flow_model = model
            self.flow_cfg = cfg
            if self.verbose:
                print(f"loaded flow model step {ckpt.get('step')} from {path}")
        except Exception as exc:
            self.flow_model = None
            if self.verbose:
                print(f"flow model unavailable: {exc}")

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            if self.verbose:
                print(f"checkpoint not found: {self.checkpoint_path}; using heuristic init only")
            return
        for dev in ([self.device, torch.device("cpu")] if self.device.type == "cuda"
                    else [self.device]):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=dev, weights_only=False)
                cfg_dict = ckpt.get("model_config", {})
                cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                                     if k in ModelConfig.__dataclass_fields__})
                model = GraphDiffusionDenoiser(cfg).to(dev)
                model.load_state_dict(ckpt["model"])
                model.eval()
                self.device = dev
                self.model = model
                self.schedule = DiffusionSchedule(cfg.timesteps, device=dev)
                if self.verbose:
                    print(f"loaded diffusion checkpoint on {dev}: {self.checkpoint_path}")
                return
            except Exception as exc:
                if self.verbose:
                    print(f"failed to load checkpoint on {dev}: {exc}")
        self.model = None
        self.schedule = None

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Rect]:
        # self-heal: if a previous case's worker overrun killed the restart
        # pool, rebuild it (no-op when the pool is alive); losing the pool
        # for the rest of the run costs far more than one rebuild
        init_worker_pool(N_RESTART_WORKERS)
        start = time.time()
        budget = _time_budget(block_count)
        deadline = start + budget
        self._first_polish_truncated = False

        # PARTNER_POST_ROUTER: one shared, time-neutral reserve for the final
        # violation-repair + second-polish post-passes.  Ported verbatim from
        # partner/postpass_router.py (see that module's docstring for the
        # measurement this rests on).  Off by default; with the flag unset
        # the module is never imported and `router` stays None, so every
        # downstream branch below is a no-op and the pipeline is unchanged.
        router = None
        if os.environ.get("PARTNER_POST_ROUTER"):
            try:
                from postpass_router import PostPassRouter
                router = PostPassRouter(block_count, budget, start)
                deadline -= router.sa_reserve
            except Exception:
                router = None
        # Time-neutral vkill: the post-pass runs inside the SAME per-case
        # budget by carving a reserve out of the SA/refine deadline (total
        # wall-clock per case is unchanged).  VKILL_CARVE=0 restores the
        # legacy additive behavior; VKILL_OFF disables the pass entirely.
        vk_deadline = None
        vk_reserve = 0.0
        if (os.environ.get("VKILL")
                and not os.environ.get("VKILL_OFF")
                and os.environ.get("VKILL_CARVE", "1") != "0"):
            if block_count >= int(_env_float("VKILL_MIN_N", 60.0)):
                reserve = min(0.25 * budget,
                              _env_float("VKILL_RESERVE_MAX", 5.0))
            else:
                reserve = min(0.1 * budget, 0.3)
            vk_deadline = deadline
            vk_reserve = reserve
            deadline = deadline - reserve

        area_targets = area_targets[:block_count].detach().float().cpu()
        constraints = constraints[:block_count].detach().float().cpu()
        target_positions = (
            target_positions[:block_count].detach().float().cpu()
            if target_positions is not None
            else torch.full((block_count, 4), -1.0)
        )
        b2b = b2b_connectivity.detach().float().cpu()
        p2b = p2b_connectivity.detach().float().cpu()
        pins = pins_pos.detach().float().cpu()

        try:
            seed_rects = _heuristic_init(area_targets, constraints, target_positions, b2b, p2b, pins)

            raw_rects = seed_rects
            if self.model is not None and self.schedule is not None:
                try:
                    raw_rects = self._diffusion_refine(
                        seed_rects, area_targets, b2b, p2b, pins,
                        constraints, target_positions, block_count,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"diffusion refine failed; using heuristic seed: {exc}")

            # direct-prediction candidates: when the restart pool is up,
            # hand a sampler to the legalizer — it reserves pool workers so
            # every prediction gets refined with a full budget in parallel.
            # The GPU sampling happens only after the seed-diffusion work
            # above AND after the column restarts are dispatched (GPU
            # contention here once starved mid-size cases' SA budget and
            # regressed the full validation to 1.262).
            import column_sa_legalizer as _lg
            direct_box: List = []
            th = None
            sample_fn = None
            # PARTNER_DIRECT_MIN (default 4.5 = historical): minimum
            # remaining budget for the direct channel.  The 4.5s gate
            # silently disables the pipeline's strongest channel for every
            # case below n~86 (budget = 0.06*e^(n/20)) — the whole 1.17x
            # mid band is column-only.  Sampling now runs concurrently
            # with the already-dispatched column restarts and small-n
            # batches are fast, so a much lower gate is viable.
            # PARTNER_DIRECT_SEAT_FIX=1 adds the second half of the
            # condition -- enough budget for the reserved workers to finish
            # rung 0, not merely to sample; see `_direct_seat_ok`.
            if self.direct_model is not None and _direct_seat_ok(
                    block_count, deadline - time.time()):
                # `gen_seed` (a generator-seed OFFSET, default 0 = the
                # reviewed baseline) lets the legalizer draw a genuinely
                # different SECOND wave on the otherwise-idle accelerator
                # while phase A runs -- see PARTNER_GPU_ARM in
                # column_sa_legalizer.py.  A caller that omits it gets the
                # historical batch bit for bit.
                if _lg._POOL is not None and _lg._POOL_READY:
                    if self.retrieval_index is not None and self.retrieval_slots > 0:
                        sample_fn = (lambda K, gen_seed=0:
                                     self._sample_portfolio_preds(
                                         block_count, area_targets, constraints,
                                         target_positions, b2b, p2b, pins, K,
                                         oversample=(deadline - time.time()) > _env_float("PARTNER_OVERSAMPLE_MIN_REM", 12.0),  # noqa: E501  (PARTNER_OVERSAMPLE_MIN_REM: remaining-seconds gate for oversampling, default 12.0 = shipped)
                                         gen_seed=gen_seed))
                    else:
                        sample_fn = (lambda K, gen_seed=0:
                                     self._sample_direct_preds(
                                         block_count, area_targets, constraints,
                                         target_positions, b2b, p2b, pins, K,
                                         oversample=(deadline - time.time()) > _env_float("PARTNER_OVERSAMPLE_MIN_REM", 12.0),
                                         gen_seed=gen_seed))
                else:
                    # no pool: fall back to the sliced in-process thread
                    th = threading.Thread(
                        target=self._direct_worker,
                        args=(block_count, area_targets, constraints,
                              target_positions, b2b, p2b, pins,
                              deadline - 0.35, direct_box),
                        daemon=True)
                    th.start()

            t_dispatch = time.time()
            column_out = legalize_rectangles(
                raw_rects, area_targets, constraints, target_positions,
                b2b_connectivity=b2b, p2b_connectivity=p2b, pins_pos=pins,
                deadline=deadline, sample_fn=sample_fn,
            )
            t_legal = time.time()
            if th is not None:
                th.join(timeout=max(0.0, deadline - time.time()) + 0.1)

            # PARTNER_VAUDIT_JSONL: per-case exact-violation audit trail.
            # Off (default, unset env var): the block below is skipped
            # entirely -- no import, no scorer build, no buffering.
            vaudit_path = os.environ.get("PARTNER_VAUDIT_JSONL", "").strip()
            vaudit_scorer = None
            vaudit_v = None
            vaudit_stages: List[Tuple[str, int]] = []
            if vaudit_path:
                from violation_killer import _violations_exact
                vaudit_scorer = (direct_box[0][1] if direct_box else
                                 _ColumnOptimizer(
                                     [tuple(map(float, r)) for r in column_out],
                                     area_targets, constraints, target_positions,
                                     b2b, p2b, pins, time.time() + 1.0, seed=0))

                def vaudit_v(rects, _scorer=vaudit_scorer,
                             _f=_violations_exact) -> int:
                    P = np.asarray([list(r) for r in rects], dtype=np.float64)
                    return int(_f(_scorer, P))

                vaudit_stages.append(("column_pre_seat", vaudit_v(column_out)))

            column_out = self._column_edge_seat(
                column_out, area_targets, constraints, target_positions,
                b2b, p2b, pins, direct_box)
            if vaudit_path:
                vaudit_stages.append(("column_edge_seat", vaudit_v(column_out)))
            out = (self._pick_best(column_out, direct_box)
                   if direct_box else column_out)
            if vaudit_path:
                vaudit_stages.append(("pick_best", vaudit_v(out)))
            t_kill = (vk_deadline if vk_deadline is not None
                      else time.time() + _env_float("VKILL_BUDGET", 6.0))
            # PARTNER_EARLY_EXIT: the vkill reserve was carved as a FIXED
            # share of the budget, but `t_kill` is absolute -- so a legalizer
            # that returned early would silently hand vkill the whole
            # reclaimed tail.  Cap it at the carved share.
            if vk_deadline is not None and _lg.early_exit_on():
                t_kill = min(t_kill, time.time() + vk_reserve)
            result = self._violation_kill(
                out, area_targets, constraints, target_positions,
                b2b, p2b, pins, t_kill)
            _t_pol = time.time()
            result = self._coord_polish(
                result, area_targets, constraints, target_positions,
                b2b, p2b, pins, elapsed=_t_pol - start)
            if router is not None:
                # Second-polish precondition for the router: `polish_layout`
                # is deterministic, so a second call is only worth its time
                # when something modified the layout after this one, or when
                # this one was cut short by its own box and therefore had not
                # converged.
                _pol_box = _env_float(
                    "PARTNER_COORD_POLISH_BUDGET_MS", 600.0) / 1000.0
                self._first_polish_truncated = (
                    (time.time() - _t_pol) >= 0.9 * _pol_box)
                self._post_polish_layout = [tuple(map(float, r)) for r in result]
            if vaudit_path:
                vaudit_stages.append(("coord_polish", vaudit_v(result)))
            result = self._final_seat(
                result, area_targets, constraints, target_positions,
                b2b, p2b, pins, direct_box)
            if vaudit_path:
                vaudit_stages.append(("final_seat", vaudit_v(result)))
            result = self._tag_compress(
                result, area_targets, constraints, target_positions,
                b2b, p2b, pins, direct_box)
            result = self._wall_repair_final(
                result, area_targets, constraints, target_positions,
                b2b, p2b, pins, direct_box)
            if vaudit_path:
                vaudit_stages.append(("tag_compress", vaudit_v(result)))
                self._vaudit_seq += 1
                from violation_killer import (_boundary_violators,
                                              _components, _mib_count)
                P = np.asarray([list(r) for r in result], dtype=np.float64)
                bnd = [(int(i), int(code))
                       for i, code in _boundary_violators(vaudit_scorer, P)]
                grp = []
                for gid, idxs in vaudit_scorer.cluster_groups.items():
                    if len(idxs) < 2:
                        continue
                    g = np.asarray(sorted(int(i) for i in idxs), dtype=np.int64)
                    comps = _components(P, g)
                    if len(comps) > 1:
                        grp.append((int(gid), len(comps)))
                _vaudit_record({
                    "test_or_seq_id": self._vaudit_seq,
                    "n": int(block_count),
                    "channel": self._last_pick_channel,
                    "stage_v": vaudit_stages,
                    "final_report": {
                        "bnd": bnd,
                        "grp": grp,
                        "mib_count": int(_mib_count(vaudit_scorer, P)),
                    },
                })
            if router is not None:
                result = self._routed_quality_pass(
                    result, area_targets, constraints, target_positions,
                    b2b, p2b, pins, router)
            if os.environ.get("PARTNER_EARLY_EXIT_DEBUG"):
                # per-case time anatomy: serial head (heuristic seed + GPU
                # seed-diffusion + parse), the deadline-bounded solve, and
                # the selection/vkill tail.  Only the middle term is what
                # EARLY_EXIT can shorten.
                now = time.time()
                print(f"[ee] n={block_count} budget={budget:.2f} "
                      f"pre={t_dispatch - start:.3f} "
                      f"solve={t_legal - t_dispatch:.3f} "
                      f"post={now - t_legal:.3f} total={now - start:.3f}",
                      file=sys.stderr, flush=True)
            # Last-line hard-legality guard (PARTNER_FINAL_AREA_GUARD, default
            # on; "0" disables): the evaluator treats a single block outside
            # the 1% area tolerance as INFEASIBLE (cost 10).  Observed once in
            # 30,800 case-runs (2026-08-26, shadow v3 tid 85, tail budget x2):
            # an MIB member of a heterogeneous-area group shipped with the
            # group's shared dims (24x13 = 312 vs target 650).  A pure check
            # on the common path; on failure fall back to the post-pick
            # layout, then to the column champion (exact-area by
            # construction).
            return _final_area_guard(result, (out, column_out), area_targets,
                                     block_count)
        except Exception as exc:
            if self.verbose:
                print(f"column optimizer failed; using row fallback: {exc}")
            return _fallback_row(area_targets, constraints, target_positions)

    def _violation_kill(self, out, at, cons, tpos, b2b, p2b, pins,
                        t_kill: float):
        """Post-pass: targeted elimination of residual soft violations
        (boundary / grouping / MIB) via exact candidate enumeration with an
        evaluator-faithful acceptance test.  Contained: any failure —
        including an absent violation_killer module — returns `out` unchanged.
        Runs until the absolute deadline `t_kill` (carved out of the case
        budget by solve, so per-case wall-clock is unchanged).  Opt-in via
        VKILL=1 (bare defaults keep the original pipeline byte-identical);
        VKILL_OFF=1 is a hard override."""
        if not os.environ.get("VKILL") or os.environ.get("VKILL_OFF"):
            return out
        try:
            from violation_killer import kill_violations
            return kill_violations(
                out, at, cons, tpos, b2b, p2b, pins,
                budget=max(0.2, t_kill - time.time()),
                verbose=self.verbose)
        except Exception:
            return out

    def _column_edge_seat(self, column_out, at, cons, tpos, b2b, p2b, pins,
                          direct_box):
        """Path coverage for `layout_refiner._edge_seat`.

        `_edge_seat` runs inside the direct-prediction refine ladder, so the
        COLUMN arm -- which wins most cases -- never saw it: its residual
        boundary-tag hovers survived to the evaluator untouched.  This runs
        the same surgical pass over the column champion before `_pick_best`
        arbitrates, so both arms are judged after their tags are seated.

        Ordering: this is deliberately BEFORE `_coord_polish`.  Polish's
        BOUNDARY=2 mode only preserves tag bits that are ALREADY satisfied,
        so seating first hands it more walls to protect.

        Opt-in via PARTNER_EDGE_SEAT_V2=1 (the same flag that widens
        `_edge_seat` itself -- the pass is only worth its cost in the widened
        form).  Off: returns the SAME list object, no import, no scorer
        build.  Contained: any failure returns `column_out` unchanged."""
        if not edge_seat_v2_on():
            return column_out
        try:
            from layout_refiner import _edge_seat
            # reuse the direct channel's scorer when there is one -- it is
            # constraint-derived, so it scores any layout of this instance
            # (this is exactly what `_pick_best` already does with it)
            scorer = direct_box[0][1] if direct_box else _ColumnOptimizer(
                [tuple(map(float, r)) for r in column_out], at, cons, tpos,
                b2b, p2b, pins, time.time() + 1.0, seed=0)
            seated = _edge_seat(scorer, column_out)
            return [tuple(map(float, r)) for r in seated]
        except Exception:
            return column_out

    def _final_seat(self, out, at, cons, tpos, b2b, p2b, pins, direct_box):
        """Run `_edge_seat` once more on the FINAL layout (PARTNER_SEAT_FINAL=1).

        Every existing call site sits upstream of a stage that can still move
        blocks: the column arm is seated in `_column_edge_seat` BEFORE
        `_pick_best` arbitrates, the direct arm is seated inside its own refine
        ladder, and `_coord_polish` then re-solves both axis coordinate
        problems on whichever layout won.  So the list this method receives has
        never been offered to `_edge_seat`, and measurably is not at its fixed
        point: replaying the pass over the shipped n>=100 layouts of a full-100
        run still earned two tags (a 0.6%-area edge dilate on one case, a
        strictly V-improving translate on another) in ~1 ms per case.

        The pass only commits a move when the evaluator's own
        boundary+grouping+MIB total STRICTLY drops, so this can only lower V.
        It is the last thing to touch the layout, so nothing downstream can
        undo it.

        Opt-in via PARTNER_SEAT_FINAL=1 and only meaningful with
        PARTNER_EDGE_SEAT_V2=1 (the narrow historical pass has nothing to add
        here).  Off: returns the SAME list object -- no import, no scorer
        build, byte-identical pipeline.  Contained: any failure returns `out`.
        """
        if not os.environ.get("PARTNER_SEAT_FINAL") or not edge_seat_v2_on():
            return out
        try:
            from layout_refiner import _edge_seat
            scorer = direct_box[0][1] if direct_box else _ColumnOptimizer(
                [tuple(map(float, r)) for r in out], at, cons, tpos,
                b2b, p2b, pins, time.time() + 1.0, seed=0)
            seated = _edge_seat(scorer, out)
            if os.environ.get("PARTNER_SEAT_FINAL_DEBUG"):
                # self-paired accounting: V before and after, on the SAME
                # layout in the SAME run.  A pass that can only lower V is
                # measurable this way without a second run, so this readout is
                # immune to the deadline-bounded SA's run-to-run drift.
                import numpy as _np
                a = _np.asarray([tuple(map(float, r)) for r in out],
                                dtype=float)
                b = _np.asarray([tuple(map(float, r)) for r in seated],
                                dtype=float)
                nmoved = (int((_np.abs(a - b) > 1e-12).any(axis=1).sum())
                          if a.shape == b.shape else -1)
                try:
                    from violation_killer import _violations_exact
                    v0, v1 = (_violations_exact(scorer, a),
                              _violations_exact(scorer, b))
                except Exception:
                    v0 = v1 = -1
                print(f"[fseat] n={len(out)} moved={nmoved} "
                      f"V={v0}->{v1}", file=sys.stderr, flush=True)
            return [tuple(map(float, r)) for r in seated]
        except Exception:
            return out

    def _wall_repair_final(self, out, at, cons, tpos, b2b, p2b, pins,
                          direct_box):
        """Run `layout_refiner._wall_repair` on the FINAL layout
        (PARTNER_WALL_REPAIR=1, default off).

        The move it makes is the one no seat pass can make on its own: a
        boundary-tagged CLUSTER member that could translate straight onto its
        wall, where doing so detaches it from its cluster.  Each half is
        exactly V-neutral, and every pass here (`_edge_seat`, `_final_seat`,
        `_cluster_seat`, `tag_compress`) commits only on a strict drop, so
        the pair is unreachable from any of them.  See `_wall_repair` for the
        classification this rests on (28 of the 57 unsatisfied boundary bits
        at n >= 76 in a shipped official-100 run are exactly this shape).

        Runs LAST, after `_tag_compress`, for the same reason `_tag_compress`
        runs after `_final_seat`: the pass is monotone (evaluator-form
        boundary+grouping+MIB total must strictly drop, and the result must
        introduce no new hard-legality problem), so nothing downstream can
        undo it and nothing upstream has to anticipate it.

        Off: returns the SAME list object -- no import, no scorer build,
        byte-identical pipeline.  Contained: any failure returns `out`."""
        if not wall_repair_on():
            return out
        try:
            from layout_refiner import _wall_repair
            scorer = direct_box[0][1] if direct_box else _ColumnOptimizer(
                [tuple(map(float, r)) for r in out], at, cons, tpos,
                b2b, p2b, pins, time.time() + 1.0, seed=0)
            P = np.asarray([tuple(map(float, r)) for r in out], dtype=float)
            R = _wall_repair(scorer, P,
                             deadline=time.time() + _env_float(
                                 "PARTNER_WALL_REPAIR_MS", 60.0) / 1000.0)
            R = np.asarray(R, dtype=float)
            if R.shape != P.shape or not np.all(np.isfinite(R)):
                return out
            if os.environ.get("PARTNER_WALL_REPAIR_DEBUG"):
                # self-paired accounting, the same shape as `[fseat]`: V
                # before and after on the SAME layout in the SAME run, so it
                # is immune to the deadline-bounded SA's run-to-run drift.
                from violation_killer import _violations_exact
                print(f"[wrep] n={len(out)} "
                      f"moved={int((np.abs(P - R) > 1e-12).any(axis=1).sum())} "
                      f"V={_violations_exact(scorer, P)}->"
                      f"{_violations_exact(scorer, R)}",
                      file=sys.stderr, flush=True)
            return [tuple(map(float, r)) for r in R]
        except Exception:
            return out

    def _tag_compress(self, out, at, cons, tpos, b2b, p2b, pins, direct_box):
        """Compact the FINAL layout onto its preplaced boundary-tag lines
        (PARTNER_TAG_COMPRESS=1).  See partner/tag_compress.py.

        `_edge_seat` (and so `_final_seat`) can only translate, dilate or
        pull at most 8 rigid outliers; the blocks that overshoot a preplaced
        tag line sit in packed chains 6-16 deep, so the shipped n>=100
        layouts still hand the evaluator a bbox edge past a hard-locked
        block's own tagged edge.  This pushes the whole chain back onto the
        line, absorbing the residual through the soft blocks' unused 1% area
        tolerance.

        Runs LAST, after `_final_seat`, because the pass is monotone: it
        commits only on a strict drop in the evaluator's own
        boundary+grouping+MIB total, or on an unchanged total with a
        strictly smaller bbox and no HPWL regression.  Nothing downstream
        can undo it.

        Off: returns the SAME list object -- no import, no scorer build,
        byte-identical pipeline.  Contained: any failure returns `out`."""
        tag_on = bool(os.environ.get("PARTNER_TAG_COMPRESS"))
        bridge_on = bool(os.environ.get("PARTNER_GROUP_BRIDGE"))
        dag_on = os.environ.get("PARTNER_GROUP_DAG_BRIDGE") == "1"
        if not (tag_on or bridge_on or dag_on):
            return out
        scorer = None
        try:
            scorer = direct_box[0][1] if direct_box else _ColumnOptimizer(
                [tuple(map(float, r)) for r in out], at, cons, tpos,
                b2b, p2b, pins, time.time() + 1.0, seed=0)
        except Exception:
            return out
        current = out
        if tag_on:
            try:
                from tag_compress import tag_compress
                current = tag_compress(scorer, current)
            except Exception:
                current = out
            if os.environ.get("PARTNER_TAG_COMPRESS_DEBUG"):
                # self-paired accounting on the SAME layout in the SAME run:
                # a pass that can only lower V is measurable without a second
                # run, so this readout is immune to the SA's run-to-run drift.
                import numpy as _np
                a = _np.asarray([tuple(map(float, r)) for r in out],
                                dtype=float)
                b = _np.asarray([tuple(map(float, r)) for r in current],
                                dtype=float)
                nmoved = (int((_np.abs(a - b) > 1e-12).any(axis=1).sum())
                          if a.shape == b.shape else -1)
                try:
                    from violation_killer import _violations_exact
                    v0, v1 = (_violations_exact(scorer, a),
                              _violations_exact(scorer, b))
                except Exception:
                    v0 = v1 = -1
                ar = [float(((p[:, 0] + p[:, 2]).max() - p[:, 0].min())
                            * ((p[:, 1] + p[:, 3]).max() - p[:, 1].min()))
                      for p in (a, b)]
                print(f"[tcomp] n={len(out)} moved={nmoved} V={v0}->{v1} "
                      f"darea={(ar[1] / max(ar[0], 1e-9) - 1) * 100:.3f}%",
                      file=sys.stderr, flush=True)
        if not (bridge_on or dag_on):
            return current
        debug_bridge = os.environ.get("PARTNER_GROUP_BRIDGE_DEBUG") == "1"
        try:
            try:
                budget = float(os.environ.get("PARTNER_GROUP_BRIDGE_BUDGET", "0.02"))
            except (TypeError, ValueError):
                budget = 0.02
            from violation_killer import bridge_grouping_violations
            started = time.perf_counter() if debug_bridge else None
            bridged = bridge_grouping_violations(scorer, current, budget)
            elapsed_ms = ((time.perf_counter() - started) * 1000
                          if debug_bridge else None)
        except Exception:
            return current
        if debug_bridge:
            try:
                from violation_killer import _grouping_count, _violations_exact, _bbox_area
                import numpy as _np
                before = current
                P0 = _np.asarray([tuple(map(float, r)) for r in before], dtype=float)
                grouping0 = _grouping_count(scorer, P0)
                violations0 = _violations_exact(scorer, P0)
                hpwl0 = float(scorer._hpwl(P0))
                bbox0 = _bbox_area(P0)
                P1 = _np.asarray([tuple(map(float, r)) for r in bridged], dtype=float)
                grouping1 = _grouping_count(scorer, P1)
                violations1 = _violations_exact(scorer, P1)
                hpwl1 = float(scorer._hpwl(P1))
                bbox1 = _bbox_area(P1)
                print(f"[gbridge] n={len(before)} ms={elapsed_ms:.3f} "
                      f"grouping={grouping0}->{grouping1} V={violations0}->{violations1} "
                      f"hpwl={hpwl0:.6f}->{hpwl1:.6f} bbox={bbox0:.6f}->{bbox1:.6f} "
                      f"committed={int(bridged != before)}", file=sys.stderr, flush=True)
            except Exception:
                pass
        if not dag_on:
            return bridged

        try:
            base_array = np.asarray(
                [tuple(map(float, r)) for r in bridged], dtype=float)
            if base_array.ndim != 2 or base_array.shape[1] != 4 \
                    or not np.all(np.isfinite(base_array)):
                return bridged
            from violation_killer import _grouping_count
            if _grouping_count(scorer, base_array) == 0:
                return bridged

            dag_debug = os.environ.get("PARTNER_GROUP_DAG_BRIDGE_DEBUG") == "1"
            try:
                dag_budget = float(os.environ.get("PARTNER_GROUP_DAG_BRIDGE_BUDGET", "0.003"))
            except (TypeError, ValueError):
                dag_budget = 0.003
            if not np.isfinite(dag_budget) or dag_budget <= 0.0:
                dag_budget = 0.003
            from violation_killer import bridge_grouping_violations_dag
            dag_started = time.perf_counter() if dag_debug else None
            dagged = bridge_grouping_violations_dag(scorer, bridged, dag_budget)
            dag_array = np.asarray([tuple(map(float, r)) for r in dagged], dtype=float)
            if dag_array.shape != base_array.shape or not np.all(np.isfinite(dag_array)):
                return bridged
            if dag_debug:
                elapsed_ms = (time.perf_counter() - dag_started) * 1000.0
                _print_dag_bridge_diag(
                    scorer, base_array, dag_array, elapsed_ms,
                    not np.array_equal(base_array, dag_array))
            return dagged
        except Exception:
            return bridged

    def _coord_polish(self, out, at, cons, tpos, b2b, p2b, pins,
                      elapsed: Optional[float] = None):
        """Post-pass: order-preserving simultaneous-axis coordinate polish of
        the final layout (see partner/coord_polish.py).  Every stage upstream
        of here optimises coordinates by coordinate descent over groups, so the
        returned layout is only locally optimal for its own topology; this pass
        re-solves both axis coordinate problems jointly under the layout's own
        separation DAG.

        Opt-in via PARTNER_COORD_POLISH=1.  With the flag unset the module is
        never imported and this method is a single dict lookup, so the default
        pipeline stays byte-identical.  The pass is time-boxed
        (PARTNER_COORD_POLISH_BUDGET_MS, default 300) and its cost is INSIDE
        the per-case timing boundary, so it is charged honestly to runtime.
        Contained: any failure returns `out` unchanged.

        PARTNER_COORD_POLISH_HEADROOM_S (default 0, disabled): once set > 0,
        skip the pass once the case has already spent that many seconds of
        wall clock (measured from solve()'s `start`, passed in as `elapsed`).
        Ported from the handover build's headroom gate; default kept at 0 so
        this stays a no-op unless explicitly enabled."""
        if not os.environ.get("PARTNER_COORD_POLISH"):
            return out
        head = _env_float("PARTNER_COORD_POLISH_HEADROOM_S", 0.0)
        if head > 0.0 and elapsed is not None and elapsed > head:
            return out
        try:
            from coord_polish import polish_layout
            return polish_layout(out, at, cons, tpos, b2b, p2b, pins,
                                 verbose=self.verbose)
        except Exception:
            return out

    def _soft_violations(self, out, cons, b2b, p2b, pins):
        """Evaluator-faithful boundary+grouping+MIB count of `out`.

        Uses `coord_polish._LiteScorer`, which is the cheap constraint-geometry
        context the polish pass already builds, so the routing feature costs a
        scorer build rather than a full `_ColumnOptimizer`.  Returns None when
        it cannot be computed, which the router reads as "no precondition".
        """
        try:
            import numpy as _np
            from coord_polish import _LiteScorer
            from violation_killer import _violations_exact
            P = _np.asarray([[float(v) for v in r] for r in out],
                            dtype=_np.float64)
            scorer = _LiteScorer(len(out), cons, b2b, p2b, pins)
            return int(_violations_exact(scorer, P))
        except Exception:
            return None

    def _routed_quality_pass(self, out, at, cons, tpos, b2b, p2b, pins,
                             router):
        """Router-driven violation repair + second polish.

        Opt-in via PARTNER_POST_ROUTER=1 (`router` is None otherwise, and this
        method is never called).  Same stage order and same failure
        containment as coord_polish/violation_kill: repair first (it can only
        reduce violations), polish second (it re-solves both axis coordinate
        problems for the repaired topology).  Each stage is bounded by an
        ABSOLUTE deadline drawn from the shared reserve, so the pair cannot
        turn the carved reserve into appended wall clock.  Ported from the
        handover build's `_routed_quality_pass`; the stage-receipt
        instrumentation was intentionally left out (see partner/postpass_router.py
        for the router itself)."""
        import postpass_router as _pr

        current = out
        # -- violation repair ------------------------------------------------
        ok, reason = router.allow(_pr.VKILL)
        if ok:
            vio = self._soft_violations(current, cons, b2b, p2b, pins)
            ok, reason = router.allow(_pr.VKILL, soft_violations=vio)
        if ok:
            t0 = time.time()
            try:
                from violation_killer import kill_violations
                nxt = kill_violations(
                    current, at, cons, tpos, b2b, p2b, pins,
                    budget=router.budget_for(_pr.VKILL),
                    verbose=self.verbose)
            except Exception:
                nxt = current
            router.spend(time.time() - t0)
            current = nxt
        else:
            router.note_skip(_pr.VKILL, reason)

        # -- second coordinate polish ---------------------------------------
        prev = getattr(self, "_post_polish_layout", None)
        dirty = None
        if prev is not None:
            try:
                dirty = [tuple(map(float, r)) for r in current] != prev
            except Exception:
                dirty = None
        ok, reason = router.allow(
            _pr.POLISH, layout_dirty=dirty,
            first_polish_truncated=self._first_polish_truncated)
        if ok:
            t0 = time.time()
            slice_s = router.budget_for(_pr.POLISH)
            try:
                from coord_polish import polish_layout
                nxt = polish_layout(
                    current, at, cons, tpos, b2b, p2b, pins,
                    deadline=t0 + slice_s, verbose=self.verbose)
            except Exception:
                nxt = current
            router.spend(time.time() - t0)
            current = nxt
        else:
            router.note_skip(_pr.POLISH, reason)

        return current

    def _sample_direct_raw_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                                 K, oversample: bool = True,
                                 gen_seed: int = 0) -> List[np.ndarray]:
        """Dispatch to the legacy or quota-first source-portfolio sampler.

        PARTNER_QUOTA_FIRST (default OFF, read the same way as the other
        boolean env toggles in this module, e.g. PARTNER_FLOW_WARM) gates the
        behavior:

          OFF (default) -- byte-identical to the pre-quota-first baseline:
          always samples the full oversampled Direct batch, then (if a flow
          model + PARTNER_FLOW_SLOTS>0) replaces a suffix with Flow samples
          via `candidate_supply.allocate_quotas`.  At a shipped six-candidate
          / ten-Flow-slot operating point this means the whole Direct batch
          is paid for in GPU latency and then thrown away.

          ON -- quota-first: source quotas are allocated *before* either
          model runs.  When the quota hands Flow the entire batch, the
          Direct conditioning tensors are never built and the Direct sampler
          never runs; in the mixed case Direct only samples its own quota.
          Flow remains failure-contained: if it raises or returns fewer than
          its quota, Direct backfills the missing capacity so the total
          candidate count handed to the refine ladder is UNCHANGED.

        Both paths populate `self._last_source_labels` / `_last_source_receipt`
        and print the same `[sources] ...` line under PARTNER_SOURCE_DEBUG=1.

        `gen_seed` is an OFFSET added to the pinned generator seeds (default 0
        -> byte-identical to the reviewed baseline when Flow is disabled).
        PARTNER_GPU_ARM uses it to draw a genuinely different second wave on
        the idle accelerator; the seeds are pinned, so calling this twice
        with the default offset would return the same batch.
        """
        quota_first = os.environ.get("PARTNER_QUOTA_FIRST", "0") in (
            "1", "true", "True", "on", "ON")
        if quota_first:
            return self._sample_direct_raw_preds_quota_first(
                n, at, cons, tpos, b2b, p2b, pins, K,
                oversample=oversample, gen_seed=gen_seed)
        return self._sample_direct_raw_preds_legacy(
            n, at, cons, tpos, b2b, p2b, pins, K,
            oversample=oversample, gen_seed=gen_seed)

    def _sample_direct_raw_preds_quota_first(self, n, at, cons, tpos, b2b,
                                             p2b, pins, K,
                                             oversample: bool = True,
                                             gen_seed: int = 0
                                             ) -> List[np.ndarray]:
        """Quota-first source portfolio (PARTNER_QUOTA_FIRST=1).

        See `_sample_direct_raw_preds` docstring for the full rationale.
        """
        # sampling is batched, so oversample cheaply and let the prescreen
        # keep the best K for the (expensive) refine workers — but only
        # when the budget affords the extra sampling latency (on ~6 s
        # mid-size cases the doubled batch starved the refine workers and
        # lost cases direct used to win)
        # PARTNER_OVERSAMPLE: sampling is batched and runs concurrently
        # with the already-dispatched column restarts, so a bigger batch is
        # nearly free in wall-clock; the prescreen keeps the best K.
        os_f = 2
        try:
            os_f = max(1, int(float(os.environ.get("PARTNER_OVERSAMPLE",
                                                   "2"))))
        except ValueError:
            os_f = 2
        # cap the GPU batch: sampling happens before the refine dispatch,
        # so an oversized batch delays every refine worker (measured cliff
        # near ~2x the validated 48-sample batch)
        ks_cap = _env_int("PARTNER_KS_CAP", 56)
        total = max(K, min(os_f * K, ks_cap)) if oversample else K
        total = max(0, int(total))
        if total == 0:
            self._last_source_labels = []
            self._last_source_receipt = {
                "requested": {"direct": 0, "flow": 0},
                "attempted": {"direct": 0, "flow": 0},
                "produced": {"direct": 0, "flow": 0},
                "latency_s": {"direct": 0.0, "flow": 0.0},
                "fallback": [], "total": 0,
            }
            return []

        # Opt-in flow-matching candidate source (default off): quota-first --
        # the total candidate count handed to the refine ladder is UNCHANGED
        # (replace-not-add) -- see candidate_supply.allocate_quotas.
        flow_slots = _env_int("PARTNER_FLOW_SLOTS", 0)
        flow_n = (min(flow_slots, total)
                  if self.flow_model is not None and flow_slots > 0 else 0)
        planned_direct = total - flow_n
        receipt = {
            "requested": {"direct": planned_direct, "flow": flow_n},
            "attempted": {"direct": 0, "flow": 0},
            "produced": {"direct": 0, "flow": 0},
            "latency_s": {"direct": 0.0, "flow": 0.0},
            "fallback": [], "total": total,
        }

        flow_preds: List[np.ndarray] = []
        if flow_n > 0:
            receipt["attempted"]["flow"] = flow_n
            started = time.perf_counter()
            try:
                produced = self._sample_flow_preds(
                    n, at, cons, tpos, b2b, p2b, pins, flow_n,
                    gen_seed=gen_seed)
                flow_preds = list(produced[:flow_n])
                if len(flow_preds) < flow_n:
                    receipt["fallback"].append("flow_short")
            except Exception as exc:
                flow_preds = []
                receipt["fallback"].append(
                    f"flow_error:{type(exc).__name__}")
            receipt["latency_s"]["flow"] = time.perf_counter() - started
        receipt["produced"]["flow"] = len(flow_preds)

        # A failed/short Flow draw backfills the remaining Direct capacity;
        # when Flow filled the whole quota, direct_n is 0 and the Direct
        # conditioning tensors / sampler are never touched.
        direct_n = max(0, total - len(flow_preds))
        direct_preds: List[np.ndarray] = []
        if direct_n > 0:
            receipt["attempted"]["direct"] = direct_n
            started = time.perf_counter()
            direct_preds = self._sample_direct_only_raw_preds(
                n, at, cons, tpos, b2b, p2b, pins, direct_n,
                gen_seed=gen_seed)
            receipt["latency_s"]["direct"] = time.perf_counter() - started

        preds = direct_preds + flow_preds
        labels = (["direct"] * len(direct_preds)
                  + ["flow"] * len(flow_preds))
        receipt["produced"]["direct"] = len(direct_preds)

        self._last_source_labels = labels
        self._last_source_receipt = receipt
        if os.environ.get("PARTNER_SOURCE_DEBUG"):
            print(f"[sources] n={n} requested={receipt['requested']} "
                  f"attempted={receipt['attempted']} "
                  f"produced={receipt['produced']} "
                  f"latency={receipt['latency_s']} "
                  f"fallback={receipt['fallback']}",
                  file=sys.stderr, flush=True)

        # *** PROBE ONLY, NEVER PROMOTABLE *** (PARTNER_ORACLE_PRED_FILE).
        # With the flag unset `ORACLE_PRED_ON` is a False constant bound at
        # import, so the production path is this one dead branch test.  When
        # set, the raw Direct batch is replaced by a ground-truth-derived
        # layout so the channel downstream of the sampler (prescreen ->
        # refine_prediction -> selector) can be measured against a PERFECT
        # prior.  See column_sa_legalizer.oracle_pred_override.
        if ORACLE_PRED_ON:
            preds = oracle_pred_override(preds, n)
            self._last_source_labels = ["oracle"] * len(preds)
        return preds

    def _sample_direct_raw_preds_legacy(self, n, at, cons, tpos, b2b, p2b,
                                        pins, K, oversample: bool = True,
                                        gen_seed: int = 0
                                        ) -> List[np.ndarray]:
        """Pre-quota-first baseline (PARTNER_QUOTA_FIRST unset/0): always
        samples the full oversampled Direct batch first, then (opt-in)
        replaces a fixed suffix with Flow samples.  Kept byte-identical to
        the reviewed baseline so PARTNER_QUOTA_FIRST=0 is a true no-op; only
        the `_last_source_labels`/`_last_source_receipt` bookkeeping (and the
        PARTNER_SOURCE_DEBUG print) is new, and is derived post-hoc without
        touching the sampling calls or their generator consumption order.
        """
        from direct_diffusion_train import fast_condition
        from direct_diffusion_model import (known_z_channels, sample_direct,
                                         sample_direct_dpmpp)
        dev = self.device
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), _cond_p2b(p2b.unsqueeze(0).to(dev)),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=self.direct_cfg.relation_feat_dim,
            node_feat_dim=self.direct_cfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        gen = torch.Generator(device=dev)
        gen.manual_seed(17 + int(gen_seed))
        # sampling is batched, so oversample cheaply and let the prescreen
        # keep the best K for the (expensive) refine workers — but only
        # when the budget affords the extra sampling latency (on ~6 s
        # mid-size cases the doubled batch starved the refine workers and
        # lost cases direct used to win)
        # PARTNER_OVERSAMPLE: sampling is batched and runs concurrently
        # with the already-dispatched column restarts, so a bigger batch is
        # nearly free in wall-clock; the prescreen keeps the best K.
        os_f = 2
        try:
            os_f = max(1, int(float(os.environ.get("PARTNER_OVERSAMPLE",
                                                   "2"))))
        except ValueError:
            os_f = 2
        # cap the GPU batch: sampling happens before the refine dispatch,
        # so an oversized batch delays every refine worker (measured cliff
        # near ~2x the validated 48-sample batch)
        ks_cap = _env_int("PARTNER_KS_CAP", 56)
        K_s = max(K, min(os_f * K, ks_cap)) if oversample else K
        if (os.environ.get("PARTNER_NOISE_OPT") == "hybrid"
                and self.direct_model is not None):
            try:
                preds = self._noise_opt_hybrid_preds(
                    n, cond, z_known, known, scale, at_d, cons_d, tpos_d,
                    b2b.to(dev), K_s)
                if preds:
                    self._last_source_labels = ["direct"] * len(preds)
                    self._last_source_receipt = {
                        "requested": {"direct": len(preds), "flow": 0},
                        "attempted": {"direct": len(preds), "flow": 0},
                        "produced": {"direct": len(preds), "flow": 0},
                        "latency_s": {"direct": 0.0, "flow": 0.0},
                        "fallback": [], "total": len(preds),
                    }
                    if os.environ.get("PARTNER_SOURCE_DEBUG"):
                        print(f"[sources] n={n} "
                              f"requested={self._last_source_receipt['requested']} "
                              f"attempted={self._last_source_receipt['attempted']} "
                              f"produced={self._last_source_receipt['produced']} "
                              f"latency={self._last_source_receipt['latency_s']} "
                              f"fallback={self._last_source_receipt['fallback']}",
                              file=sys.stderr, flush=True)
                    return preds
            except Exception as exc:  # never harm the default channel
                if self.verbose:
                    print(f"noise-opt hybrid failed: {exc}", file=sys.stderr)
        with torch.no_grad():
            cond_k = {k: (v.expand(K_s, *v.shape[1:]).contiguous()
                          if torch.is_tensor(v) else v)
                      for k, v in cond.items()}
            guide = None
            if os.environ.get("PARTNER_PHYSICS_GUIDE") == "1":
                from physics_guidance import (GuidanceConfig,
                                                     build_context,
                                                     make_guidance)
                ctx = build_context(at_d, cons_d, b2b.to(dev), scale,
                                    known).expand(K_s)
                guide = make_guidance(ctx, GuidanceConfig.from_env())
            solver = os.environ.get("PARTNER_DIRECT_SOLVER", "ddim")
            if solver == "dpmpp" and guide is None:
                z = sample_direct_dpmpp(
                    self.direct_model, cond_k, self.direct_schedule,
                    steps=_env_int("PARTNER_DDIM_STEPS", 50),
                    generator=gen,
                    z_known=z_known.expand(K_s, -1, -1),
                    known_mask=known.expand(K_s, -1, -1))
            else:
                z = sample_direct(self.direct_model, cond_k,
                                  self.direct_schedule,
                                  steps=_env_int("PARTNER_DDIM_STEPS", 50),
                                  generator=gen,
                                  z_known=z_known.expand(K_s, -1, -1),
                                  known_mask=known.expand(K_s, -1, -1),
                                  guidance=guide)
            rects = z_to_rectangles(
                z, at_d.expand(K_s, -1),
                target_positions=tpos_d.expand(K_s, -1, -1),
                constraints=cons_d.expand(K_s, -1, -1),
                z_repr=self.direct_cfg.z_repr)
        preds = [rects[k, :n].cpu().numpy().astype(np.float64)
                 for k in range(K_s)]

        # Opt-in flow-matching candidate source (default off): replaces a
        # fixed slice of the Direct batch with flow samples so the total
        # candidate count handed to the refine ladder is UNCHANGED
        # (replace-not-add) -- see candidate_supply.allocate_quotas.
        direct_n = len(preds)
        flow_n = 0
        flow_preds: List[np.ndarray] = []
        if self.flow_model is not None:
            flow_slots = _env_int("PARTNER_FLOW_SLOTS", 0)
            if flow_slots > 0 and preds:
                quotas = allocate_quotas(
                    len(preds),
                    {"direct": max(0, len(preds) - flow_slots), "flow": flow_slots},
                    ("direct", "flow"),
                )
                flow_n = quotas.get("flow", 0)
                if flow_n > 0:
                    try:
                        flow_preds = self._sample_flow_preds(
                            n, at, cons, tpos, b2b, p2b, pins, flow_n,
                            gen_seed=gen_seed)
                        direct_n = quotas.get("direct", len(preds) - flow_n)
                        preds = preds[:direct_n] + flow_preds
                    except Exception:
                        flow_preds = []
                        flow_n = 0
                        direct_n = len(preds)
                        # flow failure never harms the Direct channel

        # Bookkeeping only (does not affect `preds` or generator consumption
        # order): mirrors the quota-first receipt/labels shape so downstream
        # consumers (PARTNER_SOURCE_DEBUG, diagnostics) work identically
        # under either toggle setting.
        labels = (["direct"] * direct_n) + (["flow"] * len(flow_preds))
        receipt = {
            "requested": {"direct": direct_n, "flow": flow_n},
            "attempted": {"direct": direct_n,
                          "flow": flow_n if flow_n else 0},
            "produced": {"direct": direct_n, "flow": len(flow_preds)},
            "latency_s": {"direct": 0.0, "flow": 0.0},
            "fallback": [] if (flow_n == 0 or len(flow_preds) == flow_n)
                        else ["flow_short_or_error"],
            "total": len(preds),
        }
        self._last_source_labels = labels
        self._last_source_receipt = receipt
        if os.environ.get("PARTNER_SOURCE_DEBUG"):
            print(f"[sources] n={n} requested={receipt['requested']} "
                  f"attempted={receipt['attempted']} "
                  f"produced={receipt['produced']} "
                  f"latency={receipt['latency_s']} "
                  f"fallback={receipt['fallback']}",
                  file=sys.stderr, flush=True)

        # *** PROBE ONLY, NEVER PROMOTABLE *** (PARTNER_ORACLE_PRED_FILE).
        # With the flag unset `ORACLE_PRED_ON` is a False constant bound at
        # import, so the production path is this one dead branch test.  When
        # set, the raw Direct batch is replaced by a ground-truth-derived
        # layout so the channel downstream of the sampler (prescreen ->
        # refine_prediction -> selector) can be measured against a PERFECT
        # prior.  See column_sa_legalizer.oracle_pred_override.
        if ORACLE_PRED_ON:
            preds = oracle_pred_override(preds, n)
            self._last_source_labels = ["oracle"] * len(preds)
        return preds

    def _sample_direct_only_raw_preds(self, n, at, cons, tpos, b2b, p2b,
                                      pins, K,
                                      gen_seed: int = 0) -> List[np.ndarray]:
        """Generate exactly ``K`` Direct candidates, with no source mixing.

        `gen_seed` is an OFFSET added to the pinned generator seeds (default 0
        -> pinned draws).  Oversampling and source quotas are owned by
        `_sample_direct_raw_preds`, so a zero Direct quota never enters this
        method or builds Direct conditioning tensors.
        """
        K_s = max(0, int(K))
        if K_s == 0:
            return []
        from direct_diffusion_train import fast_condition
        from direct_diffusion_model import (known_z_channels, sample_direct,
                                         sample_direct_dpmpp)
        dev = self.device
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), _cond_p2b(p2b.unsqueeze(0).to(dev)),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=self.direct_cfg.relation_feat_dim,
            node_feat_dim=self.direct_cfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        gen = torch.Generator(device=dev)
        gen.manual_seed(17 + int(gen_seed))
        if (os.environ.get("PARTNER_NOISE_OPT") == "hybrid"
                and self.direct_model is not None):
            try:
                preds = self._noise_opt_hybrid_preds(
                    n, cond, z_known, known, scale, at_d, cons_d, tpos_d,
                    b2b.to(dev), K_s)
                if preds:
                    return preds
            except Exception as exc:  # never harm the default channel
                if self.verbose:
                    print(f"noise-opt hybrid failed: {exc}", file=sys.stderr)
        with torch.no_grad():
            cond_k = {k: (v.expand(K_s, *v.shape[1:]).contiguous()
                          if torch.is_tensor(v) else v)
                      for k, v in cond.items()}
            guide = None
            if os.environ.get("PARTNER_PHYSICS_GUIDE") == "1":
                from physics_guidance import (GuidanceConfig,
                                                     build_context,
                                                     make_guidance)
                ctx = build_context(at_d, cons_d, b2b.to(dev), scale,
                                    known).expand(K_s)
                guide = make_guidance(ctx, GuidanceConfig.from_env())
            solver = os.environ.get("PARTNER_DIRECT_SOLVER", "ddim")
            if solver == "dpmpp" and guide is None:
                z = sample_direct_dpmpp(
                    self.direct_model, cond_k, self.direct_schedule,
                    steps=_env_int("PARTNER_DDIM_STEPS", 50),
                    generator=gen,
                    z_known=z_known.expand(K_s, -1, -1),
                    known_mask=known.expand(K_s, -1, -1))
            else:
                z = sample_direct(self.direct_model, cond_k,
                                  self.direct_schedule,
                                  steps=_env_int("PARTNER_DDIM_STEPS", 50),
                                  generator=gen,
                                  z_known=z_known.expand(K_s, -1, -1),
                                  known_mask=known.expand(K_s, -1, -1),
                                  guidance=guide)
            rects = z_to_rectangles(
                z, at_d.expand(K_s, -1),
                target_positions=tpos_d.expand(K_s, -1, -1),
                constraints=cons_d.expand(K_s, -1, -1),
                z_repr=self.direct_cfg.z_repr)
        preds = [rects[k, :n].cpu().numpy().astype(np.float64)
                 for k in range(K_s)]
        return preds

    def _sample_flow_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                           K, gen_seed: int = 0) -> List[np.ndarray]:
        """Sample K layouts from the opt-in flow-matching model.

        Default path is unchanged. Opt-in flow-seed variants (all default off):
        PARTNER_FLOW_ANTITHETIC=1 (V-A, +/-z paired seeds), PARTNER_FLOW_ZORDER=1
        (V-B, zero-order neighborhood resample), PARTNER_FLOW_NOPT=hybrid
        (V-C/D, gradient noise-opt on the flow channel).

        `gen_seed` is the same generator-seed OFFSET the Direct batch takes
        (default 0 -> unchanged).  The ZORDER / NOPT sub-samplers keep their
        own pinned seeds, so under those opt-in flags a PARTNER_GPU_ARM second
        wave can repeat wave-1 flow candidates; the legalizer de-duplicates."""
        from direct_diffusion_train import fast_condition
        from direct_diffusion_model import known_z_channels
        from flow_matching_model import sample_flow
        dev = self.device
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), _cond_p2b(p2b.unsqueeze(0).to(dev)),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=self.flow_cfg.relation_feat_dim,
            node_feat_dim=self.flow_cfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        b2b_d = b2b.to(dev)
        if os.environ.get("PARTNER_FLOW_NOPT") == "hybrid":
            return self._flow_noise_opt_preds(
                n, cond, z_known, known, scale, at_d, cons_d, tpos_d, b2b_d, K)
        if os.environ.get("PARTNER_FLOW_ZORDER"):
            return self._flow_zorder_preds(
                n, cond, z_known, known, scale, at_d, cons_d, tpos_d, b2b_d, K)

        steps = _env_int("PARTNER_FLOW_STEPS", 8)
        solver = os.environ.get("PARTNER_FLOW_SOLVER", "euler")
        cond_k = {k: (v.expand(K, *v.shape[1:]).contiguous()
                      if torch.is_tensor(v) else v) for k, v in cond.items()}
        gen = torch.Generator(device=dev)
        gen.manual_seed(23 + int(gen_seed))
        if os.environ.get("PARTNER_FLOW_ANTITHETIC"):
            # V-A: antithetic +/-z coverage (arXiv 2506.06185). known_noise is
            # shared within each pair so only the free seed is mirrored.
            from noise_optimization import sample_flow_diff
            N = cond["mask"].shape[1]
            zdim = self.flow_cfg.z_dim
            valid = cond_k["mask"].unsqueeze(-1)
            has_known = bool(known.any())
            half = (K + 1) // 2
            zf = torch.randn((half, N, zdim), device=dev, generator=gen)
            z_init = torch.cat([zf, -zf], dim=0)[:K] * valid
            kn = None
            if has_known:
                knh = torch.randn((half, N, zdim), device=dev, generator=gen)
                kn = torch.cat([knh, knh], dim=0)[:K] * known.expand(K, -1, -1)
            with torch.no_grad():
                z = sample_flow_diff(
                    self.flow_model, cond_k, steps, solver, z_init,
                    z_known=z_known.expand(K, -1, -1),
                    known_mask=known.expand(K, -1, -1), known_noise=kn)
        else:
            with torch.no_grad():
                z = sample_flow(
                    self.flow_model, cond_k, steps=steps, solver=solver,
                    generator=gen, z_known=z_known.expand(K, -1, -1),
                    known_mask=known.expand(K, -1, -1)).z
        rects = z_to_rectangles(
            z, at_d.expand(K, -1), target_positions=tpos_d.expand(K, -1, -1),
            constraints=cons_d.expand(K, -1, -1), z_repr=self.flow_cfg.z_repr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]

    def _flow_setup(self, cond, K):
        """Local (ex, ex_cond, N, zdim, valid, has_known, steps, solver)."""
        def ex(t, k):
            return t.expand(k, *t.shape[1:]).contiguous()

        def ex_cond(k):
            return {kk: (ex(v, k) if torch.is_tensor(v) else v)
                    for kk, v in cond.items()}
        return (ex, ex_cond, cond["mask"].shape[1], self.flow_cfg.z_dim,
                cond["mask"].unsqueeze(-1), bool(cond["mask"].numel()),
                _env_int("PARTNER_FLOW_STEPS", 8),
                os.environ.get("PARTNER_FLOW_SOLVER", "euler"))

    def _flow_zorder_preds(self, n, cond, z_known, known, scale,
                           at_d, cons_d, tpos_d, b2b, K) -> List[np.ndarray]:
        """V-B: zero-order neighborhood resample (Ma et al. 2501.09732).

        Sample a pool, pick the top-M by layout energy, draw sigma-Gaussian
        neighbors of each, re-render everything, and keep the best K by energy.
        Pure forward, stochastic (no gradient collapse -> dodges failure mode A;
        selection stays overlap-driven)."""
        import time as _time
        from physics_guidance import build_context, guidance_energy_per_sample
        from noise_optimization import sample_flow_diff
        dev = self.device
        z_repr = self.flow_cfg.z_repr
        ex, ex_cond, N, zdim, _v, _hk, steps, solver = self._flow_setup(cond, K)
        has_known = bool(known.any())
        M = _env_int("PARTNER_FLOW_ZORDER_M", 4)
        nb = _env_int("PARTNER_FLOW_ZORDER_NB", 3)
        sig = _env_float("PARTNER_FLOW_ZORDER_SIGMA", 0.1)
        Ksample = max(K, _env_int("PARTNER_FLOW_ZORDER_K", 16))
        w_ov = _env_float("PARTNER_FLOW_ZORDER_W_OVERLAP", 1.0)
        w_bd = _env_float("PARTNER_FLOW_ZORDER_W_BOUNDARY", 0.5)
        ctx = build_context(at_d, cons_d, b2b, scale, known)

        def render(z_init):
            k = z_init.shape[0]
            valid_k = cond["mask"].unsqueeze(-1).expand(k, -1, -1)
            kn = (z_init * ex(known, k)) if has_known else None
            with torch.no_grad():
                return sample_flow_diff(
                    self.flow_model, ex_cond(k), steps, solver, z_init * valid_k,
                    z_known=ex(z_known, k), known_mask=ex(known, k), known_noise=kn)

        def energy(z0):
            k = z0.shape[0]
            return guidance_energy_per_sample(z0, ctx.expand(k), w_ov, w_bd, 0.0, 0.0)

        t0 = _time.time()
        gen = torch.Generator(device=dev).manual_seed(23)
        valid_s = cond["mask"].unsqueeze(-1).expand(Ksample, -1, -1)
        z_seed = torch.randn((Ksample, N, zdim), device=dev, generator=gen) * valid_s
        z0 = render(z_seed)
        e = energy(z0)
        top = torch.topk(e, min(M, Ksample), largest=False).indices
        neigh = []
        for idx in top.tolist():
            base = z_seed[idx:idx + 1]
            for _ in range(nb):
                neigh.append(base + sig * torch.randn_like(base))
        if neigh:
            z_nb = torch.cat(neigh, dim=0)
            z0_nb = render(z_nb)
            z_all = torch.cat([z0, z0_nb], dim=0)
        else:
            z_all = z0
        e_all = energy(z_all)
        keep = torch.topk(e_all, min(K, z_all.shape[0]), largest=False).indices
        z_keep = z_all[keep].contiguous()
        Kk = z_keep.shape[0]
        rects = z_to_rectangles(
            z_keep, at_d.expand(Kk, -1), target_positions=tpos_d.expand(Kk, -1, -1),
            constraints=cons_d.expand(Kk, -1, -1), z_repr=z_repr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[flow-zorder] n={n} Ksample={Ksample} nb={len(neigh)} "
              f"K={Kk} gpu_s={_time.time() - t0:.2f}", file=sys.stderr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(Kk)]

    def _flow_noise_opt_preds(self, n, cond, z_known, known, scale,
                              at_d, cons_d, tpos_d, b2b, K) -> List[np.ndarray]:
        """V-C/D: gradient noise-opt on the flow channel with an INVERTED energy
        (hpwl-heavy). Tests whether targeting the refine-immovable wirelength
        dimension (not the refine-fixable overlap) revives gradient noise-opt,
        and whether the flow 8-step unroll avoids the DDIM-25 hpwl stall.
        Optional batch-repulsion (V-D) via PGUIDE_W_REP."""
        import time as _time
        from physics_guidance import build_context, guidance_energy_per_sample
        from noise_optimization import NoiseOptConfig, optimize_noise, sample_flow_diff
        dev = self.device
        z_repr = self.flow_cfg.z_repr
        ex, ex_cond, N, zdim, _v, _hk, steps, solver = self._flow_setup(cond, K)
        has_known = bool(known.any())
        M = min(_env_int("PARTNER_FLOW_NOPT_TOPM", 4), K)
        w_rep = _env_float("PGUIDE_W_REP", 0.0)
        sig_rep = _env_float("PGUIDE_SIGMA_REP", 0.1)
        ctx = build_context(at_d, cons_d, b2b, scale, known)
        cfg = NoiseOptConfig.from_env()
        cfg.rounds = _env_int("PARTNER_FLOW_NOPT_ITERS", 10)
        cfg.steps, cfg.solver = steps, solver
        cfg.w_hpwl = _env_float("PARTNER_FLOW_NOPT_W_HPWL", 1.0)
        cfg.w_overlap = _env_float("PARTNER_FLOW_NOPT_W_OVERLAP", 0.2)
        cfg.w_boundary = _env_float("PARTNER_FLOW_NOPT_W_BOUNDARY", 0.2)
        cfg.w_group = 0.0

        def render(z_init):
            k = z_init.shape[0]
            valid_k = cond["mask"].unsqueeze(-1).expand(k, -1, -1)
            kn = (z_init * ex(known, k)) if has_known else None
            with torch.no_grad():
                return sample_flow_diff(
                    self.flow_model, ex_cond(k), steps, solver, z_init * valid_k,
                    z_known=ex(z_known, k), known_mask=ex(known, k), known_noise=kn)

        def base_energy(z0):
            k = z0.shape[0]
            return guidance_energy_per_sample(
                z0, ctx.expand(k), cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, 0.0)

        t0 = _time.time()
        gen = torch.Generator(device=dev).manual_seed(23)
        valid_K = cond["mask"].unsqueeze(-1).expand(K, -1, -1)
        z_seed = torch.randn((K, N, zdim), device=dev, generator=gen) * valid_K
        z0 = render(z_seed)
        e = base_energy(z0)
        best = torch.topk(e, M, largest=False).indices
        worst = torch.topk(e, M, largest=True).indices

        def flow_sampler(seed):
            kn = (seed * ex(known, M)) if has_known else None
            return sample_flow_diff(
                self.flow_model, ex_cond(M), steps, solver, seed,
                z_known=ex(z_known, M), known_mask=ex(known, M), known_noise=kn)

        energy_fn = None
        if w_rep > 0.0:
            ctx_M = ctx.expand(M)

            def energy_fn(z0m):  # V-D: SVGD-style batch repulsion
                base = guidance_energy_per_sample(
                    z0m, ctx_M, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, 0.0)
                x = z0m[..., 0] * ctx_M.scale.view(-1, 1)
                y = z0m[..., 1] * ctx_M.scale.view(-1, 1)
                m = ctx_M.mask.to(z0m.dtype)
                cx = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
                cy = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
                diag = ctx_M.scale.clamp_min(1.0)
                d2 = ((cx[:, None] - cx[None]) ** 2 + (cy[:, None] - cy[None]) ** 2)
                rep = torch.exp(-d2 / (sig_rep * diag.mean()) ** 2)
                rep = rep - torch.diag(torch.diagonal(rep))
                return base + w_rep * rep.sum(dim=1)

        z_opt = optimize_noise(
            None, ex_cond(M), ctx.expand(M), z_seed[best].contiguous(),
            ex(z_known, M), ex(known, M), cfg, energy_fn=energy_fn,
            sample_fn=flow_sampler)
        z_ref = render(z_opt)

        z_out = z0.clone()
        for j, w in enumerate(worst.tolist()):
            z_out[w] = z_ref[j]
        rects = z_to_rectangles(
            z_out, at_d.expand(K, -1), target_positions=tpos_d.expand(K, -1, -1),
            constraints=cons_d.expand(K, -1, -1), z_repr=z_repr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[flow-nopt] n={n} K={K} M={M} iters={cfg.rounds} steps={steps} "
              f"w_rep={w_rep} gpu_s={_time.time() - t0:.2f}", file=sys.stderr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]

    def _noise_opt_hybrid_preds(self, n, cond, z_known, known, scale,
                                at_d, cons_d, tpos_d, b2b, K_s) -> List[np.ndarray]:
        """Opt-in (PARTNER_NOISE_OPT=hybrid) initial-noise optimization.

        Sample K_s candidates at the production step count, rank by the
        per-sample layout energy, gradient-optimize the initial noise of the
        best M via a cheap differentiable unroll, re-render those M at the
        production step count, and REPLACE the worst M of the pool with them
        (candidate count unchanged -- replace-not-add). Any failure raises and
        the caller falls back to the untouched default sampler."""
        import time as _time
        from physics_guidance import build_context, guidance_energy_per_sample
        from noise_optimization import (NoiseOptConfig, optimize_noise,
                                      sample_direct_diff, make_direct_sampler)
        dev = self.device
        model, sched = self.direct_model, self.direct_schedule
        zdim, z_repr = self.direct_cfg.z_dim, self.direct_cfg.z_repr
        N = cond["mask"].shape[1]
        valid = cond["mask"].unsqueeze(-1)
        has_known = bool(known.any())
        render_steps = _env_int("PARTNER_DDIM_STEPS", 50)
        unroll = _env_int("PARTNER_NOPT_UNROLL", 10)
        iters = _env_int("PARTNER_NOPT_ITERS", 10)
        M = min(_env_int("PARTNER_NOPT_TOPM", 4), K_s)
        gc = os.environ.get("PARTNER_NOPT_GRAD_CKPT", "1") != "0"

        def ex(t, k):
            return t.expand(k, *t.shape[1:]).contiguous()

        def ex_cond(k):
            return {kk: (ex(v, k) if torch.is_tensor(v) else v)
                    for kk, v in cond.items()}

        def render(z_init, k, seed):
            with torch.no_grad():
                kn = (torch.randn(render_steps, k, N, zdim, device=dev,
                                  generator=torch.Generator(device=dev).manual_seed(seed))
                      if has_known else None)
                return sample_direct_diff(
                    model, ex_cond(k), sched, render_steps, z_init * ex(valid, k),
                    known_noise=kn, z_known=ex(z_known, k), known_mask=ex(known, k))

        t0 = _time.time()
        ctx = build_context(at_d, cons_d, b2b, scale, known)
        cfg = NoiseOptConfig.from_env()
        cfg.rounds, cfg.steps = iters, unroll

        # 1. sample the K_s-candidate pool (production step count), keep seeds
        gen = torch.Generator(device=dev).manual_seed(_env_int("PARTNER_NOPT_SEED", 20))
        z_seed = torch.randn((K_s, N, zdim), device=dev, generator=gen) * ex(valid, K_s)
        z_pool = render(z_seed, K_s, _env_int("PARTNER_NOPT_SEED", 20) + 1)
        e_pool = guidance_energy_per_sample(
            z_pool, ctx.expand(K_s), cfg.w_overlap, cfg.w_boundary,
            cfg.w_hpwl, cfg.w_group)
        best = torch.topk(e_pool, M, largest=False).indices
        worst = torch.topk(e_pool, M, largest=True).indices

        # 2. optimize the best-M seeds with a cheap unroll
        kn_opt = (torch.randn(unroll, M, N, zdim, device=dev,
                              generator=torch.Generator(device=dev).manual_seed(41))
                  if has_known else None)
        sampler = make_direct_sampler(
            model, ex_cond(M), sched, unroll, ex(z_known, M), ex(known, M),
            kn_opt, grad_checkpoint=gc)
        z_opt = optimize_noise(
            None, ex_cond(M), ctx.expand(M), z_seed[best].contiguous(),
            ex(z_known, M), ex(known, M), cfg, sample_fn=sampler)

        # 3. re-render the optimized seeds at production step count, replace worst
        z_ref = render(z_opt, M, 71)
        rects_pool = z_to_rectangles(
            z_pool, at_d.expand(K_s, -1), target_positions=tpos_d.expand(K_s, -1, -1),
            constraints=cons_d.expand(K_s, -1, -1), z_repr=z_repr)
        rects_ref = z_to_rectangles(
            z_ref, at_d.expand(M, -1), target_positions=tpos_d.expand(M, -1, -1),
            constraints=cons_d.expand(M, -1, -1), z_repr=z_repr)
        preds = [rects_pool[k, :n].cpu().numpy().astype(np.float64) for k in range(K_s)]
        for j, w in enumerate(worst.tolist()):
            preds[w] = rects_ref[j, :n].cpu().numpy().astype(np.float64)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[noise-opt] n={n} K_s={K_s} M={M} iters={iters} unroll={unroll} "
              f"render={render_steps} gpu_s={_time.time() - t0:.2f}", file=sys.stderr)
        return preds

    def _constraint_penalties(self, preds, n, at, cons, as_counts: bool = False):
        """Per-prediction soft-constraint surrogate.

        `as_counts=False` (historical): a NORMALISED penalty -- mean boundary
        slack in [0,1] plus mean cluster-bbox inflation in [0,3].
        `as_counts=True` (PARTNER_PROXY_ALIGN): the same two terms converted
        back to ESTIMATED VIOLATION COUNTS by multiplying each by the number
        of items it was averaged over, so the caller can price them against
        `n_soft` the way the official cost does."""
        if not os.environ.get("PARTNER_PRESCREEN_V"):
            return None
        _f, _p, _mib, clu, bnd = _parse_constraints(cons, n)
        groups = {}
        for i in range(n):
            if clu[i] > 0:
                groups.setdefault(clu[i], []).append(i)
        diag = math.sqrt(max(float(at[:n].sum().item()), 1.0))

        def _viol_est(P):
            x0, y0 = P[:, 0], P[:, 1]
            x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
            X0, Y0, X1, Y1 = x0.min(), y0.min(), x1.max(), y1.max()
            pen = 0.0
            n_b = 0
            for i in range(n):
                code = bnd[i]
                if not code:
                    continue
                n_b += 1
                distance = 0.0
                if code & 1:
                    distance = max(distance, float(x0[i] - X0))
                if code & 2:
                    distance = max(distance, float(X1 - x1[i]))
                if code & 4:
                    distance = max(distance, float(Y1 - y1[i]))
                if code & 8:
                    distance = max(distance, float(y0[i] - Y0))
                pen += min(distance / diag, 1.0)
            if n_b and not as_counts:
                pen /= n_b
            spread = 0.0
            for group in groups.values():
                if len(group) < 2:
                    continue
                group_width = float(x1[group].max() - x0[group].min())
                group_height = float(y1[group].max() - y0[group].min())
                group_area = float((P[group, 2] * P[group, 3]).sum())
                spread += min(max(0.0, group_width * group_height / max(group_area, 1e-9) - 1.2), 3.0)
            if groups and not as_counts:
                spread /= max(sum(len(group) >= 2 for group in groups.values()), 1)
            return pen + spread

        return [_viol_est(prediction) for prediction in preds]

    @staticmethod
    def _soft_norm(n, cons):
        """`_ColumnOptimizer.n_soft_den` computed from the raw constraints:
        boundary-tagged blocks + (size-1) per MIB group + (size-1) per
        cluster.  Same definition as column_sa_legalizer._build_soft_norm."""
        _f, _p, mib, clu, bnd = _parse_constraints(cons, n)
        n_soft = sum(1 for i in range(n) if bnd[i] > 0)
        for tag in (mib, clu):
            groups = {}
            for i in range(n):
                if tag[i] > 0:
                    groups.setdefault(tag[i], []).append(i)
            for idxs in groups.values():
                n_soft += max(0, len(idxs) - 1)
        return max(n_soft, 1)

    def _rank_portfolio(self, preds, n, at, cons, b2b):
        w_b2b = b2b[:n, :n].detach().cpu().numpy() if b2b is not None else None
        # PARTNER_PROXY_ALIGN (default off): rank the prescreen under the
        # official cost's functional form instead of the additive proxy, so
        # a unit of relative violation is worth 4 gap units (exp(2.) vs
        # alpha=0.5) and the bbox gap is priced at all.  See
        # candidate_supply.rank_predictions.  Off -> byte-identical.
        align = os.environ.get("PARTNER_PROXY_ALIGN", "0") in (
            "1", "true", "True", "on", "ON")
        penalties = self._constraint_penalties(preds, n, at, cons,
                                               as_counts=align)
        align_ref = None
        if align and penalties is not None:
            area_ref = float(at[:n].sum().item()) / 0.97   # UTIL_TARGET_REF
            align_ref = (area_ref, float(self._soft_norm(n, cons)))
        return rank_predictions(
            preds,
            at[:n].detach().cpu().numpy(),
            w_b2b if w_b2b is not None else np.zeros((n, n), dtype=np.float64),
            constraint_penalties=penalties,
            violation_weight=(_env_float("PARTNER_PRESCREEN_VW", 0.5)
                              if penalties is not None else 0.0),
            align_ref=align_ref,
        )

    def _sample_direct_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                             K, oversample: bool = True,
                             gen_seed: int = 0) -> List[np.ndarray]:
        """Prescreen the Direct batch exactly as the reviewed baseline does."""
        preds = self._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins,
                                              K, oversample, gen_seed=gen_seed)
        return [preds[index] for index in self._rank_portfolio(preds, n, at, cons, b2b)]

    def _empty_retrieval_batch(self, started: float, rejected: int = 0,
                               query_s: float = 0.0) -> CandidateBatch:
        return CandidateBatch(
            "retrieval", [], max(0.0, time.perf_counter() - started),
            {"source_ids": [], "retrieval_distances": [], "transforms": [],
             "match_costs": [], "match_confidences": [], "query_s": query_s,
             "match_s": 0.0, "transfer_s": 0.0, "rejected": rejected},
        )

    def _sample_retrieval_preds(self, n, at, cons, tpos, b2b, p2b, pins, K) -> CandidateBatch:
        """Query same-N training layouts and transfer one D4 match per source."""
        started = time.perf_counter()
        limit = min(
            max(0, int(K)), max(0, int(self.retrieval_slots)),
            FIRST_R4_RETRIEVAL_SLOTS,
        )
        if limit <= 0 or self.retrieval_index is None:
            return self._empty_retrieval_batch(started)

        try:
            from retrieval_features import extract_retrieval_features
            from retrieval_matching import match_blocks
            from retrieval_transfer import (remap_boundary_node_features,
                                                   transfer_layout)
            area = at[:n].detach().cpu().numpy()
            constraints = cons[:n].detach().cpu().numpy()
            target_positions = tpos[:n].detach().cpu().numpy()
            b2b_array = b2b.detach().cpu().numpy()
            p2b_array = p2b.detach().cpu().numpy()
            pins_array = pins.detach().cpu().numpy()
            target_features = extract_retrieval_features(
                area, b2b_array, p2b_array, pins_array, constraints, target_positions
            )
        except Exception:
            return self._empty_retrieval_batch(started)
        query_started = time.perf_counter()
        try:
            retrieved = self.retrieval_index.query(n, target_features.global_vector, top_k=limit)
        except Exception:
            return self._empty_retrieval_batch(
                started, query_s=time.perf_counter() - query_started
            )
        query_s = time.perf_counter() - query_started

        predictions: List[np.ndarray] = []
        source_ids: List[int] = []
        distances: List[float] = []
        transforms: List[str] = []
        match_costs: List[float] = []
        confidences: List[float] = []
        match_s = 0.0
        transfer_s = 0.0
        rejected = 0
        transform_names = ("identity", "mirror_x", "mirror_y", "transpose")
        for source_position in range(min(limit, len(retrieved.source_ids))):
            choices = []
            source_nodes = retrieved.node_features[source_position]
            for transform_order, transform in enumerate(transform_names):
                match_started = time.perf_counter()
                try:
                    result = match_blocks(
                        remap_boundary_node_features(source_nodes, transform),
                        target_features.node_matrix, max_cost=self.retrieval_max_cost,
                    )
                except Exception:
                    result = None
                match_s += time.perf_counter() - match_started
                if (result is not None and result.accepted
                        and math.isfinite(result.total_cost)):
                    choices.append((float(result.total_cost), transform_order, transform, result))
            if not choices:
                rejected += 1
                continue
            _cost, _order, transform, result = min(choices, key=lambda item: item[:2])
            transfer_started = time.perf_counter()
            try:
                prediction = transfer_layout(
                    retrieved.fp_xywh[source_position], result.target_to_source,
                    area, constraints, target_positions, transform,
                )
                fixed_or_preplaced = (constraints[:, 0] != 0) | (constraints[:, 1] != 0)
                has_shape = fixed_or_preplaced & (target_positions[:, 2] > 0) & (target_positions[:, 3] > 0)
                preplaced = ((constraints[:, 1] != 0) & (target_positions[:, 0] >= 0)
                             & (target_positions[:, 1] >= 0))
                if (prediction.shape != (n, 4) or not np.isfinite(prediction).all()
                        or not np.array_equal(prediction[has_shape, 2:4], target_positions[has_shape, 2:4])
                        or not np.array_equal(prediction[preplaced, :2], target_positions[preplaced, :2])):
                    raise ValueError("invalid transferred retrieval layout")
            except Exception:
                transfer_s += time.perf_counter() - transfer_started
                rejected += 1
                continue
            transfer_s += time.perf_counter() - transfer_started
            predictions.append(np.asarray(prediction, dtype=np.float64))
            source_ids.append(int(retrieved.source_ids[source_position]))
            distances.append(float(retrieved.distances[source_position]))
            transforms.append(transform)
            match_costs.append(float(result.total_cost))
            confidences.append(float(result.confidence))

        return CandidateBatch(
            "retrieval", predictions, max(0.0, time.perf_counter() - started),
            {"source_ids": source_ids, "retrieval_distances": distances,
             "transforms": transforms, "match_costs": match_costs,
             "match_confidences": confidences, "query_s": query_s,
             "match_s": match_s, "transfer_s": transfer_s, "rejected": rejected},
        )

    def _sample_portfolio_preds(self, n, at, cons, tpos, b2b, p2b, pins,
                                K, oversample: bool = True,
                                gen_seed: int = 0) -> List[np.ndarray]:
        """Use one source-neutral rank for the bounded Direct/retrieval portfolio."""
        retrieval_quota = min(
            max(0, int(K)), max(0, int(self.retrieval_slots)),
            FIRST_R4_RETRIEVAL_SLOTS,
        )
        if retrieval_quota <= 0 or self.retrieval_index is None:
            return self._sample_direct_preds(n, at, cons, tpos, b2b, p2b, pins,
                                             K, oversample, gen_seed=gen_seed)
        direct = self._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins,
                                               K, oversample, gen_seed=gen_seed)
        retrieved = self._sample_retrieval_preds(n, at, cons, tpos, b2b, p2b, pins, retrieval_quota)
        predictions = direct + retrieved.predictions
        sources = ["direct"] * len(direct) + ["retrieval"] * len(retrieved.predictions)
        order = self._rank_portfolio(predictions, n, at, cons, b2b)
        return _select_ranked_source_quota(predictions, sources, order, K, retrieval_quota)

    def _direct_worker(self, n, at, cons, tpos, b2b, p2b, pins,
                       deadline, box) -> None:
        """No-pool fallback: sample K layouts and push each through the
        legalize+refine glue in this process; results land in `box`."""
        try:
            K = 8 if (deadline - time.time()) > 15.0 else 4
            preds = self._sample_direct_preds(n, at, cons, tpos,
                                              b2b, p2b, pins, K)
            for rank, P in enumerate(preds):
                left = deadline - time.time()
                if left < 1.0:
                    break
                rect_list = [tuple(map(float, P[i])) for i in range(n)]
                opt = _ColumnOptimizer(rect_list, at, cons, tpos,
                                       b2b, p2b, pins, deadline,
                                       seed=31 + rank)
                sub = time.time() + max(left / 3.0, min(left, 4.0))
                out = refine_prediction(opt, P, min(sub, deadline),
                                        seed=41 + rank)
                if out is not None:
                    box.append((out, opt))
        except Exception:
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _pick_best(self, column_out: List[Rect], box) -> List[Rect]:
        """Choose among the column result and direct candidates under the
        same proxy score the restart pool uses (plus full cluster checks).

        PARTNER_PICK_EXACT_V=1 (default off) swaps the violation count fed
        into the ranking formula from `layout_refiner.full_violations` (a
        TOUCH_TOL=1e-7-slack, incomplete count) for
        `violation_killer._violations_exact` (the evaluator-exact
        boundary+grouping+MIB total).  Only the count changes -- the ranking
        formula, hpwl/area terms, and margin test are untouched.  Off path:
        no import, identical count source, byte-identical output.

        PARTNER_PICK_SCORE_REPAIRED=1 (default off): direct candidates are
        scored (hpwl/area/V) on their `_ensure_no_overlap`-repaired geometry
        instead of their raw pre-repair geometry, so the layout the proxy
        ranks is the layout that ships on a direct win (today the repair
        runs AFTER scoring, so the two can differ).  The column candidate is
        untouched -- it never goes through `_ensure_no_overlap` here.  On a
        direct win, the already-repaired list is returned without a second
        repair pass.  Off path: byte-identical (repair still runs once,
        after selection, exactly as today)."""
        try:
            scorer = box[0][1]
            score_repaired = bool(os.environ.get("PARTNER_PICK_SCORE_REPAIRED"))
            locked = None
            if score_repaired:
                locked = [scorer.kind[i] == 2 for i in range(scorer.n)]
            cands = [(np.array([list(r) for r in column_out]), column_out, False)]
            for out, _o in box:
                lst = [tuple(map(float, out[i])) for i in range(len(out))]
                if score_repaired:
                    lst = _ensure_no_overlap(lst, locked)
                    pos = np.asarray([list(r) for r in lst], dtype=np.float64)
                else:
                    pos = np.asarray(out, dtype=np.float64)
                cands.append((pos, lst, True))
            exact_v = bool(os.environ.get("PARTNER_PICK_EXACT_V"))
            if exact_v:
                from violation_killer import _violations_exact
            hps, areas, Vs = [], [], []
            for pos, _l, _d in cands:
                hps.append(scorer._hpwl(pos))
                areas.append(float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                                   * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min())))
                Vs.append(_violations_exact(scorer, pos) if exact_v
                          else full_violations(scorer, pos))
            hp_ref = max(min(hps), 1e-9)
            scores = []
            for i in range(len(cands)):
                scores.append(
                    (1.0 + 0.5 * ((hps[i] - hp_ref) / hp_ref
                                  + max(0.0, areas[i] / scorer.area_ref - 1.0)))
                    * math.exp(2.0 * Vs[i] / scorer.n_soft_den))
            if os.environ.get("REFINER_DEBUG"):
                for i in range(len(cands)):
                    pos_i = cands[i][0]
                    px0, py0 = pos_i[:, 0], pos_i[:, 1]
                    px1, py1 = px0 + pos_i[:, 2], py0 + pos_i[:, 3]
                    b = scorer._bnd_idx
                    bnd = 0
                    if len(b):
                        codes = scorer._bnd_codes
                        eps = 1e-6
                        bad = ((codes & 1) != 0) & (np.abs(px0[b] - px0.min()) >= eps)
                        bad |= ((codes & 2) != 0) & (np.abs(px1[b] - px1.max()) >= eps)
                        bad |= ((codes & 4) != 0) & (np.abs(py1[b] - py1.max()) >= eps)
                        bad |= ((codes & 8) != 0) & (np.abs(py0[b] - py0.min()) >= eps)
                        bnd = int(bad.sum())
                    print(f"[pick] cand{i} direct={cands[i][2]} "
                          f"hp={hps[i]:.1f} area={areas[i]:.0f} "
                          f"V={Vs[i]} (bnd={bnd} rest={Vs[i]-bnd}) "
                          f"score={scores[i]:.4f}", flush=True)
            # a direct candidate must beat the column result by a clear
            # margin — marginal swaps are proxy-noise coin flips
            best_i = 0
            for i in range(1, len(cands)):
                if scores[i] < scores[best_i] and scores[i] < scores[0] * PICK_MARGIN:
                    best_i = i
            pos, lst, is_direct = cands[best_i]
            if is_direct and not score_repaired:
                locked = [scorer.kind[i] == 2 for i in range(scorer.n)]
                lst = _ensure_no_overlap(lst, locked)
            # audit-only bookkeeping (PARTNER_VAUDIT_JSONL); a plain
            # attribute write, no effect on the returned layout
            self._last_pick_channel = "direct" if is_direct else "column"
            return lst
        except Exception:
            return column_out

    def _diffusion_refine(
        self,
        init_rects: List[Rect],
        area_targets: torch.Tensor,
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor,
        block_count: int,
    ) -> List[Rect]:
        at = area_targets.unsqueeze(0).to(self.device)
        cond = build_condition(
            at,
            b2b.unsqueeze(0).to(self.device),
            p2b.unsqueeze(0).to(self.device),
            pins.unsqueeze(0).to(self.device),
            constraints.unsqueeze(0).to(self.device),
            target_positions=target_positions.unsqueeze(0).to(self.device),
            relation_feat_dim=getattr(self.model.config, "relation_feat_dim", 0),
            node_feat_dim=getattr(self.model.config, "node_feat_dim", 13),
        )
        fp = torch.zeros((1, block_count, 4), dtype=torch.float32, device=self.device)
        for i, (x, y, w, h) in enumerate(init_rects):
            fp[0, i] = torch.tensor([w, h, x, y], dtype=torch.float32, device=self.device)
        z_repr = getattr(self.model.config, "z_repr", "xylogwh")
        z_init, _mask, _scale = fp_sol_to_z0(fp, at, z_repr=z_repr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.diffusion_seed)
        with torch.no_grad():
            z = ddim_refine(
                self.model,
                z_init,
                cond["node_feat"],
                cond["adj"],
                cond["mask"],
                start_t=self.refine_start_t,
                steps=max(1, int(self.steps)),
                schedule=self.schedule,
                generator=generator,
                rel_feat=cond.get("rel_feat"),
                z0_blend=0.20 if getattr(self.model.config, "predict_z0_head", False) else 0.0,
            )[0].cpu()
        return rectangles_from_z(z, area_targets, constraints, target_positions, z_repr=z_repr)


def _heuristic_init(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
) -> List[Rect]:
    """Pin/graph-weighted centroid layout used only as a relative-position
    seed for the diffusion model / column optimizer."""
    n = len(area_targets)
    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)

    shapes: List[Tuple[float, float]] = []
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            shapes.append((tw, th))
        else:
            area = max(float(area_targets[i]), 1e-6)
            shapes.append((math.sqrt(area), math.sqrt(area)))

    total_area = sum(w * h for w, h in shapes)
    scale = math.sqrt(max(total_area, 1.0))

    sx = [0.0] * n
    sy = [0.0] * n
    wsum = [0.0] * n
    n_pins = pins.shape[0]
    # PARTNER_FAST_SETUP: the two edge loops below are per-row `.item()`
    # calls over the PADDED connectivity tensors, so their cost tracks the
    # tensor height, not the instance -- 4.7 ms at n=25 up to 138 ms at
    # n=120, all of it serial in the parent before any worker sees the case.
    # The numpy twins are bit-exact (see `_pin_centroids_np`), and any
    # unexpected tensor layout falls back to the loops below.
    _fast = fast_setup_on()
    _got = _pin_centroids_np(p2b, pins, n, n_pins) if _fast else None
    if _got is not None:
        sx, sy, wsum = _got
    for edge in (() if _got is not None else p2b):
        if edge[0] == -1:
            continue
        p = int(edge[0].item())
        b = int(edge[1].item())
        if not (0 <= b < n and 0 <= p < n_pins):
            continue
        px, py = float(pins[p, 0]), float(pins[p, 1])
        if px == -1.0 or py == -1.0:
            continue
        w = max(float(edge[2].item()), 0.0)
        sx[b] += w * px
        sy[b] += w * py
        wsum[b] += w

    no_pin = [i for i in range(n) if wsum[i] <= 1e-9]
    cols = max(1, int(math.sqrt(max(len(no_pin), 1))))
    cursor = 0
    cx = [0.0] * n
    cy = [0.0] * n
    for i in range(n):
        if wsum[i] > 1e-9:
            cx[i] = sx[i] / wsum[i]
            cy[i] = sy[i] / wsum[i]
        else:
            row, col = divmod(cursor, cols)
            cx[i] = (col + 0.5) * scale / cols
            cy[i] = (row + 0.5) * scale / cols
            cursor += 1

    nx = list(cx)
    ny = list(cy)
    deg = [0.0] * n
    _got2 = _b2b_smooth_np(b2b, cx, cy, n) if _fast else None
    if _got2 is not None:
        nx, ny, deg = _got2
    for edge in (() if _got2 is not None else b2b):
        if edge[0] == -1:
            continue
        i = int(edge[0].item())
        j = int(edge[1].item())
        if not (0 <= i < n and 0 <= j < n):
            continue
        w = max(float(edge[2].item()), 0.0)
        nx[i] += 0.25 * w * cx[j]
        ny[i] += 0.25 * w * cy[j]
        nx[j] += 0.25 * w * cx[i]
        ny[j] += 0.25 * w * cy[i]
        deg[i] += 0.25 * w
        deg[j] += 0.25 * w
    fcx = [nx[i] / (1.0 + deg[i]) for i in range(n)]
    fcy = [ny[i] / (1.0 + deg[i]) for i in range(n)]

    out: List[Rect] = []
    for i in range(n):
        w, h = shapes[i]
        tx, ty, _tw, _th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0:
            out.append((tx, ty, w, h))
        else:
            out.append((fcx[i] - 0.5 * w, fcy[i] - 0.5 * h, w, h))
    return out


def _fallback_row(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
) -> List[Rect]:
    """Guaranteed-feasible fallback: preplaced blocks stay put; every other
    block is placed in a single row strictly to the right of everything."""
    n = len(area_targets)
    fixed, preplaced, _mib, _cluster, _boundary = _parse_constraints(constraints, n)
    out: List[Optional[Rect]] = [None] * n
    x_cursor = 0.0
    for i in range(n):
        tx, ty, tw, th = _target(target_positions, i)
        if preplaced[i] and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
            out[i] = (tx, ty, tw, th)
            x_cursor = max(x_cursor, tx + tw)
    x_cursor += 1.0
    for i in range(n):
        if out[i] is not None:
            continue
        tx, ty, tw, th = _target(target_positions, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            w, h = tw, th
        else:
            area = max(float(area_targets[i]), 1e-9)
            w = math.sqrt(area)
            h = area / w
        out[i] = (x_cursor, 0.0, w, h)
        x_cursor += w
    return [r for r in out]
