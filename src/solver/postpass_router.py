"""Feature router and shared time budget for the final post-processing passes.

Replaces the three independent blanket block-count gates
(``PARTNER_FINAL_TOPOLOGY_MIN_N/MAX_N``, ``PARTNER_FINAL_VKILL_MIN_N/MAX_N``,
``PARTNER_SECOND_COORD_POLISH_MIN_N/MAX_N``) with one per-case decision that
allocates a single, time-neutral reserve across the three stages.

Evidence the design is built on
-------------------------------
Eight same-base replay pairs (public + Shadow Hidden v3, two different base
solvers, ~180 changed cases) were re-scored with the official evaluator:

* **Every pass is quality-monotone.**  Zero net losers in every pair.  Each
  pass only returns a layout whose evaluator-shaped proxy strictly improved
  with non-increasing soft violations and all hard guards re-checked, and that
  acceptance test agrees with the official no-runtime cost.  There is therefore
  no population of "net loser" cases for a router to steer away from, and a
  router cannot improve the no-runtime score.  What it can do is stop spending
  time that buys nothing.

* **The block-count band is a good *value* filter.**  Under the official
  λ ∝ e^(n/12) case weighting, cases with n<95 contributed 4.4% of the
  violation-repair pass's total Shadow-v3 gain while being 15 of the 22 cases
  it changed.  Widening the band downward buys ~4% more quality for ~2x the
  runtime.  Keep n>=95.

* **Appending the passes loses on the official score.**  Re-scoring the public
  replays with the beta-hidden field median runtimes
  (``C_median_runtimes_beta_hidden_update.csv``) and the official
  ``max(0.7, RuntimeFactor^0.3)`` term: appending final-topology costs
  ``+0.0060`` and appending violation-repair + second polish costs ``+0.0115``
  on the runtime-aware weighted total, against ``-0.0020`` / ``-0.0009``
  no-runtime gains.  A *perfect oracle* router that ran each pass only on the
  cases where it strictly helps still loses (``+0.0013`` / ``+0.0062``),
  because the productive cases' own runtime outweighs their own gain.  Only a
  time-neutral integration wins: with the per-case wall clock held fixed the
  same arms are ``-0.0014`` / ``-0.0006``.

Hence the load-bearing properties of this router:

1. **Per-stage hard time boxes under one per-case reserve cap.**  A stage that
   finishes early may not hand its slack to a later stage — the first receipt
   caught the second polish absorbing the topology pass's leftovers, spending
   1.0 s across 16 cases, and changing zero layouts.
2. **Only the stage whose measured NET is negative runs by default.**  The
   quality half of that NET comes from the receipt's own paired layout
   streams; the runtime half from its own per-stage elapsed times priced
   through the official runtime term.  Neither half needs an A/B run.
3. **Appending, not carving, by default.**  Carving the reserve out of the SA
   deadline is time-neutral, but it moves the cost into the SA's search
   quality, where it is invisible to a same-call receipt and is buried under a
   full-run spread (~0.03-0.06 on Shadow v3, driven by a couple of bimodal
   hard cases) far larger than the effect.  Appending puts the whole cost in
   a term this module can measure exactly.  ``PARTNER_ROUTE_CARVE=1`` restores
   the time-neutral form for anyone who measures the carve cost separately.

Every gate reads reusable instance / layout statistics (block count, the
layout's own soft-violation count, whether an upstream stage modified the
layout).  No validation ``test_id`` is consulted anywhere.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

TOPOLOGY = "final_topology"
VKILL = "final_vkill"
POLISH = "second_polish"

_STAGES = (TOPOLOGY, VKILL, POLISH)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def case_weight_share(block_count: int, max_n: int = 120) -> float:
    """The evaluator's own unnormalised case weight, λ_i ∝ e^(n_i/12)."""
    return math.exp((float(block_count) - float(max_n)) / 12.0)


class PostPassRouter:
    """Per-case plan for the three final post-processing stages.

    Constructed once at the top of ``solve()``, before the deadline is used,
    so the reserve can be carved from the SA budget rather than appended to
    the case's wall clock.
    """

    __slots__ = ("block_count", "budget", "reserve", "in_band", "carve",
                 "start", "headroom_s", "_remaining", "_box", "_skips")

    def __init__(self, block_count: int, budget: float,
                 start: Optional[float] = None) -> None:
        self.block_count = int(block_count)
        self.budget = max(0.0, float(budget))
        self.start = float(start) if start is not None else time.time()
        self._skips = {}

        # Runtime-headroom gate.  The official runtime term is
        # max(0.7, RuntimeFactor^0.3), which is FLAT for
        # RuntimeFactor <= 0.7^(1/0.3) = 0.3006 — below that point extra
        # milliseconds are exactly free, above it they are charged at
        # 0.3 * f * dt/t.  The published beta field medians
        # (C_median_runtimes_beta_hidden_update.csv) for n>=95 run 2.71-5.47 s,
        # so our own runtime stops being free at roughly
        # 0.3006 * 2.7 = 0.81 s using the most conservative in-band median.
        # A stage therefore runs only while the case's own elapsed time is
        # still inside that flat region.  Measured effect of the gate at 0.9 s
        # (exact paired quality from the receipt streams, exact runtime from
        # its per-stage elapsed times):
        #
        #   public  ungated NET +0.00033  ->  gated NET -0.00059
        #   ShadowV3 ungated NET -0.00523 ->  gated NET -0.00634
        #
        # i.e. the gate turns a public regression into a public improvement
        # AND improves Shadow v3, because it drops exactly the spend that the
        # runtime term was charging for.  0 disables the gate.
        self.headroom_s = _envf("PARTNER_ROUTE_HEADROOM_S", 0.9)

        lo = _envi("PARTNER_ROUTE_MIN_N", 95)
        hi = _envi("PARTNER_ROUTE_MAX_N", 100000)
        self.in_band = lo <= self.block_count <= hi

        # Per-stage time boxes, in seconds.  These are *caps*, not
        # allocations: each pass has its own early exit and normally returns
        # far under its box (measured on Shadow v3, n>=95: violation repair
        # 17 ms against a 200 ms box).  The caps exist to bound the tail.
        #
        # A stage with box 0 is off.  Only violation repair ships on; the
        # other two are off by default with their boxes preserved as opt-ins.
        # Three exactly-paired same-call reps on Shadow v3 (quality delta from
        # the receipt's own layout streams, runtime delta from its per-stage
        # elapsed times priced through the official max(0.7, R^0.3) term
        # against the beta-hidden field medians):
        #
        #   violation repair  quality -0.00604   runtime +0.00112   NET -0.00493
        #   discrete topology quality -0.00452   runtime +0.00428   NET -0.00024
        #   second polish     quality  0.00000   runtime +0.00460   NET +0.00460
        #
        # The second polish changed ZERO layouts in all three reps: at n>=110
        # it needs roughly the first polish's own 600 ms box to make progress.
        # The topology pass earns real quality but spends 0.06 s on all 26
        # in-band cases to get it, which the runtime term takes straight back;
        # it is a wash (1 of 3 reps net-positive) and stays off until either a
        # cheaper bound or a time-neutral carve with a measured carve cost is
        # available.
        self._box = {
            TOPOLOGY: _envf("PARTNER_ROUTE_BOX_TOPOLOGY", 0.0),
            VKILL: _envf("PARTNER_ROUTE_BOX_VKILL", 0.20),
            POLISH: _envf("PARTNER_ROUTE_BOX_POLISH", 0.0),
        }

        # The reserve caps the WHOLE post-pass tail for this case.
        if not self.in_band:
            self.reserve = 0.0
        else:
            frac = _envf("PARTNER_ROUTE_RESERVE_FRAC", 0.35)
            cap = _envf("PARTNER_ROUTE_RESERVE_MAX", 0.28)
            floor = _envf("PARTNER_ROUTE_RESERVE_MIN", 0.06)
            self.reserve = min(cap, max(floor, frac * self.budget))
        self._remaining = self.reserve

        # PARTNER_ROUTE_CARVE=1 takes the reserve out of the SA deadline
        # (time-neutral, but the SA then searches a shorter budget and the
        # cost of that is NOT measurable from a same-call receipt).  Default
        # 0 appends instead: the runtime cost is then exactly measurable from
        # the receipt's own per-stage elapsed times, which is what lets this
        # change be promoted on evidence rather than on an A/B whose spread
        # (~0.05 on Shadow v3, driven by two bimodal hard cases) is larger
        # than the effect being measured.
        self.carve = os.environ.get("PARTNER_ROUTE_CARVE", "0") not in (
            "", "0", "false", "False", "no")

    # -- reserve accounting -------------------------------------------------

    @property
    def remaining(self) -> float:
        return max(0.0, self._remaining)

    def spend(self, elapsed: float) -> None:
        self._remaining = max(0.0, self._remaining - max(0.0, float(elapsed)))

    @property
    def sa_reserve(self) -> float:
        """Seconds to subtract from the SA deadline (0 unless carving)."""
        return self.reserve if self.carve else 0.0

    def budget_for(self, stage: str) -> float:
        """Seconds this stage may use: its own box, capped by what is left.

        A per-stage box rather than a share of the reserve, so a stage that
        finishes early cannot hand its slack to a later stage that has no
        demonstrated use for it — the failure the first receipt caught.
        """
        if not self.in_band:
            return 0.0
        return max(0.0, min(self._remaining, self._box.get(stage, 0.0)))

    # -- per-stage gates ----------------------------------------------------

    def allow(self, stage: str, *, soft_violations: Optional[int] = None,
              layout_dirty: Optional[bool] = None,
              first_polish_truncated: bool = False) -> Tuple[bool, str]:
        """Decide whether `stage` may run, and why not when it may not.

        `soft_violations` — the current layout's own boundary+grouping+MIB
        count (violation-repair precondition).  `layout_dirty` — whether any
        stage modified the layout since the first coordinate polish
        (second-polish precondition).
        """
        if not self.in_band:
            return False, "out_of_value_band"
        if self._box.get(stage, 0.0) <= 0.0:
            return False, "stage_disabled"
        if (self.headroom_s > 0.0 and not self.carve
                and (time.time() - self.start) > self.headroom_s):
            # Past the flat region of the official runtime term: from here on
            # every millisecond is charged, and no measured post-pass gain
            # covers that charge.  (A carving router is time-neutral and does
            # not face this trade, so the gate does not apply there.)
            return False, "no_runtime_headroom"
        if self.budget_for(stage) <= _envf("PARTNER_ROUTE_MIN_SLICE", 0.008):
            return False, "reserve_exhausted"

        if stage == VKILL:
            # The pass repairs boundary/grouping/MIB violations.  With none
            # present it has nothing to act on: across every same-base replay
            # measured, 21 of the 22 cases it ever changed had V>0 and the
            # single V=0 case moved the weighted total by 3e-7.
            if (soft_violations is not None
                    and soft_violations <= 0
                    and not os.environ.get("PARTNER_ROUTE_VKILL_ALWAYS")):
                return False, "no_soft_violations"

        if stage == POLISH:
            # `polish_layout` is deterministic given its input.  If nothing has
            # modified the layout since the first polish and that first polish
            # was not cut short by its own box, a second call can only redo
            # work that already converged.
            if (layout_dirty is False and not first_polish_truncated
                    and not os.environ.get("PARTNER_ROUTE_POLISH_ALWAYS")):
                return False, "layout_unchanged_since_first_polish"

        return True, "ok"

    def note_skip(self, stage: str, reason: str) -> None:
        self._skips[stage] = reason

    @property
    def skips(self):
        return dict(self._skips)
