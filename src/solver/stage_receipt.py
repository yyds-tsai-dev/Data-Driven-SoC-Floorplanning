"""Probe-only same-call receipt for the final post-processing stages.

Why this exists
---------------
The post-passes (`_final_topology_refine`, `_final_quality_pass`'s violation
repair and second coordinate polish) each carve a reserve out of the case
deadline.  Comparing two *independent* full runs — one with the pass, one
without — therefore never isolates the pass: the two runs' SA searches saw
different wall-clock budgets and produced different base layouts.  Every
"online integration" verdict built that way is confounded, and at least one
recorded verdict in this repo was
(`docs/experiments/2026-08-24-final-topology-postprocess.md`).

This module records the layout *before and after each stage of one and the
same `solve()` call*, plus the elapsed time each stage consumed and the
routing features that were visible when the routing decision was taken.
Scoring those layout streams offline with the official evaluator yields an
exactly paired, zero-variance per-case delta for each stage.

Contract
--------
- Entirely opt-in: activated only by ``PARTNER_STAGE_RECEIPT=<path>``.  With
  the variable unset every entry point here is a single dict lookup and the
  production layout is byte-identical.
- Nothing is serialised inside ``solve()``.  Snapshots are appended to a
  process-global list and written once at interpreter exit, so the evaluator's
  per-case timing is charged only for a few list appends.
- Never raises.  Any failure disables the receipt for the rest of the process.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

_RECORDS: List[Dict[str, Any]] = []
_REGISTERED = False
_DISABLED = False


def enabled() -> bool:
    """True when the probe receipt is armed for this process."""
    return (not _DISABLED) and bool(os.environ.get("PARTNER_STAGE_RECEIPT"))


def _flush() -> None:
    path = os.environ.get("PARTNER_STAGE_RECEIPT")
    if not path or not _RECORDS:
        return
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"records": _RECORDS}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


class CaseReceipt:
    """Accumulates the stage-by-stage layout stream of one `solve()` call."""

    __slots__ = ("block_count", "case_index", "stages", "features", "_t0")

    def __init__(self, case_index: int, block_count: int) -> None:
        self.case_index = int(case_index)
        self.block_count = int(block_count)
        self.stages: List[Dict[str, Any]] = []
        self.features: Dict[str, Any] = {}
        self._t0 = time.time()

    def note(self, **features: Any) -> None:
        """Record routing features observed at decision time."""
        try:
            self.features.update(features)
        except Exception:
            pass

    def stage(self, name: str, before: Sequence, after: Sequence,
              elapsed: float, ran: bool, reason: str = "") -> None:
        """Record one stage's paired before/after layout."""
        try:
            b = [[float(v) for v in row] for row in before]
            a = [[float(v) for v in row] for row in after]
            self.stages.append({
                "name": name,
                "ran": bool(ran),
                "reason": reason,
                "elapsed": float(elapsed),
                "changed": a != b,
                "before": b,
                "after": a,
            })
        except Exception:
            pass

    def close(self) -> None:
        global _REGISTERED, _DISABLED
        if _DISABLED:
            return
        try:
            _RECORDS.append({
                "case_index": self.case_index,
                "block_count": self.block_count,
                "solve_elapsed": time.time() - self._t0,
                "features": self.features,
                "stages": self.stages,
            })
            if not _REGISTERED:
                atexit.register(_flush)
                _REGISTERED = True
        except Exception:
            _DISABLED = True


def open_case(case_index: int, block_count: int) -> Optional[CaseReceipt]:
    """Return a receipt for this case, or None when the probe is off."""
    if not enabled():
        return None
    try:
        return CaseReceipt(case_index, block_count)
    except Exception:
        return None
