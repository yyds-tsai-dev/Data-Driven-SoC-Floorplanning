"""Evaluator wrapper for the default-off column split/merge G0.

It reproduces the shipped 0.3 s operating point while importing solver modules
from this worktree.  `DIRECT_CKPT` and `FLOW_CKPT` are supplied by the runner;
`PARTNER_COL_LNS_ORACLE=1` selects the treatment arm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "solver"))
sys.path.insert(0, str(ROOT / "tests"))

for key, value in {
    "VKILL_OFF": "1",
    "PARTNER_PRESCREEN_V": "1",
    "PARTNER_OVERSAMPLE": "4",
    "PARTNER_TAG_ANCHOR_EXTRA": "3",
    "PARTNER_BUDGET_SCALE": "8.498e-5",
    "PARTNER_BUDGET_TAU": "12",
    "PARTNER_BUDGET_MIN": "0.05",
    "PARTNER_BUDGET_MAX": "1.22",
    "PARTNER_POOL_GATE": "0",
    "PARTNER_DIRECT_SOLVER": "dpmpp",
    "PARTNER_DDIM_STEPS": "2",
    "PARTNER_DIRECT_MIN": "0.3",
    "PARTNER_NREF": "6",
    "PARTNER_REFINE_STALL_STOP": "1",
    "PARTNER_SA_KERNEL": "numba",
    "PARTNER_REFINE_KERNEL": "numba",
    "PARTNER_REFINE_FASTBUILD": "1",
    "PARTNER_FAST_SETUP": "1",
    "PARTNER_FLOW_SOLVER": "euler",
    "PARTNER_FLOW_SLOTS": "10",
    "PARTNER_FLOW_STEPS": "8",
    "PARTNER_FLOW_ANTITHETIC": "1",
    "PARTNER_EDGE_SEAT_V2": "1",
    "PARTNER_FRAME_WPIN": "1",
    "PARTNER_COORD_POLISH": "1",
    "PARTNER_FRAME_SCALE_LADDER": "1",
    "PARTNER_FRAME_SCALE_SET": "1.02",
    "PARTNER_SEAT_FINAL": "1",
}.items():
    os.environ.setdefault(key, value)

from contest_optimizer import MyOptimizer as ContestOptimizer  # noqa: E402


def _warm_jit_kernels() -> None:
    try:
        from refine_numeric_kernel import warm_process

        warm_process()
    except Exception:
        pass
    try:
        import time

        import column_sa_legalizer as legalizer
        import sa_numeric_kernel
        from synth_instances import build_instance

        inst = build_instance(n=21, seed=19, n_clusters=1, n_mib=1)
        opt = legalizer._ColumnOptimizer(
            inst.rects,
            inst.area_targets,
            inst.constraints,
            inst.target_positions,
            inst.b2b,
            inst.p2b,
            inst.pins,
            time.time() + 30.0,
            seed=0,
        )
        sa_numeric_kernel.try_attach(opt)
    except Exception:
        pass


if os.environ.get("PARTNER_JIT_WARM", "1") in ("1", "true", "True", "yes"):
    _warm_jit_kernels()


MyOptimizer = ContestOptimizer
