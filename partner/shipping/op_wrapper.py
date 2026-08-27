"""ICCAD 2026 submission entry point (cadc1013).

Operating point 2026-08-27 (final-sprint-0827): mid budget table + Flow-only
+ RES_FRAC 0.45 + WALL_REPAIR; official 1.09-1.10, see
docs/experiments/2026-08-21-post-beta-p0-execution.md section 15-16.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_BUDGET_TABLE_MID = "0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.239,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.302,0.308,0.314,0.050,0.327,0.050,0.340,0.346,0.353,0.050,0.050,0.374,0.050,0.050,0.050,0.050,0.050,0.420,0.429,0.437,0.050,0.454,0.463,0.472,0.482,0.491,0.501,0.050,0.324,0.531,0.334,0.339,0.563,0.349,0.585,0.359,0.608,0.370,0.632,0.380,0.386,0.391,0.397,0.697,0.710,0.724,0.419,0.753,0.431,0.437,0.798,0.448,0.454,0.460,0.418,0.454,0.493,0.536,0.583,0.634,0.689,0.748,0.813,0.884,0.961,1.068,1.135,1.110,1.220,1.154,1.220,1.220,1.439"

for k, v in {
    # Checkpoints: DIRECT_CKPT is the v2 student (s2, 20k) checkpoint;
    # FLOW_CKPT is the flow-matching v1 final checkpoint.
    "DIRECT_CKPT": str(HERE / "checkpoints" / "direct_v2_student_s2.pt"),
    "FLOW_CKPT": str(HERE / "checkpoints" / "flow_matching_v1_final.pt"),
    "VKILL_OFF": "1",
    "PARTNER_POOL": "24",
    "PARTNER_NREF": "9",
    "PARTNER_NREF_MIN_N": "95",
    "PARTNER_DIRECT_MIN": "0.3",
    "PARTNER_DIRECT_SEAT_FIX": "1",
    "PARTNER_OVERSAMPLE": "1",
    "PARTNER_KS_CAP": "6",
    "PARTNER_DIRECT_SOLVER": "dpmpp",
    "PARTNER_DDIM_STEPS": "2",
    "PARTNER_FLOW_SOLVER": "euler",
    "PARTNER_FLOW_STEPS": "8",
    "PARTNER_FLOW_ANTITHETIC": "1",
    "PARTNER_FLOW_SLOTS": "10",
    "PARTNER_PRESCREEN_V": "1",
    "PARTNER_TAG_ANCHOR_EXTRA": "3",
    # Budget curve: mid budget table (per-n seconds, n=21..120).
    "PARTNER_BUDGET_SCALE": "8.498e-5",
    "PARTNER_BUDGET_TAU": "12",
    "PARTNER_BUDGET_MIN": "0.05",
    "PARTNER_BUDGET_MAX": "1.22",
    "PARTNER_POOL_GATE": "0",
    "PARTNER_REFINE_STALL_STOP": "1",
    # Numba kernels (bit-exact vs Python paths; JIT warmed pre-fork in
    # __init__ pool warmup, first-case guard). numba and scipy are provided
    # by the evaluation environment (Beta guidelines Section 2, Case A) --
    # kernels fall back to the Python path if numba is absent.
    "PARTNER_SA_KERNEL": "numba",
    "PARTNER_REFINE_KERNEL": "numba",
    "PARTNER_REFINE_FASTBUILD": "1",
    "PARTNER_FAST_SETUP": "1",
    "PARTNER_EDGE_SEAT_V2": "1",
    "PARTNER_FRAME_WPIN": "1",
    "PARTNER_FRAME_SCALE_LADDER": "1",
    "PARTNER_FRAME_SCALE_SET": "1.02",
    "PARTNER_SEAT_FINAL": "1",
    "PARTNER_TAG_COMPRESS": "1",
    "PARTNER_GROUP_BRIDGE": "1",
    # PARTNER_COORD_POLISH deliberately unset: raw -0.014 but +0.25 s post-deadline
    # on every n>=95 case -> runtime-aware total worse in every scenario (2026-08-27).
    "PARTNER_GPU_ARM": "0",
    "PARTNER_RETRIEVAL_SLOTS": "0",
    "PARTNER_REFINE_RES_FRAC": "0.45",
    "PARTNER_WALL_REPAIR": "1",
    "PARTNER_FLOW_WARM": "1",
    "PARTNER_EARLY_EXIT": "1",
    "PARTNER_REFINE_SECURE_FALLBACK": "1",
    "PARTNER_BUDGET_TABLE": _BUDGET_TABLE_MID,
}.items():
    os.environ.setdefault(k, v)

from op_src import MyOptimizer as ContestOptimizer  # noqa: E402


def _warm_jit_kernels() -> None:
    """Compile (or cache-load) both numba kernel sets in the untimed
    module-load window.  Without this, a fresh deployment path invalidates
    the numba disk cache and the full JIT compile (~10s measured) lands in
    the FIRST CASE's runtime -- the exact first-case cliff the repack
    verification of 2026-08-06 caught.  Contained: any failure falls back
    to lazy in-case compilation, which is slow but correct."""
    try:
        from refine_numeric_kernel import warm_process
        warm_process()
    except Exception:
        pass
    try:
        import time as _time
        import column_sa_legalizer as _lg
        import sa_numeric_kernel as _snk
        from synth_instances import build_instance as _build_instance
        _inst = _build_instance(n=21, seed=19, n_clusters=1, n_mib=1)
        _opt = _lg._ColumnOptimizer(
            _inst.rects, _inst.area_targets, _inst.constraints,
            _inst.target_positions, _inst.b2b, _inst.p2b, _inst.pins,
            _time.time() + 30.0, seed=0)
        _snk.try_attach(_opt)  # builds the kernel and runs its warmup()
    except Exception:
        pass


if os.environ.get("PARTNER_JIT_WARM", "1") in ("1", "true", "True", "yes"):
    _warm_jit_kernels()

MyOptimizer = ContestOptimizer
