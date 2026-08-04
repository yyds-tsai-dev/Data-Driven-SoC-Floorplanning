"""Cold JIT compile time vs inline configuration.

The official machine installs requirements.txt and runs; it will NOT have a
warm numba on-disk cache, so cold compile time is a real per-run cost that has
to be paid out of (or before) the solve budget.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, "partner")
sys.path.insert(0, "tests")
sys.path.insert(0, "scratchpad")

SRC = pathlib.Path("partner/sa_numeric_kernel.py").read_text()
ALL = ["_interval_free", "_add_interval", "_obstacles_in", "_band_strip",
       "_solve_band", "_band_plan", "_unit_h", "_chunk_up", "_place_unit_up",
       "_chunk_down", "_place_unit_down"]

CONFIGS = {
    "none": [],
    "tiny leaves": ["_interval_free", "_chunk_up", "_chunk_down"],
    "tiny + unit_h": ["_interval_free", "_chunk_up", "_chunk_down", "_unit_h"],
    "all": ALL,
}

name = sys.argv[1]
inline_set = CONFIGS[name]

s = SRC
for f in ALL:
    base = f"@njit(cache=True, inline='always')\ndef {f}("
    plain = f"@njit(cache=True)\ndef {f}("
    cur = base if base in s else plain
    want = base if f in inline_set else plain
    s = s.replace(cur, want, 1)
mod = "cc_" + name.replace(" ", "_").replace("+", "p")
pathlib.Path(f"scratchpad/{mod}.py").write_text(s)
shutil.rmtree("scratchpad/__pycache__", ignore_errors=True)

import column_sa_legalizer as lg  # noqa: E402
from synth_instances import build_instance  # noqa: E402

m = importlib.import_module(mod)
sys.modules["sa_numeric_kernel"] = m

inst = build_instance(n=100, seed=0)
os.environ["PARTNER_SA_KERNEL"] = "numba"
os.environ["PARTNER_SA_KERNEL_WARMUP"] = "0"
opt = lg._ColumnOptimizer(
    inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
    inst.b2b, inst.p2b, inst.pins, time.time() + 1e5, seed=0)
cols = opt._init_columns(10)

t0 = time.perf_counter()
opt._layout(cols)
t_layout = time.perf_counter() - t0
t0 = time.perf_counter()
opt._violations(opt._layout(cols)[0])
t_viol = time.perf_counter() - t0

reps = 2000
t0 = time.perf_counter()
for _ in range(reps):
    opt._layout(cols)
steady = (time.perf_counter() - t0) / reps

print(f"{name:16s} cold-compile _layout {t_layout:8.2f} s  "
      f"_violations {t_viol:6.2f} s  steady {steady*1e6:7.1f} us")
