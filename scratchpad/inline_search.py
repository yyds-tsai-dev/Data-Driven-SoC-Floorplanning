"""Search which kernel functions should carry inline='always'.

Numba's cost model here is not "calls are expensive" (they are ~2 ns) but
"a callee that contains a large cold branch forces a heavy prologue on every
call, and force-inlining a *huge* callee poisons the caller's loop".  So the
right set is: inline the small hot leaves, leave the heavy ones as real calls.
"""

from __future__ import annotations

import importlib
import os
import pathlib
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
    "all": ALL,
    "all + fastmath": ALL,
    "all + no-error-model": ALL,
}

CASES = [dict(n=40, seed=3), dict(n=100, seed=0), dict(n=120, seed=7),
         dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
              frac_boundary=0.30)]


def build(name, inline_set):
    s = SRC
    for f in ALL:
        base = f"@njit(cache=True, inline='always')\ndef {f}("
        plain = f"@njit(cache=True)\ndef {f}("
        cur = base if base in s else plain
        want = base if f in inline_set else plain
        assert cur in s, f
        s = s.replace(cur, want, 1)
    s = s.replace("cache=True", "cache=False")
    if "fastmath" in name:
        s = s.replace("njit(cache=False", "njit(fastmath=True, cache=False")
    if "no-error-model" in name:
        s = s.replace("njit(cache=False", "njit(error_model='numpy', cache=False")
    mod = "inl_" + str(abs(hash(name)) % 100000)
    pathlib.Path(f"scratchpad/{mod}.py").write_text(s)
    return importlib.import_module(mod)


def measure(mod, case):
    import column_sa_legalizer as lg
    from synth_instances import build_instance
    sys.modules["sa_numeric_kernel"] = mod
    inst = build_instance(**case)
    os.environ["PARTNER_SA_KERNEL"] = "numba"
    opt = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1e4, seed=case["seed"])
    os.environ.pop("PARTNER_SA_KERNEL", None)
    opt.prepare()
    cols = opt._init_columns(10)
    k = opt._sa_kernel
    k.layout(cols)
    reps = 2000
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            k.layout(cols)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def python_ref(case):
    import column_sa_legalizer as lg
    from synth_instances import build_instance
    inst = build_instance(**case)
    os.environ.pop("PARTNER_SA_KERNEL", None)
    opt = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1e4, seed=case["seed"])
    opt.prepare()
    cols = opt._init_columns(10)
    opt._layout_full(cols)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(2000):
            opt._layout_full(cols)
        best = min(best, (time.perf_counter() - t0) / 2000)
    return best


refs = {i: python_ref(c) for i, c in enumerate(CASES)}
hdr = "  ".join(f"n{c['n']}s{c['seed']}" for c in CASES)
print(f"{'config':32s} {hdr}   (speedup vs python)")
print(f"{'python _layout_full (us)':32s} " +
      "  ".join(f"{refs[i]*1e6:8.1f}" for i in range(len(CASES))))
for name, iset in CONFIGS.items():
    mod = build(name, iset)
    outs = []
    for i, c in enumerate(CASES):
        dt = measure(mod, c)
        outs.append(f"{refs[i]/dt:8.2f}x")
    print(f"{name:32s} " + "  ".join(outs))
