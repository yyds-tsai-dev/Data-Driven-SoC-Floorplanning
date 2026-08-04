"""Ablation timing of the numba layout kernel.

Each variant patches ONE construct in a scratch copy of the kernel and
re-times it.  Results are no longer bit-exact -- the point is only to find
where the compiled time goes.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "partner")
sys.path.insert(0, "tests")
sys.path.insert(0, "scratchpad")

SRC = pathlib.Path("partner/sa_numeric_kernel.py").read_text()

VARIANTS = {
    "baseline": [],
    "return after pos init": [
        ("    has_locked = lock.shape[0] > 0\n    x = 0.0\n    pb = 0",
         "    outf[0] = 0.0\n    outf[1] = 0.0\n    outf[2] = _OK\n    return\n"
         "    has_locked = lock.shape[0] > 0\n    x = 0.0\n    pb = 0")],
    "stack_column -> no-op": [
        ("    has_locked = lock.shape[0] > 0\n    n_occ = 0\n    if has_locked:",
         "    return 0, 0.0, _OK\n"
         "    has_locked = lock.shape[0] > 0\n    n_occ = 0\n    if has_locked:")],
    "place_unit_up -> no-op": [
        ("                   static_ent):\n    if ui[u, 0] == 0:",
         "                   static_ent):\n    return y0\n    if ui[u, 0] == 0:")],
    "chunk_up -> no-op": [
        ("def _chunk_up(pos, entf, enti, ents, e_lo, e_hi, xj, wj, y0, full_w, use_full):\n    y = y0",
         "def _chunk_up(pos, entf, enti, ents, e_lo, e_hi, xj, wj, y0, full_w, use_full):\n    return y0\n    y = y0")],
    "bisect 80->8 / 48->4": [("for _ in range(80):", "for _ in range(8):"),
                             ("for _ in range(48):", "for _ in range(4):")],
    "no widen-retry": [("while col_top > H * 1.0005 and tries < 3:",
                        "while False and col_top > H * 1.0005 and tries < 3:")],
    "no width obstacle loop": [("        if has_locked:\n            for _ in range(3):",
                                "        if False and has_locked:\n            for _ in range(3):")],
    "no right-align pass": [("    if last >= 0:", "    if False and last >= 0:")],
    "no top-lift pass": [("        if (blki[top_block, 1] & 4) == 0:\n            continue",
                          "        if True or (blki[top_block, 1] & 4) == 0:\n            continue")],
    "no band plan (unit_h only)": [
        ("    _band_plan(u, w, uf, ui, bandi, chunkf, chunki, entf, enti,\n"
         "               hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp)\n"
         "    return hcf[u, 1]",
         "    return uf[u, 1] + uf[u, 0] / w")],
}


def build(name, patches):
    s = SRC.replace("@njit(cache=True)", "@njit(cache=True)")
    for a, b in patches:
        assert a in s, f"{name}: pattern not found: {a[:60]!r}"
        s = s.replace(a, b)
    mod = f"kvar_{abs(hash(name)) % 100000}"
    pathlib.Path(f"scratchpad/{mod}.py").write_text(s)
    return importlib.import_module(mod)


def time_variant(mod, case):
    import column_sa_legalizer as lg
    from synth_instances import build_instance
    sys.modules["sa_numeric_kernel"] = mod
    inst = build_instance(**case)
    os.environ["PARTNER_SA_KERNEL"] = "numba"
    opt = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1e4, seed=case["seed"])
    os.environ.pop("PARTNER_SA_KERNEL", None)
    # a FIXED column state for every variant: annealing under an ablated
    # kernel would land each variant on a different layout and confound the
    # comparison.
    opt.prepare()
    cols = opt._init_columns(10)
    k = opt._sa_kernel
    k.layout(cols)
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        k.layout(cols)
    return (time.perf_counter() - t0) / reps


case = dict(n=100, seed=0)
base = None
for name, patches in VARIANTS.items():
    mod = build(name, patches)
    dt = time_variant(mod, case)
    if base is None:
        base = dt
    print(f"{name:30s} {dt*1e6:8.1f} us   ({100*dt/base:5.1f}% of baseline)")
