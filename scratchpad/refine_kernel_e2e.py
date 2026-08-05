"""End-to-end fidelity + speed for PARTNER_REFINE_KERNEL.

`refine_prediction` on synthetic instances with a span long enough that every
stage converges (so the Python path is self-reproducible -- verified by
running it twice), kernel off vs on, layouts compared by sha1.

Usage:  uv run python scratchpad/refine_kernel_e2e.py
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "partner", ROOT / "tests"):
    sys.path.insert(0, str(p))
from synth_instances import build_instance, make_optimizer  # noqa: E402


def run(kernel, n, seed, span):
    if kernel:
        os.environ["PARTNER_REFINE_KERNEL"] = "numba"
    else:
        os.environ.pop("PARTNER_REFINE_KERNEL", None)
    import layout_refiner as mod
    inst = build_instance(n=n, seed=seed)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    dl = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=dl)
    t0 = time.time()
    out = mod.refine_prediction(opt, pred, dl, seed=11)
    el = time.time() - t0
    if out is None:
        return "NONE", el
    a = np.asarray(out, dtype=np.float64)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16], el


def main():
    ok = True
    for n, seed, span in ((40, 0, 20.0), (40, 1, 20.0), (70, 0, 25.0),
                          (70, 1, 25.0), (100, 0, 30.0), (100, 1, 30.0)):
        h1, t1 = run(False, n, seed, span)
        h2, t2 = run(False, n, seed, span)
        h3, t3 = run(True, n, seed, span)
        stable = h1 == h2
        match = h1 == h3
        ok &= stable and match
        print(f"n={n:3d} seed={seed} span={span:4.1f}: "
              f"py={h1} py2={h2} nb={h3} stable={stable} match={match} "
              f"| t_py={t1:6.2f}s t_nb={t3:6.2f}s x{t1 / max(t3, 1e-9):4.1f}")
    print(f"\nALL OK = {ok}")


if __name__ == "__main__":
    main()
