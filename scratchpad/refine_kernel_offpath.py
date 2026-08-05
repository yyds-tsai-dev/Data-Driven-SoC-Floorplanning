"""Off-path fidelity for PARTNER_REFINE_KERNEL: the pre-port `layout_refiner`
vs the dispatch-patched one, flag OFF, on spans long enough that every stage
converges (so the pristine module is self-reproducible) -> identical layouts.

Same shape as `scratchpad/anytime_bitexact.py`.  Expects a pristine copy at
`scratchpad/oldref2/layout_refiner.py` (see the header of this file for how to
produce it):

    git show <base>:partner/layout_refiner.py > scratchpad/oldref2/layout_refiner.py

Usage:  uv run python scratchpad/refine_kernel_offpath.py
"""

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "partner", ROOT / "tests"):
    sys.path.insert(0, str(p))
from synth_instances import build_instance, make_optimizer  # noqa: E402

os.environ.pop("PARTNER_REFINE_KERNEL", None)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old = load(ROOT / "scratchpad/oldref2/layout_refiner.py", "layout_refiner")
new = load(ROOT / "partner/layout_refiner.py", "layout_refiner_new")


def run(mod, n, seed, span):
    inst = build_instance(n=n, seed=seed)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    dl = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=dl)
    out = mod.refine_prediction(opt, pred, dl, seed=11)
    if out is None:
        return "NONE"
    a = np.asarray(out, dtype=np.float64)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


ok = True
for n, seed, span in ((40, 0, 20.0), (40, 1, 20.0), (70, 0, 25.0),
                      (70, 1, 25.0)):
    h1 = run(old, n, seed, span)
    h2 = run(old, n, seed, span)
    h3 = run(new, n, seed, span)
    stable = h1 == h2
    match = h1 == h3
    ok &= stable and match
    print(f"n={n} seed={seed} span={span}: old={h1} old2={h2} new={h3} "
          f"stable={stable} match={match}")
print(f"\nOFF-PATH ALL OK = {ok}")
