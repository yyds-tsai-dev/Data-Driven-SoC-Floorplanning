"""Off-path fidelity: pristine layout_refiner vs the ANYTIME-patched one,
flag OFF, generous deadline (every stage converges) -> identical layouts."""
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "partner", ROOT / "tests"):
    sys.path.insert(0, str(p))
from synth_instances import build_instance, make_optimizer


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old = load(ROOT / "scratchpad/oldref/layout_refiner.py", "layout_refiner")
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


for n, seed, span in ((40, 0, 8.0), (40, 1, 8.0), (70, 0, 10.0)):
    h_old1 = run(old, n, seed, span)
    h_old2 = run(old, n, seed, span)
    h_new = run(new, n, seed, span)
    stable = h_old1 == h_old2
    print(f"n={n} seed={seed} span={span}: old={h_old1} old2={h_old2} "
          f"new={h_new} stable={stable} match={h_old1 == h_new}")
