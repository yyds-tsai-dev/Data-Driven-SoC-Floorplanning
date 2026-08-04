#!/usr/bin/env python3
"""Synthetic-instance behaviour of the anytime ladder (no dataset, no pool).
Used to pick non-flaky spans for tests/test_partner_anytime_ladder.py."""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "partner", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import layout_refiner as rf  # noqa: E402
from synth_instances import build_instance, make_optimizer  # noqa: E402


def run(n, seed, span, on):
    if on:
        os.environ["PARTNER_ANYTIME_LADDER"] = "1"
    else:
        os.environ.pop("PARTNER_ANYTIME_LADDER", None)
    inst = build_instance(n=n, seed=seed)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + span)
    t = time.time()
    out = rf.refine_prediction(opt, pred, time.time() + span, seed=11)
    el = time.time() - t
    if out is None:
        return None, el
    pos = np.asarray(out, dtype=np.float64)
    return (float(opt._hpwl(pos)), int(rf.full_violations(opt, pos)),
            float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                  * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min()))
            / opt.area_ref), el


for n in (40, 70, 100):
    for seed in (0, 1):
        for span in (0.5, 1.0, 2.0, 4.0):
            for on in (False, True):
                r, el = run(n, seed, span, on)
                tag = "on " if on else "off"
                print(f"n={n} seed={seed} span={span:4.1f} {tag} el={el:5.2f} "
                      f"{'NONE' if r is None else r}", flush=True)
