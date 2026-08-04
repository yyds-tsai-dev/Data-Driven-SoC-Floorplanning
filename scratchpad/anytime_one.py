#!/usr/bin/env python3
"""One (case, pred, span) through refine_prediction with the current env."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "scratchpad"))
from anytime_probe import build_opt, load_case, sample_preds  # noqa: E402
import layout_refiner as rf  # noqa: E402

tid = int(sys.argv[1])
pi = int(sys.argv[2])
span = float(sys.argv[3])
case = load_case(tid)
preds = sample_preds(tid, case)
opt, dl = build_opt(case, preds[pi], span)
t = time.time()
out = rf.refine_prediction(opt, preds[pi], dl, seed=327)
el = time.time() - t
if out is None:
    print(f"RESULT el={el:.2f} NONE")
else:
    pos = np.asarray(out, dtype=np.float64)
    ar = (((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
          * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min())) / opt.area_ref
    print(f"RESULT el={el:.2f} hp={opt._hpwl(pos):.1f} a/ref={ar:.3f} "
          f"V={rf.full_violations(opt, pos)}")
