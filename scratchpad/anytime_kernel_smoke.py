import os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "partner", ROOT / "tests"):
    sys.path.insert(0, str(p))
import layout_refiner as rf
from synth_instances import build_instance, make_optimizer
inst = build_instance(n=70, seed=0)
pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
for span in (1.5,):
    dl = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=dl)
    t = time.time()
    out = rf.refine_prediction(opt, pred, dl, seed=11)
    el = time.time() - t
    ok = out is not None
    print(f"kernel={os.environ.get('PARTNER_SA_KERNEL','off')} "
          f"anytime={os.environ.get('PARTNER_ANYTIME_LADDER','0')} "
          f"span={span} el={el:.2f} delivered={ok}"
          + ("" if not ok else
             f" V={rf.full_violations(opt, np.asarray(out))}"))
