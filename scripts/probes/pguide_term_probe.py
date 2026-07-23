"""Term-attribution probe: is the overlap blow-up caused by the hpwl/boundary
attraction terms in the guidance energy (not the trajectory injection)?

k=5, eta=0.05, last_only trajectory (guidance touches only the final x0) so we
isolate guide_x0's own effect.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "partner"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

CASES = [0, 49, 88]
# (name, extra env)
CONFIGS = [
    ("k0",            dict(guide="0")),
    ("full_lastonly", dict(guide="1", PGUIDE_LAST_ONLY="1")),
    ("ov_only_last",  dict(guide="1", PGUIDE_LAST_ONLY="1",
                           PGUIDE_W_BOUNDARY="0", PGUIDE_W_HPWL="0")),
    ("ov_only_cur",   dict(guide="1",
                           PGUIDE_W_BOUNDARY="0", PGUIDE_W_HPWL="0")),
    ("no_hpwl_last",  dict(guide="1", PGUIDE_LAST_ONLY="1", PGUIDE_W_HPWL="0")),
    ("no_bnd_last",   dict(guide="1", PGUIDE_LAST_ONLY="1", PGUIDE_W_BOUNDARY="0")),
]
CLEAR = ["PGUIDE_SC_MODE", "PGUIDE_EPS_MODE", "PGUIDE_LAST_ONLY",
         "PGUIDE_W_BOUNDARY", "PGUIDE_W_HPWL"]


def main():
    import numpy as np
    import torch
    from pguide_quick_probe import load_case, _overlap_area
    from litetestLoader import FloorplanDatasetLiteTest
    from my_opt_claude import MyOptimizer

    dataset = FloorplanDatasetLiteTest("../")
    opt = MyOptimizer()
    out = []
    for cid in CASES:
        n, at, cons, tpos, b2b, p2b, pins = load_case(dataset, cid)
        for name, cfg in CONFIGS:
            for k in CLEAR:
                os.environ.pop(k, None)
            os.environ["PARTNER_PHYSICS_GUIDE"] = cfg["guide"]
            os.environ["PGUIDE_K"] = "0" if cfg["guide"] == "0" else "5"
            os.environ["PGUIDE_ETA"] = "0.05"
            for k, v in cfg.items():
                if k != "guide":
                    os.environ[k] = v
            t0 = time.time()
            preds = opt._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b,
                                                 pins, K=15)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            ov = [float(_overlap_area(p, n)) for p in preds]
            rec = dict(case=cid, cfg=name, sample_s=round(dt, 2),
                       overlap_median=round(float(np.median(ov)), 2),
                       overlap_best=round(float(np.min(ov)), 2))
            out.append(rec)
            print(rec, flush=True)
    json.dump(out, open(ROOT / "artifacts/pguide/term_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
