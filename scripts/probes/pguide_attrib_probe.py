"""Attribution probe: isolate which trajectory-injection channel makes
physics guidance worsen raw overlap. Reuses pguide_quick_probe.load_case.

Configs at fixed k=5, eta=0.05:
  k0        : no guidance (baseline)
  current   : guide every gated step, sc=guided, eps=reparam (legacy)
  sc_ung    : sc=model x0, eps=reparam
  eps_model : sc=guided, eps=model
  decouple  : sc=model x0, eps=model
  last_only : guidance only on the final emitted x0 (clean trajectory)
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
CONFIGS = {
    "k0":        dict(guide="0"),
    "current":   dict(guide="1"),
    "sc_ung":    dict(guide="1", PGUIDE_SC_MODE="unguided"),
    "eps_model": dict(guide="1", PGUIDE_EPS_MODE="model"),
    "decouple":  dict(guide="1", PGUIDE_SC_MODE="unguided", PGUIDE_EPS_MODE="model"),
    "last_only": dict(guide="1", PGUIDE_LAST_ONLY="1"),
}
FLAG_KEYS = ["PGUIDE_SC_MODE", "PGUIDE_EPS_MODE", "PGUIDE_LAST_ONLY"]


def main():
    import numpy as np
    import torch
    from pguide_quick_probe import load_case, _overlap_area
    from litetestLoader import FloorplanDatasetLiteTest
    from contest_optimizer import MyOptimizer

    dataset = FloorplanDatasetLiteTest("../")
    opt = MyOptimizer()
    out = []
    for cid in CASES:
        n, at, cons, tpos, b2b, p2b, pins = load_case(dataset, cid)
        for name, cfg in CONFIGS.items():
            for k in FLAG_KEYS:
                os.environ.pop(k, None)
            os.environ["PARTNER_PHYSICS_GUIDE"] = cfg["guide"]
            os.environ["PGUIDE_K"] = "0" if cfg["guide"] == "0" else "5"
            os.environ["PGUIDE_ETA"] = "0.05"
            for k in FLAG_KEYS:
                if k in cfg:
                    os.environ[k] = cfg[k]
            t0 = time.time()
            preds = opt._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b,
                                                 pins, K=15)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            ov = [float(_overlap_area(p, n)) for p in preds]
            pen = opt._constraint_penalties(preds, n, at, cons) or [0.0] * len(preds)
            rec = dict(case=cid, cfg=name, sample_s=round(dt, 2),
                       overlap_median=round(float(np.median(ov)), 2),
                       overlap_best=round(float(np.min(ov)), 2),
                       viol_median=round(float(np.median(pen)), 4))
            out.append(rec)
            print(rec, flush=True)
    Path(ROOT / "artifacts/pguide").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "artifacts/pguide/attrib_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
