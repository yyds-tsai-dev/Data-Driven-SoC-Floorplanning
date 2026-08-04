#!/usr/bin/env python3
"""A/B of PARTNER_ANYTIME_LADDER on the real `refine_prediction`, single
process, one validation case, real cached Direct predictions.

Usage: uv run python scratchpad/anytime_ab.py <test_id> [spans...]
"""

from __future__ import annotations

import math
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


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 95
    spans = [float(s) for s in sys.argv[2:]] or [3.2, 2.4, 1.7, 1.2, 0.8]
    case = load_case(test_id)
    preds = sample_preds(test_id, case)
    n_pred = int(os.environ.get("AB_PREDS", "4"))
    print(f"[ab] test_id={test_id} n={case[0]} preds={n_pred}", flush=True)

    rows = []
    for span in spans:
        for pi, pred in enumerate(preds[:n_pred]):
            for mode in ("off", "on"):
                if mode == "on":
                    os.environ["PARTNER_ANYTIME_LADDER"] = "1"
                else:
                    os.environ.pop("PARTNER_ANYTIME_LADDER", None)
                opt, deadline = build_opt(case, pred, span)
                tb = time.time()
                out = rf.refine_prediction(opt, pred, deadline, seed=327)
                el = time.time() - tb
                if out is None:
                    rows.append((span, pi, mode, None, None, None, el))
                    print(f"span={span:4.1f} p{pi} {mode:3s} NONE  "
                          f"el={el:5.2f} over={el - span:+5.2f}", flush=True)
                    continue
                pos = np.asarray(out, dtype=np.float64)
                hp = float(opt._hpwl(pos))
                area = float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                             * ((pos[:, 1] + pos[:, 3]).max()
                                - pos[:, 1].min()))
                V = int(rf.full_violations(opt, pos))
                rows.append((span, pi, mode, hp, area / opt.area_ref, V, el))
                print(f"span={span:4.1f} p{pi} {mode:3s} OK    "
                      f"el={el:5.2f} over={el - span:+5.2f} hp={hp:8.1f} "
                      f"a/ref={area / opt.area_ref:5.3f} V={V}", flush=True)
    os.environ.pop("PARTNER_ANYTIME_LADDER", None)

    ok = [r for r in rows if r[3] is not None]
    hp_ref = min(r[3] for r in ok) if ok else 1.0
    opt, _ = build_opt(case, preds[0], 1.0)
    den = max(getattr(opt, "n_soft_den", 1), 1)

    def proxy(r):
        if r[3] is None:
            return None
        return (1.0 + 0.5 * ((r[3] - hp_ref) / hp_ref + max(0.0, r[4] - 1.0))) \
            * math.exp(2.0 * r[5] / den)

    print("\n span  pred   off      on      delta   (proxy, lower better)")
    per_span = {}
    for span in spans:
        for pi in range(n_pred):
            o = next((proxy(r) for r in rows
                      if r[0] == span and r[1] == pi and r[2] == "off"), None)
            a = next((proxy(r) for r in rows
                      if r[0] == span and r[1] == pi and r[2] == "on"), None)
            d = (f"{a - o:+.4f}" if (o is not None and a is not None)
                 else "  --")
            print(f" {span:4.1f}  {pi}    "
                  f"{'None' if o is None else f'{o:.4f}'}  "
                  f"{'None' if a is None else f'{a:.4f}'}  {d}")
        # per-span best-of-slots (what the pool selection actually sees)
        bo = [proxy(r) for r in rows if r[0] == span and r[2] == "off"
              and r[3] is not None]
        ba = [proxy(r) for r in rows if r[0] == span and r[2] == "on"
              and r[3] is not None]
        per_span[span] = (min(bo) if bo else None, min(ba) if ba else None,
                          len(bo), len(ba))
    print("\n span  best-of-slots off / on   (delivered slots)")
    for span in spans:
        o, a, no, na = per_span[span]
        print(f" {span:4.1f}  {'None' if o is None else f'{o:.4f}'} / "
              f"{'None' if a is None else f'{a:.4f}'}   ({no}/{n_pred} vs "
              f"{na}/{n_pred})")


if __name__ == "__main__":
    main()
