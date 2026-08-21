"""alpha-curve probe, step 1: dump the REAL model prediction per case.

PROBE ONLY.  Reproduces exactly the batch the production 0.3s operating point
hands the direct channel -- `_sample_direct_preds(..., K=PARTNER_NREF,
oversample=False)` -- and stores the prescreen rank-0 layout, i.e. the model's
own best guess (no oracle selection anywhere).

Writes  alpha_pred0.json : {"<test_id>": [[x,y,w,h], ...]}   rank-0 pred
        alpha_predall.npz: every batch entry, for diversity diagnostics.

usage (from FloorSet/iccad2026contest, with the arm env exported):
    uv run python <this> [min_n]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SS = Path(__file__).resolve().parent
sys.path.insert(0, str(SS))
import gr_lib as G  # noqa: E402

REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")


def load_partner():
    p = REPO / "partner" / "contest_optimizer.py"
    spec = importlib.util.spec_from_file_location("partner_contest_optimizer", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["partner_contest_optimizer"] = mod
    spec.loader.exec_module(mod)
    return mod


def target_positions(case):
    """Byte-identical to the evaluator's opt_target_pos construction."""
    n = case.n
    tp = torch.full((n, 4), -1.0)
    for i in range(n):
        x, y, w, h = case.rects[i]
        if case.preplaced[i] != 0:
            tp[i] = torch.tensor([x, y, w, h])
        elif case.fixed[i] != 0:
            tp[i, 2] = float(w)
            tp[i, 3] = float(h)
    return tp


def main():
    min_n = int(sys.argv[1]) if len(sys.argv) > 1 else 95
    K = int(float(os.environ.get("PARTNER_NREF", "6") or 6))

    ev = G.load_evaluator()
    cases = G.load_cases(ev, "../")
    po = load_partner()
    opt = po.MyOptimizer(verbose=False)
    print(f"direct_model={opt.direct_model is not None} "
          f"flow_model={opt.flow_model is not None} "
          f"retrieval={opt.retrieval_index is not None} K={K}", flush=True)

    rank0 = {}
    allp = {}
    for c in cases:
        if c.n < min_n:
            continue
        tp = target_positions(c)
        preds = opt._sample_direct_preds(
            c.n, c.area_target, c.constraints, tp,
            c.b2b, c.p2b, c.pins, K, oversample=False)
        arrs = [np.asarray(p, dtype=np.float64)[: c.n] for p in preds]
        rank0[str(c.test_id)] = arrs[0].tolist()
        allp[str(c.test_id)] = np.stack(arrs)
        # batch diversity: mean pairwise centroid distance / frame diagonal
        cen = np.stack([a[:, :2] + 0.5 * a[:, 2:] for a in arrs])
        d = 0.0
        m = len(arrs)
        for i in range(m):
            for j in range(i + 1, m):
                d += float(np.linalg.norm(cen[i] - cen[j], axis=1).mean())
        d = d / max(1, m * (m - 1) // 2)
        x0, y0, x1, y1 = c.frame
        diag = float(np.hypot(x1 - x0, y1 - y0))
        print(f"tid {c.test_id:>3} n={c.n:>3} K={len(arrs)} "
              f"batch_spread={d / diag:.4f}", flush=True)

    (SS / "alpha_pred0.json").write_text(json.dumps(rank0))
    np.savez_compressed(SS / "alpha_predall.npz", **allp)
    print(f"wrote {len(rank0)} cases -> alpha_pred0.json")


if __name__ == "__main__":
    main()
