"""Quick physics-guidance probe: raw candidate quality vs (K, eta) on
tail/boundary-dense cases. No refine, no golden labels; ~2 min on GPU.

Run via scripts/probes/run_pguide_quick_probe.sh (cwd = FloorSet/iccad2026contest,
PYTHONPATH set) so litetestLoader.FloorplanDatasetLiteTest resolves — same
loader pattern as ContestEvaluator (scripts/iccad2026_evaluate.py).
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "partner"))

CASES = [0, 49, 79, 88, 93, 96, 99]
GRID = [dict(k=0), dict(k=3, eta=0.05), dict(k=5, eta=0.05),
        dict(k=10, eta=0.05), dict(k=5, eta=0.1)]


def load_case(dataset, idx):
    """Mirror ContestEvaluator.evaluate's per-case unpacking (scripts/iccad2026_evaluate.py),
    including _extract_baseline's polygons -> (x,y,w,h) target_pos derivation."""
    import torch
    sample = dataset[idx]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    polygons, _metrics = labels

    target_pos = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            target_pos.append((float(x_min), float(y_min),
                              float(x_max - x_min), float(y_max - y_min)))
        else:
            target_pos.append((0, 0, 1, 1))

    opt_target_pos = torch.full((block_count, 4), -1.0)
    if target_pos is not None and constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(block_count):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            if is_preplaced:
                tx, ty, tw, th = target_pos[i]
                opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
            elif is_fixed:
                _, _, tw, th = target_pos[i]
                opt_target_pos[i, 2] = tw
                opt_target_pos[i, 3] = th

    return (block_count, area_target, constraints, opt_target_pos,
            b2b_conn, p2b_conn, pins_pos)


def main():
    import numpy as np
    import torch
    from litetestLoader import FloorplanDatasetLiteTest
    from my_opt_claude import MyOptimizer

    dataset = FloorplanDatasetLiteTest("../")
    out = []
    opt = MyOptimizer()
    for cid in CASES:
        n, at, cons, tpos, b2b, p2b, pins = load_case(dataset, cid)
        for cfgi in GRID:
            os.environ["PARTNER_PHYSICS_GUIDE"] = "1" if cfgi["k"] else "0"
            os.environ["PGUIDE_K"] = str(cfgi.get("k", 0))
            os.environ["PGUIDE_ETA"] = str(cfgi.get("eta", 0.05))
            t0 = time.time()
            preds = opt._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b,
                                                 pins, K=15)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            ov = [float(_overlap_area(p, n)) for p in preds]
            pen = opt._constraint_penalties(preds, n, at, cons) or [0.0] * len(preds)
            out.append(dict(case=cid, **cfgi, sample_s=dt,
                            overlap_median=float(np.median(ov)),
                            overlap_best=float(np.min(ov)),
                            viol_median=float(np.median(pen)),
                            viol_best=float(np.min(pen))))
            print(out[-1])
    Path(ROOT / "artifacts/pguide").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "artifacts/pguide/quick_probe.json", "w"), indent=1)


def _overlap_area(P, n):
    import numpy as np
    x0, y0 = P[:n, 0], P[:n, 1]
    x1, y1 = x0 + P[:n, 2], y0 + P[:n, 3]
    ox = np.clip(np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None]), 0, None)
    oy = np.clip(np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None]), 0, None)
    m = ox * oy
    return (m.sum() - np.trace(m)) / 2.0


if __name__ == "__main__":
    main()
