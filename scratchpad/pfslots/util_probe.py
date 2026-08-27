#!/usr/bin/env python
"""Live-count probe for PARTNER_PIN_FRAME_SLOT_MIN_UTIL.

For every case of a suite: rebuild the parent-time view exactly as
contest_optimizer.solve does (heuristic seed rects + the evaluator's
opt_target_pos), then report `_has_tag_locks` and `_tag_lock_box_util`.

Usage: uv run python scratchpad/pfslots/util_probe.py <data_root> [tag]
       (data_root = the directory containing LiteTensorDataTest)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "partner", ROOT / "FloorSet" / "iccad2026contest",
           ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from litetestLoader import FloorplanDatasetLiteTest       # noqa: E402
from column_sa_legalizer import (_ColumnOptimizer,        # noqa: E402
                                 _has_tag_locks, _tag_lock_box_util)
from contest_optimizer import _heuristic_init             # noqa: E402

data = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else Path(data).name
ds = FloorplanDatasetLiteTest(data)
rows = []
for idx in range(len(ds)):
    sample = ds[idx]
    at_full, b2b, p2b, pins, cons = sample["input"]
    polygons, _m = sample["label"]
    n = int((at_full != -1).sum().item())
    nc = cons.shape[1] if cons.dim() > 1 else 0
    tpos = torch.full((n, 4), -1.0)
    for i in range(n):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if not len(valid):
            continue
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        if nc > 1 and cons[i, 1] != 0:
            tpos[i] = torch.tensor([float(x_min), float(y_min),
                                    float(x_max - x_min), float(y_max - y_min)])
        elif nc > 0 and cons[i, 0] != 0:
            tpos[i, 2] = float(x_max - x_min)
            tpos[i, 3] = float(y_max - y_min)
    at = at_full[:n].detach().float().cpu()
    c = cons[:n].detach().float().cpu()
    b2b = b2b.detach().float().cpu(); p2b = p2b.detach().float().cpu()
    pins = pins.detach().float().cpu()
    rects = _heuristic_init(at, c, tpos, b2b, p2b, pins)
    opt = _ColumnOptimizer(rects, at, c, tpos, b2b, p2b, pins,
                           time.time() + 30.0, seed=0)
    rows.append((idx, n, bool(_has_tag_locks(opt)), float(_tag_lock_box_util(opt))))
    print(f"[{tag}] tid={idx} n={n} locks={rows[-1][2]} util={rows[-1][3]:.4f}",
          flush=True)

live = [r for r in rows if r[2]]
print(f"SUMMARY {tag}: cases={len(rows)} locks={len(live)}")
for u in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.00):
    k = [r for r in live if r[3] >= u]
    print(f"  u>={u:.2f}: live={len(k):3d}  ids={[r[0] for r in k][:40]}")
