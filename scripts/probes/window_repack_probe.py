"""Diagnostic probe for the M5 windowed re-pack stage.

Drives refine_window directly on the production backbone layout for specified
validation cases, reporting the M2-failure-mode metrics the roadmap warns
about: how many tension windows were SELECTABLE (clean rectangular hole), how
many the packer improved, how many passed the full guard chain, and the
resulting full-layout HPWL delta. This tells us WHERE accepts die (selection?
packer? guards?) before spending eval_single budget.

Run via the eval-script PYTHONPATH so litetestLoader imports resolve, e.g.:

  cd FloorSet/iccad2026contest
  PYTHONPATH=... FLOORSET_COLUMN_BACKBONE=1 \
      ~/.local/bin/uv run python <repo>/scripts/probes/window_repack_probe.py --ids 50 92 95 99
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch

_THIS = Path(__file__).resolve()
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break

import floorset_arch.legalizer.column_backbone as cb  # noqa: E402
from floorset_arch.legalizer.column_backbone import solve_with_column_backbone  # noqa: E402
from floorset_arch.refine.window_repack import refine_window  # noqa: E402
from floorset_arch.refine.wirelength import hpwl  # noqa: E402
from floorset_arch.refine.guards import hard_legal  # noqa: E402

Rect = Tuple[float, float, float, float]


def _n_of(sample) -> int:
    at = sample["input"][0]
    return int((at != -1).sum().item())


def _golden_target_positions(sample) -> torch.Tensor:
    """Replicate the evaluator's opt_target_pos EXACTLY (iccad2026_evaluate.py
    ~869-881): fixed-shape blocks get (w,h) only (x/y = -1); preplaced blocks
    get full (x,y,w,h); everything else -1. Using the same tensor the backbone
    consumed keeps the probe's hard_legal check consistent with production (a
    mismatched golden tpos would spuriously fail the INCOMING layout)."""
    at, _b2b, _p2b, _pins, cons = sample["input"]
    polys, _metrics = sample["label"]
    n = _n_of(sample)
    tpos = torch.full((n, 4), -1.0)
    if cons.dim() < 2 or cons.shape[1] < 1:
        return tpos
    nc = cons.shape[1]
    for i in range(n):
        is_fixed = nc > 0 and cons[i, 0] != 0
        is_preplaced = nc > 1 and cons[i, 1] != 0
        if not (is_fixed or is_preplaced):
            continue
        block = polys[i]
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            continue
        mn = valid.min(dim=0).values
        mx = valid.max(dim=0).values
        gx, gy = float(mn[0]), float(mn[1])
        gw, gh = float(mx[0] - mn[0]), float(mx[1] - mn[1])
        if is_preplaced:
            tpos[i] = torch.tensor([gx, gy, gw, gh])
        elif is_fixed:
            tpos[i, 2] = gw
            tpos[i, 3] = gh
    return tpos


def _edge_lists(b2b, p2b, pins):
    b2b_e = [(int(r[0]), int(r[1]), float(r[2])) for r in b2b.tolist()]
    p2b_e = [(int(r[0]), int(r[1]), float(r[2])) for r in p2b.tolist()]
    pin_l = [(float(r[0]), float(r[1])) for r in pins.tolist()]
    return b2b_e, p2b_e, pin_l


def _window_budget(n: int) -> float:
    from floorset_arch.legalizer.column_backbone import _time_budget
    budget = _time_budget(n)
    refine_reserve = min(2.0, 0.3 + 0.012 * n, 0.4 * budget)
    remaining = max(0.0, budget - refine_reserve)
    if n >= 60:
        return min(3.0, 0.25 * remaining)
    return min(1.0, 0.25 * remaining)


def _run_case(sample, tag: str, log_path: str) -> None:
    at, b2b, p2b, pins, cons = sample["input"]
    n = _n_of(sample)
    at = at[:n].float().cpu()
    cons = cons[:n].float().cpu()
    b2b = b2b.float().cpu()
    p2b = p2b.float().cpu()
    pins = pins.float().cpu()
    tpos = _golden_target_positions(sample)

    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    budget = _window_budget(n)

    solver_rects = solve_with_column_backbone(n, at, b2b, p2b, pins, cons, tpos)
    solver_rects = [tuple(r) for r in solver_rects]
    pre = hpwl(solver_rects, b2b_e, p2b_e, pin_l)

    # Fresh per-case log so we can parse per-window guard_result breakdown.
    case_log = f"{log_path}.{tag}.jsonl"
    try:
        os.remove(case_log)
    except OSError:
        pass

    t0 = time.time()
    out, detail = refine_window(
        solver_rects, at, cons, tpos, b2b_e, p2b_e, pin_l,
        deadline=time.time() + budget, log_path=case_log,
    )
    elapsed = time.time() - t0
    post = hpwl(out, b2b_e, p2b_e, pin_l)
    legal = hard_legal(out, at, cons, tpos)

    # Parse per-window records for the guard-result breakdown.
    breakdown: Counter = Counter()
    try:
        import json
        with open(case_log) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("window_blocks") is not None:
                    breakdown[rec.get("guard_result", "?")] += 1
    except OSError:
        pass

    dpct = 100.0 * (pre - post) / pre if pre > 0 else 0.0
    # total-equivalent: cost gap ~ 0.5 * hpwl_gap; approximate per-case cost
    # delta using hpwl fractional reduction as a proxy for gap reduction.
    print(f"[{tag}] n={n} budget={budget:.2f}s elapsed={elapsed:.2f}s")
    print(f"    HPWL pre={pre:.3f} post={post:.3f} delta%={dpct:5.2f} legal={legal}")
    print(f"    windows tried={detail['n_windows_tried']} "
          f"accepted={detail['n_windows_accepted']} "
          f"mean_k={detail['mean_k']:.1f} guard={detail['guard_result']}")
    if breakdown:
        print(f"    per-window guard breakdown: {dict(breakdown)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[50, 92, 95, 99])
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--log", default="/tmp/window_probe")
    args = ap.parse_args()

    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")

    from litetestLoader import FloorplanDatasetLiteTest

    ds = FloorplanDatasetLiteTest(str(args.data_path))
    try:
        cb.warm_worker_pool()
    except Exception:
        pass

    for idx in args.ids:
        if 0 <= idx < len(ds):
            s = ds[idx]
            _run_case(s, str(idx), args.log)
        else:
            print(f"[skip] id {idx} out of range (len={len(ds)})")


if __name__ == "__main__":
    main()
