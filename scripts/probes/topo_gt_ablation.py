"""GT-headroom ablation for the M2 topology-search stage.

Question this answers (the M4 gate, per the roadmap): if we seed the topo_search
DAGs from the GROUND-TRUTH (fp_sol golden) layout instead of the solver-derived
layout, how much MORE HPWL can the same-budget stage recover? A large gap means
a learned ordering/topology prior (M4) has headroom; a small gap means M4 is
permanently cancelled.

Mechanism (per validation case):
  1. Run the production backbone (no refine flags) to get the solver layout.
  2. Build target_positions for preplaced blocks from the golden bbox, exactly
     as the real evaluator does (replicated from
     scripts/probes/profile_sa_throughput.py::_golden_target_positions), so the
     LOCKED pinning inside build_axis_dags matches production.
  3. Run refine_topo on the SOLVER layout (solver-derived DAG) -> paired result.
  4. Run refine_topo on the GOLDEN layout (GT-derived DAG) with the SAME budget
     and the SAME netlist -> paired result. The golden layout's bbox is frozen
     as the stage's non-growth reference, so the two runs are compared on their
     own HPWL deltas (same-run paired: never cross-run).

NOTE fp_sol golden layouts are geometry references only -- per contest QA they
may themselves violate soft constraints. We therefore report the golden run's
guard_result honestly (a golden layout that fails hard_legal / soft as the
stage input simply yields guard_result != accepted and a 0 delta for that case;
it is not silently treated as a win).

Smoke-only: this script runs 2-3 cases when invoked directly. The scheduler
runs the full 100-case ablation.

Run with FLOORSET_COLUMN_BACKBONE=1 and NO refine flags (the probe drives
refine_topo directly). Invoke via the eval-script PYTHONPATH so litetestLoader
/ iccad2026_evaluate imports resolve, e.g.:

  cd FloorSet/iccad2026contest
  PYTHONPATH=... FLOORSET_COLUMN_BACKBONE=1 \
      ~/.local/bin/uv run python <repo>/scripts/probes/topo_gt_ablation.py --cases 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Make floorset_arch importable (mirrors gt_seed_optimizer.py).
_THIS = Path(__file__).resolve()
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break

import floorset_arch.legalizer.column_backbone as cb  # noqa: E402
from floorset_arch.legalizer.column_backbone import solve_with_column_backbone  # noqa: E402
from floorset_arch.refine.topo_search import refine_topo  # noqa: E402
from floorset_arch.refine.wirelength import hpwl  # noqa: E402
from floorset_arch.refine.guards import hard_legal  # noqa: E402

Rect = Tuple[float, float, float, float]


def _n_of(sample) -> int:
    at = sample["input"][0]
    return int((at != -1).sum().item())


def _golden_target_positions(sample) -> torch.Tensor:
    """Replicate iccad2026_evaluate: preplaced blocks get their golden bbox
    (x,y,w,h) as target positions; everything else stays -1. Copied from
    profile_sa_throughput.py so LOCKED pinning matches production."""
    at, _b2b, _p2b, _pins, cons = sample["input"]
    polys, _metrics = sample["label"]
    n = _n_of(sample)
    tpos = torch.full((n, 4), -1.0)
    if cons.dim() < 2 or cons.shape[1] < 2:
        return tpos
    for i in range(n):
        if cons[i, 1] == 0:
            continue
        block = polys[i]
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            continue
        mn = valid.min(dim=0).values
        mx = valid.max(dim=0).values
        tpos[i] = torch.tensor([float(mn[0]), float(mn[1]),
                                float(mx[0] - mn[0]), float(mx[1] - mn[1])])
    return tpos


def _golden_rects(sample) -> List[Rect]:
    """Golden layout as per-block bbox rects (evaluator _extract_baseline)."""
    polys, _metrics = sample["label"]
    n = _n_of(sample)
    out: List[Rect] = []
    for i in range(n):
        block = polys[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            mn = valid.min(dim=0).values
            mx = valid.max(dim=0).values
            out.append((float(mn[0]), float(mn[1]),
                        float(mx[0] - mn[0]), float(mx[1] - mn[1])))
        else:
            out.append((0.0, 0.0, 1.0, 1.0))
    return out


def _edge_lists(b2b, p2b, pins, n):
    b2b_e = []
    for row in b2b.tolist():
        b2b_e.append((int(row[0]), int(row[1]), float(row[2])))
    p2b_e = []
    for row in p2b.tolist():
        p2b_e.append((int(row[0]), int(row[1]), float(row[2])))
    pin_l = [(float(r[0]), float(r[1])) for r in pins.tolist()]
    return b2b_e, p2b_e, pin_l


def _topo_budget(n: int) -> float:
    """Match the production carve-out: min(3.0, 0.25 * remaining_budget) with
    remaining_budget = budget - refine_reserve. Reuse the backbone's own
    budget functions so the probe's stage budget mirrors production for n>=60.
    For n<60 (where production does not carve a reserve) use a small fixed
    budget so the probe still exercises the stage."""
    from floorset_arch.legalizer.column_backbone import _time_budget
    budget = _time_budget(n)
    refine_reserve = min(2.0, 0.3 + 0.012 * n, 0.4 * budget)
    remaining = max(0.0, budget - refine_reserve)
    if n >= 60:
        return min(3.0, 0.25 * remaining)
    return min(1.0, 0.25 * remaining)


def _run_case(sample, tag: str) -> None:
    at, b2b, p2b, pins, cons = sample["input"]
    n = _n_of(sample)
    at = at[:n].float().cpu()
    cons = cons[:n].float().cpu()
    b2b = b2b.float().cpu()
    p2b = p2b.float().cpu()
    pins = pins.float().cpu()
    tpos = _golden_target_positions(sample)

    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins, n)
    budget = _topo_budget(n)

    # --- solver layout ---
    solver_rects = solve_with_column_backbone(n, at, b2b, p2b, pins, cons, tpos)
    solver_rects = [tuple(r) for r in solver_rects]
    s_pre = hpwl(solver_rects, b2b_e, p2b_e, pin_l)
    s_out, s_detail = refine_topo(
        solver_rects, at, cons, tpos, b2b_e, p2b_e, pin_l,
        deadline=time.time() + budget,
    )
    s_post = hpwl(s_out, b2b_e, p2b_e, pin_l)
    s_legal = hard_legal(s_out, at, cons, tpos)

    # --- golden layout ---
    golden = _golden_rects(sample)
    g_pre = hpwl(golden, b2b_e, p2b_e, pin_l)
    g_out, g_detail = refine_topo(
        golden, at, cons, tpos, b2b_e, p2b_e, pin_l,
        deadline=time.time() + budget,
    )
    g_post = hpwl(g_out, b2b_e, p2b_e, pin_l)
    g_legal = hard_legal(g_out, at, cons, tpos)

    def _pct(pre, post):
        return 100.0 * (pre - post) / pre if pre > 0 else 0.0

    print(f"[{tag}] n={n} budget={budget:.2f}s")
    print(f"    SOLVER  pre={s_pre:.3f} post={s_post:.3f} "
          f"delta%={_pct(s_pre, s_post):5.2f} "
          f"prop={s_detail['n_proposed']} acc={s_detail['n_accepted']} "
          f"guard={s_detail['guard_result']} elapsed={s_detail['elapsed']:.3f} "
          f"legal={s_legal}")
    print(f"    GOLDEN  pre={g_pre:.3f} post={g_post:.3f} "
          f"delta%={_pct(g_pre, g_post):5.2f} "
          f"prop={g_detail['n_proposed']} acc={g_detail['n_accepted']} "
          f"guard={g_detail['guard_result']} elapsed={g_detail['elapsed']:.3f} "
          f"legal={g_legal}")
    # Headroom signal: golden-run post HPWL vs solver-run post HPWL, normalized.
    # (Absolute cross-layout HPWL is comparable here because both use the exact
    # same netlist; this is the raw signal the scheduler aggregates over 100.)
    if s_post > 0:
        print(f"    golden_post/solver_post = {g_post / s_post:.4f} "
              f"(<1 => GT topology yields lower HPWL => M4 headroom)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3,
                    help="number of validation cases to smoke (default 3)")
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    args = ap.parse_args()

    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")

    from litetestLoader import FloorplanDatasetLiteTest

    ds = FloorplanDatasetLiteTest(str(args.data_path))
    try:
        cb.warm_worker_pool()
    except Exception:
        pass

    # Pick a spread: a small case, a mid case, and the first large (n>=100).
    picks: List[Tuple[int, str]] = []
    small = mid = big = None
    for idx in range(len(ds)):
        s = ds[idx]
        n = _n_of(s)
        if small is None and n <= 40:
            small = (idx, "small")
        if mid is None and 65 <= n <= 80:
            mid = (idx, "mid")
        if big is None and n >= 100:
            big = (idx, "large")
        if small and mid and big:
            break
    for p in (small, mid, big):
        if p is not None:
            picks.append(p)
    picks = picks[:args.cases]

    for idx, tag in picks:
        _run_case(ds[idx], tag)


if __name__ == "__main__":
    main()
