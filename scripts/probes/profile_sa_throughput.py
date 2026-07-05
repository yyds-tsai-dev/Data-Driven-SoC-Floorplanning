"""Profiling probe for the column-slicing SA (M1a enabler).

Loads three validation cases spanning the size/structure spectrum -- a small
case with preplaced obstacles, a mid case (~n=71), and the first n>=115 case --
and runs the single-threaded column-slicing annealer with a fixed short
wall-clock budget and a fixed RNG seed. For each case it reports:

  * moves attempted / second (throughput), and
  * a cProfile cumulative-time breakdown of the hot functions
    (_layout, _stack_column, _hpwl, _violations, _evaluate, _random_move).

Decision gate (M1.1): if _layout-family cumulative time is < 50% of the
move-loop time, memoization is not worth it -- stop and report.

Runnable from the repo root:  uv run python scripts/probes/profile_sa_throughput.py
Honors FLOORSET_FAST_EVAL (compare before/after by exporting the flag).
"""

from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Make src/ importable, mirroring scripts/probes/gt_seed_optimizer.py.
_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        _REPO = _p
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
else:
    _REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
    sys.path.insert(0, str(_REPO / "src"))

# FloorSet loader lives beside the submodule; add both plausible roots.
for _cand in (_REPO / "FloorSet", _REPO / "FloorSet" / "iccad2026contest"):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402

import floorset_arch.legalizer.column_backbone as cb  # noqa: E402
import floorset_arch.legalizer.column_slicing as cs  # noqa: E402

SCRATCH = Path(
    "/tmp/claude-1100/-nashome-NVL4-vdalab-yyds-dev-Data-Driven-SoC-Floorplanning/"
    "f21e985a-d31d-4b7f-8d00-a32cf7893d0b/scratchpad"
)
DATA_ROOT = str(_REPO / "FloorSet")
SEED = 20260705
BUDGET_S = 4.0
HOT = ("_layout", "_stack_column", "_hpwl", "_violations", "_evaluate", "_random_move")


def _n_of(sample) -> int:
    at = sample["input"][0]
    return int((at != -1).sum().item())


def _has_preplaced(sample) -> bool:
    cons = sample["input"][4]
    n = _n_of(sample)
    if cons.dim() < 2 or cons.shape[1] < 2:
        return False
    return bool((cons[:n, 1] != 0).any().item())


def _golden_target_positions(sample) -> torch.Tensor:
    """Replicate iccad2026_evaluate: preplaced blocks get their golden bbox
    (x,y,w,h) as target positions (which the legalizer turns into locked
    obstacle rects); everything else stays -1. Without this the obstacle
    path is never exercised."""
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


def _pick_cases(ds) -> List[Tuple[int, str]]:
    """(test_id, label): a small preplaced case, a mid case, the first n>=115."""
    small = mid = big = None
    for idx in range(len(ds)):
        s = ds[idx]
        n = _n_of(s)
        if small is None and n <= 30 and _has_preplaced(s):
            small = (idx, f"small+preplaced n={n}")
        if mid is None and 68 <= n <= 74:
            mid = (idx, f"mid n={n}")
        if big is None and n >= 115:
            big = (idx, f"large n={n}")
        if small and mid and big:
            break
    picks = [p for p in (small, mid, big) if p is not None]
    return picks


def _build_optimizer(sample, deadline: float) -> Optional[cs._ColumnOptimizer]:
    at, b2b, p2b, pins, cons = sample["input"]
    n = _n_of(sample)
    at = at[:n].float().cpu()
    cons = cons[:n].float().cpu()
    b2b = b2b.float().cpu()
    p2b = p2b.float().cpu()
    pins = pins.float().cpu()
    tpos = _golden_target_positions(sample)
    seed_rects = cb._heuristic_init(at, cons, tpos, b2b, p2b, pins)
    opt = cs._ColumnOptimizer(seed_rects, at, cons, tpos, b2b, p2b, pins,
                              deadline, seed=SEED)
    if opt.locked_only():
        return None
    opt.prepare()
    return opt


def _run_case(sample, tag: str, budget: float):
    """Anneal for `budget` seconds; return (n, moves, moves_per_s, stats)."""
    deadline = time.time() + budget
    opt = _build_optimizer(sample, deadline)
    if opt is None:
        return None
    cols = [list(c) for c in opt._cols]
    cur_cost, _ = opt._evaluate(cols)

    moves = [0]
    real_random_move = opt._random_move

    def counting_move(cols_):
        undo = real_random_move(cols_)
        if undo is not None:
            moves[0] += 1
        return undo

    opt._random_move = counting_move  # bound-method override on the instance

    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    opt._anneal(cols, deadline, cur_cost, t0=0.02, t1=0.0008)
    prof.disable()
    elapsed = time.time() - t0

    st = pstats.Stats(prof)
    st.calc_callees()
    cum = {}
    for func, (_cc, _nc, _tt, ct, _cal) in st.stats.items():  # type: ignore[attr-defined]
        cum[func[2]] = cum.get(func[2], 0.0) + ct
    return {
        "n": opt.n,
        "tag": tag,
        "moves": moves[0],
        "elapsed": elapsed,
        "mps": moves[0] / max(elapsed, 1e-9),
        "locked": len(opt.locked_rects),
        "cum": cum,
        "stats": st,
    }


def _top_entries(st: pstats.Stats, k: int = 8):
    rows = []
    for func, (_cc, nc, tt, ct, _cal) in st.stats.items():  # type: ignore[attr-defined]
        rows.append((ct, tt, nc, f"{func[2]} ({Path(func[0]).name}:{func[1]})"))
    rows.sort(reverse=True)
    return rows[:k]


def main():
    fast = os.environ.get("FLOORSET_FAST_EVAL", "0") == "1"
    mode = "FAST" if fast else "SLOW"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"[profile_sa_throughput] mode={mode} seed={SEED} budget={BUDGET_S}s "
          f"load={os.getloadavg()}")

    ds = FloorplanDatasetLiteTest(DATA_ROOT)
    picks = _pick_cases(ds)
    print(f"selected cases: {picks}")

    out_lines = []
    for idx, label in picks:
        res = _run_case(ds[idx], label, BUDGET_S)
        if res is None:
            print(f"case {idx} ({label}): locked-only, skipped")
            continue
        cum = res["cum"]
        layout_fam = sum(cum.get(f, 0.0) for f in ("_layout", "_stack_column"))
        eval_ct = cum.get("_evaluate", 0.0)
        loop_ref = max(eval_ct, res["elapsed"] * 0.5)
        frac = layout_fam / max(eval_ct, 1e-9)
        hdr = (f"\n=== case {idx} [{label}] n={res['n']} locked={res['locked']} "
               f"{mode} ===\n"
               f"  moves={res['moves']}  elapsed={res['elapsed']:.2f}s  "
               f"moves/s={res['mps']:.1f}\n"
               f"  cum(_layout+_stack_column)={layout_fam:.3f}s  "
               f"cum(_evaluate)={eval_ct:.3f}s  layout/evaluate={frac:.2%}")
        print(hdr)
        out_lines.append(hdr)
        print("  hot-function cumulative times:")
        for f in HOT:
            print(f"    {f:<16} cum={cum.get(f, 0.0):.4f}s")
        print("  top-8 by cumulative time:")
        for ct, tt, nc, name in _top_entries(res["stats"], 8):
            print(f"    cum={ct:7.3f}s tot={tt:7.3f}s calls={nc:>8} {name}")
        artifact = SCRATCH / f"sa_prof_{mode}_case{idx}_n{res['n']}.txt"
        with open(artifact, "w") as fh:
            st = res["stats"]
            st.stream = fh
            st.sort_stats("cumulative").print_stats(30)
        print(f"  wrote {artifact}")

    print("\n[decision gate] layout-family/evaluate ratio per case above.")
    print("If < 50% on all cases, the memoization kill criterion triggers.")


if __name__ == "__main__":
    main()
