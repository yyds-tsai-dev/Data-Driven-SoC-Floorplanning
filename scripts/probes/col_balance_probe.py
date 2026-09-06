#!/usr/bin/env python3
"""Offline construction check for `PARTNER_COL_BALANCE` (no full evaluation):
for all 100 official validation cases, build a `_ColumnOptimizer` under the
canonical G1 solver env (`src/icdc_engine/g1_runtime.py` `_SOLVER_ENV`), call
`_init_columns(C0)` with the flag off vs on, and report the realized initial
column width sum (`sum_c max(A_c/(H-R_c), maxrigid_w_c)`, the same formula
`_layout_full` uses per column) vs `W_est`.

Standalone probe -- does not touch solve() or any existing file.

Usage:
    uv run scripts/probes/col_balance_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "solver", REPO / "FloorSet" / "iccad2026contest",
           REPO / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from icdc_engine.data import load_test_cases  # noqa: E402
from icdc_engine.g1_runtime import _SOLVER_ENV  # noqa: E402
import iccad2026_evaluate as ev  # noqa: E402

import column_sa_legalizer as csl  # noqa: E402


def _apply_env():
    for k, v in _SOLVER_ENV.items():
        os.environ[k] = v
    # keep the probe CPU/light; irrelevant to _init_columns either way
    os.environ["DIRECT_OFF"] = "1"
    os.environ["PARTNER_FLOW_SLOTS"] = "0"


def _realized_width_sum(opt, cols) -> float:
    total = 0.0
    for ulist in cols:
        if not ulist:
            continue
        soft_a = sum(opt.units[k].eff_soft for k in ulist)
        rigid_h = sum(opt.units[k].eff_rigid_h for k in ulist)
        max_w = max((opt.units[k].max_rigid_w for k in ulist), default=0.0)
        avail = max(opt.H - rigid_h, 0.05 * opt.H)
        total += max(soft_a / avail, max_w, 0.5)
    return total


def main() -> None:
    cases = load_test_cases(ev)
    rows = []
    for c in cases:
        rects = [tuple(float(v) for v in r) for r in c["rects"]]

        os.environ.pop("PARTNER_COL_BALANCE", None)
        _apply_env()
        opt_off = csl._ColumnOptimizer(rects, c["area"], c["cons"], c["tp"],
                                       c["b2b"], c["p2b"], c["pins"],
                                       deadline=None, seed=7)
        opt_off.prepare()
        cols_off = opt_off._cols
        w_off = _realized_width_sum(opt_off, cols_off)

        _apply_env()
        os.environ["PARTNER_COL_BALANCE"] = "1"
        opt_on = csl._ColumnOptimizer(rects, c["area"], c["cons"], c["tp"],
                                      c["b2b"], c["p2b"], c["pins"],
                                      deadline=None, seed=7)
        opt_on.prepare()
        cols_on = opt_on._cols
        w_on = _realized_width_sum(opt_on, cols_on)
        os.environ.pop("PARTNER_COL_BALANCE", None)

        w_est = float(opt_off.W_est)
        delta_pct = 100.0 * (w_on - w_off) / max(w_off, 1e-9)
        rows.append({
            "test_id": c["test_id"], "n": c["n"], "C0": opt_off.C0,
            "w_est": w_est, "w_off": w_off, "w_on": w_on,
            "ratio_off": w_off / max(w_est, 1e-9),
            "ratio_on": w_on / max(w_est, 1e-9),
            "delta_pct": delta_pct,
        })

    deltas = sorted(r["delta_pct"] for r in rows)
    n = len(deltas)
    median = deltas[n // 2] if n % 2 else 0.5 * (deltas[n // 2 - 1] + deltas[n // 2])
    worst = max(deltas)
    worst_row = max(rows, key=lambda r: r["delta_pct"])

    print(f"{'id':>3} {'n':>3} {'C0':>3} {'w_est':>10} {'w_off':>10} "
          f"{'w_on':>10} {'ratio_off':>10} {'ratio_on':>10} {'delta%':>8}")
    for r in rows:
        print(f"{r['test_id']:>3} {r['n']:>3} {r['C0']:>3} {r['w_est']:>10.3f} "
              f"{r['w_off']:>10.3f} {r['w_on']:>10.3f} {r['ratio_off']:>10.3f} "
              f"{r['ratio_on']:>10.3f} {r['delta_pct']:>8.2f}")

    print()
    print(f"median delta%: {median:.3f}")
    print(f"worst  delta%: {worst:.3f}  (test_id={worst_row['test_id']})")
    print(f"gate: median <= -5.0 and worst <= 1.0 -> "
          f"{'PASS' if median <= -5.0 and worst <= 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
