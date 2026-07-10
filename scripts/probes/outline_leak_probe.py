"""Outline-leak probe: how well can the golden bbox outline (W, H) be
inferred from the LEGAL inputs solve() actually receives, without touching
the golden layout itself as an estimator input?

Hypothesis under test: pins_pos / preplaced target_positions / area_targets
are placed in the SAME coordinate frame as the golden bbox (they are derived
from the golden layout by the dataset generator), so their extents may
already leak most of the outline that the legalizer has to (re)discover.

Estimators (all built ONLY from solve()-visible inputs):
  a. pins extent      : W_pin = max(pin_x), H_pin = max(pin_y)
  b. preplaced extent  : max(x+w), max(y+h) over preplaced blocks -- only
                          available in the "with target_positions" scenario
                          (the real evaluator synthesizes preplaced target
                          positions from the golden bbox; the validation
                          loader's target_positions is None, so this
                          scenario is checked separately from "no target
                          positions").
  c. area closure      : W*H ~= sum(area_targets) / (1 - ws), where ws is a
                          GLOBAL PRIOR median whitespace fraction computed
                          from golden (once, across all 100 cases) -- this is
                          declared explicitly as a prior, not a per-case
                          leak, and reported separately.
  d. combo             : W_est = max(a_W, b_W) with aspect taken from the
                          pins bbox ratio and rescaled so W_est*H_est matches
                          the area-closure product from (c).

Golden is used ONLY as ground truth to score the estimators above -- never
as an estimator input (ws prior excepted, and clearly labeled as such).

Usage (from repo root, tcsh):
  cd FloorSet/iccad2026contest
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
      <repo>/scripts/probes/outline_leak_probe.py [--cases N] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

_THIS = Path(__file__).resolve()
_REPO = None
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        _REPO = _p
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
if _REPO is None:
    _REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
    sys.path.insert(0, str(_REPO / "src"))

Rect = Tuple[float, float, float, float]


def _band(n: int) -> str:
    if n < 60:
        return "n<60"
    if n < 100:
        return "60-99"
    return ">=100"


def _n_of(sample) -> int:
    at = sample["input"][0]
    return int((at != -1).sum().item())


def _golden_rects(sample, n: int) -> List[Rect]:
    """Golden layout as per-block bbox rects (evaluator _extract_baseline)."""
    polys = sample["label"][0]
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


def _opt_target_positions(sample, n: int, golden: List[Rect]) -> torch.Tensor:
    """Build the opt_target_pos tensor exactly as ContestEvaluator.evaluate:
    fixed blocks get (w,h); preplaced blocks get (x,y,w,h); else -1. This is
    what the REAL evaluator gives solve() as target_positions (derived from
    golden by the harness, not by us) -- the "with target_positions"
    scenario below."""
    cons = sample["input"][4]
    tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and cons[i, 0] != 0
        is_preplaced = nc > 1 and cons[i, 1] != 0
        gx, gy, gw, gh = golden[i]
        if is_preplaced:
            tpos[i] = torch.tensor([gx, gy, gw, gh])
        elif is_fixed:
            tpos[i, 2] = gw
            tpos[i, 3] = gh
    return tpos


def _pctile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(errs: Sequence[float]) -> Dict[str, float]:
    if not errs:
        return {"median": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "median": statistics.median(errs),
        "p90": _pctile(errs, 0.90),
        "n": len(errs),
    }


def run(args) -> None:
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))

    all_idx = list(range(len(ds)))
    if args.cases and args.cases < len(all_idx):
        step = max(1, len(all_idx) // args.cases)
        idxs = all_idx[::step][: args.cases]
    else:
        idxs = all_idx

    # ---- Pass 1: golden geometry + global whitespace prior. ------------
    cases = []
    ws_all: List[float] = []
    for idx in idxs:
        sample = ds[idx]
        n = _n_of(sample)
        golden = _golden_rects(sample, n)
        gx0 = min(r[0] for r in golden)
        gy0 = min(r[1] for r in golden)
        gW = max(r[0] + r[2] for r in golden)
        gH = max(r[1] + r[3] for r in golden)
        area_sum = sum(r[2] * r[3] for r in golden)
        ws = 1.0 - (area_sum / (gW * gH)) if gW > 0 and gH > 0 else float("nan")
        ws_all.append(ws)
        cases.append(dict(idx=idx, n=n, sample=sample, golden=golden,
                           gx0=gx0, gy0=gy0, gW=gW, gH=gH, area_sum=area_sum,
                           ws=ws))

    ws_median = statistics.median(ws_all)

    # ---- Pass 2: legal-input estimators, scored against golden. --------
    rows = []
    for c in cases:
        sample, n = c["sample"], c["n"]
        at, b2b, p2b, pins, cons = sample["input"]
        area_targets = [float(at[i]) for i in range(n)]
        pins_l = pins.tolist()  # [(x, y), ...] -- absolute pin coords

        # (a) pins extent -- always available (pins_pos is a solve() arg).
        if pins_l:
            px = [p[0] for p in pins_l]
            py = [p[1] for p in pins_l]
            W_pin, H_pin = max(px), max(py)
            minx_pin, miny_pin = min(px), min(py)
        else:
            W_pin = H_pin = minx_pin = miny_pin = 0.0

        # pins-on-boundary fraction (within +-1% of golden W/H) -- purely
        # diagnostic, uses golden only to VALIDATE, not to estimate.
        gW, gH = c["gW"], c["gH"]
        tol_w, tol_h = 0.01 * gW, 0.01 * gH
        on_wall = sum(
            1 for (x, y) in pins_l
            if abs(x - gW) <= tol_w or abs(x) <= tol_w
            or abs(y - gH) <= tol_h or abs(y) <= tol_h
        )
        pin_wall_frac = on_wall / len(pins_l) if pins_l else float("nan")

        # (b) preplaced extent -- only in the "with target_positions"
        # scenario (real evaluator supplies this; validation loader's
        # target_positions is None -> scenario "no_tpos" below has no (b)).
        tpos = _opt_target_positions(sample, n, c["golden"])
        preplaced_rows = [
            tpos[i] for i in range(n)
            if float(cons[i, 1]) != 0 and float(tpos[i, 2]) > 0
        ]
        if preplaced_rows:
            W_pre = max(float(r[0] + r[2]) for r in preplaced_rows)
            H_pre = max(float(r[1] + r[3]) for r in preplaced_rows)
            has_preplaced = True
        else:
            W_pre = H_pre = 0.0
            has_preplaced = False

        # (c) area closure using the GLOBAL ws prior (not per-case golden).
        area_sum = sum(area_targets)
        side_area = math.sqrt(area_sum / max(1e-6, (1.0 - ws_median)))

        # aspect prior for splitting side_area into (W,H): from pins bbox
        # ratio when available, else 1:1.
        if W_pin > 1e-6 and H_pin > 1e-6:
            aspect = W_pin / H_pin
        else:
            aspect = 1.0
        W_area = side_area * math.sqrt(aspect)
        H_area = side_area / math.sqrt(aspect)

        # Empirically (measured below) pins/preplaced extents consistently
        # OVER-estimate the golden outline (pins sit on a padded fixed
        # canvas, not a tight golden bbox), so naive max(a,b,c) is dominated
        # by the weaker signal and hurts accuracy. The combo instead uses
        # the area-closure estimate (c) as primary, and only raises it to
        # meet a pins/preplaced floor when that floor legitimately exceeds
        # the area estimate (area estimate under-shooting a hard extent).
        W_est_no = max(W_area, W_pin) if W_pin > W_area * 1.5 else W_area
        H_est_no = max(H_area, H_pin) if H_pin > H_area * 1.5 else H_area

        W_est_with = max(W_area, W_pin, W_pre) if max(W_pin, W_pre) > W_area * 1.5 else W_area
        H_est_with = max(H_area, H_pin, H_pre) if max(H_pin, H_pre) > H_area * 1.5 else H_area

        def relerr(est, gt):
            return abs(est - gt) / gt if gt > 1e-9 else float("nan")

        row = dict(
            idx=c["idx"], n=n, band=_band(n),
            gW=gW, gH=gH, gx0=c["gx0"], gy0=c["gy0"],
            W_pin=W_pin, H_pin=H_pin, minx_pin=minx_pin, miny_pin=miny_pin,
            pin_wall_frac=pin_wall_frac,
            has_preplaced=has_preplaced, W_pre=W_pre, H_pre=H_pre,
            W_area=W_area, H_area=H_area,
            err_pin_W=relerr(W_pin, gW), err_pin_H=relerr(H_pin, gH),
            err_pre_W=relerr(W_pre, gW) if has_preplaced else float("nan"),
            err_pre_H=relerr(H_pre, gH) if has_preplaced else float("nan"),
            err_area_W=relerr(W_area, gW), err_area_H=relerr(H_area, gH),
            W_est_no=W_est_no, H_est_no=H_est_no,
            err_no_W=relerr(W_est_no, gW), err_no_H=relerr(H_est_no, gH),
            W_est_with=W_est_with, H_est_with=H_est_with,
            err_with_W=relerr(W_est_with, gW), err_with_H=relerr(H_est_with, gH),
        )
        rows.append(row)

    # ---- Aggregate. ------------------------------------------------------
    def both_under(rows_, wkey, hkey, thresh):
        return sum(
            1 for r in rows_
            if r[wkey] == r[wkey] and r[hkey] == r[hkey]  # not NaN
            and r[wkey] < thresh and r[hkey] < thresh
        )

    def agg_block(rows_, tag):
        errs_w = [r[f"err_{tag}_W"] for r in rows_ if r[f"err_{tag}_W"] == r[f"err_{tag}_W"]]
        errs_h = [r[f"err_{tag}_H"] for r in rows_ if r[f"err_{tag}_H"] == r[f"err_{tag}_H"]]
        return {
            "W": _stats(errs_w),
            "H": _stats(errs_h),
            "both_lt_2pct": both_under(rows_, f"err_{tag}_W", f"err_{tag}_H", 0.02),
            "both_lt_5pct": both_under(rows_, f"err_{tag}_W", f"err_{tag}_H", 0.05),
        }

    n_cases = len(rows)
    with_rows = [r for r in rows if r["has_preplaced"]]

    summary = {
        "n_cases": n_cases,
        "ws_median_prior": ws_median,
        "ws_all_pctile": {
            "p10": _pctile(ws_all, 0.10),
            "median": ws_median,
            "p90": _pctile(ws_all, 0.90),
        },
        "golden_origin_check": {
            "max_gx0": max(r["gx0"] for r in rows),
            "max_gy0": max(r["gy0"] for r in rows),
        },
        "pin_wall_frac_median": statistics.median(
            [r["pin_wall_frac"] for r in rows if r["pin_wall_frac"] == r["pin_wall_frac"]]
        ),
        "estimator_a_pins_only": agg_block(rows, "pin"),
        "estimator_b_preplaced_only": {
            "n_cases_with_preplaced": len(with_rows),
            **(agg_block(with_rows, "pre") if with_rows else {}),
        },
        "estimator_c_area_closure_prior": agg_block(rows, "area"),
        "scenario_no_target_positions_combo": agg_block(rows, "no"),
        "scenario_with_target_positions_combo": agg_block(rows, "with"),
        "by_band": {},
    }

    for band in ("n<60", "60-99", ">=100"):
        band_rows = [r for r in rows if r["band"] == band]
        if not band_rows:
            continue
        summary["by_band"][band] = {
            "n_cases": len(band_rows),
            "no_tpos_combo": agg_block(band_rows, "no"),
            "with_tpos_combo": agg_block(band_rows, "with"),
        }

    # Leak-grade verdict.
    def grade(both2, both5, total):
        if both2 >= 0.80 * total:
            return "A"
        if both5 >= 0.50 * total:
            return "B"
        return "C"

    no_g = summary["scenario_no_target_positions_combo"]
    with_g = summary["scenario_with_target_positions_combo"]
    summary["verdict"] = {
        "no_tpos_grade": grade(no_g["both_lt_2pct"], no_g["both_lt_5pct"], n_cases),
        "with_tpos_grade": grade(with_g["both_lt_2pct"], with_g["both_lt_5pct"],
                                  n_cases if not with_rows else len(with_rows)),
    }

    print(json.dumps(summary, indent=2, default=str))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)
        print(f"\nwrote {args.out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--cases", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
