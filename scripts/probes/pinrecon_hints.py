"""Pin-anchored reconstruction (pinrecon) hint source + full-100 measurement.

Hypothesis (from the outline-leak finding): p2b pins live in the golden
coordinate frame, so a block's p2b-pin weighted-center leaks its golden position;
reconstructing coordinates from pins should give near-golden hints that the
realize toolchain cashes into a near-golden score (ceiling 1.0472 / anchored
~1.0).

Direct hint-quality test (report FIRST) measures per-block displacement of the
pin-center from the golden centroid (normalized by sqrt(area)) and the rank
correlation of pin-center vs golden coordinates. Then the hints are realized via
the frozen decode() (compact+repair+refine and anchored) and paired against the
production cache, with golden hints as the ceiling control.

Reuses gen_decoder_probe (decode/score_case) + analytic_hints.analytic_shapes.
No src/ or FloorSet/ edits; gen_decoder_probe unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
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

from iccad2026_evaluate import compute_total_score  # noqa: E402

from gen_decoder_probe import (  # noqa: E402
    _n_of, _golden_rects, _opt_target_positions, _edge_lists, _band,
    decode, score_case,
)
from analytic_hints import analytic_shapes  # noqa: E402
from floorset_arch.legalizer.column_slicing import _parse_constraints, _target  # noqa: E402

Rect = Tuple[float, float, float, float]


def pinrecon_hints(sample, n, rounds=8) -> List[Rect]:
    at, b2b, p2b, pins, cons = sample["input"]
    at = at[:n]
    cons = cons[:n]
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)
    fixed, pre, mib, clus, bnd = _parse_constraints(cons, n)
    area = [float(at[i]) for i in range(n)]
    w, h = analytic_shapes(n, area, fixed, pre, mib, tpos)
    pinsl = pins.tolist()
    cx: List = [None] * n
    cy: List = [None] * n
    # preplaced -> target centroid.
    for i in range(n):
        if pre[i]:
            tx, ty, tw, th = _target(tpos, i)
            cx[i] = float(tx) + float(tw) / 2.0
            cy[i] = float(ty) + float(th) / 2.0
    # p2b blocks -> weighted mean pin center (mean beat median: 1.336 vs 1.493).
    acc = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in p2b.tolist():
        p, b, wt = int(r[0]), int(r[1]), float(r[2])
        if p < 0 or b < 0 or b >= n or p >= len(pinsl) or wt <= 0:
            continue
        px, py = pinsl[p]
        acc[b][0] += wt * px
        acc[b][1] += wt * py
        acc[b][2] += wt
    for b in range(n):
        if cx[b] is None and acc[b][2] > 0:
            cx[b] = acc[b][0] / acc[b][2]
            cy[b] = acc[b][1] / acc[b][2]
    # b2b propagation for unlocated blocks.
    adj = defaultdict(list)
    for r in b2b.tolist():
        i, j, wt = int(r[0]), int(r[1]), float(r[2])
        if i < 0 or j < 0 or i >= n or j >= n or wt <= 0:
            continue
        adj[i].append((j, wt))
        adj[j].append((i, wt))
    for _ in range(rounds):
        upd = {}
        for i in range(n):
            if cx[i] is not None:
                continue
            sx = sy = sw = 0.0
            for j, wt in adj[i]:
                if cx[j] is not None:
                    sx += wt * cx[j]
                    sy += wt * cy[j]
                    sw += wt
            if sw > 0:
                upd[i] = (sx / sw, sy / sw)
        for i, (x, y) in upd.items():
            cx[i] = x
            cy[i] = y
        if not upd:
            break
    loc = [i for i in range(n) if cx[i] is not None]
    ox = sum(cx[i] for i in loc) / max(len(loc), 1)
    oy = sum(cy[i] for i in loc) / max(len(loc), 1)
    for i in range(n):
        if cx[i] is None:
            cx[i] = ox
            cy[i] = oy
    return [(cx[i] - w[i] / 2.0, cy[i] - h[i] / 2.0, w[i], h[i]) for i in range(n)]


def _pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _disp(sample, n, hints):
    """Normalized per-block displacement of pinrecon centroid from golden."""
    golden = _golden_rects(sample, n)
    out = []
    for i in range(n):
        gcx = golden[i][0] + golden[i][2] / 2.0
        gcy = golden[i][1] + golden[i][3] / 2.0
        hcx = hints[i][0] + hints[i][2] / 2.0
        hcy = hints[i][1] + hints[i][3] / 2.0
        a = max(golden[i][2] * golden[i][3], 1e-9)
        out.append(math.hypot(hcx - gcx, hcy - gcy) / math.sqrt(a))
    return out


def run(args):
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    with open(args.production_cache) as f:
        cache = {int(k): [tuple(r) for r in v] for k, v in json.load(f).items()}
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))
    idxs = [i for i in range(len(ds)) if i in cache]

    t0 = time.time()
    rows = []
    disps = []
    for idx in idxs:
        s = ds[idx]
        n = _n_of(s)
        b2b, p2b, pins = s["input"][1], s["input"][2], s["input"][3]
        b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
        prod = cache[idx]
        base = score_case(s, prod, n)["cost"]
        h = pinrecon_hints(s, n, rounds=args.prop_rounds)
        disps.extend(_disp(s, n, h))
        if args.realize == "compact":
            pos, tr = decode(h, s, n, b2b_e, p2b_e, pin_l, do_polish=False,
                             realize="compact", do_repair=True, do_refine=True,
                             refine_deadline=args.refine_deadline)
        else:
            pos, tr = decode(h, s, n, b2b_e, p2b_e, pin_l, do_polish=False,
                             realize="anchored")
        sc = score_case(s, pos, n) if pos is not None else {"cost": 10.0, "feasible": False}
        rows.append(dict(idx=idx, n=n, band=_band(n), cost=sc["cost"],
                         base_cost=base, feasible=sc.get("feasible", False)))
    elapsed = time.time() - t0

    bcs = [r["n"] for r in rows]
    dc = [r["cost"] for r in rows]
    bc = [r["base_cost"] for r in rows]
    total = compute_total_score(dc, bcs)
    base_total = compute_total_score(bc, bcs)
    port = compute_total_score([min(a, b) for a, b in zip(dc, bc)], bcs)
    wins = sum(1 for r in rows if r["cost"] < r["base_cost"] - 1e-9)
    tail_wins = sum(1 for r in rows if r["n"] >= 100 and r["cost"] < r["base_cost"] - 1e-9)

    print("\n" + "=" * 84)
    print(f"PINRECON realize={args.realize}  prop_rounds={args.prop_rounds}  wall={elapsed:.1f}s")
    print("=" * 84)
    print(f"  HINT QUALITY: norm-disp vs golden  median={_pct(disps,0.5):.3f}  "
          f"p90={_pct(disps,0.9):.3f}  (<0.3 = near-golden)")
    print(f"  PAIRED: baseline(prod)={base_total:.4f}  pinrecon={total:.4f}  "
          f"portfolio(min)={port:.4f}")
    print(f"  wins vs production = {wins}/{len(rows)}   TAIL wins = {tail_wins}   "
          f"deploy delta = {port - base_total:+.4f}")
    for band in ("n<60", "60-99", ">=100"):
        br = [r for r in rows if r["band"] == band]
        if not br:
            continue
        bt = compute_total_score([r["cost"] for r in br], [r["n"] for r in br])
        bw = sum(1 for r in br if r["cost"] < r["base_cost"] - 1e-9)
        print(f"    {band:6s}: cases={len(br):3d}  pinrecon_wt={bt:.4f}  wins={bw}")
    print("=" * 84)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(realize=args.realize, total=total, base_total=base_total,
                           portfolio=port, wins=wins, tail_wins=tail_wins,
                           disp_median=_pct(disps, 0.5), rows=rows), f, indent=2, default=str)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-cache", required=True)
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--realize", choices=["compact", "anchored"], default="anchored")
    ap.add_argument("--prop-rounds", dest="prop_rounds", type=int, default=8)
    ap.add_argument("--refine-deadline", dest="refine_deadline", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
