"""P2 prototype: analytic (QP) layout hints realized through the faithful decoder.

P1 (perturb_probe) killed local search over the faithful realizer (the hint-floor
is a ratchet). P2 asks the orthogonal question the NO-GO left open: can a
SELF-CONSISTENT analytic layout, realized faithfully (no SA), beat production on
any case -- especially the n>=100 tail?

The analytic placer is a classic quadratic global placement, one independent
sparse linear system per axis:
  - b2b connectivity  -> quadratic springs (clique model on the net graph)
  - p2b connectivity  -> springs pulling a block toward its fixed pin coordinate
  - boundary tags     -> wall-anchor springs (faithful has NO snap fallback, so
                         wall adhesion must be baked into the hint coordinates)
  - tiny regularization toward a grid init (islands / unconnected blocks spread)
  - preplaced blocks pinned to their golden coordinate (moved to the RHS)
then a light force-directed de-overlap so most pairs are cleanly separated (so
faithful's geo-axis extraction is effective, not the centroid fallback).

Shapes respect the faithful keep-path contract (reservation #3): exact area
w*h == area_target (square by default), MIB groups share one shape, fixed /
preplaced pass through their exact (w,h).

Realization + scoring reuse the FROZEN faithful path: gen_decoder_probe.decode(
realize="faithful") + score_case. gen_decoder_probe.py / src/ / FloorSet/ are
untouched. Paired vs the gate0 production cache (same D1 instrument).

NB (caveat): the cache baseline is the OLD production config (total 1.2724); this
is a prototype GATE, not a promotion claim -- a promotion-grade claim needs a
fresh production rerun.

Usage (from repo root, tcsh -> run via bash):
  cd FloorSet/iccad2026contest
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
      <repo>/scripts/probes/analytic_hints.py \
      --production-cache <gate0 cache.json> --out analytic.json
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
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- make src/ and this probe dir importable (mirrors gen_decoder_probe.py) ---
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
    _n_of,
    _golden_rects,
    _opt_target_positions,
    _edge_lists,
    _band,
    decode,
    score_case,
)
from floorset_arch.legalizer.column_slicing import (  # noqa: E402
    _parse_constraints,
    _target,
)

Rect = Tuple[float, float, float, float]

BOUND_LEFT = 1
BOUND_RIGHT = 2
BOUND_TOP = 4
BOUND_BOTTOM = 8


# =============================================================================
# Shapes (exact-area, faithful keep-path compatible)
# =============================================================================
def analytic_shapes(n, area_targets, fixed, preplaced, mib, tpos):
    """(w,h) per block: square exact-area by default; MIB groups share one
    square; fixed/preplaced pass through their exact target (w,h)."""
    w = [0.0] * n
    h = [0.0] * n
    # MIB shared side = sqrt(mean area) (members are equal-area by definition;
    # mean is defensive). All members land on an identical shape.
    mib_side: Dict[int, float] = {}
    mib_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        if mib[i] > 0:
            mib_members[mib[i]].append(i)
    for g, members in mib_members.items():
        a = sum(float(area_targets[i]) for i in members) / len(members)
        mib_side[g] = math.sqrt(max(a, 1e-9))
    for i in range(n):
        tx, ty, tw, th = _target(tpos, i)
        if (fixed[i] or preplaced[i]) and tw > 0 and th > 0:
            w[i], h[i] = float(tw), float(th)
            continue
        if mib[i] > 0:
            w[i] = h[i] = mib_side[mib[i]]
            continue
        s = math.sqrt(max(float(area_targets[i]), 1e-9))
        w[i] = h[i] = s
    return w, h


# =============================================================================
# Quadratic placement (one axis)
# =============================================================================
def solve_axis(n, b2b_e, p2b_pull, fixed_mask, fixed_coord, p0, reg, wall_pull):
    """Solve L c = b for the free blocks (fixed blocks pinned to fixed_coord and
    moved to the RHS). b2b_e: (i,j,w). p2b_pull: (block, coord, w). wall_pull:
    (block, target_coord, w). p0: grid-init anchor for the reg term."""
    L = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    for i, j, wt in b2b_e:
        if i < 0 or j < 0 or i >= n or j >= n or wt <= 0:
            continue
        L[i, i] += wt
        L[j, j] += wt
        L[i, j] -= wt
        L[j, i] -= wt
    for bl, coord, wt in p2b_pull:
        L[bl, bl] += wt
        b[bl] += wt * coord
    for bl, tgt, wt in wall_pull:
        L[bl, bl] += wt
        b[bl] += wt * tgt
    for i in range(n):
        L[i, i] += reg
        b[i] += reg * p0[i]
    free = [i for i in range(n) if not fixed_mask[i]]
    fix = [i for i in range(n) if fixed_mask[i]]
    if not free:
        return [float(fixed_coord[i]) for i in range(n)]
    Lff = L[np.ix_(free, free)]
    bf = b[free].copy()
    if fix:
        cfix = np.array([fixed_coord[i] for i in fix], dtype=float)
        bf = bf - L[np.ix_(free, fix)] @ cfix
    try:
        cf = np.linalg.solve(Lff, bf)
    except np.linalg.LinAlgError:
        cf = np.linalg.lstsq(Lff, bf, rcond=None)[0]
    c = [0.0] * n
    for k, i in enumerate(free):
        c[i] = float(cf[k])
    for i in fix:
        c[i] = float(fixed_coord[i])
    return c


def _spread(cx, cy, w, h, preplaced, n, iters, pad=0.0):
    """Light force-directed de-overlap: push overlapping pairs apart along the
    min-overlap axis; preplaced blocks are immovable."""
    for _ in range(iters):
        any_ov = False
        for i in range(n):
            for j in range(i + 1, n):
                ox = (min(cx[i] + w[i] / 2, cx[j] + w[j] / 2)
                      - max(cx[i] - w[i] / 2, cx[j] - w[j] / 2))
                oy = (min(cy[i] + h[i] / 2, cy[j] + h[j] / 2)
                      - max(cy[i] - h[i] / 2, cy[j] - h[j] / 2))
                if ox <= pad or oy <= pad:
                    continue
                any_ov = True
                pi = not preplaced[i]
                pj = not preplaced[j]
                if not (pi or pj):
                    continue
                if ox < oy:
                    d = ox / (2.0 if (pi and pj) else 1.0) + 1e-6
                    s = 1.0 if cx[i] <= cx[j] else -1.0
                    if pi:
                        cx[i] -= s * d
                    if pj:
                        cx[j] += s * d
                else:
                    d = oy / (2.0 if (pi and pj) else 1.0) + 1e-6
                    s = 1.0 if cy[i] <= cy[j] else -1.0
                    if pi:
                        cy[i] -= s * d
                    if pj:
                        cy[j] += s * d
        if not any_ov:
            break
    return cx, cy


def analytic_hints(sample, n, cfg) -> List[Rect]:
    at = sample["input"][0][:n]
    b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
    cons = sample["input"][4][:n]
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)
    fixed, preplaced, mib, cluster, boundary = _parse_constraints(cons, n)
    area_targets = [float(at[i]) for i in range(n)]

    w, h = analytic_shapes(n, area_targets, fixed, preplaced, mib, tpos)

    # Fixed-position (preplaced) centroids -> golden centroid.
    fixed_mask = [bool(preplaced[i]) for i in range(n)]
    fx = [0.0] * n
    fy = [0.0] * n
    for i in range(n):
        if preplaced[i]:
            tx, ty, tw, th = _target(tpos, i)
            fx[i] = float(tx) + float(tw) / 2.0
            fy[i] = float(ty) + float(th) / 2.0

    # Grid init (reg anchor): spread islands, PD system.
    cols = max(1, int(math.ceil(math.sqrt(n))))
    step = 1.3 * max(max(w), max(h))
    p0x = [(i % cols) * step for i in range(n)]
    p0y = [(i // cols) * step for i in range(n)]

    # p2b pulls per axis.
    p2b_px = []
    p2b_py = []
    for p, bl, wt in p2b_e:
        if p < 0 or bl < 0 or bl >= n or p >= len(pin_l) or wt <= 0:
            continue
        p2b_px.append((bl, pin_l[p][0], wt * cfg["pin_w"]))
        p2b_py.append((bl, pin_l[p][1], wt * cfg["pin_w"]))

    reg = cfg["reg"]
    # Round 0: no walls.
    cx = solve_axis(n, b2b_e, p2b_px, fixed_mask, fx, p0x, reg, [])
    cy = solve_axis(n, b2b_e, p2b_py, fixed_mask, fy, p0y, reg, [])

    # Wall-anchor rounds: pull boundary-tagged edges to the current bbox walls.
    for _ in range(cfg["wall_rounds"]):
        x0 = min(cx[i] - w[i] / 2 for i in range(n))
        x1 = max(cx[i] + w[i] / 2 for i in range(n))
        y0 = min(cy[i] - h[i] / 2 for i in range(n))
        y1 = max(cy[i] + h[i] / 2 for i in range(n))
        wall_x = []
        wall_y = []
        for i in range(n):
            if preplaced[i]:
                continue
            code = boundary[i]
            if code == 0:
                continue
            if code & BOUND_LEFT:
                wall_x.append((i, x0 + w[i] / 2, cfg["wall_w"]))
            if code & BOUND_RIGHT:
                wall_x.append((i, x1 - w[i] / 2, cfg["wall_w"]))
            if code & BOUND_BOTTOM:
                wall_y.append((i, y0 + h[i] / 2, cfg["wall_w"]))
            if code & BOUND_TOP:
                wall_y.append((i, y1 - h[i] / 2, cfg["wall_w"]))
        cx = solve_axis(n, b2b_e, p2b_px, fixed_mask, fx, p0x, reg, wall_x)
        cy = solve_axis(n, b2b_e, p2b_py, fixed_mask, fy, p0y, reg, wall_y)

    # De-overlap so most pairs are cleanly separated for geo-axis extraction.
    cx, cy = _spread(cx, cy, w, h, preplaced, n, cfg["spread_iters"])

    return [(cx[i] - w[i] / 2.0, cy[i] - h[i] / 2.0, w[i], h[i]) for i in range(n)]


# =============================================================================
# Driver (paired vs production cache, D1-style)
# =============================================================================
def run(args):
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    with open(args.production_cache) as f:
        cache = {int(k): [tuple(r) for r in v] for k, v in json.load(f).items()}
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))

    idxs = [i for i in range(len(ds)) if i in cache]
    if args.cases and args.cases < len(idxs):
        step = max(1, len(idxs) // args.cases)
        idxs = idxs[::step][: args.cases]

    cfg = dict(pin_w=args.pin_w, reg=args.reg, wall_w=args.wall_w,
               wall_rounds=args.wall_rounds, spread_iters=args.spread_iters)

    t0 = time.time()
    rows = []
    for idx in idxs:
        sample = ds[idx]
        n = _n_of(sample)
        b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
        b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)

        prod = cache[idx]
        base = score_case(sample, prod, n)

        hints = analytic_hints(sample, n, cfg)
        pos, tr = decode(hints, sample, n, b2b_e, p2b_e, pin_l,
                         do_polish=False, realize=args.realize,
                         anchor_hint_weight=args.anchor_hint_weight,
                         anchor_wall_weight=args.anchor_wall_weight)
        if pos is None:
            sc = {"cost": 10.0, "feasible": False, "hpwl_gap": 0.0,
                  "area_gap": 0.0, "v_rel": 1.0, "boundary_v": -1,
                  "grouping_v": -1, "mib_v": -1, "overlap": -1}
        else:
            sc = score_case(sample, pos, n)
        rows.append(dict(
            idx=idx, n=n, band=_band(n), cost=sc["cost"],
            feasible=sc["feasible"], hpwl_gap=sc["hpwl_gap"],
            area_gap=sc["area_gap"], v_rel=sc["v_rel"],
            boundary_v=sc["boundary_v"], grouping_v=sc["grouping_v"],
            mib_v=sc["mib_v"],
            base_cost=base["cost"], base_hpwl_gap=base["hpwl_gap"],
            base_area_gap=base["area_gap"], base_v_rel=base["v_rel"],
            fallback=tr.get("legal_fallback"),
        ))
    elapsed = time.time() - t0
    _report(rows, elapsed, args, cfg)


def _report(rows, elapsed, args, cfg):
    bcs = [r["n"] for r in rows]
    dc = [r["cost"] for r in rows]
    bc = [r["base_cost"] for r in rows]
    port = [min(r["cost"], r["base_cost"]) for r in rows]
    feas = sum(1 for r in rows if r["feasible"])
    total = compute_total_score(dc, bcs)
    base_total = compute_total_score(bc, bcs)
    port_total = compute_total_score(port, bcs)
    wins = sum(1 for r in rows if r["cost"] < r["base_cost"] - 1e-9)
    losses = sum(1 for r in rows if r["cost"] > r["base_cost"] + 1e-9)
    fb = sum(1 for r in rows if r["fallback"])

    print("\n" + "=" * 86)
    print(f"ANALYTIC-HINT (QP) + faithful realize   cases={len(rows)}   wall={elapsed:.1f}s")
    print(f"  cfg: pin_w={cfg['pin_w']} reg={cfg['reg']} wall_w={cfg['wall_w']} "
          f"wall_rounds={cfg['wall_rounds']} spread_iters={cfg['spread_iters']}")
    print("=" * 86)
    print(f"  PAIRED: baseline(prod cache)={base_total:.4f}   decoded(analytic)={total:.4f}   "
          f"portfolio(min)={port_total:.4f}")
    print(f"  analytic wins {wins} / loses {losses} / ties {len(rows)-wins-losses}   "
          f"feasible={feas}/{len(rows)}   legal-fallback fired={fb}   "
          f"(deploy delta = {port_total - base_total:+.4f})")
    print("-" * 86)
    for band in ("n<60", "60-99", ">=100"):
        br = [r for r in rows if r["band"] == band]
        if not br:
            continue
        bt = compute_total_score([r["cost"] for r in br], [r["n"] for r in br])
        bbt = compute_total_score([r["base_cost"] for r in br], [r["n"] for r in br])
        bw = sum(1 for r in br if r["cost"] < r["base_cost"] - 1e-9)
        mh = sum(r["hpwl_gap"] for r in br) / len(br)
        ma = sum(r["area_gap"] for r in br) / len(br)
        mv = sum(r["v_rel"] for r in br) / len(br)
        print(f"  {band:6s}: cases={len(br):3d}  analytic_wt={bt:.4f}  base_wt={bbt:.4f}  "
              f"wins={bw:3d}  mean hpwl_gap={mh:+.3f} area_gap={ma:+.3f} v_rel={mv:.3f}")
    # tail detail.
    print("-" * 86)
    print("  TAIL n>=100 per-case (analytic vs base):")
    tail = sorted([r for r in rows if r["n"] >= 100], key=lambda r: r["idx"])
    for r in tail:
        flag = "WIN " if r["cost"] < r["base_cost"] - 1e-9 else "loss"
        print(f"    idx={r['idx']:3d} n={r['n']:3d} {flag} "
              f"analytic={r['cost']:.4f}(hg{r['hpwl_gap']:+.2f} ag{r['area_gap']:+.2f} "
              f"vr{r['v_rel']:.2f} bnd{r['boundary_v']} grp{r['grouping_v']}) "
              f"base={r['base_cost']:.4f}(hg{r['base_hpwl_gap']:+.2f} ag{r['base_area_gap']:+.2f} "
              f"vr{r['base_v_rel']:.2f})")
    print("=" * 86)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(cfg=cfg, base_total=base_total, decoded_total=total,
                           portfolio_total=port_total, wins=wins, losses=losses,
                           feasible=feas, rows=rows), f, indent=2, default=str)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-cache", required=True)
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--cases", type=int, default=0)
    ap.add_argument("--pin-w", dest="pin_w", type=float, default=1.0)
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--wall-w", dest="wall_w", type=float, default=2.0)
    ap.add_argument("--wall-rounds", dest="wall_rounds", type=int, default=2)
    ap.add_argument("--spread-iters", dest="spread_iters", type=int, default=25)
    ap.add_argument("--realize", choices=["compact", "faithful", "anchored"],
                    default="faithful",
                    help="realization mode for the analytic hints (A2 uses "
                         "'anchored' for the compaction-capable realizer)")
    ap.add_argument("--anchor-hint-weight", dest="anchor_hint_weight",
                    type=float, default=2.0)
    ap.add_argument("--anchor-wall-weight", dest="anchor_wall_weight",
                    type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
