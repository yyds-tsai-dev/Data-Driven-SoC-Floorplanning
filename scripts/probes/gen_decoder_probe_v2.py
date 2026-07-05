"""Track-B exact decoder, iteration 2: noise-tolerant (pin-order repair).

Extends gen_decoder_probe (E1) with the fixes from the decoder-iteration brief:

  Item 1  pin-anchored order repair  -> _probe_repair.repair_pin_order
  Item 3  slack-aware compaction     -> --slack-aware (best-of-k axis flips on
                                        LOW-confidence locked pin pairs)
  Item 4  production refine post-pass -> --refine (refine.api.refine_layout with
                                        window-repack, on the LEGAL decoded layout)
  Item 5  full E2 diagnostic          -> default run: 100 cases, GNN 0514 hints,
                                        official offline scoring, honest fallback
                                        count, per-step attribution.

Item 2 (learned pair head) is MEASURED here (--pair-blend) but DEFAULT OFF: on
gnn_best_0514 the pair head's axis-choice agreement vs golden is 48.5% (chance),
vs 98.7% for geometric anchor extraction, so blending it regresses. The gate is
kept for a future better-trained checkpoint.

Reuses E1's shapes/snap/polish/scoring verbatim from gen_decoder_probe; the only
new decode step is the repair replacing the naive longest-path compaction.

Usage (from FloorSet/iccad2026contest, env sourced as in eval scripts):
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" uv run python \
    <repo>/scripts/probes/gen_decoder_probe_v2.py \
      --gnn-checkpoint <ckpt> [--refine] [--slack-aware] [--out out.json]
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

import torch

_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break

from iccad2026_evaluate import compute_total_score  # noqa: E402

import gen_decoder_probe as e1  # noqa: E402
from _probe_repair import (  # noqa: E402
    repair_pin_order,
    repair_pin_shapes,
    evict_overlaps,
)
from floorset_arch.refine import guards as refine_guards  # noqa: E402
from floorset_arch.refine.wirelength import hpwl  # noqa: E402

Rect = Tuple[float, float, float, float]
SEP_TOL = 1e-6


def _pins_of(preplaced, tpos, n) -> Tuple[Dict[int, float], Dict[int, float]]:
    xp: Dict[int, float] = {}
    yp: Dict[int, float] = {}
    for i in range(n):
        if preplaced[i]:
            tx, ty, tw, th = e1._target(tpos, i)
            if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                xp[i] = float(tx)
                yp[i] = float(ty)
    return xp, yp


def decode_v2(
    hints: List[Rect],
    sample,
    n: int,
    b2b_e,
    p2b_e,
    pin_l,
    do_polish: bool = True,
    do_refine: bool = False,
    refine_deadline: float = 2.0,
    args_slack_aware: bool = False,
) -> Tuple[Optional[List[Rect]], Dict[str, object]]:
    """Order-faithful decode with pin-order repair; reuses E1 shapes/snap/polish.

    Post-compaction pipeline is identical to E1's decode() from the repaired
    coords onward, so per-step attribution is comparable to the E1 table.
    """
    at = sample["input"][0][:n]
    cons = sample["input"][4][:n]
    golden = e1._golden_rects(sample, n)
    tpos = e1._opt_target_positions(sample, n, golden)

    fixed, preplaced, mib, cluster, boundary = e1._parse_constraints(cons, n)
    area_targets = [float(at[i]) for i in range(n)]

    trace: Dict[str, object] = {}

    # Step 3: shapes (E1, unchanged).
    ws, hs = e1.decode_shapes(hints, area_targets, fixed, preplaced, mib, tpos, n)

    # Steps 2+4: order DAGs (E1) then PIN-ORDER + PIN-SHAPE REPAIR (new).
    x_edges, y_edges = e1.build_order_dags(hints, ws, hs, n)
    x_pins, y_pins = _pins_of(preplaced, tpos, n)
    if args_slack_aware:
        rects, rtr = repair_pin_shapes(
            hints, x_edges, y_edges, ws, hs, x_pins, y_pins,
            preplaced, fixed, mib, area_targets, n,
            hpwl_fn=hpwl, hpwl_args=(b2b_e, p2b_e, pin_l),
        )
        trace["repair_status"] = rtr.get("status")
        trace["repair_iters"] = rtr.get("order_iters")
        trace["repair_edits"] = rtr.get("order_edits")
        trace["reshaped"] = rtr.get("reshaped")
    else:
        rects, rtr = repair_pin_order(
            x_edges, y_edges, ws, hs, x_pins, y_pins, preplaced, n
        )
        trace["repair_status"] = rtr.get("status")
        trace["repair_iters"] = rtr.get("iters")
        trace["repair_edits"] = rtr.get("repairs")
    if rects is None:
        trace["fail"] = "repair_" + str(rtr.get("status"))
        return None, trace
    trace["repair_legal"] = refine_guards.hard_legal(rects, at, cons, tpos)
    trace["hpwl_after_compact"] = hpwl(rects, b2b_e, p2b_e, pin_l)

    # Step 5b: boundary wall-snap (E1, unchanged).
    snapped = e1.boundary_snap(rects, boundary, preplaced, n)
    if refine_guards.hard_legal(snapped, at, cons, tpos):
        rects = snapped
        trace["snap_applied"] = True
    else:
        trace["snap_applied"] = False
    trace["hpwl_after_snap"] = hpwl(rects, b2b_e, p2b_e, pin_l)

    # Step 6: weighted-median projection polish (E1, unchanged, opt-in).
    projected = rects
    if do_polish:
        try:
            from floorset_arch.refine.constraint_graph import build_axis_dags
            from floorset_arch.refine.slack_solve import project_axis

            gx, gy = build_axis_dags(rects, cons, tpos, b2b_e, p2b_e, pin_l)
            dimx = [rects[i][2] for i in range(n)]
            dimy = [rects[i][3] for i in range(n)]
            ax = e1._anchors_by_block(n, rects, b2b_e, p2b_e, pin_l, axis=0)
            ay = e1._anchors_by_block(n, rects, b2b_e, p2b_e, pin_l, axis=1)
            xcoord = [rects[i][0] for i in range(n)]
            ycoord = [rects[i][1] for i in range(n)]
            nx, _ = project_axis(gx, xcoord, dimx, ax)
            ny, _ = project_axis(gy, ycoord, dimy, ay)
            cand = [(nx[i], ny[i], dimx[i], dimy[i]) for i in range(n)]
            if refine_guards.hard_legal(cand, at, cons, tpos):
                projected = cand
                trace["polish_applied"] = True
            else:
                trace["polish_applied"] = False
        except Exception as exc:  # noqa: BLE001
            trace["polish_applied"] = False
            trace["polish_error"] = repr(exc)

    # Item 4: production refine post-pass (window-repack + slack). Only on a
    # LEGAL layout (refine_layout requires and preserves legality; it returns
    # its input on any doubt, so it can only help).
    trace["refine_applied"] = False
    if do_refine and refine_guards.hard_legal(projected, at, cons, tpos):
        try:
            from floorset_arch.refine.api import refine_layout

            b2b = sample["input"][1]
            p2b = sample["input"][2]
            pins = sample["input"][3]
            dl = time.time() + refine_deadline
            refined = refine_layout(
                projected, at, cons, tpos, b2b, p2b, pins,
                deadline=dl, topo_deadline=dl,
            )
            if (refined is not projected
                    and refine_guards.hard_legal(refined, at, cons, tpos)):
                projected = refined
                trace["refine_applied"] = True
        except Exception as exc:  # noqa: BLE001
            trace["refine_error"] = repr(exc)

    # Final legality floor. Prefer the NON-destructive eviction floor (park only
    # the still-overlapping soft blocks on a shelf above the bbox, keep the ~95%
    # good decode) over E1's whole-layout shelf-pack. Eviction scores 3.7-6.8 vs
    # shelf-pack's flat 10.0 on the 6 residual GNN cases. Shelf-pack is the last
    # resort if eviction is somehow still illegal.
    if not refine_guards.hard_legal(projected, at, cons, tpos):
        ev, n_ev = evict_overlaps(projected, preplaced, n)
        if refine_guards.hard_legal(ev, at, cons, tpos):
            projected = ev
            trace["legal_fallback"] = "evict"
            trace["evicted"] = n_ev
        else:
            fb = e1._legal_shelf_fallback(
                hints, at, cons, tpos, preplaced, fixed, mib, area_targets, n
            )
            if fb is not None and refine_guards.hard_legal(fb, at, cons, tpos):
                projected = fb
                trace["legal_fallback"] = "shelf"
            else:
                trace["legal_fallback"] = "failed"
    trace["hpwl_final"] = hpwl(projected, b2b_e, p2b_e, pin_l)
    return projected, trace


def run(args) -> None:
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    if args.refine:
        # Enable the production window-repack + topo stages inside refine_layout.
        os.environ["FLOORSET_WINDOW_REPACK"] = "1"
        os.environ.setdefault("FLOORSET_TOPO_SEARCH", "0")
    from lite_dataset_test import FloorplanDatasetLiteTest

    ds = FloorplanDatasetLiteTest(str(args.data_path))
    idxs = list(range(len(ds)))
    if args.cases and args.cases < len(idxs):
        step = max(1, len(idxs) // args.cases)
        idxs = idxs[::step][: args.cases]

    from _probe_hints import GnnHintProvider

    prov = GnnHintProvider(args.gnn_checkpoint)

    rows = []
    costs = []
    bcs = []
    t0 = time.time()
    for idx in idxs:
        sample = ds[idx]
        n = e1._n_of(sample)
        b2b, p2b, pins = (
            sample["input"][1],
            sample["input"][2],
            sample["input"][3],
        )
        b2b_e, p2b_e, pin_l = e1._edge_lists(b2b, p2b, pins)

        ti = time.time()
        hints = prov.hints(sample, n)
        infer_ms = 1000.0 * (time.time() - ti)

        dt0 = time.time()
        pos, tr = decode_v2(
            hints, sample, n, b2b_e, p2b_e, pin_l,
            do_polish=not args.no_polish,
            do_refine=args.refine,
            refine_deadline=args.refine_deadline,
            args_slack_aware=args.slack_aware,
        )
        dec_ms = 1000.0 * (time.time() - dt0)

        if pos is None:
            sc = {
                "cost": 10.0, "feasible": False, "hpwl_gap": 0.0,
                "area_gap": 0.0, "v_rel": 1.0, "overlap": -1, "area_viol": -1,
                "dim_viol": -1, "boundary_v": -1, "grouping_v": -1,
                "mib_v": -1, "n_soft": 0,
            }
        else:
            sc = e1.score_case(sample, pos, n)

        row = {
            "idx": idx, "n": n, "band": e1._band(n),
            "dec_ms": round(dec_ms, 1), "infer_ms": round(infer_ms, 1),
            **sc, **{f"tr_{k}": v for k, v in tr.items()},
        }
        rows.append(row)
        costs.append(sc["cost"])
        bcs.append(n)
        if args.verbose:
            print(
                f"  idx={idx:3d} n={n:3d} cost={sc['cost']:.4f} "
                f"feas={int(sc['feasible'])} status={tr.get('repair_status')} "
                f"fb={tr.get('legal_fallback')} hpwl_gap={sc['hpwl_gap']:+.3f} "
                f"area_gap={sc['area_gap']:+.3f} dec_ms={dec_ms:.0f}",
                flush=True,
            )

    total = compute_total_score(costs, bcs)
    elapsed = time.time() - t0
    _report(rows, total, elapsed, args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"total": total, "rows": rows}, f, indent=2, default=str
            )
        print(f"wrote {args.out}")


def _report(rows, total, elapsed, args) -> None:
    def _fb(r):
        v = r.get("tr_legal_fallback")
        return v not in (None, False)

    feas = sum(1 for r in rows if r["feasible"])
    fb_fired = sum(1 for r in rows if _fb(r))
    # Repair-status census.
    status = defaultdict(int)
    for r in rows:
        status[r.get("tr_repair_status")] += 1
    # Legal-before-fallback = decoded hard-legal (repair produced legal AND no
    # fallback needed). We flag repair_legal True on the raw compaction.
    repair_legal = sum(1 for r in rows if r.get("tr_repair_legal") is True)
    no_fb = [r for r in rows if not _fb(r)]
    fb = [r for r in rows if _fb(r)]

    def mean(rr, key):
        return sum(r[key] for r in rr) / len(rr) if rr else float("nan")

    band_stats = defaultdict(lambda: {"costs": [], "ns": [], "feas": 0})
    for r in rows:
        b = band_stats[r["band"]]
        b["costs"].append(r["cost"])
        b["ns"].append(r["n"])
        b["feas"] += int(r["feasible"])

    print("\n" + "=" * 78)
    print(
        f"DECODER v2  cases={len(rows)}  refine={args.refine}  "
        f"slack_aware={args.slack_aware}  polish={not args.no_polish}"
    )
    print(
        f"WEIGHTED TOTAL (exp(n/12), no-runtime) = {total:.4f}   "
        f"feasible={feas}/{len(rows)}   wall={elapsed:.1f}s"
    )
    print(
        f"  legal-fallback fired = {fb_fired}/{len(rows)}   "
        f"repair-legal (no fallback needed) = {len(no_fb)}/{len(rows)}"
    )
    print(f"  repair status census: {dict(status)}")
    print(
        f"  NON-fallback cases: mean_cost={mean(no_fb,'cost'):.4f} "
        f"hpwl_gap={mean(no_fb,'hpwl_gap'):+.3f} "
        f"area_gap={mean(no_fb,'area_gap'):+.3f} v_rel={mean(no_fb,'v_rel'):.3f}"
    )
    if fb:
        print(
            f"  FALLBACK cases:     mean_cost={mean(fb,'cost'):.4f} "
            f"(n={len(fb)})"
        )
    print(
        f"  decode {mean(rows,'dec_ms'):.0f} ms/case   "
        f"inference {mean(rows,'infer_ms'):.0f} ms/case   "
        f"max_decode={max(r['dec_ms'] for r in rows):.0f} ms"
    )
    print("-" * 78)
    for band in ("n<60", "60-99", ">=100"):
        if band not in band_stats:
            continue
        st = band_stats[band]
        cs = st["costs"]
        sub = compute_total_score(cs, st["ns"])
        print(
            f"  {band:7s}: cases={len(cs):3d} feas={st['feas']:3d} "
            f"mean_cost={sum(cs)/len(cs):.4f} weighted={sub:.4f} "
            f"max_cost={max(cs):.4f}"
        )
    max_n = max(r["n"] for r in rows)
    contrib = [
        (r["cost"] * math.exp((r["n"] - max_n) / 12.0), r) for r in rows
    ]
    contrib.sort(key=lambda t: -t[0])
    print("-" * 78)
    print("  TOP-8 worst (by weighted contribution):")
    for c, r in contrib[:8]:
        print(
            f"    idx={r['idx']:3d} n={r['n']:3d} cost={r['cost']:.4f} "
            f"w={c:.4f} fb={r.get('tr_legal_fallback')} "
            f"status={r.get('tr_repair_status')} "
            f"hpwl_gap={r['hpwl_gap']:+.3f} area_gap={r['area_gap']:+.3f} "
            f"v_rel={r['v_rel']:.3f}"
        )
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-checkpoint", required=True)
    ap.add_argument("--cases", type=int, default=0)
    ap.add_argument(
        "--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../")
    )
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--refine", action="store_true",
                    help="item 4: production window-repack + slack post-pass")
    ap.add_argument("--refine-deadline", type=float, default=2.0)
    ap.add_argument("--slack-aware", action="store_true",
                    help="item 3: bounded critical-path axis flips (stub gate)")
    ap.add_argument("--pair-blend", action="store_true",
                    help="item 2: blend pair-head axis (MEASURED WORSE on 0514)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
