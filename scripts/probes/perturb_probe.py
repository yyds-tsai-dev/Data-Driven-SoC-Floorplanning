"""P1 search gate: perturbation-smoothness probe for the faithful realizer.

Gate 1 (gen_decoder_probe --realize faithful) proved the decoder is a FAITHFUL
realizer: a legal exact-area hint round-trips to itself (tax +0.000). Fidelity is
necessary but not sufficient for local search -- the landscape must also be
SMOOTH (small move -> small dCost, no cliffs) and contain PREY (a measurable
fraction of improving moves), especially on the n>=100 tail (77% of the prize
pool). This probe measures that.

It CONSUMES the frozen faithful decode components from gen_decoder_probe (import
reuse -- decode_shapes / build_order_dags / longest_path_coords / score_case) and
orchestrates them into a single-seam `reflow()` that mirrors decode(realize=
"faithful") exactly on the unperturbed input (asserted at startup), with three
perturbation seams:

  (a) order swap    -- flip one tight (touching) same-axis separation edge, then
                       faithful reflow (floor = original coords; longest-path max
                       resolves the induced conflict).
  (b) centroid swap -- two similar-area free blocks trade hint centroids; the
                       geometry re-extracts the pairwise order naturally.
  (c) shape perturb -- one free block's aspect +/-10-20% at EXACT area
                       (w*h == area_target by construction).

Reports per move-type x band: dCost distribution (median / p90 |dCost|),
improving rate, legal rate; a k=1,2,4,8 stacked-perturbation curve; and a single
GO / NO-GO line (GO iff the n>=100 tail has a measurable improving rate AND the
tail single-move median |dCost| <= 0.05).

Does NOT modify gen_decoder_probe.py (faithful is frozen), src/, or FloorSet/.

Usage (from repo root, tcsh -> run via bash):
  cd FloorSet/iccad2026contest
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
      <repo>/scripts/probes/perturb_probe.py \
      --production-cache <gate0 cache.json> --out perturb.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Frozen faithful components (import reuse; no logic copied).
from gen_decoder_probe import (  # noqa: E402
    _n_of,
    _golden_rects,
    _opt_target_positions,
    _edge_lists,
    _band,
    decode,
    decode_shapes,
    build_order_dags,
    longest_path_coords,
    score_case,
)
from floorset_arch.legalizer.column_slicing import (  # noqa: E402
    _parse_constraints,
    _target,
)
from floorset_arch.refine import guards as refine_guards  # noqa: E402

Rect = Tuple[float, float, float, float]
TIGHT_TOL = 1e-4
IMPROVE_TOL = 1e-9


# =============================================================================
# Faithful reflow (single seam; identical to decode(realize="faithful") on legal
# input, plus an optional edge flip for the order-swap move class)
# =============================================================================
def _decode_shapes_ctx(ctx, hints):
    return decode_shapes(
        hints, ctx["area_targets"], ctx["fixed"], ctx["preplaced"],
        ctx["mib"], ctx["tpos"], ctx["n"], keep_valid_hint_shapes=True,
    )


def reflow(ctx, hints, flips=None) -> Optional[List[Rect]]:
    """Realize ``hints`` under the probe's realize mode.

    faithful (default): keep-shape + geo-axis + hint-floored longest path, with
    optional single-edge flips injected between edge-build and reflow.
    anchored: route through the frozen decode(realize="anchored") -- order-swap
    flips are applied as a coordinate swap of the tight pair (anchored re-extracts
    order from geometry, so a coordinate swap flips it). Either way returns None
    if the result cycles, is hard-illegal, or (anchored) needs the shelf floor --
    all counted as a rejected move; the probe never scores a shelf-packed layout."""
    if ctx.get("realize") == "anchored":
        h = [tuple(r) for r in hints]
        if flips:
            for axis, u, v in flips:
                hu = list(h[u])
                hv = list(h[v])
                hu[axis], hv[axis] = hv[axis], hu[axis]
                h[u], h[v] = tuple(hu), tuple(hv)
        pos, tr = decode(h, ctx["sample"], ctx["n"], ctx["b2b_e"],
                         ctx["p2b_e"], ctx["pin_l"], do_polish=False,
                         realize="anchored")
        if pos is None or tr.get("legal_fallback"):
            return None
        return pos
    n = ctx["n"]
    ws, hs = _decode_shapes_ctx(ctx, hints)
    x_edges, y_edges = build_order_dags(hints, ws, hs, n, geo_faithful=True)
    if flips:
        x_edges = list(x_edges)
        y_edges = list(y_edges)
        for axis, u, v in flips:
            if axis == 0:
                x_edges = [e for e in x_edges if {e[0], e[1]} != {u, v}]
                x_edges.append((v, u, ws[v]))
            else:
                y_edges = [e for e in y_edges if {e[0], e[1]} != {u, v}]
                y_edges.append((v, u, hs[v]))
    xf = [hints[i][0] for i in range(n)]
    yf = [hints[i][1] for i in range(n)]
    xc = longest_path_coords(n, x_edges, ctx["x_pins"], floor=xf)
    yc = longest_path_coords(n, y_edges, ctx["y_pins"], floor=yf)
    if xc is None or yc is None:
        return None
    rects = [(xc[i], yc[i], ws[i], hs[i]) for i in range(n)]
    if not refine_guards.hard_legal(rects, ctx["at"], ctx["cons"], ctx["tpos"]):
        return None
    return rects


def _tight_edges(ctx, cur) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Same-axis TOUCHING separation pairs in ``cur`` (candidates for a local
    reorder: their DAG edge is realized at equality)."""
    n = ctx["n"]
    ws, hs = _decode_shapes_ctx(ctx, cur)
    x_edges, y_edges = build_order_dags(cur, ws, hs, n, geo_faithful=True)
    tx = [(u, v) for (u, v, g) in x_edges
          if abs(cur[v][0] - (cur[u][0] + g)) <= TIGHT_TOL]
    ty = [(u, v) for (u, v, g) in y_edges
          if abs(cur[v][1] - (cur[u][1] + g)) <= TIGHT_TOL]
    return tx, ty


# =============================================================================
# Move classes: (ctx, cur, rng) -> (perturbed_hints, flips) or None
# =============================================================================
def move_order(ctx, cur, rng):
    tx, ty = _tight_edges(ctx, cur)
    pool = [(0, u, v) for (u, v) in tx] + [(1, u, v) for (u, v) in ty]
    if not pool:
        return None
    axis, u, v = rng.choice(pool)
    return list(cur), [(axis, u, v)]


def move_centroid(ctx, cur, rng):
    n = ctx["n"]
    free = [i for i in range(n) if not ctx["preplaced"][i]]
    if len(free) < 2:
        return None
    rng.shuffle(free)
    i = free[0]
    ai = ctx["area_targets"][i]
    j = None
    for cand in free[1:]:
        if abs(ctx["area_targets"][cand] - ai) <= 0.2 * max(ai, 1e-9):
            j = cand
            break
    if j is None:
        return None
    xi, yi, wi, hi = cur[i]
    xj, yj, wj, hj = cur[j]
    cxi, cyi = xi + wi / 2.0, yi + hi / 2.0
    cxj, cyj = xj + wj / 2.0, yj + hj / 2.0
    hints = list(cur)
    hints[i] = (cxj - wi / 2.0, cyj - hi / 2.0, wi, hi)
    hints[j] = (cxi - wj / 2.0, cyi - hj / 2.0, wj, hj)
    return hints, None


def move_shape(ctx, cur, rng):
    n = ctx["n"]
    cands = [i for i in range(n)
             if not ctx["fixed"][i] and not ctx["preplaced"][i]
             and ctx["mib"][i] == 0 and cur[i][2] > 1e-9 and cur[i][3] > 1e-9]
    if not cands:
        return None
    i = rng.choice(cands)
    x, y, w, h = cur[i]
    a = w * h
    asp = w / h
    f = rng.uniform(0.10, 0.20) * rng.choice((-1.0, 1.0))
    asp2 = asp * (1.0 + f)
    nw = math.sqrt(a * asp2)   # nw*nh == a exactly (reservation #3)
    nh = math.sqrt(a / asp2)
    cx, cy = x + w / 2.0, y + h / 2.0
    hints = list(cur)
    hints[i] = (cx - nw / 2.0, cy - nh / 2.0, nw, nh)
    return hints, None


MOVES = {"order": move_order, "centroid": move_centroid, "shape": move_shape}


# =============================================================================
# Context
# =============================================================================
def build_ctx(ds, cache, idx, realize="faithful"):
    sample = ds[idx]
    n = _n_of(sample)
    at = sample["input"][0][:n]
    cons = sample["input"][4][:n]
    b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)
    fixed, preplaced, mib, cluster, boundary = _parse_constraints(cons, n)
    area_targets = [float(at[i]) for i in range(n)]
    prod = [tuple(float(v) for v in r) for r in cache[idx]]
    x_pins: Dict[int, float] = {}
    y_pins: Dict[int, float] = {}
    for i in range(n):
        if preplaced[i]:
            tx, ty, tw, th = _target(tpos, i)
            if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                x_pins[i] = float(tx)
                y_pins[i] = float(ty)
    ctx = dict(
        sample=sample, n=n, at=at, cons=cons, tpos=tpos, fixed=fixed,
        preplaced=preplaced, mib=mib, boundary=boundary,
        area_targets=area_targets, base=prod, x_pins=x_pins, y_pins=y_pins,
        band=_band(n), realize=realize,
        b2b_e=b2b_e, p2b_e=p2b_e, pin_l=pin_l,
    )
    ctx["base_cost"] = score_case(sample, prod, n)["cost"]
    # Reference for dCost: the realized cost of the UNPERTURBED hint under this
    # realizer (the SA operating point). faithful -> identity -> == base_cost;
    # anchored -> the re-solved base, so perturbation dCost is not swamped by the
    # constant re-solve offset.
    r0 = reflow(ctx, prod)
    ctx["ref_cost"] = (score_case(sample, r0, n)["cost"] if r0 is not None
                       else ctx["base_cost"])
    return ctx


# =============================================================================
# Stats helpers
# =============================================================================
def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _summ(deltas):
    """Summary over the LEGAL dCost list."""
    if not deltas:
        return dict(n=0, median=float("nan"), median_abs=float("nan"),
                    p90_abs=float("nan"), max_abs=float("nan"),
                    improve_rate=float("nan"), best=float("nan"))
    ab = [abs(d) for d in deltas]
    imp = sum(1 for d in deltas if d < -IMPROVE_TOL)
    return dict(
        n=len(deltas), median=_pct(deltas, 0.5), median_abs=_pct(ab, 0.5),
        p90_abs=_pct(ab, 0.9), max_abs=max(ab), improve_rate=imp / len(deltas),
        best=min(deltas),
    )


# =============================================================================
# Driver
# =============================================================================
def run(args):
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    with open(args.production_cache) as f:
        cache = {int(k): v for k, v in json.load(f).items()}
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))

    all_idx = [i for i in range(len(ds)) if i in cache]
    if args.cases and args.cases < len(all_idx):
        step = max(1, len(all_idx) // args.cases)
        all_idx = all_idx[::step][: args.cases]

    t0 = time.time()

    # Startup sanity: reflow(base) reproduces base cost (faithful identity).
    max_id_gap = 0.0
    base_reflow_gaps = []
    for idx in all_idx[: min(5, len(all_idx))]:
        ctx = build_ctx(ds, cache, idx, realize=args.realize)
        r = reflow(ctx, ctx["base"])
        assert r is not None, f"idx {idx}: base reflow illegal"
        c = score_case(ctx["sample"], r, ctx["n"])["cost"]
        gap = c - ctx["base_cost"]
        max_id_gap = max(max_id_gap, abs(gap))
        base_reflow_gaps.append(gap)
    # faithful is an identity realizer (gap ~0); anchored re-solves, so
    # reflow(base) != base -- report the mean signed gap as the reference offset
    # (dCost is always measured against base_cost, so a nonzero reflow(base) just
    # shifts every dCost; it does not corrupt the smoothness/improving signal).
    if args.realize == "faithful":
        assert max_id_gap < 1e-3, f"faithful identity broken: {max_id_gap:.2e}"
    print(f"[reflow(base) vs base] realize={args.realize}  max|gap|={max_id_gap:.2e}"
          f"  mean signed gap={sum(base_reflow_gaps)/len(base_reflow_gaps):+.4f}",
          flush=True)

    # (type, band) -> list of legal dCost ; (type, band) -> [legal, illegal]
    single: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    legal_cnt: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    # (band, k) -> list of legal cumulative dCost
    stack: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    # improving moves worth logging for P1 design.
    prey: List[dict] = []

    ks = [1, 2, 4, 8]
    movefns = list(MOVES.values())

    for idx in all_idx:
        ctx = build_ctx(ds, cache, idx, realize=args.realize)
        band = ctx["band"]

        # --- single moves ---
        for tname, mfn in MOVES.items():
            rng = random.Random(args.seed * 1_000_003 + idx * 97 + hash(tname) % 97)
            got = 0
            tries = 0
            cap = args.samples_per * 8
            while got < args.samples_per and tries < cap:
                tries += 1
                res = mfn(ctx, ctx["base"], rng)
                if res is None:
                    continue
                hints, flips = res
                rects = reflow(ctx, hints, flips)
                if rects is None:
                    legal_cnt[(tname, band)][1] += 1
                    got += 1
                    continue
                d = score_case(ctx["sample"], rects, ctx["n"])["cost"] - ctx["ref_cost"]
                single[(tname, band)].append(d)
                legal_cnt[(tname, band)][0] += 1
                got += 1
                if d < -IMPROVE_TOL and len(prey) < 60:
                    prey.append(dict(idx=idx, n=ctx["n"], band=band,
                                     move=tname, dcost=round(d, 5)))

        # --- k-stacked random-walk perturbations ---
        for s in range(args.stack_samples):
            rng = random.Random(args.seed * 7_919 + idx * 131 + s)
            cur = list(ctx["base"])
            applied = 0
            for k in ks:
                target = k
                guard = 0
                while applied < target and guard < target * 6:
                    guard += 1
                    mfn = rng.choice(movefns)
                    res = mfn(ctx, cur, rng)
                    if res is None:
                        continue
                    hints, flips = res
                    rects = reflow(ctx, hints, flips)
                    if rects is None:
                        continue
                    cur = [tuple(r) for r in rects]
                    applied += 1
                if applied >= k:
                    c = score_case(ctx["sample"], cur, ctx["n"])["cost"]
                    stack[(band, k)].append(c - ctx["ref_cost"])

        if args.verbose:
            print(f"  idx={idx:3d} n={ctx['n']:3d} band={band} "
                  f"done ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    _report(single, legal_cnt, stack, prey, ks, elapsed, args)


def _report(single, legal_cnt, stack, prey, ks, elapsed, args):
    bands = ["n<60", "60-99", ">=100"]
    print("\n" + "=" * 84)
    print(f"PERTURB SMOOTHNESS PROBE   seed={args.seed}   "
          f"samples/type/case={args.samples_per}   stack_samples={args.stack_samples}   "
          f"wall={elapsed:.1f}s")
    print("=" * 84)

    # Per move-type x band table.
    print("\n  MOVE x BAND   (dCost over LEGAL moves; +worse / -better)")
    hdr = (f"  {'move':8s} {'band':6s} {'legal':>5s} {'legal%':>6s} "
           f"{'med':>8s} {'med|d|':>7s} {'p90|d|':>7s} {'max|d|':>7s} "
           f"{'impr%':>6s} {'best':>8s}")
    print(hdr)
    print("  " + "-" * 80)
    type_band_summ = {}
    for tname in MOVES:
        for band in bands:
            deltas = single.get((tname, band), [])
            lc = legal_cnt.get((tname, band), [0, 0])
            tot = lc[0] + lc[1]
            lr = lc[0] / tot if tot else float("nan")
            s = _summ(deltas)
            type_band_summ[(tname, band)] = (s, lr, tot)
            print(f"  {tname:8s} {band:6s} {lc[0]:5d} {lr*100:5.0f}% "
                  f"{s['median']:+8.4f} {s['median_abs']:7.4f} {s['p90_abs']:7.4f} "
                  f"{s['max_abs']:7.3f} {s['improve_rate']*100:5.1f}% {s['best']:+8.4f}")
        print()

    # Pooled single-move stats per band (all move types).
    print("  POOLED single-move per band (all move types):")
    tail_median_abs = float("nan")
    tail_improve = float("nan")
    pooled = {}
    for band in bands:
        alld = []
        for tname in MOVES:
            alld += single.get((tname, band), [])
        s = _summ(alld)
        pooled[band] = s
        print(f"    {band:6s} legal={s['n']:5d} med={s['median']:+.4f} "
              f"med|d|={s['median_abs']:.4f} p90|d|={s['p90_abs']:.4f} "
              f"impr%={s['improve_rate']*100:.1f} best={s['best']:+.4f}")
    if pooled.get(">=100"):
        tail_median_abs = pooled[">=100"]["median_abs"]
        tail_improve = pooled[">=100"]["improve_rate"]

    # k-stacking curve per band.
    print("\n  k-STACKED random-walk cumulative dCost (median|d| / p90|d| / max|d|):")
    print(f"    {'band':6s} " + "  ".join(f"k={k:<2d}" for k in ks))
    for band in bands:
        cells = []
        for k in ks:
            xs = stack.get((band, k), [])
            ab = [abs(x) for x in xs]
            if ab:
                cells.append(f"{_pct(ab,0.5):.3f}/{_pct(ab,0.9):.3f}/{max(ab):.2f}")
            else:
                cells.append("  -  ")
        print(f"    {band:6s} " + "  ".join(cells))
    # monotonicity check on tail medians.
    tail_meds = []
    for k in ks:
        ab = [abs(x) for x in stack.get((">=100", k), [])]
        tail_meds.append(_pct(ab, 0.5) if ab else float("nan"))
    mono = all(
        (math.isnan(tail_meds[i]) or math.isnan(tail_meds[i + 1])
         or tail_meds[i + 1] >= tail_meds[i] - 1e-6)
        for i in range(len(tail_meds) - 1)
    )

    # Per-move-type tail improving rate (which move class is the prey source).
    print("\n  TAIL (n>=100) improving-rate by move type:")
    tail_type_impr = {}
    for tname in MOVES:
        s, lr, tot = type_band_summ.get((tname, ">=100"), (_summ([]), float("nan"), 0))
        tail_type_impr[tname] = s["improve_rate"]
        print(f"    {tname:8s} impr%={s['improve_rate']*100:5.1f}  "
              f"med|d|={s['median_abs']:.4f}  best={s['best']:+.4f}  legal={s['n']}")

    # GO / NO-GO.
    tail_has_prey = (not math.isnan(tail_improve)) and tail_improve > 0.0
    tail_smooth = (not math.isnan(tail_median_abs)) and tail_median_abs <= 0.05
    go = tail_has_prey and tail_smooth
    print("\n" + "=" * 84)
    print(f"  DECISION: {'GO' if go else 'NO-GO'}   | tail improving-rate="
          f"{tail_improve*100:.2f}%  tail median|dCost|={tail_median_abs:.4f} "
          f"(<=0.05? {tail_smooth})  k-mono(tail)={mono}")
    print("=" * 84)

    if args.out:
        out = dict(
            seed=args.seed, samples_per=args.samples_per,
            stack_samples=args.stack_samples, wall_s=elapsed,
            decision="GO" if go else "NO-GO",
            tail_improve_rate=tail_improve, tail_median_abs=tail_median_abs,
            tail_k_mono=mono,
            move_band={f"{t}|{b}": type_band_summ[(t, b)][0]
                       for t in MOVES for b in bands
                       if (t, b) in type_band_summ},
            pooled={b: pooled[b] for b in bands if b in pooled},
            stack={f"{b}|k{k}": {
                "median_abs": _pct([abs(x) for x in stack.get((b, k), [])], 0.5),
                "p90_abs": _pct([abs(x) for x in stack.get((b, k), [])], 0.9),
                "n": len(stack.get((b, k), [])),
            } for b in bands for k in ks},
            prey=prey,
        )
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-cache", required=True,
                    help="JSON cache of production layouts keyed by case idx "
                         "(same gate0 cache the D1 probe uses)")
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--cases", type=int, default=0,
                    help="0 = all cached; else stratified subsample of size N")
    ap.add_argument("--samples-per", dest="samples_per", type=int, default=20,
                    help="single-move samples per move type per case (>=20)")
    ap.add_argument("--stack-samples", dest="stack_samples", type=int, default=6,
                    help="random-walk trajectories per case for the k-curve")
    ap.add_argument("--realize", choices=["faithful", "anchored"], default="faithful",
                    help="realizer under test (A3 uses anchored)")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
