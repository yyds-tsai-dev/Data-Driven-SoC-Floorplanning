"""P-A1/A3 follow-up: SA pilot over the anchored realizer (cash-in test).

A3 showed the anchored realizer has an exploitable search landscape (17% tail
improving moves, magnitude up to -1.0, shape moves smooth) -- unlike faithful's
ratchet. This pilot cashes that in: a short greedy+low-T SA per case, genotype =
the (realized) layout, phenotype/objective = decode(realize="anchored") scored by
the exact evaluator (score_case), shape-dominant move set, seeded from the
anchored realization of the production hint.

Case subset: the A1 anchored-vs-production WIN cases (auto-detected) plus the
heaviest tail cases (largest n) filled to --n-cases (default 10).

Decision (written into the report):
  GO      : >= 3/10 converge below production base_cost (per-case portfolio wins
            are real score), OR the A1 wins' lead widens materially.
  WEAK GO : 1-2 wins, or several within +/-0.02 of production (landscape has meat
            but 200 steps / this move set is not enough).
  NO-GO   : all plateau near seed, or > 0.1 from production (area-ratchet ceiling
            confirmed -> skeleton-level compaction moves must go first).

Reuses perturb_probe (build_ctx / reflow / move_*) and gen_decoder_probe
(score_case). No src/ or FloorSet/ edits; gen_decoder_probe unchanged.

Usage (from repo root, tcsh -> bash):
  cd FloorSet/iccad2026contest
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
      <repo>/scripts/probes/anchored_sa_pilot.py \
      --production-cache <gate0 cache.json> --out sa_pilot.json
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
from typing import Dict, List

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

from gen_decoder_probe import score_case  # noqa: E402
from perturb_probe import (  # noqa: E402
    build_ctx,
    reflow,
    move_shape,
    move_centroid,
    move_order,
)

MOVE_FNS = {"shape": move_shape, "order": move_order, "centroid": move_centroid}


def _pick_move(rng, shape_frac, order_frac):
    r = rng.random()
    if r < shape_frac:
        return "shape"
    if r < shape_frac + order_frac:
        return "order"
    return "centroid"


def sa_case(ctx, args, rng) -> Dict[str, object]:
    """Greedy + low-T SA. State = realized anchored layout; objective = exact
    score_case cost. Returns per-case trajectory summary + move acceptance."""
    base_cost = ctx["base_cost"]          # production (cache) cost
    seed_rects = reflow(ctx, ctx["base"])  # anchored realization of production
    if seed_rects is None:
        return dict(idx=None, seed_cost=float("nan"))
    seed_cost = score_case(ctx["sample"], seed_rects, ctx["n"])["cost"]

    # Seed protection: if the anchored realization damages the production seed
    # (> base + eps), start the SA from the production layout ITSELF (a legal
    # layout; moves still realize through anchored). With the legal-hint skeleton
    # this rarely triggers -- anchored(production) == production -- but it caps
    # the seed at production so the search never starts in a hole.
    if seed_cost > base_cost + args.seed_eps:
        seed_rects = [tuple(r) for r in ctx["base"]]
        seed_cost = base_cost

    cur = [tuple(r) for r in seed_rects]
    cur_cost = seed_cost
    best_cost = seed_cost
    best_rects = seed_rects

    T = args.t0
    cooling = (args.t_final / args.t0) ** (1.0 / max(1, args.moves))
    stats = {m: [0, 0, 0] for m in MOVE_FNS}  # propose, accept, improve
    accepted_steps = 0
    proposed = 0
    deadline = time.time() + args.sec_per_seed if args.sec_per_seed > 0 else None

    for _ in range(args.moves):
        if deadline is not None and time.time() > deadline:
            break
        proposed += 1
        mt = _pick_move(rng, args.shape_frac, args.order_frac)
        res = MOVE_FNS[mt](ctx, cur, rng)
        stats[mt][0] += 1
        if res is None:
            T *= cooling
            continue
        hints, flips = res
        rects = reflow(ctx, hints, flips)
        if rects is None:
            T *= cooling
            continue
        c = score_case(ctx["sample"], rects, ctx["n"])["cost"]
        d = c - cur_cost
        if d < 0.0 or rng.random() < math.exp(-d / max(T, 1e-9)):
            cur = [tuple(r) for r in rects]
            cur_cost = c
            accepted_steps += 1
            stats[mt][1] += 1
            if d < 0.0:
                stats[mt][2] += 1
            if c < best_cost:
                best_cost = c
                best_rects = rects
        T *= cooling

    return dict(
        idx=ctx.get("_idx"), n=ctx["n"], band=ctx["band"],
        base_cost=base_cost, seed_cost=seed_cost, best_cost=best_cost,
        gain=seed_cost - best_cost, vs_base=best_cost - base_cost,
        win=bool(best_cost < base_cost - 1e-9), accepted=accepted_steps,
        proposed=proposed,
        move_stats={m: {"propose": stats[m][0], "accept": stats[m][1],
                        "improve": stats[m][2]} for m in MOVE_FNS},
    )


def select_cases(ds, cache, n_cases):
    """A1 anchored-vs-production winners + heaviest tail cases -> n_cases."""
    from perturb_probe import _n_of
    seed_costs = {}
    base_costs = {}
    ns = {}
    for idx in cache:
        ctx = build_ctx(ds, cache, idx, realize="anchored")
        r = reflow(ctx, ctx["base"])
        ns[idx] = ctx["n"]
        base_costs[idx] = ctx["base_cost"]
        seed_costs[idx] = (score_case(ctx["sample"], r, ctx["n"])["cost"]
                           if r is not None else float("inf"))
    winners = sorted([i for i in cache if seed_costs[i] < base_costs[i] - 1e-9],
                     key=lambda i: seed_costs[i] - base_costs[i])
    chosen = list(winners)
    # fill with heaviest tail (largest n) not already chosen.
    for idx in sorted(cache, key=lambda i: -ns[i]):
        if len(chosen) >= n_cases:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen[:n_cases], winners, seed_costs, base_costs, ns


def run(args):
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    with open(args.production_cache) as f:
        cache = {int(k): [tuple(r) for r in v] for k, v in json.load(f).items()}
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))

    t0 = time.time()
    if args.case_ids:
        chosen = [int(x) for x in args.case_ids.split(",") if x.strip() != ""]
        winners, seed_all, base_all, ns = [], {}, {}, {}
        from perturb_probe import _n_of
        for idx in cache:
            ns[idx] = _n_of(ds[idx])
            base_all[idx] = None
    else:
        chosen, winners, seed_all, base_all, ns = select_cases(ds, cache, args.n_cases)
    print(f"[select] A1 winners={winners}  chosen={chosen}  "
          f"seeds/case={args.n_seeds}  moves/seed={args.moves}", flush=True)

    rows = []
    any_win = False
    for idx in chosen:
        for s in range(args.n_seeds):
            ctx = build_ctx(ds, cache, idx, realize="anchored")
            ctx["_idx"] = idx
            base_all[idx] = ctx["base_cost"]
            rng = random.Random(args.seed * 1_000_003 + idx * 131 + s * 7_919)
            r = sa_case(ctx, args, rng)
            r["seed_i"] = s
            rows.append(r)
            flag = "  *** WIN ***" if r["win"] else ""
            if r["win"]:
                any_win = True
            print(f"  idx={idx:3d} seed{s} n={r['n']:3d} base={r['base_cost']:.4f} "
                  f"best={r['best_cost']:.4f} gain={r['gain']:+.4f} "
                  f"vs_base={r['vs_base']:+.4f} mv={r.get('proposed','?')} "
                  f"acc={r['accepted']}{flag}", flush=True)
    elapsed = time.time() - t0
    if args.n_seeds > 1 or args.case_ids:
        _report_confirm(rows, any_win, elapsed, args)
    else:
        _report(rows, winners, seed_all, base_all, ns, cache, elapsed, args)


def _report_confirm(rows, any_win, elapsed, args):
    """Focused evidence table for the confirmation run: per (case, seed)."""
    print("\n" + "=" * 88)
    print(f"CONFIRMATION RUN   cases x seeds = {len(rows)}   moves/seed={args.moves}   "
          f"wall={elapsed:.1f}s")
    print("=" * 88)
    print(f"  {'idx':>3s} {'seed':>4s} {'n':>3s} {'base':>8s} {'best':>8s} "
          f"{'gain':>8s} {'vs_base':>8s} {'mv':>5s} {'acc':>5s}  result")
    for r in rows:
        res = "*** WIN ***" if r["win"] else ("tie" if abs(r["vs_base"]) < 1e-6 else "")
        print(f"  {r['idx']:3d} {r['seed_i']:4d} {r['n']:3d} {r['base_cost']:8.4f} "
              f"{r['best_cost']:8.4f} {r['gain']:+8.4f} {r['vs_base']:+8.4f} "
              f"{r.get('proposed',0):5d} {r['accepted']:5d}  {res}")
    n_win = sum(1 for r in rows if r["win"])
    best_vs = min(r["vs_base"] for r in rows)
    print("-" * 88)
    print(f"  ANY WIN across all (case,seed) = {any_win}   wins={n_win}/{len(rows)}   "
          f"best vs_base seen = {best_vs:+.4f}")
    print(f"  VERDICT: {'!!! UNEXPECTED WIN -- FLAG !!!' if any_win else 'DISPROVED (no run < base)'}")
    print("=" * 88)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(any_win=any_win, wins=n_win, rows=rows,
                           cfg=dict(moves=args.moves, n_seeds=args.n_seeds,
                                    seed=args.seed)), f, indent=2, default=str)
        print(f"wrote {args.out}")


def _report(rows, winners, seed_all, base_all, ns, cache, elapsed, args):
    wins = [r for r in rows if r["win"]]
    near = [r for r in rows if (not r["win"]) and abs(r["vs_base"]) <= 0.02]
    plateau = [r for r in rows if r["gain"] <= 0.01]
    n_win = len(wins)

    print("\n" + "=" * 88)
    print(f"ANCHORED SA PILOT   cases={len(rows)}   moves/case={args.moves}   "
          f"shape/order/centroid={args.shape_frac}/{args.order_frac}/"
          f"{1-args.shape_frac-args.order_frac:.2f}   wall={elapsed:.1f}s")
    print(f"  T0={args.t0} T_final={args.t_final} seed={args.seed}")
    print("=" * 88)
    print(f"  wins (best<base) = {n_win}/{len(rows)}   near(+/-0.02) = {len(near)}   "
          f"plateau(gain<=0.01) = {len(plateau)}")
    print("-" * 88)
    print(f"  {'idx':>3s} {'n':>3s} {'seed':>8s} {'best':>8s} {'base':>8s} "
          f"{'gain':>8s} {'vs_base':>8s}  result")
    for r in sorted(rows, key=lambda r: r["vs_base"]):
        print(f"  {r['idx']:3d} {r['n']:3d} {r['seed_cost']:8.4f} {r['best_cost']:8.4f} "
              f"{r['base_cost']:8.4f} {r['gain']:+8.4f} {r['vs_base']:+8.4f}  "
              f"{'WIN' if r['win'] else ('near' if abs(r['vs_base'])<=0.02 else '')}")
    # move acceptance pooled.
    print("-" * 88)
    pooled = {m: [0, 0, 0] for m in MOVE_FNS}
    for r in rows:
        for m in MOVE_FNS:
            s = r["move_stats"][m]
            pooled[m][0] += s["propose"]
            pooled[m][1] += s["accept"]
            pooled[m][2] += s["improve"]
    print("  move acceptance (pooled):")
    for m in MOVE_FNS:
        p, a, im = pooled[m]
        print(f"    {m:8s} propose={p:5d} accept={a:5d} ({100*a/max(p,1):4.0f}%) "
              f"improve={im:5d} ({100*im/max(p,1):4.0f}%)")

    # Decision.
    if n_win >= 3:
        decision = "GO"
    elif n_win >= 1 or len(near) >= 2:
        decision = "WEAK-GO"
    else:
        decision = "NO-GO"
    print("=" * 88)
    print(f"  DECISION: {decision}   wins={n_win}/{len(rows)}  near={len(near)}  "
          f"mean gain={sum(r['gain'] for r in rows)/len(rows):+.4f}  "
          f"mean vs_base={sum(r['vs_base'] for r in rows)/len(rows):+.4f}")

    # Full-100 portfolio extrapolation (only if any signal).
    if decision != "NO-GO":
        bcs = [ns[i] for i in cache]
        base_costs = [base_all[i] for i in cache]
        base_total = compute_total_score(base_costs, bcs)
        # Per-case pilot win rate by band -> extrapolate a portfolio gain.
        by_band = defaultdict(lambda: [0, 0.0])  # band -> [count, sum_win_gain]
        for r in rows:
            g = max(0.0, r["base_cost"] - r["best_cost"])  # portfolio gain
            by_band[r["band"]][0] += 1
            by_band[r["band"]][1] += g
        # Apply mean per-case portfolio gain (by band) to all cache cases; the
        # anchored seed already never loses portfolio (min(base, anchored)), so
        # this is a floor-ish estimate: expected best_cost = base - est_gain.
        def band_of(nn):
            return "n<60" if nn < 60 else ("60-99" if nn < 100 else ">=100")
        est_costs = []
        for i in cache:
            b = band_of(ns[i])
            cnt, tot = by_band.get(b, [0, 0.0])
            eg = (tot / cnt) if cnt else 0.0
            est_costs.append(max(0.0, base_all[i] - eg))
        est_total = compute_total_score(est_costs, bcs)
        print("-" * 88)
        print(f"  EXTRAPOLATION (rough): production total={base_total:.4f}  "
              f"est. full-100 portfolio(min) ~= {est_total:.4f}  "
              f"(delta {est_total - base_total:+.4f})")
        print("    per-band mean portfolio gain: " + "  ".join(
            f"{b}={(by_band[b][1]/by_band[b][0] if by_band[b][0] else 0):+.4f}"
            f"(n={by_band[b][0]})" for b in ("n<60", "60-99", ">=100")))
    print("=" * 88)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                decision=decision, wins=n_win, cases=len(rows),
                winners_a1=list(winners), rows=rows,
                cfg=dict(moves=args.moves, t0=args.t0, t_final=args.t_final,
                         shape_frac=args.shape_frac, order_frac=args.order_frac,
                         seed=args.seed),
            ), f, indent=2, default=str)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-cache", required=True)
    ap.add_argument("--data-path", default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--n-cases", dest="n_cases", type=int, default=10)
    ap.add_argument("--case-ids", dest="case_ids", default=None,
                    help="comma-separated case indices (overrides auto-select; "
                         "confirmation run)")
    ap.add_argument("--n-seeds", dest="n_seeds", type=int, default=1,
                    help="SA restarts per case with distinct RNG seeds")
    ap.add_argument("--moves", type=int, default=200)
    ap.add_argument("--sec-per-seed", dest="sec_per_seed", type=float, default=0.0,
                    help="wall-clock cap per SA seed in seconds (0 = no cap); "
                         "moves stop early when exceeded (records actual proposed)")
    ap.add_argument("--t0", type=float, default=0.03)
    ap.add_argument("--t-final", dest="t_final", type=float, default=0.003)
    ap.add_argument("--shape-frac", dest="shape_frac", type=float, default=0.80)
    ap.add_argument("--order-frac", dest="order_frac", type=float, default=0.20,
                    help="centroid fraction = 1 - shape - order; default 0 "
                         "(centroid dropped: 2% accept in the pilot, dead weight)")
    ap.add_argument("--seed-eps", dest="seed_eps", type=float, default=1e-3,
                    help="seed protection: fall back to the production layout as "
                         "SA start when anchored(seed) > base + this")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
