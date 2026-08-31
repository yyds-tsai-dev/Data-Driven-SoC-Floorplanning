#!/usr/bin/env python
"""Where do the violations enter?  Sample the shipped Flow direct channel
exactly as contest_optimizer._sample_flow_preds does (CPU), score every RAW
sample with the evaluator's own predicates, then push each sample through
refine_prediction + the shipped post-pass chain and score again.

Run from FloorSet/iccad2026contest.  One tid per process:
  uv run python .../vprobe.py --tid 99 --k 16 --budget 5 --jsonl out/99.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "partner"), str(_REPO / "FloorSet" / "iccad2026contest"),
           str(_REPO / "FloorSet"), str(_REPO / "scripts" / "probes"),
           str(_REPO / "scratchpad" / "rtaware")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from litetestLoader import FloorplanDatasetLiteTest
import iccad2026_evaluate as EV
from column_sa_legalizer import _ColumnOptimizer
from layout_refiner import refine_prediction
from violation_killer import _violations_exact
from golden_probe import build, ev, _Shim


def overlap_frac(P: np.ndarray, at) -> float:
    """Pairwise overlap area / total target area."""
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = (np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])).clip(0)
    oy = (np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])).clip(0)
    a = ox * oy
    np.fill_diagonal(a, 0.0)
    tot = float(a.sum()) * 0.5
    return tot / max(float(np.asarray(at).sum()), 1e-9)


def score_layout(P, base, cons, b2b, p2b, pins, at, tpos):
    m = ev(P, base, cons, b2b, p2b, pins, at, tpos)
    return dict(cost=float(m.cost), feas=bool(m.is_feasible),
                hpwl=float(m.hpwl_gap), area=float(m.area_gap),
                V=float(m.violations_relative), bnd=int(m.boundary_violations),
                grp=int(m.grouping_violations), mib=int(m.mib_violations),
                M=int(m.max_possible_violations))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--raw-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("VP_THREADS", "2")))

    ds = FloorplanDatasetLiteTest(a.data_path)
    n, at, cons, tpos, b2b, p2b, pins, golden, base = build(ds, a.tid)
    from contest_optimizer import MyOptimizer
    opt_m = MyOptimizer(device="cpu")
    assert opt_m.flow_model is not None, "flow model not loaded (FLOW_CKPT / PARTNER_FLOW_SLOTS)"

    t0 = time.time()
    preds = opt_m._sample_flow_preds(n, at, cons, tpos, b2b, p2b, pins,
                                     a.k, gen_seed=0)
    t_sample = time.time() - t0

    shim = _Shim()
    row = {"tid": a.tid, "n": n, "k": len(preds), "t_sample": round(t_sample, 2),
           "budget": a.budget}
    row["golden"] = score_layout(golden, base, cons, b2b, p2b, pins, at, tpos)
    raws, refs, posts = [], [], []
    for P in preds:
        P = np.asarray(P, dtype=np.float64)
        r = score_layout(P, base, cons, b2b, p2b, pins, at, tpos)
        r["ovl"] = round(overlap_frac(P, at), 5)
        o0 = _ColumnOptimizer([tuple(map(float, q)) for q in P], at, cons, tpos,
                              b2b, p2b, pins, time.time() + 1.0, seed=0)
        try:
            r["Vx"] = int(_violations_exact(o0, P))
        except Exception as e:
            r["Vx"] = -1; r["Vx_err"] = str(e)[:40]
        raws.append(r)
        if a.raw_only:
            continue
        o = _ColumnOptimizer([tuple(map(float, q)) for q in P], at, cons, tpos,
                             b2b, p2b, pins, time.time() + a.budget, seed=0)
        try:
            ref = refine_prediction(o, P.copy(), time.time() + a.budget, seed=41)
        except Exception as e:
            ref = None
        if ref is None:
            refs.append(None); posts.append(None); continue
        refs.append(score_layout(ref, base, cons, b2b, p2b, pins, at, tpos))
        box = [(np.asarray([list(map(float, q)) for q in ref]), o)]
        cur = [tuple(map(float, q)) for q in ref]
        try:
            for fn, kw in ((MyOptimizer._coord_polish, {"elapsed": 0.0}),
                           (MyOptimizer._final_seat, None),
                           (MyOptimizer._tag_compress, None),
                           (MyOptimizer._wall_repair_final, None)):
                cur = (fn(shim, cur, at, cons, tpos, b2b, p2b, pins, box) if kw is None
                       else fn(shim, cur, at, cons, tpos, b2b, p2b, pins, **kw))
            posts.append(score_layout(cur, base, cons, b2b, p2b, pins, at, tpos))
        except Exception as e:
            posts.append(None); row.setdefault("post_err", str(e)[:80])
    row["raw"] = raws
    row["ref"] = refs
    row["post"] = posts
    line = json.dumps(row)
    print(line, flush=True)
    if a.jsonl:
        Path(a.jsonl).write_text(line + "\n")


if __name__ == "__main__":
    main()
