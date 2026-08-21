"""alpha-curve probe, CONTROL LADDER: overlap-free at every alpha.

The coordinate-interpolation ladder (alpha_build.py) is non-monotone in the one
input property that actually drives the downstream score -- overlap.  Raw
prediction overlaps 0.44% of block area, golden 0%, but the midpoints overlap
MORE than either endpoint (peak 0.97% at alpha=0.5), because averaging two
different discrete packings is not a packing.  Any flat-then-takeoff shape in
the primary curve is therefore unreadable.

This builds a ladder that is provably overlap-free at every alpha:

  1. Take the golden layout's pair->axis separation certificate E (every pair
     is separated on exactly one axis, so E certifies overlap-freeness).
  2. Project the raw prediction onto the polyhedron {c : c_i + s_i <= c_j,
     (i,j) in E} using the PREDICTION's own shapes -> P0L.  Both endpoints now
     satisfy the SAME linear edge set.
  3. Interpolate positions LINEARLY and shapes GEOMETRICALLY.  The edge
     constraint is affine in (c, s), so the linear-shape combination is
     feasible; the geometric shape is <= the linear one, so it is feasible too
     AND preserves area exactly.

alpha = 0 is therefore "the model's own prediction, legalised"; the ladder
isolates coordinate fidelity from legality.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SS = Path(__file__).resolve().parent
sys.path.insert(0, str(SS))
import gr_lib as G  # noqa: E402
from alpha_build import interp, overlap_frac  # noqa: E402


def project(c0, s0, edges, order):
    """Feasible point of {c : c_i + s_i <= c_j} closest-ish to c0.

    Forward pass in topological order pushes each block just past its
    predecessors; the backward pass then pulls every block back toward its
    target as far as its successors allow.  Both passes preserve feasibility.
    """
    n = len(c0)
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for i, j in edges:
        succ[i].append(j)
        pred[j].append(i)
    c = np.array(c0, dtype=np.float64)
    for i in order:                                   # forward: satisfy preds
        lo = max((c[j] + s0[j] for j in pred[i]), default=-np.inf)
        c[i] = max(c0[i], lo) if lo > -np.inf else c0[i]
    for i in reversed(order):                         # backward: pull back
        hi = min((c[k] - s0[i] for k in succ[i]), default=np.inf)
        lo = max((c[j] + s0[j] for j in pred[i]), default=-np.inf)
        want = c0[i]
        v = min(c[i], hi) if hi < np.inf else c[i]
        v = max(v, want) if want <= v else v
        c[i] = max(v, lo) if lo > -np.inf else v
    return c


def main():
    alphas = [float(a) for a in sys.argv[1:]] or [0.0, 0.25, 0.5, 0.75]
    ev = G.load_evaluator()
    cases = {c.test_id: c for c in G.load_cases(ev, "../")}
    pred = {int(k): np.asarray(v, float)
            for k, v in json.load(open(SS / "alpha_pred0.json")).items()}
    gold = {int(k): np.asarray(v, float)
            for k, v in json.load(open(SS / "gr_layouts3.json")).items()}

    files = {a: {} for a in alphas}
    stats = {a: [] for a in alphas}
    proj_report = []
    for tid in sorted(pred):
        c = cases[tid]
        P0, P1 = pred[tid].copy(), gold[tid][: c.n]
        hor, ver, _ov = G.build_topology(c.rects)
        # golden coordinate order is a valid topological order for each DAG
        ox = sorted(range(c.n), key=lambda i: (c.rects[i][0], i))
        oy = sorted(range(c.n), key=lambda i: (c.rects[i][1], i))
        P0L = P0.copy()
        P0L[:, 0] = project(P0[:, 0], P0[:, 2], hor, ox)
        P0L[:, 1] = project(P0[:, 1], P0[:, 3], ver, oy)
        # preplaced blocks must stay exactly on their conditioned position
        pp = c.preplaced != 0
        P0L[pp] = P1[pp]
        shift = float(np.abs(P0L[:, :2] - P0[:, :2]).mean())
        proj_report.append((tid, c.n, shift, overlap_frac(P0L)))
        for a in alphas:
            Pa = interp(P0L, P1, a, c.preplaced, c.fixed)
            files[a][str(tid)] = Pa.tolist()
            at = c.area_target[: c.n].numpy().astype(float)
            at = at[:, 0] if at.ndim > 1 else at
            rects = [tuple(map(float, r)) for r in Pa]
            hb = ev.calculate_hpwl_b2b(rects, c.b2b)
            hp = ev.calculate_hpwl_p2b(rects, c.p2b, c.pins)
            area = ev.calculate_bbox_area(rects)
            stats[a].append((
                c.n,
                (hb + hp) / c.baseline["hpwl_baseline"] - 1.0,
                area / c.baseline["area_baseline"] - 1.0,
                overlap_frac(Pa),
                float(G.evaluate(ev, c, rects).cost_no_runtime)))

    band = [r for r in proj_report if r[1] >= 103]
    print(f"projection onto the golden certificate, live band ({len(band)} cases): "
          f"mean |shift| = {np.mean([r[2] for r in band]):.4f} units, "
          f"resulting overlap = {np.mean([r[3] for r in band]) * 100:.4f}%")
    print(f"\n{'alpha':>6} {'hpwl_gap':>9} {'area_gap':>9} {'q_proxy':>8} "
          f"{'ovl%':>7} {'cost_asis':>10}")
    for a in alphas:
        rows = [r for r in stats[a] if r[0] >= 103]
        mx = max(r[0] for r in rows)
        w = np.array([math.exp((r[0] - mx) / 12) for r in rows])
        w /= w.sum()

        def agg(i):
            return float((w * np.array([r[i] for r in rows])).sum())

        hg, ag = agg(1), agg(2)
        q = 1.0 + max(0.0, hg) / 2 + max(0.0, ag) / 2
        print(f"{a:>6.2f} {hg:>9.4f} {ag:>9.4f} {q:>8.4f} {agg(3) * 100:>7.4f} "
              f"{agg(4):>10.4f}")
        name = f"alpha_L{int(round(a * 100)):02d}.json"
        (SS / name).write_text(json.dumps(files[a]))


if __name__ == "__main__":
    main()
