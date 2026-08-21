"""alpha-curve probe: INPUT quality of each injected layout, overlap-free.

The evaluator caps any overlapping layout at cost 10, so the injected
alpha<1 layouts have no readable cost.  This measures the two quantities the
cost is actually built from -- hpwl_gap and area_gap -- directly on the
injected layout, plus the overlap fraction.  It separates "the interpolated
prior is genuinely better" from "the interpolated prior just overlaps more".
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

EPS = 1e-12


def overlap_frac(P):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.maximum(0.0, np.minimum(x1[:, None], x1[None, :])
                    - np.maximum(x0[:, None], x0[None, :]))
    oy = np.maximum(0.0, np.minimum(y1[:, None], y1[None, :])
                    - np.maximum(y0[:, None], y0[None, :]))
    ov = ox * oy
    np.fill_diagonal(ov, 0.0)
    return float(np.triu(ov, 1).sum() / max(float((P[:, 2] * P[:, 3]).sum()), EPS))


def main():
    ev = G.load_evaluator()
    cases = {c.test_id: c for c in G.load_cases(ev, "../")}
    files = [("a00", "alpha_a00.json", 0.00), ("a25", "alpha_a25.json", 0.25),
             ("a50", "alpha_a50.json", 0.50), ("a75", "alpha_a75.json", 0.75),
             ("o", "gr_layouts3.json", 1.00)]
    print(f"{'arm':>5} {'alpha':>6} {'hpwl_gap':>9} {'area_gap':>9} "
          f"{'q_proxy':>8} {'ovl%':>7} {'d/diag':>7}")
    for arm, fn, a in files:
        lay = {int(k): np.asarray(v, float)
               for k, v in json.load(open(SS / fn)).items()}
        rows = []
        for tid, P in lay.items():
            c = cases.get(tid)
            if c is None or c.n < 103:
                continue
            P = P[: c.n]
            rects = [tuple(map(float, r)) for r in P]
            hb = ev.calculate_hpwl_b2b(rects, c.b2b)
            hp = ev.calculate_hpwl_p2b(rects, c.p2b, c.pins)
            area = ev.calculate_bbox_area(rects)
            hgap = (hb + hp) / c.baseline["hpwl_baseline"] - 1.0
            agap = area / c.baseline["area_baseline"] - 1.0
            gold = np.asarray(json.load(open(SS / "gr_layouts3.json"))[str(tid)],
                              float)[: c.n]
            cen = P[:, :2] + 0.5 * P[:, 2:]
            cg = gold[:, :2] + 0.5 * gold[:, 2:]
            x0, y0, x1, y1 = c.frame
            d = float(np.linalg.norm(cen - cg, axis=1).mean()) / math.hypot(
                x1 - x0, y1 - y0)
            rows.append((c.n, hgap, agap, overlap_frac(P), d))
        mx = max(r[0] for r in rows)
        w = np.array([math.exp((r[0] - mx) / 12) for r in rows])
        w /= w.sum()

        def agg(i):
            return float((w * np.array([r[i] for r in rows])).sum())

        hg, ag = agg(1), agg(2)
        # the evaluator's quality factor: 1 + max(0,hpwl_gap)/2 + max(0,area_gap)/2
        q = 1.0 + max(0.0, hg) / 2 + max(0.0, ag) / 2
        print(f"{arm:>5} {a:>6.2f} {hg:>9.4f} {ag:>9.4f} {q:>8.4f} "
              f"{agg(3) * 100:>7.3f} {agg(4):>7.4f}")


if __name__ == "__main__":
    main()
