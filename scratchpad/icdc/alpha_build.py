"""alpha-curve probe, step 2: build the alpha-interpolated injection files.

pred (alpha=0, the real model rank-0 sample) --> repaired golden (alpha=1,
gr_layouts3.json), per case per block.

  * x, y            : LINEAR interpolation.
  * w, h            : GEOMETRIC (log-space) interpolation, so w*h moves
                      log-linearly between the two endpoint areas.  Linear
                      shape interpolation is area-CONVEX (w0h0 = w1h1 = A
                      forces the interpolant above A, peaking at alpha=0.5),
                      which would plant a spurious area-violation hump exactly
                      in the middle of the curve we are trying to read.
  * preplaced block : golden (x, y, w, h) -- position IS a conditioning input.
  * fixed-shape     : golden (w, h), x/y still interpolated -- only the shape
                      is conditioned.

Overlaps created by the interpolation are injected AS IS: the direct channel
already eats raw overlapping predictions and refine_prediction legalizes them.

usage: alpha_build.py 0.0 0.25 0.5 0.75
"""
from __future__ import annotations

import json
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
    tot = float((P[:, 2] * P[:, 3]).sum())
    return float(np.triu(ov, 1).sum() / max(tot, EPS))


def interp(P0, P1, a, preplaced, fixed):
    out = np.empty_like(P1)
    out[:, 0] = (1 - a) * P0[:, 0] + a * P1[:, 0]
    out[:, 1] = (1 - a) * P0[:, 1] + a * P1[:, 1]
    w0 = np.maximum(P0[:, 2], EPS)
    h0 = np.maximum(P0[:, 3], EPS)
    w1 = np.maximum(P1[:, 2], EPS)
    h1 = np.maximum(P1[:, 3], EPS)
    out[:, 2] = np.exp((1 - a) * np.log(w0) + a * np.log(w1))
    out[:, 3] = np.exp((1 - a) * np.log(h0) + a * np.log(h1))
    fx = fixed != 0
    out[fx, 2] = P1[fx, 2]
    out[fx, 3] = P1[fx, 3]
    pp = preplaced != 0
    out[pp] = P1[pp]
    return out


def main():
    alphas = [float(a) for a in sys.argv[1:]] or [0.0, 0.25, 0.5, 0.75]
    ev = G.load_evaluator()
    cases = {c.test_id: c for c in G.load_cases(ev, "../")}
    pred = {int(k): np.asarray(v, float)
            for k, v in json.load(open(SS / "alpha_pred0.json")).items()}
    gold = {int(k): np.asarray(v, float)
            for k, v in json.load(open(SS / "gr_layouts3.json")).items()}

    files = {a: {} for a in alphas}
    print(f"{'a':>5} {'tid':>4} {'n':>4} {'d/diag':>7} {'ovl':>7} "
          f"{'areaviol':>9} {'cost':>8}")
    stats = {a: [] for a in alphas}
    lin_hump = []
    for tid in sorted(pred):
        c = cases[tid]
        P0, P1 = pred[tid], gold[tid][: c.n]
        assert P0.shape == P1.shape, (tid, P0.shape, P1.shape)
        at = c.area_target[: c.n].numpy().astype(float)
        at = at[:, 0] if at.ndim > 1 else at
        x0, y0, x1, y1 = c.frame
        diag = float(np.hypot(x1 - x0, y1 - y0))
        # how far does the raw model sample already respect the conditioning?
        pp = c.preplaced != 0
        fx = c.fixed != 0
        for a in alphas:
            Pa = interp(P0, P1, a, c.preplaced, c.fixed)
            files[a][str(tid)] = Pa.tolist()
            cen_a = Pa[:, :2] + 0.5 * Pa[:, 2:]
            cen_g = P1[:, :2] + 0.5 * P1[:, 2:]
            d = float(np.linalg.norm(cen_a - cen_g, axis=1).mean()) / diag
            ov = overlap_frac(Pa)
            av = float(np.maximum(0.0, at - Pa[:, 2] * Pa[:, 3]).sum()
                       / max(at.sum(), EPS))
            m = G.evaluate(ev, c, [tuple(map(float, r)) for r in Pa])
            stats[a].append((tid, c.n, d, ov, av, float(m.cost_no_runtime)))
            if a == 0.5:
                # what the naive LINEAR shape interpolation would have added
                lw = 0.5 * (P0[:, 2] + P1[:, 2])
                lh = 0.5 * (P0[:, 3] + P1[:, 3])
                lw[fx] = P1[fx, 2]; lh[fx] = P1[fx, 3]
                lin_hump.append(float((lw * lh).sum() / (Pa[:, 2] * Pa[:, 3]).sum()))
        if tid in (74, 99):
            print(f"  tid {tid}: preplaced={int(pp.sum())} fixed={int(fx.sum())} "
                  f"pred-vs-golden max|dxy| on preplaced="
                  f"{float(np.abs(P0[pp, :2] - P1[pp, :2]).max()) if pp.any() else 0.0:.4g} "
                  f"max|dwh| on fixed="
                  f"{float(np.abs(P0[fx, 2:] - P1[fx, 2:]).max()) if fx.any() else 0.0:.4g}")

    for a in alphas:
        rows = stats[a]
        band = [r for r in rows if r[1] >= 103]
        mx = max(r[1] for r in band)
        w = np.array([np.exp((r[1] - mx) / 12) for r in band])
        w /= w.sum()
        print(f"alpha={a:.2f}  live-band(n>=103,{len(band)}c): "
              f"d/diag={np.mean([r[2] for r in band]):.4f} "
              f"ovl={np.mean([r[3] for r in band]):.4f} "
              f"areaviol={np.mean([r[4] for r in band]):.4f} "
              f"wcost_asis={float((w * np.array([r[5] for r in band])).sum()):.4f}")
        name = f"alpha_a{int(round(a * 100)):02d}.json"
        (SS / name).write_text(json.dumps(files[a]))
        print(f"   -> {name}")
    if lin_hump:
        print(f"\nlinear-shape area bulge at alpha=0.5 vs geometric: "
              f"x{np.mean(lin_hump):.4f} (mean over cases)")


if __name__ == "__main__":
    main()
