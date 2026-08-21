"""alpha-curve probe, step 5: per-arm channel attrition from PARTNER_PSEL_DUMP.

For each arm: how many live-band cases did the direct/phaseB channel win, and
how much of the injected fidelity survived refine vs the selector.

usage: alpha_psel.py <arm>=<npz>[,<injected json>] ...
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


def band_rows(npz):
    d = np.load(npz)
    off = d["cand_pos_off"]
    pos = d["pos_all"]
    ends = np.append(off[1:], pos.shape[0])
    cands = {}
    for j in range(len(d["cand_case"])):
        cands.setdefault(int(d["cand_case"][j]), []).append({
            "chan": int(d["cand_chan"][j]),
            "sel": bool(d["cand_sel"][j]),
            "pos": pos[off[j]:ends[j]],
        })
    return cands


def main():
    ev = G.load_evaluator()
    cases = {c.test_id: c for c in G.load_cases(ev, "../")}
    print(f"{'arm':>5} {'cases':>6} {'dirwin':>7} {'dir>col':>8} "
          f"{'inject':>8} {'dir*':>8} {'col*':>8} {'pool*':>8} {'sel':>8} "
          f"{'refdmg':>8} {'seldmg':>8}")
    for spec in sys.argv[1:]:
        arm, rest = spec.split("=", 1)
        parts = rest.split(",")
        npz = parts[0]
        inj = None
        if len(parts) > 1 and parts[1]:
            inj = {int(k): np.asarray(v, float)
                   for k, v in json.load(open(parts[1])).items()}
        cands = band_rows(npz)
        rows = []
        for tid in sorted(cands):
            c = cases.get(tid)
            if c is None or c.n < 103:
                continue
            costs = {0: [], 1: [], 2: []}
            sel = None
            for cd in cands[tid]:
                P = cd["pos"]
                if P.shape[0] != c.n:
                    continue
                m = G.evaluate(ev, c, [tuple(map(float, r)) for r in P])
                cst = float(m.cost_no_runtime)
                costs[cd["chan"]].append(cst)
                if cd["sel"]:
                    sel = (cst, cd["chan"])
            if not any(costs.values()):
                continue
            bd = min(costs[1] + costs[2]) if (costs[1] or costs[2]) else float("nan")
            bc = min(costs[0]) if costs[0] else float("nan")
            bp = min([v for v in (bd, bc) if v == v], default=float("nan"))
            ic = float("nan")
            if inj is not None and tid in inj:
                P = inj[tid][: c.n]
                ic = float(G.evaluate(
                    ev, c, [tuple(map(float, r)) for r in P]).cost_no_runtime)
            rows.append((c.n, ic, bd, bc, bp, sel[0] if sel else float("nan"),
                         sel[1] if sel else -1))
        if not rows:
            print(f"{arm:>5}  (no live-band rows)")
            continue
        mx = max(r[0] for r in rows)
        w = np.array([math.exp((r[0] - mx) / 12) for r in rows])

        def agg(i):
            v = np.array([r[i] for r in rows], float)
            ok = np.isfinite(v)
            return float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")

        dw = sum(1 for r in rows if r[6] in (1, 2))
        beat = sum(1 for r in rows if r[2] < r[3] - 1e-9)
        print(f"{arm:>5} {len(rows):>6} {dw:>7} {beat:>8} {agg(1):>8.4f} "
              f"{agg(2):>8.4f} {agg(3):>8.4f} {agg(4):>8.4f} {agg(5):>8.4f} "
              f"{agg(2) - agg(1):>+8.4f} {agg(5) - agg(4):>+8.4f}")


if __name__ == "__main__":
    main()
