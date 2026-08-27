#!/usr/bin/env python
"""Classify the residual BOUNDARY violations of a shipped 100-case run.

Reads the evaluator's own `--output` JSON (which carries `positions` per
case), rebuilds the instance-side view of each case from the FloorSet
validation loader, and asks, for every unsatisfied boundary tag bit, WHY the
block is not sitting on its wall:

  free      the block can translate straight onto the wall with no clash
            (so the seat pass either never tried it or its strict V gate
            rejected the intermediate)
  swap1     exactly ONE block sits between it and the wall, and that block
            is itself flush on the wall  -> wall-swap territory
  blockN    N>=2 blocks stand between it and the wall (a packed band)
  interior  the blockers are not on the wall at all (the tag is in a
            different band; a local move cannot reach)
  locked    the tagged block is preplaced (kind 2): it cannot move, so the
            WALL has to come to it (compaction, not seating)

Usage:
  uv run python scripts/probes/wall_seat_diag.py RESULT.json [--data-path ../]
      [--ids 88,89,90,92,99] [--min-n 102] [--jsonl OUT.jsonl]
Run it from FloorSet/iccad2026contest (the loader resolves its own paths).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "partner"),
           str(_REPO / "FloorSet" / "iccad2026contest"),
           str(_REPO / "FloorSet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from litetestLoader import FloorplanDatasetLiteTest      # noqa: E402
from column_sa_legalizer import _ColumnOptimizer          # noqa: E402
from violation_killer import (_boundary_violators, _violations_exact,  # noqa: E402
                              B_EPS)

SEP = 1e-7          # same separation tolerance the seat pass uses
BITS = ((1, 0, 0), (2, 0, 1), (4, 1, 1), (8, 1, 0))   # bit, axis, side


def build_opt(dataset, idx):
    sample = dataset[idx]
    area_target, b2b, p2b, pins, cons = sample["input"]
    polygons, _metrics = sample["label"]
    n = int((area_target != -1).sum().item())
    tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if not len(valid):
            continue
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        if nc > 1 and cons[i, 1] != 0:
            tpos[i] = torch.tensor([float(x_min), float(y_min),
                                    float(x_max - x_min), float(y_max - y_min)])
        elif nc > 0 and cons[i, 0] != 0:
            tpos[i, 2] = float(x_max - x_min)
            tpos[i, 3] = float(y_max - y_min)
    at = area_target[:n].detach().float().cpu()
    cons = cons[:n].detach().float().cpu()
    rects = [(0.0, 0.0, 1.0, 1.0)] * n
    import time as _t
    opt = _ColumnOptimizer(rects, at, cons, tpos,
                           b2b.detach().float().cpu(),
                           p2b.detach().float().cpu(),
                           pins.detach().float().cpu(),
                           _t.time() + 1.0, seed=0)
    return opt, n


def band_blockers(P, i, axis, side, wall):
    """Blocks standing between block i's tagged edge and its wall, inside
    i's own extent on the other axis."""
    o = 1 - axis
    lo_i = float(P[i, axis])
    hi_i = lo_i + float(P[i, axis + 2])
    olo = float(P[i, o])
    ohi = olo + float(P[i, o + 2])
    if side == 0:
        strip = (wall, lo_i)
    else:
        strip = (hi_i, wall)
    out = []
    for j in range(len(P)):
        if j == i:
            continue
        jlo, jhi = float(P[j, axis]), float(P[j, axis]) + float(P[j, axis + 2])
        klo, khi = float(P[j, o]), float(P[j, o]) + float(P[j, o + 2])
        if min(jhi, strip[1]) - max(jlo, strip[0]) <= SEP:
            continue
        if min(khi, ohi) - max(klo, olo) <= SEP:
            continue
        out.append(j)
    return out


def translate_clash(P, i, axis, side, wall):
    """Blocks hit by translating i straight onto the wall."""
    rect = P[i].copy()
    rect[axis] = wall if side == 0 else wall - rect[axis + 2]
    x0, y0, w, h = rect
    cx = np.minimum(x0 + w, P[:, 0] + P[:, 2]) - np.maximum(x0, P[:, 0])
    cy = np.minimum(y0 + h, P[:, 1] + P[:, 3]) - np.maximum(y0, P[:, 1])
    hit = (cx > SEP) & (cy > SEP)
    hit[i] = False
    return [int(j) for j in np.nonzero(hit)[0]]


def _frame_area(P, opt, axis, side, tag):
    """Bbox area if the wall on (axis, side) were pulled back to `tag`."""
    X0, Y0 = float(P[:, 0].min()), float(P[:, 1].min())
    X1 = float((P[:, 0] + P[:, 2]).max())
    Y1 = float((P[:, 1] + P[:, 3]).max())
    lo = [X0, Y0]
    hi = [X1, Y1]
    if side == 0:
        lo[axis] = tag
    else:
        hi[axis] = tag
    return (hi[0] - lo[0]) * (hi[1] - lo[1])


def classify(opt, P):
    X0, Y0 = float(P[:, 0].min()), float(P[:, 1].min())
    X1 = float((P[:, 0] + P[:, 2]).max())
    Y1 = float((P[:, 1] + P[:, 3]).max())
    walls = {(0, 0): X0, (0, 1): X1, (1, 0): Y0, (1, 1): Y1}
    span = (X1 - X0, Y1 - Y0)
    recs = []
    for i, code in _boundary_violators(opt, P):
        for bit, axis, side in BITS:
            if not (code & bit):
                continue
            wall = walls[(axis, side)]
            lo = float(P[i, axis])
            hi = lo + float(P[i, axis + 2])
            g = (lo - wall) if side == 0 else (wall - hi)
            if abs(g) < B_EPS:
                continue                        # this bit is satisfied
            rec = {
                "i": int(i), "code": int(code), "bit": bit,
                "gap": round(float(g), 4),
                "gap_frac": round(float(g) / max(span[axis], 1e-9), 4),
                "kind": int(opt.kind[i]),
                "cluster": int(opt.cluster[i]), "mib": int(opt.mib[i]),
                "area": round(float(opt.areas[i]), 1),
            }
            if opt.kind[i] == 2:
                # The tag line cannot move: the frame wall has to come back
                # to it.  Measure the overshoot population that defines the
                # wall instead -- that is what any repair would have to move.
                rec["cls"] = "locked"
                tag = float(P[i, axis]) if side == 0 else \
                    float(P[i, axis] + P[i, axis + 2])
                if side == 0:
                    over = [j for j in range(len(P))
                            if float(P[j, axis]) < tag - B_EPS]
                    exc = [tag - float(P[j, axis]) for j in over]
                else:
                    over = [j for j in range(len(P))
                            if float(P[j, axis] + P[j, axis + 2]) > tag + B_EPS]
                    exc = [float(P[j, axis] + P[j, axis + 2]) - tag
                           for j in over]
                rec["over_n"] = len(over)
                rec["over_max"] = round(max(exc), 3) if exc else 0.0
                rec["over_locked"] = sum(1 for j in over if opt.kind[j] == 2)
                rec["over_rigid"] = sum(1 for j in over if opt.kind[j] == 1)
                rec["over_clu"] = sum(1 for j in over if opt.cluster[j] > 0)
                # how much slack the layout has inside the tag frame
                rec["util"] = round(float(sum(opt.areas))
                                    / max(_frame_area(P, opt, axis, side, tag),
                                          1e-9), 4)
                recs.append(rec)
                continue
            blockers = band_blockers(P, i, axis, side, wall)
            tclash = translate_clash(P, i, axis, side, wall)
            rec["depth"] = len(blockers)
            rec["tclash"] = len(tclash)
            # which of the blockers actually sit ON the wall
            onwall = []
            for j in blockers:
                jlo = float(P[j, axis])
                jhi = jlo + float(P[j, axis + 2])
                d = (jlo - wall) if side == 0 else (wall - jhi)
                if abs(d) < 1e-4:
                    onwall.append(j)
            rec["onwall"] = len(onwall)
            if not tclash:
                rec["cls"] = "free"
                # would the seat pass's strict gate have taken it?
                Q = P.copy()
                Q[i, axis] = wall if side == 0 else wall - Q[i, axis + 2]
                rec["dV"] = int(_violations_exact(opt, Q)
                                - _violations_exact(opt, P))
            elif len(blockers) == 1 and len(onwall) == 1:
                j = onwall[0]
                rec["cls"] = "swap1"
                rec["peer"] = int(j)
                rec["peer_kind"] = int(opt.kind[j])
                rec["peer_tag"] = int(opt.boundary[j])
                rec["peer_cluster"] = int(opt.cluster[j])
                rec["peer_mib"] = int(opt.mib[j])
                rec["peer_area_ratio"] = round(
                    float(P[j, 2] * P[j, 3]) / max(float(P[i, 2] * P[i, 3]), 1e-9), 3)
                # same extent on the free axis? (a pure exchange needs it)
                o = 1 - axis
                rec["peer_o_ratio"] = round(
                    float(P[j, o + 2]) / max(float(P[i, o + 2]), 1e-9), 3)
            elif onwall:
                rec["cls"] = f"block{min(len(blockers), 9)}"
            else:
                rec["cls"] = "interior"
            recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--data-path", default=str(_REPO / "FloorSet"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    d = json.load(open(args.result))
    rows = {r["test_id"]: r for r in d["test_results"]}
    want = ([int(t) for t in args.ids.split(",") if t.strip()]
            if args.ids else sorted(rows))
    ds = FloorplanDatasetLiteTest(args.data_path)

    tally = Counter()
    wtally = Counter()
    lines = []
    for tid in want:
        r = rows.get(tid)
        if r is None or r.get("positions") is None:
            continue
        if r["block_count"] < args.min_n:
            continue
        opt, n = build_opt(ds, tid)
        P = np.asarray([list(p) for p in r["positions"]], dtype=np.float64)
        recs = classify(opt, P)
        bnd = int(r.get("boundary_violations", 0))
        print(f"tid={tid} n={n} bnd_eval={bnd} bits_unsat={len(recs)} "
              f"grp={r.get('grouping_violations')} mib={r.get('mib_violations')} "
              f"v_rel={r.get('violations_relative'):.4f} "
              f"cost={r.get('cost_no_runtime'):.4f}")
        for rec in sorted(recs, key=lambda x: -x["gap"]):
            print("   " + " ".join(f"{k}={v}" for k, v in rec.items()))
            tally[rec["cls"]] += 1
            if r["block_count"] >= 102:
                wtally[rec["cls"]] += 1
            lines.append(dict(rec, tid=tid, n=n))
    print("\nCLASS TALLY (all requested cases):", dict(tally))
    if wtally:
        print("CLASS TALLY (n>=102):", dict(wtally))
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")
        print("wrote", args.jsonl)


if __name__ == "__main__":
    main()
