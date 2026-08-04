#!/usr/bin/env python3
"""Prototype: value-ordered (anytime) direct-refine ladder vs the shipped one.

Shipped order: fixed-frame rung -> tight expand rungs -> loose rung -> tighten
-> refiner -> repair tail.  At short spans the failing rungs eat everything and
the pipeline returns None (or a loose-frame layout with a zero-second tail).

Anytime order: secure a legal layout with the rung that (almost) always
legalizes, then spend the rest monotonically -- frame anneal (`_tighten`),
refiner (`run` = HPWL + squeeze + deflate), repair/seat tail -- every stage
interruptible with a legal layout already in hand.

Usage: uv run python scratchpad/anytime_proto.py <test_id> [spans...]
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "scratchpad"))
from anytime_probe import build_opt, load_case, sample_preds  # noqa: E402

import layout_refiner as rf  # noqa: E402


def _proxy(opt, pos):
    hp = float(opt._hpwl(pos))
    area = float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                 * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min()))
    V = int(rf.full_violations(opt, pos))
    return hp, area, V


def anytime_refine(opt, pred, deadline, seed=327, loose=0.28,
                   f_secure=0.45, f_tighten=0.30, f_reserve=0.18,
                   trace=None):
    """Anytime ladder prototype (mirrors the pieces of refine_prediction)."""
    t0 = time.time()
    span = max(0.0, deadline - t0)
    t_hard = deadline
    res = f_reserve * span
    dl = deadline - res

    P0 = np.array(pred, dtype=np.float64, copy=True)
    # step 1: MIB shape unification (verbatim from refine_prediction)
    for idxs in opt.mib_groups.values():
        if len(idxs) < 2:
            continue
        ref = next((i for i in idxs if opt.fixed[i] or opt.preplaced[i]), None)
        if ref is not None:
            w, h = float(opt.rw[ref]), float(opt.rh[ref])
        else:
            la = float(np.mean([math.log(P0[i, 2] / P0[i, 3]) for i in idxs]))
            area = float(opt.areas[idxs[0]])
            w = math.sqrt(area * math.exp(la))
            h = area / w
        for i in idxs:
            if opt.kind[i] != 2 and not (opt.fixed[i] or opt.preplaced[i]):
                cx = P0[i, 0] + 0.5 * P0[i, 2]
                cy = P0[i, 1] + 0.5 * P0[i, 3]
                P0[i] = (cx - 0.5 * w, cy - 0.5 * h, w, h)
    pred_c = np.stack([P0[:, 0] + 0.5 * P0[:, 2], P0[:, 1] + 0.5 * P0[:, 3]],
                      axis=1)
    saved_cg = opt.cluster_groups

    # -- phase A: secure a legal layout with the loose rung ----------------
    t_sec = time.time() + f_secure * max(0.0, dl - time.time())
    legal = None
    for expand in (loose, 0.55):
        if legal is not None or time.time() >= dl:
            break
        try:
            opt.cluster_groups = {}
            r = rf._Refiner(opt, P0, seed)
        finally:
            opt.cluster_groups = saved_cg
        r._pred_c = pred_c
        r.xmax += expand * (r.xmax - r.xmin)
        r.ymax += expand * (r.ymax - r.ymin)
        for g in r.groups:
            g.pin_x = g.pin_y = False
        r.satL[:] = r.satR[:] = r.satB[:] = r.satT[:] = False
        r._anchor_frame_to_tags()
        ok = r.legalize_soft(deadline=min(dl, t_sec))
        if trace is not None:
            trace.append(("secure%.2f" % expand, time.time() - t0, ok))
        if not ok:
            continue
        # tag recovery (verbatim intent from the pin-less rung)
        snap_t = r.P.copy()
        r._seed_tags()
        if not r.legalize_soft(10, deadline=min(dl, t_sec)):
            r.P[...] = snap_t
            for g in r.groups:
                g.pin_x = g.pin_y = False
            r.satL[:] = r.satR[:] = r.satB[:] = r.satT[:] = False
        r._assemble_clusters(saved_cg)
        if not r._has_overlap():
            legal = r.P.copy()
    if legal is None:
        return None, []

    # -- phase B: frame anneal, monotone + interruptible -------------------
    t_tig = time.time() + f_tighten * max(0.0, dl - time.time())
    r._tighten(min(dl, t_tig))
    r._assemble_clusters(saved_cg)
    if not r._has_overlap():
        legal = r.P.copy()
    if trace is not None:
        trace.append(("tighten", time.time() - t0, None))

    # -- phase C: refiner (HPWL + squeeze + deflate) -----------------------
    r2 = rf._Refiner(opt, legal, seed + 1)
    r2._pred_c = pred_c
    r2.enable_deflate = True
    r2._anchor_frame_to_tags()
    out = r2.run(dl - 0.02)
    if trace is not None:
        trace.append(("run", time.time() - t0, None))

    # -- phase D: repair + seat tail (inside the reserve) ------------------
    hp, area, V = _proxy(opt, np.asarray(out, dtype=np.float64))
    if V > 0 and time.time() < t_hard - 0.15:
        Q = np.array(out, dtype=np.float64, copy=True)
        try:
            r3 = rf._Refiner(opt, Q, seed + 2)
            r3._pred_c = pred_c
            r3._anchor_frame_to_tags()
            r3._seed_tags()
            if r3.legalize_soft(10, deadline=min(t_hard - 0.05,
                                                 time.time() + 0.5 * res)):
                r3._assemble_clusters(saved_cg)
                if not r3._has_overlap():
                    hp1, a1, V1 = _proxy(opt, r3.P)
                    if V1 < V or (V1 == V and hp1 < hp):
                        out = r3.P.copy()
        except Exception:
            pass
    out = rf._cluster_seat(opt, rf._edge_seat(opt, out),
                           deadline=t_hard - 0.02)
    if trace is not None:
        trace.append(("tail", time.time() - t0, None))
    return out, trace


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 95
    spans = [float(s) for s in sys.argv[2:]] or [3.2, 2.4, 1.7, 1.2, 0.8]
    case = load_case(test_id)
    preds = sample_preds(test_id, case)
    print(f"[proto] test_id={test_id} n={case[0]}", flush=True)

    rows = []
    for span in spans:
        for pi, pred in enumerate(preds[:3]):
            for mode in ("ship", "anytime"):
                opt, deadline = build_opt(case, pred, span)
                tb = time.time()
                tr = []
                if mode == "ship":
                    out = rf.refine_prediction(opt, pred, deadline, seed=327)
                else:
                    out, tr = anytime_refine(opt, pred, deadline, seed=327,
                                             trace=tr)
                el = time.time() - tb
                if out is None:
                    print(f"span={span:4.1f} p{pi} {mode:8s} NONE "
                          f"el={el:5.2f}", flush=True)
                    rows.append((span, pi, mode, None, None, None, el))
                    continue
                pos = np.asarray(out, dtype=np.float64)
                hp, area, V = _proxy(opt, pos)
                rows.append((span, pi, mode, hp, area / opt.area_ref, V, el))
                print(f"span={span:4.1f} p{pi} {mode:8s} OK   el={el:5.2f} "
                      f"over={el - span:+5.2f} hp={hp:8.1f} "
                      f"a/ref={area / opt.area_ref:5.3f} V={V} "
                      f"{[(a, round(b, 2), c) for a, b, c in tr]}", flush=True)

    ok = [r for r in rows if r[3] is not None]
    hp_ref = min(r[3] for r in ok) if ok else 1.0
    opt, _ = build_opt(case, preds[0], 1.0)
    den = max(getattr(opt, "n_soft_den", 1), 1)
    print("\n span  pred  mode      proxy   (lower is better)")
    for (span, pi, mode, hp, ar, V, el) in rows:
        if hp is None:
            print(f" {span:4.1f}  {pi}     {mode:8s}  ---")
            continue
        s = (1.0 + 0.5 * ((hp - hp_ref) / hp_ref + max(0.0, ar - 1.0))) \
            * math.exp(2.0 * V / den)
        print(f" {span:4.1f}  {pi}     {mode:8s}  {s:.4f}")


if __name__ == "__main__":
    main()
