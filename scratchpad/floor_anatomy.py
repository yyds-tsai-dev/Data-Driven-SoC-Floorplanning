#!/usr/bin/env python3
"""Per-case fixed-cost anatomy at the goal operating point (budget 0.05 s).

Single process, single thread, no pool, no evaluator.  Replays what
legalize_rectangles -> _parallel_solve -> _worker_solve do for one case and
times each stage separately: parent build, numba kernel attach, payload
marshal, pickle/unpickle (pool dispatch proxy), worker build, prepare, and
finish() at the real worker span (with its deadline overshoot).

Usage:  FLOORSET_MAIN=<root> uv run python scratchpad/floor_anatomy.py [n ...]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PARTNER_SA_KERNEL", "numba")
os.environ.setdefault("PARTNER_REFINE_KERNEL", "numba")
os.environ.setdefault("PARTNER_REFINE_FASTBUILD", "1")
os.environ.setdefault("PARTNER_REFINE_STALL_STOP", "1")
os.environ.setdefault("PARTNER_POOL_GATE", "0")
os.environ.setdefault("PARTNER_BUDGET_SCALE", "5e-5")
os.environ.setdefault("PARTNER_BUDGET_TAU", "12")
os.environ.setdefault("PARTNER_BUDGET_MIN", "0.05")
os.environ.setdefault("PARTNER_BUDGET_MAX", "0.75")
os.environ.setdefault("VKILL_OFF", "1")

import torch  # noqa: E402

WT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ["FLOORSET_MAIN"])
for p in (WT / "partner", MAIN / "FloorSet", MAIN / "FloorSet/iccad2026contest"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch.set_num_threads(1)

BUDGET = float(os.environ.get("FLOOR_BUDGET", "0.05"))
K_CONFIGS = int(os.environ.get("FLOOR_CONFIGS", "24"))
REPS = int(os.environ.get("FLOOR_REPS", "7"))


def load_cases(targets):
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(MAIN / "FloorSet"))
    ns = []
    for tid in range(100):
        s = ds[tid]
        a = s["input"][0]
        ns.append((tid, int((a != -1).sum().item())))
    picked = []
    for t in targets:
        tid, n = min(ns, key=lambda kv: (abs(kv[1] - t), kv[0]))
        picked.append((tid, n))
    out = []
    for tid, n in picked:
        s = ds[tid]
        area_target, b2b, p2b, pins, cons = s["input"]
        polygons, _m = s["label"]
        tpos = torch.full((n, 4), -1.0)
        nc = cons.shape[1] if cons.dim() > 1 else 0
        for i in range(n):
            blk = polygons[i]
            valid = blk[blk[:, 0] != -1]
            if not len(valid):
                continue
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            is_fixed = nc > 0 and cons[i, 0] != 0
            is_pre = nc > 1 and cons[i, 1] != 0
            if is_pre:
                tpos[i] = torch.tensor([float(x0), float(y0),
                                        float(x1 - x0), float(y1 - y0)])
            elif is_fixed:
                tpos[i, 2] = float(x1 - x0)
                tpos[i, 3] = float(y1 - y0)
        out.append(dict(tid=tid, n=n, at=area_target[:n].float(),
                        cons=cons[:n].float(), tpos=tpos,
                        b2b=b2b.float(), p2b=p2b.float(), pins=pins.float()))
    return out


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def anatomy(case):
    import column_sa_legalizer as lg
    from contest_optimizer import _heuristic_init
    at = case["at"]
    cons = case["cons"]
    tpos = case["tpos"]
    b2b = case["b2b"]
    p2b = case["p2b"]
    pins = case["pins"]
    th = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        rects = _heuristic_init(at, cons, tpos, b2b, p2b, pins)
        th.append(time.perf_counter() - t0)

    def build(kernel=True, span=BUDGET):
        old = os.environ.get("PARTNER_SA_KERNEL", "")
        if not kernel:
            os.environ["PARTNER_SA_KERNEL"] = ""
        t0 = time.perf_counter()
        opt = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                  time.time() + span, seed=11)
        dt = time.perf_counter() - t0
        os.environ["PARTNER_SA_KERNEL"] = old
        return opt, dt

    res = {"heuristic_init": med(th)}
    t_build = []
    t_build_nk = []
    for _ in range(REPS):
        _o, dt = build(True)
        t_build.append(dt)
        _o2, dt2 = build(False)
        t_build_nk.append(dt2)
    res["parent_build"] = med(t_build)
    res["parent_build_nokernel"] = med(t_build_nk)
    res["kernel_attach"] = res["parent_build"] - res["parent_build_nokernel"]

    def np_of(t):
        return None if t is None else t.detach().cpu().numpy()

    cfgs = [("N", None, 11 + 7 * k, 1.0, 1.0) for k in range(K_CONFIGS)]
    wd = time.time() + BUDGET
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        payloads = [(list(rects), np_of(at), np_of(cons), np_of(tpos),
                     np_of(b2b), np_of(p2b), np_of(pins),
                     orient, cf, sd, wd, vw, hs)
                    for (orient, cf, sd, vw, hs) in cfgs]
        ts.append(time.perf_counter() - t0)
    res["marshal"] = med(ts)

    ts = []
    tu = []
    nbytes = 0
    for _ in range(REPS):
        t0 = time.perf_counter()
        blobs = [pickle.dumps(p, protocol=pickle.HIGHEST_PROTOCOL)
                 for p in payloads]
        ts.append(time.perf_counter() - t0)
        nbytes = len(blobs[0])
        t0 = time.perf_counter()
        pickle.loads(blobs[0])
        tu.append(time.perf_counter() - t0)
    res["pickle_K"] = med(ts)
    res["unpickle_1"] = med(tu)
    res["payload_bytes"] = nbytes
    args = payloads[0]

    def worker_setup(span):
        (rs, areas_np, cons_np, tpos_np, b2b_np, p2b_np, pins_np,
         _orient, _cf, seed, _dl, vw, hs) = args
        t0 = time.perf_counter()
        a = torch.from_numpy(areas_np)
        c = torch.from_numpy(cons_np) if cons_np is not None else None
        tp = torch.from_numpy(tpos_np) if tpos_np is not None else None
        bb = torch.from_numpy(b2b_np) if b2b_np is not None else None
        pb = torch.from_numpy(p2b_np) if p2b_np is not None else None
        pn = torch.from_numpy(pins_np) if pins_np is not None else None
        t_cast = time.perf_counter() - t0
        t0 = time.perf_counter()
        opt = lg._ColumnOptimizer(rs, a, c, tp, bb, pb, pn,
                                  time.time() + span, seed=seed,
                                  v_weight=vw, h_scale=hs)
        t_bld = time.perf_counter() - t0
        t0 = time.perf_counter()
        opt.prepare()
        t_prep = time.perf_counter() - t0
        return opt, t_cast, t_bld, t_prep

    tc = []
    tb = []
    tp = []
    for _ in range(REPS):
        _o, a_, b_, c_ = worker_setup(60.0)
        tc.append(a_)
        tb.append(b_)
        tp.append(c_)
    res["worker_cast"] = med(tc)
    res["worker_build"] = med(tb)
    res["worker_prepare"] = med(tp)

    head = res["parent_build"] + res["marshal"] + res["pickle_K"]
    span = max(BUDGET - head, 0.005) * 0.85
    tf = []
    tover = []
    for _ in range(REPS):
        opt, _a, _b, _c = worker_setup(span)
        t0 = time.perf_counter()
        opt.finish(time.time() + span, max_runs=1)
        dt = time.perf_counter() - t0
        tf.append(dt)
        tover.append(dt - span)
    res["worker_span_planned"] = span
    res["finish"] = med(tf)
    res["finish_overshoot"] = med(tover)
    return res


KEYS = ("heuristic_init", "parent_build", "parent_build_nokernel",
        "kernel_attach", "marshal", "pickle_K", "unpickle_1",
        "worker_cast", "worker_build", "worker_prepare",
        "worker_span_planned", "finish", "finish_overshoot")


def main():
    targets = [int(a) for a in sys.argv[1:]] or [25, 50, 70]
    for case in load_cases(targets):
        r = anatomy(case)
        print("")
        print("=== tid=%d n=%d budget=%s K=%d ===" % (
            case["tid"], case["n"], BUDGET, K_CONFIGS))
        for k in KEYS:
            print("  %-22s %9.2f ms" % (k, 1e3 * r[k]))
        print("  %-22s %9d" % ("payload_bytes", r["payload_bytes"]))
        serial = r["parent_build"] + r["marshal"] + r["pickle_K"]
        wall = (serial + r["unpickle_1"] + r["worker_cast"]
                + r["worker_build"] + r["worker_prepare"] + r["finish"])
        print("  -> serial head %.1f ms | modelled wall %.1f ms" % (
            1e3 * serial, 1e3 * wall))


if __name__ == "__main__":
    main()
