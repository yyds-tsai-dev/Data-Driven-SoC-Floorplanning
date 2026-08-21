#!/usr/bin/env python3
"""Sub-anatomy of `_Refiner.run`'s `discrete` bucket (PARTNER_REFINE_PROF_DISC).

Single process, single thread, deterministic synthetic instances (no dataset,
no evaluator, no pool) -- the same discipline as `refine_kernel_e2e.py`, so a
teammate's eval batch on the box cannot contaminate the split.

`discrete` is 58.4% of `run()` wall clock on the real-case profile; this asks
what INSIDE it is worth moving to numba.  The sub-buckets partition `discrete`
only (see `_DPROF_KEYS` in `partner/layout_refiner.py`); the `_PROF_KEYS`
invariant `sum + other == span` is untouched.

Usage:  uv run python scratchpad/disc_micro.py [span]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "tests", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The real records (144 of them, /tmp/qsweep_prof/0806_2235) are ALL
# `refine_positions` calls -- n in [115, 120], span ~0.23 s, `enable_deflate`
# False -- i.e. the pool `_worker_refine` tail on the large band, not the
# `refine_prediction` step-5 refiner.  Reproduce THAT: a legalized layout, a
# 0.23 s deadline, many seeds.  (Driving `refine_prediction` instead puts 86%
# of the span in `_deflate`, which is 0.000 s in the real profile.)
CASES = [
    dict(n=118, seed=0),
    dict(n=116, seed=53, n_preplaced=5, n_clusters=8, anchor_clusters=4),
    dict(n=120, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]
REPS = int(os.environ.get("DISC_MICRO_REPS", "8"))


def run_case(case, span, outdir, env):
    for k, v in env.items():
        os.environ[k] = v
    import layout_refiner as rf
    from synth_instances import build_instance, make_optimizer
    inst = build_instance(**case)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    # one legalization, reused by every rep: `refine_positions` is documented
    # to take a LEGAL layout, and re-legalizing per rep would be measured too
    opt0 = make_optimizer(inst, seed=3, deadline=time.time() + 600.0)
    r0 = rf._Refiner(opt0, pred, seed=11)
    r0.legalize(60, deadline=time.time() + 60.0)
    legal = r0.P.copy()
    el = 0.0
    for rep in range(REPS):
        dl = time.time() + span
        opt = make_optimizer(inst, seed=3, deadline=dl)
        t0 = time.time()
        rf.refine_positions(opt, legal, dl, seed=100 + rep)
        el += time.time() - t0
    return legal, el / REPS


def collect(outdir):
    recs = []
    for f in sorted(Path(outdir).glob("dprof.jsonl.*")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                recs.append(json.loads(ln))
    return recs


def report(recs, label):
    import layout_refiner as rf
    span = sum(r["span"] for r in recs)
    disc = sum(r["discrete"] for r in recs)
    sw = sum(r.get("swaps", -1) for r in recs)
    bt = sum(r["batches"] for r in recs)
    print(f"\n=== {label}: {len(recs)} run() calls, "
          f"span {span:.3f}s, discrete {disc:.3f}s "
          f"({100.0 * disc / max(span, 1e-9):.1f}% of span), "
          f"{bt} batches, {sw} accepts ===")
    print(f"{'top bucket':>14s} {'sec':>9s} {'% span':>8s}")
    for k in sorted(rf._PROF_KEYS, key=lambda k: -sum(r[k] for r in recs)):
        t = sum(r[k] for r in recs)
        if t <= 0:
            continue
        print(f"{k:>14s} {t:9.4f} {100.0 * t / max(span, 1e-9):7.2f}%")
    oth = sum(r["other"] for r in recs)
    print(f"{'other':>14s} {oth:9.4f} {100.0 * oth / max(span, 1e-9):7.2f}%")

    if "d_enum_optpt" not in recs[0]:
        print("  (no DISC sub-buckets in these records)")
        return
    print(f"\n{'discrete sub':>14s} {'sec':>9s} {'% disc':>8s} "
          f"{'% span':>8s} {'calls':>10s}")
    rows = []
    for k in rf._DPROF_KEYS:
        t = sum(r["d_" + k] for r in recs)
        c = sum(r["dc_" + k] for r in recs)
        rows.append((t, k, c))
    rows.sort(reverse=True)
    for t, k, c in rows:
        if t <= 0 and c == 0:
            continue
        print(f"{k:>14s} {t:9.4f} {100.0 * t / max(disc, 1e-9):7.2f}% "
              f"{100.0 * t / max(span, 1e-9):7.2f}% {c:10d}")
    un = sum(r["d_unattributed"] for r in recs)
    print(f"{'unattributed':>14s} {un:9.4f} "
          f"{100.0 * un / max(disc, 1e-9):7.2f}% "
          f"{100.0 * un / max(span, 1e-9):7.2f}%")

    # the two families a kernel port would actually target
    hp = sum(r["d_enum_gain"] + r["d_screen_delta"] + r["d_m_cost"]
             for r in recs)
    op = sum(r["d_enum_optpt"] + r["d_m_enum"] for r in recs)
    lc = sum(r["d_lc_axis"] + r["d_lc_ovl"] + r["d_lc_key"] for r in recs)
    print(f"\n  grouped: _block_hp family  {hp:8.4f}s "
          f"({100.0 * hp / max(disc, 1e-9):5.1f}% disc)")
    print(f"           _optimal_point     {op:8.4f}s "
          f"({100.0 * op / max(disc, 1e-9):5.1f}% disc)")
    print(f"           _legal_check core  {lc:8.4f}s "
          f"({100.0 * lc / max(disc, 1e-9):5.1f}% disc)  [already numba]")


def main():
    span = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    outdir = Path(os.environ.get(
        "DISC_MICRO_OUT",
        "/tmp/disc_micro_%d" % os.getpid()))
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("dprof.jsonl.*"):
        f.unlink()
    env = {
        # production refine env (see .env) + the two diagnostics
        "PARTNER_REFINE_KERNEL": "numba",
        "PARTNER_REFINE_FASTBUILD": "1",
        "PARTNER_REFINE_PROF": "1",
        "PARTNER_REFINE_PROF_DISC": "1",
        "PARTNER_REFINE_PROF_FILE": str(outdir / "dprof.jsonl"),
    }
    if os.environ.get("DISC_MICRO_QSWEEP", "1") not in ("0", ""):
        env["PARTNER_REFINE_KERNEL_QSWEEP"] = "1"
    if os.environ.get("DISC_MICRO_MATCH"):
        env["PARTNER_MATCH"] = "1"
    if os.environ.get("DISC_MICRO_DISC"):
        env["PARTNER_REFINE_KERNEL_DISC"] = "1"
        env["PARTNER_REFINE_DISC_PARTS"] = os.environ.get(
            "DISC_MICRO_PARTS", "fuse,hp")

    for case in CASES:
        out, el = run_case(case, span, outdir, env)
        tag = "n%ds%d" % (case["n"], case["seed"])
        print(f"[{tag}] refine_positions x{REPS} mean {el:.3f}s", flush=True)

    recs = collect(outdir)
    if not recs:
        print("no prof records written")
        return
    import layout_refiner  # noqa: F401  (report needs the key tuples)
    report(recs, "all cases, span=%.1fs" % span)


if __name__ == "__main__":
    main()
