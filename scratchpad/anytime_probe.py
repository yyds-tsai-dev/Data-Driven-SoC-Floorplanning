#!/usr/bin/env python3
"""Single-threaded micro-run: time anatomy of `refine_prediction` under short
worker deadlines (the MAX=2.0 / worker deadline ~1.7 s regime).

No fork pool, no evaluator: one real validation case, real Direct predictions
sampled once on CPU and cached, then `refine_prediction` replayed at a sweep
of deadlines with every stage timed.

Usage:  uv run python scratchpad/anytime_probe.py <test_id> [spans...]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

WT = Path(__file__).resolve().parents[1]
MAIN = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
for p in (WT / "partner", MAIN / "FloorSet", MAIN / "FloorSet/iccad2026contest"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CACHE = WT / "scratchpad" / "anytime_preds"
CACHE.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# case + predictions
# --------------------------------------------------------------------------
def load_case(test_id: int):
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(MAIN / "FloorSet"))
    sample = ds[test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b, p2b, pins, cons = inputs
    n = int((area_target != -1).sum().item())
    polygons, _metrics = labels
    tpos_gt = []
    for i in range(n):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if len(valid):
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            tpos_gt.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            tpos_gt.append((0.0, 0.0, 1.0, 1.0))
    opt_tpos = torch.full((n, 4), -1.0)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and cons[i, 0] != 0
        is_pre = nc > 1 and cons[i, 1] != 0
        if is_pre:
            opt_tpos[i] = torch.tensor(list(tpos_gt[i]))
        elif is_fixed:
            opt_tpos[i, 2] = tpos_gt[i][2]
            opt_tpos[i, 3] = tpos_gt[i][3]
    return (n, area_target[:n].float(), cons[:n].float(), opt_tpos,
            b2b.float(), p2b.float(), pins.float(), tpos_gt)


def sample_preds(test_id: int, case, k: int = 6):
    f = CACHE / f"preds_{test_id}.npz"
    if f.exists():
        d = np.load(f)
        return [d[key] for key in sorted(d.files)]
    n, at, cons, tpos, b2b, p2b, pins, _ = case
    os.environ.setdefault("DIRECT_CKPT",
                          str(MAIN / "partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"))
    os.environ.setdefault("PARTNER_DIRECT_SOLVER", "dpmpp")
    os.environ.setdefault("PARTNER_DDIM_STEPS", "10")
    os.environ["PARTNER_OVERSAMPLE"] = "1"
    os.environ["PARTNER_FLOW_SLOTS"] = "0"
    import contest_optimizer as co
    co.init_worker_pool = lambda *a, **kw: None
    obj = co.MyOptimizer.__new__(co.MyOptimizer)
    obj.verbose = False
    obj.device = torch.device("cpu")
    obj.direct_model = None
    obj.flow_model = None
    obj.retrieval_index = None
    obj.retrieval_slots = 0
    obj.model = None
    obj.schedule = None
    t0 = time.time()
    co.MyOptimizer._load_direct_model(obj)
    print(f"[probe] direct model loaded in {time.time() - t0:.1f}s", flush=True)
    assert obj.direct_model is not None, "no direct checkpoint"
    t0 = time.time()
    preds = co.MyOptimizer._sample_direct_raw_preds(
        obj, n, at, cons, tpos, b2b, p2b, pins, k, oversample=False)
    print(f"[probe] sampled {len(preds)} preds in {time.time() - t0:.1f}s",
          flush=True)
    np.savez(f, **{f"p{i:02d}": p for i, p in enumerate(preds)})
    return preds


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------
class Trace:
    def __init__(self):
        self.ev = []
        self.t0 = 0.0
        self._patched = []

    def start(self):
        self.ev = []
        self.t0 = time.time()

    def rec(self, name, t_beg, t_end, extra=""):
        self.ev.append((name, t_beg - self.t0, t_end - t_beg, extra))

    def wrap_method(self, cls, name, label=None, extra_fn=None):
        real = getattr(cls, name)
        lbl = label or name
        tr = self

        def _w(self_, *a, **kw):
            tb = time.time()
            try:
                return real(self_, *a, **kw)
            finally:
                ex = ""
                try:
                    if extra_fn is not None:
                        ex = extra_fn(self_)
                except Exception:
                    ex = "?"
                tr.rec(lbl, tb, time.time(), ex)
        setattr(cls, name, _w)
        self._patched.append((cls, name, real))

    def wrap_func(self, mod, name):
        real = getattr(mod, name)
        tr = self

        def _w(*a, **kw):
            tb = time.time()
            try:
                return real(*a, **kw)
            finally:
                tr.rec(name, tb, time.time())
        setattr(mod, name, _w)
        self._patched.append((mod, name, real))

    def summary(self):
        agg = {}
        for name, beg, dur, _ex in self.ev:
            a = agg.setdefault(name, [0, 0.0])
            a[0] += 1
            a[1] += dur
        return agg


def build_opt(case, pred, span):
    import column_sa_legalizer as lg
    n, at, cons, tpos, b2b, p2b, pins, _ = case
    rect_list = [tuple(map(float, pred[i])) for i in range(n)]
    deadline = time.time() + span
    opt = lg._ColumnOptimizer(rect_list, at, cons, tpos, b2b, p2b, pins,
                              deadline, seed=317, v_weight=1.0)
    opt._tag_anchor = False
    return opt, deadline


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 95
    spans = [float(s) for s in sys.argv[2:]] or [3.2, 2.4, 1.7, 1.2, 0.8]
    case = load_case(test_id)
    n = case[0]
    print(f"[probe] test_id={test_id} n={n}", flush=True)
    preds = sample_preds(test_id, case)

    import layout_refiner as rf

    tr = Trace()
    tr.wrap_method(rf._Refiner, "__init__", "build")
    tr.wrap_method(rf._Refiner, "legalize_soft", "lsoft",
                   extra_fn=lambda s: f"ovl={s._overlap_count()}")
    tr.wrap_method(rf._Refiner, "_tighten", "tighten")
    tr.wrap_method(rf._Refiner, "run", "run")
    tr.wrap_method(rf._Refiner, "_assemble_clusters", "assemble")
    tr.wrap_func(rf, "_edge_seat")
    tr.wrap_func(rf, "_cluster_seat")
    tr.wrap_func(rf, "_lock_compact")

    rows = []
    for span in spans:
        for pi, pred in enumerate(preds[:3]):
            opt, deadline = build_opt(case, pred, span)
            tr.start()
            t_beg = time.time()
            out = rf.refine_prediction(opt, pred, deadline, seed=327)
            el = time.time() - t_beg
            if out is None:
                rows.append(dict(span=span, pred=pi, ok=False, elapsed=el,
                                 ev=list(tr.ev)))
                print(f"span={span:4.1f} pred={pi} NONE  elapsed={el:5.2f}"
                      f"  over={el - span:+.2f}", flush=True)
            else:
                pos = np.asarray(out, dtype=np.float64)
                hp = float(opt._hpwl(pos))
                area = float(((pos[:, 0] + pos[:, 2]).max() - pos[:, 0].min())
                             * ((pos[:, 1] + pos[:, 3]).max() - pos[:, 1].min()))
                V = int(rf.full_violations(opt, pos))
                rows.append(dict(span=span, pred=pi, ok=True, elapsed=el,
                                 hp=hp, area=area, V=V, ev=list(tr.ev)))
                print(f"span={span:4.1f} pred={pi} OK    elapsed={el:5.2f}"
                      f"  over={el - span:+.2f}  hp={hp:11.1f}"
                      f"  area/ref={area / opt.area_ref:5.3f}  V={V}",
                      flush=True)
            for name, beg, dur, ex in tr.ev:
                print(f"      {beg:6.3f} +{dur:6.3f}  {name:10s} {ex}",
                      flush=True)

    # proxy scores (same shape as _parallel_solve.score)
    okrows = [r for r in rows if r["ok"]]
    if okrows:
        hp_ref = min(r["hp"] for r in okrows)
        opt, _ = build_opt(case, preds[0], 1.0)
        den = max(getattr(opt, "n_soft_den", 1), 1)
        aref = opt.area_ref
        print("\n[proxy] span pred score")
        for r in okrows:
            s = (1.0 + 0.5 * ((r["hp"] - hp_ref) / hp_ref
                              + max(0.0, r["area"] / aref - 1.0))) \
                * math.exp(2.0 * r["V"] / den)
            r["score"] = s
            print(f"  {r['span']:4.1f} {r['pred']}  {s:.4f}")
    out_f = WT / "scratchpad" / f"anytime_probe_{test_id}.json"
    with open(out_f, "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "ev"} for r in rows],
                  fh, indent=1)
    print(f"\n[probe] wrote {out_f}")


if __name__ == "__main__":
    main()
