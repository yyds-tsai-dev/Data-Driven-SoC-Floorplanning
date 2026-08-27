"""Bit-exact check: pristine vs patched `_sample_direct_raw_preds`, across
the PARTNER_QUOTA_FIRST toggle and flow on/off.

Four checks:
  (a) flow off, toggle off  -- must equal pristine (baseline no-op check)
  (b) flow off, toggle on   -- must equal pristine (quota-first degenerates
                                to all-Direct when there is no flow model)
  (c) flow ON,  toggle OFF  -- must equal pristine (PARTNER_QUOTA_FIRST=0 is
                                a true no-op, restores the legacy
                                replace-a-suffix behavior verbatim)
  (d) flow ON,  toggle ON   -- reported, not required to match pristine;
                                whether it matches tells us if the Flow
                                generator's seed consumption is independent
                                of how many Direct samples ran before it.

Builds a synthetic instance (same recipe as `_warm_flow_sampler`) and
compares raw predictions from pristine vs patched module copies with
identical seeding.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PARTNER_DIRECT_SOLVER", "dpmpp")
os.environ.setdefault("PARTNER_DDIM_STEPS", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault(
    "DIRECT_CKPT",
    "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning/"
    "artifacts/icdc_topology/checkpoints_s2_20k/best.pt")
FLOW_CKPT = (
    "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning/"
    "submission/cadc1013/checkpoints/flow_matching_v1_final.pt")

ROOT = Path(__file__).resolve().parents[2]
PARTNER = ROOT / "partner"
CONTEST = ROOT / "FloorSet" / "iccad2026contest"
FLOORSET = ROOT / "FloorSet"
sys.path.insert(0, str(PARTNER))
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(FLOORSET))

import importlib.util
import numpy as np
import torch


def _load(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_instance(n=100, seed=23456):
    g = torch.Generator().manual_seed(seed)
    at = torch.rand(n, generator=g) * 40.0 + 10.0
    cons = torch.zeros(n, 5)
    tpos = torch.full((n, 4), -1.0)
    cons[0, 1] = 1.0
    anchor_side = float(torch.sqrt(at[0]))
    tpos[0] = torch.tensor([0.0, 0.0, anchor_side, anchor_side])
    e = torch.randint(0, n, (3 * n, 2), generator=g).float()
    b2b = torch.cat([e, torch.ones(3 * n, 1)], dim=1)
    npin = 16
    pe = torch.stack([
        torch.randint(0, npin, (n,), generator=g).float(),
        torch.arange(n, dtype=torch.float32),
    ], dim=1)
    p2b = torch.cat([pe, torch.ones(n, 1)], dim=1)
    side = float(torch.sqrt(at.sum() / 0.96))
    pins = torch.rand(npin, 2, generator=g) * side
    return n, at, cons, tpos, b2b, p2b, pins


def _compare(label, preds_a, preds_b):
    equal_len = len(preds_a) == len(preds_b)
    ok = equal_len
    detail = []
    if not equal_len:
        detail.append(f"len mismatch {len(preds_a)} vs {len(preds_b)}")
    else:
        for i, (a, b) in enumerate(zip(preds_a, preds_b)):
            same = np.array_equal(a, b)
            ok = ok and same
            if not same:
                detail.append(f"pred[{i}] differs")
    print(f"[{label}] n_preds_a={len(preds_a)} n_preds_b={len(preds_b)} "
          f"EQUAL={ok}" + (f" ({'; '.join(detail)})" if detail else ""))
    return ok


def main():
    n, at, cons, tpos, b2b, p2b, pins = _build_instance()

    # ---- flow-off pair (checks a, b) --------------------------------
    for v in ("FLOW_CKPT", "PARTNER_FLOW_SLOTS", "PARTNER_FLOW_STEPS",
              "PARTNER_FLOW_ANTITHETIC", "PARTNER_FLOW_SOLVER",
              "PARTNER_QUOTA_FIRST"):
        os.environ.pop(v, None)

    pristine_noflow = _load("co_orig_noflow", PARTNER / "contest_optimizer_orig.py")
    patched_noflow = _load("co_patched_noflow", PARTNER / "contest_optimizer.py")
    opt_a1 = pristine_noflow.MyOptimizer(verbose=False)
    opt_b1 = patched_noflow.MyOptimizer(verbose=False)
    assert opt_a1.flow_model is None and opt_b1.flow_model is None

    ref_a = opt_a1._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K=9)

    os.environ["PARTNER_QUOTA_FIRST"] = "0"
    preds_b_off = opt_b1._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K=9)
    ok_a = _compare("a: flow=off toggle=off", ref_a, preds_b_off)

    os.environ["PARTNER_QUOTA_FIRST"] = "1"
    preds_b_on = opt_b1._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K=9)
    ok_b = _compare("b: flow=off toggle=on", ref_a, preds_b_on)

    # ---- flow-on pair (checks c, d) ----------------------------------
    os.environ["FLOW_CKPT"] = FLOW_CKPT
    os.environ["PARTNER_FLOW_SLOTS"] = "10"
    os.environ["PARTNER_FLOW_STEPS"] = "8"
    os.environ["PARTNER_FLOW_ANTITHETIC"] = "1"
    os.environ["PARTNER_FLOW_SOLVER"] = "euler"
    os.environ.pop("PARTNER_QUOTA_FIRST", None)

    pristine_flow = _load("co_orig_flow", PARTNER / "contest_optimizer_orig.py")
    patched_flow = _load("co_patched_flow", PARTNER / "contest_optimizer.py")
    opt_a2 = pristine_flow.MyOptimizer(verbose=False)
    opt_b2 = patched_flow.MyOptimizer(verbose=False)
    assert opt_a2.flow_model is not None and opt_b2.flow_model is not None

    ref_a2 = opt_a2._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K=9)

    os.environ["PARTNER_QUOTA_FIRST"] = "0"
    preds_b2_off = opt_b2._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K=9)
    ok_c = _compare("c: flow=ON toggle=OFF", ref_a2, preds_b2_off)

    os.environ["PARTNER_QUOTA_FIRST"] = "1"
    preds_b2_on = opt_b2._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b, pins, K=9)
    ok_d = _compare("d: flow=ON toggle=ON", ref_a2, preds_b2_on)

    print(f"SUMMARY a={ok_a} b={ok_b} c={ok_c} d={ok_d}")
    # (a), (b), (c) are hard requirements; (d) is informational only.
    if not (ok_a and ok_b and ok_c):
        sys.exit(1)


if __name__ == "__main__":
    main()
