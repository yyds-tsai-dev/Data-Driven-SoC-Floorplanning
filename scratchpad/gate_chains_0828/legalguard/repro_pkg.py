"""Reproduce fixedheavy_n21_s1 through the WORKING TREE partner/ with the
packaged env (op_wrapper's defaults, applied via setdefault the same way)."""
import os, sys, time, json
REPO = "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning"
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, "/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/pack_test6/x/cadc1013")
sys.path.insert(0, os.path.join(REPO, "FloorSet", "iccad2026contest"))
sys.path.insert(0, os.path.join(REPO, "FloorSet"))

_TAB = open(os.path.join(REPO, "artifacts/p0_newbox/budget_table_mid.txt")).read().strip()
for k, v in {
    "DIRECT_CKPT": os.path.join(REPO, "artifacts/icdc_topology/checkpoints_s2_20k/best.pt"),
    "FLOW_CKPT": os.path.join(REPO, "submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt"),
    "VKILL_OFF": "1", "PARTNER_POOL": "24", "PARTNER_NREF": "9", "PARTNER_NREF_MIN_N": "95",
    "PARTNER_DIRECT_MIN": "0.3", "PARTNER_DIRECT_SEAT_FIX": "1", "PARTNER_OVERSAMPLE": "1",
    "PARTNER_KS_CAP": "6", "PARTNER_DIRECT_SOLVER": "dpmpp", "PARTNER_DDIM_STEPS": "2",
    "PARTNER_FLOW_SOLVER": "euler", "PARTNER_FLOW_STEPS": "8", "PARTNER_FLOW_ANTITHETIC": "1",
    "PARTNER_FLOW_SLOTS": "10", "PARTNER_PRESCREEN_V": "1", "PARTNER_TAG_ANCHOR_EXTRA": "3",
    "PARTNER_BUDGET_SCALE": "8.498e-5", "PARTNER_BUDGET_TAU": "12", "PARTNER_BUDGET_MIN": "0.05",
    "PARTNER_BUDGET_MAX": "1.22", "PARTNER_POOL_GATE": "0", "PARTNER_REFINE_STALL_STOP": "1",
    "PARTNER_SA_KERNEL": "numba", "PARTNER_REFINE_KERNEL": "numba", "PARTNER_REFINE_FASTBUILD": "1",
    "PARTNER_FAST_SETUP": "1", "PARTNER_EDGE_SEAT_V2": "1", "PARTNER_FRAME_WPIN": "1",
    "PARTNER_FRAME_SCALE_LADDER": "1", "PARTNER_FRAME_SCALE_SET": "1.02", "PARTNER_SEAT_FINAL": "1",
    "PARTNER_TAG_COMPRESS": "1", "PARTNER_GROUP_BRIDGE": "1", "PARTNER_GPU_ARM": "0",
    "PARTNER_RETRIEVAL_SLOTS": "0", "PARTNER_REFINE_RES_FRAC": "0.45", "PARTNER_WALL_REPAIR": "1",
    "PARTNER_FLOW_WARM": "1", "PARTNER_EARLY_EXIT": "1", "PARTNER_REFINE_SECURE_FALLBACK": "1",
    "PARTNER_BUDGET_TABLE": _TAB,
}.items():
    os.environ.setdefault(k, v)

import torch
from synth_instances import build_instance
from iccad2026_evaluate import evaluate_solution
from op_wrapper import MyOptimizer


def make_target_positions(n, constraints, tpos):
    out = torch.full((n, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_pre = nc > 1 and constraints[i, 1] != 0
        if is_pre:
            out[i] = torch.tensor(list(tpos[i]))
        elif is_fixed:
            out[i, 2] = tpos[i][2]; out[i, 3] = tpos[i][3]
    return out


def overlaps(pos, eps=1e-9):
    bad = []
    n = len(pos)
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > eps and oy > eps:
                bad.append((i, j, ox, oy, ox * oy))
    return bad


def run(kw, reps=1, label=""):
    n = kw["n"]
    inst = build_instance(**kw)
    tp = make_target_positions(n, inst.constraints, inst.target_positions)
    opt = MyOptimizer()
    for r in range(reps):
        t0 = time.time()
        pos = opt.solve(n, inst.area_targets, inst.b2b, inst.p2b, inst.pins,
                        inst.constraints, tp)
        dt = time.time() - t0
        m = evaluate_solution({'positions': pos, 'runtime': dt}, {}, inst.constraints,
                              inst.b2b, inst.p2b, inst.pins, inst.area_targets,
                              inst.target_positions, median_runtime=1.0)
        ov = overlaps(pos)
        print(f"{label}{kw} rep{r}: feasible={m.is_feasible} overlap_v={m.overlap_violations} "
              f"dim_v={m.dimension_violations} fixed_v={m.fixed_violations} "
              f"prepl_v={m.preplaced_violations} area_v={m.area_violations} rt={dt:.3f} "
              f"pairs={[(a,b,round(c,4),round(d,4)) for a,b,c,d,_ in ov]}")
        sys.stdout.flush()
    return inst, tp


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run(dict(n=21, seed=1, frac_fixed=0.6), reps=reps)
