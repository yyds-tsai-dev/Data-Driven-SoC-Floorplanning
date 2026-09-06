"""Free baseline: the UNDISTILLED teacher's quality vs sampler step count.

Measures, on held-out training samples with the deployment sampler contract
(euler, hard-anchor imposition, self-conditioning), the raw quantities the
candidate probe ranks on: coordinate L1 vs golden and overlap fraction.
If the teacher already survives 1-2 steps, distillation has little to buy.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = str(Path(__file__).resolve().parents[2])
for p in (f"{ROOT}/src/solver", f"{ROOT}/FloorSet/iccad2026contest", f"{ROOT}/FloorSet"):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=f"{ROOT}/checkpoints/flow_matching_v1/final.pt")
ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--offset", type=int, default=500_000,
                help="start index into the 1M training set")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

import direct_diffusion_train as V1
from diffusion_data import batch_to_device, fp_sol_to_z0, z_to_rectangles
from diffusion_train import known_target_positions_from_fp
from direct_diffusion_model import known_z_channels
from flow_matching_distill import load_teacher
from flow_matching_model import sample_flow
from iccad2026_evaluate import FloorplanDatasetLite, train_floorplan_collate

dev = torch.device(a.device)
model, cfg, _ = load_teacher(Path(a.checkpoint), dev)
ds = FloorplanDatasetLite(f"{ROOT}/FloorSet")
# NOTE: the 1M lite set is the teacher's *training* set, so absolute numbers
# here are optimistic.  The within-model step-count ratios, which is what this
# probe is read for, are unaffected.
idx = list(range(a.offset, a.offset + a.batches * a.batch_size))
loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(ds, idx), batch_size=a.batch_size,
    collate_fn=train_floorplan_collate, num_workers=4)

acc = {s: {"pos": 0.0, "ov": 0.0, "n": 0} for s in a.steps}
for batch in loader:
    batch = batch_to_device(batch, dev)
    area, b2b, p2b, pins, cons, _t, fp, _m = batch
    tp = known_target_positions_from_fp(fp, cons)
    z0, mask, scale = fp_sol_to_z0(fp, area, z_repr="xyaspect")
    cond = V1.fast_condition(area, b2b, p2b, pins, cons, target_positions=tp,
                             relation_feat_dim=cfg.relation_feat_dim,
                             node_feat_dim=cfg.node_feat_dim)
    zk, kn = known_z_channels(area, cons, tp, scale)
    for s in a.steps:
        g = torch.Generator().manual_seed(777)
        with torch.no_grad():
            z = sample_flow(model, cond, steps=s, solver="euler", generator=g,
                            z_known=zk, known_mask=kn).z
        rects = z_to_rectangles(z, area, target_positions=tp, constraints=cons,
                                z_repr="xyaspect")
        pos = (((z[..., :2] - z0[..., :2]).abs().sum(-1)) * mask).sum() / mask.sum().clamp_min(1)
        ov = V1.overlap_fraction(rects, mask, scale).mean()
        acc[s]["pos"] += float(pos)
        acc[s]["ov"] += float(ov)
        acc[s]["n"] += 1

base = None
print(f"teacher = {a.checkpoint.split('/')[-2]}/{a.checkpoint.split('/')[-1]}  "
      f"n={a.batches * a.batch_size} samples")
print(f"{'steps':>6} {'pos_l1':>9} {'overlap':>9} {'pos vs st8':>12} {'ov vs st8':>11}")
ref = acc[max(a.steps)]
rpos, rov = ref["pos"] / ref["n"], ref["ov"] / ref["n"]
for s in a.steps:
    d = acc[s]
    pos, ov = d["pos"] / d["n"], d["ov"] / d["n"]
    print(f"{s:>6} {pos:>9.5f} {ov:>9.5f} {pos / rpos:>11.3f}x {ov / max(rov, 1e-9):>10.3f}x")
