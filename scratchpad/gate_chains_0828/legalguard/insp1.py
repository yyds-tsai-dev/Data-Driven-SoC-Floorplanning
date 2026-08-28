import os, sys
REPO = "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning"
sys.path.insert(0, os.path.join(REPO, "tests"))
from synth_instances import build_instance
import torch
inst = build_instance(n=21, seed=1, frac_fixed=0.6)
c = inst.constraints
print("constraints shape", c.shape)
print("area_targets", [round(float(a),3) for a in inst.area_targets])
print("idx fixed preplaced boundary? cols:", c.shape[1])
for i in range(21):
    print(i, [float(x) for x in c[i]], "tpos", [round(float(x),3) for x in inst.target_positions[i]], "rect", [round(float(x),3) for x in inst.rects[i]] if hasattr(inst,'rects') else None)
