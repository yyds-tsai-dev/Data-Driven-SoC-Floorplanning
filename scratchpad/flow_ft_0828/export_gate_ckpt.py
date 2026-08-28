#!/usr/bin/env python3
"""Strip a fine-tune snapshot down to the shipped FLOW_CKPT format.

The trainer's step_*.pt carries optimizer + sched state (~860 MB).  The
shipped ``flow_matching_v1_final.pt`` carries only
{model, ema, model_config, args, step} (~430 MB), which is what
``contest_optimizer._load_flow_model`` reads (it prefers ``ema``).  This
writes that stripped form so the gate runs against exactly the artifact
that would ship.

Usage: export_gate_ckpt.py <step_XXXXXXXX.pt> [out.pt]
"""
import sys
from pathlib import Path

import torch

src = Path(sys.argv[1])
ck = torch.load(src, map_location="cpu", weights_only=False)
step = int(ck["step"])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    src.parent.parent / f"flow_ft0828_tailT24_lr2e-5_step{step // 1000}k.pt")
slim = {k: ck[k] for k in ("model", "ema", "model_config", "args")}
slim["step"] = step
tmp = out.with_suffix(".pt.tmp")
torch.save(slim, tmp)
tmp.replace(out)
print(f"{out}  step={step}  method={slim['args'].get('training_method')}")
