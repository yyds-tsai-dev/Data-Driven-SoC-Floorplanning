#!/usr/bin/env python3
"""Export an EMA-only shipping checkpoint: {ema, model_config, args, step}.

`contest_optimizer._load_flow_model` loads `ckpt.get("ema") or ckpt["model"]`,
so dropping the raw `model` tensors halves the file (430 MB instead of 860 MB)
without changing what gets loaded.  Usage: export_ema_only.py <src.pt> <out.pt>
"""
import sys
from pathlib import Path
import torch

src, out = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(src, map_location="cpu", weights_only=False)
slim = {"ema": ck["ema"], "model_config": ck["model_config"], "args": ck["args"], "step": int(ck["step"])}
tmp = out.with_suffix(".pt.tmp"); torch.save(slim, tmp); tmp.replace(out)
print(f"{out} step={slim['step']} method={slim['args'].get('training_method')} bytes={out.stat().st_size}")
