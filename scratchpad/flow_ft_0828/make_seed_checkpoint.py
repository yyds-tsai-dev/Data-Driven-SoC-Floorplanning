#!/usr/bin/env python3
"""Turn the shipped (stripped) flow v1 checkpoint into a resumable latest.pt.

``submission/cadc1013/checkpoints/flow_matching_v1_final.pt`` carries only
{model, ema, model_config, args, step}; ``direct_diffusion_train.main`` also
requires ``optimizer`` and ``sched`` on resume.  This builds the same objects
main() would build for the fine-tune's own args, loads v1's weights and EMA
shadow into them, and writes ``<ckpt-dir>/latest.pt`` at step 0 so a bare
launch auto-resumes into a fresh LR schedule with v1's parameters.

Usage: make_seed_checkpoint.py <ckpt_dir> <lr> <warmup> <max_steps> <ema_decay>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim import AdamW

REPO = Path("/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning")
for _p in (REPO / "partner", REPO / "FloorSet" / "iccad2026contest", REPO / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math  # noqa: E402

from direct_diffusion_model import DirectDenoiser, DirectModelConfig, EMA  # noqa: E402

SRC = REPO / "submission/cadc1013/checkpoints/flow_matching_v1_final.pt"


def main():
    ckdir = Path(sys.argv[1])
    lr = float(sys.argv[2])
    warmup = int(sys.argv[3])
    max_steps = int(sys.argv[4])
    ema_decay = float(sys.argv[5])
    ckdir.mkdir(parents=True, exist_ok=True)

    src = torch.load(SRC, map_location="cpu", weights_only=False)
    cfg = DirectModelConfig(**{k: v for k, v in src["model_config"].items()
                               if k in DirectModelConfig.__dataclass_fields__})
    model = DirectDenoiser(cfg)
    model.load_state_dict(src["model"])
    ema = EMA(model, ema_decay)
    ema.load_state_dict(src["ema"])

    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        (no_decay if p.dim() <= 1 else decay).append(p)
    optimizer = AdamW([{"params": decay, "weight_decay": 0.01},
                       {"params": no_decay, "weight_decay": 0.0}],
                      lr=lr, betas=(0.9, 0.99))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(max_steps - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args = dict(src["args"])
    args["training_method"] = "flow_matching_v1"  # keep the resume validator happy
    state = {"model": model.state_dict(), "ema": ema.state_dict(),
             "optimizer": optimizer.state_dict(), "sched": sched.state_dict(),
             "step": 0, "model_config": cfg.__dict__, "args": args}
    out = ckdir / "latest.pt"
    tmp = out.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(out)
    print(f"seed written: {out}  (from {SRC}, src step {src.get('step')})")
    print(f"  lr={lr} warmup={warmup} max_steps={max_steps} ema_decay={ema_decay}")


if __name__ == "__main__":
    main()
