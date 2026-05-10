from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model,
    model_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": dict(model_config),
        "extra": extra or {},
    }
    torch.save(payload, output)


def build_run_tag(args, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%m%d")
    parts = [
        stamp,
        f"ns{int(args.num_samples)}",
        f"ep{int(args.epochs)}",
        f"h{int(args.hidden_dim)}",
        f"l{int(args.layers)}",
        f"acc{int(getattr(args, 'accumulation_steps', 1))}",
    ]
    return "_".join(parts)


def anchor_checkpoint_payload(
    model, args, epoch: int, train_stats, val_stats, optimizer=None
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model.state_dict(),
        "node_feat_dim": model.node_feat_dim,
        "hidden_dim": model.hidden_dim,
        "layers": model.num_layers,
        "dropout": getattr(model, "dropout_p", None),
        "epoch": epoch,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_anchor_checkpoint(
    path: str | Path,
    model,
    args,
    epoch: int,
    train_stats,
    val_stats,
    optimizer=None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        anchor_checkpoint_payload(
            model, args, epoch, train_stats, val_stats, optimizer=optimizer
        ),
        output,
    )
    return output


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
