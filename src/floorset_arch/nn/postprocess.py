from __future__ import annotations

import math

import torch

from floorset_arch.models import Instance, ModelPrediction, Rect


def _layout_scale(inst: Instance) -> torch.Tensor:
    total_area = torch.clamp(inst.area_targets[: inst.block_count], min=1.0).sum()
    pin_span = torch.tensor(0.0, dtype=inst.area_targets.dtype, device=inst.area_targets.device)
    if inst.pins_pos.numel() > 0:
        pin_span = torch.clamp(inst.pins_pos.max() - inst.pins_pos.min(), min=1.0)
    return torch.maximum(torch.sqrt(total_area) * 2.0, pin_span)


def predictions_to_positions(inst: Instance, pred: ModelPrediction) -> torch.Tensor:
    n = inst.block_count
    dtype = pred.centers.dtype
    device = pred.centers.device
    areas = torch.clamp(inst.area_targets[:n].to(device=device, dtype=dtype), min=1.0)
    scale = _layout_scale(inst).to(device=device, dtype=dtype)
    aspect = torch.exp(pred.log_aspect[:n])
    widths = torch.sqrt(areas * aspect)
    heights = torch.sqrt(areas / aspect)
    centers = pred.centers[:n] * scale
    x = centers[:, 0] - widths / 2.0
    y = centers[:, 1] - heights / 2.0
    positions = torch.stack([x, y, widths, heights], dim=1)

    rows = []
    for i in range(n):
        if i in inst.preplaced and i in inst.target_rects:
            target = inst.target_rects[i]
            rows.append(torch.tensor(target.as_tuple(), dtype=dtype, device=device))
        elif i in inst.fixed and i in inst.target_rects:
            target = inst.target_rects[i]
            rows.append(torch.stack([positions[i, 0], positions[i, 1], torch.tensor(target.width, dtype=dtype, device=device), torch.tensor(target.height, dtype=dtype, device=device)]))
        else:
            rows.append(positions[i])
    return torch.stack(rows, dim=0)


def predictions_to_rects(inst: Instance, pred: ModelPrediction) -> dict[int, Rect]:
    positions = predictions_to_positions(inst, pred).detach().cpu()
    hints = {}
    for i in range(inst.block_count):
        x, y, w, h = [float(v) for v in positions[i].tolist()]
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h):
            hints[i] = Rect(max(0.0, x), max(0.0, y), max(1e-6, w), max(1e-6, h))
    return hints

