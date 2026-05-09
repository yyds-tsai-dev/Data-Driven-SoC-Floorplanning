from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from floorset_arch.models import Instance, Rect


def _trim_tensor(tensor: Optional[torch.Tensor], cols: int) -> torch.Tensor:
    if tensor is None:
        return torch.empty(0, cols)
    tensor = torch.as_tensor(tensor).detach().cpu().float()
    if tensor.numel() == 0:
        return torch.empty(0, cols)
    if tensor.dim() == 1:
        tensor = tensor.reshape(0, cols)
    if tensor.shape[-1] != cols:
        tensor = tensor.reshape(-1, cols)
    return tensor[tensor[:, 0] >= 0]


def _constraints_tensor(constraints: Optional[torch.Tensor], block_count: int) -> torch.Tensor:
    if constraints is None:
        return torch.zeros(block_count, 5)
    out = torch.as_tensor(constraints).detach().cpu().float()
    if out.numel() == 0:
        return torch.zeros(block_count, 5)
    if out.dim() == 1:
        out = out.reshape(block_count, -1)
    out = out[:block_count]
    if out.shape[1] < 5:
        pad = torch.zeros(out.shape[0], 5 - out.shape[1])
        out = torch.cat([out, pad], dim=1)
    return out[:, :5]


def parse_instance(
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: Optional[torch.Tensor],
    target_positions: Optional[torch.Tensor] = None,
) -> Instance:
    areas = torch.as_tensor(area_targets).detach().cpu().float()[:block_count]
    cons = _constraints_tensor(constraints, block_count)
    pins = torch.as_tensor(pins_pos).detach().cpu().float()
    if pins.numel() == 0:
        pins = torch.empty(0, 2)
    pins = pins.reshape(-1, 2)
    valid_b2b = _trim_tensor(b2b_connectivity, 3)
    valid_p2b = _trim_tensor(p2b_connectivity, 3)
    b2b_by_block: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i_f, j_f, weight_f in valid_b2b.tolist():
        i, j, weight = int(i_f), int(j_f), float(weight_f)
        if 0 <= i < block_count and 0 <= j < block_count:
            b2b_by_block[i].append((j, weight))
            b2b_by_block[j].append((i, weight))
    p2b_by_block: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pin_f, block_f, weight_f in valid_p2b.tolist():
        pin_idx, block, weight = int(pin_f), int(block_f), float(weight_f)
        if 0 <= block < block_count and pin_idx >= 0:
            p2b_by_block[block].append((pin_idx, weight))

    fixed = {i for i in range(block_count) if cons[i, 0].item() != 0}
    preplaced = {i for i in range(block_count) if cons[i, 1].item() != 0}
    boundary = {
        i: int(cons[i, 4].item())
        for i in range(block_count)
        if cons[i, 4].item() > 0
    }

    mib_groups: dict[int, list[int]] = defaultdict(list)
    cluster_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(block_count):
        mib_id = int(cons[i, 2].item())
        cluster_id = int(cons[i, 3].item())
        if mib_id > 0:
            mib_groups[mib_id].append(i)
        if cluster_id > 0:
            cluster_groups[cluster_id].append(i)

    targets = None
    if target_positions is not None:
        targets = torch.as_tensor(target_positions).detach().cpu().float()
    target_rects: dict[int, Rect] = {}
    if targets is not None and targets.numel() > 0:
        targets = targets.reshape(-1, 4)
        for i in range(min(block_count, targets.shape[0])):
            x, y, w, h = [float(v) for v in targets[i].tolist()]
            if i in fixed or i in preplaced:
                if w > 0 and h > 0:
                    target_rects[i] = Rect(
                        x if x >= 0 else 0.0,
                        y if y >= 0 else 0.0,
                        w,
                        h,
                    )

    return Instance(
        block_count=block_count,
        area_targets=areas,
        b2b_connectivity=torch.as_tensor(b2b_connectivity).detach().cpu().float(),
        p2b_connectivity=torch.as_tensor(p2b_connectivity).detach().cpu().float(),
        pins_pos=pins,
        constraints=cons,
        fixed=fixed,
        preplaced=preplaced,
        mib_groups=dict(mib_groups),
        cluster_groups=dict(cluster_groups),
        boundary=boundary,
        target_rects=target_rects,
        valid_b2b=valid_b2b,
        valid_p2b=valid_p2b,
        b2b_by_block=dict(b2b_by_block),
        p2b_by_block=dict(p2b_by_block),
    )
