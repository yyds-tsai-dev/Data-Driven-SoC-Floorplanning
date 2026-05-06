from __future__ import annotations

from dataclasses import dataclass

import torch

from floorset_arch.models import Instance


@dataclass
class ModelInputs:
    block_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    pin_features: torch.Tensor


def build_model_inputs(inst: Instance) -> ModelInputs:
    n = inst.block_count
    device = inst.area_targets.device
    dtype = inst.area_targets.dtype
    areas = torch.clamp(inst.area_targets[:n].to(dtype=dtype), min=1.0)
    log_area = torch.log(areas).unsqueeze(1)
    log_area = log_area / torch.clamp(log_area.abs().max(), min=1.0)

    fixed = torch.zeros(n, 1, dtype=dtype, device=device)
    preplaced = torch.zeros(n, 1, dtype=dtype, device=device)
    mib = torch.zeros(n, 1, dtype=dtype, device=device)
    cluster = torch.zeros(n, 1, dtype=dtype, device=device)
    boundary_bits = torch.zeros(n, 4, dtype=dtype, device=device)
    for i in range(n):
        fixed[i, 0] = 1.0 if i in inst.fixed else 0.0
        preplaced[i, 0] = 1.0 if i in inst.preplaced else 0.0
        mib[i, 0] = float(inst.constraints[i, 2].item()) / 32.0
        cluster[i, 0] = float(inst.constraints[i, 3].item()) / 32.0
        code = inst.boundary.get(i, 0)
        boundary_bits[i] = torch.tensor(
            [1.0 if code & bit else 0.0 for bit in (1, 2, 4, 8)],
            dtype=dtype,
            device=device,
        )

    degree = torch.zeros(n, 1, dtype=dtype, device=device)
    pin_agg = torch.zeros(n, 3, dtype=dtype, device=device)
    edge_pairs = []
    edge_weights = []
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if i >= n or j >= n:
            continue
        weight = float(weight_f)
        degree[i, 0] += weight
        degree[j, 0] += weight
        edge_pairs.extend([(i, j), (j, i)])
        edge_weights.extend([weight, weight])

    for pin_f, block_f, weight_f in inst.valid_p2b.tolist():
        pin_idx, block = int(pin_f), int(block_f)
        if block >= n or pin_idx >= inst.pins_pos.shape[0]:
            continue
        weight = float(weight_f)
        px = float(inst.pins_pos[pin_idx, 0])
        py = float(inst.pins_pos[pin_idx, 1])
        pin_agg[block, 0] += weight * px
        pin_agg[block, 1] += weight * py
        pin_agg[block, 2] += weight
    pin_norm = torch.clamp(pin_agg[:, 2:3], min=1.0)
    pin_agg[:, :2] = pin_agg[:, :2] / pin_norm
    coord_scale = torch.clamp(pin_agg[:, :2].abs().max(), min=1.0)
    pin_agg[:, :2] = pin_agg[:, :2] / coord_scale
    degree = degree / torch.clamp(degree.max(), min=1.0)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=dtype, device=device)
        edge_weight = edge_weight / torch.clamp(edge_weight.max(), min=1.0)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        edge_weight = torch.empty(0, dtype=dtype, device=device)

    block_features = torch.cat(
        [log_area, fixed, preplaced, mib, cluster, boundary_bits, degree, pin_agg],
        dim=1,
    )
    return ModelInputs(
        block_features=block_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pin_features=pin_agg,
    )

