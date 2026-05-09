from __future__ import annotations

import torch

from floorset_arch.models import Instance


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def build_anchor_node_features(inst: Instance, device: torch.device | None = None) -> tuple[torch.Tensor, float]:
    """Feature builder compatible with arch_new Anchor-GNN checkpoints."""
    n = inst.block_count
    device = device or inst.area_targets.device
    area = torch.clamp(inst.area_targets[:n].float().to(device), min=1.0)
    sqrt_area = torch.sqrt(area)
    total_area = torch.clamp(area.sum(), min=1.0)
    scale = float(torch.sqrt(total_area).item())

    degree = torch.zeros(n, device=device)
    pin_degree = torch.zeros(n, device=device)
    pin_cx = torch.zeros(n, device=device)
    pin_cy = torch.zeros(n, device=device)
    pin_wsum = torch.zeros(n, device=device)

    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i = int(i_f)
        j = int(j_f)
        weight = _safe_float(weight_f)
        if 0 <= i < n and 0 <= j < n:
            degree[i] += weight
            degree[j] += weight

    for pin_f, block_f, weight_f in inst.valid_p2b.tolist():
        pin_idx = int(pin_f)
        block_idx = int(block_f)
        weight = _safe_float(weight_f)
        if 0 <= block_idx < n and 0 <= pin_idx < inst.pins_pos.shape[0]:
            px = _safe_float(inst.pins_pos[pin_idx, 0])
            py = _safe_float(inst.pins_pos[pin_idx, 1])
            if px != -1.0 and py != -1.0:
                pin_degree[block_idx] += weight
                pin_cx[block_idx] += weight * px
                pin_cy[block_idx] += weight * py
                pin_wsum[block_idx] += weight

    has_pin = pin_wsum > 0.0
    pin_cx = torch.where(has_pin, pin_cx / pin_wsum.clamp_min(1e-6), torch.zeros_like(pin_cx))
    pin_cy = torch.where(has_pin, pin_cy / pin_wsum.clamp_min(1e-6), torch.zeros_like(pin_cy))
    max_degree = torch.clamp(degree.max(), min=1.0)
    max_pin_degree = torch.clamp(pin_degree.max(), min=1.0)

    fixed = torch.zeros(n, device=device)
    preplaced = torch.zeros(n, device=device)
    mib_flag = torch.zeros(n, device=device)
    cluster_flag = torch.zeros(n, device=device)
    left_b = torch.zeros(n, device=device)
    right_b = torch.zeros(n, device=device)
    top_b = torch.zeros(n, device=device)
    bottom_b = torch.zeros(n, device=device)
    mib_norm = torch.zeros(n, device=device)
    cluster_norm = torch.zeros(n, device=device)

    if inst.constraints is not None and inst.constraints.dim() > 1:
        c = inst.constraints[:n].to(device)
        if c.shape[1] > 0:
            fixed = (c[:, 0] != 0).float()
        if c.shape[1] > 1:
            preplaced = (c[:, 1] != 0).float()
        if c.shape[1] > 2:
            mib_id = c[:, 2].float()
            mib_flag = (mib_id != 0).float()
            mib_norm = mib_id / torch.clamp(mib_id.abs().max(), min=1.0)
        if c.shape[1] > 3:
            cluster_id = c[:, 3].float()
            cluster_flag = (cluster_id != 0).float()
            cluster_norm = cluster_id / torch.clamp(cluster_id.abs().max(), min=1.0)
        if c.shape[1] > 4:
            bc = c[:, 4].long()
            left_b = ((bc & 1) != 0).float()
            right_b = ((bc & 2) != 0).float()
            top_b = ((bc & 4) != 0).float()
            bottom_b = ((bc & 8) != 0).float()

    feat = torch.stack(
        [
            sqrt_area / max(scale, 1.0),
            torch.log1p(area) / 10.0,
            degree / max_degree,
            torch.log1p(degree) / 10.0,
            pin_degree / max_pin_degree,
            has_pin.float(),
            pin_cx / max(scale, 1.0),
            pin_cy / max(scale, 1.0),
            fixed,
            preplaced,
            mib_flag,
            cluster_flag,
            left_b,
            right_b,
            top_b,
            bottom_b,
            mib_norm,
            cluster_norm,
        ],
        dim=1,
    )
    return feat.float(), scale


def build_anchor_edge_tensors(inst: Instance, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge tensors compatible with arch_new Anchor-GNN checkpoints."""
    n = inst.block_count
    device = device or inst.area_targets.device
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i = int(i_f)
        j = int(j_f)
        weight = max(_safe_float(weight_f), 0.0)
        if 0 <= i < n and 0 <= j < n and i != j:
            ww = torch.log1p(torch.tensor(weight)).item()
            src.extend([i, j])
            dst.extend([j, i])
            weights.extend([ww, ww])
    if not src:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, 1), dtype=torch.float32, device=device),
        )
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
    edge_attr = edge_attr / edge_attr.max().clamp_min(1.0)
    return edge_index, edge_attr
