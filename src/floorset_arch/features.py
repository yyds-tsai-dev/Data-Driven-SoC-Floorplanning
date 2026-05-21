from __future__ import annotations

from dataclasses import dataclass

import torch

from floorset_arch.models import Instance


ANCHOR_EDGE_TYPE_B2B = 0
ANCHOR_EDGE_TYPE_PIN = 1
ANCHOR_EDGE_TYPE_CLUSTER = 2
ANCHOR_EDGE_TYPE_MIB = 3
ANCHOR_EDGE_TYPE_BOUNDARY = 4
ANCHOR_EDGE_TYPE_COUNT = 5


@dataclass(frozen=True)
class AnchorTransformerGraphInputs:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_type: torch.Tensor
    node_structural_features: torch.Tensor
    edge_type_count: int = ANCHOR_EDGE_TYPE_COUNT


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


def _append_bidirectional_context_edges(
    src: list[int],
    dst: list[int],
    weights: list[float],
    edge_types: list[int],
    blocks: list[int],
    edge_type: int,
    weight: float,
) -> None:
    unique_blocks = sorted({int(block) for block in blocks})
    for i, block_i in enumerate(unique_blocks):
        for block_j in unique_blocks[i + 1 :]:
            if block_i == block_j:
                continue
            src.extend([block_i, block_j])
            dst.extend([block_j, block_i])
            weights.extend([weight, weight])
            edge_types.extend([edge_type, edge_type])


def _spectral_positional_features(
    n: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
    dims: int = 2,
) -> torch.Tensor:
    if n == 0 or dims <= 0:
        return torch.empty((n, 0), dtype=torch.float32, device=device)
    if n == 1 or edge_index.numel() == 0:
        return torch.zeros((n, dims), dtype=torch.float32, device=device)

    adjacency = torch.zeros((n, n), dtype=torch.float32, device=device)
    src = edge_index[0]
    dst = edge_index[1]
    weight = edge_attr.flatten().float().clamp_min(0.0)
    adjacency[dst, src] = torch.maximum(adjacency[dst, src], weight)
    adjacency = torch.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(dim=1)
    deg_inv_sqrt = degree.clamp_min(1e-6).rsqrt()
    normalized = deg_inv_sqrt[:, None] * adjacency * deg_inv_sqrt[None, :]
    laplacian = torch.eye(n, dtype=torch.float32, device=device) - normalized
    try:
        eigvec = torch.linalg.eigh(laplacian).eigenvectors[:, 1 : dims + 1]
    except RuntimeError:
        return torch.zeros((n, dims), dtype=torch.float32, device=device)
    if eigvec.shape[1] < dims:
        eigvec = torch.cat(
            [eigvec, torch.zeros((n, dims - eigvec.shape[1]), dtype=eigvec.dtype, device=device)],
            dim=1,
        )
    return eigvec.float()


def build_anchor_transformer_graph_inputs(
    inst: Instance, device: torch.device | None = None
) -> AnchorTransformerGraphInputs:
    """Build graph-transformer-only structural inputs from the hetero floorplan graph.

    Existing MPNN checkpoints keep using ``build_anchor_edge_tensors``. The
    transformer encoder gets additional factor-derived block context so pins,
    grouping, MIB, and boundary constraints influence attention without changing
    the block-level output contract.
    """
    from floorset_arch.hetero_graph import build_hetero_floorplan_graph

    n = inst.block_count
    device = device or inst.area_targets.device
    base_edge_index, base_edge_attr = build_anchor_edge_tensors(inst, device=device)
    src = base_edge_index[0].detach().cpu().tolist()
    dst = base_edge_index[1].detach().cpu().tolist()
    weights = base_edge_attr.flatten().detach().cpu().tolist()
    edge_types = [ANCHOR_EDGE_TYPE_B2B for _ in weights]

    graph = build_hetero_floorplan_graph(inst)
    factor_members: dict[tuple[str, int], list[int]] = {}
    for edge in graph.edges:
        if edge.src_type == "block" and edge.dst_type in {"pin", "cluster", "mib", "boundary"}:
            factor_members.setdefault((edge.dst_type, edge.dst), []).append(edge.src)
        elif edge.dst_type == "block" and edge.src_type in {"pin", "cluster", "mib", "boundary"}:
            factor_members.setdefault((edge.src_type, edge.src), []).append(edge.dst)

    factor_type = {
        "pin": ANCHOR_EDGE_TYPE_PIN,
        "cluster": ANCHOR_EDGE_TYPE_CLUSTER,
        "mib": ANCHOR_EDGE_TYPE_MIB,
        "boundary": ANCHOR_EDGE_TYPE_BOUNDARY,
    }
    factor_weight = {
        "pin": 0.75,
        "cluster": 0.85,
        "mib": 0.80,
        "boundary": 0.65,
    }
    for (factor, _local_id), blocks in factor_members.items():
        _append_bidirectional_context_edges(
            src,
            dst,
            weights,
            edge_types,
            blocks,
            factor_type[factor],
            factor_weight[factor],
        )

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_attr = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
        edge_attr = edge_attr / edge_attr.max().clamp_min(1.0)
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)

    b2b_degree = torch.zeros(n, dtype=torch.float32, device=device)
    pin_degree = torch.zeros(n, dtype=torch.float32, device=device)
    context_degree = torch.zeros(n, dtype=torch.float32, device=device)
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i = int(i_f)
        j = int(j_f)
        weight = max(_safe_float(weight_f), 0.0)
        if 0 <= i < n and 0 <= j < n:
            b2b_degree[i] += weight
            b2b_degree[j] += weight
    for _pin_f, block_f, weight_f in inst.valid_p2b.tolist():
        block = int(block_f)
        if 0 <= block < n:
            pin_degree[block] += max(_safe_float(weight_f), 0.0)
    if edge_index.numel() > 0:
        context_degree.index_add_(0, edge_index[1], edge_attr.flatten())

    cluster_size = torch.zeros(n, dtype=torch.float32, device=device)
    for blocks in inst.cluster_groups.values():
        size = float(len(blocks))
        for block in blocks:
            if 0 <= block < n:
                cluster_size[block] = size
    mib_size = torch.zeros(n, dtype=torch.float32, device=device)
    for blocks in inst.mib_groups.values():
        size = float(len(blocks))
        for block in blocks:
            if 0 <= block < n:
                mib_size[block] = size
    boundary_count = torch.zeros(n, dtype=torch.float32, device=device)
    for block, code in inst.boundary.items():
        if 0 <= block < n:
            boundary_count[block] = float(sum(1 for bit in (1, 2, 4, 8) if int(code) & bit))

    structural = torch.stack(
        [
            b2b_degree / b2b_degree.max().clamp_min(1.0),
            pin_degree / pin_degree.max().clamp_min(1.0),
            context_degree / context_degree.max().clamp_min(1.0),
            cluster_size / max(float(n), 1.0),
            mib_size / max(float(n), 1.0),
            boundary_count / 4.0,
        ],
        dim=1,
    )
    spectral = _spectral_positional_features(n, edge_index, edge_attr, device=device, dims=2)
    structural = torch.cat([structural, spectral], dim=1).float()

    return AnchorTransformerGraphInputs(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        node_structural_features=structural,
    )
