from __future__ import annotations

from itertools import combinations

import torch

from floorset_arch import features
from floorset_arch.diffusion.contracts import DiffusionGraphInputs, Relation
from floorset_arch.features import build_anchor_node_features
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.models import Instance


def _id_tensor(values: dict[int, int], block_count: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(block_count, dtype=torch.long, device=device)
    for block, value in values.items():
        if 0 <= block < block_count:
            out[block] = int(value)
    return out


def _constraint_ids(inst: Instance, column: int, device: torch.device) -> torch.Tensor:
    if inst.constraints is None or inst.constraints.dim() < 2 or inst.constraints.shape[1] <= column:
        return torch.zeros(inst.block_count, dtype=torch.long, device=device)
    return inst.constraints[: inst.block_count, column].long().to(device)


def _block_mask(blocks: set[int], block_count: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(block_count, dtype=torch.bool, device=device)
    for block in blocks:
        if 0 <= block < block_count:
            mask[block] = True
    return mask


def _pair_set_from_instance(inst: Instance) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i_f, j_f, _weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if 0 <= i < inst.block_count and 0 <= j < inst.block_count and i != j:
            pairs.add(tuple(sorted((i, j))))
    for groups in (inst.cluster_groups, inst.mib_groups):
        for blocks in groups.values():
            valid_blocks = sorted({int(block) for block in blocks if 0 <= block < inst.block_count})
            pairs.update(tuple(pair) for pair in combinations(valid_blocks, 2))
    if not pairs:
        pairs.update(combinations(range(inst.block_count), 2))
    return sorted(pairs)


def _b2b_weights(inst: Instance) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if 0 <= i < inst.block_count and 0 <= j < inst.block_count and i != j:
            pair = tuple(sorted((i, j)))
            weights[pair] = weights.get(pair, 0.0) + max(float(weight_f), 0.0)
    return weights


def _pair_features(
    inst: Instance,
    pairs: list[tuple[int, int]],
    area: torch.Tensor,
    boundary_codes: torch.Tensor,
    cluster_ids: torch.Tensor,
    mib_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    b2b_weights = _b2b_weights(inst)
    max_b2b = max(b2b_weights.values(), default=1.0)
    rows: list[list[float]] = []
    area_cpu = area.detach().cpu().float().clamp_min(1e-6)
    boundary_cpu = boundary_codes.detach().cpu().long()
    cluster_cpu = cluster_ids.detach().cpu().long()
    mib_cpu = mib_ids.detach().cpu().long()
    for i, j in pairs:
        pair = tuple(sorted((i, j)))
        b2b_weight = b2b_weights.get(pair, 0.0)
        same_cluster = cluster_cpu[i].item() != 0 and cluster_cpu[i].item() == cluster_cpu[j].item()
        same_mib = mib_cpu[i].item() != 0 and mib_cpu[i].item() == mib_cpu[j].item()
        same_boundary = (
            boundary_cpu[i].item() != 0 and boundary_cpu[i].item() == boundary_cpu[j].item()
        )
        rows.append(
            [
                1.0 if b2b_weight > 0.0 else 0.0,
                b2b_weight / max(max_b2b, 1.0),
                1.0 if same_cluster else 0.0,
                1.0 if same_mib else 0.0,
                1.0 if same_boundary else 0.0,
                float(torch.log(area_cpu[i] / area_cpu[j]).item()),
            ]
        )
    if not rows:
        return torch.empty((0, 6), dtype=torch.float32, device=device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _typed_edges(
    inst: Instance,
    node_features: dict[str, torch.Tensor],
    block_features: torch.Tensor,
    device: torch.device,
) -> tuple[dict[Relation, torch.Tensor], dict[Relation, torch.Tensor]]:
    graph = build_hetero_floorplan_graph(inst, block_features=block_features.detach().cpu())
    edge_src: dict[Relation, list[int]] = {}
    edge_dst: dict[Relation, list[int]] = {}
    edge_weight: dict[Relation, list[float]] = {}
    canonical = set(features.ANCHOR_HGT_RELATION_SPECS)

    for edge in graph.edges:
        relation = (edge.src_type, edge.edge_type, edge.dst_type)
        if relation not in canonical:
            continue
        if edge.src >= node_features[edge.src_type].shape[0]:
            continue
        if edge.dst >= node_features[edge.dst_type].shape[0]:
            continue
        weight = max(float(edge.weight), 0.0)
        if edge.edge_type in {"connects", "pin_connects"}:
            weight = float(torch.log1p(torch.tensor(weight)).item())
        edge_src.setdefault(relation, []).append(int(edge.src))
        edge_dst.setdefault(relation, []).append(int(edge.dst))
        edge_weight.setdefault(relation, []).append(weight)

    edge_index: dict[Relation, torch.Tensor] = {}
    edge_attr: dict[Relation, torch.Tensor] = {}
    for relation in features.ANCHOR_HGT_RELATION_SPECS:
        if relation not in edge_src:
            edge_index[relation] = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr[relation] = torch.empty((0, 1), dtype=torch.float32, device=device)
            continue
        weights = torch.tensor(edge_weight[relation], dtype=torch.float32, device=device).view(-1, 1)
        weights = weights / weights.max().clamp_min(1.0)
        edge_index[relation] = torch.tensor(
            [edge_src[relation], edge_dst[relation]], dtype=torch.long, device=device
        )
        edge_attr[relation] = weights
    return edge_index, edge_attr


def _global_features(
    inst: Instance,
    area: torch.Tensor,
    scale: float,
    pair_count: int,
    fixed_mask: torch.Tensor,
    preplaced_mask: torch.Tensor,
    movable_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    n = max(float(inst.block_count), 1.0)
    values = [
        float(inst.block_count),
        float(inst.pins_pos.shape[0]),
        float(inst.valid_b2b.shape[0]),
        float(inst.valid_p2b.shape[0]),
        float(pair_count),
        float(fixed_mask.float().mean().item()) if inst.block_count else 0.0,
        float(preplaced_mask.float().mean().item()) if inst.block_count else 0.0,
        float(movable_mask.float().mean().item()) if inst.block_count else 0.0,
        float(area.sum().item()),
        float(scale),
    ]
    norm = torch.tensor([n, max(n, 1.0), max(n, 1.0), max(n, 1.0), max(n * n, 1.0), 1, 1, 1, max(float(area.sum().item()), 1.0), max(float(scale), 1.0)], dtype=torch.float32, device=device)
    return torch.tensor(values, dtype=torch.float32, device=device) / norm


def build_diffusion_graph_inputs(
    inst: Instance,
    device: torch.device | None = None,
) -> DiffusionGraphInputs:
    device = device or inst.area_targets.device
    block_features, scale = build_anchor_node_features(inst, device=device)
    hetero_graph = build_hetero_floorplan_graph(inst, block_features=block_features.detach().cpu())
    node_features = {
        node_type: value.to(device=device, dtype=torch.float32)
        for node_type, value in hetero_graph.node_features.items()
    }
    for node_type, width in (("block", block_features.shape[1]), ("pin", 2), ("cluster", 0), ("mib", 0), ("boundary", 4)):
        if node_type not in node_features:
            node_features[node_type] = torch.empty((0, width), dtype=torch.float32, device=device)

    edge_index, edge_attr = _typed_edges(inst, node_features, block_features, device)
    area = inst.area_targets[: inst.block_count].float().to(device)
    fixed_mask = _block_mask(inst.fixed, inst.block_count, device)
    preplaced_mask = _block_mask(inst.preplaced, inst.block_count, device)
    movable_mask = ~(fixed_mask | preplaced_mask)
    boundary_codes = _id_tensor(inst.boundary, inst.block_count, device)
    cluster_ids = _constraint_ids(inst, 3, device)
    mib_ids = _constraint_ids(inst, 2, device)

    pairs = _pair_set_from_instance(inst)
    pair_index = (
        torch.tensor(pairs, dtype=torch.long, device=device)
        if pairs
        else torch.empty((0, 2), dtype=torch.long, device=device)
    )
    raw_pair_features = _pair_features(
        inst,
        pairs,
        area,
        boundary_codes,
        cluster_ids,
        mib_ids,
        device,
    )

    return DiffusionGraphInputs(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        relation_specs=features.ANCHOR_HGT_RELATION_SPECS,
        raw_block_features=block_features,
        raw_pair_features=raw_pair_features,
        pair_index=pair_index,
        area=area,
        scale=float(scale),
        fixed_mask=fixed_mask,
        preplaced_mask=preplaced_mask,
        movable_mask=movable_mask,
        boundary_codes=boundary_codes,
        cluster_ids=cluster_ids,
        mib_ids=mib_ids,
        global_features=_global_features(
            inst,
            area,
            float(scale),
            len(pairs),
            fixed_mask,
            preplaced_mask,
            movable_mask,
            device,
        ),
    )
