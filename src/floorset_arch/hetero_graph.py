from __future__ import annotations

from dataclasses import dataclass, field

import torch

from floorset_arch.features import build_anchor_node_features
from floorset_arch.models import Instance


@dataclass(frozen=True)
class HeteroEdge:
    src_type: str
    src: int
    dst_type: str
    dst: int
    edge_type: str
    weight: float = 1.0


@dataclass
class HeteroFloorplanGraph:
    node_features: dict[str, torch.Tensor] = field(default_factory=dict)
    edges: list[HeteroEdge] = field(default_factory=list)
    block_to_cluster: dict[int, int] = field(default_factory=dict)
    block_to_mib: dict[int, int] = field(default_factory=dict)
    block_to_boundary: dict[int, int] = field(default_factory=dict)

    def degree(self, block: int) -> float:
        total = 0.0
        for edge in self.edges:
            if edge.src_type == "block" and edge.src == block:
                total += edge.weight
            elif edge.dst_type == "block" and edge.dst == block:
                total += edge.weight
        return total


def build_hetero_floorplan_graph(inst: Instance) -> HeteroFloorplanGraph:
    """Build the runtime factor graph used by the beam decoder.

    The arch_new checkpoint still consumes the compact block graph in
    ``features.py``. This hetero graph is the solver-side representation that
    keeps pins and constraints explicit for masking and scoring decisions.
    """
    block_features, _scale = build_anchor_node_features(inst)
    graph = HeteroFloorplanGraph(node_features={"block": block_features})

    if inst.pins_pos.numel() > 0:
        pins = inst.pins_pos.float()
        coord_scale = torch.clamp(pins.abs().max(), min=1.0)
        graph.node_features["pin"] = pins / coord_scale
    else:
        graph.node_features["pin"] = torch.empty((0, 2), dtype=torch.float32)

    cluster_ids = sorted(inst.cluster_groups)
    mib_ids = sorted(inst.mib_groups)
    boundary_codes = sorted(set(inst.boundary.values()))
    graph.node_features["cluster"] = torch.eye(max(1, len(cluster_ids)), dtype=torch.float32)[: len(cluster_ids)]
    graph.node_features["mib"] = torch.eye(max(1, len(mib_ids)), dtype=torch.float32)[: len(mib_ids)]
    graph.node_features["boundary"] = torch.tensor(
        [[1.0 if code & bit else 0.0 for bit in (1, 2, 4, 8)] for code in boundary_codes],
        dtype=torch.float32,
    ).view(len(boundary_codes), 4)

    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i = int(i_f)
        j = int(j_f)
        weight = float(weight_f)
        if 0 <= i < inst.block_count and 0 <= j < inst.block_count and i != j:
            graph.edges.append(HeteroEdge("block", i, "block", j, "connects", weight))
            graph.edges.append(HeteroEdge("block", j, "block", i, "connects", weight))

    for pin_f, block_f, weight_f in inst.valid_p2b.tolist():
        pin = int(pin_f)
        block = int(block_f)
        weight = float(weight_f)
        if 0 <= block < inst.block_count and 0 <= pin < inst.pins_pos.shape[0]:
            graph.edges.append(HeteroEdge("pin", pin, "block", block, "pin_connects", weight))
            graph.edges.append(HeteroEdge("block", block, "pin", pin, "pin_connects", weight))

    for local_id, cluster_id in enumerate(cluster_ids):
        for block in inst.cluster_groups[cluster_id]:
            if 0 <= block < inst.block_count:
                graph.block_to_cluster[block] = local_id
                graph.edges.append(HeteroEdge("block", block, "cluster", local_id, "member_of", 1.0))
                graph.edges.append(HeteroEdge("cluster", local_id, "block", block, "has_member", 1.0))

    for local_id, mib_id in enumerate(mib_ids):
        for block in inst.mib_groups[mib_id]:
            if 0 <= block < inst.block_count:
                graph.block_to_mib[block] = local_id
                graph.edges.append(HeteroEdge("block", block, "mib", local_id, "same_shape_as", 1.0))
                graph.edges.append(HeteroEdge("mib", local_id, "block", block, "has_member", 1.0))

    boundary_lookup = {code: idx for idx, code in enumerate(boundary_codes)}
    for block, code in inst.boundary.items():
        if 0 <= block < inst.block_count and code in boundary_lookup:
            local_id = boundary_lookup[code]
            graph.block_to_boundary[block] = local_id
            graph.edges.append(HeteroEdge("block", block, "boundary", local_id, "wants_boundary", 1.0))
            graph.edges.append(HeteroEdge("boundary", local_id, "block", block, "has_member", 1.0))

    return graph
