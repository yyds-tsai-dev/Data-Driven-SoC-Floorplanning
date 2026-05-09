import torch

from floorset_arch.features import build_anchor_edge_tensors, build_anchor_node_features
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.parser import parse_instance


def test_anchor_gnn_forward_matches_runtime_features():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 1.0]]),
        torch.tensor([[0.0, 2.0, 1.5]]),
        torch.tensor([[10.0, 20.0]]),
        torch.zeros(3, 5),
        None,
    )
    node_feat, _scale = build_anchor_node_features(inst)
    edge_index, edge_attr = build_anchor_edge_tensors(inst)
    model = FloorplanGNN(node_feat_dim=node_feat.shape[1], hidden_dim=16, num_layers=2)

    pred = model(node_feat, edge_index, edge_attr)

    assert pred["anchor"].shape == (3, 2)
    assert pred["priority"].shape == (3,)
    assert pred["log_aspect"].shape == (3,)


def test_hetero_graph_keeps_constraints_as_first_class_nodes():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0]]),
        torch.tensor([[0.0, 2.0, 1.5]]),
        torch.tensor([[10.0, 20.0]]),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 7.0, 1.0],
                [0.0, 0.0, 1.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0],
            ]
        ),
        None,
    )

    graph = build_hetero_floorplan_graph(inst)

    assert {"block", "pin", "cluster", "mib", "boundary"} <= set(graph.node_features)
    assert graph.block_to_cluster[0] == graph.block_to_cluster[1]
    assert graph.block_to_mib[0] == graph.block_to_mib[1]
    assert 2 in graph.block_to_boundary
