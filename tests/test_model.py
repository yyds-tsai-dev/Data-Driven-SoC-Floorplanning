import torch

from floorset_arch import features
from floorset_arch.features import build_anchor_edge_tensors, build_anchor_node_features
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.parser import parse_instance
from floorset_arch.training.losses import (
    build_pairwise_relation_targets,
    fp_sol_soft_violations,
    is_constraint_clean_training_sample,
)
from floorset_arch.training.checkpoint import anchor_checkpoint_payload


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


def test_floorplan_gnn_pairwise_logits_shape():
    model = FloorplanGNN(node_feat_dim=18, hidden_dim=16, num_layers=1)
    node_feat = torch.randn(4, 18)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.empty(0, 1)
    pairs = torch.tensor([[0, 1], [2, 3]])

    output = model(node_feat, edge_index, edge_attr, pairs=pairs)

    assert output["pair_logits"].shape == (2, 2)


def test_floorplan_gnn_graph_transformer_encoder_matches_output_contract():
    model = FloorplanGNN(
        node_feat_dim=18,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        encoder_type="graph-transformer",
        num_heads=4,
    )
    node_feat = torch.randn(4, 18)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.ones(3, 1)
    pairs = torch.tensor([[0, 1], [2, 3]])

    output = model(node_feat, edge_index, edge_attr, pairs=pairs)

    assert model.encoder_type == "graph-transformer"
    assert output["anchor"].shape == (4, 2)
    assert output["priority"].shape == (4,)
    assert output["log_aspect"].shape == (4,)
    assert output["pair_logits"].shape == (2, 2)


def test_graph_transformer_accepts_structural_and_edge_type_inputs():
    model = FloorplanGNN(
        node_feat_dim=18,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        encoder_type="graph-transformer",
        num_heads=4,
        structural_feat_dim=6,
        edge_type_count=5,
    )
    node_feat = torch.randn(4, 18)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.ones(3, 1)
    edge_type = torch.tensor([0, 2, 4], dtype=torch.long)
    structural_feat = torch.randn(4, 6)

    output = model(
        node_feat,
        edge_index,
        edge_attr,
        edge_type=edge_type,
        structural_feat=structural_feat,
        pairs=torch.tensor([[0, 3]]),
    )

    assert output["anchor"].shape == (4, 2)
    assert output["pair_logits"].shape == (1, 2)


def test_transformer_graph_inputs_project_hetero_context_to_anchor_encoder():
    inst = parse_instance(
        4,
        torch.tensor([4.0, 9.0, 16.0, 25.0]),
        torch.empty(0, 3),
        torch.tensor([[0.0, 0.0, 2.0], [0.0, 1.0, 3.0]]),
        torch.tensor([[10.0, 20.0]]),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 7.0, 1.0],
                [0.0, 0.0, 1.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 7.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 2.0],
            ]
        ),
        None,
    )

    graph_inputs = features.build_anchor_transformer_graph_inputs(inst)

    assert graph_inputs.node_structural_features.shape[0] == 4
    assert graph_inputs.node_structural_features.shape[1] >= 6
    assert graph_inputs.edge_index.shape[1] > 0
    assert graph_inputs.edge_type.shape[0] == graph_inputs.edge_index.shape[1]
    assert len(set(graph_inputs.edge_type.tolist())) >= 3


def test_anchor_checkpoint_payload_records_encoder_config():
    model = FloorplanGNN(
        node_feat_dim=18,
        hidden_dim=16,
        num_layers=2,
        encoder_type="graph-transformer",
        num_heads=4,
    )
    args = type("Args", (), {"example": "value"})()

    payload = anchor_checkpoint_payload(model, args, epoch=1, train_stats={}, val_stats={})

    assert payload["encoder_type"] == "graph-transformer"
    assert payload["num_heads"] == 4
    assert payload["has_pair_head"] is True


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


def test_pairwise_relation_targets_ignore_ambiguous_pairs():
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 5.0, 0.0],
            [2.0, 2.0, 5.5, 5.0],
            [2.0, 2.0, 6.0, 6.0],
        ]
    )
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3]])

    targets = build_pairwise_relation_targets(fp_sol, pairs, min_gap=1.0, clear_ratio=1.25)

    assert targets["x_label"].tolist() == [1.0, 0.0, 0.0]
    assert targets["y_label"].tolist() == [0.0, 1.0, 0.0]
    assert targets["mask"].tolist() == [True, True, False]


def test_constraint_clean_training_sample_detects_boundary_group_and_mib():
    constraints = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )
    clean_fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 0.0],
            [2.0, 2.0, 4.0, 0.0],
        ]
    )
    dirty_fp_sol = torch.tensor(
        [
            [2.0, 2.0, 5.0, 0.0],
            [3.0, 3.0, 10.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
        ]
    )

    assert fp_sol_soft_violations(
        clean_fp_sol,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
    ) == (0, 0, 0)
    assert is_constraint_clean_training_sample(
        clean_fp_sol,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
    )
    assert fp_sol_soft_violations(
        dirty_fp_sol,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
    ) == (2, 1, 1)
    assert not is_constraint_clean_training_sample(
        dirty_fp_sol,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
    )
