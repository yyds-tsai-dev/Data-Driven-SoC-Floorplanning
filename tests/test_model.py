import sys

import pytest
import torch

from floorset_arch import features
from floorset_arch.features import build_anchor_edge_tensors, build_anchor_node_features
from floorset_arch.nn import model as model_module
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.parser import parse_instance
from floorset_arch.training.losses import (
    build_pairwise_relation_targets,
    fp_sol_soft_violations,
    is_constraint_clean_training_sample,
)
from floorset_arch.training.pseudo_targets import (
    PseudoTargetConfig,
    TrainingTargetSource,
    build_training_target_record,
)
from floorset_arch.training.checkpoint import anchor_checkpoint_payload, build_run_tag
from floorset_arch.training.selection import (
    CheckpointMetricRecord,
    better_checkpoint_metric,
)
from floorset_arch.training import train as train_module


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


def _hgt_sample_instance():
    return parse_instance(
        4,
        torch.tensor([4.0, 9.0, 16.0, 25.0]),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 3.0, 3.0]]),
        torch.tensor([[10.0, 20.0], [30.0, 5.0]]),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 7.0, 1.0],
                [0.0, 0.0, 1.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 7.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 4.0],
            ]
        ),
        None,
    )


def test_hgt_graph_inputs_keep_typed_local_relations():
    inst = _hgt_sample_instance()

    graph_inputs = features.build_anchor_hgt_graph_inputs(inst)

    assert {"block", "pin", "cluster", "mib", "boundary"} <= set(graph_inputs.node_features)
    assert graph_inputs.node_features["block"].shape == (4, 18)
    assert graph_inputs.node_feat_dims == {
        node_type: feat.shape[1] for node_type, feat in graph_inputs.node_features.items()
    }
    relations = set(graph_inputs.relation_specs)
    assert ("block", "connects", "block") in relations
    assert ("pin", "pin_connects", "block") in relations
    assert ("block", "pin_connects", "pin") in relations
    assert ("cluster", "has_member", "block") in relations
    assert ("mib", "has_member", "block") in relations
    assert ("boundary", "has_member", "block") in relations
    for relation in graph_inputs.relation_specs:
        edge_index = graph_inputs.edge_index[relation]
        edge_attr = graph_inputs.edge_attr[relation]
        assert edge_index.shape[0] == 2
        assert edge_attr.shape == (edge_index.shape[1], 1)


def test_hgt_relation_specs_are_canonical_even_when_edges_are_absent():
    inst = parse_instance(
        2,
        torch.tensor([4.0, 9.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(2, 5),
        None,
    )

    graph_inputs = features.build_anchor_hgt_graph_inputs(inst)

    assert graph_inputs.relation_specs == features.ANCHOR_HGT_RELATION_SPECS
    for relation in features.ANCHOR_HGT_RELATION_SPECS:
        assert relation in graph_inputs.edge_index
        assert graph_inputs.edge_index[relation].shape == (2, 0)
        assert graph_inputs.edge_attr[relation].shape == (0, 1)


def test_hgt_graph_inputs_reuse_precomputed_block_features(monkeypatch):
    inst = _hgt_sample_instance()
    block_features, scale = build_anchor_node_features(inst)

    def fail_recompute(*_args, **_kwargs):
        raise AssertionError("block features should be reused")

    monkeypatch.setattr(features, "build_anchor_node_features", fail_recompute)
    import floorset_arch.hetero_graph as hetero_graph

    monkeypatch.setattr(hetero_graph, "build_anchor_node_features", fail_recompute)

    graph_inputs = features.build_anchor_hgt_graph_inputs(
        inst, block_features=block_features, scale=scale
    )

    assert torch.equal(graph_inputs.node_features["block"], block_features)


def test_hgt_edge_softmax_does_not_require_torch_unique(monkeypatch):
    scores = torch.tensor(
        [[1.0, 0.0], [2.0, 1.0], [0.5, 0.5], [0.0, 2.0]], dtype=torch.float32
    )
    dst = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def fail_unique(*_args, **_kwargs):
        raise AssertionError("edge softmax should use vectorized scatter ops")

    monkeypatch.setattr(torch, "unique", fail_unique)

    alpha = model_module._edge_softmax_by_dst(scores, dst, dst_count=2)

    expected = torch.stack(
        [
            torch.softmax(scores[:2, 0], dim=0),
            torch.softmax(scores[:2, 1], dim=0),
            torch.softmax(scores[2:, 0], dim=0),
            torch.softmax(scores[2:, 1], dim=0),
        ]
    )
    assert torch.allclose(alpha[:, 0], torch.cat([expected[0], expected[2]]))
    assert torch.allclose(alpha[:, 1], torch.cat([expected[1], expected[3]]))


def test_batch_anchor_hgt_graph_inputs_offsets_nodes_and_edges():
    left = features.build_anchor_hgt_graph_inputs(_hgt_sample_instance())
    right = features.build_anchor_hgt_graph_inputs(_hgt_sample_instance())

    batch = features.batch_anchor_hgt_graph_inputs([left, right])

    assert batch.node_features["block"].shape[0] == 8
    assert batch.node_batch["block"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    relation = ("block", "connects", "block")
    left_edges = left.edge_index[relation].shape[1]
    assert torch.equal(batch.edge_index[relation][:, :left_edges], left.edge_index[relation])
    assert torch.equal(
        batch.edge_index[relation][:, left_edges:],
        right.edge_index[relation] + left.node_features["block"].shape[0],
    )
    assert batch.sample_block_slices == [(0, 4), (4, 8)]


def test_floorplan_gnn_hgt_encoder_matches_output_contract():
    inst = _hgt_sample_instance()
    graph_inputs = features.build_anchor_hgt_graph_inputs(inst)
    model = FloorplanGNN(
        node_feat_dim=graph_inputs.node_features["block"].shape[1],
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        encoder_type="hgt",
        num_heads=4,
        hgt_node_feat_dims=graph_inputs.node_feat_dims,
        hgt_relation_specs=graph_inputs.relation_specs,
    )
    pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    output = model(
        graph_inputs.node_features["block"],
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 1)),
        hgt_node_features=graph_inputs.node_features,
        hgt_edge_index=graph_inputs.edge_index,
        hgt_edge_attr=graph_inputs.edge_attr,
        pairs=pairs,
    )

    assert model.encoder_type == "hgt"
    assert output["anchor"].shape == (inst.block_count, 2)
    assert output["priority"].shape == (inst.block_count,)
    assert output["log_aspect"].shape == (inst.block_count,)
    assert output["pair_logits"].shape == (2, 2)


def test_floorplan_gnn_hgt_batched_forward_matches_separate_graphs():
    inst = _hgt_sample_instance()
    first = features.build_anchor_hgt_graph_inputs(inst)
    second = features.build_anchor_hgt_graph_inputs(inst)
    batch = features.batch_anchor_hgt_graph_inputs([first, second])
    model = FloorplanGNN(
        node_feat_dim=first.node_features["block"].shape[1],
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        encoder_type="hgt",
        num_heads=4,
        hgt_node_feat_dims=first.node_feat_dims,
        hgt_relation_specs=first.relation_specs,
    )
    model.eval()
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1))

    separate = model(
        first.node_features["block"],
        edge_index,
        edge_attr,
        hgt_node_features=first.node_features,
        hgt_edge_index=first.edge_index,
        hgt_edge_attr=first.edge_attr,
    )
    batched = model(
        batch.node_features["block"],
        edge_index,
        edge_attr,
        hgt_node_features=batch.node_features,
        hgt_edge_index=batch.edge_index,
        hgt_edge_attr=batch.edge_attr,
        block_batch=batch.node_batch["block"],
    )

    assert torch.allclose(batched["anchor"][:4], separate["anchor"], atol=1e-6)
    assert torch.allclose(batched["anchor"][4:], separate["anchor"], atol=1e-6)


def test_anchor_checkpoint_payload_records_hgt_config():
    inst = _hgt_sample_instance()
    graph_inputs = features.build_anchor_hgt_graph_inputs(inst)
    model = FloorplanGNN(
        node_feat_dim=graph_inputs.node_features["block"].shape[1],
        hidden_dim=32,
        num_layers=2,
        encoder_type="hgt",
        num_heads=4,
        hgt_node_feat_dims=graph_inputs.node_feat_dims,
        hgt_relation_specs=graph_inputs.relation_specs,
    )
    args = type("Args", (), {"example": "value"})()

    payload = anchor_checkpoint_payload(model, args, epoch=1, train_stats={}, val_stats={})

    assert payload["encoder_type"] == "hgt"
    assert payload["hgt_node_feat_dims"] == graph_inputs.node_feat_dims
    assert payload["hgt_relation_specs"] == list(graph_inputs.relation_specs)


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


def test_checkpoint_metric_prefers_no_runtime_over_val_loss():
    low_val_loss_bad_eval = CheckpointMetricRecord(
        checkpoint="bad_eval.pt",
        epoch=3,
        metric_source="tail_eval",
        feasible=100,
        val_loss=0.001,
        total_score_no_runtime=2.70,
        tail_weighted_no_runtime=2.70,
        soft_violations=10,
        avg_runtime=1.0,
    )
    higher_val_loss_good_eval = CheckpointMetricRecord(
        checkpoint="good_eval.pt",
        epoch=2,
        metric_source="tail_eval",
        feasible=100,
        val_loss=0.010,
        total_score_no_runtime=2.05,
        tail_weighted_no_runtime=2.05,
        soft_violations=4,
        avg_runtime=1.2,
    )

    assert better_checkpoint_metric(higher_val_loss_good_eval, low_val_loss_bad_eval)
    assert not better_checkpoint_metric(low_val_loss_bad_eval, higher_val_loss_good_eval)


def test_checkpoint_metric_manifest_round_trips(tmp_path):
    from floorset_arch.training.selection import append_metric_record, read_metric_records

    manifest = tmp_path / "checkpoint_metrics.jsonl"
    record = CheckpointMetricRecord(
        checkpoint="ckpt.pt",
        epoch=4,
        metric_source="full_eval",
        feasible=100,
        val_loss=0.004,
        total_score_no_runtime=2.01,
        tail_weighted_no_runtime=2.03,
        soft_violations=5,
        avg_runtime=1.5,
    )

    append_metric_record(manifest, record)

    assert read_metric_records(manifest) == [record]


def test_hgt_run_tag_records_batch_size():
    args = type(
        "Args",
        (),
        {
            "num_samples": 500000,
            "epochs": 3,
            "encoder": "hgt",
            "hidden_dim": 256,
            "layers": 4,
            "accumulation_steps": 32,
            "batch_size": 8,
            "num_heads": 4,
        },
    )()

    tag = build_run_tag(args)

    assert "_bs8_" in tag
    assert tag.endswith("_heads4")


def test_training_parser_accepts_hgt_encoder(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--encoder", "hgt", "--batch-size", "8"])

    args = train_module.parse_args()

    assert args.encoder == "hgt"
    assert args.batch_size == 8


def test_hgt_decoder_ranking_multipliers_target_high_risk_samples():
    constraints = torch.zeros(112, 5)
    constraints[:16, 4] = torch.tensor([1.0, 2.0, 4.0, 8.0] * 4)
    inst = parse_instance(
        112,
        torch.ones(112),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        None,
    )
    args = type(
        "Args",
        (),
        {
            "high_risk_order_multiplier": 1.7,
            "high_risk_pairwise_multiplier": 1.6,
            "high_risk_min_blocks": 110,
            "high_risk_min_constraints": 12,
        },
    )()

    order_multiplier, pairwise_multiplier = train_module.decoder_ranking_multipliers(
        inst, args, encoder_type="hgt"
    )
    mpnn_order, mpnn_pairwise = train_module.decoder_ranking_multipliers(
        inst, args, encoder_type="mpnn"
    )

    assert order_multiplier == 1.7
    assert pairwise_multiplier == 1.6
    assert (mpnn_order, mpnn_pairwise) == (1.0, 1.0)


def test_dirty_sample_builds_repaired_pseudo_target():
    block_count = 2
    area_targets = torch.tensor([4.0, 4.0])
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 8.0, 8.0],
        ]
    )
    inst = parse_instance(
        block_count,
        area_targets,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        fp_sol,
    )

    record = build_training_target_record(
        inst,
        fp_sol,
        PseudoTargetConfig(enabled=True, clean_enough_soft_violations=0),
    )

    assert record.source in {
        TrainingTargetSource.DIRTY_REPAIRED,
        TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH,
    }
    assert record.target_fp_sol.shape == (block_count, 4)
    assert record.original_soft_violations[1] > 0


def test_repaired_clean_enough_target_enables_dirty_order_weight():
    config = PseudoTargetConfig(
        enabled=True,
        dirty_pseudo_order_weight=0.25,
        dirty_pseudo_clean_enough_order_weight=0.35,
        clean_enough_soft_violations=0,
    )
    record = train_module._target_weight_policy(
        is_clean=False,
        target_source=TrainingTargetSource.DIRTY_REPAIRED_CLEAN_ENOUGH,
        config=config,
        existing_dirty_sample_weight=0.25,
    )

    assert record.sample_weight == 0.25
    assert record.order_weight_multiplier == 0.35
    assert record.pairwise_weight_multiplier == 0.35


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
