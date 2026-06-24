import pytest
import torch

from floorset_arch.features import ANCHOR_HGT_RELATION_SPECS
from floorset_arch.diffusion import DiffusionGraphInputs
from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
from floorset_arch.parser import parse_instance


def _sample_instance():
    return parse_instance(
        4,
        torch.tensor([4.0, 9.0, 16.0, 25.0]),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 3.0, 3.0]]),
        torch.tensor([[10.0, 20.0], [30.0, 5.0]]),
        torch.tensor(
            [
                [1.0, 0.0, 1.0, 7.0, 1.0],
                [0.0, 1.0, 1.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 7.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 4.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 3.0, 3.0],
                [-1.0, -1.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0],
            ]
        ),
    )


def test_diffusion_graph_inputs_keep_raw_side_channels_and_typed_relations():
    graph_inputs = build_diffusion_graph_inputs(_sample_instance())

    assert isinstance(graph_inputs, DiffusionGraphInputs)
    assert graph_inputs.block_count == 4
    assert graph_inputs.pair_count == graph_inputs.pair_index.shape[0]
    assert {"block", "pin", "cluster", "mib", "boundary"} <= set(graph_inputs.node_features)
    assert graph_inputs.node_features["pin"].shape[1] == 3
    assert graph_inputs.node_features["cluster"].shape[1] == 4
    assert graph_inputs.node_features["mib"].shape[1] == 4
    assert graph_inputs.node_features["boundary"].shape[1] == 5
    assert graph_inputs.raw_block_features.shape[0] == 4
    assert graph_inputs.area.shape == (4,)
    assert graph_inputs.fixed_mask.tolist() == [True, False, False, False]
    assert graph_inputs.preplaced_mask.tolist() == [False, True, False, False]
    assert graph_inputs.movable_mask.tolist() == [False, False, True, True]

    relations = set(graph_inputs.relation_specs)
    assert ("block", "connects", "block") in relations
    assert ("boundary", "has_member", "block") in relations
    for relation in ANCHOR_HGT_RELATION_SPECS:
        assert relation in graph_inputs.edge_index
        assert relation in graph_inputs.edge_attr
        assert graph_inputs.edge_index[relation].shape[0] == 2
        assert graph_inputs.edge_attr[relation].shape == (
            graph_inputs.edge_index[relation].shape[1],
            1,
        )

    assert graph_inputs.pair_index.dim() == 2
    assert graph_inputs.pair_index.shape[1] == 2
    assert graph_inputs.raw_pair_features.shape[0] == graph_inputs.pair_index.shape[0]
    assert graph_inputs.raw_pair_features.shape[1] == 6
    assert torch.equal(graph_inputs.cluster_ids, torch.tensor([7, 7, 7, 0]))
    assert torch.equal(graph_inputs.mib_ids, torch.tensor([1, 1, 0, 0]))
    assert torch.equal(graph_inputs.boundary_codes, torch.tensor([1, 0, 2, 4]))


def test_parse_instance_stays_model_agnostic():
    inst = _sample_instance()

    assert not hasattr(inst, "diffusion_graph_inputs")
    graph_inputs = build_diffusion_graph_inputs(inst)

    assert isinstance(graph_inputs, DiffusionGraphInputs)
    assert graph_inputs.global_features.shape[0] > 0


def test_diffusion_graph_inputs_keep_factor_feature_widths_stable_across_samples():
    left = build_diffusion_graph_inputs(_sample_instance())
    right = build_diffusion_graph_inputs(
        parse_instance(
            4,
            torch.tensor([4.0, 9.0, 16.0, 25.0]),
            torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
            torch.tensor([[0.0, 0.0, 2.0], [1.0, 3.0, 3.0]]),
            torch.tensor([[10.0, 20.0], [30.0, 5.0]]),
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 4.0],
                ]
            ),
            None,
        )
    )

    assert {
        node_type: features.shape[1]
        for node_type, features in left.node_features.items()
    } == {
        node_type: features.shape[1]
        for node_type, features in right.node_features.items()
    }


def test_diffusion_graph_inputs_reject_empty_instances_before_feature_building():
    inst = parse_instance(
        0,
        torch.empty(0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.empty(0, 5),
        None,
    )

    with pytest.raises(ValueError, match="block_count|positive"):
        build_diffusion_graph_inputs(inst)
