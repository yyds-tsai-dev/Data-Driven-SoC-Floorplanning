import torch

from floorset_arch import features
from floorset_arch.diffusion import DiffusionGraphInputs, build_diffusion_graph_inputs
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
    assert {"block", "pin", "cluster", "mib", "boundary"} <= set(graph_inputs.node_features)
    assert graph_inputs.raw_block_features.shape[0] == 4
    assert graph_inputs.area.shape == (4,)
    assert graph_inputs.fixed_mask.tolist() == [True, False, False, False]
    assert graph_inputs.preplaced_mask.tolist() == [False, True, False, False]
    assert graph_inputs.movable_mask.tolist() == [False, False, True, True]

    relations = set(graph_inputs.relation_specs)
    assert ("block", "connects", "block") in relations
    assert ("boundary", "has_member", "block") in relations
    for relation in features.ANCHOR_HGT_RELATION_SPECS:
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
