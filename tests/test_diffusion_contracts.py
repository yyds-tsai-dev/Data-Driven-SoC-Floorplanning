import subprocess
import sys

import pytest
import torch

from floorset_arch.diffusion import (
    DiffusionGraphInputs,
    DiffusionPlacementPrior,
    PlacementTensorBatch,
)


def _graph_inputs(**overrides):
    values = {
        "node_features": {"block": torch.zeros(3, 2)},
        "edge_index": {},
        "edge_attr": {},
        "relation_specs": (),
        "raw_block_features": torch.zeros(3, 2),
        "raw_pair_features": torch.zeros(2, 1),
        "pair_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "area": torch.ones(3),
        "scale": 1.0,
        "fixed_mask": torch.tensor([True, False, False]),
        "preplaced_mask": torch.tensor([False, True, False]),
        "movable_mask": torch.tensor([False, False, True]),
        "boundary_codes": torch.tensor([1, 0, 2], dtype=torch.long),
        "cluster_ids": torch.tensor([7, 7, 0], dtype=torch.long),
        "mib_ids": torch.tensor([1, 1, 0], dtype=torch.long),
        "global_features": torch.zeros(2),
    }
    values.update(overrides)
    return DiffusionGraphInputs(**values)


def _typed_graph_inputs(**overrides):
    relation = ("block", "connects", "block")
    values = {
        "node_features": {"block": torch.zeros(3, 2)},
        "edge_index": {relation: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)},
        "edge_attr": {relation: torch.ones(2, 1)},
        "relation_specs": (relation,),
    }
    values.update(overrides)
    return _graph_inputs(**values)


def test_diffusion_public_import_keeps_graph_builder_lazy():
    code = """
import sys
import floorset_arch.diffusion as diffusion
assert "floorset_arch.diffusion.graph_inputs" not in sys.modules
assert callable(diffusion.build_diffusion_graph_inputs)
assert "floorset_arch.diffusion.graph_inputs" in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_diffusion_prior_validates_center_and_aspect_shapes():
    prior = DiffusionPlacementPrior(
        centers=torch.zeros(3, 4, 2),
        log_aspect=torch.zeros(3, 4),
        pairwise_axis_logits=torch.zeros(3, 2, 2),
        pair_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        quality_pred=torch.zeros(3),
    )

    assert prior.sample_count == 3
    assert prior.block_count == 4
    assert prior.pair_count == 2

    with pytest.raises(ValueError, match="centers"):
        DiffusionPlacementPrior(
            centers=torch.zeros(3, 4),
            log_aspect=torch.zeros(3, 4),
            pairwise_axis_logits=torch.zeros(3, 2, 2),
            pair_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        )

    with pytest.raises(ValueError, match="log_aspect"):
        DiffusionPlacementPrior(
            centers=torch.zeros(3, 4, 2),
            log_aspect=torch.zeros(3, 5),
            pairwise_axis_logits=torch.zeros(3, 2, 2),
            pair_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        )


def test_diffusion_prior_rejects_mismatched_pair_logits():
    with pytest.raises(ValueError, match="pairwise_axis_logits"):
        DiffusionPlacementPrior(
            centers=torch.zeros(2, 4, 2),
            log_aspect=torch.zeros(2, 4),
            pairwise_axis_logits=torch.zeros(2, 3, 2),
            pair_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        )


def test_diffusion_prior_rejects_invalid_pair_index_semantics():
    with pytest.raises(ValueError, match="pair_index must be torch.long"):
        DiffusionPlacementPrior(
            centers=torch.zeros(2, 4, 2),
            log_aspect=torch.zeros(2, 4),
            pairwise_axis_logits=torch.zeros(2, 1, 2),
            pair_index=torch.tensor([[0.0, 1.0]]),
        )

    with pytest.raises(ValueError, match="pair_index values must reference valid blocks"):
        DiffusionPlacementPrior(
            centers=torch.zeros(2, 4, 2),
            log_aspect=torch.zeros(2, 4),
            pairwise_axis_logits=torch.zeros(2, 1, 2),
            pair_index=torch.tensor([[0, 4]], dtype=torch.long),
        )


def test_diffusion_prior_rejects_mixed_devices():
    with pytest.raises(ValueError, match="all tensors must be on the same device"):
        DiffusionPlacementPrior(
            centers=torch.zeros(2, 4, 2),
            log_aspect=torch.zeros(2, 4),
            pairwise_axis_logits=torch.zeros(2, 1, 2),
            pair_index=torch.tensor([[0, 1]], dtype=torch.long),
            quality_pred=torch.zeros(2, device="meta"),
        )


def test_diffusion_graph_inputs_validate_masks_ids_pairs_and_devices():
    with pytest.raises(ValueError, match="pair_index must be torch.long"):
        _graph_inputs(pair_index=torch.tensor([[0.0, 1.0]]))

    with pytest.raises(ValueError, match="pair_index values must reference valid blocks"):
        _graph_inputs(pair_index=torch.tensor([[0, 3]], dtype=torch.long), raw_pair_features=torch.zeros(1, 1))

    with pytest.raises(ValueError, match="fixed_mask must be torch.bool"):
        _graph_inputs(fixed_mask=torch.tensor([1, 0, 0], dtype=torch.long))

    with pytest.raises(ValueError, match="cluster_ids must be torch.long"):
        _graph_inputs(cluster_ids=torch.tensor([7.0, 7.0, 0.0]))

    with pytest.raises(ValueError, match="all tensors must be on the same device"):
        _graph_inputs(global_features=torch.zeros(2, device="meta"))


def test_diffusion_graph_inputs_validate_relation_tensor_presence_and_shapes():
    relation = ("block", "connects", "block")

    with pytest.raises(ValueError, match="edge_index"):
        _typed_graph_inputs(edge_index={}, relation_specs=(relation,))

    with pytest.raises(ValueError, match="edge_attr"):
        _typed_graph_inputs(edge_attr={}, relation_specs=(relation,))

    with pytest.raises(ValueError, match="edge_index"):
        _typed_graph_inputs(edge_index={relation: torch.zeros(3, 2, dtype=torch.long)})

    with pytest.raises(ValueError, match="edge_attr"):
        _typed_graph_inputs(edge_attr={relation: torch.ones(1, 1)})


def test_diffusion_graph_inputs_validate_relation_dtypes_and_index_bounds():
    relation = ("block", "connects", "block")

    with pytest.raises(ValueError, match="edge_index"):
        _typed_graph_inputs(edge_index={relation: torch.tensor([[0.0], [1.0]])})

    with pytest.raises(ValueError, match="node_features.*float"):
        _typed_graph_inputs(node_features={"block": torch.ones((3, 2), dtype=torch.long)})

    with pytest.raises(ValueError, match="edge_attr.*float"):
        _typed_graph_inputs(edge_attr={relation: torch.ones(2, 1, dtype=torch.long)})

    with pytest.raises(ValueError, match="edge_index values must reference valid nodes"):
        _typed_graph_inputs(
            edge_index={relation: torch.tensor([[0], [3]], dtype=torch.long)},
            edge_attr={relation: torch.ones(1, 1)},
        )


def test_placement_tensor_batch_exposes_topk_slice():
    batch = PlacementTensorBatch(
        rect_xywh=torch.arange(5 * 4 * 4, dtype=torch.float32).reshape(5, 4, 4),
        pairwise_axis_logits=torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2),
        pair_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        score_features=torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3),
        source="unit-test",
    )

    selected = batch.select(torch.tensor([3, 1], dtype=torch.long))

    assert selected.rect_xywh.shape == (2, 4, 4)
    assert torch.equal(selected.rect_xywh, batch.rect_xywh[[3, 1]])
    assert torch.equal(selected.pairwise_axis_logits, batch.pairwise_axis_logits[[3, 1]])
    assert torch.equal(selected.pair_index, batch.pair_index)
    assert torch.equal(selected.score_features, batch.score_features[[3, 1]])
    assert selected.source == "unit-test"


def test_placement_tensor_batch_validates_pair_index_and_device():
    with pytest.raises(ValueError, match="pair_index must be torch.long"):
        PlacementTensorBatch(
            rect_xywh=torch.zeros(2, 3, 4),
            pairwise_axis_logits=torch.zeros(2, 1, 2),
            pair_index=torch.tensor([[0.0, 1.0]]),
        )

    with pytest.raises(ValueError, match="pair_index values must reference valid blocks"):
        PlacementTensorBatch(
            rect_xywh=torch.zeros(2, 3, 4),
            pairwise_axis_logits=torch.zeros(2, 1, 2),
            pair_index=torch.tensor([[0, 3]], dtype=torch.long),
        )

    with pytest.raises(ValueError, match="all tensors must be on the same device"):
        PlacementTensorBatch(
            rect_xywh=torch.zeros(2, 3, 4),
            pairwise_axis_logits=torch.zeros(2, 1, 2),
            pair_index=torch.tensor([[0, 1]], dtype=torch.long),
            score_features=torch.zeros(2, 1, device="meta"),
        )
