import pytest
import torch

from floorset_arch.diffusion import DiffusionPlacementPrior, PlacementTensorBatch


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
