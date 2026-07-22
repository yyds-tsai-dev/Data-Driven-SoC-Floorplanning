import pytest
import torch

from flow_matching_claude import flow_path
from flow_train_claude import checkpoint_method, masked_flow_loss


def test_zero_error_flow_velocity_has_zero_primary_loss():
    z0 = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]])
    noise = torch.zeros_like(z0)
    t = torch.tensor([0.5])
    z_t, target = flow_path(z0, noise, t)

    loss, endpoint = masked_flow_loss(
        target, target, z_t, z0, t, torch.tensor([[True]])
    )

    assert loss.item() == 0.0
    torch.testing.assert_close(endpoint, z0)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"args": {"training_method": "diffusion"}},
        {"args": {}},
        {},
    ],
)
def test_checkpoint_method_rejects_non_flow_checkpoint(checkpoint):
    with pytest.raises(ValueError, match="flow_matching_v1"):
        checkpoint_method(checkpoint)


def test_flow_primary_loss_backpropagates_and_checkpoint_is_tagged():
    predicted = torch.nn.Parameter(torch.zeros(1, 2, 4))
    z0 = torch.ones(1, 2, 4)
    noise = torch.zeros_like(z0)
    t = torch.tensor([0.5])
    z_t, target = flow_path(z0, noise, t)

    loss, _endpoint = masked_flow_loss(
        predicted, target, z_t, z0, t, torch.tensor([[True, False]])
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert checkpoint_method({"args": {"training_method": "flow_matching_v1"}}) == (
        "flow_matching_v1"
    )
