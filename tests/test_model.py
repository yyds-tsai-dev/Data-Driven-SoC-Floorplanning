import torch

from floorset_arch.features import build_model_inputs
from floorset_arch.nn.model import SimpleGraphFloorplanner
from floorset_arch.nn.postprocess import predictions_to_positions
from floorset_arch.parser import parse_instance
from floorset_arch.training.losses import compute_v1_loss


def test_model_forward_outputs_positions_and_loss_backpropagates():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 9.0, 16.0]),
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 1.0]]),
        torch.tensor([[0.0, 2.0, 1.5]]),
        torch.tensor([[10.0, 20.0]]),
        torch.zeros(3, 5),
        None,
    )
    inputs = build_model_inputs(inst)
    model = SimpleGraphFloorplanner(input_dim=inputs.block_features.shape[1], hidden_dim=16, layers=2)

    pred = model(inputs)
    positions = predictions_to_positions(inst, pred)
    target = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [2.0, 0.0, 3.0, 3.0],
            [0.0, 3.0, 4.0, 4.0],
        ]
    )
    metrics = torch.tensor([29.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.5])

    loss, parts = compute_v1_loss(positions, target, inst, metrics, return_parts=True)
    loss.backward()

    assert positions.shape == (3, 4)
    assert parts["supervised"].item() >= 0.0
    assert any(param.grad is not None for param in model.parameters())

