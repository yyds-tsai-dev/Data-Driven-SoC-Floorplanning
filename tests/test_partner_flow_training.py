import importlib
import sys
from pathlib import Path

import pytest
import torch

from flow_matching_claude import flow_path
import flow_train_claude as flow_train
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


def test_module_import_does_not_mutate_sys_path():
    before = sys.path.copy()
    sys.modules.pop("flow_train_claude", None)
    try:
        importlib.import_module("flow_train_claude")
        assert sys.path == before
    finally:
        sys.path[:] = before


def _v1_resume_environment(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "FloorSet"))
    monkeypatch.syspath_prepend(str(root / "FloorSet" / "iccad2026contest"))
    import direct_train_claude as trainer

    restored = False

    class TinyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_state_dict(self, state_dict, *args, **kwargs):
            nonlocal restored
            restored = True
            raise AssertionError("V1 restored checkpoint before flow validation")

    monkeypatch.setattr(trainer, "DirectDenoiser", TinyModel)
    monkeypatch.setattr(trainer, "parse_args", trainer.parse_args)
    monkeypatch.setattr(trainer, "train_step", trainer.train_step)
    return trainer, lambda: restored


@pytest.mark.parametrize(
    "args_payload",
    [{"training_method": "diffusion"}, {}],
    ids=["diffusion", "untagged"],
)
def test_flow_cli_rejects_invalid_resume_before_v1_state_restore(
    monkeypatch, tmp_path, args_payload
):
    _trainer, restored = _v1_resume_environment(monkeypatch)
    checkpoint = tmp_path / "resume.pt"
    torch.save({"args": args_payload, "model": {}}, checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flow_train_claude.py",
            "--device",
            "cpu",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--resume",
            str(checkpoint),
        ],
    )

    with pytest.raises(ValueError, match="flow_matching_v1"):
        flow_train.main()
    assert not restored()
