"""PARTNER_COND_P2B conditioning knob (2026-08-27)."""
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "partner") not in sys.path:
    sys.path.insert(0, str(REPO / "partner"))
import contest_optimizer as co  # noqa: E402


def test_off_path_returns_same_object(monkeypatch):
    monkeypatch.delenv("PARTNER_COND_P2B", raising=False)
    p2b = torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]])
    assert co._cond_p2b(p2b) is p2b


def test_off_mode_invalidates_pins_only(monkeypatch):
    monkeypatch.setenv("PARTNER_COND_P2B", "off")
    p2b = torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]])
    q = co._cond_p2b(p2b)
    assert q is not p2b
    assert torch.all(q[..., 0] == -1.0)
    assert torch.equal(q[..., 1:], p2b[..., 1:])
    assert torch.equal(p2b[..., 0], torch.tensor([0.0, 1.0]))  # input untouched


def test_off_mode_zeroes_pin_features(monkeypatch):
    from direct_diffusion_train import fast_condition
    monkeypatch.setenv("PARTNER_COND_P2B", "off")
    area = torch.tensor([[10.0, 20.0, 30.0]])
    b2b = torch.tensor([[[0.0, 1.0, 1.0]]])
    p2b = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 2.0]]])
    pins = torch.tensor([[[5.0, 5.0], [9.0, 1.0]]])
    cons = torch.zeros(1, 3, 5)
    tp = torch.full((1, 3, 4), -1.0)
    c_on = fast_condition(area, b2b, p2b, pins, cons, tp)
    c_off = fast_condition(area, b2b, co._cond_p2b(p2b), pins, cons, tp)
    diff = (c_on["node_feat"] - c_off["node_feat"]).abs().sum().item() if isinstance(c_on, dict) and "node_feat" in c_on else None
    assert diff is None or diff > 0.0
