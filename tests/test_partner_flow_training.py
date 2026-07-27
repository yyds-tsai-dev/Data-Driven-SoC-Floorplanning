import importlib
import sys
from pathlib import Path

import pytest
import torch

from flow_matching_claude import (
    endpoint_snr_weight,
    endpoint_time_weight,
    flow_path,
    sample_flow_t,
)
import flow_train_claude as flow_train
from flow_train_claude import (
    checkpoint_method,
    masked_flow_loss,
    mib_objective,
    parse_flow_extras,
)


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


@pytest.mark.parametrize(
    "method",
    [
        "flow_matching_v1",
        "flow_matching_v2",
        "flow_matching_v2_1",
        "flow_matching_v3",
    ],
)
def test_checkpoint_method_accepts_every_flow_generation(method):
    # v3 is current; older tags stay loadable for the candidate probe.
    assert checkpoint_method({"args": {"training_method": method}}) == method


# --- v3: golden-referenced MIB aspect hinge ---------------------------------


def test_mib_hinge_is_zero_when_prediction_is_more_symmetric_than_golden():
    # Trained models sit at ~0.42-0.51 against a golden ~0.59-0.67, so this is
    # the common case: the unhinged term was charging for symmetry the ground
    # truth does not have.
    mib_pred = torch.tensor([0.42, 0.10, 0.00])
    mib_gt = torch.tensor([0.59, 0.67, 0.00])

    hinged = mib_objective(mib_pred, mib_gt, hinge=True)

    assert torch.equal(hinged, torch.zeros(3))


def test_mib_hinge_charges_only_the_excess_over_golden():
    mib_pred = torch.tensor([0.80, 0.60])
    mib_gt = torch.tensor([0.50, 0.65])

    hinged = mib_objective(mib_pred, mib_gt, hinge=True)

    torch.testing.assert_close(hinged, torch.tensor([0.30, 0.00]))
    assert float(hinged[0]) > 0.0


def test_mib_hinge_scores_the_golden_layout_itself_at_zero():
    mib_gt = torch.tensor([0.589, 0.0, 1.25])

    assert torch.equal(mib_objective(mib_gt, mib_gt, hinge=True), torch.zeros(3))


def test_mib_hinge_off_is_bit_exact_with_the_v1_objective():
    mib_pred = torch.tensor([0.42, 0.80, 0.00])
    mib_gt = torch.tensor([0.59, 0.50, 0.00])

    # off must return the prediction untouched, ignoring golden entirely
    assert mib_objective(mib_pred, mib_gt, hinge=False) is mib_pred
    assert mib_objective(mib_pred, None, hinge=False) is mib_pred


def test_mib_hinge_defaults_off_so_the_objective_stays_v1():
    known, _rest = parse_flow_extras([])
    assert known.mib_hinge is False
    assert parse_flow_extras(["--mib-hinge"])[0].mib_hinge is True


def test_mib_hinge_gradient_flows_only_through_the_excess():
    mib_pred = torch.tensor([0.80, 0.42], requires_grad=True)
    mib_gt = torch.tensor([0.50, 0.59])

    mib_objective(mib_pred, mib_gt, hinge=True).sum().backward()

    # worse-than-golden sample carries gradient, better-than-golden does not
    torch.testing.assert_close(mib_pred.grad, torch.tensor([1.0, 0.0]))


def test_mib_aspect_is_d4_invariant_so_golden_reference_survives_augmentation():
    """The hinge reads mib_gt off the *augmented* golden rects, so the metric
    must be D4-invariant or the reference would move with the augmentation."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "FloorSet" / "iccad2026contest"))
    sys.path.insert(0, str(root / "FloorSet"))
    import direct_train_claude as trainer

    # two blocks in MIB group 1, one in group 2, one ungrouped
    w = torch.tensor([[4.0, 2.0, 3.0, 5.0]])
    h = torch.tensor([[1.0, 3.0, 6.0, 5.0]])
    rects = torch.stack([torch.zeros_like(w), torch.zeros_like(w), w, h], dim=-1)
    cons = torch.zeros(1, 4, 5)
    cons[..., 2] = torch.tensor([[1.0, 1.0, 2.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    base = trainer.mib_aspect(rects, cons, mask)
    # transpose: w <-> h  (negates every log-aspect)
    transposed = torch.stack([torch.zeros_like(w), torch.zeros_like(w), h, w], dim=-1)
    # mirror: w/h untouched, only positions move
    mirrored = torch.stack([w, h, w, h], dim=-1)

    torch.testing.assert_close(trainer.mib_aspect(transposed, cons, mask), base)
    torch.testing.assert_close(trainer.mib_aspect(mirrored, cons, mask), base)
    assert float(base) > 0.0


def test_v2_1_defaults_reproduce_the_validated_v1_objective():
    """v2's two loss-recipe changes are OFF by default after the 0727 diagnosis.

    SNR endpoint weighting was measured net-negative and terminal-t coverage
    unproven, so the default objective is exactly v1's; both stay reachable by
    flag for the ablation arms.
    """
    known, rest = parse_flow_extras([])
    assert known.x0_time_weighting == "none"
    assert known.term_t_prob == 0.0
    assert rest == []

    arm, _ = parse_flow_extras(["--x0-time-weighting", "snr", "--term-t-prob", "0.1"])
    assert arm.x0_time_weighting == "snr" and arm.term_t_prob == 0.1


def test_parse_flow_extras_passes_through_trainer_flags():
    known, rest = parse_flow_extras(["--term-band", "0.05", "--batch-size", "12"])
    assert known.term_band == 0.05
    assert rest == ["--batch-size", "12"]


def test_endpoint_time_weight_none_is_identity_and_snr_matches_helper():
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95])

    none = endpoint_time_weight(t, "none", 5.0)
    assert torch.equal(none, torch.ones(5))

    torch.testing.assert_close(
        endpoint_time_weight(t, "snr", 5.0), endpoint_snr_weight(t, 5.0)
    )
    with pytest.raises(ValueError, match="none.*snr"):
        endpoint_time_weight(t, "cosine", 5.0)


def test_sample_flow_t_defaults_to_uniform_time():
    # The v2.1 default must not put a spike at the data endpoint.
    gen = torch.Generator().manual_seed(3)
    t = sample_flow_t(20000, torch.device("cpu"), gen)
    assert float((t >= 0.98).float().mean()) < 0.035


def test_uniform_t_consumes_the_generator_exactly_like_v1():
    """v3 is only a clean single-variable control against v1 if the default
    t-sampling draws the identical RNG stream v1's bare torch.rand did --
    otherwise every downstream noise draw shifts and the runs diverge for a
    reason that has nothing to do with the mib hinge."""
    a = torch.Generator().manual_seed(7)
    b = torch.Generator().manual_seed(7)

    ours = sample_flow_t(12, torch.device("cpu"), a)
    v1_style = torch.rand((12,), device=torch.device("cpu"), generator=b)

    assert torch.equal(ours, v1_style)
    # and the generators must be left in the same state for the noise draw
    assert torch.equal(torch.randn(5, generator=a), torch.randn(5, generator=b))


def test_endpoint_snr_weight_downweights_low_t_and_clamps():
    # Low-t (high-noise) endpoints get a small weight; the weight rises with t
    # and saturates at gamma once t**2/(1-t)**2 exceeds it.
    gamma = 5.0
    t = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999])
    w = endpoint_snr_weight(t, gamma)

    assert w.shape == t.shape
    assert float(w[0]) == 0.0
    assert (w[1:] >= w[:-1]).all()          # monotone non-decreasing in t
    assert float(w[3]) == pytest.approx(1.0, abs=1e-5)  # t=0.5 -> SNR 1
    assert w.max() <= gamma + 1e-6
    assert float(w[-1]) == pytest.approx(gamma, abs=1e-5)  # clamped near t=1


def test_sample_flow_t_covers_terminal_band_and_stays_in_unit_interval():
    gen = torch.Generator().manual_seed(0)
    t = sample_flow_t(20000, torch.device("cpu"), gen, term_prob=0.10, term_band=0.02)

    assert t.shape == (20000,)
    assert float(t.min()) >= 0.0 and float(t.max()) < 1.0
    frac_terminal = float((t >= 0.98).float().mean())
    assert 0.07 <= frac_terminal <= 0.13   # ~10% land in [1-band, 1)

    uniform = sample_flow_t(5000, torch.device("cpu"), gen, term_prob=0.0, term_band=0.02)
    assert float((uniform >= 0.98).float().mean()) < 0.05  # no terminal spike


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
    monkeypatch.setattr(
        trainer, "get_training_dataloader", trainer.get_training_dataloader
    )
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


def test_flow_cli_num_samples_supports_official_train_loader_signature(
    monkeypatch, tmp_path
):
    trainer, _restored = _v1_resume_environment(monkeypatch)
    calls = {}

    def official_train_loader(data_path, batch_size, num_samples, shuffle):
        calls.update(
            data_path=data_path,
            batch_size=batch_size,
            num_samples=num_samples,
            shuffle=shuffle,
        )
        return []

    monkeypatch.setattr(trainer, "get_training_dataloader", official_train_loader)
    monkeypatch.setattr(trainer.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flow_train_claude.py",
            "--device",
            "cpu",
            "--fresh",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--num-samples",
            "256",
            "--num-workers",
            "0",
            "--max-steps",
            "0",
        ],
    )

    flow_train.main()
    assert calls == {
        "data_path": "../",
        "batch_size": 16,
        "num_samples": 256,
        "shuffle": False,
    }
