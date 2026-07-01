from __future__ import annotations

import pytest
import torch


class _TinyEvalProbeDataset:
    def __init__(self, _root):
        self.samples = [
            {
                "input": (
                    torch.tensor([4.0, 9.0]),
                    torch.tensor([[0.0, 1.0, 2.0]]),
                    torch.empty(0, 3),
                    torch.empty(0, 2),
                    torch.zeros(2, 5),
                ),
                "label": (
                    torch.empty(0, 3),
                    torch.tensor(
                        [
                            [2.0, 2.0, 0.0, 0.0],
                            [3.0, 3.0, 3.0, 0.0],
                        ]
                    ),
                    torch.tensor([13.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 6.0]),
                ),
            }
            for _ in range(3)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[int(index)]


def test_parse_args_defaults_to_lite_dataset_mode():
    from floorset_arch.training.train_diffusion import parse_args

    args = parse_args([])

    assert args.dataset_mode == "lite"


def test_eval_probe_rejects_nonzero_tree_weight(capsys):
    from floorset_arch.training.train_diffusion import parse_args

    with pytest.raises(SystemExit, match="2"):
        parse_args(["--dataset-mode", "eval-probe", "--tree-weight", "0.25"])

    captured = capsys.readouterr()
    assert "eval-probe requires --tree-weight 0" in captured.err


def test_eval_probe_rejects_train_time_evaluator(capsys):
    from floorset_arch.training.train_diffusion import parse_args

    with pytest.raises(SystemExit, match="2"):
        parse_args(
            [
                "--dataset-mode",
                "eval-probe",
                "--tree-weight",
                "0",
                "--train-evaluate-each-epoch",
            ]
        )

    captured = capsys.readouterr()
    assert "eval-probe uses final-only evaluator" in captured.err


def test_make_loaders_uses_eval_probe_dataset_without_validation(monkeypatch):
    import floorset_arch.training.train_diffusion as train_diffusion

    monkeypatch.setattr(train_diffusion, "EvalProbeDataset", _TinyEvalProbeDataset)
    args = train_diffusion.parse_args(
        [
            "--dataset-mode",
            "eval-probe",
            "--tree-weight",
            "0",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
        ]
    )

    train_loader, val_loader, ts, te, vs, ve, total = train_diffusion._make_loaders(args)

    assert val_loader is None
    assert (ts, te, vs, ve) == (0, 2, -1, -1)
    assert total == 3
    first_batch = next(iter(train_loader))
    tree_sol = first_batch[5]
    fp_sol = first_batch[6]
    assert tree_sol.shape[-1] == 3
    assert fp_sol.shape[-1] == 4
