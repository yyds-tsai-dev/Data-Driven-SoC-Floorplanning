import importlib.util
from pathlib import Path

import torch

from floorset_arch.diffusion.contracts import DiffusionPlacementPrior
from floorset_arch.optimizer import (
    ArchitectureV4Optimizer,
    ArchitectureV5Optimizer,
    ArchitectureV11Optimizer,
)


def _load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v11_optimizer_aliases_preserve_legacy_names():
    assert ArchitectureV5Optimizer is ArchitectureV11Optimizer
    assert ArchitectureV4Optimizer is ArchitectureV11Optimizer


def test_v11_wrapper_exports_contest_optimizer():
    module = _load_wrapper(Path("src/architecture_v11_optimizer.py"))
    assert module.MyOptimizer is ArchitectureV11Optimizer
    assert module.ContestOptimizer is ArchitectureV11Optimizer


def test_v5_wrapper_forwards_to_v11_optimizer():
    module = _load_wrapper(Path("src/architecture_v5_optimizer.py"))
    assert module.MyOptimizer is ArchitectureV11Optimizer
    assert module.ContestOptimizer is ArchitectureV11Optimizer


def test_active_scripts_reference_v11_wrapper():
    for script in ["scripts/validate.sh", "scripts/eval_single.sh", "scripts/eval_total.sh"]:
        text = Path(script).read_text(encoding="utf-8")
        assert "architecture_v11_optimizer.py" in text


def _tiny_problem():
    return {
        "block_count": 2,
        "area_targets": torch.tensor([4.0, 9.0]),
        "b2b_connectivity": torch.tensor([[0.0, 1.0, 1.0]]),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(2, 5),
        "target_positions": torch.full((2, 4), -1.0),
    }


def test_v11_falls_back_when_diffusion_checkpoint_is_missing(monkeypatch):
    # Legacy-path test: __init__ loads .env, which defaults FLOORSET_COLUMN_BACKBONE=1;
    # setenv wins because the dotenv load uses override=False.
    monkeypatch.setenv("FLOORSET_COLUMN_BACKBONE", "0")
    monkeypatch.delenv("FLOORSET_DIFFUSION_CHECKPOINT", raising=False)
    optimizer = ArchitectureV11Optimizer()

    positions = optimizer.solve(**_tiny_problem())

    assert len(positions) == 2
    assert optimizer.last_solve_metadata["fallback"] == "v5_no_diffusion_checkpoint"


def test_v11_uses_diffusion_prior_without_anchor_guidance(monkeypatch):
    # Legacy-path test: keep the column backbone off so solve() reaches the diffusion path.
    monkeypatch.setenv("FLOORSET_COLUMN_BACKBONE", "0")
    optimizer = ArchitectureV11Optimizer()

    def fake_prior(inst):
        return DiffusionPlacementPrior(
            centers=torch.tensor([[[1.0, 1.0], [5.0, 5.0]]]),
            log_aspect=torch.zeros(1, 2),
            pairwise_axis_logits=torch.zeros(1, 1, 3),
            pair_index=torch.tensor([[0, 1]]),
            variant="unit",
        )

    def fail_anchor(_inst):
        raise AssertionError("diffusion path must not request AnchorGuidance")

    monkeypatch.setattr(optimizer, "_try_diffusion_prior", fake_prior)
    monkeypatch.setattr(optimizer, "_try_anchor_guidance", fail_anchor)

    positions = optimizer.solve(**_tiny_problem())

    assert len(positions) == 2
    assert optimizer.last_solve_metadata["diffusion_variant"] == "unit"
    assert optimizer.last_solve_metadata["fallback"] == ""
