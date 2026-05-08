import os

import torch

from floorset_arch.models import SolverConfig
from floorset_arch.nn.model import SimpleGraphFloorplanner
from floorset_arch.optimizer import ArchitectureV1Optimizer
from floorset_arch.parser import parse_instance
from floorset_arch.training.checkpoint import save_checkpoint


def _tiny_problem():
    return {
        "block_count": 2,
        "area_targets": torch.tensor([4.0, 4.0]),
        "b2b_connectivity": torch.tensor([[0.0, 1.0, 1.0]]),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(2, 5),
        "target_positions": torch.full((2, 4), -1.0),
    }


def test_checkpoint_relative_path_resolves_from_repo_root(tmp_path, monkeypatch):
    problem = _tiny_problem()
    inst = parse_instance(**problem)
    input_dim = 13
    model = SimpleGraphFloorplanner(input_dim=input_dim, hidden_dim=4, layers=1)
    checkpoint = tmp_path / "repo-relative.pt"
    save_checkpoint(checkpoint, model, {"hidden_dim": 4, "layers": 1, "input_dim": input_dim})

    monkeypatch.chdir("FloorSet/iccad2026contest")
    monkeypatch.setenv("FLOORSET_V1_CHECKPOINT", os.path.relpath(checkpoint, start="FloorSet/iccad2026contest"))

    optimizer = ArchitectureV1Optimizer()

    assert optimizer._try_model_hints(inst) is not None


def test_checkpoint_model_is_cached_between_solves(tmp_path, monkeypatch):
    problem = _tiny_problem()
    input_dim = 13
    model = SimpleGraphFloorplanner(input_dim=input_dim, hidden_dim=4, layers=1)
    checkpoint = tmp_path / "cached.pt"
    save_checkpoint(checkpoint, model, {"hidden_dim": 4, "layers": 1, "input_dim": input_dim})
    monkeypatch.setenv("FLOORSET_V1_CHECKPOINT", str(checkpoint))

    import floorset_arch.training.checkpoint as checkpoint_module

    original_loader = checkpoint_module.load_checkpoint
    calls = 0

    def counted_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", counted_loader)
    optimizer = ArchitectureV1Optimizer(config=SolverConfig(max_candidates_per_block=8))

    optimizer.solve(**problem)
    optimizer.solve(**problem)

    assert calls == 1
