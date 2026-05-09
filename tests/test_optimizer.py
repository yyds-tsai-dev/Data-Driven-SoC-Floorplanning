from pathlib import Path

import torch

from floorset_arch.models import SolverConfig
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.optimizer import ArchitectureV2Optimizer
from floorset_arch.parser import parse_instance


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


def _write_anchor_checkpoint(path: Path, node_feat_dim: int = 18):
    model = FloorplanGNN(node_feat_dim=node_feat_dim, hidden_dim=8, num_layers=1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_feat_dim": node_feat_dim,
            "hidden_dim": 8,
            "layers": 1,
        },
        path,
    )


def test_checkpoint_relative_path_resolves_from_repo_root(tmp_path, monkeypatch):
    checkpoint = tmp_path / "repo-relative.pt"
    _write_anchor_checkpoint(checkpoint)
    problem = _tiny_problem()
    inst = parse_instance(**problem)

    monkeypatch.chdir("FloorSet/iccad2026contest")
    monkeypatch.setenv("FLOORSET_GNN_CHECKPOINT", str(checkpoint))
    optimizer = ArchitectureV2Optimizer()

    assert optimizer._try_anchor_guidance(inst) is not None


def test_checkpoint_model_is_cached_between_solves(tmp_path, monkeypatch):
    checkpoint = tmp_path / "cached.pt"
    _write_anchor_checkpoint(checkpoint)
    monkeypatch.setenv("FLOORSET_GNN_CHECKPOINT", str(checkpoint))

    import floorset_arch.training.checkpoint as checkpoint_module

    original_loader = checkpoint_module.load_checkpoint
    calls = 0

    def counted_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", counted_loader)
    optimizer = ArchitectureV2Optimizer(config=SolverConfig(max_candidates_per_block=8, beam_width=1))

    optimizer.solve(**_tiny_problem())
    optimizer.solve(**_tiny_problem())

    assert calls == 1


def test_default_checkpoint_loads_root_anchor_gnn(monkeypatch):
    monkeypatch.delenv("FLOORSET_GNN_CHECKPOINT", raising=False)
    checkpoint = Path("checkpoints/gnn_best.pt")
    if not checkpoint.exists():
        return
    problem = _tiny_problem()
    inst = parse_instance(**problem)
    optimizer = ArchitectureV2Optimizer()

    guidance = optimizer._try_anchor_guidance(inst)

    assert guidance is not None
    assert optimizer._checkpoint_kind == "anchor_gnn_v2"
    assert len(guidance.rect_priors) == problem["block_count"]
