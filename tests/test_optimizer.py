from pathlib import Path

import torch

from floorset_arch.models import Placement, Rect
from floorset_arch.models import AnchorGuidance, SolverConfig
from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.optimizer import ArchitectureV4Optimizer, CandidateSpec
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
    optimizer = ArchitectureV4Optimizer()

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
    optimizer = ArchitectureV4Optimizer(config=SolverConfig(max_candidates_per_block=8, beam_width=1))

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
    optimizer = ArchitectureV4Optimizer()

    guidance = optimizer._try_anchor_guidance(inst)

    assert guidance is not None
    assert optimizer._checkpoint_kind == "anchor_gnn_v2"
    assert len(guidance.rect_priors) == problem["block_count"]


def test_optimizer_loads_repo_dotenv_for_checkpoint_env(tmp_path, monkeypatch):
    checkpoint = tmp_path / "from-dotenv.pt"
    _write_anchor_checkpoint(checkpoint)
    env_file = Path(".env")
    original = env_file.read_text() if env_file.exists() else None
    try:
        env_file.write_text(f"FLOORSET_GNN_CHECKPOINT={checkpoint}\n", encoding="utf-8")
        monkeypatch.delenv("FLOORSET_GNN_CHECKPOINT", raising=False)

        optimizer = ArchitectureV4Optimizer()
        inst = parse_instance(**_tiny_problem())

        assert optimizer._try_anchor_guidance(inst) is not None
        assert optimizer._checkpoint_key == checkpoint.resolve()
    finally:
        if original is None:
            env_file.unlink(missing_ok=True)
        else:
            env_file.write_text(original, encoding="utf-8")


def test_runtime_calibration_env_does_not_sleep(monkeypatch):
    block_count = 21
    problem = {
        "block_count": block_count,
        "area_targets": torch.full((block_count,), 4.0),
        "b2b_connectivity": torch.empty(0, 3),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(block_count, 5),
        "target_positions": torch.full((block_count, 4), -1.0),
    }

    def fail_sleep(_seconds):
        raise AssertionError("runtime calibration must not sleep")

    monkeypatch.setattr("time.sleep", fail_sleep)
    monkeypatch.setenv("FLOORSET_RUNTIME_CALIBRATION_SECONDS", "10")

    optimizer = ArchitectureV4Optimizer()

    assert len(optimizer.solve(**problem)) == block_count


def test_anchor_guidance_can_store_pairwise_logits():
    guidance = AnchorGuidance()

    guidance.pairwise_axis[(0, 1)] = (2.0, -1.0)

    assert guidance.pairwise_axis[(0, 1)] == (2.0, -1.0)


def test_large_case_candidate_specs_include_relative_order_profiles(monkeypatch):
    monkeypatch.setenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", "1")
    monkeypatch.delenv("FLOORSET_INCLUDE_BEAM_CANDIDATES", raising=False)
    monkeypatch.delenv("FLOORSET_INCLUDE_NO_GUIDANCE_CANDIDATE", raising=False)
    block_count = 118
    problem = {
        "block_count": block_count,
        "area_targets": torch.full((block_count,), 4.0),
        "b2b_connectivity": torch.empty(0, 3),
        "p2b_connectivity": torch.empty(0, 3),
        "pins_pos": torch.empty(0, 2),
        "constraints": torch.zeros(block_count, 5),
        "target_positions": torch.full((block_count, 4), -1.0),
    }
    inst = parse_instance(**problem)
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    names = {(spec.name, spec.profile, spec.repair_profile) for spec in specs}
    assert ("adaptive_relative_order", optimizer._adaptive_profile_order(inst)[0], "normal") in names
    assert {spec.profile for spec in specs} == {"soft", "compact"}
    assert {spec.repair_profile for spec in specs} == {"normal"}
    assert all(spec.kind == "relative_order" for spec in specs)


def test_high_risk_case_gets_candidate_portfolio_when_enabled(monkeypatch):
    monkeypatch.setenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", "1")
    monkeypatch.setenv("FLOORSET_HIGH_RISK_REPAIR_PROFILES", "normal")
    monkeypatch.delenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", raising=False)
    block_count = 112
    constraints = torch.zeros(block_count, 5)
    constraints[:18, 4] = torch.tensor([1.0, 2.0, 4.0, 8.0] * 4 + [1.0, 2.0])
    constraints[18:30, 3] = 1.0
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert optimizer._is_high_risk_case(inst)
    assert len(specs) > 1
    assert {"soft", "compact"}.issubset({spec.profile for spec in specs})
    assert {spec.repair_profile for spec in specs} == {"normal"}


def test_high_risk_repair_profiles_are_configurable(monkeypatch):
    monkeypatch.setenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", "1")
    monkeypatch.setenv("FLOORSET_HIGH_RISK_REPAIR_PROFILES", "normal,boundary_first,grouping_first,quality_refine")
    block_count = 112
    constraints = torch.zeros(block_count, 5)
    constraints[:18, 4] = torch.tensor([1.0, 2.0, 4.0, 8.0] * 4 + [1.0, 2.0])
    constraints[18:30, 3] = 1.0
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert {"normal", "boundary_first", "grouping_first", "quality_refine"}.issubset(
        {spec.repair_profile for spec in specs}
    )


def test_auto_high_risk_portfolio_targets_extreme_tail_cases(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", raising=False)
    monkeypatch.delenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", raising=False)
    block_count = 120
    b2b = torch.tensor([[float(i % block_count), float((i + 1) % block_count), 1.0] for i in range(9700)])
    p2b = torch.tensor([[float(i % 100), float(i % block_count), 1.0] for i in range(10)])
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        b2b,
        p2b,
        torch.zeros(100, 2),
        torch.zeros(block_count, 5),
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert optimizer._is_targeted_high_risk_case(inst)
    assert len(specs) > 1
    assert {"normal", "boundary_first", "grouping_first", "quality_refine"}.issubset(
        {spec.repair_profile for spec in specs}
    )


def test_auto_high_risk_portfolio_skips_broad_risk_only_cases(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", raising=False)
    monkeypatch.delenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", raising=False)
    block_count = 119
    constraints = torch.zeros(block_count, 5)
    constraints[:31, 4] = torch.tensor(([1.0, 2.0, 4.0, 8.0] * 8)[:31])
    constraints[31:52, 3] = 1.0
    b2b = torch.tensor([[float(i % block_count), float((i + 1) % block_count), 1.0] for i in range(1699)])
    p2b = torch.tensor([[float(i % 128), float(i % block_count), 1.0] for i in range(908)])
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        b2b,
        p2b,
        torch.zeros(128, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert optimizer._is_high_risk_case(inst)
    assert not optimizer._is_targeted_high_risk_case(inst)
    assert len(specs) == 1
    assert specs[0].repair_profile == "normal"


def test_low_risk_large_case_keeps_single_default_candidate(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", raising=False)
    monkeypatch.delenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", raising=False)
    block_count = 120
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(block_count, 5),
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert not optimizer._is_high_risk_case(inst)
    assert len(specs) == 1
    assert specs[0].repair_profile == "normal"


def test_dense_medium_case_is_high_risk(monkeypatch):
    monkeypatch.setenv("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", "1")
    monkeypatch.delenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", raising=False)
    block_count = 91
    constraints = torch.zeros(block_count, 5)
    constraints[:27, 4] = torch.tensor(([1.0, 2.0, 4.0, 8.0] * 7)[:27])
    constraints[27:54, 3] = 1.0
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert optimizer._is_high_risk_case(inst)
    assert len(specs) > 1
    assert {"soft", "compact", "wide", "tall"}.issubset({spec.profile for spec in specs})


def test_large_case_boundary_repair_profile_is_opt_in(monkeypatch):
    monkeypatch.setenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", "1")
    monkeypatch.setenv("FLOORSET_LARGE_CASE_REPAIR_PROFILES", "normal,large_boundary")
    block_count = 120
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(block_count, 5),
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert {spec.repair_profile for spec in specs} == {"normal", "large_boundary"}


def test_large_case_candidate_matrix_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", raising=False)
    block_count = 120
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(block_count, 5),
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    specs = optimizer._candidate_specs(inst)

    assert len(specs) == 1
    assert specs[0].name in {"soft_relative_order", "compact_relative_order"}


def test_adaptive_profile_uses_compact_for_low_pin_high_b2b_cases():
    block_count = 111
    b2b_edges = torch.tensor([[float(i), float((i + 1) % block_count), 1.0] for i in range(2400)])
    p2b_edges = torch.tensor([[float(i), float(i % block_count), 1.0] for i in range(50)])
    constraints = torch.zeros(block_count, 5)
    constraints[:15, 0] = 1.0
    for offset, cluster_id in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        constraints[20 + offset * 3 : 23 + offset * 3, 3] = cluster_id
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        b2b_edges,
        p2b_edges,
        torch.zeros(50, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    assert optimizer._adaptive_profile_order(inst) == ["compact"]


def test_adaptive_profile_uses_soft_for_sparse_pin_low_b2b_cases():
    block_count = 100
    b2b_edges = torch.tensor([[float(i), float((i + 1) % block_count), 1.0] for i in range(700)])
    p2b_edges = torch.tensor([[float(i), float(i % block_count), 1.0] for i in range(75)])
    constraints = torch.zeros(block_count, 5)
    constraints[:7, 1] = 1.0
    for offset, cluster_id in enumerate([1.0, 2.0, 3.0]):
        constraints[20 + offset * 3 : 23 + offset * 3, 3] = cluster_id
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        b2b_edges,
        p2b_edges,
        torch.zeros(75, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    assert optimizer._adaptive_profile_order(inst) == ["soft"]


def test_adaptive_profile_uses_compact_for_very_high_b2b_low_pin_cases():
    block_count = 117
    b2b_edges = torch.tensor([[float(i), float((i + 1) % block_count), 1.0] for i in range(6700)])
    p2b_edges = torch.tensor([[float(i), float(i % block_count), 1.0] for i in range(93)])
    constraints = torch.zeros(block_count, 5)
    constraints[:17, 0] = 1.0
    for offset, cluster_id in enumerate([1.0, 2.0, 3.0, 4.0]):
        constraints[30 + offset * 3 : 33 + offset * 3, 3] = cluster_id
    inst = parse_instance(
        block_count,
        torch.full((block_count,), 4.0),
        b2b_edges,
        p2b_edges,
        torch.zeros(93, 2),
        constraints,
        torch.full((block_count, 4), -1.0),
    )
    optimizer = ArchitectureV4Optimizer()

    assert optimizer._adaptive_profile_order(inst) == ["compact"]


def test_candidate_workers_are_bounded_by_candidate_count(monkeypatch):
    optimizer = ArchitectureV4Optimizer()
    monkeypatch.setenv("FLOORSET_CANDIDATE_WORKERS", "8")

    assert optimizer._candidate_workers(3) == 3
    assert optimizer._candidate_workers(1) == 1


def test_invalid_candidate_workers_falls_back_to_sequential(monkeypatch):
    optimizer = ArchitectureV4Optimizer()
    monkeypatch.setenv("FLOORSET_CANDIDATE_WORKERS", "many")

    assert optimizer._candidate_workers(3) == 1


def test_profiled_candidate_build_is_sequential(monkeypatch):
    optimizer = ArchitectureV4Optimizer()
    monkeypatch.setenv("FLOORSET_CANDIDATE_WORKERS", "8")
    specs = [
        CandidateSpec("normal", "soft", repair_profile="normal"),
        CandidateSpec("boundary", "soft", repair_profile="boundary_first"),
    ]

    assert optimizer._candidate_workers(len(specs)) == 2
    assert optimizer._candidate_worker_count_for_specs(specs) == 1


def test_candidate_selection_prefers_fewer_soft_violations():
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((3, 4), -1.0),
    )
    compact_dirty = Placement(
        {
            0: Rect(4.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
        }
    )
    larger_clean = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(10.0, 0.0, 2.0, 2.0),
            2: Rect(12.0, 0.0, 2.0, 2.0),
        }
    )
    optimizer = ArchitectureV4Optimizer()

    best = optimizer._select_best_candidate(inst, [compact_dirty, larger_clean])

    assert best is larger_clean


def test_default_candidate_selection_keeps_soft_first_policy():
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((3, 4), -1.0),
    )
    compact_dirty = Placement(
        {
            0: Rect(4.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
        }
    )
    huge_clean = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(200.0, 0.0, 2.0, 2.0),
            2: Rect(202.0, 0.0, 2.0, 2.0),
        }
    )
    optimizer = ArchitectureV4Optimizer()

    best = optimizer._select_best_candidate(inst, [compact_dirty, huge_clean])

    assert best is huge_clean


def test_no_runtime_proxy_candidate_selection_is_opt_in(monkeypatch):
    monkeypatch.setenv("FLOORSET_CANDIDATE_RANK_POLICY", "no_runtime_proxy")
    constraints = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        torch.full((3, 4), -1.0),
    )
    compact_dirty = Placement(
        {
            0: Rect(4.0, 0.0, 2.0, 2.0),
            1: Rect(0.0, 0.0, 2.0, 2.0),
            2: Rect(2.0, 0.0, 2.0, 2.0),
        }
    )
    huge_clean = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(200.0, 0.0, 2.0, 2.0),
            2: Rect(202.0, 0.0, 2.0, 2.0),
        }
    )
    optimizer = ArchitectureV4Optimizer()

    best = optimizer._select_best_candidate(inst, [compact_dirty, huge_clean])

    assert best is compact_dirty


def test_quality_refine_moves_block_to_better_frontier_without_soft_regression():
    inst = parse_instance(
        3,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.tensor([[0.0, 1.0, 10.0]]),
        torch.empty(0, 3),
        torch.empty(0, 2),
        torch.zeros(3, 5),
        torch.full((3, 4), -1.0),
    )
    placement = Placement(
        {
            0: Rect(0.0, 0.0, 2.0, 2.0),
            1: Rect(100.0, 0.0, 2.0, 2.0),
            2: Rect(50.0, 0.0, 2.0, 2.0),
        }
    )
    optimizer = ArchitectureV4Optimizer()
    before = optimizer._no_runtime_proxy_cost(inst, placement, optimizer._placement_metrics(inst, placement))

    refined = optimizer._quality_refine_candidate(inst, placement)
    after = optimizer._no_runtime_proxy_cost(inst, refined, optimizer._placement_metrics(inst, refined))

    assert after < before
    assert refined.rects != placement.rects
