from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from icdc_engine.audit_topology_prior import retained_gain_summary


def _checkpoint(model_fill: float = 1.0, ema_fill: float = 1.0):
    return {
        "model_config": {
            "node_feat_dim": 26,
            "relation_feat_dim": 9,
            "z_dim": 4,
            "z_repr": "xyaspect",
            "d_model": 8,
            "layers": 1,
            "heads": 1,
            "dropout": 0.0,
            "timesteps": 10,
            "self_conditioning": True,
        },
        "model": {"weight": torch.full((2, 2), model_fill)},
        "ema": {"weight": torch.full((2, 2), ema_fill)},
    }


def test_checkpoint_contract_requires_same_shape_and_frozen_3d3f():
    from icdc_engine.train_topology_prior import (
        checkpoint_contract,
        frozen_portfolio_contract,
    )

    base = _checkpoint()
    candidate = copy.deepcopy(base)
    portfolio = frozen_portfolio_contract(flow_slots=3, nref=6)
    assert checkpoint_contract(
        base,
        candidate,
        sampler_method="dpmpp",
        sampler_steps=2,
        candidate_count=6,
        portfolio_contract_sha256=portfolio["sha256"],
    )["ok"]
    assert not checkpoint_contract(
        base,
        candidate,
        sampler_method="dpmpp",
        sampler_steps=2,
        candidate_count=4,
        portfolio_contract_sha256=portfolio["sha256"],
    )["ok"]
    changed = copy.deepcopy(candidate)
    changed["ema"]["weight"] = torch.zeros(3)
    assert not checkpoint_contract(
        base,
        changed,
        sampler_method="dpmpp",
        sampler_steps=2,
        candidate_count=6,
        portfolio_contract_sha256=portfolio["sha256"],
    )["ok"]


def test_source_contract_rejects_c0_that_is_not_source_ema():
    from icdc_engine.train_topology_prior import verify_source_contract

    source = _checkpoint(model_fill=1.0, ema_fill=2.0)
    control = _checkpoint(model_fill=3.0, ema_fill=3.0)
    result = verify_source_contract(source, control)
    assert not result["ok"]
    assert "source_ema_equals_c0" in result["failed"]
    matching = _checkpoint(model_fill=2.0, ema_fill=2.0)
    assert verify_source_contract(source, matching)["ok"]


def test_trainer_ast_forbids_validation_test_and_golden_reads():
    path = Path("src/icdc_engine/train_topology_prior.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    findings = []
    forbidden_names = {"load_test_cases", "FloorplanDatasetLiteTest"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in forbidden_names:
                findings.append((node.lineno, node.id))
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            findings.append((node.lineno, node.attr))
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "golden"
            and isinstance(node.ctx, ast.Load)
        ):
            findings.append((node.lineno, "golden"))
    assert findings == []


def test_load_paired_records_binds_receipt_case_and_sparse_label(tmp_path):
    from icdc_engine.train_topology_prior import load_paired_records
    from icdc_engine.topology_data import fingerprint_case

    case = {
        "instance_id": "worker_0/layouts_2.th#3",
        "n": 2,
        "area": [4.0, 4.0],
        "cons": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "tp": [[-1.0] * 4, [-1.0] * 4],
        "b2b": [],
        "p2b": [],
        "pins": [],
        "hpwl_ref": 1.0,
        "area_ref": 8.0,
    }
    receipt = {
        "relative_path": "worker_0/layouts_2.th",
        "file_sha256": "a" * 64,
        "layout_index": 3,
        "fingerprint": fingerprint_case(case),
    }
    label = {
        "instance_id": case["instance_id"],
        "n": 2,
        "sample_seed": 17,
        "teacher_cost": 1.0,
        "base_cost": 1.2,
        "record_weight": 1.2,
        "edges": [
            {"src": 0, "dst": 1, "axis": 0, "margin": 0.0,
             "kind": "sep", "weight": 1.0}
        ],
        "contacts": [],
        "pin_paths": [],
    }
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text(
        json.dumps({"receipt": receipt, "case": case}, sort_keys=True) + "\n",
        encoding="ascii",
    )
    labels.write_text(
        json.dumps({"receipt": receipt, "label": label}, sort_keys=True) + "\n",
        encoding="ascii",
    )
    records = load_paired_records(corpus, labels)
    assert len(records) == 1
    assert records[0].case["instance_id"] == label["instance_id"]
    assert records[0].label.edges[0].src == 0
    bad = {**label, "instance_id": "worker_9/layouts_9.th#9"}
    labels.write_text(
        json.dumps({"receipt": receipt, "label": bad}, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="binding"):
        load_paired_records(corpus, labels)


def test_export_corpus_reconstructs_only_authorized_training_inputs(tmp_path):
    from icdc_engine.export_corpus import export_corpus
    from icdc_engine.topology_data import (
        CorpusSourceReceipt,
        verified_training_fp_row,
    )

    root = tmp_path / "floorset_lite"
    shard = root / "worker_0" / "layouts_2.th"
    shard.parent.mkdir(parents=True)
    inp = torch.tensor(
        [[[4.0, 0, 1, 0, 0, 1], [4.0, 0, 0, 0, 0, 0]]],
        dtype=torch.float32,
    )
    empty3 = torch.empty((1, 0, 3), dtype=torch.float32)
    empty2 = torch.empty((1, 0, 2), dtype=torch.float32)
    source = [
        inp,
        empty3,
        empty3.clone(),
        empty2,
        torch.zeros((1, 1, 3), dtype=torch.float32),
        torch.tensor(
            [[[2.0, 2.0, 10.0, 20.0], [2.0, 2.0, 30.0, 40.0]]],
            dtype=torch.float32,
        ),
        torch.ones((1, 8), dtype=torch.float32),
    ]
    torch.save(source, shard)
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    provisional = CorpusSourceReceipt(
        "worker_0/layouts_2.th", digest, 0, "0" * 64
    )
    verified = verified_training_fp_row(source, provisional)
    receipt = {
        "relative_path": provisional.relative_path,
        "file_sha256": digest,
        "layout_index": 0,
        "fingerprint": verified.input_fingerprint,
    }
    label = {
        "instance_id": verified.instance_id,
        "n": 2,
        "sample_seed": 17,
        "teacher_cost": 1.0,
        "base_cost": 1.2,
        "record_weight": 1.2,
        "edges": [],
        "contacts": [],
        "pin_paths": [],
    }
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"receipt": receipt, "label": label}) + "\n",
        encoding="ascii",
    )
    out = tmp_path / "out"
    manifest = export_corpus(labels, root, out, canonical_root=root)
    assert manifest["record_count"] == 1
    assert manifest["selection_mod"] == 1
    assert json.loads((out / "labels.jsonl").read_text())["label"] == label
    row = json.loads((out / "corpus.jsonl").read_text())
    assert row["case"]["tp"][0] == [10.0, 20.0, 2.0, 2.0]
    assert row["case"]["tp"][1] == [-1.0, -1.0, -1.0, -1.0]
    payload = (out / "corpus.jsonl").read_bytes()
    assert b"fp_sol" not in payload
    assert b"golden" not in payload


def test_three_trajectory_sparse_loss_backpropagates_through_real_tiny_model():
    from direct_diffusion_model import DirectDenoiser, DirectModelConfig
    from diffusion_model import DiffusionSchedule
    from icdc_engine.topology_data import CorpusSourceReceipt, SparseEdge, TopologyLabel
    from icdc_engine.train_topology_prior import TrainingRecord, _trajectory_loss
    from icdc_engine.topology_data import fingerprint_case

    case = {
        "instance_id": "worker_0/layouts_2.th#3",
        "n": 2,
        "area": [4.0, 4.0],
        "cons": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "tp": [[-1.0] * 4, [-1.0] * 4],
        "b2b": [],
        "p2b": [],
        "pins": [],
        "hpwl_ref": 1.0,
        "area_ref": 8.0,
    }
    label = TopologyLabel(
        case["instance_id"],
        2,
        17,
        1.0,
        1.2,
        1.2,
        (SparseEdge(0, 1, 0, 10.0, "sep", 1.0),),
        (),
        (),
    )
    record = TrainingRecord(
        CorpusSourceReceipt(
            "worker_0/layouts_2.th", "a" * 64, 3, fingerprint_case(case)
        ),
        case,
        label,
    )
    config = DirectModelConfig(
        d_model=8,
        layers=1,
        heads=1,
        timesteps=10,
        node_feat_dim=26,
        relation_feat_dim=9,
    )
    model = DirectDenoiser(config)
    schedule = DiffusionSchedule(config.timesteps, device="cpu")
    losses, spread = _trajectory_loss(
        model,
        schedule,
        [record],
        samples=3,
        steps=2,
        device=torch.device("cpu"),
        grad_steps=2,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(spread)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_retained_gain_is_weighted_over_positive_teacher_gain_only():
    rows = [
        {
            "instance_id": "a", "n": 100, "weight": 2.0,
            "base_cost": 1.4, "teacher_cost": 1.0,
            "student_cost": 1.1, "admitted": True,
        },
        {
            "instance_id": "b", "n": 110, "weight": 1.0,
            "base_cost": 1.2, "teacher_cost": 1.0,
            "student_cost": 1.1, "admitted": True,
        },
        {
            "instance_id": "ignored", "n": 120, "weight": 100.0,
            "base_cost": 1.0, "teacher_cost": 1.0,
            "student_cost": 10.0, "admitted": False,
        },
    ]
    summary = retained_gain_summary(rows)
    assert summary["eligible_count"] == 2
    assert summary["admitted_count"] == 2
    assert summary["teacher_gain_weighted_sum"] == pytest.approx(1.0)
    assert summary["student_gain_weighted_sum"] == pytest.approx(0.7)
    assert summary["retained_gain_fraction"] == pytest.approx(0.7)
    assert not summary["retained_gain_pass"]
