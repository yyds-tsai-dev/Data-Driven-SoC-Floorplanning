import dataclasses
import ast
import copy
import errno
import collections.abc
import gc
import hashlib
import io
import importlib
import importlib.util
import inspect
import os
import stat
import sys
import weakref
import json
import math
import tracemalloc
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Dict, Iterator, Optional, get_args, get_origin, get_type_hints

import pytest
import torch
from icdc.data import BandFileSampler, target_positions_from_rects

from icdc.topology_data import (
    CorpusSourceReceipt,
    fingerprint_case,
    load_sanitized_corpus,
    save_sanitized_corpus,
    split_for_id,
)
from icdc.topology_data import (
    ContactLabel,
    SparseEdge,
    SparseTopologyBatch,
    TopologyLabel,
    canonical_jsonl,
    canonical_jsonl_sha256,
    collate_labels,
    write_sha256_manifest,
    source_instance_id,
)

import icdc.topology_data as topology_data
import icdc.topology_prior as topology_prior


# Task 4B2 RED contract: publication must be a single, injectable
# renameat2(RENAME_NOREPLACE) attempt, with identity-bound cleanup.
def test_teacher_publish_b2_atomic_success_uses_one_noreplace_attempt(tmp_path, monkeypatch):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"
    stage.mkdir(); (stage / "sentinel").write_text("owned")
    lease = t._new_staging_lease(stage)
    original_inode = stage.stat().st_ino; calls = []

    def rename_once(source, dest):
        calls.append((Path(source), Path(dest)))
        os.rename(source, dest)

    monkeypatch.setattr(t, "_renameat2_noreplace", rename_once, raising=False)
    t._publish_staging(lease, destination)
    assert len(calls) == 1 and not stage.exists()
    assert destination.stat().st_ino == original_inode


@pytest.mark.parametrize("err", [errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL, errno.EXDEV])
def test_teacher_publish_b2_platform_errors_propagate_without_fallback(tmp_path, monkeypatch, err):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"; stage.mkdir(); lease = t._new_staging_lease(stage)
    calls = []
    def fail_once(source, dest):
        calls.append((source, dest)); raise OSError(err, os.strerror(err))
    monkeypatch.setattr(t, "_renameat2_noreplace", fail_once, raising=False)
    monkeypatch.setattr(os, "rename", lambda *a: pytest.fail("fallback rename"))
    monkeypatch.setattr(os, "replace", lambda *a: pytest.fail("fallback replace"))
    with pytest.raises(OSError) as exc: t._publish_staging(lease, destination)
    assert exc.value.errno == err and len(calls) == 1
    assert stage.exists() and not destination.exists()


@pytest.mark.parametrize("err", [errno.EEXIST, errno.ENOTEMPTY])
def test_teacher_publish_b2_existing_destination_is_untouched(tmp_path, monkeypatch, err):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"
    stage.mkdir(); lease = t._new_staging_lease(stage); destination.mkdir(); (destination / "sentinel").write_text("foreign")
    inode = destination.stat().st_ino
    calls = []
    def fail_once(*args): calls.append(args); raise OSError(err, "busy")
    monkeypatch.setattr(t, "_renameat2_noreplace", fail_once, raising=False)
    with pytest.raises(ValueError, match="existing output"): t._publish_staging(lease, destination)
    assert len(calls) == 1
    assert destination.stat().st_ino == inode and (destination / "sentinel").read_text() == "foreign"


def test_teacher_publish_b2_eintr_precommit_is_not_retried(tmp_path, monkeypatch):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"; stage.mkdir(); lease = t._new_staging_lease(stage)
    inode = stage.stat().st_ino; calls = []
    def interrupted(*args):
        calls.append(args); raise InterruptedError(errno.EINTR, "interrupted")
    monkeypatch.setattr(t, "_renameat2_noreplace", interrupted, raising=False)
    with pytest.raises(InterruptedError): t._publish_staging(lease, destination)
    assert len(calls) == 1 and stage.stat().st_ino == inode and not destination.exists()


def test_teacher_publish_b2_eintr_postcommit_accepts_exact_lease(tmp_path, monkeypatch):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"; stage.mkdir(); lease = t._new_staging_lease(stage)
    inode = stage.stat().st_ino
    def moved_then_interrupted(source, dest):
        os.rename(source, dest); raise InterruptedError(errno.EINTR, "interrupted")
    monkeypatch.setattr(t, "_renameat2_noreplace", moved_then_interrupted, raising=False)
    t._publish_staging(lease, destination)
    assert not stage.exists() and destination.stat().st_ino == inode


def test_teacher_publish_b2_lease_cleanup_requires_exact_identity(tmp_path):
    t = _teacher(); cleanup = t._cleanup_owned_staging
    exact = tmp_path / "exact"; exact.mkdir(); lease = t._new_staging_lease(exact)
    assert cleanup(lease) is True and not exact.exists()
    for kind in ("symlink", "foreign"):
        stage = tmp_path / kind; stage.mkdir(); lease = t._new_staging_lease(stage)
        sentinel = tmp_path / (kind + "-sentinel"); sentinel.mkdir(); (sentinel / "keep").write_text("keep")
        moved = tmp_path / (kind + "-moved-owned")
        stage.rename(moved)
        if kind == "symlink": stage.symlink_to(sentinel, target_is_directory=True)
        else: stage.mkdir(); (stage / "keep").write_text("keep")
        assert cleanup(lease) is False and (sentinel / "keep").exists()
        if kind == "symlink":
            assert stage.is_symlink()
        else:
            assert stage.is_dir() and (stage / "keep").read_text() == "keep"
        assert moved.exists()


def test_teacher_publish_b2_lease_creation_failure_preserves_replacements(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    out = tmp_path / "out"; _task4_fake_runtime(t, monkeypatch)
    real_new = t._new_staging; real_lease = t._new_staging_lease; moved = tmp_path / "moved-owned"; marker = RuntimeError("lease marker")
    observed = {}
    def attack(stage):
        observed["stage"] = stage
        Path(stage).rename(moved); Path(stage).mkdir(); (Path(stage) / "foreign").write_text("foreign")
        raise marker
    monkeypatch.setattr(t, "_new_staging_lease", attack)
    cleanup_calls = []
    monkeypatch.setattr(t, "_cleanup_owned_staging", lambda lease: cleanup_calls.append(lease) or True)
    with pytest.raises(RuntimeError) as exc:
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert exc.value is marker
    assert not cleanup_calls
    assert out.exists() is False and moved.exists()
    assert (Path(observed["stage"]) / "foreign").read_text() == "foreign"


def test_teacher_publish_b2_ambiguous_eintr_never_deletes_foreign_destination(tmp_path, monkeypatch):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"; stage.mkdir(); lease = t._new_staging_lease(stage)
    moved = tmp_path / "moved-owned"
    def ambiguous(source, dest):
        os.rename(source, moved); Path(dest).mkdir(); (Path(dest) / "foreign").write_text("foreign"); raise InterruptedError(errno.EINTR, "interrupted")
    monkeypatch.setattr(t, "_renameat2_noreplace", ambiguous, raising=False)
    with pytest.raises(RuntimeError, match="ambiguous"):
        t._publish_staging(lease, destination)
    assert destination.exists() and (destination / "foreign").read_text() == "foreign" and moved.exists()


def test_teacher_publish_b2_eintr_ambiguous_source_replacement_is_untouched(tmp_path, monkeypatch):
    t = _teacher(); stage = tmp_path / "stage"; destination = tmp_path / "out"; stage.mkdir(); lease = t._new_staging_lease(stage)
    moved = tmp_path / "moved-owned"
    def ambiguous(source, dest):
        os.rename(source, moved); Path(source).mkdir(); (Path(source) / "foreign").write_text("foreign"); raise InterruptedError(errno.EINTR, "interrupted")
    monkeypatch.setattr(t, "_renameat2_noreplace", ambiguous, raising=False)
    with pytest.raises(RuntimeError, match="ambiguous"):
        t._publish_staging(lease, destination)
    assert not destination.exists() and (stage / "foreign").read_text() == "foreign" and moved.exists()


@pytest.mark.parametrize("replacement", ["symlink", "foreign"])
def test_teacher_publish_b2_transaction_cleanup_refuses_replacement(tmp_path, monkeypatch, replacement):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    _task4_fake_runtime(t, monkeypatch); stages = []
    monkeypatch.setattr(t, "_validate_outcome", lambda value: value)
    real_new = t._new_staging
    def new_staging(destination):
        stage = real_new(destination); stages.append(Path(getattr(stage, "path", stage))); return stage
    monkeypatch.setattr(t, "_new_staging", new_staging)
    moved = tmp_path / "moved-owned"; sentinel = tmp_path / "sentinel"; sentinel.mkdir(); (sentinel / "keep").write_text("keep")
    marker = RuntimeError("publish sentinel")
    publish_calls = []
    def publish(stage, destination):
        publish_calls.append((stage, destination))
        path = Path(getattr(stage, "path", stage)); os.rename(path, moved)
        if replacement == "symlink": path.symlink_to(sentinel, target_is_directory=True)
        else: path.mkdir(); (path / "foreign").write_text("foreign")
        raise marker
    monkeypatch.setattr(t, "_publish_staging", publish)
    with pytest.raises(RuntimeError, match="publish sentinel") as exc: t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert exc.value is marker and len(publish_calls) == 1
    assert not out.exists() and moved.exists() and sentinel.exists()
    replacement_path = stages[-1]
    if replacement == "symlink":
        assert replacement_path.is_symlink() and (sentinel / "keep").exists()
    else:
        assert replacement_path.is_dir() and (replacement_path / "foreign").read_text() == "foreign"


def test_teacher_publish_b2_transaction_existing_destination_race(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    _task4_fake_runtime(t, monkeypatch); stages = []
    monkeypatch.setattr(t, "_validate_outcome", lambda value: value)
    real_new = t._new_staging
    def new_staging(destination):
        stage = real_new(destination); stages.append(Path(getattr(stage, "path", stage))); return stage
    monkeypatch.setattr(t, "_new_staging", new_staging)
    calls = []
    def race(source, destination):
        calls.append((source, destination)); Path(destination).mkdir(); (Path(destination) / "sentinel").write_text("foreign")
        raise OSError(errno.EEXIST, "exists")
    monkeypatch.setattr(t, "_renameat2_noreplace", race, raising=False)
    with pytest.raises(ValueError, match="existing output"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert len(calls) == 1 and out.is_dir() and (out / "sentinel").read_text() == "foreign"
    assert not stages[-1].exists()


# ---------------------------------------------------------------------------
# Task 4 RED contract: deterministic topology teacher
#
# The import is deliberately delayed.  This keeps the legacy topology-prior
# suite collectible while making each frozen teacher contract fail loudly
# until scripts/probes/icdc_topology_teacher.py is implemented.
def _teacher():
    name = "scripts.probes.icdc_topology_teacher"
    if name in sys.modules:
        return sys.modules[name]
    path = Path("scripts/probes/icdc_topology_teacher.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def test_teacher_policy_is_frozen_and_has_exact_fields(tmp_path):
    t = _teacher()
    identity = {"identity_schema": "icdc_canonical_state_v1", "model_config_sha256": "a"*64, "model_keyset_sha256": "b"*64, "ema_keyset_sha256": "c"*64, "ema_state_sha256": "d"*64}
    policy = t.TeacherTrustPolicy(Path("/canonical"), "a" * 64, identity, "b" * 64, "contract", "2.0")
    assert [f.name for f in dataclasses.fields(policy)] == [
        "canonical_root", "expected_checkpoint_sha256", "allowed_model_identity",
        "expected_scorer_sha256", "scorer_contract", "shapely_version"]
    with pytest.raises(FrozenInstanceError):
        policy.scorer_contract = "changed"


def test_teacher_public_entrypoint_rejects_noncanonical_root_before_load(tmp_path, monkeypatch):
    t = _teacher()
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))
    canonical = tmp_path / "canonical"; canonical.mkdir(); out = tmp_path / "out"
    identity = {"identity_schema": "icdc_canonical_state_v1", "model_config_sha256": "a"*64, "model_keyset_sha256": "b"*64, "ema_keyset_sha256": "c"*64, "ema_state_sha256": "d"*64}
    policy = t.TeacherTrustPolicy(canonical, "a"*64, identity, "b"*64, "contract", "2.0")
    args = ["--data-root", str(tmp_path / "other"), "--out-dir", str(out),
            "--index-out", str(out / "training_index.json"),
            "--checkpoint", str(tmp_path / "m.th")]
    with pytest.raises(ValueError, match="(canonical|provenance|validation)"):
        t.teacher_main(args, _trust_policy=policy)
    link = tmp_path / "link"; link.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(ValueError, match="(canonical|provenance|symlink)"):
        t.teacher_main([*args[:1], str(link), *args[2:]], _trust_policy=policy)


def test_teacher_sample_seed_is_stable_order_independent_and_signed63():
    t = _teacher()
    vals = [t._sample_seed(17, "case-1", i) for i in range(4)]
    assert vals == [t._sample_seed(17, "case-1", i) for i in range(4)]
    assert all(0 <= x < 2**63 for x in vals)
    assert len({t._sample_seed(17, "case-1", 0), t._sample_seed(17, "case-2", 0), t._sample_seed(17, "case-1", 1)}) == 3


def test_teacher_intent_parser_requires_exact_realized_geometry():
    t = _teacher(); rects = torch.tensor(
        [[0., 0., 1., 1.], [2., 0., 1., 1.]], dtype=torch.float64)
    case = {"n": 2, "area": [1., 1.], "cons": [[0, 1, 5, 5, 5], [0, 0, 5, 5, 5]], "pins": [], "tp": [[0., 0., 1., 1.], [-1]*4], "groups": [[0, 1]]}
    assert t._proposal_intent_holds("base", rects, case)
    axis = rects.clone(); axis[1, 0] = 1.
    assert t._proposal_intent_holds("axis:0:1:0:1", axis, case)
    assert not t._proposal_intent_holds("axis:0:1:1:1", axis, case)
    assert not t._proposal_intent_holds("axis:malformed", rects, case)
    assert t._proposal_intent_holds("pin:0:1:0:1", torch.tensor(
        [[0., 0., 1., 1.], [1., 0., 1., 1.]], dtype=torch.float64), case)
    assert t._proposal_intent_holds("contact:5:0:1:0:1", torch.tensor(
        [[0., 0., 1., 1.], [1., .25, 1., 1.]], dtype=torch.float64), case)
    gap = torch.tensor(
        [[0., 0., 1., 1.], [1. + torch.finfo(torch.float64).eps, .25, 1., 1.]],
        dtype=torch.float64)
    assert not t._proposal_intent_holds("contact:5:0:1:0:1", gap, case)
    assert not t._proposal_intent_holds("contact:99:0:1:0:1", rects, case)


def test_teacher_official_winner_uses_cost_ordinal_name_and_keeps_base():
    t = _teacher()
    rows = [{"name": "base", "ordinal": 0, "cost_no_runtime": 10, "energy": 1, "feasible": True},
            {"name": "z", "ordinal": 2, "cost_no_runtime": 4, "energy": 0, "feasible": True},
            {"name": "a", "ordinal": 1, "cost_no_runtime": 4, "energy": 9, "feasible": True}]
    assert t._select_official_winner(rows) == 2
    assert t._select_official_winner([{**rows[0], "cost_no_runtime": 1}]) == 0
    for bad in [rows[1:], rows + [dict(rows[0])], rows + [dict(rows[1], ordinal=2)], rows + [dict(rows[1], name="base")], rows + [dict(rows[1], cost_no_runtime=float("nan"))], rows + [dict(rows[1], feasible=False)]]:
        with pytest.raises((ValueError, TypeError)):
            t._select_official_winner(bad)


def test_teacher_weighted_population_is_sorted_and_exact():
    t = _teacher()
    rows = [{"relative_path": "b.json", "layout_index": 1, "instance_id": "b", "n": 12, "base_cost": 10., "teacher_cost": 7.},
            {"relative_path": "a.json", "layout_index": 0, "instance_id": "a", "n": 0, "base_cost": 4., "teacher_cost": 3.}]
    out = t._weighted_population(list(reversed(rows)))
    assert out["B_H"] == pytest.approx((10.*math.e+4.)/(1+math.e))
    assert out["T_H"] == pytest.approx((7.*math.e+3.)/(1+math.e))
    assert out["Delta_H"] == pytest.approx(out["B_H"] - out["T_H"])
    assert out["denominator"] == pytest.approx(1 + math.e)
    assert out["population_sha256"] == t._weighted_population(rows)["population_sha256"]
    forged = [dict(r, weight=999.) for r in rows]
    try:
        forged_out = t._weighted_population(forged)
    except (ValueError, TypeError):
        forged_out = out
    assert forged_out == out
    assert t._weighted_population([dict(rows[0], n=13), rows[1]])["population_sha256"] != out["population_sha256"]


def test_teacher_manifest_hash_omits_self_and_binds_support_hashes():
    t = _teacher(); m = {"status": "complete", "self_sha256": "old", "support_hashes": {"a": "1"}}
    assert t._manifest_self_sha256(m) == t._manifest_self_sha256({**m, "self_sha256": "new"})
    assert t._manifest_self_sha256({**m, "support_hashes": {"a": "2"}}) != t._manifest_self_sha256(m)


def test_teacher_checkpoint_hashes_bytes_before_torch_load(tmp_path, monkeypatch):
    t = _teacher(); p = tmp_path / "model.th"; payload = b"checkpoint-A"; p.write_bytes(payload)
    state = {"layer.weight": torch.ones(1), "layer.bias": torch.zeros(1)}
    def canon(v):
        return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    def encoded_state(s, *, include_values):
        chunks = []
        for key in sorted(s):
            tensor = s[key].detach().cpu().contiguous()
            dtype = str(tensor.dtype).removeprefix("torch.")
            chunks.extend((key.encode("utf-8"), b"\0", dtype.encode("ascii"), b"\0",
                           canon(list(tensor.shape)), b"\0"))
            if include_values:
                chunks.append(tensor.view(torch.uint8).numpy().tobytes())
        return b"".join(chunks)
    ema = {k: v.clone() for k, v in state.items()}
    identity = {
        "identity_schema": "icdc_canonical_state_v1",
        "model_config_sha256": hashlib.sha256(canon({"d": 1})).hexdigest(),
        "model_keyset_sha256": hashlib.sha256(encoded_state(state, include_values=False)).hexdigest(),
        "ema_keyset_sha256": hashlib.sha256(encoded_state(ema, include_values=False)).hexdigest(),
        "ema_state_sha256": hashlib.sha256(encoded_state(ema, include_values=True)).hexdigest(),
    }
    policy = t.TeacherTrustPolicy(tmp_path, hashlib.sha256(payload).hexdigest(), identity, "b"*64, "c", "2")
    seen = []
    kwargs = {}
    def fake_load(fh, **k):
        kwargs.update(k); seen.append(fh.read())
        return {"model": state, "ema": ema, "model_config": {"d": 1}}
    monkeypatch.setattr(torch, "load", fake_load)
    loaded, actual_identity = t._load_verified_checkpoint_bytes(p, policy)
    assert loaded["ema"] and seen == [payload]
    assert actual_identity == {**identity, "checkpoint_sha256": policy.expected_checkpoint_sha256}
    assert kwargs == {"weights_only": True, "map_location": "cpu"}
    policy_bad = dataclasses.replace(policy, expected_checkpoint_sha256="0"*64)
    with pytest.raises(ValueError, match="hash"):
        t._load_verified_checkpoint_bytes(p, policy_bad)
    identity_bad = {**identity, "ema_state_sha256": "0"*64}
    with pytest.raises(ValueError, match="(identity|EMA|ema)"):
        t._load_verified_checkpoint_bytes(
            p, dataclasses.replace(policy, allowed_model_identity=identity_bad))


def test_teacher_checkpoint_identity_schema_and_ema_requirements():
    t = _teacher()
    assert t._checkpoint_identity_schema() == "icdc_canonical_state_v1"
    for bad in [
        {"model": {}, "ema": {}, "model_config": {}},
        {"model": {"w": torch.ones(1)}, "model_config": {"d": 1}},
        {"model": {"w": torch.ones(1)}, "ema": {}, "model_config": {"d": 1}},
    ]:
        with pytest.raises((ValueError, TypeError)):
            t._checkpoint_identity(bad)


def test_checkpoint_identity_codec_has_canonical_vectors():
    codec = importlib.import_module("icdc.checkpoint_identity")
    assert codec.IDENTITY_SCHEMA == "icdc_canonical_state_v1"

    logical = torch.arange(12, dtype=torch.float64).reshape(2, 6)[:, ::2]
    assert not logical.is_contiguous()
    reversed_state = {"z": torch.ones(1), "a": logical}
    sorted_state = {"a": logical.contiguous(), "z": torch.ones(1)}
    assert codec.canonical_keyset_sha256(reversed_state) == codec.canonical_keyset_sha256(sorted_state)
    assert codec.canonical_state_sha256(reversed_state) == codec.canonical_state_sha256(sorted_state)

    exact_header = b"w\0float32\0[2]\0"
    assert codec.canonical_keyset_sha256({"w": torch.zeros(2)}) == hashlib.sha256(exact_header).hexdigest()
    assert codec.canonical_keyset_sha256({"w": torch.zeros(2)}) != codec.canonical_keyset_sha256({"w": torch.zeros(1, 2)})
    assert codec.canonical_config_sha256({"z": 1, "a": [2]}) == codec.canonical_config_sha256({"a": [2], "z": 1})
    with pytest.raises((ValueError, TypeError)):
        codec.canonical_config_sha256({"bad": float("nan")})

    checkpoint = {"model_config": {"d": 1}, "model": sorted_state,
                  "ema": {key: value.clone() for key, value in sorted_state.items()}}
    identity = codec.canonical_checkpoint_identity(checkpoint)
    assert tuple(identity) == (
        "identity_schema", "model_config_sha256", "model_keyset_sha256",
        "ema_keyset_sha256", "ema_state_sha256")
    assert identity == codec.canonical_checkpoint_identity(checkpoint)
    with pytest.raises((ValueError, TypeError)):
        codec.canonical_checkpoint_identity({**checkpoint, "ema": {}})


def test_checkpoint_identity_streams_tensor_bytes_incrementally(monkeypatch):
    codec = importlib.import_module("icdc.checkpoint_identity")
    real_sha256 = hashlib.sha256
    updates = []

    class RecordingHash:
        def __init__(self, data=b""):
            self._inner = real_sha256(data)

        def update(self, data):
            updates.append(len(data))
            self._inner.update(data)

        def hexdigest(self):
            return self._inner.hexdigest()

    monkeypatch.setattr(codec.hashlib, "sha256", RecordingHash)
    state = {"a": torch.arange(64, dtype=torch.float64),
             "b": torch.arange(96, dtype=torch.float32)}
    codec.canonical_state_sha256(state)
    assert len(updates) >= 4
    assert max(updates) <= max(t.numel() * t.element_size() for t in state.values())


def test_teacher_ast_guard_forbids_legacy_data_and_energy_shortlist():
    path = Path("scripts/probes/icdc_topology_teacher.py")
    tree = ast.parse(path.read_text())
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    def resolve(node):
        if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value); return f"{base}.{node.attr}" if base else node.attr
        return ""
    for _ in range(2):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                value = resolve(node.value)
                if value and value != node.targets[0].id and isinstance(node.value, (ast.Name, ast.Attribute)):
                    aliases[node.targets[0].id] = value
    calls = [resolve(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = ("load_test_cases", "FloorplanDatasetLiteTest", "shelf_fallback", "BandFileSampler._instance", "engine.load_model", "engine.sample_bank", "sample_bank")
    assert not any(any(x == bad or x.endswith("." + bad) for bad in forbidden) for x in calls)
    assert not any(x == "collate" or x.endswith(".collate") for x in calls)
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and resolve(n.func) == "torch.load"]
    assert len(loads) == 2
    for node in loads:
        assert len(node.args) == 1 and isinstance(node.args[0], ast.Call) and resolve(node.args[0].func) == "io.BytesIO"
        assert len(node.args[0].args) == 1 and isinstance(node.args[0].args[0], ast.Name) and not node.args[0].keywords
        kw = {k.arg: k.value for k in node.keywords}
        assert set(kw) == {"weights_only", "map_location"}
        assert kw["weights_only"].value is True and kw["map_location"].value == "cpu"

def _task4_static_forbidden(source, *, require_exact_loads=False):
    if _task4_dynamic_forbidden(source):
        return False
    tree = ast.parse(source); aliases = {}
    def resolve(n):
        if isinstance(n, ast.Name): return aliases.get(n.id, n.id)
        if isinstance(n, ast.Attribute): return f"{resolve(n.value)}.{n.attr}"
        return ""
    changed = True
    while changed:
        changed = False
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for x in n.names:
                    aliases[x.asname or x.name.split(".")[0]] = (f"{n.module}.{x.name}" if isinstance(n, ast.ImportFrom) and n.module else x.name)
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                v = resolve(n.value)
                if (isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                        and n.value.func.id == "getattr" and len(n.value.args) == 2
                        and isinstance(n.value.args[1], ast.Constant)
                        and n.value.args[1].value == "load" and resolve(n.value.args[0]) == "torch"):
                    v = "torch.load"
                if v and aliases.get(n.targets[0].id) != v: aliases[n.targets[0].id] = v; changed = True
    bad = ("load_test_cases", "floorplandatasetlitetest", "bandfilesampler._instance", "shelf_fallback", "engine.load_model", "engine.sample_bank", "collate", "tfdl")
    literals = {"golden", "validation", "test", "test_id", "validation_case", "golden_target"}
    def forbidden_literal(value):
        key = str(value).lower()
        return key in literals or key.startswith(("test_", "validation_", "golden_"))
    forbidden_runtime = ("proposal", "admission", "tfdl", "hard-legal", "hard_legal", "intent", "energy", "evaluate_solution", "official_score", "winner")
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            path = resolve(n.func)
            if any(part in path.lower() for part in bad): return False
            if isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and any(x in str(n.args[1].value).lower() for x in (*bad, "load_model", "sample_bank")): return False
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and forbidden_literal(n.slice.value): return False
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args and isinstance(n.args[0], ast.Constant) and forbidden_literal(n.args[0].value): return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"_sample_direct_once", "_build_teacher_batches"}:
            if any(
                isinstance(x, ast.Call)
                and not (node.name == "_sample_direct_once" and resolve(x.func) == "icdc.energy.decode_rects")
                and any(part in resolve(x.func).lower() for part in forbidden_runtime)
                for x in ast.walk(node)
            ):
                return False
    loads = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        path = resolve(n.func)
        is_load = path == "torch.load" or (isinstance(n.func, ast.Call) and isinstance(n.func.func, ast.Name) and n.func.func.id == "getattr" and len(n.func.args) >= 2 and isinstance(n.func.args[1], ast.Constant) and n.func.args[1].value == "load" and resolve(n.func.args[0]) == "torch")
        if is_load:
            loads.append(n)
            if len(n.args) != 1 or not isinstance(n.args[0], ast.Call) or resolve(n.args[0].func) != "io.BytesIO" or len(n.args[0].args) != 1 or n.args[0].keywords or not isinstance(n.args[0].args[0], ast.Name): return False
            kw = {k.arg: k.value for k in n.keywords}
            if set(kw) != {"weights_only", "map_location"} or not isinstance(kw["weights_only"], ast.Constant) or kw["weights_only"].value is not True or not isinstance(kw["map_location"], ast.Constant) or kw["map_location"].value != "cpu": return False
    if require_exact_loads and len(loads) != 2: return False
    return True


def _task4_dynamic_forbidden(source):
    """Reject dynamic execution/import calls while allowing model.eval()."""
    tree = ast.parse(source)
    aliases = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".")[0]
                aliases[local] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                local = imported.asname or imported.name
                aliases[local] = f"{node.module}.{imported.name}"

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return f"{base}.{node.attr}" if base else f".{node.attr}"
        return ""

    for _ in range(len(tree.body) + 2):
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            resolved = resolve(value)
            if resolved and aliases.get(target.id) != resolved:
                aliases[target.id] = resolved
                changed = True
        if not changed:
            break

    bare = {
        "eval", "exec", "__import__", "getattr", "importlib.import_module",
        "builtins.eval", "__builtins__.eval",
    }
    qualified_suffixes = {"exec", "__import__", "getattr", "import_module"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolve(node.func)
        if name in bare or any(name.endswith("." + suffix) for suffix in qualified_suffixes):
            return True
    return False

def test_task4_static_checker_rejects_synthetic_legacy_paths_and_accepts_safe():
    assert _task4_static_forbidden(Path("scripts/probes/icdc_topology_teacher.py").read_text(), require_exact_loads=True)
    assert _task4_dynamic_forbidden("eval('1')")
    assert _task4_dynamic_forbidden("danger = eval\ndanger('1')")
    assert _task4_dynamic_forbidden("import builtins\nbuiltins.eval('1')")
    assert _task4_dynamic_forbidden("import builtins as b\nb.eval('1')")
    assert _task4_dynamic_forbidden("from builtins import eval as run\nrun('1')")
    assert _task4_dynamic_forbidden("__builtins__.eval('1')")
    assert not _task4_dynamic_forbidden("model.eval()")
    assert not _task4_static_forbidden("import icdc.engine as e\nx=e\ny=x\nz=y\na=z\nb=a\ngetattr(b, 'load_model')()")
    assert not _task4_static_forbidden("case={'golden': 1}\ncase.get('golden')\ne.sample_bank()")
    assert not _task4_static_forbidden("from icdc import tfdl as q\nq(x)")
    assert _task4_static_forbidden("def process_case(x):\n  fake_admission(x)\n  official_score(x)")
    assert not _task4_static_forbidden("def _sample_direct_once(x):\n  fake_admission(x)")
    assert not _task4_static_forbidden("def _build_teacher_batches(x):\n  official_score(x)")
    assert _task4_static_forbidden("import io, torch\na='x'; b='y'\ntorch.load(io.BytesIO(a), weights_only=True, map_location='cpu'); torch.load(io.BytesIO(b), weights_only=True, map_location='cpu')", require_exact_loads=True)
    assert not _task4_static_forbidden("import io, torch\na='x'; b='y'; c='z'\ntorch.load(io.BytesIO(a), weights_only=True, map_location='cpu'); torch.load(io.BytesIO(b), weights_only=True, map_location='cpu'); torch_alias=torch; getattr(torch_alias, 'load')(io.BytesIO(c), weights_only=True, map_location='cpu')", require_exact_loads=True)
    assert not _task4_static_forbidden("import io, torch\na='x'; b='y'; c='z'\ntorch.load(io.BytesIO(a), weights_only=True, map_location='cpu'); torch.load(io.BytesIO(b), weights_only=True, map_location='cpu'); torch_alias=torch; loader=getattr(torch_alias, 'load'); loader(io.BytesIO(c), weights_only=True, map_location='cpu')", require_exact_loads=True)
    assert not _task4_static_forbidden("case['test_id']")
    assert not _task4_static_forbidden("case.get('validation_case')")
    assert not _task4_static_forbidden("case['golden_target']")
    assert not _task4_static_forbidden("import io, torch\na='x'; b='y'\ntorch.load(io.BytesIO(a, extra=True), weights_only=True, map_location='cpu'); torch.load(io.BytesIO(b), weights_only=True, map_location='cpu')", require_exact_loads=True)


def _task4_teacher_payload():
    """Small in-memory checkpoint with deliberately different EMA weights."""
    # The repository test configuration exposes partner/; keep this helper
    # free of permanent interpreter-path mutation.
    from direct_diffusion_model import DirectDenoiser, DirectModelConfig
    cfg = DirectModelConfig(node_feat_dim=26, relation_feat_dim=9, z_dim=4,
                            z_repr="xyaspect", d_model=8, layers=1, heads=1,
                            dropout=0.0, timesteps=8, self_conditioning=True)
    model = DirectDenoiser(cfg)
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema_state = {k: (v.detach().clone().add(1.0) if i == 0 and v.is_floating_point() else v.detach().clone())
                 for i, (k, v) in enumerate(model_state.items())}
    return {"model_config": dataclasses.asdict(cfg), "model": model_state,
            "ema": ema_state}


def _task4_case_input(t, seed=17):
    case = {"instance_id": "x.jsonl#0", "n": 3,
            "area": [1.0, 1.0, 1.0000001],
            "cons": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "tp": [[-1.0] * 4] * 3,
            "b2b": [[0, 1, 1.5]], "p2b": [[0, 2, 2.0]],
            "pins": [[.25, .75]], "hpwl_ref": 4.0, "area_ref": 4.0}
    ci = getattr(t, "_CaseInput", _Task4CaseInput)
    receipt = CorpusSourceReceipt(relative_path="x.jsonl", file_sha256="0" * 64,
                                  layout_index=0, fingerprint="1" * 64)
    return ci(case, receipt, "train", seed)

def _task4_anchored_case_input(t, seed=17):
    ci = _task4_case_input(t, seed)
    case = dict(ci.case)
    case["area"] = [6.0, 20.0, 1.0000001]
    case["cons"] = [[1, 0, 3, 5, 1], [0, 1, 4, 5, 2], [0, 0, 0, 6, 4]]
    case["tp"] = [[-1.0, -1.0, 2.0, 3.0], [7.0, 8.0, 4.0, 5.0], [-1.0] * 4]
    return dataclasses.replace(ci, case=case)


def test_task4_materializes_ema_eval_and_frozen_from_memory(monkeypatch):
    t = _teacher(); payload = _task4_teacher_payload()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("reopened checkpoint"))
    real = t.DirectDenoiser.load_state_dict; calls = []
    def wrapped(self, state_dict, *args, **kwargs):
        calls.append(kwargs.copy()); return real(self, state_dict, *args, **kwargs)
    monkeypatch.setattr(t.DirectDenoiser, "load_state_dict", wrapped)
    state = t._materialize_teacher_model(payload, torch.device("cpu"))
    assert calls and calls[-1].get("strict") is True
    assert isinstance(state, t._TeacherModelState)
    assert state.model.training is False and state.schedule.timesteps == 8
    assert all(not p.requires_grad for p in state.model.parameters())
    assert list(state.model.state_dict()) == list(payload["ema"])
    assert all(torch.equal(state.model.state_dict()[k], payload["ema"][k]) for k in payload["ema"])
    differing = {k for k in payload["ema"] if not torch.equal(payload["model"][k], payload["ema"][k])}
    assert differing
    assert all(torch.equal(state.model.state_dict()[k], payload["ema"][k]) for k in payload["ema"])
    assert all(p.dtype == torch.float32 for p in state.model.parameters())
    cpu_before = torch.get_rng_state()
    t._materialize_teacher_model(payload, torch.device("cpu"))
    assert torch.equal(cpu_before, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["nonmapping", "missing_config", "config_nonmapping", "config_incomplete",
    "same_key_model_order", "same_key_ema_order", "model_ema_reversed", "ema_dtype", "model_nonfinite", "ema_nonfinite",
    "missing_model", "model_nonmapping", "model_empty",
    "missing_ema", "ema_nonmapping", "ema_empty", "key_mismatch", "extra_ema", "shape",
    "nontensor", "inf", "zdim", "zrepr", "unknown", "model_shape", "model_dtype",
    "timesteps_zero", "timesteps_bool", "dropout_nan", "dropout_negative", "config_reversed"])
def test_task4_materializer_rejects_malformed_payload(kind):
    bad = _task4_teacher_payload()
    if kind == "nonmapping": bad = []
    elif kind == "missing_config": bad.pop("model_config")
    elif kind == "config_nonmapping": bad["model_config"] = []
    elif kind == "config_incomplete": bad["model_config"].pop("z_dim")
    elif kind == "same_key_model_order": bad["model"] = dict(reversed(list(bad["model"].items())))
    elif kind == "same_key_ema_order": bad["ema"] = dict(reversed(list(bad["ema"].items())))
    elif kind == "model_ema_reversed":
        bad["model"] = dict(reversed(list(bad["model"].items())))
        bad["ema"] = dict(reversed(list(bad["ema"].items())))
    elif kind == "ema_dtype":
        k = next(k for k, v in bad["ema"].items() if v.is_floating_point()); bad["ema"][k] = bad["ema"][k].double()
    elif kind == "model_nonfinite":
        k = next(k for k, v in bad["model"].items() if v.is_floating_point()); bad["model"][k].fill_(float("nan"))
    elif kind == "ema_nonfinite":
        k = next(k for k, v in bad["ema"].items() if v.is_floating_point()); bad["ema"][k].fill_(float("nan"))
    elif kind == "missing_model": bad.pop("model")
    elif kind == "model_nonmapping": bad["model"] = []
    elif kind == "model_empty": bad["model"] = {}
    elif kind == "missing_ema": bad.pop("ema")
    elif kind == "ema_nonmapping": bad["ema"] = []
    elif kind == "ema_empty": bad["ema"] = {}
    elif kind == "key_mismatch": bad["ema"].pop(next(iter(bad["ema"])))
    elif kind == "extra_ema": bad["ema"]["extra"] = torch.ones(1)
    elif kind == "shape":
        k = next(iter(bad["ema"])); v = bad["ema"][k]; bad["ema"][k] = torch.cat((v.reshape(-1), v.new_zeros(1)))
    elif kind == "nontensor": bad["ema"][next(iter(bad["ema"]))] = 1
    elif kind == "inf": bad["ema"][next(iter(bad["ema"]))].fill_(float("inf"))
    elif kind == "zdim": bad["model_config"]["z_dim"] = 3
    elif kind == "zrepr": bad["model_config"]["z_repr"] = "bad"
    elif kind == "unknown": bad["model_config"]["unknown"] = 1
    elif kind == "model_shape":
        k = next(iter(bad["model"])); v = bad["model"][k]; bad["model"][k] = torch.cat((v.reshape(-1), v.new_zeros(1)))
    elif kind == "model_dtype":
        k = next(k for k, v in bad["model"].items() if v.is_floating_point()); bad["model"][k] = bad["model"][k].double()
    elif kind == "timesteps_zero": bad["model_config"]["timesteps"] = 0
    elif kind == "timesteps_bool": bad["model_config"]["timesteps"] = True
    elif kind == "dropout_nan": bad["model_config"]["dropout"] = float("nan")
    elif kind == "dropout_negative": bad["model_config"]["dropout"] = -0.1
    elif kind == "config_reversed": bad["model_config"] = dict(reversed(list(bad["model_config"].items())))
    with pytest.raises(ValueError):
        _teacher()._materialize_teacher_model(bad, torch.device("cpu"))


def test_task4_materializer_rejects_unsupported_device_before_construction(monkeypatch):
    t = _teacher()
    monkeypatch.setattr(t, "DirectDenoiser", lambda *args, **kwargs: pytest.fail("constructed"))
    with pytest.raises(ValueError):
        t._materialize_teacher_model(_task4_teacher_payload(), torch.device("meta"))


def test_task4_teacher_batch_adapter_has_frozen_shapes_dtypes_and_scale():
    t = _teacher(); payload = _task4_teacher_payload(); state = t._materialize_teacher_model(payload, torch.device("cpu")); case = _task4_anchored_case_input(t).case
    case = t._sanitize_case(case, artifact=True)
    direct, diagnostic, _ = t._build_teacher_batches(case, torch.device("cpu"), state.cfg)
    assert direct["area"].shape == (1, 3) and direct["tp"].shape == (1, 3, 4)
    assert direct["cons"].shape == (1, 3, 5) and direct["scale"].shape == (1,)
    for key in ("area", "tp", "b2b", "p2b", "pins", "scale"):
        assert direct[key].dtype == torch.float32 and direct[key].device.type == "cpu"
    assert direct["cons"].dtype == torch.int64 and direct["cons"].shape == (1, 3, 5)
    assert direct["b2b"].shape == (1, 1, 3) and direct["p2b"].shape == (1, 1, 3)
    assert direct["pins"].shape == (1, 1, 2)
    for key in ("area", "tp", "b2b", "p2b", "pins", "scale"):
        assert diagnostic[key].dtype == torch.float64 and diagnostic[key].device.type == "cpu"
    assert diagnostic["cons"].dtype == torch.int64 and diagnostic["cons"].device.type == "cpu"
    assert diagnostic["scale"].item() == direct["scale"].double().item()
    for key, value in (("hpwl_ref", 4.0), ("area_ref", 4.0)):
        assert diagnostic[key].shape == (1,) and diagnostic[key].dtype == torch.float64 and diagnostic[key].device.type == "cpu"
        assert diagnostic[key].item() == value
    for batch in (direct, diagnostic):
        assert batch["area"].shape == (1, 3) and batch["tp"].shape == (1, 3, 4)
        assert batch["cons"].shape == (1, 3, 5) and batch["scale"].shape == (1,)
        for key in ("area", "cons", "tp", "b2b", "p2b", "pins"):
            expected = torch.as_tensor(case[key], dtype=batch[key].dtype).unsqueeze(0)
            if key in ("b2b", "p2b", "pins"): expected = expected.reshape(batch[key].shape)
            assert torch.equal(batch[key], expected)
    expected_scale = torch.sqrt(torch.as_tensor(case["area"], dtype=torch.float32)[torch.as_tensor(case["area"], dtype=torch.float32) > 0].sum()).clamp_min(1.0)
    assert direct["scale"].item() == expected_scale.item()
    assert diagnostic["scale"].item() == expected_scale.double().item()
    f64_scale = torch.sqrt(torch.tensor(case["area"], dtype=torch.float64).sum())
    assert diagnostic["scale"].item() != f64_scale.item()
    tiny = t._sanitize_case(dict(_task4_case_input(t).case, area=[0.1, 0.2, 0.3]), artifact=True)
    tiny_direct, tiny_diag, _ = t._build_teacher_batches(tiny, torch.device("cpu"), state.cfg)
    assert tiny_direct["scale"].item() == 1.0 and tiny_diag["scale"].item() == 1.0
    assert direct["cons"].data_ptr() != diagnostic["cons"].data_ptr()
    direct["cons"][0, 0, 0] = 0
    assert diagnostic["cons"][0, 0, 0].item() == case["cons"][0][0]


def test_task4_teacher_batch_adapter_accepts_empty_relation_tails():
    t = _teacher(); state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    case = dict(_task4_case_input(t).case, b2b=[], p2b=[], pins=[])
    case = t._sanitize_case(case, artifact=True)
    direct, diagnostic, _ = t._build_teacher_batches(case, torch.device("cpu"), state.cfg)
    assert direct["b2b"].shape == diagnostic["b2b"].shape == (1, 0, 3)
    assert direct["p2b"].shape == diagnostic["p2b"].shape == (1, 0, 3)
    assert direct["pins"].shape == diagnostic["pins"].shape == (1, 0, 2)
    assert direct["b2b"].dtype is torch.float32 and diagnostic["b2b"].dtype is torch.float64
    assert direct["p2b"].dtype is torch.float32 and diagnostic["p2b"].dtype is torch.float64
    assert direct["pins"].dtype is torch.float32 and diagnostic["pins"].dtype is torch.float64
    assert direct["node_feat"].shape[-1] == state.cfg.node_feat_dim


def test_task4_teacher_batch_adapter_rejects_float32_scale_overflow():
    t = _teacher(); state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    case = dict(_task4_case_input(t).case, n=2, area=[3e38, 3e38],
                cons=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                tp=[[-1.0] * 4, [-1.0] * 4], b2b=[], p2b=[], pins=[])
    case = t._sanitize_case(case, artifact=True)
    with pytest.raises(ValueError):
        t._build_teacher_batches(case, torch.device("cpu"), state.cfg)


def _task4_condition_case(t):
    state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    case = t._sanitize_case(_task4_case_input(t).case, artifact=True)
    _, _, condition = t._build_teacher_batches(case, torch.device("cpu"), state.cfg)
    return state, case, condition


@pytest.mark.parametrize("kind", [
    "missing", "extra", "non_tensor",
    "node_shape", "adj_shape", "mask_shape", "scale_shape", "rel_shape",
    "node_dtype", "adj_dtype", "mask_dtype", "scale_dtype", "rel_dtype",
    "node_nan", "adj_inf", "scale_inf", "rel_nan", "scale_different",
])
def test_task4_teacher_batch_adapter_rejects_malformed_condition(kind, monkeypatch):
    t = _teacher(); state, case, valid = _task4_condition_case(t)
    condition = {key: value.clone() for key, value in valid.items()}
    n = case["n"]
    if kind == "missing":
        condition.pop("node_feat")
    elif kind == "extra":
        condition["extra"] = torch.zeros(1)
    elif kind == "non_tensor":
        condition["node_feat"] = object()
    elif kind == "node_shape":
        condition["node_feat"] = torch.zeros((1, n, state.cfg.node_feat_dim + 1), dtype=torch.float32)
    elif kind == "adj_shape":
        condition["adj"] = torch.zeros((1, n, n + 1), dtype=torch.float32)
    elif kind == "mask_shape":
        condition["mask"] = torch.zeros((1, n, 1), dtype=torch.bool)
    elif kind == "scale_shape":
        condition["scale"] = torch.zeros((2,), dtype=torch.float32)
    elif kind == "rel_shape":
        condition["rel_feat"] = torch.zeros((1, n, n, state.cfg.relation_feat_dim + 1), dtype=torch.float32)
    elif kind == "node_dtype":
        condition["node_feat"] = condition["node_feat"].double()
    elif kind == "adj_dtype":
        condition["adj"] = condition["adj"].double()
    elif kind == "mask_dtype":
        condition["mask"] = condition["mask"].to(torch.float32)
    elif kind == "scale_dtype":
        condition["scale"] = condition["scale"].double()
    elif kind == "rel_dtype":
        condition["rel_feat"] = condition["rel_feat"].double()
    elif kind == "node_nan":
        condition["node_feat"][0, 0, 0] = float("nan")
    elif kind == "adj_inf":
        condition["adj"][0, 0, 0] = float("inf")
    elif kind == "scale_inf":
        condition["scale"].fill_(float("inf"))
    elif kind == "rel_nan":
        condition["rel_feat"][0, 0, 0, 0] = float("nan")
    elif kind == "scale_different":
        condition["scale"] = condition["scale"] + 1.0
    monkeypatch.setattr(t._ENGINE, "build_cond", lambda direct, cfg: condition)
    with pytest.raises(ValueError):
        t._build_teacher_batches(case, torch.device("cpu"), state.cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_task4_teacher_batch_adapter_rejects_condition_wrong_device(monkeypatch):
    t = _teacher(); payload = _task4_teacher_payload(); state = t._materialize_teacher_model(payload, torch.device("cuda"))
    case = t._sanitize_case(_task4_case_input(t).case, artifact=True)
    _, _, valid = t._build_teacher_batches(case, state.device, state.cfg)
    condition = {key: value.clone() for key, value in valid.items()}
    condition["node_feat"] = condition["node_feat"].to("cpu")
    monkeypatch.setattr(t._ENGINE, "build_cond", lambda direct, cfg: condition)
    with pytest.raises(ValueError):
        t._build_teacher_batches(case, state.device, state.cfg)


@pytest.mark.parametrize("kind", [
    "not_pair", "arity", "z_shape", "mask_shape", "z_dtype", "z_nonfinite",
    "z_device", "mask_dtype", "mask_device",
])
def test_task4_sample_rejects_malformed_known_channels_before_sampler(kind, monkeypatch):
    t = _teacher(); state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    case = t._sanitize_case(_task4_case_input(t).case, artifact=True)
    z = torch.zeros((1, 3, 4), dtype=torch.float32)
    mask = torch.zeros((1, 3, 4), dtype=torch.bool)
    if kind == "not_pair":
        malformed = {"z": z, "mask": mask}
    elif kind == "arity":
        malformed = (z,)
    else:
        if kind == "z_shape":
            z = torch.zeros((1, 3, 3), dtype=torch.float32)
        elif kind == "mask_shape":
            mask = torch.zeros((1, 3), dtype=torch.bool)
        elif kind == "z_dtype":
            z = z.double()
        elif kind == "z_nonfinite":
            z[0, 0, 0] = float("nan")
        elif kind == "z_device":
            if not torch.cuda.is_available():
                pytest.skip("CUDA unavailable")
            z = z.to("cuda")
        elif kind == "mask_dtype":
            mask = mask.to(torch.float32)
        elif kind == "mask_device":
            if not torch.cuda.is_available():
                pytest.skip("CUDA unavailable")
            mask = mask.to("cuda")
        malformed = (z, mask)
    monkeypatch.setattr(t._ENGINE, "known_channels", lambda direct: malformed)
    monkeypatch.setattr(t, "_SAMPLE_DIRECT_DPM", lambda *args, **kwargs: pytest.fail("sampler invoked"), raising=False)
    with pytest.raises(ValueError):
        t._sample_direct_once(state, case, 9)


def test_task4_sample_uses_materialized_cfg_condition_once(monkeypatch):
    t = _teacher(); state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    case = t._sanitize_case(_task4_case_input(t).case, artifact=True)
    import icdc.engine as engine
    original = engine.build_cond; calls = []
    def spy(batch, cfg):
        calls.append((batch, cfg))
        return original(batch, cfg)
    monkeypatch.setattr(engine, "build_cond", spy)
    monkeypatch.setattr(t, "_SAMPLE_DIRECT_DPM", lambda model, cond, schedule, **kwargs: torch.zeros((1, 3, 4), dtype=torch.float32), raising=False)
    def decoder(raw, area, cons, tp, scale):
        out = raw.detach().to(torch.float64).clone()
        out[..., 2:] = 1.0
        return out
    monkeypatch.setattr(t, "_DECODE_RECTS", decoder, raising=False)
    t._sample_direct_once(state, case, 7)
    assert len(calls) == 1 and calls[0][1] is state.cfg


def test_task4_sample_direct_once_calls_dpmpp_once_and_decodes_once(monkeypatch):
    t = _teacher(); payload = _task4_teacher_payload(); state = t._materialize_teacher_model(payload, torch.device("cpu"))
    calls = {"sample": [], "decode": []}
    def sampler(*args, **kwargs):
        calls["sample"].append((args, kwargs, kwargs["generator"].initial_seed()))
        return torch.rand((1, 3, 4), generator=kwargs["generator"])
    monkeypatch.setattr(t, "_SAMPLE_DIRECT_DPM", sampler, raising=False)
    def decoder(raw, area, cons, tp, scale):
        calls["decode"].append((raw, area, cons, tp, scale))
        out = raw.double().clone()
        out[..., 2:] = out[..., 2:].abs() + 1.0
        return out
    monkeypatch.setattr(t, "_DECODE_RECTS", decoder, raising=False)
    case = _task4_anchored_case_input(t).case
    case = t._sanitize_case(case, artifact=True)
    cpu_before = torch.get_rng_state(); cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    out1 = t._sample_direct_once(state, case, 31)
    assert torch.equal(cpu_before, torch.get_rng_state())
    if cuda_before is not None: assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    out2 = t._sample_direct_once(state, case, 31)
    out3 = t._sample_direct_once(state, case, 32)
    for out in (out1, out2, out3):
        assert out.shape == (3, 4) and out.dtype == torch.float64 and out.device.type == "cpu"
        assert torch.isfinite(out).all() and (out[..., 2:] > 0).all()
    assert torch.equal(out1, out2) and not torch.equal(out1, out3)
    assert len(calls["sample"]) == len(calls["decode"]) == 3
    assert [x[2] for x in calls["sample"]] == [31, 31, 32]
    for args, kwargs, _ in calls["sample"]:
        assert len(args) == 3 and args[0] is state.model and args[2] is state.schedule
        cond = args[1]
        assert isinstance(cond, collections.abc.Mapping)
        assert cond["node_feat"].shape == (1, 3, 26) and cond["node_feat"].dtype == torch.float32 and cond["node_feat"].device.type == "cpu"
        assert cond["mask"].shape == (1, 3) and cond["mask"].dtype == torch.bool and bool(cond["mask"].all())
        assert cond["adj"].shape == (1, 3, 3) and cond["adj"].dtype == torch.float32
        if "rel_feat" in cond:
            assert cond["rel_feat"].shape == (1, 3, 3, 9) and cond["rel_feat"].dtype == torch.float32
        assert set(kwargs) == {"steps", "generator", "z_known", "known_mask"} and kwargs["steps"] == 2
        assert kwargs["z_known"].shape == (1, 3, 4) and kwargs["known_mask"].shape == (1, 3, 4)
        assert kwargs["z_known"].dtype == torch.float32 and kwargs["z_known"].device.type == "cpu"
        assert kwargs["known_mask"].dtype == torch.bool and kwargs["known_mask"].device.type == "cpu"
        direct, _, _ = t._build_teacher_batches(case, torch.device("cpu"), state.cfg)
        import icdc.engine as engine
        expected_cond = engine.build_cond(direct, state.cfg)
        assert set(cond) == set(expected_cond)
        assert all(torch.equal(cond[k], expected_cond[k]) for k in expected_cond)
        expected_z, expected_mask = engine.known_channels(direct)
        assert torch.equal(kwargs["z_known"], expected_z)
        assert torch.equal(kwargs["known_mask"], expected_mask)
        assert kwargs["known_mask"].any()
        assert torch.equal(kwargs["known_mask"], torch.tensor([[[False, False, True, False], [True, True, True, False], [False, False, False, False]]]))
        assert direct["node_feat"].shape == (1, 3, 26) and direct["node_feat"].dtype == torch.float32
        assert direct["mask"].shape == (1, 3) and direct["mask"].dtype == torch.bool and bool(direct["mask"].all())
        assert direct["adj"].shape == (1, 3, 3) and direct["adj"].dtype == torch.float32
        if "rel_feat" in direct:
            assert direct["rel_feat"].shape == (1, 3, 3, 9) and direct["rel_feat"].dtype == torch.float32
    for raw, area, cons, tp, scale in calls["decode"]:
        assert raw.dtype == area.dtype == tp.dtype == scale.dtype == torch.float64
        assert cons.dtype == torch.int64
        assert all(x.device.type == "cpu" for x in (raw, area, cons, tp, scale))

@pytest.mark.parametrize("bad", [torch.zeros((1, 3, 4, 1)), torch.zeros((2, 3, 4)),
    torch.tensor([[[0., 0., 1., 1.]] * 3], dtype=torch.float32),
    torch.zeros((1, 2, 4)), torch.full((1, 3, 4), float("nan")),
    torch.full((1, 3, 4), float("inf")), torch.tensor([[[0., 0., 0., 1.]] * 3]),
    torch.tensor([[[0., 0., 1., -1.]] * 3])])
def test_task4_sample_rejects_malformed_decoder_results(monkeypatch, bad):
    t = _teacher(); state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    monkeypatch.setattr(t, "_SAMPLE_DIRECT_DPM", lambda *a, **k: torch.zeros((1, 3, 4)), raising=False)
    monkeypatch.setattr(t, "_DECODE_RECTS", lambda *a, **k: bad, raising=False)
    with pytest.raises(ValueError):
        t._sample_direct_once(state, _task4_anchored_case_input(t).case, 9)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_task4_sample_rejects_cuda_decoder_result(monkeypatch):
    t = _teacher(); state = t._materialize_teacher_model(_task4_teacher_payload(), torch.device("cpu"))
    monkeypatch.setattr(t, "_SAMPLE_DIRECT_DPM", lambda *a, **k: torch.zeros((1, 3, 4)), raising=False)
    monkeypatch.setattr(t, "_DECODE_RECTS", lambda *a, **k: torch.tensor([[[0., 0., 1., 1.]] * 3], dtype=torch.float64, device="cuda"), raising=False)
    with pytest.raises(ValueError):
        t._sample_direct_once(state, _task4_anchored_case_input(t).case, 9)


def test_task4_runtime_materializes_lazily_caches_and_never_reopens(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "trusted"; root.mkdir()
    payload = _task4_teacher_payload(); path = root / "unused.th"; torch.save(payload, path)
    identity = t._checkpoint_identity(payload)
    policy = t.TeacherTrustPolicy(root, hashlib.sha256(path.read_bytes()).hexdigest(), identity,
                                  t._SCORER_SHA256, "iccad2026_evaluate_cost_no_runtime_v1", "2.0.5")
    runtime = t._runtime_hooks(); assert runtime.authorizing is True
    runtime.preflight(policy, path)
    materialize_calls = []; sample_calls = []; materialized = object()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("reopened checkpoint"))
    import icdc.engine as engine
    if hasattr(engine, "load_model"):
        monkeypatch.setattr(engine, "load_model", lambda *a, **k: pytest.fail("legacy reopen"))
    monkeypatch.setattr(t, "_select_teacher_device", lambda: torch.device("cpu"), raising=False)
    monkeypatch.setattr(t, "_materialize_teacher_model", lambda p, d: materialize_calls.append((p, d)) or materialized, raising=False)
    monkeypatch.setattr(t, "_sample_direct_once", lambda s, c, seed: sample_calls.append((s, seed)) or torch.ones((3, 4), dtype=torch.float64), raising=False)
    monkeypatch.setattr(t, "_run_candidate_lifecycle", lambda *args, **kwargs: object(), raising=False)
    monkeypatch.setattr(t, "_outcome_from_lifecycle", lambda *args: object(), raising=False)
    ci1 = _task4_case_input(t, 101); ci2 = _task4_case_input(t, 202)
    runtime.process_case(ci1)
    runtime.process_case(ci2)
    assert len(materialize_calls) == 1 and materialize_calls[0][1].type == "cpu"
    assert t._checkpoint_identity(materialize_calls[0][0]) == identity
    assert [seed for _, seed in sample_calls] == [101, 202]

def test_task4_runtime_preflight_replaces_cached_payload_and_failed_preflight_clears(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "trusted"; root.mkdir()
    p1, p2, p3 = (_task4_teacher_payload() for _ in range(3))
    key = next(iter(p1["ema"]))
    p2["ema"][key] = p2["ema"][key] + 2
    p3["ema"][key] = p3["ema"][key] + 3
    paths = [root / "one.th", root / "two.th", root / "three.th"]
    for path, payload in zip(paths, (p1, p2, p3)): torch.save(payload, path)
    def policy(path, payload):
        return t.TeacherTrustPolicy(root, hashlib.sha256(path.read_bytes()).hexdigest(),
            t._checkpoint_identity(payload), t._SCORER_SHA256,
            "iccad2026_evaluate_cost_no_runtime_v1", "2.0.5")
    runtime = t._runtime_hooks(); materialized = []; sampled = []; refs = []
    class Token: pass
    def materialize(payload, device):
        materialized.append(payload)
        token = Token(); refs.append(weakref.ref(token)); return token
    monkeypatch.setattr(t, "_select_teacher_device", lambda: torch.device("cpu"), raising=False)
    monkeypatch.setattr(t, "_materialize_teacher_model", materialize, raising=False)
    monkeypatch.setattr(t, "_sample_direct_once", lambda s, c, seed: sampled.append(seed) or torch.ones((3, 4), dtype=torch.float64), raising=False)
    def process_with_seams(seed):
        with monkeypatch.context() as m:
            m.setattr(t, "_run_candidate_lifecycle", lambda *args, **kwargs: object(), raising=False)
            m.setattr(t, "_outcome_from_lifecycle", lambda *args: object(), raising=False)
            runtime.process_case(_task4_case_input(t, seed))
    runtime.preflight(policy(paths[0], p1), paths[0]); process_with_seams(101)
    assert [t._checkpoint_identity(x) for x in materialized] == [t._checkpoint_identity(p1)]
    assert refs[0]() is not None
    runtime.preflight(policy(paths[1], p2), paths[1]); gc.collect(); assert refs[0]() is None
    process_with_seams(202); assert refs[1]() is not None
    with pytest.raises(ValueError, match="scorer_sha256"):
        runtime.preflight(dataclasses.replace(policy(paths[1], p2), expected_scorer_sha256="0" * 64), paths[1])
    gc.collect(); assert refs[1]() is None
    with pytest.raises(RuntimeError, match="before trusted preflight"):
        runtime.process_case(_task4_case_input(t, 303))
    runtime.preflight(policy(paths[2], p3), paths[2]); process_with_seams(303); assert refs[2]() is not None
    assert [t._checkpoint_identity(x) for x in materialized] == [
        t._checkpoint_identity(p1), t._checkpoint_identity(p2), t._checkpoint_identity(p3)
    ]
    assert sampled == [101, 202, 303]


_TASK4_FILES = (
    "train_corpus.jsonl", "heldout_corpus.jsonl", "train_labels.jsonl",
    "heldout_labels.jsonl", "proposals.jsonl", "rejections.jsonl",
    "training_index.json", "g0_manifest.json",
)
_TASK4_JSONL = _TASK4_FILES[:6]
_TASK4_SQLITE_MAGIC = b"SQLite format 3\x00"


def _task4_private_case_spool(stage, expected_identity=None):
    """Return the sole lease-owned SQLite spool, excluding every public artifact."""
    stage = Path(stage)
    private_entries = [entry for entry in stage.iterdir()
                       if entry.name not in _TASK4_FILES]
    sqlite_mains = []
    for entry in private_entries:
        mode = entry.lstat().st_mode
        assert not stat.S_ISLNK(mode), f"private spool entry must not be a symlink: {entry.name}"
        assert stat.S_ISREG(mode), f"private spool entry must be a regular file: {entry.name}"
        with entry.open("rb") as handle:
            header = handle.read(len(_TASK4_SQLITE_MAGIC))
        if header == _TASK4_SQLITE_MAGIC:
            sqlite_mains.append(entry)
    assert len(sqlite_mains) == 1, (
        "replay requires exactly one unambiguous private SQLite case spool; "
        f"found {[entry.name for entry in sqlite_mains]} among {[entry.name for entry in private_entries]}"
    )
    spool = sqlite_mains[0]
    assert spool.parent == stage
    allowed_private_names = {
        f"{spool.name}{suffix}" for suffix in ("", "-journal", "-wal", "-shm")
    }
    assert {entry.name for entry in private_entries} <= allowed_private_names, (
        "only the private case spool and its SQLite sidecars may share the lease: "
        f"{[entry.name for entry in private_entries]}"
    )
    identity = (spool, spool.stat().st_dev, spool.stat().st_ino)
    if expected_identity is not None:
        assert identity == expected_identity, "the private SQLite spool changed during replay"
    return identity, tuple(private_entries)


def _task4_stage_fd_targets(stage):
    """Open descriptors bound to a staging tree, including deleted children."""
    fd_root = Path("/proc/self/fd")
    if not fd_root.is_dir():
        pytest.skip("/proc/self/fd is unavailable")
    stage_text = str(Path(stage).absolute())
    targets = []
    for fd in fd_root.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target == stage_text or target.startswith(stage_text + os.sep):
            targets.append((fd.name, target))
    return targets


@dataclasses.dataclass(frozen=True)
class _Task4CaseInput:
    case: collections.abc.Mapping
    receipt: CorpusSourceReceipt
    partition: str
    sample_seed: int


@dataclasses.dataclass(frozen=True)
class _Task4CaseOutcome:
    label_row: collections.abc.Mapping
    proposal_rows: collections.abc.Sequence
    rejection_rows: collections.abc.Sequence
    base_cost: float
    teacher_cost: float
    legal: bool
    covered: bool
    case_status: str


@dataclasses.dataclass(frozen=True)
class _Task4Runtime:
    preflight: object
    process_case: object
    authorizing: bool


class _Task4ProcessFailure(RuntimeError):
    pass


class _Task4PublishFailure(RuntimeError):
    pass


def _task4_tensors(*, soft_delta=0, metric_delta=0):
    input_data = torch.tensor(
        [[[4, 0, 0, 0, 0, 0], [6, 1, 0, 0, 0, 0], [9, 0, 1, 0, 2, 5]],
         [[5, 0, 0, 0, 0, 0], [20, 1, 0, 0, 0, 0], [20, 0, 1, 0, 2, 5]]],
        dtype=torch.float32,
    )
    fp = torch.tensor(
        [[[71, 73, 701, 703], [2, 3, 401, 403], [3, 3, 10, 11]],
         [[79, 83, 709, 719], [5, 4, 809, 811], [4, 5, 20, 21]]],
        dtype=torch.float32,
    )
    if soft_delta:
        fp[0, 0, 2:] += soft_delta
        fp[0, 1, 2:] += soft_delta
        fp[1, 0, 2:] += soft_delta
        fp[1, 1, 2:] += soft_delta
    metrics = torch.tensor(
        [[100 + metric_delta, 0, 0, 0, 0, 0, 2, 3],
         [200 + metric_delta, 0, 0, 0, 0, 0, 5, 7]],
        dtype=torch.float32,
    )
    return (
        input_data,
        torch.zeros((2, 0, 3), dtype=torch.float32),
        torch.zeros((2, 0, 3), dtype=torch.float32),
        torch.zeros((2, 0, 2), dtype=torch.float32),
        torch.zeros((2, 2, 3), dtype=torch.float32),
        fp,
        metrics,
    )


def _task4_shard(root, *, soft_delta=0, metric_delta=0,
                 relative_path="worker_2/layouts_0.th"):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_task4_tensors(soft_delta=soft_delta, metric_delta=metric_delta), path)
    return path


def _task4_single_shard(root, relative_path, *, metric_delta=0):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = _task4_tensors(metric_delta=metric_delta)
    torch.save(tuple(array[:1] for array in tensors), path)
    return path


def _task4_expected_case(index):
    if index == 0:
        return {
            "instance_id": "worker_2/layouts_0.th#0", "n": 3,
            "area": [4.0, 6.0, 9.0],
            "cons": [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 2, 5]],
            "tp": [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 2.0, 3.0], [10.0, 11.0, 3.0, 3.0]],
            "b2b": [], "p2b": [], "pins": [], "hpwl_ref": 5.0, "area_ref": 100.0,
        }
    return {
        "instance_id": "worker_2/layouts_0.th#1", "n": 3,
        "area": [5.0, 20.0, 20.0],
        "cons": [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 2, 5]],
        "tp": [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 5.0, 4.0], [20.0, 21.0, 4.0, 5.0]],
        "b2b": [], "p2b": [], "pins": [], "hpwl_ref": 12.0, "area_ref": 200.0,
    }


def _task4_fake_runtime(t, monkeypatch, *, calls=None, fail_at=None,
                        preflight_calls=None, events=None, outcome_factory=None):
    calls = [] if calls is None else calls
    preflight_calls = [] if preflight_calls is None else preflight_calls
    events = [] if events is None else events
    case_input_type = getattr(t, "_CaseInput", _Task4CaseInput)
    # P1-C freezes this schema independently of the production dataclass so
    # teacher_main validation, rather than test-fixture construction, is the
    # first RED boundary until the bridge lands.
    case_outcome_type = _Task4CaseOutcome
    runtime_type = getattr(t, "_TeacherRuntime", _Task4Runtime)
    if case_input_type is not _Task4CaseInput:
        assert [field.name for field in dataclasses.fields(case_input_type)] == [
            "case", "receipt", "partition", "sample_seed"]
    if runtime_type is not _Task4Runtime:
        assert [field.name for field in dataclasses.fields(runtime_type)] == [
            "preflight", "process_case", "authorizing"]

    def preflight(policy, checkpoint):
        preflight_calls.append((policy, checkpoint))
        return {
            "trust_ok": True, "input_ok": True, "scorer_ok": True,
            "checkpoint_sha256": policy.expected_checkpoint_sha256,
            "model_identity": dict(policy.allowed_model_identity),
            "scorer_sha256": policy.expected_scorer_sha256,
            "scorer_contract": policy.scorer_contract,
            "shapely_version": policy.shapely_version,
        }

    def process_case(case_input):
        calls.append(case_input)
        events.append(("process", case_input.receipt.relative_path,
                       case_input.receipt.layout_index))
        if fail_at is not None and len(calls) - 1 == fail_at:
            raise _Task4ProcessFailure(f"process failure at {fail_at}")
        return (outcome_factory(t, case_input) if outcome_factory else
                _task4_good_outcome(t, case_input))

    monkeypatch.setattr(
        t, "_CaseOutcome", case_outcome_type,
    )
    monkeypatch.setattr(
        t, "_runtime_hooks", lambda: runtime_type(preflight, process_case, True),
        raising=False,
    )
    return calls


def _task4_args(root, out, *, n_min=1, heldout_mod=2, max_files=None):
    args = ["--data-root", str(root), "--out-dir", str(out),
            "--index-out", str(out / "training_index.json"),
            "--checkpoint", str(root / "unused.th"), "--seed", "20260813",
            "--heldout-mod", str(heldout_mod), "--n-min", str(n_min)]
    if max_files is not None:
        args.extend(["--max-files", str(max_files)])
    return args


def _task4_run(tmp_path, monkeypatch, *, n_min=1, max_files=None,
               soft_delta=0, relative_path="worker_2/layouts_0.th", fail_at=None,
               preflight_calls=None, events=None, policy=None, outcome_factory=None):
    t = _teacher()
    root = tmp_path / "floorset_lite"
    _task4_shard(root, soft_delta=soft_delta, relative_path=relative_path)
    out = tmp_path / "out"
    calls = _task4_fake_runtime(t, monkeypatch, fail_at=fail_at,
                                preflight_calls=preflight_calls, events=events,
                                outcome_factory=outcome_factory)
    policy = _policy_for(root) if policy is None else policy
    result = t.teacher_main(
        _task4_args(root, out, n_min=n_min, max_files=max_files),
        _trust_policy=policy,
    )
    return t, root, out, calls, result


def _task4_jsonl(path):
    raw = path.read_bytes()
    if not raw:
        return []
    assert raw.endswith(b"\n")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def _task4_canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _task4_receipt(path, index, case):
    return _task4_receipt_bytes(path.read_bytes(), index, case)


def _task4_receipt_bytes(source_bytes, index, case):
    return dataclasses.asdict(CorpusSourceReceipt(
        relative_path="worker_2/layouts_0.th",
        file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        layout_index=index, fingerprint=fingerprint_case(case),
    ))


def _task4_envelope(case, receipt, partition, sample_seed):
    return {"receipt": receipt, "instance_id": case["instance_id"],
            "partition": partition, "sample_seed": sample_seed, "n": case["n"]}


def _task4_normalize_known_hashes(value, *, kind):
    """Normalize only receipt hashes and root manifest hashes for soft-fp comparison."""
    if kind == "index":
        normalized = dict(value)
        rows = normalized["rows"]
        normalized["rows"] = _task4_normalize_known_hashes(rows, kind="label")
        return normalized
    if kind in {"label", "proposal"}:
        rows = value if isinstance(value, list) else [value]
        normalized = []
        for row in rows:
            row = dict(row)
            receipt = dict(row["receipt"])
            receipt["file_sha256"] = "SOURCE_SHA_SENTINEL"
            row["receipt"] = receipt
            normalized.append(row)
        return normalized if isinstance(value, list) else normalized[0]
    if kind == "manifest":
        value = dict(value)
        value.pop("support_hashes", None)
        value.pop("self_sha256", None)
        return value
    return value


def _task4_artifact_projection(out):
    projection = {}
    for name in _TASK4_FILES:
        path = out / name
        if name in _TASK4_JSONL:
            kind = "label" if "labels" in name else "proposal" if name == "proposals.jsonl" else "corpus"
            projection[name] = _task4_normalize_known_hashes(_task4_jsonl(path), kind=kind)
        elif name == "training_index.json":
            projection[name] = _task4_normalize_known_hashes(
                json.loads(path.read_text()), kind="index"
            )
        else:
            projection[name] = _task4_normalize_known_hashes(
                json.loads(path.read_text()), kind="manifest"
            )
    return projection


_TASK4_DIGEST_FIELDS = {
    "file_sha256", "fingerprint", "checkpoint_sha256", "model_config_sha256",
    "model_keyset_sha256", "ema_keyset_sha256", "ema_state_sha256",
    "scorer_sha256", "population_sha256", "self_sha256",
}


def _task4_assert_no_forbidden_semantic_values(values, forbidden):
    forbidden_numbers = set(forbidden)
    forbidden_strings = {str(value) for value in forbidden_numbers}

    def walk(value, field=None):
        if field in _TASK4_DIGEST_FIELDS or field == "support_hashes":
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, key)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)) and value in forbidden_numbers:
            raise AssertionError(f"forbidden semantic numeric value: {value!r}")
        if isinstance(value, str) and value in forbidden_strings:
            raise AssertionError(f"forbidden semantic string value: {value!r}")

    walk(values)


def _task4_expected_trust(policy):
    return {
        "trust_ok": True, "input_ok": True, "scorer_ok": True,
        "checkpoint_sha256": policy.expected_checkpoint_sha256,
        "model_identity": dict(policy.allowed_model_identity),
        "scorer_sha256": policy.expected_scorer_sha256,
        "scorer_contract": policy.scorer_contract,
        "shapely_version": policy.shapely_version,
    }


# Task 4 source-fix A: transaction-level RED probes for the evidence boundary.
def _task4_padded_tensors():
    source = list(_task4_tensors())
    padding_input = torch.tensor([[[-1, 0, 0, 0, 0, 0]]] * 2, dtype=torch.float32)
    source[0] = torch.cat((source[0], padding_input), dim=1)
    source[1] = torch.tensor(
        [[[0, 1, 1.5], [-1, -1, -1]], [[1, 2, 2.5], [-1, -1, -1]]],
        dtype=torch.float32,
    )
    source[2] = torch.tensor(
        [[[0, 2, 3], [-1, -1, -1]], [[0, 0, 4], [-1, -1, -1]]],
        dtype=torch.float32,
    )
    source[3] = torch.tensor(
        [[[1.25, 2.5], [-1, -1]], [[3.5, 4.5], [-1, -1]]],
        dtype=torch.float32,
    )
    source[4] = torch.zeros((2, 3, 3), dtype=torch.float32)
    source[5] = torch.cat(
        (source[5], torch.full((2, 1, 4), -1.0, dtype=torch.float32)), dim=1
    )
    return tuple(source)


_TASK4_P1C_PROPOSAL_KEYS = {
    "ordinal", "name", "intended_intent", "raw_topology_fingerprint",
    "intended_topology_fingerprint", "realized_topology_fingerprint",
    "admission_status", "admission_reason", "drift", "hard",
    "hard_status", "intent_status", "official_cost", "feasible",
    "official_status", "diagnostic_energy", "energy_status", "winner",
    "status",
}


def _task4_good_label(case_input, *, base_cost=1.1, teacher_cost=1.1):
    """A valid base-winner label, derived from this exact corpus case."""
    raw = _task4_p1c_raw(case_input.case)
    return _task4_p1c_label(
        case_input, raw, 0, "base", base_cost, teacher_cost,
    )


def _task4_good_proposal(t, case_input, *, cost=1.1):
    """A valid base-no-improvement proposal with real topology SHA evidence."""
    raw = _task4_p1c_raw(case_input.case)
    base = _task4_p1c_record(
        t, 0, "base", raw, raw, cost=cost, energy=None,
        energy_status="unavailable", hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    lifecycle = t._CandidateLifecycle((base,), 0, cost, cost)
    return _task4_p1c_expected_row(case_input, raw, base, lifecycle)


def _task4_good_outcome(t, case_input, *, cost=1.1):
    """The shared default fixture is logically valid base-no-improvement."""
    return t._CaseOutcome(
        _task4_good_label(case_input, base_cost=cost, teacher_cost=cost),
        (_task4_good_proposal(t, case_input, cost=cost),), (),
        cost, cost, True, True, "winner_base_no_improvement",
    )


def _task4_install_custom_runtime(t, monkeypatch, *, preflight_factory=None,
                                  outcome_factory=None, process_calls=None,
                                  preflight_calls=None):
    process_calls = [] if process_calls is None else process_calls
    preflight_calls = [] if preflight_calls is None else preflight_calls

    def preflight(policy, checkpoint):
        preflight_calls.append((policy, checkpoint))
        value = _task4_expected_trust(policy)
        return preflight_factory(dict(value), policy) if preflight_factory else value

    def process(case_input):
        process_calls.append(case_input)
        return (outcome_factory(t, case_input) if outcome_factory else
                _task4_good_outcome(t, case_input))

    monkeypatch.setattr(t, "_CaseOutcome", _Task4CaseOutcome)
    monkeypatch.setattr(
        t, "_runtime_hooks", lambda: t._TeacherRuntime(preflight, process, True)
    )
    return process_calls, preflight_calls


def _task4_execute_tensors(tmp_path, monkeypatch, tensors, *, heldout_mod=2,
                           preflight_factory=None, outcome_factory=None):
    t = _teacher(); root = tmp_path / "floorset_lite"
    path = root / "worker_2" / "layouts_0.th"; path.parent.mkdir(parents=True)
    torch.save(tensors, path); out = tmp_path / "out"
    calls, preflight_calls = _task4_install_custom_runtime(
        t, monkeypatch, preflight_factory=preflight_factory,
        outcome_factory=outcome_factory,
    )
    policy = _policy_for(root)
    result = t.teacher_main(
        _task4_args(root, out, heldout_mod=heldout_mod), _trust_policy=policy
    )
    return t, root, out, calls, preflight_calls, policy, result


def test_teacher_evidence_a_cli_help_exposes_data_root():
    import subprocess
    proc = subprocess.run(
        [sys.executable, "scripts/probes/icdc_topology_teacher.py", "--help"],
        cwd=Path(__file__).parents[1], capture_output=True, text=True,
    )
    assert proc.returncode == 0 and "--data-root" in proc.stdout


def test_teacher_evidence_a_padded_connectivity_transaction(tmp_path, monkeypatch):
    t, _root, out, calls, _preflight, _policy, result = _task4_execute_tensors(
        tmp_path, monkeypatch, _task4_padded_tensors(),
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert result == 0 and len(calls) == 2
    expected_connections = (
        ([[0, 1, 1.5]], [[0, 2, 3.0]], [[1.25, 2.5]]),
        ([[1, 2, 2.5]], [[0, 0, 4.0]], [[3.5, 4.5]]),
    )
    for case_input, (b2b, p2b, pins) in zip(calls, expected_connections):
        assert case_input.case["n"] == 3
        assert (case_input.case["b2b"], case_input.case["p2b"],
                case_input.case["pins"]) == (b2b, p2b, pins)
    corpora = (_task4_jsonl(out / "train_corpus.jsonl") +
               _task4_jsonl(out / "heldout_corpus.jsonl"))
    assert {row["instance_id"] for row in corpora} == {
        "worker_2/layouts_0.th#0", "worker_2/layouts_0.th#1"
    }
    index = json.loads((out / "training_index.json").read_text())
    assert all(row["block_count"] == 3 for row in index["rows"])
    by_id = {row["instance_id"]: row for row in corpora}
    for index_value, (b2b, p2b, pins) in enumerate(expected_connections):
        emitted = by_id[f"worker_2/layouts_0.th#{index_value}"]
        assert emitted["n"] == 3
        assert (emitted["b2b"], emitted["p2b"], emitted["pins"]) == (
            b2b, p2b, pins,
        )
        assert all(len(emitted[field]) == 3 for field in ("area", "cons", "tp"))
    assert all(
        row["receipt"]["fingerprint"] == fingerprint_case(by_id[row["instance_id"]])
        for row in index["rows"]
    )


_TASK4_BAD_SOURCE_KINDS = (
    "nan", "bool", "requires_grad", "wrong_width", "wrong_batch",
    "wrong_tree", "area_after_pad", "partial_b2b", "partial_p2b",
    "partial_pins", "edge_after_pad", "p2b_after_pad", "pin_after_pad",
    "negative_weight", "p2b_negative_weight", "b2b_range", "p2b_pin_range",
    "p2b_block_range", "fractional_constraint",
)

_TASK4_BAD_SOURCE_ERRORS = {
    "nan": "source tensors",
    "bool": "source tensors",
    "requires_grad": "source tensors",
    "wrong_width": "source shapes",
    "wrong_batch": "source batch",
    "wrong_tree": "source tree",
    "area_after_pad": "area padding",
    "partial_b2b": "partial padding",
    "partial_p2b": "partial padding",
    "partial_pins": "partial padding",
    "edge_after_pad": "noncontiguous padding",
    "p2b_after_pad": "noncontiguous padding",
    "pin_after_pad": "noncontiguous padding",
    "negative_weight": "b2b weight",
    "p2b_negative_weight": "p2b weight",
    "b2b_range": "b2b endpoint",
    "p2b_pin_range": "p2b endpoint",
    "p2b_block_range": "p2b endpoint",
    "fractional_constraint": "constraint",
}


def _task4_bad_source(kind):
    source = list(_task4_padded_tensors())
    if kind == "nan": source[6] = source[6].clone(); source[6][0, 0] = float("nan")
    elif kind == "bool": source[6] = source[6].to(torch.bool)
    elif kind == "requires_grad": source = [value.clone().requires_grad_() for value in source]
    elif kind == "wrong_width": source[1] = source[1][..., :2]
    elif kind == "wrong_batch": source[1] = source[1][:1]
    elif kind == "wrong_tree": source[4] = source[4][:, :2]
    elif kind == "area_after_pad": source[0] = source[0].clone(); source[0][0, 1, 0] = -1
    elif kind == "partial_b2b": source[1] = source[1].clone(); source[1][0, 1] = torch.tensor([0, -1, -1.])
    elif kind == "partial_p2b": source[2] = source[2].clone(); source[2][0, 1] = torch.tensor([0, -1, -1.])
    elif kind == "partial_pins": source[3] = source[3].clone(); source[3][0, 1] = torch.tensor([0, -1.])
    elif kind == "edge_after_pad": source[1] = source[1].clone(); source[1][0] = torch.tensor([[-1, -1, -1.], [0, 1, 1.]])
    elif kind == "p2b_after_pad": source[2] = source[2].clone(); source[2][0] = torch.tensor([[-1, -1, -1.], [0, 1, 1.]])
    elif kind == "pin_after_pad": source[3] = source[3].clone(); source[3][0] = torch.tensor([[-1, -1.], [1, 2.]])
    elif kind == "negative_weight": source[1] = source[1].clone(); source[1][0, 0, 2] = -2
    elif kind == "p2b_negative_weight": source[2] = source[2].clone(); source[2][0, 0, 2] = -2
    elif kind == "b2b_range": source[1] = source[1].clone(); source[1][0, 0, 1] = 9
    elif kind == "p2b_pin_range": source[2] = source[2].clone(); source[2][0, 0, 0] = 9
    elif kind == "p2b_block_range": source[2] = source[2].clone(); source[2][0, 0, 1] = 9
    elif kind == "fractional_constraint": source[0] = source[0].clone(); source[0][0, 0, 1] = .5
    else: raise AssertionError(kind)
    return tuple(source)


@pytest.mark.parametrize("kind", _TASK4_BAD_SOURCE_KINDS)
def test_teacher_evidence_a_malformed_source_transaction(tmp_path, monkeypatch, kind):
    t = _teacher(); root = tmp_path / "floorset_lite"
    path = root / "worker_2" / "layouts_0.th"; path.parent.mkdir(parents=True)
    torch.save(_task4_bad_source(kind), path); out = tmp_path / "out"
    calls, _ = _task4_install_custom_runtime(t, monkeypatch)
    with pytest.raises(ValueError, match=_TASK4_BAD_SOURCE_ERRORS[kind]):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert calls == [] and not out.exists()


def test_teacher_evidence_a_sparse_layout_rejected():
    t = _teacher(); source = list(_task4_tensors()); source[6] = source[6].to_sparse()
    with pytest.raises(ValueError): t._validate_source_shard(tuple(source))


_TASK4_PREFLIGHT_FIELDS = (
    "trust_ok", "input_ok", "scorer_ok", "checkpoint_sha256",
    "model_identity", "scorer_sha256", "scorer_contract", "shapely_version",
)


_TASK4_BAD_PREFLIGHT_KINDS = tuple(
    [f"missing:{key}" for key in _TASK4_PREFLIGHT_FIELDS] +
    ["extra"] + [f"mismatch:{key}" for key in (
        "checkpoint_sha256", "model_identity", "scorer_sha256",
        "scorer_contract", "shapely_version")] +
    [f"false:{key}" for key in ("trust_ok", "input_ok", "scorer_ok")] +
    [f"nonbool:{key}" for key in ("trust_ok", "input_ok", "scorer_ok")]
)


@pytest.mark.parametrize("kind", _TASK4_BAD_PREFLIGHT_KINDS)
def test_teacher_evidence_a_preflight_transaction(tmp_path, monkeypatch, kind):
    def mutate(value, _policy):
        action, _, field = kind.partition(":")
        if action == "missing": value.pop(field)
        elif action == "extra": value["extra"] = 1
        elif action == "mismatch": value[field] = "mismatch"
        elif action == "false": value[field] = False
        elif action == "nonbool": value[field] = 1
        return value
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls, _ = _task4_install_custom_runtime(t, monkeypatch, preflight_factory=mutate)
    with pytest.raises(ValueError):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert calls == [] and not out.exists()


_TASK4_PROTECTED_FIELDS = (
    "receipt", "instance_id", "partition", "sample_seed", "n",
    "base_cost", "teacher_cost", "record_weight",
)


_TASK4_BAD_OUTCOME_KINDS = (
    "base_nan", "base_inf", "base_zero", "base_negative", "teacher_nan",
    "teacher_inf", "teacher_zero", "teacher_negative", "teacher_gt_base",
    "ratio_overflow", "legal_nonbool", "covered_nonbool", "label_extra",
    "label_protected", "label_nan",
    *(f"proposal_protected:{key}" for key in _TASK4_PROTECTED_FIELDS),
    *(f"rejection_protected:{key}" for key in _TASK4_PROTECTED_FIELDS),
    "no_winner", "two_winners", "nonbase_winner", "ordinal_bool",
    "ordinal_negative", "name_empty", "name_duplicate", "ordinal_duplicate",
    "feasible_false", "status_bad", "official_bool", "official_nan",
    "official_zero", "official_mismatch", "hard_nan", "diagnostic_energy_nan",
    "drift_nan", "missing_proposal_field",
)


def _task4_bad_outcome(t, case_input, kind):
    valid = _task4_good_outcome(t, case_input)
    label = dict(valid.label_row)
    proposal = dict(valid.proposal_rows[0]); proposals = [proposal]
    rejections = []; base_cost = teacher_cost = 1.1; legal = covered = True
    if kind.startswith("base_"):
        base_cost = {"base_nan": float("nan"), "base_inf": float("inf"),
                     "base_zero": 0.0, "base_negative": -1.0}[kind]
    elif kind.startswith("teacher_"):
        teacher_cost = {"teacher_nan": float("nan"), "teacher_inf": float("inf"),
                        "teacher_zero": 0.0, "teacher_negative": -1.0,
                        "teacher_gt_base": 2.0}[kind]
        proposal["official_cost"] = teacher_cost
    elif kind == "ratio_overflow": base_cost, teacher_cost = 1e308, 1e-308; proposal["official_cost"] = teacher_cost
    elif kind == "legal_nonbool": legal = 1
    elif kind == "covered_nonbool": covered = "yes"
    elif kind == "label_extra": label["extra"] = 1
    elif kind == "label_protected": label["receipt"] = {}
    elif kind == "label_nan": label["edges"] = [{"weight": float("nan")}]
    elif kind.startswith("proposal_protected:"):
        proposal[kind.partition(":")[2]] = "forged"
    elif kind.startswith("rejection_protected:"):
        rejections = [{kind.partition(":")[2]: "forged"}]
    elif kind == "no_winner": proposal["winner"] = False
    elif kind == "two_winners": proposals.append({**proposal, "name": "other", "ordinal": 1})
    elif kind == "nonbase_winner": proposal["winner"] = False; proposals.append({**proposal, "name": "other", "ordinal": 1, "winner": True})
    elif kind == "ordinal_bool": proposal["ordinal"] = False
    elif kind == "ordinal_negative": proposal["ordinal"] = -1
    elif kind == "name_empty": proposal["name"] = ""
    elif kind == "name_duplicate": proposals.append({**proposal, "ordinal": 1, "winner": False})
    elif kind == "ordinal_duplicate": proposals.append({**proposal, "name": "other", "winner": False})
    elif kind == "feasible_false": proposal["feasible"] = False
    elif kind == "status_bad": proposal["status"] = "candidate"
    elif kind == "official_bool": proposal["official_cost"] = True
    elif kind == "official_nan": proposal["official_cost"] = float("nan")
    elif kind == "official_zero": proposal["official_cost"] = 0.0
    elif kind == "official_mismatch": proposal["official_cost"] = .5
    elif kind == "hard_nan": proposal["hard"] = {"legal": float("nan")}
    elif kind == "diagnostic_energy_nan": proposal["diagnostic_energy"] = float("nan")
    elif kind == "drift_nan": proposal["drift"] = {"max_abs": float("nan")}
    elif kind == "missing_proposal_field": proposal.pop("admission_status")
    else: raise AssertionError(kind)
    return t._CaseOutcome(label, tuple(proposals), tuple(rejections),
                          base_cost, teacher_cost, legal, covered,
                          "winner_base_no_improvement")


@pytest.mark.parametrize("kind", _TASK4_BAD_OUTCOME_KINDS)
def test_teacher_evidence_a_outcome_transaction(tmp_path, monkeypatch, kind):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls, _ = _task4_install_custom_runtime(
        t, monkeypatch, outcome_factory=lambda module, case_input: _task4_bad_outcome(
            module, case_input, kind,
        )
    )
    with pytest.raises(ValueError):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert len(calls) == 1 and not out.exists()


def test_teacher_evidence_a_ratio_transaction(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls, _ = _task4_install_custom_runtime(
        t, monkeypatch,
        outcome_factory=lambda module, case_input: _task4_p1c_mutation_outcome(
            module, case_input, base_cost=2.5, teacher_cost=1.25,
        ),
    )
    result = t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert result == 0 and len(calls) == 2
    labels = (_task4_jsonl(out / "train_labels.jsonl") +
              _task4_jsonl(out / "heldout_labels.jsonl"))
    assert [row["record_weight"] for row in labels] == [2.0, 2.0]
    population = json.loads((out / "g0_manifest.json").read_text())["population"]
    assert (population["B_H"], population["T_H"], population["Delta_H"]) == (2.5, 1.25, 1.25)


@pytest.mark.parametrize("kind", ("worker", "shard"))
def test_teacher_evidence_a_numeric_symlink_rejected_before_load(tmp_path, monkeypatch, kind):
    t = _teacher(); root = tmp_path / "floorset_lite"; root.mkdir(); external = tmp_path / "external"
    _task4_shard(external)
    if kind == "worker": (root / "worker_2").symlink_to(external / "worker_2", target_is_directory=True)
    else:
        worker = root / "worker_2"; worker.mkdir()
        (worker / "layouts_0.th").symlink_to(external / "worker_2" / "layouts_0.th")
    loads = []; calls, _ = _task4_install_custom_runtime(t, monkeypatch)
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append((a, k)))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="symlink"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert loads == [] and calls == [] and not out.exists()


def test_teacher_evidence_a_nonnumeric_symlink_decoys_are_ignored(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    external = tmp_path / "external"; _task4_shard(external)
    (root / "worker_bad").symlink_to(external / "worker_2", target_is_directory=True)
    (root / "worker_2" / "layouts_bad.th").symlink_to(external / "worker_2" / "layouts_0.th")
    out = tmp_path / "out"; calls, _ = _task4_install_custom_runtime(
        t, monkeypatch, outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 0
    assert len(calls) == 2


@pytest.mark.parametrize("heldout_mod,expected_train,expected_heldout", ((3, 2, 0), (1, 0, 2)))
def test_teacher_evidence_a_one_sided_split_is_coverage_kill(
        tmp_path, monkeypatch, heldout_mod, expected_train, expected_heldout):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls, _ = _task4_install_custom_runtime(t, monkeypatch)
    result = t.teacher_main(
        _task4_args(root, out, heldout_mod=heldout_mod), _trust_policy=_policy_for(root)
    )
    assert result != 0 and len(calls) == 2
    manifest = json.loads((out / "g0_manifest.json").read_text())
    assert manifest["state"] == "KILLED_LEGALITY_OR_COVERAGE"
    assert manifest["training_authorized"] is False
    assert manifest["coverage"] == {
        "eligible_train": expected_train, "eligible_heldout": expected_heldout,
        "heldout_winners": expected_heldout, "legal": True, "covered": False,
        "mutation_proposals": 0, "mutation_admitted": 0,
        "mutation_intent_survived": 0, "positive_gain_heldout": 0,
        "eligible_heldout_weight": expected_heldout * math.exp(3 / 12),
        "positive_gain_heldout_weight": 0.0,
        "rejection_counts": {
            "admission_failed": 0, "hard_audit_failed": 0,
            "intent_not_survived": 0, "official_infeasible": 0,
            "official_invalid_cost": 0, "official_evaluator_error": 0,
            "base_unavailable": 0,
        },
    }


def _task4_p1c_coverage(calls, outcomes):
    reasons = (
        "admission_failed", "hard_audit_failed", "intent_not_survived",
        "official_infeasible", "official_invalid_cost", "official_evaluator_error",
        "base_unavailable",
    )
    heldout = [(case_input, outcome) for case_input, outcome in zip(calls, outcomes)
               if case_input.partition == "heldout"]
    return {
        "eligible_train": sum(case_input.partition == "train" for case_input in calls),
        "eligible_heldout": len(heldout),
        "heldout_winners": sum(outcome.case_status != "base_unavailable"
                                for _, outcome in heldout),
        "legal": all(outcome.legal for outcome in outcomes),
        "covered": (all(outcome.covered for outcome in outcomes)
                    and any(case_input.partition == "train" for case_input in calls)
                    and bool(heldout)
                    and all(outcome.case_status != "base_unavailable" for _, outcome in heldout)),
        "mutation_proposals": sum(
            row["ordinal"] != 0
            for outcome in outcomes for row in outcome.proposal_rows
        ),
        "mutation_admitted": sum(
            row["ordinal"] != 0 and row["admission_status"] == "admitted"
            for outcome in outcomes for row in outcome.proposal_rows
        ),
        "mutation_intent_survived": sum(
            row["ordinal"] != 0 and row["intent_status"] == "passed"
            for outcome in outcomes for row in outcome.proposal_rows
        ),
        "positive_gain_heldout": sum(
            outcome.covered and outcome.teacher_cost < outcome.base_cost
            for _, outcome in heldout
        ),
        "eligible_heldout_weight": sum(
            math.exp(case_input.case["n"] / 12) for case_input, _ in heldout
        ),
        "positive_gain_heldout_weight": sum(
            math.exp(case_input.case["n"] / 12)
            for case_input, outcome in heldout
            if outcome.covered and outcome.teacher_cost < outcome.base_cost
        ),
        "rejection_counts": {
            reason: sum(row["reason"] == reason
                        for outcome in outcomes for row in outcome.rejection_rows)
            for reason in reasons
        },
    }


def _task4_assert_success_artifacts(t, root, out, calls, policy,
                                    preflight_calls, *, bounded=False,
                                    training_authorized=False,
                                    state="STOP_HARD_GAIN_MISSED", source_bytes=None,
                                    outcome_factory=_task4_good_outcome):
    assert sorted(path.name for path in out.iterdir()) == sorted(_TASK4_FILES)
    expected_cases = [_task4_expected_case(0), _task4_expected_case(1)]
    if source_bytes is None:
        source_bytes = (root / "worker_2/layouts_0.th").read_bytes()
    expected_receipts = [_task4_receipt_bytes(source_bytes, i, expected_cases[i])
                         for i in range(2)]
    assert [item.case for item in calls] == expected_cases
    assert [source_instance_id(item.receipt) for item in calls] == [case["instance_id"] for case in expected_cases]
    assert [item.partition for item in calls] == ["train", "heldout"]
    assert [item.sample_seed for item in calls] == [
        t._sample_seed(20260813, case["instance_id"], 0) for case in expected_cases]
    assert len(preflight_calls) == 1
    assert preflight_calls[0] == (policy, root / "unused.th")

    train_rows = _task4_jsonl(out / "train_corpus.jsonl")
    heldout_rows = _task4_jsonl(out / "heldout_corpus.jsonl")
    assert train_rows == [expected_cases[0]] and heldout_rows == [expected_cases[1]]
    env = lambda i, partition: _task4_envelope(
        expected_cases[i], expected_receipts[i], partition,
        t._sample_seed(20260813, expected_cases[i]["instance_id"], 0),
    )
    expected_outcomes = [outcome_factory(t, case_input) for case_input in calls]
    expected_labels = [
        {**env(index, partition), **dict(outcome.label_row)}
        for (index, partition), outcome in zip(((0, "train"), (1, "heldout")), expected_outcomes)
    ]
    expected_proposals = [
        {**env(index, partition), **dict(row)}
        for (index, partition), outcome in zip(((0, "train"), (1, "heldout")), expected_outcomes)
        for row in outcome.proposal_rows
    ]
    expected_rejections = [
        {**env(index, partition), **dict(row)}
        for (index, partition), outcome in zip(((0, "train"), (1, "heldout")), expected_outcomes)
        for row in outcome.rejection_rows
    ]
    assert _task4_jsonl(out / "train_labels.jsonl") == [expected_labels[0]]
    assert _task4_jsonl(out / "heldout_labels.jsonl") == [expected_labels[1]]
    assert _task4_jsonl(out / "proposals.jsonl") == expected_proposals
    assert _task4_jsonl(out / "rejections.jsonl") == expected_rejections

    index = json.loads((out / "training_index.json").read_text())
    assert index["schema"] == "icdc_topology_training_index_v1"
    assert index["rows"] == [
        {"receipt": expected_receipts[i], "instance_id": expected_cases[i]["instance_id"],
         "source_row_count": 2, "block_count": 3,
         "partition": ("train" if i == 0 else "heldout"), "sample_ordinal": 0,
         "sample_seed": t._sample_seed(20260813, expected_cases[i]["instance_id"], 0),
         "status": expected_outcomes[i].case_status}
        for i in range(2)]

    expected_population = _task4_p1c_canonical_population([
        {"relative_path": item.receipt.relative_path, "layout_index": item.receipt.layout_index,
         "instance_id": item.case["instance_id"], "n": item.case["n"],
         "base_cost": outcome.base_cost, "teacher_cost": outcome.teacher_cost}
        for item, outcome in zip(calls, expected_outcomes) if item.partition == "heldout"
    ])
    manifest = json.loads((out / "g0_manifest.json").read_text())
    assert set(manifest) == {"schema", "status", "state", "secondary_reasons",
                             "training_authorized", "bounded_max_files", "trust",
                             "population", "coverage", "support_hashes", "self_sha256"}
    assert manifest["schema"] == "icdc_topology_teacher_g0_v1"
    assert manifest["status"] == "complete"
    assert manifest["state"] == state
    assert manifest["secondary_reasons"] == []
    assert manifest["training_authorized"] is training_authorized
    assert manifest["bounded_max_files"] is bounded
    assert manifest["trust"] == _task4_expected_trust(policy)
    assert manifest["population"] == expected_population
    assert manifest["coverage"] == _task4_p1c_coverage(calls, expected_outcomes)
    assert manifest["support_hashes"] == {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in _TASK4_FILES if name != "g0_manifest.json"
    }
    assert manifest["self_sha256"] == t._manifest_self_sha256(manifest)

    for name in _TASK4_JSONL:
        rows = _task4_jsonl(out / name)
        expected_bytes = b"".join(_task4_canonical_json(row) + b"\n" for row in rows)
        assert (out / name).read_bytes() == expected_bytes
    for name in ("training_index.json", "g0_manifest.json"):
        parsed = json.loads((out / name).read_text())
        assert (out / name).read_bytes() == _task4_canonical_json(parsed) + b"\n"
    parsed_artifacts = [
        *[_task4_jsonl(out / name) for name in _TASK4_JSONL],
        json.loads((out / "training_index.json").read_text()),
        json.loads((out / "g0_manifest.json").read_text()),
    ]
    _task4_assert_no_forbidden_semantic_values(
        parsed_artifacts, (701, 703, 401, 403, 709, 719, 809, 811))
    blobs = b"".join((out / name).read_bytes() for name in _TASK4_FILES)
    assert b"timestamp" not in blobs and str(root).encode() not in blobs
    return manifest


def test_teacher_source_transaction_publishes_exact_sanitized_evidence(tmp_path, monkeypatch):
    preflight_calls = []
    policy = _policy_for(tmp_path / "floorset_lite")
    t, root, out, calls, result = _task4_run(tmp_path, monkeypatch,
                                             preflight_calls=preflight_calls,
                                             policy=policy,
                                             outcome_factory=_task4_p1c_mutation_outcome)
    assert result == 0
    _task4_assert_success_artifacts(
        t, root, out, calls, policy, preflight_calls,
        training_authorized=True, state="TARGET_GAIN_MET",
        outcome_factory=_task4_p1c_mutation_outcome,
    )


def test_teacher_source_reads_each_file_once_from_same_bytesio_under_path_swap(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; path = _task4_shard(root)
    out = tmp_path / "out"; calls = []
    preflight_calls = []
    _task4_fake_runtime(
        t, monkeypatch, calls=calls, preflight_calls=preflight_calls,
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    original_bytes = path.read_bytes(); seen = []
    reader_calls = []; real_read = t._read_verified_shard

    def read(root_path, worker, layout):
        reader_calls.append((worker, layout))
        return real_read(root_path, worker, layout)

    real_load = torch.load

    def load_from_verified_bytes(source, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert kwargs == {"weights_only": True, "map_location": "cpu"}
        seen.append(source.getvalue())
        path.write_bytes(b"replaced-after-read")
        return real_load(source, **kwargs)

    monkeypatch.setattr(t, "_read_verified_shard", read)
    monkeypatch.setattr(torch, "load", load_from_verified_bytes)
    policy = _policy_for(root)
    result = t.teacher_main(_task4_args(root, out), _trust_policy=policy)
    assert result == 0 and reader_calls == [(2, 0)]
    assert seen == [original_bytes] and len(calls) == 2
    expected_sha = hashlib.sha256(original_bytes).hexdigest()
    index = json.loads((out / "training_index.json").read_text())
    assert {row["receipt"]["file_sha256"] for row in index["rows"]} == {expected_sha}
    _task4_assert_success_artifacts(
        t, root, out, calls, policy, preflight_calls,
        training_authorized=True, state="TARGET_GAIN_MET",
        source_bytes=original_bytes, outcome_factory=_task4_p1c_mutation_outcome,
    )


@pytest.mark.parametrize("soft_delta", [1000, 2000])
def test_teacher_soft_fp_variants_only_change_narrow_source_hashes(tmp_path, monkeypatch, soft_delta):
    base_dir = tmp_path / "base"; soft_dir = tmp_path / "soft"
    base_root = base_dir / "floorset_lite"; soft_root = soft_dir / "floorset_lite"
    base_path = _task4_shard(base_root); soft_path = _task4_shard(soft_root, soft_delta=soft_delta)
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() != hashlib.sha256(soft_path.read_bytes()).hexdigest()
    t = _teacher(); base_out = base_dir / "out"; soft_out = soft_dir / "out"
    _task4_fake_runtime(t, monkeypatch, outcome_factory=_task4_p1c_mutation_outcome)
    assert t.teacher_main(_task4_args(base_root, base_out), _trust_policy=_policy_for(base_root)) == 0
    _task4_fake_runtime(t, monkeypatch, outcome_factory=_task4_p1c_mutation_outcome)
    assert t.teacher_main(_task4_args(soft_root, soft_out), _trust_policy=_policy_for(soft_root)) == 0
    assert _task4_artifact_projection(base_out) == _task4_artifact_projection(soft_out)
    changed = (701 + soft_delta, 703 + soft_delta, 401 + soft_delta,
               403 + soft_delta, 709 + soft_delta, 719 + soft_delta,
               809 + soft_delta, 811 + soft_delta)
    parsed_artifacts = [
        *[_task4_jsonl(soft_out / name) for name in _TASK4_JSONL],
        json.loads((soft_out / "training_index.json").read_text()),
        json.loads((soft_out / "g0_manifest.json").read_text()),
    ]
    _task4_assert_no_forbidden_semantic_values(parsed_artifacts, changed)


def test_teacher_fresh_outputs_are_byte_identical_and_existing_output_is_untouched(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    first, second = tmp_path / "first", tmp_path / "second"
    policy = _policy_for(root)
    calls_first = []; preflight_first = []
    _task4_fake_runtime(
        t, monkeypatch, calls=calls_first, preflight_calls=preflight_first,
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert t.teacher_main(_task4_args(root, first), _trust_policy=policy) == 0
    _task4_assert_success_artifacts(
        t, root, first, calls_first, policy, preflight_first,
        training_authorized=True, state="TARGET_GAIN_MET",
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    calls_second = []; preflight_second = []
    _task4_fake_runtime(
        t, monkeypatch, calls=calls_second, preflight_calls=preflight_second,
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert t.teacher_main(_task4_args(root, second), _trust_policy=policy) == 0
    _task4_assert_success_artifacts(
        t, root, second, calls_second, policy, preflight_second,
        training_authorized=True, state="TARGET_GAIN_MET",
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert {name: (first / name).read_bytes() for name in _TASK4_FILES} == {
        name: (second / name).read_bytes() for name in _TASK4_FILES}
    before = {name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in _TASK4_FILES}
    calls = _task4_fake_runtime(t, monkeypatch)
    with pytest.raises(ValueError, match="existing output"):
        t.teacher_main(_task4_args(root, first), _trust_policy=_policy_for(root))
    assert calls == []
    assert before == {name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in _TASK4_FILES}


@pytest.mark.parametrize("fail_at", [0, 1])
def test_teacher_process_failure_is_transactional_for_each_eligible_row(tmp_path, monkeypatch, fail_at):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    _task4_fake_runtime(t, monkeypatch, fail_at=fail_at)
    with pytest.raises(_Task4ProcessFailure, match=rf"process failure at {fail_at}"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert not out.exists()


def test_teacher_late_invalid_shard_aborts_ingestion_before_runtime_or_destination(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_2/layouts_0.th")
    invalid = root / "worker_2/layouts_2.th"
    torch.save(_task4_tensors()[:-1], invalid)
    calls = []; _task4_fake_runtime(t, monkeypatch, calls=calls)
    stages = []; real_staging = t._new_staging
    monkeypatch.setattr(
        t, "_new_staging",
        lambda destination: (stages.append(Path(real_staging(destination))) or stages[-1]),
    )
    with pytest.raises(ValueError, match="source schema"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert calls == [] and not out.exists()
    assert stages and not stages[-1].exists()


def test_teacher_duplicate_discovery_entry_aborts_ingestion_before_runtime_or_destination(
        tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls = []; _task4_fake_runtime(t, monkeypatch, calls=calls)
    discovered = t._iter_approved_shards(root)
    assert len(discovered) == 1
    monkeypatch.setattr(t, "_iter_approved_shards", lambda _root: [*discovered, *discovered])
    stages = []; real_staging = t._new_staging
    monkeypatch.setattr(
        t, "_new_staging",
        lambda destination: (stages.append(Path(real_staging(destination))) or stages[-1]),
    )
    with pytest.raises(ValueError, match="duplicate"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert calls == [] and not out.exists()
    assert stages and not stages[-1].exists()


def test_teacher_private_case_spool_is_staged_during_replay_and_absent_at_publish(
        tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls = []; _task4_fake_runtime(t, monkeypatch, calls=calls,
                                    outcome_factory=_task4_p1c_mutation_outcome)
    stages = []; real_staging = t._new_staging
    monkeypatch.setattr(
        t, "_new_staging",
        lambda destination: (stages.append(Path(real_staging(destination))) or stages[-1]),
    )
    spool_identity = None
    observed_spool_paths = set()

    def private_spool():
        nonlocal spool_identity
        assert stages
        spool_identity, entries = _task4_private_case_spool(stages[-1], spool_identity)
        observed_spool_paths.update(entries)
        return spool_identity

    runtime = t._runtime_hooks(); real_process = runtime.process_case
    def process(case_input):
        private_spool()
        return real_process(case_input)
    monkeypatch.setattr(t, "_runtime_hooks", lambda: dataclasses.replace(runtime, process_case=process))

    real_write = t._JsonlWriter.write
    def write(writer, name, value):
        private_spool()
        return real_write(writer, name, value)
    monkeypatch.setattr(t._JsonlWriter, "write", write)

    import builtins
    real_open = builtins.open

    class IndexWriterSpy:
        def __init__(self, handle):
            self._handle = handle

        def write(self, data):
            private_spool()
            return self._handle.write(data)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def spool_aware_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "w" in mode and Path(path).name == "training_index.json":
            return IndexWriterSpy(handle)
        return handle

    monkeypatch.setattr(t, "open", spool_aware_open, raising=False)
    real_fsync = t.os.fsync
    real_sha256 = hashlib.sha256

    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if stat.S_ISDIR(t.os.fstat(fd).st_mode) and path == stages[-1]:
            if not (path / "g0_manifest.json").exists():
                assert not [entry for entry in path.iterdir() if entry.name not in _TASK4_FILES]
                assert spool_identity is not None
                assert all(not entry.exists() for entry in observed_spool_paths)
        return real_fsync(fd)

    monkeypatch.setattr(t.os, "fsync", fsync)

    def sha256(data=b"", *args, **kwargs):
        if stages and stages[-1].exists() and not (stages[-1] / "g0_manifest.json").exists():
            support_payloads = [
                (stages[-1] / name).read_bytes()
                for name in _TASK4_FILES if name != "g0_manifest.json"
                and (stages[-1] / name).exists()
            ]
            if data in support_payloads:
                assert not [entry for entry in stages[-1].iterdir()
                            if entry.name not in _TASK4_FILES]
                assert all(not entry.exists() for entry in observed_spool_paths)
        return real_sha256(data, *args, **kwargs)

    monkeypatch.setattr(t.hashlib, "sha256", sha256)

    published = []
    def publish(lease, destination):
        stage = Path(getattr(lease, "path", lease))
        assert sorted(path.name for path in stage.iterdir()) == sorted(_TASK4_FILES)
        assert spool_identity is not None
        assert all(not entry.exists() for entry in observed_spool_paths)
        published.append((stage, destination))
    monkeypatch.setattr(t, "_publish_staging", publish)

    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 0
    assert len(calls) == 2 and len(published) == 1
    assert observed_spool_paths and all(not path.exists() for path in observed_spool_paths)


@pytest.mark.parametrize("seam", ("runtime", "writer", "index"))
def test_teacher_staged_case_spool_failure_is_transactional(tmp_path, monkeypatch, seam):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    _task4_fake_runtime(t, monkeypatch, outcome_factory=_task4_p1c_mutation_outcome)
    stages = []; real_staging = t._new_staging
    monkeypatch.setattr(
        t, "_new_staging",
        lambda destination: (stages.append(Path(real_staging(destination))) or stages[-1]),
    )
    sentinel = RuntimeError(f"{seam} replay failure")
    spool_identity = None
    observed_spool_paths = set()

    def private_spool():
        nonlocal spool_identity
        assert stages
        spool_identity, entries = _task4_private_case_spool(stages[-1], spool_identity)
        observed_spool_paths.update(entries)

    if seam == "runtime":
        runtime = t._runtime_hooks()
        def process(_case_input):
            private_spool()
            raise sentinel
        monkeypatch.setattr(t, "_runtime_hooks", lambda: dataclasses.replace(runtime, process_case=process))
    elif seam == "writer":
        real_write = t._JsonlWriter.write
        def write(writer, name, value):
            private_spool()
            real_write(writer, name, value)
            raise sentinel
        monkeypatch.setattr(t._JsonlWriter, "write", write)
    else:
        import builtins
        real_open = builtins.open

        class FailingIndexWriter:
            def __init__(self, handle):
                self._handle = handle

            def write(self, data):
                private_spool()
                self._handle.write(data)
                raise sentinel

            def __enter__(self):
                self._handle.__enter__()
                return self

            def __exit__(self, *args):
                return self._handle.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._handle, name)

        def failing_index_open(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            if "w" in mode and Path(path).name == "training_index.json":
                return FailingIndexWriter(handle)
            return handle

        monkeypatch.setattr(t, "open", failing_index_open, raising=False)

    with pytest.raises(RuntimeError, match=rf"{seam} replay failure"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert not out.exists() and stages and not stages[-1].exists()
    assert spool_identity is not None
    assert observed_spool_paths and all(not path.exists() for path in observed_spool_paths)
    assert not _task4_stage_fd_targets(stages[-1]), (
        "transactional failure must close every descriptor into the removed staging tree"
    )


def _task4_spooled_replay_record(source_path, source_row_index):
    """The semantic replay binding, deliberately independent of private storage."""
    case = _task4_expected_case(source_row_index)
    file_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    fingerprint = fingerprint_case(case)
    return {
        "worker": 2,
        "layout": 0,
        "relative_path": "worker_2/layouts_0.th",
        "file_sha256": file_sha256,
        "source_row_count": 2,
        "source_row_index": source_row_index,
        "instance_id": f"worker_2/layouts_0.th#{source_row_index}",
        "case_json": _task4_canonical_json(case).decode("ascii"),
        "case": copy.deepcopy(case),
        "fingerprint": fingerprint,
        "receipt": CorpusSourceReceipt("worker_2/layouts_0.th", file_sha256,
                                        source_row_index, fingerprint),
    }


def _task4_trusted_shard_summary(source_path, *, worker=2, layout=0, source_row_count=2):
    """Verified-ingestion facts replay must use, never candidate-row metadata."""
    return {
        "worker": worker,
        "layout": layout,
        "relative_path": f"worker_{worker}/layouts_{layout}.th",
        "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_row_count": source_row_count,
    }


def _task4_spooled_replay_leaves(record):
    receipt = record["receipt"]
    return {
        "worker": record["worker"],
        "layout": record["layout"],
        "relative_path": record["relative_path"],
        "file_sha256": record["file_sha256"],
        "source_row_count": record["source_row_count"],
        "source_row_index": record["source_row_index"],
        "instance_id": record["instance_id"],
        "case_json": record["case_json"],
        "case": record["case"],
        "fingerprint": record["fingerprint"],
        "receipt.relative_path": receipt.relative_path,
        "receipt.file_sha256": receipt.file_sha256,
        "receipt.layout_index": receipt.layout_index,
        "receipt.fingerprint": receipt.fingerprint,
    }


def _task4_normalized_spooled_validation(value):
    """Replay consumes the validator's compact, explicit (case, receipt) result."""
    assert isinstance(value, tuple) and len(value) == 2
    case, receipt = value
    assert isinstance(case, collections.abc.Mapping)
    assert type(receipt) is CorpusSourceReceipt
    return case, receipt


def _task4_semantic_value(value, name):
    return value[name] if isinstance(value, collections.abc.Mapping) else getattr(value, name)


def _task4_summary_values(summary):
    return {name: _task4_semantic_value(summary, name) for name in (
        "worker", "layout", "relative_path", "file_sha256", "source_row_count",
    )}


def _task4_replace_summary_value(summary, name, replacement):
    if isinstance(summary, collections.abc.Mapping):
        changed = dict(summary)
        changed[name] = replacement
        return changed
    if dataclasses.is_dataclass(summary):
        return dataclasses.replace(summary, **{name: replacement})
    raise TypeError("semantic shard summary must be a mapping or dataclass")


@pytest.mark.parametrize("poison, changed_leaf", [
    ("worker", "worker"),
    ("layout", "layout"),
    ("relative_path", "relative_path"),
    ("source_row_index_in_range", "source_row_index"),
    ("source_row_index_out_of_range", "source_row_index"),
    ("source_row_count", "source_row_count"),
    ("fingerprint", "fingerprint"),
    ("receipt_fingerprint", "receipt.fingerprint"),
    ("receipt_relative_path", "receipt.relative_path"),
    ("instance_id", "instance_id"),
    ("case_payload", "case"),
    ("case_instance_id", "case"),
    ("case_json_reordered", "case_json"),
    ("case_json_whitespace", "case_json"),
    ("case_json_trailing_newline", "case_json"),
    ("case_json_non_ascii", "case_json"),
    ("malformed_file_sha256", "file_sha256"),
    ("wrong_file_sha256", "file_sha256"),
    ("malformed_receipt_sha256", "receipt.file_sha256"),
    ("wrong_receipt_sha256", "receipt.file_sha256"),
])
def test_teacher_spooled_row_validator_rejects_spoofed_semantic_metadata(
        tmp_path, poison, changed_leaf):
    """The real replay validator rejects every mismatched sealed binding."""
    t = _teacher()
    validator = getattr(t, "_validate_spooled_case_row", None)
    assert callable(validator), "missing real _validate_spooled_case_row semantic replay seam"
    source_path = _task4_shard(tmp_path / "floorset_lite")
    summary = _task4_trusted_shard_summary(source_path)
    pristine_records = [_task4_spooled_replay_record(source_path, index) for index in range(2)]
    for pristine in pristine_records:
        assert pristine["case_json"].isascii()
        assert _task4_normalized_spooled_validation(
            validator(copy.deepcopy(pristine), copy.deepcopy(summary))) == (
            pristine["case"], pristine["receipt"]
        )
    pristine = pristine_records[0]
    spoofed = copy.deepcopy(pristine)
    if poison == "worker":
        spoofed["worker"] = 99
    elif poison == "layout":
        spoofed["layout"] = 7
    elif poison == "relative_path":
        spoofed["relative_path"] = "worker_2/layouts_7.th"
    elif poison == "source_row_index_in_range":
        spoofed["source_row_index"] = 1
    elif poison == "source_row_index_out_of_range":
        spoofed["source_row_index"] = 2
    elif poison == "source_row_count":
        # Row zero remains in range, but its sealed source has two rows.
        spoofed["source_row_count"] = 1
    elif poison == "fingerprint":
        spoofed["fingerprint"] = "0" * 64
    elif poison == "receipt_fingerprint":
        spoofed["receipt"] = dataclasses.replace(spoofed["receipt"], fingerprint="0" * 64)
    elif poison == "receipt_relative_path":
        spoofed["receipt"] = dataclasses.replace(
            spoofed["receipt"], relative_path="worker_2/layouts_7.th")
    elif poison == "instance_id":
        spoofed["instance_id"] = "worker_2/layouts_0.th#99"
    elif poison == "case_payload":
        spoofed["case"]["area"][0] += 0.25
    elif poison == "case_instance_id":
        spoofed["case"]["instance_id"] = "worker_2/layouts_0.th#99"
    elif poison == "case_json_reordered":
        spoofed["case_json"] = json.dumps(spoofed["case"], sort_keys=False,
                                           separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    elif poison == "case_json_whitespace":
        spoofed["case_json"] = json.dumps(spoofed["case"], sort_keys=True, indent=1,
                                           ensure_ascii=True, allow_nan=False)
    elif poison == "case_json_trailing_newline":
        spoofed["case_json"] += "\n"
    elif poison == "case_json_non_ascii":
        spoofed["case_json"] = spoofed["case_json"].replace("worker_2", "wørker_2", 1)
    elif poison == "malformed_file_sha256":
        spoofed["file_sha256"] = "not-a-sha256"
    elif poison == "wrong_file_sha256":
        spoofed["file_sha256"] = "0" * 64
    elif poison == "malformed_receipt_sha256":
        spoofed["receipt"] = dataclasses.replace(spoofed["receipt"], file_sha256="not-a-sha256")
    else:
        spoofed["receipt"] = dataclasses.replace(spoofed["receipt"], file_sha256="0" * 64)
    if poison in {"case_json_reordered", "case_json_whitespace", "case_json_trailing_newline"}:
        assert json.loads(spoofed["case_json"]) == pristine["case"]
    assert [name for name, value in _task4_spooled_replay_leaves(pristine).items()
            if value != _task4_spooled_replay_leaves(spoofed)[name]] == [changed_leaf]
    with pytest.raises(ValueError):
        validator(spoofed, copy.deepcopy(summary))


def test_teacher_replay_uses_real_spooled_row_validator_immediately_before_runtime(
        tmp_path, monkeypatch):
    """Replay alternates one real validation result with its matching runtime call."""
    t = _teacher()
    validator = getattr(t, "_validate_spooled_case_row", None)
    assert callable(validator), "missing real _validate_spooled_case_row semantic replay seam"
    root = tmp_path / "floorset_lite"; source_path = _task4_shard(root); out = tmp_path / "out"
    expected = [_task4_spooled_replay_record(source_path, index) for index in range(2)]
    expected_summary = _task4_trusted_shard_summary(source_path)
    events = []; calls = []

    def wrapped_validator(record, shard_summary):
        summary = {name: _task4_semantic_value(shard_summary, name)
                   for name in expected_summary}
        assert summary == expected_summary
        for name, value in summary.items():
            assert _task4_semantic_value(record, name) == value
        result = validator(record, shard_summary)
        normalized = _task4_normalized_spooled_validation(result)
        events.append(("validate", _task4_semantic_value(record, "instance_id"), summary,
                       copy.deepcopy(normalized[0]), normalized[1]))
        return result

    monkeypatch.setattr(t, "_validate_spooled_case_row", wrapped_validator)
    _task4_fake_runtime(t, monkeypatch, calls=calls,
                        outcome_factory=_task4_p1c_mutation_outcome)
    runtime = t._runtime_hooks(); process_case = runtime.process_case

    def wrapped_process_case(case_input):
        events.append(("process", case_input.case["instance_id"],
                       copy.deepcopy(case_input.case), case_input.receipt))
        return process_case(case_input)

    monkeypatch.setattr(t, "_runtime_hooks",
                        lambda: dataclasses.replace(runtime, process_case=wrapped_process_case))
    t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert [(event[0], event[1]) for event in events] == [
        (kind, expected[index]["instance_id"])
        for index in range(2) for kind in ("validate", "process")
    ]
    assert len(calls) == len(expected)
    for index, expected_record in enumerate(expected):
        validate_event, process_event = events[2 * index:2 * index + 2]
        assert validate_event[2:] == (
            expected_summary, expected_record["case"], expected_record["receipt"]
        )
        assert process_event[2:] == validate_event[3:]


def test_teacher_verified_shard_summary_binds_raw_source_and_numeric_coordinates(tmp_path):
    """Verified ingestion produces a replay summary from bytes and source, not a row."""
    t = _teacher()
    builder = getattr(t, "_verified_shard_summary", None)
    assert callable(builder), "missing real _verified_shard_summary semantic ingestion seam"
    root = tmp_path / "floorset_lite"
    sources = [
        (2, 0, _task4_shard(root)),
        (2, 1, _task4_single_shard(root, "worker_2/layouts_1.th")),
    ]
    for worker, layout, path in sources:
        raw, source = t._read_verified_shard(root, worker, layout)
        source_row_count, _blocks = t._validate_source_shard(source)
        summary = builder(raw, source, worker, layout)
        assert raw == path.read_bytes()
        assert _task4_summary_values(summary) == _task4_trusted_shard_summary(
            path, worker=worker, layout=layout, source_row_count=source_row_count
        )


def test_teacher_replay_rejects_builder_returned_count_not_matching_spooled_row(
        tmp_path, monkeypatch):
    """A replaceable summary cannot rewrite an independently decoded replay record."""
    t = _teacher()
    builder = getattr(t, "_verified_shard_summary", None)
    validator = getattr(t, "_validate_spooled_case_row", None)
    assert callable(builder), "missing real _verified_shard_summary semantic ingestion seam"
    assert callable(validator), "missing real _validate_spooled_case_row semantic replay seam"
    root = tmp_path / "floorset_lite"; source_path = _task4_shard(root); out = tmp_path / "out"
    expected_summary = _task4_trusted_shard_summary(source_path)
    calls = []
    _task4_fake_runtime(t, monkeypatch, calls=calls,
                        outcome_factory=_task4_p1c_mutation_outcome)
    stages = []; real_staging = t._new_staging
    monkeypatch.setattr(
        t, "_new_staging",
        lambda destination: (stages.append(Path(real_staging(destination))) or stages[-1]),
    )
    real_builder = builder
    returned_summaries = []

    def return_wrong_count(raw, source, worker, layout):
        verified = real_builder(raw, source, worker, layout)
        changed = _task4_replace_summary_value(verified, "source_row_count", 1)
        returned_summaries.append((verified, changed))
        return changed

    monkeypatch.setattr(t, "_verified_shard_summary", return_wrong_count)
    real_validator = validator
    validation_calls = []

    def validate_record(record, shard_summary):
        validation_calls.append((copy.deepcopy(record), copy.deepcopy(shard_summary)))
        return real_validator(record, shard_summary)

    monkeypatch.setattr(t, "_validate_spooled_case_row", validate_record)
    with pytest.raises(ValueError):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert len(returned_summaries) == len(validation_calls) == 1
    pristine_summary, changed_summary = returned_summaries[0]
    assert _task4_summary_values(pristine_summary) == expected_summary
    assert [name for name, value in _task4_summary_values(pristine_summary).items()
            if value != _task4_summary_values(changed_summary)[name]] == ["source_row_count"]
    record, passed_summary = validation_calls[0]
    assert _task4_semantic_value(record, "source_row_count") == 2
    assert _task4_summary_values(passed_summary) == _task4_summary_values(changed_summary)
    assert calls == []
    assert not out.exists() and stages and not stages[-1].exists()


def test_teacher_publish_failure_inspects_complete_staging_and_leaves_no_destination(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    policy = _policy_for(root); calls = []; preflight_calls = []
    _task4_fake_runtime(t, monkeypatch, calls=calls, preflight_calls=preflight_calls)
    inspected = {}

    def fail_publish(staging, destination):
        stage_path = getattr(staging, "path", staging)
        assert destination == out and stage_path.is_dir()
        _task4_assert_success_artifacts(t, root, stage_path, calls, policy, preflight_calls)
        inspected["ok"] = True
        raise _Task4PublishFailure("publish failure")

    monkeypatch.setattr(t, "_publish_staging", fail_publish, raising=False)
    with pytest.raises(_Task4PublishFailure, match="publish failure"):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert inspected == {"ok": True} and not out.exists()


def test_teacher_spool_ingests_all_shards_before_ordered_runtime_replay(tmp_path, monkeypatch):
    """P1-C globally registers before replaying one spooled case at a time."""
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    heldout_mod = 2
    _task4_shard(root, relative_path="worker_2/layouts_0.th")
    _task4_shard(root, relative_path="worker_2/layouts_2.th", metric_delta=2)
    _task4_shard(root, relative_path="worker_2/layouts_10.th", metric_delta=10)
    calls = []; events = []
    _task4_fake_runtime(
        t, monkeypatch, calls=calls, events=events,
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    created = []
    real_new_staging = t._new_staging

    def new_staging(destination):
        stage = real_new_staging(destination); created.append(stage); return stage

    reader_calls = []; real_read = t._read_verified_shard
    def read(root_path, worker, layout):
        reader_calls.append((worker, layout))
        return real_read(root_path, worker, layout)

    real_load = torch.load
    def load(source, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert kwargs == {"weights_only": True, "map_location": "cpu"}
        assert created, "staging must exist before any shard is loaded"
        stage = created[-1]
        assert all((stage / name).exists() for name in _TASK4_JSONL)
        events.append(("load", source.getvalue()))
        return real_load(source, **kwargs)

    real_register = t._PopulationAccumulator.register
    def register(population, row):
        events.append(("register", row["instance_id"]))
        return real_register(population, row)

    real_write = t._JsonlWriter.write
    def write(writer, name, value):
        events.append(("write", name, value.get("instance_id")))
        return real_write(writer, name, value)

    monkeypatch.setattr(t, "_new_staging", new_staging)
    monkeypatch.setattr(t, "_read_verified_shard", read)
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(t._PopulationAccumulator, "register", register)
    monkeypatch.setattr(t._JsonlWriter, "write", write)
    assert t.teacher_main(
        _task4_args(root, out, heldout_mod=heldout_mod), _trust_policy=_policy_for(root)
    ) == 0

    expected_paths = ["worker_2/layouts_0.th", "worker_2/layouts_2.th",
                      "worker_2/layouts_10.th"]
    assert reader_calls == [(2, 0), (2, 2), (2, 10)]
    assert [event[1] for event in events if event[0] == "load"] == [
        (root / relative_path).read_bytes() for relative_path in expected_paths]
    expected_all = [
        f"{relative_path}#{index}"
        for relative_path in expected_paths
        for index in range(2)
    ]
    heldout_ids = [
        instance_id for instance_id in expected_all
        if split_for_id(instance_id, heldout_mod) == "heldout"
    ]
    assert [event[1] for event in events if event[0] == "register"] == heldout_ids
    first_process = next(index for index, event in enumerate(events) if event[0] == "process")
    assert all(index < first_process for index, event in enumerate(events)
               if event[0] in {"load", "register"})
    first_write = next(index for index, event in enumerate(events) if event[0] == "write")
    assert first_process < first_write

    assert [case.case["instance_id"] for case in calls] == expected_all
    assert len(calls) == len(expected_all) == len({case.case["instance_id"] for case in calls})
    assert {case.case["instance_id"] for case in calls} == set(expected_all)
    for name in _TASK4_JSONL:
        rows = _task4_jsonl(out / name)
        assert (out / name).read_bytes() == b"".join(_task4_canonical_json(row) + b"\n" for row in rows)
    expected_outcomes = [_task4_p1c_mutation_outcome(t, case_input) for case_input in calls]
    expected_proposals = [
        {
            **_task4_envelope(
                case_input.case, dataclasses.asdict(case_input.receipt),
                case_input.partition, case_input.sample_seed,
            ),
            **dict(proposal),
        }
        for case_input, outcome in zip(calls, expected_outcomes)
        for proposal in sorted(
            outcome.proposal_rows,
            key=lambda row: (row["ordinal"], row["name"]),
        )
    ]
    actual_proposals = _task4_jsonl(out / "proposals.jsonl")
    assert actual_proposals == expected_proposals
    assert len(actual_proposals) == 2 * len(calls)
    assert len(actual_proposals) == sum(
        len(outcome.proposal_rows) for outcome in expected_outcomes
    )
    expected_rejections = [
        {
            **_task4_envelope(
                case_input.case, dataclasses.asdict(case_input.receipt),
                case_input.partition, case_input.sample_seed,
            ),
            **dict(rejection),
        }
        for case_input, outcome in zip(calls, expected_outcomes)
        for rejection in sorted(
            outcome.rejection_rows,
            key=lambda row: (row.get("ordinal", 0), row.get("name", "")),
        )
    ]
    actual_rejections = _task4_jsonl(out / "rejections.jsonl")
    assert actual_rejections == expected_rejections
    assert len(actual_rejections) == sum(
        len(outcome.rejection_rows) for outcome in expected_outcomes
    )
    train_ids = [c.case["instance_id"] for c in calls if c.partition == "train"]
    held_ids = [c.case["instance_id"] for c in calls if c.partition == "heldout"]
    assert [r["instance_id"] for r in _task4_jsonl(out / "train_corpus.jsonl")] == train_ids
    assert [r["instance_id"] for r in _task4_jsonl(out / "heldout_corpus.jsonl")] == held_ids
    assert [r["instance_id"] for r in _task4_jsonl(out / "train_labels.jsonl")] == train_ids
    assert [r["instance_id"] for r in _task4_jsonl(out / "heldout_labels.jsonl")] == held_ids
    assert [r["instance_id"] for r in actual_proposals] == [
        row["instance_id"] for row in expected_proposals
    ]
    assert [r["instance_id"] for r in actual_rejections] == [
        row["instance_id"] for row in expected_rejections
    ]
    assert len(actual_proposals) == len(expected_proposals)
    assert len(actual_rejections) == len(expected_rejections)


def test_teacher_streaming_b1_releases_prior_source_before_next_shard(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_2/layouts_0.th")
    _task4_shard(root, relative_path="worker_2/layouts_2.th")
    _task4_fake_runtime(t, monkeypatch)
    refs = []; reads = 0; real_read = t._read_verified_shard
    def read(*args):
        nonlocal reads
        if reads:
            gc.collect()
            assert all(ref() is None for ref in refs[-1])
        raw, source = real_read(*args)
        refs.append(tuple(weakref.ref(tensor) for tensor in source)); reads += 1
        return raw, source
    monkeypatch.setattr(t, "_read_verified_shard", read)
    t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))


def test_teacher_streaming_b1_teacher_main_has_no_row_sequence_accumulators():
    tree = ast.parse(Path("scripts/probes/icdc_topology_teacher.py").read_text())
    teacher = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "teacher_main")
    forbidden = {"train_c", "held_c", "train_l", "held_l", "proposals", "rejections"}
    assert not ({node.id for node in ast.walk(teacher) if isinstance(node, ast.Name)} & forbidden)
    assert not ({node.arg for node in ast.walk(teacher) if isinstance(node, ast.arg)} & forbidden)

    def assigned_names(target):
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(assigned_names(element) for element in target.elts))
        return set()

    def is_mutable_container(expression):
        if isinstance(expression, (ast.List, ast.Dict, ast.Set,
                                   ast.ListComp, ast.DictComp, ast.SetComp)):
            return True
        if not isinstance(expression, ast.Call):
            return False
        if isinstance(expression.func, ast.Name):
            return expression.func.id in {"list", "dict", "set", "defaultdict", "deque"}
        return (isinstance(expression.func, ast.Attribute)
                and expression.func.attr in {"defaultdict", "deque"})

    def annotation_is_mutable_container(annotation):
        return any(isinstance(node, ast.Name) and node.id in {"list", "dict", "set",
                                                               "List", "Dict", "Set"}
                   for node in ast.walk(annotation))

    def annotation_is_scalar(annotation):
        return (isinstance(annotation, ast.Name)
                and annotation.id in {"int", "float", "bool"})

    def scalar_expression(expression):
        if isinstance(expression, ast.Constant):
            return type(expression.value) in {int, float, bool, type(None)}
        return (isinstance(expression, ast.UnaryOp)
                and isinstance(expression.op, (ast.UAdd, ast.USub))
                and isinstance(expression.operand, ast.Constant)
                and type(expression.operand.value) in {int, float})

    def payload_expression(expression):
        if isinstance(expression, (ast.List, ast.Tuple, ast.Dict, ast.Set,
                                   ast.ListComp, ast.Tuple, ast.DictComp, ast.SetComp,
                                   ast.JoinedStr)):
            return True
        if isinstance(expression, ast.Constant):
            return isinstance(expression.value, (str, bytes, bytearray))
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                return expression.func.id in {
                    "list", "tuple", "dict", "set", "bytes", "bytearray", "str",
                    "defaultdict", "deque",
                }
            return (isinstance(expression.func, ast.Attribute)
                    and expression.func.attr in {"defaultdict", "deque"})
        return False

    def contains_name(expression, name):
        return any(isinstance(child, ast.Name) and child.id == name
                   for child in ast.walk(expression))

    replay_candidates = [
        node for node in ast.walk(teacher) if isinstance(node, ast.For)
        and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                and child.func.attr == "process_case" for child in ast.walk(node))
    ]
    replay_loops = [
        node for node in replay_candidates
        if not any(node is not child and isinstance(child, ast.For)
                   and child in replay_candidates for child in ast.walk(node))
    ]
    assert len(replay_loops) == 1
    replay_nodes = {id(node) for node in ast.walk(replay_loops[0])}
    persistent_mutables = {}
    outer_bindings = {}
    for node in ast.walk(teacher):
        if id(node) in replay_nodes:
            continue
        if isinstance(node, ast.AnnAssign):
            names = assigned_names(node.target)
            is_mutable = (annotation_is_mutable_container(node.annotation)
                          or (node.value is not None and is_mutable_container(node.value)))
            scalar = annotation_is_scalar(node.annotation) or (
                node.value is not None and scalar_expression(node.value))
            payload = (not scalar and (annotation_is_mutable_container(node.annotation)
                       or (node.value is not None and payload_expression(node.value))))
        elif isinstance(node, ast.Assign):
            names = set().union(*(assigned_names(target) for target in node.targets))
            is_mutable = is_mutable_container(node.value)
            scalar = scalar_expression(node.value)
            payload = not scalar and payload_expression(node.value)
        else:
            continue
        outer_bindings.update({
            name: {"scalar": scalar, "payload": payload, "line": node.lineno}
            for name in names
        })
        if is_mutable:
            persistent_mutables.update({name: node.lineno for name in names})
    allowed_persistent_mutables = {"files", "rejection_counts", "coverage", "manifest"}
    assert set(persistent_mutables) <= allowed_persistent_mutables, (
        "teacher_main may not retain a mutable corpus/index container across replay; "
        f"found {persistent_mutables}"
    )

    def mutable_receiver(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id
        return None

    def numeric_increment(expression):
        return (isinstance(expression, ast.Constant)
                and type(expression.value) in {int, float}
                and not isinstance(expression.value, bool))

    def numeric_expression(expression):
        if isinstance(expression, ast.Constant):
            return type(expression.value) in {int, float} and not isinstance(expression.value, bool)
        if isinstance(expression, ast.UnaryOp):
            return isinstance(expression.op, (ast.UAdd, ast.USub)) and numeric_expression(expression.operand)
        if isinstance(expression, ast.BinOp):
            return (isinstance(expression.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                                                ast.Mod, ast.Pow))
                    and numeric_expression(expression.left)
                    and numeric_expression(expression.right))
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                return expression.func.id in {"abs", "float", "int", "len", "max", "min", "round", "sum"}
            return isinstance(expression.func, ast.Attribute) and expression.func.attr in {"exp", "log", "sqrt"}
        return False

    def scalar_update_expression(expression):
        return (numeric_expression(expression)
                or (isinstance(expression, ast.BoolOp)
                    and isinstance(expression.op, (ast.And, ast.Or))))

    def fixed_counter_increment(node):
        if isinstance(node, ast.AugAssign):
            return (isinstance(node.target, ast.Subscript)
                    and mutable_receiver(node.target) == "rejection_counts"
                    and isinstance(node.op, ast.Add)
                    and numeric_increment(node.value))
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and mutable_receiver(node.targets[0]) == "rejection_counts"
                and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
                and numeric_increment(node.value.right)
                and isinstance(node.value.left, ast.Subscript)):
            return False
        return ast.dump(node.value.left) == ast.dump(node.targets[0])

    replay_mutations = []
    for node in ast.walk(replay_loops[0]):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "extend", "add", "update", "setdefault",
                                       "__setitem__"}):
            name = mutable_receiver(node.func.value)
            if name is not None:
                replay_mutations.append((name, node.lineno, node.func.attr))
            continue
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript) and fixed_counter_increment(node):
                continue
            name = mutable_receiver(node.target)
            if name == "rejection_counts":
                replay_mutations.append((name, node.lineno, "augassign"))
            elif name is not None and not numeric_expression(node.value):
                replay_mutations.append((name, node.lineno, "augassign"))
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if not isinstance(node, ast.Assign) else node.targets[0]
            value = node.value
            if isinstance(target, ast.Subscript):
                if not fixed_counter_increment(node):
                    replay_mutations.append((mutable_receiver(target), node.lineno, "subscript"))
                continue
            name = mutable_receiver(target)
            if name == "rejection_counts":
                replay_mutations.append((name, node.lineno, "rebind"))
            elif (name is not None and name in outer_bindings
                  and contains_name(value, name)
                  and not (outer_bindings[name]["scalar"]
                           and scalar_update_expression(value))):
                replay_mutations.append((name, node.lineno, "self-reference"))
            continue
    assert not replay_mutations, (
        "replay may write only sinks and fixed counters, not retain or grow local state: "
        f"{replay_mutations}"
    )


def test_teacher_streaming_b1_index_rows_are_emitted_during_replay(tmp_path, monkeypatch):
    """A later runtime case proves the earlier index row was already streamed."""
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root)
    calls = []
    _task4_fake_runtime(t, monkeypatch, calls=calls,
                        outcome_factory=_task4_p1c_mutation_outcome)
    import builtins
    real_open = builtins.open
    index_writes = []

    class IndexWriterSpy:
        def __init__(self, handle):
            self._handle = handle

        def write(self, data):
            index_writes.append(data if isinstance(data, bytes) else data.encode("utf-8"))
            return self._handle.write(data)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def streaming_index_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "w" in mode and Path(path).name.startswith("training_index"):
            return IndexWriterSpy(handle)
        return handle

    monkeypatch.setattr(t, "open", streaming_index_open, raising=False)
    runtime = t._runtime_hooks(); real_process = runtime.process_case

    def process(case_input):
        if calls:
            first = calls[0]
            first_index_row = {
                "receipt": dataclasses.asdict(first.receipt),
                "instance_id": first.case["instance_id"],
                "source_row_count": 2,
                "block_count": first.case["n"],
                "partition": first.partition,
                "sample_ordinal": 0,
                "sample_seed": first.sample_seed,
                "status": "winner_mutation",
            }
            assert _task4_canonical_json(first_index_row) in b"".join(index_writes), (
                "the complete canonical first row must reach the streaming index "
                "before replay continues"
            )
        return real_process(case_input)

    monkeypatch.setattr(t, "_runtime_hooks",
                        lambda: dataclasses.replace(runtime, process_case=process))
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 0
    assert index_writes
    assert json.loads((out / "training_index.json").read_text())["rows"]


def test_teacher_streaming_b1_large_replay_does_not_retain_index_rows(tmp_path, monkeypatch):
    """Large source identities make any retained corpus/index roster observable in RAM."""
    t = _teacher(); root = tmp_path / "floorset_lite"; root.mkdir(); out = tmp_path / "out"
    source = _task4_tensors()
    files = []
    for index in range(256):
        worker_text = f"{index + 1:04000d}"
        files.append((int(worker_text), 0,
                      root / f"worker_{worker_text}" / "layouts_0.th"))
    monkeypatch.setattr(t, "_iter_approved_shards", lambda _root: files)
    monkeypatch.setattr(t, "_read_verified_shard", lambda *_args: (b"verified", source))
    processed = 0
    observed_delta = None

    def preflight(policy, checkpoint):
        return {
            "trust_ok": True, "input_ok": True, "scorer_ok": True,
            "checkpoint_sha256": policy.expected_checkpoint_sha256,
            "model_identity": dict(policy.allowed_model_identity),
            "scorer_sha256": policy.expected_scorer_sha256,
            "scorer_contract": policy.scorer_contract,
            "shapely_version": policy.shapely_version,
        }

    def process(case_input):
        nonlocal processed, observed_delta
        processed += 1
        if processed == 384:
            gc.collect()
            observed_delta = tracemalloc.get_traced_memory()[0] - baseline
        return _task4_good_outcome(t, case_input)

    runtime_type = getattr(t, "_TeacherRuntime", _Task4Runtime)
    monkeypatch.setattr(
        t, "_runtime_hooks", lambda: runtime_type(preflight, process, True), raising=False,
    )
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) in (0, 1)
        assert processed == 2 * len(files)
        assert observed_delta is not None and observed_delta < 2 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_teacher_streaming_b1_durability_precedes_publish(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_0/layouts_0.th")
    _task4_fake_runtime(t, monkeypatch)
    staging_paths = []
    real_new_staging = t._new_staging
    def new_staging(destination):
        stage = real_new_staging(destination); staging_paths.append(Path(getattr(stage, "path", stage))); return stage
    monkeypatch.setattr(t, "_new_staging", new_staging)
    events = []; published = []
    real_fsync = t.os.fsync
    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        mode = t.os.fstat(fd).st_mode
        stage = staging_paths[-1]
        if stat.S_ISDIR(mode):
            manifest_exists = (path / "g0_manifest.json").exists()
            if not manifest_exists:
                assert sorted(child.name for child in path.iterdir()) == sorted(
                    set(_TASK4_FILES) - {"g0_manifest.json"})
            events.append(("dir", path.resolve(), manifest_exists))
        else:
            events.append((path.name, path.parent.resolve()))
        return real_fsync(fd)
    monkeypatch.setattr(t.os, "fsync", fsync)
    def publish(staging, destination):
        stage = Path(getattr(staging, "path", staging))
        manifest = json.loads((stage / "g0_manifest.json").read_text())
        assert manifest["support_hashes"] == {
            name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
            for name in _TASK4_FILES if name != "g0_manifest.json"}
        assert manifest["self_sha256"] == t._manifest_self_sha256(manifest)
        published.append((stage, destination)); events.append("publish")
    monkeypatch.setattr(t, "_publish_staging", publish)
    t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert staging_paths
    stage = staging_paths[-1].resolve()
    assert events == [*((name, stage) for name in _TASK4_JSONL),
                      ("training_index.json", stage), ("dir", stage, False),
                      ("g0_manifest.json", stage), ("dir", stage, True), "publish"]


@pytest.mark.parametrize("fail_target", ["train_corpus.jsonl", "training_index.json",
                                           "dir_pre", "g0_manifest.json", "dir_post"])
def test_teacher_streaming_b1_durability_failure_is_transactional(tmp_path, monkeypatch, fail_target):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_0/layouts_0.th")
    _task4_fake_runtime(t, monkeypatch)
    staging_paths = []; real_new_staging = t._new_staging
    def new_staging(destination):
        stage = real_new_staging(destination); staging_paths.append(Path(getattr(stage, "path", stage))); return stage
    monkeypatch.setattr(t, "_new_staging", new_staging)
    sentinel = OSError("durability sentinel"); real_fsync = t.os.fsync
    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}")); mode = t.os.fstat(fd).st_mode
        stage = staging_paths[-1].resolve() if staging_paths else None
        owned = path.resolve() == stage if stat.S_ISDIR(mode) else path.parent.resolve() == stage
        target = ("dir_pre" if owned and stat.S_ISDIR(mode) and not (path / "g0_manifest.json").exists()
                  else "dir_post" if owned and stat.S_ISDIR(mode) else path.name if owned else None)
        if target == fail_target: raise sentinel
        return real_fsync(fd)
    monkeypatch.setattr(t.os, "fsync", fsync)
    published = []; monkeypatch.setattr(t, "_publish_staging", lambda *args: published.append(args))
    with pytest.raises(OSError) as exc:
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert exc.value is sentinel
    assert published == [] and not out.exists()
    assert staging_paths and not staging_paths[-1].exists()


def test_teacher_streaming_b1_review_population_is_bounded_exact_and_ordered():
    t = _teacher()
    rows = [
        {"relative_path": "worker_2/layouts_0.th", "layout_index": 0, "instance_id": "worker_2/layouts_0.th#0", "n": 4, "base_cost": 8.5, "teacher_cost": 7.25},
        {"relative_path": "worker_10/layouts_0.th", "layout_index": 0, "instance_id": "worker_10/layouts_0.th#0", "n": 12, "base_cost": 3.5, "teacher_cost": 2.25},
    ]
    expected = _task4_p1c_canonical_population(rows)
    acc = t._PopulationAccumulator()
    for row in rows:
        acc.register({key: row[key] for key in ("relative_path", "layout_index", "instance_id", "n")})
        acc.record_winner(row["instance_id"], row["base_cost"], row["teacher_cost"])
    assert acc.finish() == expected
    reversed_acc = t._PopulationAccumulator()
    for row in reversed(rows):
        reversed_acc.register({key: row[key] for key in ("relative_path", "layout_index", "instance_id", "n")})
        reversed_acc.record_winner(row["instance_id"], row["base_cost"], row["teacher_cost"])
    assert reversed_acc.finish() == expected
    tracemalloc.start()
    try:
        bounded = t._PopulationAccumulator()
        bounded.register({**{key: rows[0][key] for key in ("relative_path", "layout_index", "instance_id", "n")}, "instance_id": "warmup"})
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        for index in range(2000):
            bounded.register({"relative_path": "worker_2/layouts_0.th", "layout_index": index + 1,
                              "instance_id": f"worker_2/layouts_0.th#{index}-" + ("x" * 4096),
                              "n": 4})
        gc.collect()
        assert tracemalloc.get_traced_memory()[0] - baseline < 2 * 1024 * 1024
    finally:
        bounded.abort()
        tracemalloc.stop()


def test_teacher_streaming_b1_population_roster_is_sqlite_only():
    tree = ast.parse(Path("scripts/probes/icdc_topology_teacher.py").read_text())
    population = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef)
                      and node.name == "_PopulationAccumulator")

    def self_attribute(node):
        return (node.attr if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "self" else None)

    def self_targets(target):
        attribute = self_attribute(target)
        if attribute is not None:
            return {attribute}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(self_targets(element) for element in target.elts))
        return set()

    def contains_self_attribute(expression, attribute):
        return any(self_attribute(node) == attribute for node in ast.walk(expression))

    def numeric_summary_expression(expression):
        if isinstance(expression, ast.Constant):
            return (type(expression.value) in {int, float}
                    and not isinstance(expression.value, bool))
        if isinstance(expression, ast.Attribute):
            return self_attribute(expression) in scalar_summaries
        if isinstance(expression, ast.UnaryOp):
            return (isinstance(expression.op, (ast.UAdd, ast.USub))
                    and numeric_summary_expression(expression.operand))
        if isinstance(expression, ast.BinOp):
            return (isinstance(expression.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                                ast.FloorDiv, ast.Mod, ast.Pow))
                    and numeric_summary_expression(expression.left)
                    and numeric_summary_expression(expression.right))
        return False

    sqlite_state = {"_db", "_db_path"}
    finished_snapshot = {"_finished"}
    scalar_summaries = {"count", "denominator", "base_total", "teacher_total"}
    allowed_state = sqlite_state | finished_snapshot | scalar_summaries
    assigned_state = []
    for node in ast.walk(population):
        if isinstance(node, ast.AnnAssign):
            assigned_state.extend((attribute, node.lineno)
                                  for attribute in self_targets(node.target))
        elif isinstance(node, ast.Assign):
            assigned_state.extend((attribute, node.lineno)
                                  for target in node.targets
                                  for attribute in self_targets(target))
        elif isinstance(node, ast.AugAssign):
            assigned_state.extend((attribute, node.lineno)
                                  for attribute in self_targets(node.target))
    assert {attribute for attribute, _ in assigned_state} <= allowed_state, (
        "the population roster may retain only SQLite state, its bounded terminal snapshot, "
        f"and finite summaries; found {assigned_state}"
    )

    growing_self_state = []
    for node in ast.walk(population):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add", "append", "extend", "update", "setdefault"}):
            receiver = node.func.value
            attr = self_attribute(receiver)
            if attr is None and isinstance(receiver, ast.Subscript):
                attr = self_attribute(receiver.value)
            if attr is not None:
                growing_self_state.append((attr, node.lineno, node.func.attr))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = ([node.target] if not isinstance(node, ast.Assign) else node.targets)
            for target in targets:
                if isinstance(target, ast.Subscript):
                    attr = self_attribute(target.value)
                    if attr is not None:
                        growing_self_state.append((attr, node.lineno, "subscript"))
        if isinstance(node, ast.AugAssign):
            attr = self_attribute(node.target)
            if attr is not None and (attr not in scalar_summaries
                                     or not numeric_summary_expression(node.value)):
                growing_self_state.append((attr, node.lineno, "augassign"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
            for target in targets:
                attr = self_attribute(target)
                if (attr is not None and contains_self_attribute(node.value, attr)
                        and (attr not in sqlite_state
                             and (attr not in scalar_summaries
                                  or not numeric_summary_expression(node.value)))):
                    growing_self_state.append((attr, node.lineno, "self-reference"))
    assert not growing_self_state, (
        "population identity/resolution state must be SQLite rows, not growing Python state: "
        f"{growing_self_state}"
    )

    methods = {node.name: node for node in population.body
               if isinstance(node, ast.FunctionDef)}
    expected_sql = {
        "register": "insert", "record_winner": "update", "finish": "select",
    }
    for name, operation in expected_sql.items():
        assert name in methods
        statements = [
            node.args[0].value.lower() for node in ast.walk(methods[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute" and node.args
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
        ]
        assert any(operation in statement and "population" in statement for statement in statements), (
            f"{name} must {operation} canonical population SQLite rows, not issue a dummy query"
        )


def test_teacher_streaming_b1_population_roster_register_and_winner_are_sqlite_rows():
    t = _teacher(); acc = t._PopulationAccumulator()
    identity = {"relative_path": "worker_2/layouts_0.th", "layout_index": 0,
                "instance_id": "worker_2/layouts_0.th#0", "n": 4}
    try:
        acc.register(identity)
        assert acc._db is not None
        registered = acc._db.execute(
            "SELECT relative_path, layout_index, instance_id, n "
            "FROM population WHERE instance_id = ?", (identity["instance_id"],)
        ).fetchone()
        assert registered is not None
        assert (registered[0], int(registered[1]), registered[2], int(registered[3])) == (
            identity["relative_path"], identity["layout_index"],
            identity["instance_id"], identity["n"],
        )
        acc.record_winner(identity["instance_id"], 8.5, 7.25)
        winner = acc._db.execute(
            "SELECT base_cost, teacher_cost FROM population WHERE instance_id = ?",
            (identity["instance_id"],),
        ).fetchone()
        assert winner == (8.5, 7.25)
    finally:
        acc.abort()


def test_teacher_streaming_b1_population_finish_uses_sqlite_roster_for_incomplete_result():
    t = _teacher(); acc = t._PopulationAccumulator()
    rows = [
        {"relative_path": "worker_2/layouts_0.th", "layout_index": 0,
         "instance_id": "worker_2/layouts_0.th#0", "n": 4},
        {"relative_path": "worker_2/layouts_1.th", "layout_index": 1,
         "instance_id": "worker_2/layouts_1.th#1", "n": 8},
        {"relative_path": "worker_10/layouts_0.th", "layout_index": 0,
         "instance_id": "worker_10/layouts_0.th#0", "n": 12},
    ]
    try:
        for row in rows:
            acc.register(row)
        acc.record_winner(rows[0]["instance_id"], 8.5, 7.25)
        assert acc._db is not None
        acc._db.execute("DELETE FROM population WHERE instance_id = ?", (rows[1]["instance_id"],))
        acc._db.commit()
        expected = _task4_p1c_canonical_population([
            {**rows[0], "base_cost": 8.5, "teacher_cost": 7.25}, rows[2],
        ])
        assert acc.finish() == expected
    finally:
        acc.abort()


@pytest.mark.parametrize("mutate", [
    lambda r: {**r, "extra": 1}, lambda r: {k: v for k, v in r.items() if k != "n"},
    lambda r: {**r, "layout_index": -1}, lambda r: {**r, "layout_index": True},
    lambda r: {**r, "instance_id": ""},
    lambda r: {**r, "n": -1}, lambda r: {**r, "n": True},
])
def test_teacher_streaming_b1_review_population_register_rejects_noncanonical_identity(mutate):
    t = _teacher()
    row = {"relative_path": "worker_2/layouts_0.th", "layout_index": 0,
           "instance_id": "worker_2/layouts_0.th#0", "n": 4}
    with pytest.raises(ValueError):
        t._PopulationAccumulator().register(mutate(row))


@pytest.mark.parametrize("n,base_cost,teacher_cost,weighted_product_overflows", [
    (10000, 8.5, 7.25, False), (4, float("nan"), 7.25, False),
    (4, float("inf"), 7.25, False), (4, 8.5, float("nan"), False),
    (4, 8.5, float("inf"), False), (8, 1e308, 7.25, True),
])
def test_teacher_streaming_b1_review_population_record_winner_rejects_overflow_and_nonfinite(
        n, base_cost, teacher_cost, weighted_product_overflows):
    t = _teacher(); acc = t._PopulationAccumulator()
    identity = {"relative_path": "worker_2/layouts_0.th", "layout_index": 0,
                "instance_id": "a", "n": n}
    acc.register(identity)
    if weighted_product_overflows:
        assert math.isfinite(math.exp(n / 12))
        assert not math.isfinite(math.exp(n / 12) * base_cost)
    with pytest.raises(ValueError):
        acc.record_winner("a", base_cost, teacher_cost)
    acc.abort()


def test_teacher_streaming_b1_review_population_rejects_duplicate_source_layout():
    t = _teacher(); acc = t._PopulationAccumulator()
    row = {"relative_path": "worker_2/layouts_0.th", "layout_index": 0, "instance_id": "a", "n": 4}
    acc.register(row)
    with pytest.raises(ValueError):
        acc.register({**row, "instance_id": "b"})
    acc.abort()


def test_teacher_streaming_b1_review_population_rejects_duplicate_instance():
    t = _teacher(); acc = t._PopulationAccumulator()
    row = {"relative_path": "worker_2/layouts_0.th", "layout_index": 0, "instance_id": "a", "n": 4}
    acc.register(row)
    with pytest.raises(ValueError):
        acc.register({**row, "relative_path": "worker_2/layouts_1.th", "layout_index": 1})
    with pytest.raises(ValueError):
        acc.record_winner("unknown", 8.5, 7.25)
    acc.record_winner("a", 8.5, 7.25)
    with pytest.raises(ValueError):
        acc.record_winner("a", 8.5, 7.25)
    acc.abort()


def test_teacher_streaming_b1_cleanup_population_spool_connect_failure_removes_all_artifacts(tmp_path, monkeypatch):
    t = _teacher()
    db_path = tmp_path / "population.sqlite"
    sentinel = RuntimeError("sqlite connect sentinel")

    class TempFile:
        name = str(db_path)
        def close(self):
            pass

    monkeypatch.setattr(t.tempfile, "NamedTemporaryFile", lambda **kwargs: TempFile())
    monkeypatch.setattr(t.sqlite3, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(sentinel))
    for suffix in ("", "-journal", "-wal", "-shm"):
        (tmp_path / f"population.sqlite{suffix}").touch()
    with pytest.raises(RuntimeError) as exc:
        t._PopulationAccumulator()
    assert exc.value is sentinel
    assert all(not Path(f"{db_path}{suffix}").exists() for suffix in ("", "-journal", "-wal", "-shm"))


def test_teacher_streaming_b1_cleanup_baseexception_failure_cleans_transaction(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_0/layouts_0.th"); _task4_fake_runtime(t, monkeypatch)
    class CustomBase(BaseException):
        pass
    sentinel = CustomBase("writer baseexception")
    staging_paths = []; writers = []; populations = []
    real_staging = t._new_staging
    monkeypatch.setattr(t, "_new_staging", lambda destination: (staging_paths.append(Path(real_staging(destination))) or staging_paths[-1]))
    real_writer_init = t._JsonlWriter.__init__
    def writer_init(writer, staging):
        writers.append(writer); real_writer_init(writer, staging)
    monkeypatch.setattr(t._JsonlWriter, "__init__", writer_init)
    real_population_init = t._PopulationAccumulator.__init__
    def population_init(population):
        real_population_init(population); populations.append(population)
    monkeypatch.setattr(t._PopulationAccumulator, "__init__", population_init)
    monkeypatch.setattr(t._JsonlWriter, "write", lambda *args, **kwargs: (_ for _ in ()).throw(sentinel))
    published = []; monkeypatch.setattr(t, "_publish_staging", lambda *args: published.append(args))
    with pytest.raises(CustomBase) as exc:
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert exc.value is sentinel
    assert published == [] and not out.exists()
    assert staging_paths and not staging_paths[-1].exists()
    assert writers and all(handle.closed for handle in writers[0]._files.values())
    assert populations
    db_path = populations[0]._db_path
    assert all(not Path(f"{db_path}{suffix}").exists() for suffix in ("", "-journal", "-wal", "-shm"))


def test_teacher_streaming_b1_review_writer_init_is_transactional(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_0/layouts_0.th"); _task4_fake_runtime(t, monkeypatch)
    stages = []; real_stage = t._new_staging
    monkeypatch.setattr(t, "_new_staging", lambda destination: (stages.append(real_stage(destination)) or stages[-1]))
    import builtins
    real_open = builtins.open; handles = []; sentinel = RuntimeError("third staging open")
    def flaky_open(path, mode="r", *args, **kwargs):
        if len(handles) >= 2 and str(path).endswith(".jsonl"):
            raise sentinel
        handle = real_open(path, mode, *args, **kwargs); handles.append(handle); return handle
    monkeypatch.setattr(t, "open", flaky_open, raising=False)
    with pytest.raises(RuntimeError) as exc:
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert exc.value is sentinel and not out.exists() and stages and not Path(stages[0]).exists()
    assert all(handle.closed for handle in handles)


def test_teacher_streaming_b1_review_writer_close_preserves_primary_and_closes_all(monkeypatch):
    t = _teacher(); writer = t._JsonlWriter.__new__(t._JsonlWriter)
    class Fake:
        closed = False
        def __init__(self, primary=False): self.primary = primary; self.calls = 0
        def flush(self):
            if self.primary: raise primary_error
        def fileno(self): return 0
        def close(self):
            self.calls += 1; self.closed = True
            if self.primary: raise cleanup
    primary_error = RuntimeError("primary"); primary = primary_error; cleanup = RuntimeError("cleanup")
    files = [Fake(i == 0) for i in range(6)]; writer._files = dict(zip(t._TASK4_JSONL, files))
    monkeypatch.setattr(t.os, "fsync", lambda fd: None)
    with pytest.raises(RuntimeError) as exc: writer.close()
    assert exc.value is primary and [f.calls for f in files] == [1] * 6


def test_teacher_streaming_b1_review_directory_fd_closes_on_fsync_error(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; out = tmp_path / "out"
    _task4_shard(root, relative_path="worker_0/layouts_0.th"); _task4_fake_runtime(t, monkeypatch)
    stages = []; real_stage = t._new_staging
    monkeypatch.setattr(t, "_new_staging", lambda destination: (stages.append(Path(real_stage(destination))) or stages[-1]))
    real_open, real_close, real_fsync = t.os.open, t.os.close, t.os.fsync; owned = []; sentinel = OSError("directory fsync")
    def spy_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path).resolve() == stages[-1].resolve(): owned.append(fd)
        return fd
    closed = []
    monkeypatch.setattr(t.os, "open", spy_open); monkeypatch.setattr(t.os, "close", lambda fd: (closed.append(fd), real_close(fd))[1])
    failed = []
    monkeypatch.setattr(t.os, "fsync", lambda fd: (failed.append(fd), (_ for _ in ()).throw(sentinel))[1] if fd in owned else real_fsync(fd))
    published = []; monkeypatch.setattr(t, "_publish_staging", lambda *a: published.append(a))
    with pytest.raises(OSError) as exc: t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert exc.value is sentinel and failed and failed[0] in closed and not out.exists() and not stages[-1].exists() and published == []


def test_teacher_excluded_rows_publish_empty_non_authorizing_terminal_evidence(tmp_path, monkeypatch):
    preflight_calls = []
    policy = _policy_for(tmp_path / "floorset_lite")
    t, root, out, calls, result = _task4_run(
        tmp_path, monkeypatch, n_min=4, preflight_calls=preflight_calls, policy=policy)
    assert result != 0 and calls == []
    assert sorted(path.name for path in out.iterdir()) == sorted(_TASK4_FILES)
    assert preflight_calls == [(policy, root / "unused.th")]
    expected_cases = [_task4_expected_case(0), _task4_expected_case(1)]
    expected_receipts = [_task4_receipt(root / "worker_2/layouts_0.th", i, expected_cases[i])
                         for i in range(2)]
    index = json.loads((out / "training_index.json").read_text())
    assert index["schema"] == "icdc_topology_training_index_v1"
    assert index["rows"] == [
        {"receipt": expected_receipts[i], "instance_id": expected_cases[i]["instance_id"],
         "source_row_count": 2, "block_count": 3, "partition": None,
         "sample_ordinal": None, "sample_seed": None, "status": "excluded_n_min"}
        for i in range(2)]
    assert all((out / name).read_bytes() == b"" for name in _TASK4_JSONL)
    manifest = json.loads((out / "g0_manifest.json").read_text())
    assert set(manifest) == {"schema", "status", "state", "secondary_reasons",
                             "training_authorized", "bounded_max_files", "trust",
                             "population", "coverage", "support_hashes", "self_sha256"}
    assert manifest["status"] == "complete"
    assert manifest["state"] == "KILLED_LEGALITY_OR_COVERAGE"
    assert manifest["secondary_reasons"] == []
    assert manifest["training_authorized"] is False
    assert manifest["bounded_max_files"] is False
    assert manifest["trust"] == _task4_expected_trust(policy)
    assert manifest["population"] == _task4_p1c_canonical_population([])
    assert manifest["coverage"]["eligible_train"] == 0
    assert manifest["coverage"]["eligible_heldout"] == 0
    assert manifest["coverage"]["heldout_winners"] == 0
    assert manifest["coverage"]["legal"] is True
    assert manifest["coverage"]["covered"] is False
    assert manifest["coverage"] == {
        "eligible_train": 0, "eligible_heldout": 0, "heldout_winners": 0,
        "legal": True, "covered": False,
        "mutation_proposals": 0, "mutation_admitted": 0,
        "mutation_intent_survived": 0, "positive_gain_heldout": 0,
        "eligible_heldout_weight": 0.0, "positive_gain_heldout_weight": 0.0,
        "rejection_counts": {
            "admission_failed": 0, "hard_audit_failed": 0,
            "intent_not_survived": 0, "official_infeasible": 0,
            "official_invalid_cost": 0, "official_evaluator_error": 0,
            "base_unavailable": 0,
        },
    }
    assert manifest["support_hashes"] == {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in _TASK4_FILES if name != "g0_manifest.json"
    }
    assert manifest["self_sha256"] == t._manifest_self_sha256(manifest)
    assert (out / "training_index.json").read_bytes() == _task4_canonical_json(index) + b"\n"
    assert (out / "g0_manifest.json").read_bytes() == _task4_canonical_json(manifest) + b"\n"


def test_teacher_bounded_max_files_is_non_authorizing_even_with_positive_fake_gain(tmp_path, monkeypatch):
    preflight_calls = []
    policy = _policy_for(tmp_path / "floorset_lite")
    t, root, out, calls, result = _task4_run(
        tmp_path, monkeypatch, max_files=1, preflight_calls=preflight_calls, policy=policy,
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert result == 1
    manifest = _task4_assert_success_artifacts(
        t, root, out, calls, policy, preflight_calls,
        bounded=True, training_authorized=False, state="TARGET_GAIN_MET",
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    assert len(calls) == 2
    assert manifest["state"] == "TARGET_GAIN_MET"
    assert manifest["coverage"] == {
        "eligible_train": 1, "eligible_heldout": 1, "heldout_winners": 1,
        "legal": True, "covered": True,
        "mutation_proposals": 2, "mutation_admitted": 2,
        "mutation_intent_survived": 2, "positive_gain_heldout": 1,
        "eligible_heldout_weight": math.exp(3 / 12),
        "positive_gain_heldout_weight": math.exp(3 / 12),
        "rejection_counts": {
            "admission_failed": 0, "hard_audit_failed": 0,
            "intent_not_survived": 0, "official_infeasible": 0,
            "official_invalid_cost": 0, "official_evaluator_error": 0,
            "base_unavailable": 0,
        },
    }


def test_teacher_numeric_discovery_filters_decoys_and_preserves_order(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"
    _task4_shard(root)
    _task4_single_shard(root, "worker_2/layouts_2.th", metric_delta=2)
    _task4_single_shard(root, "worker_2/layouts_10.th", metric_delta=3)
    _task4_single_shard(root, "worker_10/layouts_0.th", metric_delta=4)
    decoy_a = root / "worker_2" / "layouts_bad.th"; decoy_a.write_bytes(b"bad")
    decoy_b = root / "worker_bad" / "layouts_0.th"; decoy_b.parent.mkdir(); decoy_b.write_bytes(b"bad")
    calls = []; events = []
    _task4_fake_runtime(
        t, monkeypatch, calls=calls, events=events,
        outcome_factory=_task4_p1c_mutation_outcome,
    )
    loaded = []
    reader_calls = []; real_read = t._read_verified_shard

    def read(root_path, worker, layout):
        reader_calls.append((worker, layout))
        return real_read(root_path, worker, layout)

    real_load = torch.load

    def count_load(source, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert kwargs == {"weights_only": True, "map_location": "cpu"}
        loaded.append(source.getvalue())
        events.append(("load", source.getvalue(), kwargs.copy()))
        return real_load(source, **kwargs)

    monkeypatch.setattr(t, "_read_verified_shard", read)
    monkeypatch.setattr(torch, "load", count_load)
    out = tmp_path / "out"
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 0
    assert [item.receipt.relative_path for item in calls] == [
        "worker_2/layouts_0.th", "worker_2/layouts_0.th",
        "worker_2/layouts_2.th", "worker_2/layouts_10.th",
        "worker_10/layouts_0.th"]
    expected_paths = ["worker_2/layouts_0.th", "worker_2/layouts_2.th",
                      "worker_2/layouts_10.th", "worker_10/layouts_0.th"]
    assert reader_calls == [(2, 0), (2, 2), (2, 10), (10, 0)]
    assert len(loaded) == 4
    expected_events = [
        ("load", (root / relative_path).read_bytes(),
         {"weights_only": True, "map_location": "cpu"})
        for relative_path in expected_paths
    ]
    for relative_path in expected_paths:
        expected_events.extend(
            ("process", relative_path, index)
            for index in ([0, 1] if relative_path == "worker_2/layouts_0.th" else [0])
        )
    assert events == expected_events


def test_teacher_evidence_a_canonical_ascii_discovery_binds_paths_and_ignores_unicode_decoys(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    # These names must not become valid after parse-to-int/path reconstruction.
    _task4_single_shard(root, "worker_02/layouts_00.th", metric_delta=9)
    _task4_single_shard(root, "worker_²/layouts_⁰.th", metric_delta=8)
    _task4_single_shard(root, "worker_2/layouts_０.th", metric_delta=7)
    _task4_single_shard(root, "worker_١/layouts_١.th", metric_delta=6)
    calls = []; _task4_fake_runtime(
        t, monkeypatch, calls=calls, outcome_factory=_task4_p1c_mutation_outcome,
    )
    out = tmp_path / "out"
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 0
    assert [c.receipt.relative_path for c in calls] == [
        "worker_2/layouts_0.th", "worker_2/layouts_0.th"]
    assert [c.receipt.layout_index for c in calls] == [0, 1]
    assert all("02" not in c.receipt.relative_path for c in calls)


def test_teacher_evidence_a_validate_source_shard_uses_bounded_finite_checks(monkeypatch):
    t = _teacher(); source = _task4_padded_tensors()
    monkeypatch.setattr(torch, "cat", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("validation concatenated source tensors")))
    assert t._validate_source_shard(source) == (2, 4)


def test_teacher_evidence_a_json_serialization_always_disables_nan(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    _task4_fake_runtime(
        t, monkeypatch, outcome_factory=_task4_p1c_mutation_outcome,
    ); out = tmp_path / "out"
    real_dumps = t.json.dumps; seen = []
    def spy(value, *args, **kwargs):
        seen.append(kwargs.get("allow_nan", "missing"))
        return real_dumps(value, *args, **kwargs)
    monkeypatch.setattr(t.json, "dumps", spy)
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 0
    assert seen and all(value is False for value in seen)


@pytest.mark.parametrize("field,value,match", [
    ("diagnostic_energy", "bad", "diagnostic_energy"),
    ("diagnostic_energy", True, "diagnostic_energy"),
    ("diagnostic_energy", float("inf"), "(diagnostic_energy|noncanonical runtime evidence)"),
    ("drift", [], "drift"), ("drift", {}, "drift"),
    ("drift", {"max_abs": True}, "drift"),
    ("drift", {"max_abs": -1.0}, "drift"),
    ("drift", {"max_abs": float("nan")}, "(drift|noncanonical runtime evidence)"),
    ("hard", {}, "hard"), ("hard", {1: True}, "hard"),
    ("hard", {"legal": 1}, "hard"), ("hard", {"legal": False}, "hard"),
])
def test_teacher_evidence_a_diagnostic_evidence_schema_is_semantically_rejected(
        tmp_path, monkeypatch, field, value, match):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    def malformed(module, case_input):
        valid = _task4_good_outcome(module, case_input)
        return module._CaseOutcome(
            valid.label_row,
            (dict(valid.proposal_rows[0], **{field: value}),), (),
            valid.base_cost, valid.teacher_cost, valid.legal, valid.covered,
            valid.case_status,
        )
    out = tmp_path / "out"; calls, _ = _task4_install_custom_runtime(
        t, monkeypatch, outcome_factory=malformed,
    )
    with pytest.raises(ValueError, match=match):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert len(calls) == 1 and not out.exists()


def test_teacher_ast_guard_resolves_imports_aliases_and_bytesio_source_load_contract():
    tree = ast.parse(Path("scripts/probes/icdc_topology_teacher.py").read_text())
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
                else:
                    aliases[item.name.split(".")[0]] = item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return f"{base}.{node.attr}"
        return ""

    # Resolve ordinary one-level assignment aliases after imports, including
    # aliases nested in a function body.
    for _ in range(2):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            resolved = resolve(value)
            if resolved and resolved != target.id and (isinstance(value, (ast.Name, ast.Attribute))):
                aliases[target.id] = resolved

    loads = []
    banned = ("load_test_cases", "FloorplanDatasetLiteTest", "BandFileSampler._instance",
              "golden", "validation")
    dynamic = {"__import__", "importlib.import_module", "getattr", "exec"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [item.name for item in node.names]
            assert not any(any(symbol == bad or symbol.endswith("." + bad)
                               for bad in banned + ("_instance",)) for symbol in imported)
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"golden", "_instance"}
        if isinstance(node, ast.Name):
            assert node.id != "golden"
        if isinstance(node, ast.Subscript):
            literal = node.slice.value if isinstance(node.slice, ast.Constant) else None
            assert literal != "golden"
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"):
            assert not any(
                isinstance(argument, ast.Constant) and argument.value == "golden"
                for argument in node.args
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolve(node.func)
        assert name not in dynamic and not any(name.endswith("." + item) for item in dynamic)
        assert not any(name == item or name.endswith("." + item) for item in banned)
        assert not any(keyword.arg is None for keyword in node.keywords)
        assert name != "torch.serialization.load"
        if name == "torch.load":
            loads.append(node)
            assert len(node.args) == 1
            source = node.args[0]
            assert isinstance(source, ast.Call) and resolve(source.func) == "io.BytesIO"
            assert len(source.args) == 1
            assert isinstance(source.args[0], ast.Name) and not source.keywords
            keyword_values = {
                keyword.arg: keyword.value.value
                for keyword in node.keywords
                if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
            }
            assert keyword_values == {"weights_only": True, "map_location": "cpu"}
            assert len(keyword_values) == len(node.keywords) == 2
    assert loads


def test_teacher_g0_state_precedence_literals():
    t = _teacher()
    base = {"trust_ok": True, "scorer_ok": True, "input_ok": True, "legal": True, "coverage": True,
            "teacher_mean": 0.0, "delta": 0}
    expected = ["KILLED_INPUT_CHECKPOINT_OR_SCORER", "KILLED_LEGALITY_OR_COVERAGE", "KILLED_TEACHER_GT_1_5", "STOP_HARD_GAIN_MISSED", "TARGET_GAIN_MISSED_NO_TRAINING_AUTHORITY", "TARGET_GAIN_MET"]
    cases = [{**base, "trust_ok": False}, {**base, "legal": False}, {**base, "teacher_mean": 2}, {**base, "delta": 0.01}, {**base, "delta": 0.0181504738793652}, {**base, "delta": 0.0261247299384228}]
    assert [t._g0_state(c) for c in cases] == expected
    assert t._g0_state({**base, "trust_ok": False, "legal": False, "teacher_mean": 2, "delta": .03}) == "KILLED_INPUT_CHECKPOINT_OR_SCORER"


def test_teacher_g0_requires_teacher_mean_after_all_boolean_gates_pass():
    with pytest.raises(ValueError):
        _teacher()._g0_state({"trust_ok": True, "scorer_ok": True, "input_ok": True,
                              "legal": True, "coverage": True, "delta": 0.03})


# Additional Task 4 fail-closed review contracts.  These deliberately exercise
# validation before any checkpoint load or pipeline work is reached.
def _policy_for(root):
    t = _teacher()
    identity = {"identity_schema": "icdc_canonical_state_v1", "model_config_sha256": "a" * 64,
                "model_keyset_sha256": "b" * 64, "ema_keyset_sha256": "c" * 64,
                "ema_state_sha256": "d" * 64}
    return t.TeacherTrustPolicy(root, "a" * 64, identity, "b" * 64, "contract", "2.0")


def _task4_verified_checkpoint(t, root):
    """Create the smallest canonical checkpoint and bind policy to its bytes."""
    checkpoint = _task4_teacher_payload()
    path = root / "unused.th"
    torch.save(checkpoint, path)
    identity = t._checkpoint_identity(checkpoint)
    scorer_path = (t._REPO / "scripts" / "iccad2026_evaluate.py").resolve()
    assert hashlib.sha256(scorer_path.read_bytes()).hexdigest() == t._SCORER_SHA256
    policy = t.TeacherTrustPolicy(
        root, hashlib.sha256(path.read_bytes()).hexdigest(), identity,
        t._SCORER_SHA256,
        "iccad2026_evaluate_cost_no_runtime_v1", "2.0.5",
    )
    return path, policy


def test_teacher_runtime_preflight_binds_verified_checkpoint_and_literal_scorer_contract(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    scorer_path = (t._REPO / "scripts" / "iccad2026_evaluate.py").resolve()
    assert Path(t._EVALUATOR.__file__).resolve() == scorer_path
    monkeypatch.chdir(tmp_path)
    runtime = t._runtime_hooks()
    assert runtime.authorizing is True
    assert runtime.preflight(policy, checkpoint) == _task4_expected_trust(policy)
    with pytest.raises(ValueError, match="case input"):
        runtime.process_case(None)


def test_teacher_scorer_source_file_seam_rejects_symlink_and_is_static(tmp_path, monkeypatch):
    t = _teacher()
    scorer_path = (t._REPO / "scripts" / "iccad2026_evaluate.py").absolute()
    assert t._SCORER_PATH == scorer_path
    t._verify_scorer_source_file(scorer_path, t._SCORER_SHA256)
    symlink = tmp_path / "iccad2026_evaluate.py"
    symlink.symlink_to(scorer_path)
    with pytest.raises(ValueError, match="scorer_sha256"):
        t._verify_scorer_source_file(symlink, t._SCORER_SHA256)

    monkeypatch.setattr(t, "_SCORER_PATH", symlink)
    root = tmp_path / "canonical"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    with pytest.raises(ValueError, match="scorer_sha256"):
        t._runtime_hooks().preflight(policy, checkpoint)


def test_teacher_scorer_source_seam_is_frozen_before_static_evaluator_import():
    path = Path("scripts/probes/icdc_topology_teacher.py")
    tree = ast.parse(path.read_text())
    evaluator_imports = [
        node for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "iccad2026_evaluate" and alias.asname == "_EVALUATOR"
                for alias in node.names)
    ]
    assert len(evaluator_imports) == 1
    evaluator_line = evaluator_imports[0].lineno
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"_SCORER_SHA256", "_SCORER_PATH"}:
                assignments[target.id] = node.lineno
    assert set(assignments) == {"_SCORER_SHA256", "_SCORER_PATH"}
    helper_defs = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_verify_scorer_source_file"
    ]
    assert len(helper_defs) == 1
    module_validation_calls = [
        node.value for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert any(
        isinstance(node.func, ast.Name)
        and node.func.id == "_verify_scorer_source_file"
        and len(node.args) == 2
        and all(isinstance(arg, ast.Name) for arg in node.args)
        and [arg.id for arg in node.args] == ["_SCORER_PATH", "_SCORER_SHA256"]
        for node in module_validation_calls
    )
    assert all(line < evaluator_line for line in (*assignments.values(), helper_defs[0].lineno,
                                                    *(node.lineno for node in module_validation_calls
                                                      if isinstance(node.func, ast.Name)
                                                      and node.func.id == "_verify_scorer_source_file")))
    dynamic_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))
    ]
    assert not dynamic_imports


@pytest.mark.parametrize("bad_kind", ["bool", "int"])
def test_teacher_runtime_preflight_rejects_boolean_median_runtime_signature(tmp_path, monkeypatch, bad_kind):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    evaluator = t._EVALUATOR

    if bad_kind == "bool":
        def bad_evaluate_solution(
            solution, baseline_metrics, target_constraints, b2b_connectivity,
            p2b_connectivity, pins_pos, target_areas, target_positions=None,
            median_runtime=True,
        ):
            return None
    else:
        def bad_evaluate_solution(
            solution, baseline_metrics, target_constraints, b2b_connectivity,
            p2b_connectivity, pins_pos, target_areas, target_positions=None,
            median_runtime=1,
        ):
            return None

    monkeypatch.setattr(evaluator, "evaluate_solution", bad_evaluate_solution)
    with pytest.raises(ValueError, match="evaluate_solution_signature"):
        t._runtime_hooks().preflight(policy, checkpoint)


def test_teacher_runtime_preflight_failure_clears_trusted_process_state(tmp_path):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    runtime = t._runtime_hooks()
    assert runtime.preflight(policy, checkpoint) == _task4_expected_trust(policy)
    bad_policy = dataclasses.replace(policy, expected_scorer_sha256="0" * 64)
    with pytest.raises(ValueError, match="scorer_sha256"):
        runtime.preflight(bad_policy, checkpoint)
    with pytest.raises(RuntimeError, match="before trusted preflight"):
        runtime.process_case(None)


def test_teacher_runtime_preflight_rejects_non_dataclass_metrics(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    monkeypatch.setattr(t._EVALUATOR, "SolutionMetrics", object)
    with pytest.raises(ValueError, match="cost_no_runtime"):
        t._runtime_hooks().preflight(policy, checkpoint)


@pytest.mark.parametrize("broken, expected_label", [
    ("scorer_sha256", "scorer_sha256"),
    ("shapely_available", "shapely_available"),
    ("shapely_version", "shapely_version"),
    ("evaluate_solution_signature", "evaluate_solution_signature"),
    ("cost_no_runtime", "cost_no_runtime"),
    ("compute_total_score_weighting", "compute_total_score_weighting"),
])
def test_teacher_runtime_preflight_rejects_broken_literal_scorer_contract(tmp_path, monkeypatch, broken, expected_label):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    evaluator = t._EVALUATOR
    if broken == "shapely_version":
        monkeypatch.setattr(t.shapely, "__version__", "0.0.0")
    if broken == "scorer_sha256":
        policy = dataclasses.replace(policy, expected_scorer_sha256="0" * 64)
    elif broken == "shapely_available":
        monkeypatch.setattr(evaluator, "SHAPELY_AVAILABLE", False)
    elif broken == "evaluate_solution_signature":
        monkeypatch.setattr(evaluator, "evaluate_solution", lambda bad: bad)
    elif broken == "cost_no_runtime":
        field_map = dict(evaluator.SolutionMetrics.__dataclass_fields__)
        field_map.pop("cost_no_runtime", None)
        monkeypatch.setattr(evaluator.SolutionMetrics, "__dataclass_fields__", field_map)
    else:
        monkeypatch.setattr(evaluator, "compute_total_score", lambda costs, counts: sum(costs))
    with pytest.raises(ValueError, match=expected_label):
        t._runtime_hooks().preflight(policy, checkpoint)


@pytest.mark.parametrize("root_kind", ["other", "lexical", "symlink"])
def test_teacher_rejects_noncanonical_and_symlink_roots_before_pipeline(tmp_path, root_kind):
    t = _teacher(); canonical = tmp_path / "canonical"; canonical.mkdir()
    other = tmp_path / "other"; other.mkdir()
    if root_kind == "other": root = other
    elif root_kind == "lexical": root = canonical / ".." / "other"
    else:
        root = tmp_path / "canonical-link"; root.symlink_to(canonical, target_is_directory=True)
    out = tmp_path / "out"
    args = ["--data-root", str(root), "--out-dir", str(out), "--index-out", str(out / "training_index.json"),
            "--checkpoint", str(tmp_path / "missing.th")]
    with pytest.raises(ValueError): t.teacher_main(args, _trust_policy=_policy_for(canonical))


def test_teacher_valid_canonical_root_runs_the_authorizing_bridge_without_claiming_g0_gain(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir(); out = tmp_path / "out"
    _task4_shard(root); _checkpoint, policy = _task4_verified_checkpoint(t, root)
    args = _task4_args(root, out)
    _task4_fake_runtime(t, monkeypatch)
    assert t.teacher_main(args, _trust_policy=policy) == 1
    assert out.exists()


def test_teacher_default_policy_binds_frozen_production_inputs():
    t = _teacher()
    policy = t._production_trust_policy()
    assert policy.canonical_root == Path("FloorSet/floorset_lite").resolve()
    assert policy.expected_checkpoint_sha256 == "508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659"
    assert policy.allowed_model_identity == {
        "identity_schema": "icdc_canonical_state_v1",
        "model_config_sha256": "4c6a1e19f0574af348efa81a758c05524522ad3501d46f18e839774fa933971b",
        "model_keyset_sha256": "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7",
        "ema_keyset_sha256": "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7",
        "ema_state_sha256": "92838740993a697a56f3afdfba4402eb83c8dc095fe43462f8bdaffdb4ef5ecb",
    }
    assert policy.expected_scorer_sha256 == "7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1"
    assert policy.scorer_contract == "iccad2026_evaluate_cost_no_runtime_v1"
    assert policy.shapely_version == "2.0.5"


def test_teacher_production_identity_is_immutable_and_stable():
    t = _teacher()
    first = t._production_trust_policy()
    second = t._production_trust_policy()
    assert first.allowed_model_identity == second.allowed_model_identity
    with pytest.raises(TypeError):
        first.allowed_model_identity["ema_state_sha256"] = "forged"
    assert second.allowed_model_identity["ema_state_sha256"] == "92838740993a697a56f3afdfba4402eb83c8dc095fe43462f8bdaffdb4ef5ecb"


def test_teacher_default_runtime_is_authorizing_after_process_slice():
    runtime = _teacher()._runtime_hooks()
    assert runtime.authorizing is True


@pytest.mark.parametrize("bad", ["--scorer", "--unknown"])
def test_teacher_cli_rejects_unapproved_flags(bad, tmp_path):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir(); out = tmp_path / "out"
    args = ["--data-root", str(root), "--out-dir", str(out), "--index-out", str(out / "training_index.json"),
            "--checkpoint", str(tmp_path / "missing.th"), bad, "x"]
    with pytest.raises(SystemExit): t.teacher_main(args, _trust_policy=_policy_for(root))


@pytest.mark.parametrize("field,value", [("n", 1), ("n", True), ("cons", "bad"), ("tp", "bad")])
def test_teacher_intent_rejects_malformed_case_contract(field, value):
    t = _teacher(); rects = torch.tensor([[0., 0., 1., 1.], [1., 0., 1., 1.]])
    case = {"n": 2, "cons": [[0, 1, 0, 0, 0]] * 2, "tp": [[-1.] * 4] * 2}
    case[field] = value
    assert not t._proposal_intent_holds("base", rects, case)


def test_teacher_intent_accepts_uniform_two_column_constraints():
    t = _teacher()
    case = {"n": 2, "area": [1., 1.], "cons": [[0, 1], [0, 0]],
            "tp": [[0., 0., 1., 1.], [-1., -1., 1., 1.]]}
    rects = torch.tensor([[0., 0., 1., 1.], [1., 0., 1., 1.]], dtype=torch.float64)
    assert t._proposal_intent_holds("base", rects, case)
    assert t._proposal_intent_holds("axis:0:1:0:1", rects, case)
    assert t._proposal_intent_holds("pin:0:1:0:1", rects, case)
    assert not t._proposal_intent_holds("contact:5:0:1:0:1", rects, case)


@pytest.mark.parametrize("cons", [
    [[0, 0], [0, 0, 0, 0, 0]],
    [[2, 0], [0, 0]],
    [[0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, -1, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 99], [0, 0, 0, 0, 0]],
])
def test_teacher_intent_rejects_constraint_width_and_metadata_gaps(cons):
    case = {"n": 2, "area": [1., 1.], "cons": cons,
            "tp": [[0., 0., 1., 1.], [-1., -1., 1., 1.]]}
    rects = torch.tensor([[0., 0., 1., 1.], [1., 0., 1., 1.]], dtype=torch.float64)
    assert not _teacher()._proposal_intent_holds("base", rects, case)


@pytest.mark.parametrize("field,value", [
    ("area", None), ("area", [1.]), ("area", [0., 1.]),
    ("area", [float("nan"), 1.]), ("area", [True, 1.]),
    ("tp", [[0., 0., -1., 1.], [-1., -1., 1., 1.]]),
    ("tp", [[-1., 0., 1., 1.], [-1., -1., 1., 1.]]),
])
def test_teacher_intent_rejects_area_and_authorized_geometry_gaps(field, value):
    case = {"n": 2, "area": [1., 1.], "cons": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            "tp": [[0., 0., 1., 1.], [0., 0., 1., 1.]]}
    case[field] = value
    rects = torch.tensor([[0., 0., 1., 1.], [1., 0., 1., 1.]], dtype=torch.float64)
    assert not _teacher()._proposal_intent_holds("base", rects, case)


def test_teacher_intent_rejects_wrong_rect_contract_and_unauthorized_pin():
    t = _teacher()
    case = {"n": 2, "area": [1., 1.], "cons": [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            "tp": [[-1., -1., 1., 1.], [-1.] * 4]}
    assert not t._proposal_intent_holds(
        "base", torch.tensor([[0., 0., 1., 1.]]), case)
    assert not t._proposal_intent_holds(
        "base", torch.tensor([[0, 0, 1, 1], [1, 0, 1, 1]]), case)
    rects = torch.tensor(
        [[-1., -1., 1., 1.], [0., -1., 1., 1.]], dtype=torch.float64)
    assert not t._proposal_intent_holds("pin:0:1:0:1", rects, case)
    authorized = {**case, "tp": [[0., 0., 1., 1.], [-1.] * 4]}
    drift = torch.tensor(
        [[torch.nextafter(torch.tensor(0., dtype=torch.float64),
                          torch.tensor(1., dtype=torch.float64)), 0., 1., 1.],
         [1., 0., 1., 1.]], dtype=torch.float64)
    assert not t._proposal_intent_holds("pin:0:1:0:1", drift, authorized)


def test_teacher_contact_requires_exact_face_and_positive_perpendicular_overlap():
    t = _teacher(); case = {"n": 2, "area": [4., 4.], "cons": [[0, 0, 0, 5, 0]] * 2, "tp": [[-1.] * 4] * 2}
    partial = torch.tensor([[0., 0., 2., 2.], [2., 1.5, 2., 2.]])
    no_overlap = torch.tensor([[0., 0., 2., 2.], [2., 2., 2., 2.]])
    face_gap = torch.tensor([[0., 0., 2., 2.], [2.01, 0., 2., 2.]])
    exact = torch.tensor([[0., 0., 2., 2.], [2., 0., 2., 2.]])
    assert t._proposal_intent_holds("contact:5:0:1:0:1", partial, case)
    assert not t._proposal_intent_holds("contact:5:0:1:0:1", no_overlap, case)
    assert not t._proposal_intent_holds("contact:5:0:1:0:1", face_gap, case)
    assert t._proposal_intent_holds("contact:5:0:1:0:1", exact, case)


@pytest.mark.parametrize("bad", [None, "x", True, float("nan"), float("inf"), float("-inf")])
def test_teacher_g0_rejects_malformed_metrics(bad):
    t = _teacher(); base = {"trust_ok": True, "scorer_ok": True, "input_ok": True, "legal": True, "coverage": True,
                            "teacher_mean": 0.1, "delta": 0.03}
    for field in ("teacher_mean", "delta"):
        with pytest.raises((ValueError, TypeError)):
            t._g0_state({**base, field: bad})


def test_teacher_g0_failures_precede_missing_metrics():
    t = _teacher()
    assert t._g0_state({"trust_ok": False, "scorer_ok": True, "input_ok": True}) == "KILLED_INPUT_CHECKPOINT_OR_SCORER"
    assert t._g0_state({"trust_ok": True, "scorer_ok": True, "input_ok": True, "legal": False}) == "KILLED_LEGALITY_OR_COVERAGE"


@pytest.mark.parametrize("bad", [{"relative_path": "../x"}, {"relative_path": "/x"}, {"relative_path": ""},
                                  {"relative_path": "a\\b"}, {"layout_index": -1}, {"n": -1},
                                  {"n": 12.0}, {"n": True}, {"n": "12"},
                                  {"base_cost": "x"}, {"teacher_cost": float("nan")}, {"extra": 1}])
def test_teacher_weighted_population_rejects_untrusted_rows(bad):
    t = _teacher(); row = {"relative_path": "a.json", "layout_index": 0, "instance_id": "a", "n": 1,
                           "base_cost": 2., "teacher_cost": 1.}
    with pytest.raises((ValueError, TypeError, OverflowError)):
        t._weighted_population([{**row, **bad}])


def test_teacher_weighted_population_rejects_duplicate_source_tuple_and_instance_id():
    t = _teacher(); row = {"relative_path": "a.json", "layout_index": 0, "instance_id": "a", "n": 1,
                           "base_cost": 2., "teacher_cost": 1.}
    with pytest.raises((ValueError, TypeError)):
        t._weighted_population([row, {**row, "instance_id": "b"}])
    with pytest.raises((ValueError, TypeError)):
        t._weighted_population([row, {**row, "relative_path": "b.json"}])


def test_teacher_weighted_population_rejects_nonfinite_weighted_arithmetic():
    t = _teacher()
    base = {"relative_path": "a.json", "layout_index": 0, "instance_id": "a",
            "n": 1, "base_cost": 2., "teacher_cost": 1.}
    for bad in [
        {**base, "n": 100_000},
        {**base, "n": 700, "base_cost": 1e308, "teacher_cost": 1e308},
    ]:
        with pytest.raises((ValueError, OverflowError)):
            t._weighted_population([bad])


def test_teacher_population_hash_omits_costs_but_metrics_change():
    t = _teacher(); row = {"relative_path": "a.json", "layout_index": 0, "instance_id": "a", "n": 1,
                           "base_cost": 2., "teacher_cost": 1.}
    a = t._weighted_population([row]); b = t._weighted_population([{**row, "base_cost": 3., "teacher_cost": 2.}])
    assert a["population_sha256"] == b["population_sha256"]
    assert a["B_H"] != b["B_H"]


@pytest.mark.parametrize("bad", [{"cost_no_runtime": "x"}, {"cost_no_runtime": True}, {"ordinal": -1},
                                  {"ordinal": True}, {"name": ""}, {"name": "a\x00b"}])
def test_teacher_winner_rejects_malformed_candidate_rows(bad):
    t = _teacher(); base = {"name": "base", "ordinal": 0, "cost_no_runtime": 10., "energy": 1., "feasible": True}
    with pytest.raises((ValueError, TypeError)):
        t._select_official_winner([base, {"name": "z", "ordinal": 1, "cost_no_runtime": 4., "energy": 0., "feasible": True, **bad}])


def test_teacher_winner_rejects_nonmapping_rows():
    with pytest.raises((ValueError, TypeError)):
        _teacher()._select_official_winner([None])


@pytest.mark.parametrize("out_kind", ["existing", "symlink", "traversal"])
def test_teacher_rejects_existing_or_escaping_output_paths(tmp_path, out_kind):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir(); out = tmp_path / "out"
    if out_kind == "existing": out.mkdir()
    elif out_kind == "symlink":
        target = tmp_path / "target"; target.mkdir(); out.symlink_to(target, target_is_directory=True)
    else: out = tmp_path / "canonical" / ".." / "escape"
    args = ["--data-root", str(root), "--out-dir", str(out), "--index-out", str(out / "training_index.json"),
            "--checkpoint", str(tmp_path / "missing.th")]
    with pytest.raises(ValueError):
        t.teacher_main(args, _trust_policy=_policy_for(root))
from icdc.topology_prior import topology_losses, extract_sparse_label
from icdc.topology_prior import (
    ProposalConfig, ProposalResult, generate_proposals,
    pin_feasible_then_exact_tfdl, is_acyclic, matches_preplaced_origins,
    has_exact_positive_contact,
)


@pytest.fixture
def proposal_fixture():
    """Small legal seed: one fixed block and a disconnected grouping pair."""
    raw = torch.tensor([[0., 0., 2., 2.], [4., 0., 2., 2.],
                        [0., 5., 2., 2.], [4., 5., 2., 2.]], dtype=torch.float64)
    case = {"instance_id": "proposal-4", "n": 4, "area": [4.] * 4,
            "cons": [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                     [0, 0, 0, 5, 0], [0, 0, 0, 5, 0]],
            "tp": [[0., 0., 2., 2.], [-1., -1., -1., -1.],
                   [-1., -1., -1., -1.], [-1., -1., -1., -1.]],
            "b2b": [], "p2b": [], "pins": [], "hpwl_ref": 1., "area_ref": 16.}
    return raw, case


def test_proposal_dataclasses_are_frozen_and_have_exact_defaults():
    assert [f.name for f in fields(ProposalConfig)] == ["axis_exchange_cap", "pin_repair_cap", "group_contact_cap", "total_cap"]
    assert ProposalConfig() == ProposalConfig(8, 8, 8, 32)
    assert [f.name for f in fields(ProposalResult)] == ["name", "rects", "legal", "drift", "hard_checks", "cost", "label"]
    with pytest.raises(FrozenInstanceError): ProposalConfig().total_cap = 1


@pytest.mark.parametrize("kwargs", [{"axis_exchange_cap": -1}, {"pin_repair_cap": -1}, {"group_contact_cap": -1}, {"total_cap": -1}, {"total_cap": 1.5}])
def test_proposal_config_rejects_bad_caps(kwargs):
    with pytest.raises((TypeError, ValueError)): ProposalConfig(**kwargs)


def test_proposals_are_deterministic_named_and_capped(proposal_fixture):
    raw, case = proposal_fixture
    cfg = ProposalConfig(1, 1, 1, 3)
    first, second = list(generate_proposals(raw, case, cfg)), list(generate_proposals(raw.clone(), case, cfg))
    assert [x[0] for x in first] == [x[0] for x in second]
    assert [x[1].tolist() for x in first] == [x[1].tolist() for x in second]
    assert len(first) <= 3 and len({x[0] for x in first}) == len(first)
    assert all(n == "base" or n.startswith(("axis:", "pin:", "contact:")) for n, _ in first)


def test_base_is_float64_clone_and_input_is_untouched(proposal_fixture):
    raw, case = proposal_fixture; before = raw.clone()
    got = list(generate_proposals(raw, case, ProposalConfig(0, 0, 0, 1)))
    assert got[0][0] == "base" and got[0][1].dtype == torch.float64
    assert not got[0][1].data_ptr() == raw.data_ptr() and torch.equal(raw, before)


def test_zero_and_individual_caps_are_respected(proposal_fixture):
    raw, case = proposal_fixture
    assert list(generate_proposals(raw, case, ProposalConfig(total_cap=0))) == []
    out = list(generate_proposals(raw, case, ProposalConfig(0, 0, 0, 32)))
    assert [n for n, _ in out] == ["base"]


def test_all_emitted_rects_are_finite_positive_and_sizes_preserved(proposal_fixture):
    raw, case = proposal_fixture
    for _, rects in generate_proposals(raw, case, ProposalConfig()):
        assert rects.shape == raw.shape and torch.isfinite(rects).all() and (rects[:, 2:] > 0).all()
        assert torch.equal(rects[:, 2:], raw[:, 2:].to(torch.float64))


def test_axis_variant_changes_realized_pair_and_preserves_pin(proposal_fixture):
    raw, case = proposal_fixture; out = list(generate_proposals(raw, case, ProposalConfig(8, 0, 0, 32)))
    assert any(name.startswith("axis:") and not torch.equal(r[:, :2], raw[:, :2].to(torch.float64)) for name, r in out)
    for _, r in out: assert torch.equal(r[0, :2], raw[0, :2].to(torch.float64))


def test_pin_and_contact_predicates_cover_contract_edges():
    assert is_acyclic(3, [(0, 1), (1, 2)]) and not is_acyclic(3, [(0, 1), (1, 2), (2, 0)])
    exact = torch.tensor([[0., 0., 1., 2.], [1., .5, 1., 2.]], dtype=torch.float64)
    assert has_exact_positive_contact(exact, 0, 1, 0, True, 1.)
    corner = torch.tensor([[0., 0., 1., 1.], [1., 1., 1., 1.]], dtype=torch.float64)
    assert not has_exact_positive_contact(corner, 0, 1, 0, True, .1)


def test_matches_preplaced_accepts_batched_and_rejects_drift(proposal_fixture):
    raw, case = proposal_fixture
    assert matches_preplaced_origins(raw, case) and matches_preplaced_origins(raw.unsqueeze(0), case)
    drift = raw.clone(); drift[0, 0] += 0.25
    assert not matches_preplaced_origins(drift, case)


def test_proposal_result_keeps_optional_cost_and_label_none():
    r = ProposalResult("base", torch.zeros((1, 4)), torch.zeros((1, 4)), torch.zeros((1, 2)), {"ok": True}, None, None)
    assert r.cost is None and r.label is None and isinstance(r.hard_checks, dict)


def test_total_cap_limits_after_base(proposal_fixture):
    raw, case = proposal_fixture
    assert len(list(generate_proposals(raw, case, ProposalConfig(total_cap=1)))) == 1


def test_generator_rejects_wrong_rank(proposal_fixture):
    raw, case = proposal_fixture
    with pytest.raises((TypeError, ValueError)): list(generate_proposals(raw.unsqueeze(0), case, ProposalConfig()))


@pytest.mark.parametrize("bad", [torch.ones((4, 3)), torch.ones((3, 4)), torch.tensor([[0., 0., -1., 1.]] * 4)])
def test_generator_rejects_malformed_or_nonpositive_raw(proposal_fixture, bad):
    with pytest.raises((TypeError, ValueError)): list(generate_proposals(bad, proposal_fixture[1], ProposalConfig()))


def test_generator_rejects_nonfinite_raw(proposal_fixture):
    raw, case = proposal_fixture; raw[0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)): list(generate_proposals(raw, case, ProposalConfig()))


def test_generator_accepts_float32_and_casts_float64(proposal_fixture):
    raw, case = proposal_fixture
    assert all(r.dtype == torch.float64 for _, r in generate_proposals(raw, case, ProposalConfig(0, 0, 0, 1)))


def test_acyclic_rejects_self_loop_and_bad_endpoints():
    assert not is_acyclic(2, [(0, 0)]) and not is_acyclic(2, [(0, 3)])


def test_acyclic_rejects_bad_n_or_edges():
    assert not is_acyclic(0, [(0, 1)]) and not is_acyclic(2, [("x", 1)])


def test_contact_rejects_gap_and_wrong_order():
    exact = torch.tensor([[0., 0., 1., 2.], [1., .5, 1., 2.]], dtype=torch.float64)
    gap = exact.clone(); gap[1, 0] += 1e-4
    assert not has_exact_positive_contact(gap, 0, 1, 0, True, 1.)
    assert not has_exact_positive_contact(exact, 0, 1, 0, False, 1.)


def test_contact_rejects_malformed_rects():
    assert not has_exact_positive_contact(torch.ones((2, 3)), 0, 1, 0, True, 1.)


def test_matches_preplaced_rejects_one_ulp_drift(proposal_fixture):
    raw, case = proposal_fixture; drift = raw.clone().to(torch.float64); drift[0, 0] = torch.nextafter(drift[0, 0], torch.tensor(1.))
    assert not matches_preplaced_origins(drift, case)


def test_matches_preplaced_rejects_malformed():
    assert not matches_preplaced_origins(torch.ones((3, 4)), {"n": 4, "cons": [[0, 1]] * 4})


def test_pin_admission_calls_nonexact_then_exact_on_original_seed(proposal_fixture, monkeypatch):
    raw, case = proposal_fixture
    calls = []
    class TSpy:
        def tfdl(self, rects, mask, pinned, *, pin_xy, boundary_code, exact=False):
            calls.append((rects.clone(), mask.clone(), pinned.clone(), pin_xy.clone(), boundary_code.clone(), exact))
            return rects.clone(), torch.zeros((1, 4, 2), dtype=rects.dtype)
    class EngineSpy:
        def verify_hard_legal(self, p, area, cons, tp):
            return {"all": True}
    monkeypatch.setattr(topology_prior, "T", TSpy(), raising=False)
    monkeypatch.setattr(topology_prior, "engine", EngineSpy(), raising=False)
    result = pin_feasible_then_exact_tfdl(raw, case)
    assert result is not None and result[0].shape == (4, 4) and result[1].shape == (4, 2)
    assert [c[-1] for c in calls] == [False, True]
    assert all(torch.equal(c[0], raw.unsqueeze(0)) for c in calls)
    assert all(torch.equal(c[2], torch.tensor([[True, False, False, False]])) for c in calls)
    assert all(torch.equal(c[3], torch.tensor([[[0., 0.], [-1., -1.], [-1., -1.], [-1., -1.]]])) for c in calls)
    assert all(torch.equal(c[4], torch.tensor([[0, 0, 0, 0]])) for c in calls)


def test_contact_topology_not_collapsed_into_base(proposal_fixture):
    raw, case = proposal_fixture
    names = [n for n, _ in generate_proposals(raw, case, ProposalConfig(0, 0, 8, 32))]
    assert names[0] == "base" and len(names) == len(set(names))


def test_proposals_never_mutate_case_or_raw(proposal_fixture):
    raw, case = proposal_fixture; snapshot = repr(case); before = raw.clone()
    list(generate_proposals(raw, case, ProposalConfig()))
    assert repr(case) == snapshot and torch.equal(raw, before)


def test_proposal_names_have_stable_kind_order(proposal_fixture):
    raw, case = proposal_fixture
    names = [n for n, _ in generate_proposals(raw, case, ProposalConfig())]
    kinds = [n.split(":", 1)[0] for n in names]
    assert kinds == sorted(kinds, key=lambda k: {"base": 0, "axis": 1, "pin": 2, "contact": 3}[k])


def test_topology_prior_loss_axis1_and_empty_backward():
    rects = torch.tensor([[[0., 0., 1., 1.], [0., 2., 1., 1.]]], requires_grad=True)
    empty = SparseTopologyBatch(*(torch.empty(0, dtype=torch.long) if n not in ('edge_margin','edge_weight','contact_margin','contact_weight') else torch.empty(0) for n in SparseTopologyBatch.__dataclass_fields__))
    out = topology_losses(rects, empty, torch.ones(1))
    assert out['total'].item() == 0
    out['total'].backward()
    assert rects.grad is not None


def test_topology_prior_extracts_margin_and_record_weight():
    legal = torch.tensor([[0., 0., 2., 1.], [3., 0., 1., 1.]], dtype=torch.float64)
    label = extract_sparse_label(legal, {'n': 2, 'cons': [[0, 0], [0, 0]]}, 'x', 4, 2., 6.)
    assert label.record_weight == 3.
    assert label.edges[0].margin == 1.


def test_topology_prior_separation_and_contact_gradients_are_directional():
    rects = torch.tensor([[[0., 0., 1., 1.], [2., 0., 1., 1.]]], requires_grad=True)
    batch = SparseTopologyBatch(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), torch.tensor([0]),
        torch.tensor([0.5]), torch.tensor([1.]), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0),
    )
    out = topology_losses(rects, batch, torch.ones(1))
    assert out["separation"].item() == 0
    out["total"].backward()
    assert rects.grad[0, 1, 0].item() == 0


def test_broken_separation_has_finite_nonzero_size_gradients_and_zero_fixture():
    rects = torch.tensor([[[0., 0., 2., 2.], [1., 0., 2., 2.]]], dtype=torch.float64, requires_grad=True)
    out = topology_losses(rects, _edge_batch(margin=1.), torch.ones(1, dtype=rects.dtype))
    out["total"].backward()
    assert torch.isfinite(rects.grad).all() and (rects.grad[..., 2:] != 0).any()
    zero = torch.ones((1, 2, 4), dtype=torch.float64, requires_grad=True)
    z = topology_losses(zero, _zero_batch(), torch.ones(1, dtype=zero.dtype))
    z["total"].backward()
    assert z["total"].item() == 0 and torch.isfinite(zero.grad).all()


def test_extract_pin_paths_preserve_reduced_edges_and_ignore_forbidden_sources():
    legal = torch.tensor([[0., 0., 1., 1.], [2., 0., 1., 1.], [4., 0., 1., 1.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]}
    label = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    assert label.pin_paths == ((0, 1, 2),)
    assert [(e.src, e.dst, e.kind) for e in label.edges].count((0, 1, "pin")) == 1


def _batch(edge_margin=0.5, contact_margin=0.1, *, dtype=torch.float64):
    empty_i = torch.empty(0, dtype=torch.long)
    empty_f = torch.empty(0, dtype=dtype)
    return SparseTopologyBatch(torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long), torch.tensor([edge_margin], dtype=dtype), torch.ones(1, dtype=dtype), empty_i, empty_i, empty_i, empty_i, empty_i, empty_f, empty_f)


def _empty_i(dtype=torch.long):
    return torch.empty(0, dtype=dtype)


def _empty_f(dtype=torch.float64):
    return torch.empty(0, dtype=dtype)


def _edge_batch(*, src=0, dst=1, axis=0, margin=1., weight=1., batch=0, dtype=torch.float64):
    return SparseTopologyBatch(
        torch.tensor([batch], dtype=torch.long), torch.tensor([src], dtype=torch.long),
        torch.tensor([dst], dtype=torch.long), torch.tensor([axis], dtype=torch.long),
        torch.tensor([margin], dtype=dtype), torch.tensor([weight], dtype=dtype),
        _empty_i(), _empty_i(), _empty_i(), _empty_i(), _empty_i(), _empty_f(dtype), _empty_f(dtype),
    )


def _contact_batch(*, a=0, b=1, axis=0, order=1, margin=2., weight=1., batch=0, dtype=torch.float64):
    return SparseTopologyBatch(
        _empty_i(), _empty_i(), _empty_i(), _empty_i(), _empty_f(dtype), _empty_f(dtype),
        torch.tensor([batch], dtype=torch.long), torch.tensor([a], dtype=torch.long),
        torch.tensor([b], dtype=torch.long), torch.tensor([axis], dtype=torch.long),
        torch.tensor([order], dtype=torch.long), torch.tensor([margin], dtype=dtype),
        torch.tensor([weight], dtype=dtype),
    )


def _zero_batch(dtype=torch.float64):
    return SparseTopologyBatch(*([_empty_i()] * 4 + [_empty_f(dtype)] * 2 + [_empty_i()] * 5 + [_empty_f(dtype)] * 2))


def test_separation_axis0_has_hand_computed_loss_and_directional_gradient():
    rects = torch.tensor([[[0., 0., 2., 2.], [1., 0., 2., 2.]]], dtype=torch.float64, requires_grad=True)
    out = topology_losses(rects, _edge_batch(), torch.ones(1, dtype=rects.dtype))
    assert out["separation"].item() == pytest.approx(2.)
    out["total"].backward()
    assert rects.grad[0, 1, 0] < 0 and rects.grad[0, 0, 0] > 0


def test_separation_axis1_has_hand_computed_loss_and_directional_gradient():
    rects = torch.tensor([[[0., 0., 2., 2.], [0., 1., 2., 2.]]], dtype=torch.float64, requires_grad=True)
    out = topology_losses(rects, _edge_batch(axis=1), torch.ones(1, dtype=rects.dtype))
    assert out["separation"].item() == pytest.approx(2.)
    out["total"].backward()
    assert rects.grad[0, 1, 1] < 0 and rects.grad[0, 0, 1] > 0


def test_contact_margin_is_scaled_and_has_positive_gradients():
    rects = torch.tensor([[[0., 0., 2., 2.], [2.5, .5, 2., 2.]]], dtype=torch.float64, requires_grad=True)
    out = topology_losses(rects, _contact_batch(), torch.tensor([10.], dtype=rects.dtype))
    assert out["contact"].item() == pytest.approx(.1)
    out["total"].backward()
    assert rects.grad[0, 1, 0] > 0 and rects.grad[0, 1, 1] > 0


@pytest.mark.parametrize("axis,order,rects", [
    (0, 1, [[[0., 0., 2., 2.], [2., 0., 2., 2.]]]),
    (0, 0, [[[2., 0., 2., 2.], [0., 0., 2., 2.]]]),
    (1, 1, [[[0., 0., 2., 2.], [0., 2., 2., 2.]]]),
    (1, 0, [[[0., 2., 2., 2.], [0., 0., 2., 2.]]]),
])
def test_exact_contacts_have_zero_loss(axis, order, rects):
    value = torch.tensor(rects, dtype=torch.float64, requires_grad=True)
    out = topology_losses(value, _contact_batch(axis=axis, order=order, margin=2.), torch.ones(1, dtype=value.dtype))
    assert out["contact"].item() == pytest.approx(0.)


def test_contact_perpendicular_deficit_is_not_clamped():
    rects = torch.tensor([[[0., 0., 2., 1.], [2., .5, 2., 1.]]], dtype=torch.float64)
    out = topology_losses(rects, _contact_batch(margin=2.), torch.ones(1, dtype=rects.dtype))
    assert out["contact"].item() == pytest.approx(1.5)


def test_contact_negative_overlap_and_exact_margin_scale():
    rects = torch.tensor([[[0., 0., 2., 1.], [2., 2., 2., 1.]]], dtype=torch.float64)
    # perpendicular intervals have overlap -1; gap 0, margin 2, scale 1 => 3
    assert topology_losses(rects, _contact_batch(margin=2.), torch.ones(1, dtype=rects.dtype))["contact"].item() == pytest.approx(3.)


def test_empty_labels_are_zero_same_dtype_and_backward_safe():
    rects = torch.ones((1, 2, 4), dtype=torch.float64, requires_grad=True)
    out = topology_losses(rects, _zero_batch(), torch.ones(1, dtype=rects.dtype))
    assert out["total"].dtype == rects.dtype and out["total"].item() == 0.
    out["total"].backward()
    assert rects.grad is not None


def test_cross_batch_scales_are_normalized_before_global_mean():
    rects = torch.tensor([[[0., 0., 2., 2.], [1., 0., 2., 2.]], [[0., 0., 2., 2.], [1., 0., 2., 2.]]], dtype=torch.float64)
    b = dataclasses.replace(_edge_batch(), edge_batch=torch.tensor([0]), edge_weight=torch.tensor([1.], dtype=rects.dtype))
    b = dataclasses.replace(b, edge_src=torch.tensor([0, 0]), edge_dst=torch.tensor([1, 1]), edge_batch=torch.tensor([0, 1]), edge_axis=torch.tensor([0, 0]), edge_margin=torch.tensor([1., 1.], dtype=rects.dtype), edge_weight=torch.tensor([1., 1.], dtype=rects.dtype))
    assert topology_losses(rects, b, torch.tensor([1., 100.], dtype=rects.dtype))["separation"].item() == pytest.approx(1.01)
    assert topology_losses(rects, b, torch.tensor([0., 1.], dtype=rects.dtype))["separation"].item() == pytest.approx(1_000_000.01)


def test_cross_batch_unequal_effective_weights_are_global():
    rects = torch.tensor([[[0., 0., 2., 2.], [1., 0., 2., 2.]], [[0., 0., 2., 2.], [1., 0., 2., 2.]]], dtype=torch.float64)
    b = dataclasses.replace(_edge_batch(), edge_batch=torch.tensor([0, 1]), edge_src=torch.tensor([0, 0]), edge_dst=torch.tensor([1, 1]), edge_axis=torch.tensor([0, 0]), edge_margin=torch.tensor([1., 1.], dtype=rects.dtype), edge_weight=torch.tensor([1., 3.], dtype=rects.dtype))
    assert topology_losses(rects, b, torch.tensor([1., 100.], dtype=rects.dtype))["separation"].item() == pytest.approx(.515)


def test_direct_weights_are_detached_and_nonpositive_weights_rejected():
    rects = torch.tensor([[[0., 0., 2., 2.], [1., 0., 2., 2.]]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
    out = topology_losses(rects, dataclasses.replace(_edge_batch(), edge_weight=weights), torch.ones(1, dtype=rects.dtype))
    out["total"].backward(); assert rects.grad is not None and weights.grad is None
    for value in (0., -1.):
        with pytest.raises(ValueError): topology_losses(rects.detach(), dataclasses.replace(_edge_batch(), edge_weight=torch.tensor([value])), torch.ones(1, dtype=rects.dtype))


@pytest.mark.parametrize("bad", [torch.tensor([-1.]), torch.tensor([float('nan')])])
def test_scale_zero_negative_nan_contract(bad):
    with pytest.raises(ValueError): topology_losses(torch.ones((1, 2, 4), dtype=torch.float64), _edge_batch(), bad.to(torch.float64))


def test_nonempty_zero_contact_margin_rejected_but_positive_accepted():
    rects = torch.ones((1, 2, 4), dtype=torch.float64)
    with pytest.raises(ValueError): topology_losses(rects, _contact_batch(margin=0.), torch.ones(1, dtype=rects.dtype))
    assert torch.isfinite(topology_losses(rects, _contact_batch(margin=1.), torch.ones(1, dtype=rects.dtype))["total"])


@pytest.mark.parametrize("field,value", [("edge_batch", 1), ("edge_src", 2), ("edge_axis", 2), ("contact_order", 2)])
def test_sparse_batch_indices_axes_and_orders_rejected(field, value):
    with pytest.raises(ValueError):
        topology_losses(torch.ones((1, 2, 4), dtype=torch.float64), dataclasses.replace(_edge_batch(), **{field: torch.tensor([value])}), torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("bad", [
    dataclasses.replace(_contact_batch(order=2), contact_order=torch.tensor([2])),
    dataclasses.replace(_contact_batch(), contact_a=torch.tensor([2])),
    dataclasses.replace(_contact_batch(), contact_batch=torch.tensor([1])),
    dataclasses.replace(_contact_batch(), contact_axis=torch.tensor([2])),
    dataclasses.replace(_contact_batch(), contact_order=torch.tensor([1], dtype=torch.int32)),
])
def test_each_contact_batch_validation_branch_is_reached(bad):
    with pytest.raises(ValueError): topology_losses(torch.ones((1, 2, 4), dtype=torch.float64), bad, torch.ones(1, dtype=torch.float64))


def test_contact_batch_length_mismatch_is_rejected():
    bad = dataclasses.replace(_contact_batch(), contact_margin=torch.tensor([1., 2.]))
    with pytest.raises(ValueError): topology_losses(torch.ones((1, 2, 4), dtype=torch.float64), bad, torch.ones(1, dtype=torch.float64))


def test_extractor_tie_breaks_axis_then_lower_id_and_chain_has_no_transitive_edge():
    legal = torch.tensor([[0., 0., 1., 1.], [2., 2., 1., 1.], [4., 2., 1., 1.]], dtype=torch.float64)
    label = extract_sparse_label(legal, {"n": 3, "cons": [[0, 0], [0, 0], [0, 0]]}, "x", 1, 1., 2.)
    assert (label.edges[0].axis, label.edges[0].src) == (0, 0)
    assert not any(e.kind == "sep" and e.src == 0 and e.dst == 2 for e in label.edges)


def test_extractor_equal_axis_centres_choose_lower_id_and_axis_zero_tie():
    legal = torch.tensor([[0., 0., 2., 2.], [0., 2., 2., 2.]], dtype=torch.float64)
    label = extract_sparse_label(legal, {"n": 2, "cons": [[0, 0], [0, 0]]}, "x", 1, 1., 2.)
    assert any(e.src == 0 and e.dst == 1 for e in label.edges)


def test_cluster_contact_requires_positive_overlap():
    legal = torch.tensor([[0., 0., 2., 2.], [2., .5, 2., 2.]], dtype=torch.float64)
    label = extract_sparse_label(legal, {"n": 2, "cons": [[0, 0, 0, 7, 0], [0, 0, 0, 7, 0]]}, "x", 1, 1., 2.)
    assert len(label.contacts) == 1


def test_cluster_contact_exact_face_margin_and_transitive_edges():
    legal = torch.tensor([[0., 0., 2., 2.], [2., 0., 2., 2.], [4., 0., 2., 2.]], dtype=torch.float64)
    label = extract_sparse_label(legal, {"n": 3, "cons": [[0, 0, 7, 7, 0], [0, 0, 7, 7, 0], [0, 0, 7, 7, 0]]}, "x", 1, 1., 2.)
    assert { (c.a, c.b) for c in label.contacts } == {(0, 1), (1, 2)}
    assert all(c.perp_margin == 2 for c in label.contacts)


def test_cluster_contact_rejects_disconnected_members():
    legal = torch.tensor([[0., 0., 2., 2.], [2., 2.5, 2., 2.]], dtype=torch.float64)
    with pytest.raises(ValueError):
        extract_sparse_label(legal, {"n": 2, "cons": [[0, 0, 0, 7, 0], [0, 0, 0, 7, 0]]}, "x", 1, 1., 2.)


def test_cluster_contact_requires_literal_positive_overlap_and_face_contact():
    base = {"n": 2, "cons": [[0, 0, 0, 7, 0], [0, 0, 0, 7, 0]]}
    exact = torch.tensor([[0., 0., 2., 2.], [2., 0., 2., 2.]], dtype=torch.float64)
    assert len(extract_sparse_label(exact, base, "x", 1, 1., 2.).contacts) == 1
    near = exact.clone(); near[1, 0] += 1e-10
    with pytest.raises(ValueError): extract_sparse_label(near, base, "x", 1, 1., 2.)
    corner = torch.tensor([[0., 0., 2., 2.], [2., 2., 2., 2.]], dtype=torch.float64)
    with pytest.raises(ValueError): extract_sparse_label(corner, base, "x", 1, 1., 2.)


def test_cluster_contact_square_kruskal_is_deterministic_and_canonical():
    legal = torch.tensor([[0., 0., 2., 2.], [2., 0., 2., 2.], [0., 2., 2., 2.], [2., 2., 2., 2.]], dtype=torch.float64)
    case = {"n": 4, "cons": [[0, 0, 0, 9, 0]] * 4}
    a = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    b = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    assert a == b
    assert {(c.a, c.b, c.axis) for c in a.contacts} == {(0, 1, 0), (2, 3, 0), (0, 2, 1)}
    reversed_ids = torch.tensor([[2., 0., 2., 2.], [0., 0., 2., 2.]], dtype=torch.float64)
    contact = extract_sparse_label(reversed_ids, {"n": 2, "cons": [[0, 0, 0, 9, 0]] * 2}, "x", 1, 1., 2.).contacts[0]
    assert (contact.a, contact.b, contact.axis, contact.a_before_b) == (0, 1, 0, False)


def test_pin_path_uses_greatest_end_predecessor_on_x():
    legal = torch.tensor([[0., 0., 2., 2.], [1., 10., 2., 2.], [6., 6., 2., 2.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0], [0, 0], [0, 1]]}
    label = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    incoming = {(e.src, e.dst) for e in label.edges if e.kind == "sep" and e.axis == 0}
    assert {(0, 2), (1, 2)} <= incoming
    assert label.pin_paths == ((1, 2),)


def test_pin_path_x_tie_uses_lower_id_predecessor():
    legal = torch.tensor([[0., 0., 3., 2.], [1., 10., 2., 2.], [7., 6., 2., 2.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0], [0, 0], [0, 1]]}
    label = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    incoming = {(e.src, e.dst) for e in label.edges if e.kind == "sep" and e.axis == 0}
    assert {(0, 2), (1, 2)} <= incoming
    assert label.pin_paths == ((0, 2),)


def test_pin_path_uses_greatest_end_predecessor_on_y():
    legal = torch.tensor([[0., 0., 2., 2.], [10., 1., 2., 2.], [6., 7., 2., 2.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0], [0, 0], [0, 1]]}
    label = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    assert label.pin_paths == ((1, 2),)


def test_collate_preserves_sep_and_pin_duplicates_with_record_weight():
    legal = torch.tensor([[0., 0., 1., 1.], [2., 0., 1., 1.], [4., 0., 1., 1.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0], [0, 0], [0, 1]]}
    label = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    batch = collate_labels([label], torch.device("cpu"), torch.float64)
    matches = [(s.item(), d.item(), ax.item(), w.item()) for s, d, ax, w in zip(batch.edge_src, batch.edge_dst, batch.edge_axis, batch.edge_weight) if (s.item(), d.item(), ax.item()) == (0, 1, 0)]
    assert len(matches) == 2 and all(weight == 2. for *_rest, weight in matches)


@pytest.mark.parametrize("legal,case", [
    (torch.ones((1, 3, 4), dtype=torch.float64), {"n": 3, "cons": [[0, 0]] * 3}),
    (torch.tensor([[0., 0., float("nan"), 1.]], dtype=torch.float64), {"n": 1, "cons": [[0, 0]]}),
])
def test_extractor_rejects_exact_legal_shape_and_value_gaps(legal, case):
    with pytest.raises(ValueError): extract_sparse_label(legal, case, "x", 1, 1., 2.)


@pytest.mark.parametrize("teacher,base", [(float("nan"), 2.), (1., float("inf")), (0., 2.), (-1., 2.), (2., 0.)])
def test_extractor_rejects_nonfinite_and_nonpositive_costs(teacher, base):
    with pytest.raises(ValueError): extract_sparse_label(torch.ones((1, 4), dtype=torch.float64), {"n": 1, "cons": [[0, 0]]}, "x", 1, teacher, base)


@pytest.mark.parametrize("cons", [
    [[0, 0, -1, 1, 0]], [[0, 0, 0, -1, 0]], [[0, 0, 0, 1, 16]], [[2, 0, 0, 1, 0]], [[0, 2, 0, 1, 0]],
])
def test_extractor_rejects_invalid_fixed_preplaced_cluster_metadata(cons):
    with pytest.raises(ValueError): extract_sparse_label(torch.ones((1, 4), dtype=torch.float64), {"n": 1, "cons": cons}, "x", 1, 1., 2.)


def test_extractor_rejects_float32_nonfinite_nonpositive_and_metadata():
    base = {"n": 1, "cons": [[0, 0]]}
    for legal in (torch.ones((1, 4), dtype=torch.float32), torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64), torch.tensor([[0., 0., float('nan'), 1.]], dtype=torch.float64)):
        with pytest.raises(ValueError): extract_sparse_label(legal, base, "x", 1, 1., 2.)
    for seed, iid, teacher, cost in [(True, "x", 1., 2.), (1, "", 1., 2.), (1, "x", 3., 2.), (1, "x", 1., 0.)]:
        with pytest.raises(ValueError): extract_sparse_label(torch.ones((1, 4), dtype=torch.float64), base, iid, seed, teacher, cost)


def test_mixed_constraint_row_widths_are_rejected():
    with pytest.raises(ValueError):
        extract_sparse_label(
            torch.ones((2, 4), dtype=torch.float64),
            {"n": 2, "cons": [[0, 0], [0, 0, 0, 0, 0]]},
            "x", 1, 1., 2.,
        )


def test_scale_dtype_mismatch_rejected():
    with pytest.raises(ValueError):
        topology_losses(torch.zeros((1, 2, 4), dtype=torch.float64), _batch(), torch.ones(1, dtype=torch.float32))


def test_direct_margin_contracts_reject_bad_values():
    bad_edge = _batch(edge_margin=-1)
    with pytest.raises(ValueError): topology_losses(torch.zeros((1, 2, 4), dtype=torch.float64) + 1, bad_edge, torch.ones(1, dtype=torch.float64))
    bad = _batch()
    bad = dataclasses.replace(bad, contact_margin=torch.zeros(0, dtype=torch.float64))
    topology_losses(torch.ones((1, 2, 4), dtype=torch.float64), bad, torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("bad", [torch.tensor([-1.]), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_scale_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        topology_losses(torch.ones((1, 2, 4), dtype=torch.float64), _batch(), bad.to(torch.float64))


def test_extractor_rejects_bad_cost_and_seed_types():
    legal = torch.ones((2, 4), dtype=torch.float64)
    case = {"n": 2, "cons": [[0, 0], [0, 0]]}
    with pytest.raises(ValueError): extract_sparse_label(legal, case, "x", True, 1., 2.)
    with pytest.raises(ValueError): extract_sparse_label(legal, case, "x", 1, 2., 1.)


def test_extractor_transitive_chain_and_repeat_determinism():
    legal = torch.tensor([[0., 0., 1., 1.], [2., 0., 1., 1.], [4., 0., 1., 1.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0], [0, 0], [0, 0]]}
    a = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    b = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    assert a == b and not any((e.src, e.dst) == (0, 2) and e.kind == "sep" for e in a.edges)


def test_extractor_is_independent_of_non_constraint_metadata():
    legal = torch.tensor([[0., 0., 1., 1.], [2., 0., 1., 1.]], dtype=torch.float64)
    base = {"n": 2, "cons": [[0, 0], [0, 0]]}
    altered = dict(base, p2b=[[99, 99, 99]], pins=[[88, 88]], b2b=[[77, 77, 77]], golden=[123])
    assert extract_sparse_label(legal, base, "x", 1, 1., 2.) == extract_sparse_label(legal, altered, "x", 1, 1., 2.)


def test_topology_prior_ast_has_no_forbidden_data_dependencies():
    path = Path(__file__).parents[1] / "partner/icdc/topology_prior.py"
    tree = ast.parse(path.read_text())
    allowed = {"math", "typing", "dataclasses", "torch", "icdc.topology_data", "topology_data", "tfdl", "engine"}
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module is not None}
    imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports <= allowed | {"__future__", "topology_data"}
    forbidden = {"golden", "p2b", "pins", "b2b"}
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and any(isinstance(a, ast.Constant) and a.value in forbidden for a in n.args) for n in ast.walk(tree))


def _called_names(fn_node):
    return {
        node.func.attr
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_proposal_functions_are_topology_only_and_do_not_import_heavy_helpers():
    path = Path(__file__).parents[1] / "partner/icdc/topology_prior.py"
    tree = ast.parse(path.read_text())
    forbidden = {"energy", "coord_polish", "violation_killer", "shelf_fallback", "extract_sparse_label"}
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("generate_proposals", "pin_feasible_then_exact_tfdl"):
        assert forbidden.isdisjoint(_called_names(funcs[name]))
        assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(funcs[name]))


def test_proposals_have_independent_per_kind_caps_when_supply_is_available():
    raw = torch.tensor([[float(i * 5), 0., 2., 2.] for i in range(6)], dtype=torch.float64)
    case = {"n": 6, "area": [4.] * 6,
            "cons": [[0, 0, 0, 9 if i in (0, 1, 2) else 0, 0] for i in range(6)],
            "tp": [[-1., -1., -1., -1.]] * 6}
    cfg = ProposalConfig(axis_exchange_cap=2, pin_repair_cap=2, group_contact_cap=2, total_cap=20)
    out = list(generate_proposals(raw, case, cfg))
    counts = {kind: sum(name.startswith(kind + ":") for name, _ in out) for kind in ("axis", "pin", "contact")}
    caps = {"axis": cfg.axis_exchange_cap, "pin": cfg.pin_repair_cap, "contact": cfg.group_contact_cap}
    assert all(counts[k] <= caps[k] for k in counts)
    assert len(out) <= cfg.total_cap


def _topology_fingerprint(rects, cons):
    return topology_prior._proposal_fingerprint(rects, cons)


def test_axis_candidate_changes_topology_fingerprint_and_preserves_sizes_and_pins(proposal_fixture):
    raw, case = proposal_fixture
    out = list(generate_proposals(raw, case, ProposalConfig(8, 0, 0, 32)))
    base_fp = _topology_fingerprint(raw, case["cons"])
    axis = next((r for name, r in out if name.startswith("axis:")), None)
    assert axis is not None and _topology_fingerprint(axis, case["cons"]) != base_fp
    assert torch.equal(axis[:, 2:], raw[:, 2:])
    assert torch.equal(axis[0, :2], raw[0, :2])


def test_predicate_reverse_orientation_and_invalid_inputs():
    rects = torch.tensor([[3., 0., 1., 2.], [0., .5, 1., 2.]], dtype=torch.float64)
    assert not has_exact_positive_contact(rects, 0, 1, 0, False, 1.)
    assert not has_exact_positive_contact(rects, 0, 1, 0, True, 1.)
    assert not has_exact_positive_contact(rects, 0, 1, 0, False, 3.)
    assert not has_exact_positive_contact(rects, 0, 1, 2, False, 1.)
    assert not has_exact_positive_contact(torch.ones((2, 3)), 0, 1, 0, False, 1.)


def test_shared_pin_path_root_deduplicates_root_pin_and_sep_edges():
    legal = torch.tensor([[0., 0., 1., 1.], [2., 0., 1., 1.], [4., 0., 1., 1.]], dtype=torch.float64)
    case = {"n": 3, "cons": [[0, 0], [0, 1], [0, 1]]}
    label = extract_sparse_label(legal, case, "x", 1, 1., 2.)
    assert label.pin_paths == ((0, 1), (0, 1, 2))
    assert [(e.src, e.dst, e.kind) for e in label.edges].count((0, 1, "pin")) == 1
    assert [(e.src, e.dst, e.kind) for e in label.edges].count((0, 1, "sep")) == 1


@pytest.mark.parametrize("cons", [[], [[0, 0]], [[0, 0, 1]], [[0, 0, 1, 0, 0]], [[0, 0, 1, 0, 16]]])
def test_extractor_malformed_constraints_are_controlled(cons):
    with pytest.raises(ValueError): extract_sparse_label(torch.ones((2, 4), dtype=torch.float64), {"n": 2, "cons": cons}, "x", 1, 1., 2.)


# Task 3 RED regressions: these deliberately pin the public proposal contract.
def test_contact_predicate_reverse_gap_and_y_orientation_are_exact():
    x_gap = torch.tensor([[3., 0., 1., 2.], [0., .5, 1., 2.]], dtype=torch.float64)
    x_exact = torch.tensor([[3., 0., 1., 2.], [2., .5, 1., 2.]], dtype=torch.float64)
    y_gap = torch.tensor([[0., 3., 2., 1.], [.5, 0., 2., 1.]], dtype=torch.float64)
    y_exact = torch.tensor([[0., 3., 2., 1.], [.5, 2., 2., 1.]], dtype=torch.float64)
    assert not has_exact_positive_contact(x_gap, 0, 1, 0, False, 1.)
    assert has_exact_positive_contact(x_exact, 0, 1, 0, False, 1.)
    assert not has_exact_positive_contact(y_gap, 0, 1, 1, False, 1.)
    assert has_exact_positive_contact(y_exact, 0, 1, 1, False, 1.)


def test_proposal_public_annotations_and_result_fields_are_exact():
    hints = get_type_hints(ProposalResult)
    assert hints == {"name": str, "rects": torch.Tensor, "legal": torch.Tensor,
                     "drift": torch.Tensor, "hard_checks": Dict[str, bool],
                     "cost": Optional[float], "label": Optional[TopologyLabel]}
    for fn in (generate_proposals, pin_feasible_then_exact_tfdl):
        fh = get_type_hints(fn)
        assert fh
    assert get_origin(get_type_hints(generate_proposals)["return"]) is collections.abc.Iterator
    assert dataclasses.is_dataclass(ProposalResult) and ProposalResult.__dataclass_params__.frozen


def _cap_case(raw, cons=None, tp=None):
    n = raw.shape[0]
    return {"instance_id": "caps", "n": n, "area": [float(r[2] * r[3]) for r in raw],
            "cons": cons or [[0, 0, 0, 0, 0] for _ in range(n)],
            "tp": tp or [[-1., -1., -1., -1.] for _ in range(n)]}


def test_each_kind_cap_is_exact_when_supply_exists():
    raw = torch.tensor([[0., 0., 2., 2.], [10., 0., 2., 2.], [0., 20., 2., 2.],
                        [10., 20., 2., 2.], [0., 40., 2., 2.], [10., 40., 2., 2.]], dtype=torch.float64)
    cons = [[0, 0, 0, 11 if i % 2 == 0 else 0, 0] for i in range(6)]
    out = list(generate_proposals(raw, _cap_case(raw, cons), ProposalConfig(2, 0, 2, 20)))
    assert sum(n.startswith("axis:") for n, _ in out) == 2
    assert sum(n.startswith("contact:") for n, _ in out) == 2
    assert len(out) == 5


def test_proposal_names_encode_pair_axis_order_and_contact_intent(proposal_fixture):
    raw, case = proposal_fixture
    names = [n for n, _ in generate_proposals(raw, case, ProposalConfig(8, 8, 8, 32))]
    for name in names:
        if name == "base":
            continue
        parts = name.split(":")
        expected = {"axis": 5, "pin": 5, "contact": 6}[parts[0]]
        assert len(parts) == expected and all(part.lstrip("-").isdigit() for part in parts[1:])
        assert parts[0] in {"axis", "pin", "contact"}


def test_generator_rejects_float32_and_malformed_case_before_any_seed(proposal_fixture):
    raw, case = proposal_fixture
    # CPU floating inputs (including float32) are admitted and normalized to float64.
    got = list(generate_proposals(raw.float(), case, ProposalConfig(0, 0, 0, 1)))
    assert got and got[0][1].dtype == torch.float64
    for bad in (dict(case, area=[1.]), dict(case, tp=[[0., 0., 1., 1.]] * 3), dict(case, cons=[])):
        with pytest.raises((TypeError, ValueError)):
            list(generate_proposals(raw, bad, ProposalConfig()))


def test_admission_rejects_displaced_preplaced_and_float32_without_tfdl(monkeypatch, proposal_fixture):
    raw, case = proposal_fixture
    calls = []
    class Spy:
        def tfdl(self, *args, **kwargs): calls.append(1); raise AssertionError("called")
    monkeypatch.setattr(topology_prior, "T", Spy())
    displaced = raw.clone(); displaced[0, 0] += 1
    assert pin_feasible_then_exact_tfdl(displaced, case) is None
    assert pin_feasible_then_exact_tfdl(raw.float(), case) is None
    assert not calls


def test_fingerprint_distinguishes_axis_and_contact_relations_independently():
    cons = [[0, 0, 0, 7, 0]] * 2
    a = torch.tensor([[0., 0., 2., 2.], [2., 1., 2., 2.]], dtype=torch.float64)
    b = a.clone(); b[1, 1] = 2.
    assert _topology_fingerprint(a, cons) != _topology_fingerprint(b, cons)


def _topology_case(raw, cons, tp=None):
    n = raw.shape[0]
    return {"instance_id": "task3", "n": n, "area": (raw[:, 2] * raw[:, 3]).tolist(),
            "cons": cons, "tp": tp or [[-1., -1., -1., -1.] for _ in range(n)],
            "b2b": [], "p2b": [], "pins": []}


def test_pin_generator_restores_preplaced_block_and_encodes_peer_axis_order():
    raw = torch.tensor([[2., 3., 2., 2.], [20., 0., 3., 4.], [40., 0., 5., 6.]], dtype=torch.float64)
    tp = [[2., 3., 2., 2.], [-1., -1., -1., -1.], [-1., -1., -1., -1.]]
    case = _topology_case(raw, [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], tp)
    out = list(generate_proposals(raw, case, ProposalConfig(0, 8, 0, 32)))
    pins = [(name, rects) for name, rects in out if name.startswith("pin:")]
    assert pins
    for name, rects in pins:
        parts = name.split(":")
        assert len(parts) == 5 and parts[0] == "pin"
        _, p, peer, axis, order = parts
        p, peer, axis, order = map(int, (p, peer, axis, order))
        assert p == 0 and peer != 0
        assert axis in (0, 1) and order in (0, 1)
        assert torch.equal(rects[0], raw[0])
        assert torch.equal(rects[:, 2:], raw[:, 2:])
        assert _topology_fingerprint(rects, case["cons"]) != _topology_fingerprint(raw, case["cons"])


def test_pin_generator_has_no_supply_when_all_blocks_are_preplaced():
    raw = torch.tensor([[0., 0., 2., 2.], [4., 0., 2., 2.]], dtype=torch.float64)
    case = _topology_case(raw, [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], [[0., 0., 2., 2.], [4., 0., 2., 2.]])
    assert not any(name.startswith("pin:") for name, _ in generate_proposals(raw, case, ProposalConfig(0, 8, 0, 32)))


def test_contact_generator_uses_only_cluster_pairs_and_literal_contact_geometry():
    raw = torch.tensor([[0., 0., 2., 2.], [50., 0., 2., 2.], [10., 0., 2., 2.]], dtype=torch.float64)
    cons = [[0, 0, 0, 7, 0], [0, 0, 0, 0, 0], [0, 0, 0, 7, 0]]
    case = _topology_case(raw, cons)
    out = [(n, r) for n, r in generate_proposals(raw, case, ProposalConfig(0, 0, 8, 32)) if n.startswith("contact:")]
    assert out
    for name, rects in out:
        parts = name.split(":"); assert len(parts) == 6
        _, gid, a, b, axis, order = parts; gid, a, b, axis, order = map(int, (gid, a, b, axis, order))
        assert gid == 7
        assert {a, b} == {0, 2} and axis in (0, 1) and order in (0, 1)
        assert has_exact_positive_contact(rects, a, b, axis, bool(order), 1e-12)
        assert torch.equal(rects[:, 2:], raw[:, 2:])


def test_contact_generator_no_supply_when_cluster_endpoints_preplaced_and_disconnected():
    raw = torch.tensor([[0., 0., 2., 2.], [40., 0., 2., 2.]], dtype=torch.float64)
    case = _topology_case(raw, [[0, 1, 0, 7, 0], [0, 1, 0, 7, 0]], [[0., 0., 2., 2.], [40., 0., 2., 2.]])
    assert not any(name.startswith("contact:") for name, _ in generate_proposals(raw, case, ProposalConfig(0, 0, 8, 32)))


def test_caps_are_exact_with_three_independent_pin_and_contact_supplies():
    raw = torch.tensor([[0., 0., 2., 2.], [20., 0., 2., 2.], [40., 0., 2., 2.],
                        [0., 20., 2., 2.], [20., 20., 2., 2.], [40., 20., 2., 2.]], dtype=torch.float64)
    cons = [[0, 1, 0, 11, 0], [0, 0, 0, 11, 0], [0, 1, 0, 12, 0],
            [0, 0, 0, 12, 0], [0, 1, 0, 13, 0], [0, 0, 0, 13, 0]]
    tp = [[0., 0., 2., 2.], [-1., -1., -1., -1.], [40., 0., 2., 2.],
          [-1., -1., -1., -1.], [20., 20., 2., 2.], [-1., -1., -1., -1.]]
    case = _topology_case(raw, cons, tp)
    out = list(generate_proposals(raw, case, ProposalConfig(0, 2, 2, 32)))
    assert sum(n.startswith("pin:") for n, _ in out) == 2
    assert sum(n.startswith("contact:") for n, _ in out) == 2


def test_generator_fingerprints_are_unique_and_inputs_immutable():
    raw = torch.tensor([[0., 0., 2., 2.], [8., 0., 2., 2.], [0., 8., 2., 2.]], dtype=torch.float64)
    case = _topology_case(raw, [[0, 0, 0, 9, 0]] * 3); before = (raw.clone(), repr(case))
    out = list(generate_proposals(raw, case, ProposalConfig()))
    assert len({_topology_fingerprint(rects, case["cons"]) for _, rects in out}) == len(out)
    assert torch.equal(raw, before[0]) and repr(case) == before[1]


def test_generator_deduplicates_realized_fingerprints_across_proposal_kinds(proposal_fixture):
    """Axis and pin proposals must not re-emit the same realized topology."""
    raw, case = proposal_fixture
    out = list(generate_proposals(raw, case, ProposalConfig()))
    fingerprints = [_topology_fingerprint(rects, case["cons"]) for _, rects in out]

    assert len(set(fingerprints)) == len(out)
    earlier = {
        fingerprint
        for (name, _), fingerprint in zip(out, fingerprints)
        if name == "base" or name.startswith("axis:")
    }
    assert not any(
        name.startswith("pin:") and fingerprint in earlier
        for (name, _), fingerprint in zip(out, fingerprints)
    )


def test_axis_reverse_boundary_moves_block_exactly_and_is_named():
    raw = torch.tensor([[0., 0., 2., 2.], [4., 0., 2., 2.]], dtype=torch.float64)
    out = dict(generate_proposals(raw, _topology_case(raw, [[0, 0]] * 2), ProposalConfig(8, 0, 0, 32)))
    candidate = out["axis:0:1:0:0"]
    expected = math.nextafter(1.0, -math.inf) - 1.0
    assert candidate[1, 0].item() == expected
    assert topology_prior._pair_state(candidate, 0, 1) == (0, 0)
    assert topology_prior._proposal_fingerprint(candidate, [[0, 0]] * 2) == (
        ("pair", 0, 1, 0, 0),
    )


def test_hard_sizes_are_normalized_only_in_emitted_proposals():
    raw = torch.tensor([[0., 0., 9., 8.], [20., 0., 7., 6.]], dtype=torch.float64)
    tp = [[0., 0., 2., 3.], [-1., -1., -1., -1.]]
    case = _topology_case(raw, [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], tp)
    out = list(generate_proposals(raw, case, ProposalConfig(8, 0, 0, 32)))
    assert torch.equal(raw[:, 2:], torch.tensor([[9., 8.], [7., 6.]], dtype=torch.float64))
    assert out
    assert all(torch.equal(rects[0, 2:], torch.tensor([2., 3.])) for _, rects in out)
    assert all(torch.equal(rects[1, 2:], torch.tensor([7., 6.])) for _, rects in out)


def test_mismatched_preplaced_pin_repairs_peer_and_reverses_repaired_pair_state():
    raw = torch.tensor([[-20., 0., 2., 2.], [10., 0., 2., 2.]], dtype=torch.float64)
    case = _topology_case(raw, [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                          [[0., 0., 2., 2.], [-1., -1., -1., -1.]])
    out = [(name, rects) for name, rects in
           generate_proposals(raw, case, ProposalConfig(0, 8, 0, 32))
           if name.startswith("pin:")]
    assert out
    pin_base = raw.clone()
    pin_base[0, :2] = torch.tensor([0., 0.])
    repaired_state = topology_prior._pair_state(pin_base, 0, 1)
    for name, rects in out:
        _, p, peer, axis, order = name.split(":")
        p, peer, axis, order = map(int, (p, peer, axis, order))
        assert (p, peer) == (0, 1)
        assert torch.equal(rects[0, :2], torch.tensor([0., 0.]))
        assert not torch.equal(rects[1, :2], pin_base[1, :2])
        assert topology_prior._proposal_fingerprint(rects, case["cons"])[0] == (
            "pair", 0, 1, axis, order
        )
        assert (axis, order) == (repaired_state[0], 1 - repaired_state[1])


def test_tfdl_public_function_does_not_call_shelf_fallback():
    tree = ast.parse(Path("partner/icdc/tfdl.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tfdl")
    calls = [node.func.id for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "shelf_fallback" not in calls


def test_axis_generation_covers_strict_y_state_beyond_sixteen_ulps():
    raw = torch.tensor([
        [-99.06493409184773, -3.451637618002451, .1977101460089326, 2.652096962005815],
        [-27.707874173400654, 86.47846882804183, .7517050889780816, .2384398881245246],
    ], dtype=torch.float64)
    case = _topology_case(raw, [[0, 0], [0, 0]], [[-1., -1., -1., -1.]] * 2)
    out = dict(generate_proposals(raw, case, ProposalConfig(64, 0, 0, 32)))
    assert "axis:0:1:1:0" in out
    candidate = out["axis:0:1:1:0"]
    assert topology_prior._pair_state(candidate, 0, 1) == (1, 0)
    assert torch.isfinite(candidate).all()


@pytest.mark.parametrize("raw", [
    [[1e308, 0., 1e308, 1.], [-1e308, 0., 1e308, 1.]],
    [[1e308, 0., 1e308, 1.], [0., 0., 1e308, 1.]],
])
def test_generator_rejects_finite_but_unrepresentable_geometry(raw):
    raw = torch.tensor(raw, dtype=torch.float64)
    case = _topology_case(raw, [[0, 0], [0, 0]], [[-1., -1., -1., -1.]] * 2)
    with pytest.raises(ValueError):
        list(generate_proposals(raw, case, ProposalConfig(8, 0, 0, 32)))


def test_place_for_pair_returns_already_valid_seed_without_ulp_search(monkeypatch):
    raw = torch.tensor([[0., 0., 2., 2.], [4., 0., 2., 2.]], dtype=torch.float64)
    calls = 0
    original = topology_prior.math.nextafter

    def spy(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(topology_prior.math, "nextafter", spy)
    got = topology_prior._place_for_pair(raw, 0, 1, 0, 1, 1)
    assert torch.equal(got, raw)
    assert calls == 0


def test_proposal_config_rejects_bool_and_result_is_frozen():
    with pytest.raises((TypeError, ValueError)): ProposalConfig(total_cap=True)
    result = ProposalResult("x", torch.zeros((1, 4)), torch.zeros((1, 4)), torch.zeros((1, 2)), {}, None, None)
    with pytest.raises(FrozenInstanceError): result.name = "y"


@pytest.mark.parametrize("failure", ["drift", "nonfinite", "pin_mismatch"])
@pytest.mark.parametrize("stage", [False, True])
def test_admission_rejects_tfdl_failure_matrix_without_running_later_stage(monkeypatch, proposal_fixture, failure, stage):
    raw, case = proposal_fixture; calls = []
    class Spy:
        def tfdl(self, rects, mask, pinned, *, pin_xy, boundary_code, exact=False):
            calls.append((exact, rects.clone(), mask.clone(), pinned.clone(), pin_xy.clone(), boundary_code.clone()))
            if exact == stage:
                legal = rects.clone()
                drift = torch.zeros((1, 4, 2), dtype=rects.dtype)
                if failure == "drift": drift[0, 1, 0] = 1.
                if failure == "nonfinite": legal[0, 1, 0] = float("nan")
                if failure == "pin_mismatch": legal[0, 0, 0] += 1.
                return legal, drift
            return rects.clone(), torch.zeros((1, 4, 2), dtype=rects.dtype)
    monkeypatch.setattr(topology_prior, "T", Spy())
    assert pin_feasible_then_exact_tfdl(raw, case) is None
    assert len(calls) == (1 if not stage else 2)
    assert all(torch.equal(c[1], raw.unsqueeze(0)) for c in calls)


@pytest.mark.parametrize("verification", [False, {}, {"ok": 1}, RuntimeError("boom")])
def test_admission_rejects_verifier_failure_matrix(monkeypatch, proposal_fixture, verification):
    raw, case = proposal_fixture; calls = []
    class TSpy:
        def tfdl(self, rects, mask, pinned, *, pin_xy, boundary_code, exact=False):
            calls.append(exact); return rects.clone(), torch.zeros((1, 4, 2), dtype=rects.dtype)
    class EngineSpy:
        def verify_hard_legal(self, *args):
            if isinstance(verification, Exception): raise verification
            return verification
    monkeypatch.setattr(topology_prior, "T", TSpy()); monkeypatch.setattr(topology_prior, "engine", EngineSpy())
    assert pin_feasible_then_exact_tfdl(raw, case) is None
    assert calls == [False, True]


def test_admission_success_spy_receives_exact_original_inputs(monkeypatch, proposal_fixture):
    raw, case = proposal_fixture; seen = {}
    class TSpy:
        def tfdl(self, rects, mask, pinned, *, pin_xy, boundary_code, exact=False):
            seen.setdefault("tfdl", []).append((rects.clone(), mask.clone(), pinned.clone(), pin_xy.clone(), boundary_code.clone(), exact))
            return rects.clone(), torch.zeros((1, 4, 2), dtype=rects.dtype)
    class EngineSpy:
        def verify_hard_legal(self, legal, area, cons, tp):
            seen["verify"] = (legal.copy(), area.copy(), cons.copy(), tp.copy()); return {"ok": True}
    monkeypatch.setattr(topology_prior, "T", TSpy()); monkeypatch.setattr(topology_prior, "engine", EngineSpy())
    result = pin_feasible_then_exact_tfdl(raw, case)
    assert result is not None
    assert all(torch.equal(x[0], raw.unsqueeze(0)) for x in seen["tfdl"])
    assert seen["verify"][1].tolist() == case["area"] and seen["verify"][2].tolist() == case["cons"] and seen["verify"][3].tolist() == case["tp"]

CANONICAL_ROOT = Path("FloorSet/floorset_lite").resolve()


def _case(**extra):
    row = {
        "instance_id": "train-7",
        "n": 3,
        "area": [4.0, 12.0, 30.0],
        "cons": [[0, 0], [1, 0], [0, 1]],
        "tp": [[7.0, 8.0, 2.0, 2.0], [9.0, 9.0, 3.0, 4.0], [5.0, 6.0, 5.0, 6.0]],
        "b2b": [],
        "p2b": [],
        "pins": [],
        "hpwl_ref": 10.0,
        "area_ref": 100.0,
    }
    row.update(extra)
    return row


def _save(path, cases, **kwargs):
    path = Path(path)
    root = path.parent / "floorset_lite"
    root.mkdir(parents=True, exist_ok=True)
    old_root = topology_data._CANONICAL_ROOT
    topology_data._CANONICAL_ROOT = root.resolve()
    source_cases = []
    receipts = []
    for index, case in enumerate(cases):
        try:
            n = int(case["n"])
            cons = torch.as_tensor(case["cons"], dtype=torch.float64)
            if cons.shape[0] != n:
                raise ValueError("constraint row count")
            if cons.shape[1] == 2:
                source_cons = torch.zeros((n, 5), dtype=torch.float64)
                source_cons[:, :2] = cons
            else:
                source_cons = cons
            area = torch.as_tensor(case["area"], dtype=torch.float64)
            tp = torch.as_tensor(case["tp"], dtype=torch.float64)
            fp = []
            for row in tp:
                width = float(row[2]) if float(row[2]) > 0 else 1.0
                height = float(row[3]) if float(row[3]) > 0 else 1.0
                x = float(row[0]) if float(row[0]) >= 0 else 0.0
                y = float(row[1]) if float(row[1]) >= 0 else 0.0
                fp.append([width, height, x, y])
            input_rows = torch.cat((area.reshape(n, 1), source_cons), dim=1)
            shard = [
                input_rows.unsqueeze(0),
                torch.as_tensor(case["b2b"], dtype=torch.float64).reshape(1, -1, 3),
                torch.as_tensor(case["p2b"], dtype=torch.float64).reshape(1, -1, 3),
                torch.as_tensor(case["pins"], dtype=torch.float64).reshape(1, -1, 2),
                torch.zeros((1, max(n - 1, 0), 3), dtype=torch.float64),
                torch.tensor(fp, dtype=torch.float64).unsqueeze(0),
                torch.tensor(
                    [
                        [
                            float(case["area_ref"]),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            float(case["hpwl_ref"]) - 2.0,
                        ]
                    ],
                    dtype=torch.float64,
                ),
            ]
            relative = "worker_0/layouts.th"
            shard_path = root / relative
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(shard, shard_path)
            digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            receipt = CorpusSourceReceipt(
                relative_path=relative,
                file_sha256=digest,
                layout_index=0,
                fingerprint="0" * 64,
            )
            source_cases.append(
                dict(
                    case,
                    instance_id=source_instance_id(receipt),
                    cons=source_cons.tolist(),
                )
            )
            receipts.append(receipt)
        except Exception:
            source_cases.append(case)
            receipts.append(None)
    for i, receipt in enumerate(receipts):
        if receipt is not None:
            try:
                receipts[i] = dataclasses.replace(
                    receipt,
                    fingerprint=fingerprint_case(source_cases[i]),
                )
            except Exception:
                receipts[i] = None
    try:
        return save_sanitized_corpus(
            path,
            source_cases,
            source_root=root,
            source_receipts=receipts,
            **kwargs,
        )
    finally:
        topology_data._CANONICAL_ROOT = old_root


def _fake_bound_source(tmp_path, monkeypatch, count=2):
    root = (tmp_path / "floorset_lite").resolve()
    shard_path = root / "worker_0" / "layouts.th"
    shard_path.parent.mkdir(parents=True)
    raw_inputs = []
    raw_fps = []
    metrics = []
    cases = []
    for index in range(count):
        input_rows = torch.tensor(
            [
                [198.0 + index, 1.0, 1.0, 2.0, 7.0, 3.0],
                [104.0 + index, 0.0, 0.0, 4.0, 9.0, 0.0],
            ],
            dtype=torch.float64,
        )
        fp = torch.tensor(
            [[22.0, 9.0, 40.0, 45.0], [13.0, 8.0, 3.0, 4.0]],
            dtype=torch.float64,
        )
        raw_inputs.append(input_rows)
        raw_fps.append(fp)
        metrics.append(
            torch.tensor(
                [100.0 + index, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
                dtype=torch.float64,
            )
        )
    shard = [
        torch.stack(raw_inputs),
        torch.zeros((count, 0, 3), dtype=torch.float64),
        torch.zeros((count, 0, 3), dtype=torch.float64),
        torch.zeros((count, 0, 2), dtype=torch.float64),
        torch.zeros((count, 1, 3), dtype=torch.float64),
        torch.stack(raw_fps),
        torch.stack(metrics),
    ]
    torch.save(shard, shard_path)
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    for index in range(count):
        receipt = CorpusSourceReceipt(
            relative_path="worker_0/layouts.th",
            file_sha256=digest,
            layout_index=index,
            fingerprint="0" * 64,
        )
        raw = raw_fps[index]
        rects = [
            (float(row[2]), float(row[3]), float(row[0]), float(row[1])) for row in raw
        ]
        case = {
            "instance_id": source_instance_id(receipt),
            "n": 2,
            "area": [198.0 + index, 104.0 + index],
            "cons": [[1, 1, 2, 7, 3], [0, 0, 4, 9, 0]],
            "tp": target_positions_from_rects(
                rects, raw_inputs[index][:, 1:], 2
            ).tolist(),
            "b2b": [],
            "p2b": [],
            "pins": [],
            "hpwl_ref": 5.0,
            "area_ref": 100.0 + index,
        }
        receipt = dataclasses.replace(receipt, fingerprint=fingerprint_case(case))
        cases.append(case)
        if index == 0:
            first_receipt = receipt
        else:
            second_receipt = receipt
    monkeypatch.setattr(topology_data, "_CANONICAL_ROOT", root)
    return root, cases, [first_receipt, second_receipt]


def test_source_receipt_schema_is_frozen_and_bound(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    assert [field.name for field in fields(CorpusSourceReceipt)] == [
        "relative_path",
        "file_sha256",
        "layout_index",
        "fingerprint",
    ]
    assert dataclasses.is_dataclass(CorpusSourceReceipt)
    assert CorpusSourceReceipt.__dataclass_params__.frozen
    assert get_type_hints(CorpusSourceReceipt) == {
        "relative_path": str,
        "file_sha256": str,
        "layout_index": int,
        "fingerprint": str,
    }
    assert source_instance_id(receipts[0]) == "worker_0/layouts.th#0"
    save_sanitized_corpus(
        tmp_path / "bound.jsonl",
        [cases[0]],
        source_root=root,
        source_receipts=[receipts[0]],
    )
    assert load_sanitized_corpus(tmp_path / "bound.jsonl")[0]["n"] == 2


def test_receipt_rejects_arbitrary_case_and_preserves_existing_output(
    tmp_path, monkeypatch
):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    output = tmp_path / "bound.jsonl"
    output.write_bytes(b"before")
    tampered = dict(cases[0], area=[999.0, 104.0])
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            output,
            [tampered],
            source_root=root,
            source_receipts=[receipts[0]],
        )
    assert output.read_bytes() == b"before"


def test_receipt_for_different_raw_row_rejects(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "different-row",
            [cases[0]],
            source_root=root,
            source_receipts=[receipts[1]],
        )


@pytest.mark.parametrize("field", ["instance_id", "fingerprint", "file_sha256"])
def test_receipt_rejects_tampered_binding_fields(tmp_path, monkeypatch, field):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    altered_case = dict(cases[0])
    altered_receipt = receipts[0]
    if field == "instance_id":
        altered_case["instance_id"] = "spoofed"
    elif field == "fingerprint":
        altered_receipt = dataclasses.replace(altered_receipt, fingerprint="f" * 64)
    else:
        altered_receipt = dataclasses.replace(altered_receipt, file_sha256="0" * 64)
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / f"bad-{field}",
            [altered_case],
            source_root=root,
            source_receipts=[altered_receipt],
        )


@pytest.mark.parametrize(
    "relative_path",
    ["../worker_0/layouts.th", "/tmp/layouts.th", "", "worker/link.th"],
)
def test_receipt_rejects_unsafe_paths(tmp_path, monkeypatch, relative_path):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    altered = dataclasses.replace(receipts[0], relative_path=relative_path)
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "bad-path",
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )


def test_receipt_rejects_symlink_escape(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    outside = tmp_path / "outside.th"
    outside.write_bytes((root / receipts[0].relative_path).read_bytes())
    link = root / "worker_0" / "escape.th"
    link.symlink_to(outside)
    altered = dataclasses.replace(
        receipts[0],
        relative_path="worker_0/escape.th",
        file_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "symlink-escape",
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )


@pytest.mark.parametrize("layout_index", [True, -1, 2, 1.0, "0"])
def test_receipt_rejects_bad_layout_index(tmp_path, monkeypatch, layout_index):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    altered = dataclasses.replace(receipts[0], layout_index=layout_index)
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "bad-index",
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )


def test_receipt_rejects_wrong_type_count_and_order(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    for bad in (object(), [receipts[0], receipts[1]], [receipts[1]]):
        with pytest.raises(ValueError):
            save_sanitized_corpus(
                tmp_path / "bad-receipts",
                cases[:1],
                source_root=root,
                source_receipts=bad,
            )
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "bad-order",
            cases,
            source_root=root,
            source_receipts=[receipts[1], receipts[0]],
        )


def test_repeated_shard_path_and_integral_float_source_rows_are_valid(
    tmp_path, monkeypatch
):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    save_sanitized_corpus(
        tmp_path / "repeated.jsonl",
        cases,
        source_root=root,
        source_receipts=receipts,
    )
    assert len(load_sanitized_corpus(tmp_path / "repeated.jsonl")) == 2


def test_corrupt_source_shard_rejects_without_partial_write(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    shard_path = root / receipts[0].relative_path
    shard_path.write_bytes(b"corrupt")
    output = tmp_path / "corrupt.jsonl"
    output.write_bytes(b"before")
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            output,
            [cases[0]],
            source_root=root,
            source_receipts=[receipts[0]],
        )
    assert output.read_bytes() == b"before"


def test_tensor_batch_scalar_and_mismatched_lengths_reject_without_partial(
    tmp_path, monkeypatch
):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    shard_path = root / receipts[0].relative_path
    output = tmp_path / "malformed-batch.jsonl"
    output.write_bytes(b"before")
    malformed = [
        torch.tensor(1.0),
        torch.zeros((2, 0, 3), dtype=torch.float64),
        torch.zeros((2, 0, 3), dtype=torch.float64),
        torch.zeros((2, 0, 2), dtype=torch.float64),
        torch.zeros((2, 2, 1), dtype=torch.float64),
        torch.zeros((2, 2, 4), dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
    ]
    torch.save(malformed, shard_path)
    altered = dataclasses.replace(
        receipts[0],
        file_sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            output,
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )
    assert output.read_bytes() == b"before"

    malformed[0] = torch.zeros((2, 6), dtype=torch.float64)
    malformed[1] = torch.zeros((1, 0, 3), dtype=torch.float64)
    torch.save(malformed, shard_path)
    altered = dataclasses.replace(
        receipts[0],
        file_sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            output,
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )
    assert output.read_bytes() == b"before"


def test_receipt_source_requires_exactly_seven_arrays(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    shard_path = root / receipts[0].relative_path
    source = torch.load(shard_path, map_location="cpu", weights_only=False)
    source.append(torch.zeros((2, 1), dtype=torch.float64))
    torch.save(source, shard_path)
    altered = dataclasses.replace(
        receipts[0],
        file_sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "extra-array",
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (0, (2, 2, 6)),
        (1, (2, 1, 2)),
        (2, (2, 1, 4)),
        (3, (2, 1, 3)),
        (4, (2, 2, 2)),
        (5, (2, 2, 3)),
        (6, (2, 3)),
    ],
)
def test_receipt_source_requires_exact_real_shard_shapes(
    tmp_path, monkeypatch, bad_shape
):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    shard_path = root / receipts[0].relative_path
    source = torch.load(shard_path, map_location="cpu", weights_only=False)
    source[bad_shape[0]] = torch.zeros(bad_shape[1], dtype=torch.float64)
    torch.save(source, shard_path)
    altered = dataclasses.replace(
        receipts[0],
        file_sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    )
    output = tmp_path / "bad-shape"
    output.write_bytes(b"before")
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            output,
            [cases[0]],
            source_root=root,
            source_receipts=[altered],
        )
    assert output.read_bytes() == b"before"


def test_source_verification_loads_the_verified_bytes_once(tmp_path, monkeypatch):
    root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    shard_path = root / receipts[0].relative_path
    original_bytes = shard_path.read_bytes()
    replacement = torch.load(
        io.BytesIO(original_bytes), map_location="cpu", weights_only=False
    )
    replacement[0] = replacement[0].clone()
    replacement[0][0, 0, 0] = 9999.0
    replacement_path = tmp_path / "replacement.th"
    torch.save(replacement, replacement_path)
    original_load = topology_data.torch.load

    def load_wrapper(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert kwargs.get("weights_only") is True
        shard_path.write_bytes(replacement_path.read_bytes())
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(topology_data.torch, "load", load_wrapper)
    output = tmp_path / "verified-bytes.jsonl"
    save_sanitized_corpus(
        output,
        [cases[0]],
        source_root=root,
        source_receipts=[receipts[0]],
    )
    assert load_sanitized_corpus(output)[0]["area"][0] == 198.0
    assert original_bytes != shard_path.read_bytes()


@pytest.mark.skipif(
    not Path("FloorSet/floorset_lite/worker_26/layouts_5488.th").is_file(),
    reason="real training shard is unavailable",
)
def test_real_training_shard_row_zero_is_receipt_bound(tmp_path, monkeypatch):
    root = Path("FloorSet/floorset_lite").resolve()
    shard_path = root / "worker_26/layouts_5488.th"
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    source = torch.load(shard_path, map_location="cpu", weights_only=False)
    raw_case = BandFileSampler._instance(source, 0)
    receipt = CorpusSourceReceipt(
        relative_path="worker_26/layouts_5488.th",
        file_sha256=digest,
        layout_index=0,
        fingerprint="0" * 64,
    )
    case = dict(raw_case, instance_id=source_instance_id(receipt))
    receipt = dataclasses.replace(receipt, fingerprint=fingerprint_case(case))
    monkeypatch.setattr(topology_data, "_CANONICAL_ROOT", root)
    output = tmp_path / "real.jsonl"
    save_sanitized_corpus(
        output,
        [case],
        source_root=root,
        source_receipts=[receipt],
    )
    row = load_sanitized_corpus(output)[0]
    assert row["n"] == 73
    assert len(row["cons"]) == 73
    assert len(row["cons"][0]) == 5
    assert len(row["b2b"]) == 319
    assert len(row["p2b"]) == 1656
    assert len(row["pins"]) == 255
    for constraints, target in zip(row["cons"], row["tp"]):
        fixed, preplaced = constraints[:2]
        if preplaced:
            assert target[0] >= 0 and target[1] >= 0
        else:
            assert target[0] == -1 and target[1] == -1
        if fixed or preplaced:
            assert target[2] > 0 and target[3] > 0
        else:
            assert target == [-1.0, -1.0, -1.0, -1.0]


def test_sanitize_excludes_golden_and_masks_non_input_geometry(tmp_path):
    case = _case(golden=[[99.0, 99.0, 99.0, 99.0]])
    _save(tmp_path / "c.jsonl", [case])
    row = load_sanitized_corpus(tmp_path / "c.jsonl")[0]
    assert set(row) == {
        "instance_id",
        "n",
        "area",
        "cons",
        "tp",
        "b2b",
        "p2b",
        "pins",
        "hpwl_ref",
        "area_ref",
    }
    assert row["tp"] == [
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, 3.0, 4.0],
        [5.0, 6.0, 5.0, 6.0],
    ]
    assert "golden" not in row and "test_id" not in row


def test_corpus_rejects_duplicates_bad_values_and_provenance(tmp_path):
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(), _case()])
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(area=[float("nan"), 1, 1])])
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(cons=[[0, 0]])])
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(source_split="validation")])


def test_fingerprint_and_split_are_deterministic():
    assert fingerprint_case(_case()) == fingerprint_case(_case())
    assert fingerprint_case(_case(instance_id="other-id")) == fingerprint_case(_case())
    assert split_for_id("abc") in {"train", "heldout"}
    assert split_for_id("abc") == split_for_id("abc")


def test_rejects_test_id_and_explicit_none_root(tmp_path):
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(test_id=1)])
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "x", [_case()], source_root=None, source_receipts=[]
        )


def test_duplicate_content_under_distinct_ids_and_invalid_split():
    with pytest.raises(ValueError):
        _save("/tmp/dup.jsonl", [_case(), _case(instance_id="other")])
    for args in (("", 10), ("x", True), ("x", 0)):
        with pytest.raises(ValueError):
            split_for_id(*args)


@pytest.mark.parametrize("bad", [None, "abc", {"x": 1}, [[1, 2]]])
def test_nested_schema_errors_are_value_errors(tmp_path, bad):
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(b2b=bad)])


def test_geometry_indices_and_recursive_canonical_forbidden(tmp_path):
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(b2b=[[True, 1, 1]])])
    with pytest.raises(ValueError):
        _save(tmp_path / "x", [_case(b2b=[[0, 9, 1]])])
    with pytest.raises(ValueError):
        canonical_jsonl(tmp_path / "x", [{"nested": [{"golden": 1}]}])


def test_manifest_hash_and_collate_contract(tmp_path):
    p = tmp_path / "x.jsonl"
    canonical_jsonl(p, [{"b": 1, "a": 2}])
    assert canonical_jsonl_sha256(p) == canonical_jsonl_sha256(p)
    assert write_sha256_manifest(tmp_path / "m", [p])[str(p)] == canonical_jsonl_sha256(
        p
    )
    label = TopologyLabel(
        "x", 2, 1, 1.0, 1.0, 1.0, (SparseEdge(0, 1, 0, 1.0, "sep", 1.0),), (), ()
    )
    batch = collate_labels([label], torch.device("cpu"), torch.float32)
    assert batch.edge_batch.shape == (1,) and batch.edge_weight.dtype == torch.float32
    empty = collate_labels([], torch.device("cpu"), torch.float64)
    assert all(getattr(empty, f).shape == (0,) for f in empty.__dataclass_fields__)


def test_schema_field_order_and_frozen_contract():
    assert [f.name for f in fields(SparseEdge)] == [
        "src",
        "dst",
        "axis",
        "margin",
        "kind",
        "weight",
    ]
    assert [f.name for f in fields(ContactLabel)] == [
        "a",
        "b",
        "axis",
        "a_before_b",
        "perp_margin",
        "weight",
    ]
    assert [f.name for f in fields(TopologyLabel)] == [
        "instance_id",
        "n",
        "sample_seed",
        "teacher_cost",
        "base_cost",
        "record_weight",
        "edges",
        "contacts",
        "pin_paths",
    ]
    with pytest.raises(FrozenInstanceError):
        SparseEdge(0, 1, 0, 1.0, "x", 1.0).src = 2


def test_five_column_constraints_are_retained_and_two_column_masks(tmp_path):
    row = _case(cons=[[0, 0, 2, 7, 10], [0, 1, 2, 7, 5], [1, 0, 0, 0, 0]])
    _save(tmp_path / "five", [row])
    got = load_sanitized_corpus(tmp_path / "five")[0]
    assert got["cons"] == row["cons"]
    assert got["tp"] == [
        [-1.0, -1.0, -1.0, -1.0],
        [9.0, 9.0, 3.0, 4.0],
        [-1.0, -1.0, 5.0, 6.0],
    ]


@pytest.mark.parametrize(
    "cons",
    [
        [[0, 0, -1, 0, 0]],
        [[0, 0, 1.5, 0, 0]],
        [[0, 0, 1, 0, 16]],
        [[0, 0, 1, -2, 0]],
    ],
)
def test_five_column_group_and_boundary_validation(tmp_path, cons):
    one_case = _case(
        n=1,
        area=[1.0],
        tp=[[1.0, 1.0, 1.0, 1.0]],
        cons=cons,
    )
    with pytest.raises(ValueError):
        _save(tmp_path / "bad", [one_case])


def test_empty_case_cannot_be_bound_to_a_training_shard(tmp_path):
    row = _case(n=0, area=[], cons=[], tp=[], b2b=[], p2b=[], pins=[])
    with pytest.raises(ValueError):
        _save(tmp_path / "empty", [row])


@pytest.mark.parametrize(
    "root", [None, "/definitely/wrong", "wrong-relative", object()]
)
def test_source_root_restrictions(tmp_path, root):
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "x", [_case()], source_root=root, source_receipts=[]
        )


def test_explicit_canonical_source_root(tmp_path, monkeypatch):
    fake_root, cases, receipts = _fake_bound_source(tmp_path, monkeypatch)
    save_sanitized_corpus(
        tmp_path / "x", cases[:1], source_root=fake_root, source_receipts=receipts[:1]
    )


@pytest.mark.parametrize(
    "row",
    [{"test_id": 1}, {"validation": True}, {"loader": {}}, {"provenance": {}}],
)
def test_load_rejects_forbidden_rows(tmp_path, row):
    p = tmp_path / "bad"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_sanitized_corpus(p)


def test_canonical_order_and_recursive_provenance_no_partial(tmp_path):
    p = tmp_path / "x"
    canonical_jsonl(p, [{"z": 1, "a": [2]}])
    before = p.read_bytes()
    assert before == b'{"a":[2],"z":1}\n'
    with pytest.raises(ValueError):
        canonical_jsonl(p, [{"nested": {"validation": 1}}])
    assert p.read_bytes() == before


def test_manifest_order_duplicate_and_no_partial(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    out = tmp_path / "m"
    result = write_sha256_manifest(out, [b, a])
    assert list(result) == sorted(result)
    before = out.read_bytes()
    with pytest.raises(ValueError):
        write_sha256_manifest(out, [a, a])
    assert out.read_bytes() == before


def test_collate_exact_contact_values_and_effective_weights():
    label = TopologyLabel(
        "x",
        3,
        2,
        2.0,
        3.0,
        4.0,
        (SparseEdge(0, 2, 1, 1.5, "pin", 0.5),),
        (ContactLabel(1, 2, 0, False, 2.0, 0.25),),
        ((0, 1, 2),),
    )
    batch = collate_labels([label], torch.device("cpu"), torch.float64)
    assert batch.edge_src.tolist() == [0] and batch.edge_weight.tolist() == [2.0]
    assert batch.contact_a.tolist() == [1]
    assert batch.contact_order.tolist() == [0]
    assert batch.contact_weight.tolist() == [1.0]
    assert all(not tensor.requires_grad for tensor in batch.__dict__.values())


def test_collate_multi_label_all_fields_order_dtype_device_and_weights():
    labels = [
        TopologyLabel(
            "first",
            3,
            4,
            2.0,
            3.0,
            2.0,
            (SparseEdge(0, 1, 0, 1.5, "a", 0.5),),
            (ContactLabel(1, 2, 1, True, 0.25, 0.75),),
            (),
        ),
        TopologyLabel(
            "second",
            2,
            5,
            2.0,
            3.0,
            3.0,
            (SparseEdge(1, 0, 1, 2.5, "b", 0.25),),
            (ContactLabel(0, 1, 0, False, 0.5, 0.5),),
            (),
        ),
    ]
    batch = collate_labels(labels, torch.device("cpu"), torch.float64)
    integer_fields = (
        "edge_batch",
        "edge_src",
        "edge_dst",
        "edge_axis",
        "contact_batch",
        "contact_a",
        "contact_b",
        "contact_axis",
        "contact_order",
    )
    float_fields = (
        "edge_margin",
        "edge_weight",
        "contact_margin",
        "contact_weight",
    )
    expected = {
        "edge_batch": [0, 1],
        "edge_src": [0, 1],
        "edge_dst": [1, 0],
        "edge_axis": [0, 1],
        "edge_margin": [1.5, 2.5],
        "edge_weight": [1.0, 0.75],
        "contact_batch": [0, 1],
        "contact_a": [1, 0],
        "contact_b": [2, 1],
        "contact_axis": [1, 0],
        "contact_order": [1, 0],
        "contact_margin": [0.25, 0.5],
        "contact_weight": [1.5, 1.5],
    }
    for name in integer_fields:
        assert getattr(batch, name).dtype == torch.long
        assert getattr(batch, name).tolist() == expected[name]
    for name in float_fields:
        tensor = getattr(batch, name)
        assert tensor.dtype == torch.float64
        assert tensor.tolist() == pytest.approx(expected[name])
    for tensor in batch.__dict__.values():
        assert tensor.device == torch.device("cpu")
        assert not tensor.requires_grad


def test_collate_empty_all_fields_have_expected_shapes_and_types():
    batch = collate_labels([], torch.device("cpu"), torch.float32)
    for name in (
        "edge_batch",
        "edge_src",
        "edge_dst",
        "edge_axis",
        "contact_batch",
        "contact_a",
        "contact_b",
        "contact_axis",
        "contact_order",
    ):
        assert getattr(batch, name).shape == (0,)
        assert getattr(batch, name).dtype == torch.long
    for name in ("edge_margin", "edge_weight", "contact_margin", "contact_weight"):
        assert getattr(batch, name).shape == (0,)
        assert getattr(batch, name).dtype == torch.float32


@pytest.mark.parametrize(
    "label",
    [
        TopologyLabel("", 1, 1, 1.0, 1.0, 1.0, (), (), ()),
        TopologyLabel("x", 1, 1, 1.0, 1.0, 0.0, (), (), ()),
        TopologyLabel("x", 1, True, 1.0, 1.0, 1.0, (), (), ()),
    ],
)
def test_collate_rejects_label_scalars(label):
    with pytest.raises(ValueError):
        collate_labels([label], torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("bad", ["abc", {"x": 1}, None, 4])
def test_sequence_inputs_reject_non_sequences(tmp_path, bad):
    with pytest.raises(ValueError):
        save_sanitized_corpus(
            tmp_path / "bad",
            bad,
            source_root=CANONICAL_ROOT,
            source_receipts=[],
        )


def test_source_root_is_required():
    with pytest.raises(TypeError):
        save_sanitized_corpus("unused", [_case()])


def test_tensor_inputs_and_floor_set_adapter_orientation(tmp_path):
    raw_fp = torch.tensor([[22.0, 9.0, 40.0, 45.0], [22.0, 9.0, 40.0, 45.0]])
    rects = [
        (float(row[2]), float(row[3]), float(row[0]), float(row[1])) for row in raw_fp
    ]
    cons = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    expected = target_positions_from_rects(rects, cons, 2)
    row = _case(
        n=2,
        area=torch.tensor([198.0, 198.0]),
        cons=cons,
        tp=expected,
        b2b=torch.empty((0, 3)),
        p2b=torch.empty((0, 3)),
        pins=torch.empty((0, 2)),
    )
    _save(tmp_path / "tensor", [row])
    got = load_sanitized_corpus(tmp_path / "tensor")[0]
    assert got["tp"] == [[-1.0, -1.0, 22.0, 9.0], [40.0, 45.0, 22.0, 9.0]]


def test_tensor_integral_constraint_and_endpoint_values_are_normalized(tmp_path):
    row = _case(
        cons=torch.tensor([[0.0, 0.0, 2.0, 7.0, 10.0]] * 3),
        b2b=torch.tensor([[0.0, 1.0, 2.0]]),
        p2b=torch.tensor([[0.0, 1.0, 0.5]]),
        pins=torch.tensor([[2.0, 3.0]]),
    )
    _save(tmp_path / "integral", [row])
    got = load_sanitized_corpus(tmp_path / "integral")[0]
    assert got["cons"] == [[0, 0, 2, 7, 10]] * 3
    assert got["b2b"] == [[0, 1, 2.0]]
    assert got["p2b"] == [[0, 1, 0.5]]


def test_requires_grad_tensor_is_rejected(tmp_path):
    area = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(ValueError):
        _save(tmp_path / "requires-grad", [_case(area=area)])


def test_load_rejects_sanitized_artifact_contamination(tmp_path):
    row = _case()
    p = tmp_path / "contaminated"
    p.write_text(json.dumps({**row, "golden": [[1, 2, 3, 4]]}) + "\n")
    with pytest.raises(ValueError):
        load_sanitized_corpus(p)


def test_canonical_generator_is_rejected_without_partial_write(tmp_path):
    p = tmp_path / "canonical"
    p.write_bytes(b"before")
    with pytest.raises(ValueError):
        canonical_jsonl(p, ({"x": i} for i in range(2)))
    assert p.read_bytes() == b"before"


def test_manifest_generator_and_path_errors_are_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest = tmp_path / "manifest"
    result = write_sha256_manifest(manifest, (path for path in (second, first)))
    assert list(result) == sorted(result)
    with pytest.raises(ValueError):
        write_sha256_manifest(manifest, (object() for _ in range(1)))


def test_collate_rejects_nonfloating_dtype_and_weight_overflow():
    label = TopologyLabel(
        "x",
        2,
        1,
        1.0,
        1.0,
        1.0,
        (SparseEdge(0, 1, 0, 1.0, "sep", 1.0),),
        (),
        (),
    )
    with pytest.raises(ValueError):
        collate_labels([label], torch.device("cpu"), torch.long)
    huge = TopologyLabel(
        "x",
        2,
        1,
        1.0,
        1.0,
        1e308,
        (SparseEdge(0, 1, 0, 1.0, "sep", 1e308),),
        (),
        (),
    )
    with pytest.raises(ValueError):
        collate_labels([huge], torch.device("cpu"), torch.float32)

    cast_overflow = TopologyLabel(
        "x",
        2,
        1,
        1.0,
        1.0,
        1e20,
        (SparseEdge(0, 1, 0, 1.0, "sep", 1e20),),
        (),
        (),
    )
    with pytest.raises(ValueError):
        collate_labels([cast_overflow], torch.device("cpu"), torch.float32)

    with pytest.raises(ValueError):
        collate_labels([label], torch.device("meta"), torch.float32)


def test_collate_rejects_float8_dtypes():
    label = TopologyLabel(
        "x",
        2,
        1,
        1.0,
        1.0,
        1.0,
        (SparseEdge(0, 1, 0, 1.0, "sep", 1.0),),
        (),
        (),
    )
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    ):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            with pytest.raises(ValueError):
                collate_labels([label], torch.device("cpu"), dtype)


def test_collate_rejects_tensor_record_weight():
    label = TopologyLabel(
        "x",
        2,
        1,
        1.0,
        1.0,
        torch.tensor(1.0),
        (),
        (),
        (),
    )
    with pytest.raises(ValueError):
        collate_labels([label], torch.device("cpu"), torch.float32)


def test_dataclass_annotations_and_frozen_contracts():
    assert get_type_hints(SparseEdge) == {
        "src": int,
        "dst": int,
        "axis": int,
        "margin": float,
        "kind": str,
        "weight": float,
    }
    assert get_type_hints(ContactLabel) == {
        "a": int,
        "b": int,
        "axis": int,
        "a_before_b": bool,
        "perp_margin": float,
        "weight": float,
    }
    edges_type = get_type_hints(TopologyLabel)["edges"]
    assert get_origin(edges_type) is tuple
    assert get_args(edges_type) == (SparseEdge, Ellipsis)
    assert all(
        annotation is torch.Tensor
        for annotation in get_type_hints(SparseTopologyBatch).values()
    )
    for cls in (SparseEdge, ContactLabel, TopologyLabel, SparseTopologyBatch):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen


@pytest.mark.parametrize(
    "edge",
    [
        object(),
        SparseEdge(True, 1, 0, 0.0, "x", 1.0),
        SparseEdge(0, 1, 2, 0.0, "x", 1.0),
        SparseEdge(0, 2, 0, 0.0, "x", 1.0),
        SparseEdge(0, 0, 0, 0.0, "x", 1.0),
        SparseEdge(0, 1, 0, -1.0, "x", 1.0),
        SparseEdge(0, 1, 0, float("nan"), "x", 1.0),
        SparseEdge(0, 1, 0, 0.0, " ", 1.0),
        SparseEdge(0, 1, 0, 0.0, "x", 0.0),
        SparseEdge(0, 1, 0, 0.0, "x", float("inf")),
    ],
)
def test_collate_rejects_malformed_edges(edge):
    label = TopologyLabel("x", 2, 1, 1.0, 1.0, 1.0, (edge,), (), ())
    with pytest.raises(ValueError):
        collate_labels([label], torch.device("cpu"), torch.float32)


@pytest.mark.parametrize(
    "contact",
    [
        object(),
        ContactLabel(True, 1, 0, True, 1.0, 1.0),
        ContactLabel(0, 2, 0, True, 1.0, 1.0),
        ContactLabel(0, 0, 0, True, 1.0, 1.0),
        ContactLabel(0, 1, 2, True, 1.0, 1.0),
        ContactLabel(0, 1, 0, 1, 1.0, 1.0),
        ContactLabel(0, 1, 0, True, 0.0, 1.0),
        ContactLabel(0, 1, 0, True, float("nan"), 1.0),
        ContactLabel(0, 1, 0, True, 1.0, 0.0),
    ],
)
def test_collate_rejects_malformed_contacts(contact):
    label = TopologyLabel("x", 2, 1, 1.0, 1.0, 1.0, (), (contact,), ())
    with pytest.raises(ValueError):
        collate_labels([label], torch.device("cpu"), torch.float32)


@pytest.mark.parametrize(
    "paths",
    [
        ([],),
        ((0, 0),),
        ((0, 2),),
        ((True, 1),),
        ((0, 1), [0, 1]),
    ],
)
def test_collate_rejects_malformed_pin_paths(paths):
    label = TopologyLabel("x", 2, 1, 1.0, 1.0, 1.0, (), (), paths)
    with pytest.raises(ValueError):
        collate_labels([label], torch.device("cpu"), torch.float32)


def test_collate_rejects_non_tuple_label_fields_and_nonfinite_costs():
    labels = [
        TopologyLabel("x", 2, 1, float("nan"), 1.0, 1.0, (), (), ()),
        TopologyLabel("x", 2, 1, 1.0, 1.0, 1.0, [], (), ()),
        TopologyLabel("x", 2, 1, 1.0, 1.0, 1.0, (), [], ()),
        TopologyLabel("x", 2, 1, 1.0, 1.0, 1.0, (), (), []),
    ]
    for label in labels:
        with pytest.raises(ValueError):
            collate_labels([label], torch.device("cpu"), torch.float32)


def test_zero_hpwl_and_self_b2b_allowed(tmp_path):
    row = _case(hpwl_ref=0.0, b2b=[[0, 0, 0.0]])
    _save(tmp_path / "ok", [row])
    assert load_sanitized_corpus(tmp_path / "ok")[0]["hpwl_ref"] == 0.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "0"])
def test_bad_hpwl_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        _save(tmp_path / "bad", [_case(hpwl_ref=bad)])


def test_canonical_nonfinite_unserializable_and_no_partial(tmp_path):
    p = tmp_path / "x"
    p.write_text("original")
    for row in ({"x": float("nan")}, {"x": object()}):
        with pytest.raises(ValueError):
            canonical_jsonl(p, [row])
        assert p.read_text() == "original"


def test_manifest_missing_and_no_partial(tmp_path):
    p = tmp_path / "manifest"
    p.write_text("original")
    with pytest.raises(ValueError):
        write_sha256_manifest(p, [tmp_path / "missing"])
    assert p.read_text() == "original"


@pytest.mark.parametrize("bad", [True, 1.0, -1, "1"])
def test_split_invalid_matrix(bad):
    with pytest.raises(ValueError):
        split_for_id("x", bad)


@pytest.mark.parametrize("bad", [object(), None, {"src": 1}, "x"])
def test_collate_rejects_bad_labels_container(bad):
    with pytest.raises(ValueError):
        collate_labels(bad, torch.device("cpu"), torch.float32)


def test_collate_rejects_wrong_label_instance_type():
    with pytest.raises(ValueError):
        collate_labels([object()], torch.device("cpu"), torch.float32)


# P1-B RED: teacher-owned, in-memory candidate lifecycle contract.
def _p1b_case():
    return {
        "n": 2,
        "area": [1.0, 1.0],
        "cons": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "b2b": [[0, 1, 2.5]],
        "p2b": [[0, 1, 3.5]],
        "pins": [[7.0, 11.0]],
        "tp": [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
        "hpwl_ref": 13.0,
        "area_ref": 2.0,
    }


def _p1b_raw():
    return torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]],
        dtype=torch.float64,
    )


def _p1b_stream():
    return [
        ("base", _p1b_raw()),
        (
            "axis:0:1:0:0",
            torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [-2.0, 0.0, 1.0, 1.0]],
                dtype=torch.float64,
            ),
        ),
    ]


class _P1BScorer:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def evaluate_solution(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        result = self.results.pop(0) if self.results else (True, 1.0)
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, tuple):
            return __import__("types").SimpleNamespace(
                is_feasible=result[0], cost_no_runtime=result[1]
            )
        return result


def _p1b_run(
    monkeypatch,
    scorer,
    *,
    stream=None,
    admissions=None,
    hard_results=None,
    intent_results=None,
    energy_results=None,
    raw=None,
    case=None,
    cfg=None,
    generator_error=None,
    trace_out=None,
):
    t = _teacher()
    raw = _p1b_raw() if raw is None else raw
    case = _p1b_case() if case is None else case
    cfg = topology_prior.ProposalConfig() if cfg is None else cfg
    stream = _p1b_stream() if stream is None else stream
    raw_before = raw.clone() if isinstance(raw, torch.Tensor) else raw
    case_before = repr(case)
    trace = {
        "generator": [],
        "yielded": [],
        "admit": [],
        "hard": [],
        "intent": [],
        "energy": [],
    }
    if trace_out is not None:
        trace_out.update(trace)

    def generate(rects, received_case, received_cfg):
        trace["generator"].append(
            (rects, rects.clone(), repr(received_case), received_cfg)
        )
        if generator_error is not None:
            raise generator_error
        for name, template in stream:
            candidate = template.clone() if isinstance(template, torch.Tensor) else template
            trace["yielded"].append(candidate)
            yield name, candidate

    admission_queue = None if admissions is None else iter(admissions)

    def admit(proposal, received_case):
        trace["admit"].append((proposal, proposal.clone(), repr(received_case)))
        if admission_queue is None:
            return proposal.clone(), torch.zeros((proposal.shape[0], 2), dtype=torch.float64)
        result = next(admission_queue)
        if result == "pass":
            return proposal.clone(), torch.zeros((proposal.shape[0], 2), dtype=torch.float64)
        return result

    hard_queue = iter(hard_results if hard_results is not None else [{"legal": True}] * len(stream))

    def verify_hard(legal, received_case):
        trace["hard"].append((legal, legal.clone(), repr(received_case)))
        return next(hard_queue)

    intent_queue = iter(intent_results if intent_results is not None else [True] * len(stream))

    def intent(name, legal, received_case):
        trace["intent"].append((name, legal, legal.clone(), repr(received_case)))
        return next(intent_queue)

    energy_queue = iter(energy_results if energy_results is not None else [0.0] * len(stream))

    def diagnostic_energy(legal, received_case):
        trace["energy"].append((legal, legal.clone(), repr(received_case)))
        result = next(energy_queue)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(t, "_GENERATE_PROPOSALS", generate, raising=False)
    monkeypatch.setattr(t, "_ADMIT_PROPOSAL", admit, raising=False)
    monkeypatch.setattr(t, "_VERIFY_HARD_LEGAL", verify_hard, raising=False)
    monkeypatch.setattr(t, "_proposal_intent_holds", intent, raising=False)
    monkeypatch.setattr(t, "_DIAGNOSTIC_ENERGY", diagnostic_energy, raising=False)
    out = t._run_candidate_lifecycle(raw, case, scorer=scorer, cfg=cfg)
    if isinstance(raw_before, torch.Tensor):
        assert torch.equal(raw, raw_before)
    assert repr(case) == case_before
    return out, trace


def _p1b_assert_accounting(out):
    winners = [record for record in out.candidates if record.rejection_reason is None]
    if out.winner_ordinal is None:
        assert winners == []
    else:
        assert len(winners) == 1 and winners[0].ordinal == out.winner_ordinal
    assert all(
        record.rejection_reason is None
        or (isinstance(record.rejection_reason, str) and record.rejection_reason)
        for record in out.candidates
    )


def test_task4_p1b_normal_schema_order_immutability_and_accounting(monkeypatch):
    raw = _p1b_raw()
    case = _p1b_case()
    stream = _p1b_stream()
    raw_before = raw.clone()
    case_before = repr(case)
    scorer = _P1BScorer((True, 3.0), (True, 2.0))
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        raw=raw,
        case=case,
        stream=stream,
        energy_results=[-100.0, 100.0],
    )
    assert torch.equal(raw, raw_before) and repr(case) == case_before
    assert dataclasses.is_dataclass(out) and dataclasses.is_dataclass(out.candidates[0])
    assert [field.name for field in fields(out)] == [
        "candidates", "winner_ordinal", "base_cost", "teacher_cost"
    ]
    assert [field.name for field in fields(out.candidates[0])] == [
        "ordinal", "name", "original", "legal", "drift", "hard",
        "official_cost", "diagnostic_energy", "energy_status",
        "rejection_reason",
    ]
    assert isinstance(out.candidates, tuple)
    assert [record.ordinal for record in out.candidates] == [0, 1]
    assert [record.name for record in out.candidates] == ["base", "axis:0:1:0:0"]
    assert sum(record.name == "base" for record in out.candidates) == 1
    assert len(trace["generator"]) == 1
    generated_raw, generated_value, generated_case, generated_cfg = trace["generator"][0]
    assert generated_raw.data_ptr() == raw.data_ptr()
    assert torch.equal(generated_value, _p1b_raw())
    assert generated_case == repr(_p1b_case())
    assert generated_cfg == topology_prior.ProposalConfig()
    for index, record in enumerate(out.candidates):
        assert torch.equal(record.original, stream[index][1])
        assert record.original.data_ptr() != trace["yielded"][index].data_ptr()
        assert record.original.data_ptr() != generated_raw.data_ptr()
        assert record.original.data_ptr() != trace["admit"][index][0].data_ptr()
        assert record.legal.device.type == "cpu"
        assert record.legal.dtype == torch.float64
        assert record.drift == {"max_abs": 0.0}
        assert record.hard == {"legal": True}
    assert out.candidates[0].original.data_ptr() != out.candidates[1].original.data_ptr()
    assert out.winner_ordinal == 1 and out.base_cost == 3.0 and out.teacher_cost == 2.0
    assert out.candidates[0].rejection_reason == "not_selected"
    assert out.candidates[1].rejection_reason is None
    _p1b_assert_accounting(out)
    with pytest.raises(FrozenInstanceError):
        out.winner_ordinal = 0
    with pytest.raises(FrozenInstanceError):
        out.candidates[0].name = "changed"


@pytest.mark.parametrize(
    "stream",
    [
        [],
        [("axis:0:1:0:0", _p1b_raw())],
        [("axis:0:1:0:0", _p1b_raw()), ("base", _p1b_raw())],
        [("base", _p1b_raw()), ("base", _p1b_raw())],
        [("base", _p1b_raw()), ("axis:0:1:0:0", _p1b_raw()), ("axis:0:1:0:0", _p1b_raw())],
        [("base", _p1b_raw()), ("mutation", _p1b_raw())],
        [("base", _p1b_raw()), ("contact:1:0:1:0:1", _p1b_raw()), ("axis:0:1:0:0", _p1b_raw())],
        [("base", _p1b_raw()), ("pin:0:1:0:1", _p1b_raw()), ("axis:0:1:0:0", _p1b_raw())],
        [("base", _p1b_raw()), ("contact:1:0:1:0:1", _p1b_raw()), ("pin:0:1:0:1", _p1b_raw())],
        [("", _p1b_raw())],
        [(1, _p1b_raw())],
        ["not-a-pair"],
        [("base", [[0.0, 0.0, 1.0, 1.0]])],
        [("base", torch.ones((1, 4), dtype=torch.float64))],
        [("base", torch.ones((2, 4), dtype=torch.float32))],
        [("base", torch.tensor([[0.0, 0.0, 1.0, 1.0], [float("nan"), 0.0, 1.0, 1.0]], dtype=torch.float64))],
        [("base", torch.tensor([[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 1.0, 1.0]], dtype=torch.float64))],
    ],
)
def test_task4_p1b_invalid_generator_stream_fails_before_candidate_work(monkeypatch, stream):
    scorer = _P1BScorer()
    trace = {}
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _p1b_run(monkeypatch, scorer, stream=stream, trace_out=trace)
    assert scorer.calls == []
    assert trace["admit"] == trace["hard"] == trace["intent"] == trace["energy"] == []


def test_task4_p1b_generator_exception_fails_closed(monkeypatch):
    scorer = _P1BScorer()
    trace = {}
    with pytest.raises((ValueError, RuntimeError)):
        _p1b_run(
            monkeypatch,
            scorer,
            generator_error=RuntimeError("generator"),
            trace_out=trace,
        )
    assert scorer.calls == []
    assert trace["admit"] == trace["hard"] == trace["intent"] == trace["energy"] == []


@pytest.mark.parametrize(
    "raw,case,cfg",
    [
        (torch.ones((1, 4), dtype=torch.float64), _p1b_case(), topology_prior.ProposalConfig()),
        (_p1b_raw().float(), _p1b_case(), topology_prior.ProposalConfig()),
        (torch.tensor([[0.0, 0.0, 1.0, 1.0], [float("inf"), 0.0, 1.0, 1.0]], dtype=torch.float64), _p1b_case(), topology_prior.ProposalConfig()),
        (torch.tensor([[0.0, 0.0, -1.0, 1.0], [2.0, 0.0, 1.0, 1.0]], dtype=torch.float64), _p1b_case(), topology_prior.ProposalConfig()),
        (_p1b_raw(), {**_p1b_case(), "area": [1.0]}, topology_prior.ProposalConfig()),
        (_p1b_raw(), _p1b_case(), object()),
    ],
)
def test_task4_p1b_invalid_input_fails_before_generation(monkeypatch, raw, case, cfg):
    scorer = _P1BScorer()
    trace = {}
    with pytest.raises((TypeError, ValueError)):
        _p1b_run(
            monkeypatch,
            scorer,
            raw=raw,
            case=case,
            cfg=cfg,
            trace_out=trace,
        )
    assert scorer.calls == []
    assert all(trace[key] == [] for key in trace)


def test_task4_p1b_admission_failure_is_opaque_and_processing_continues(monkeypatch):
    scorer = _P1BScorer((True, 2.0))
    out, trace = _p1b_run(monkeypatch, scorer, admissions=["pass", None], energy_results=[7.0])
    assert len(trace["admit"]) == 2
    assert len(trace["hard"]) == len(trace["intent"]) == len(trace["energy"]) == 1
    assert len(scorer.calls) == 1
    assert out.winner_ordinal == 0 and out.base_cost == out.teacher_cost == 2.0
    assert out.candidates[0].rejection_reason is None
    assert out.candidates[1].rejection_reason == "admission_failed"
    assert out.candidates[1].energy_status == "not_reached"
    _p1b_assert_accounting(out)


@pytest.mark.parametrize(
    "hard_result",
    [
        {"legal": False},
        {},
        {"legal": "true"},
        {"legal": 1},
        {"legal": True, "evidence": 1},
        {"legal": True, "evidence": False},
    ],
)
def test_task4_p1b_hard_failure_never_reaches_intent_score_or_energy(monkeypatch, hard_result):
    scorer = _P1BScorer((True, 2.0))
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        hard_results=[{"legal": True}, hard_result],
        intent_results=[True],
        energy_results=[5.0],
    )
    assert len(trace["hard"]) == 2
    assert len(trace["intent"]) == len(scorer.calls) == len(trace["energy"]) == 1
    assert out.candidates[1].rejection_reason == "hard_audit_failed"
    assert out.candidates[1].energy_status == "not_reached"
    _p1b_assert_accounting(out)


def test_task4_p1b_intent_loss_never_reaches_score_or_energy(monkeypatch):
    scorer = _P1BScorer((True, 2.0))
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        intent_results=[True, False],
        energy_results=[5.0],
    )
    assert len(trace["intent"]) == 2
    assert len(scorer.calls) == len(trace["energy"]) == 1
    assert out.candidates[1].rejection_reason == "intent_not_survived"
    assert out.candidates[1].energy_status == "not_reached"
    _p1b_assert_accounting(out)


def test_task4_p1b_all_downstream_stages_use_admitted_legal_geometry(monkeypatch):
    original = _p1b_raw()
    legal = torch.tensor(
        [[10.0, 5.0, 1.0, 1.0], [12.0, 5.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    scorer = _P1BScorer((True, 2.0))
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        stream=[("base", original)],
        admissions=[(legal, torch.zeros((2, 2), dtype=torch.float64))],
        energy_results=[9.0],
    )
    assert torch.equal(out.candidates[0].original, original)
    assert torch.equal(out.candidates[0].legal, legal)
    assert out.candidates[0].original.data_ptr() != out.candidates[0].legal.data_ptr()
    assert torch.equal(trace["hard"][0][1], legal)
    assert torch.equal(trace["intent"][0][2], legal)
    assert torch.equal(trace["energy"][0][1], legal)
    args, kwargs = scorer.calls[0]
    assert args[0]["positions"] == legal.tolist()
    assert kwargs == {"median_runtime": 1.0}


@pytest.mark.parametrize(
    "scorer",
    [
        lambda *args, **kwargs: None,
        object(),
        __import__("types").SimpleNamespace(evaluate_solution=1),
    ],
)
def test_task4_p1b_scorer_requires_callable_evaluate_solution_before_generation(monkeypatch, scorer):
    trace = {}
    with pytest.raises((TypeError, ValueError)):
        _p1b_run(monkeypatch, scorer, trace_out=trace)
    assert all(trace[key] == [] for key in trace)


def test_task4_p1b_official_evaluator_adapter_is_exact(monkeypatch):
    scorer = _P1BScorer((True, 2.0))
    out, _trace = _p1b_run(
        monkeypatch,
        scorer,
        stream=[("base", _p1b_raw())],
        energy_results=[0.0],
    )
    assert out.winner_ordinal == 0
    assert len(scorer.calls) == 1
    args, kwargs = scorer.calls[0]
    assert len(args) == 8 and kwargs == {"median_runtime": 1.0}
    solution, baseline, cons, b2b, p2b, pins, area, tp = args
    assert solution == {
        "positions": [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]],
        "runtime": 1.0,
    }
    assert all(type(value) is float for row in solution["positions"] for value in row)
    assert baseline == {"hpwl_baseline": 13.0, "area_baseline": 2.0}
    assert cons.device.type == "cpu" and cons.dtype == torch.int64
    assert tuple(cons.shape) == (2, 5) and cons.tolist() == _p1b_case()["cons"]
    expected = [
        (b2b, (1, 3), _p1b_case()["b2b"]),
        (p2b, (1, 3), _p1b_case()["p2b"]),
        (pins, (1, 2), _p1b_case()["pins"]),
        (area, (2,), _p1b_case()["area"]),
    ]
    for tensor, shape, values in expected:
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float64
        assert tuple(tensor.shape) == shape and tensor.tolist() == values
    assert tp == _p1b_case()["tp"]


def test_task4_p1b_official_cost_beats_inverted_energy(monkeypatch):
    scorer = _P1BScorer((True, 3.0), (True, 2.0))
    out, trace = _p1b_run(
        monkeypatch, scorer, energy_results=[-100.0, 100.0]
    )
    assert len(trace["energy"]) == 2
    assert [record.official_cost for record in out.candidates] == [3.0, 2.0]
    assert [record.diagnostic_energy for record in out.candidates] == [-100.0, 100.0]
    assert [record.energy_status for record in out.candidates] == ["recorded", "recorded"]
    assert out.winner_ordinal == 1
    assert out.candidates[0].rejection_reason == "not_selected"
    assert out.candidates[1].rejection_reason is None
    _p1b_assert_accounting(out)


@pytest.mark.parametrize("mutation_cost", [2.0, 3.0])
def test_task4_p1b_tie_or_worse_mutation_keeps_base(monkeypatch, mutation_cost):
    scorer = _P1BScorer((True, 2.0), (True, mutation_cost))
    out, _trace = _p1b_run(monkeypatch, scorer, energy_results=[10.0, -10.0])
    assert out.winner_ordinal == 0
    assert out.base_cost == out.teacher_cost == 2.0
    assert out.candidates[0].rejection_reason is None
    assert out.candidates[1].official_cost == mutation_cost
    assert out.candidates[1].rejection_reason == "not_selected"
    _p1b_assert_accounting(out)


_P1B_MISSING_FEASIBLE = __import__("types").SimpleNamespace(cost_no_runtime=1.0)
_P1B_MISSING_COST = __import__("types").SimpleNamespace(is_feasible=True)


@pytest.mark.parametrize(
    "result,reason",
    [
        (RuntimeError("scorer"), "official_evaluator_error"),
        (None, "official_evaluator_error"),
        (object(), "official_evaluator_error"),
        ({"is_feasible": True, "cost_no_runtime": 1.0}, "official_evaluator_error"),
        (_P1B_MISSING_FEASIBLE, "official_evaluator_error"),
        (_P1B_MISSING_COST, "official_evaluator_error"),
        ((None, 1.0), "official_evaluator_error"),
        ((1, 1.0), "official_evaluator_error"),
        ((False, 1.0), "official_infeasible"),
        ((True, True), "official_invalid_cost"),
        ((True, "1"), "official_invalid_cost"),
        ((True, float("nan")), "official_invalid_cost"),
        ((True, float("inf")), "official_invalid_cost"),
        ((True, 0.0), "official_invalid_cost"),
        ((True, -1.0), "official_invalid_cost"),
    ],
)
def test_task4_p1b_official_failure_has_one_reason_and_no_energy(monkeypatch, result, reason):
    scorer = _P1BScorer((True, 3.0), result)
    out, trace = _p1b_run(monkeypatch, scorer, energy_results=[4.0])
    assert len(scorer.calls) == 2
    assert len(trace["energy"]) == 1
    assert out.candidates[0].diagnostic_energy == 4.0
    assert out.candidates[1].official_cost is None
    assert out.candidates[1].diagnostic_energy is None
    assert out.candidates[1].energy_status == "not_reached"
    assert out.candidates[1].rejection_reason == reason
    _p1b_assert_accounting(out)


@pytest.mark.parametrize(
    "energy_result",
    [RuntimeError("energy"), float("nan"), float("inf"), True, "bad"],
)
def test_task4_p1b_unavailable_energy_does_not_block_scored_winner(monkeypatch, energy_result):
    scorer = _P1BScorer((True, 1.0))
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        stream=[("base", _p1b_raw())],
        energy_results=[energy_result],
    )
    assert len(scorer.calls) == len(trace["energy"]) == 1
    assert out.winner_ordinal == 0 and out.base_cost == out.teacher_cost == 1.0
    assert out.candidates[0].official_cost == 1.0
    assert out.candidates[0].diagnostic_energy is None
    assert out.candidates[0].energy_status == "unavailable"
    assert out.candidates[0].rejection_reason is None
    _p1b_assert_accounting(out)


def test_task4_p1b_base_unavailable_keeps_scored_mutation_evidence(monkeypatch):
    scorer = _P1BScorer((True, 2.0))
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        admissions=[None, "pass"],
        energy_results=[7.0],
    )
    assert len(trace["admit"]) == 2
    assert len(trace["hard"]) == len(trace["intent"]) == 1
    assert len(scorer.calls) == len(trace["energy"]) == 1
    assert torch.equal(trace["energy"][0][1], _p1b_stream()[1][1])
    assert out.winner_ordinal is None
    assert out.base_cost is None and out.teacher_cost is None
    assert out.candidates[0].official_cost is None
    assert out.candidates[0].rejection_reason == "admission_failed"
    assert out.candidates[1].official_cost == 2.0
    assert out.candidates[1].diagnostic_energy == 7.0
    assert out.candidates[1].energy_status == "recorded"
    assert out.candidates[1].rejection_reason == "base_unavailable"
    _p1b_assert_accounting(out)


@pytest.mark.parametrize(
    "base_failure,expected_reason",
    [
        ("hard", "hard_audit_failed"),
        ("intent", "intent_not_survived"),
        ("official", "official_evaluator_error"),
    ],
)
def test_task4_p1b_base_late_failure_keeps_mutation_evidence_without_winner(
    monkeypatch, base_failure, expected_reason
):
    scorer_results = (
        [RuntimeError("base scorer"), (True, 2.0)]
        if base_failure == "official"
        else [(True, 2.0)]
    )
    hard_results = (
        [{"legal": False}, {"legal": True}]
        if base_failure == "hard"
        else [{"legal": True}, {"legal": True}]
    )
    intent_results = (
        [False, True]
        if base_failure == "intent"
        else [True] if base_failure == "hard" else [True, True]
    )
    scorer = _P1BScorer(*scorer_results)
    out, trace = _p1b_run(
        monkeypatch,
        scorer,
        hard_results=hard_results,
        intent_results=intent_results,
        energy_results=[7.0],
    )
    expected_scorer_calls = 2 if base_failure == "official" else 1
    assert len(scorer.calls) == expected_scorer_calls
    assert len(trace["energy"]) == 1
    assert out.winner_ordinal is None
    assert out.base_cost is None and out.teacher_cost is None
    assert out.candidates[0].rejection_reason == expected_reason
    assert out.candidates[1].official_cost == 2.0
    assert out.candidates[1].diagnostic_energy == 7.0
    assert out.candidates[1].rejection_reason == "base_unavailable"
    _p1b_assert_accounting(out)


# P1-B review RED contracts: production energy plumbing must use the complete
# collate schema, sanitized zero-edge cases must remain evaluable, and the
# lifecycle must reject malformed/over-cap generator streams before sinks.
def test_task4_p1b_energy_batch_is_exact_scaled_cpu_f64(monkeypatch):
    t = _teacher()
    case = _p1b_case() | {
        "area": [4.0, 5.0], "hpwl_ref": 7.0, "area_ref": 9.0,
    }
    seen = {}

    def energy(rects, batch):
        seen["rects"] = rects
        seen["batch"] = batch
        return {"E": torch.tensor([1.25], dtype=torch.float64)}

    monkeypatch.setattr(t._ENERGY, "energy", energy)
    assert t._DIAGNOSTIC_ENERGY(_p1b_raw(), case) == 1.25
    assert seen["rects"].shape == (1, 2, 4)
    assert seen["rects"].device.type == "cpu" and seen["rects"].dtype == torch.float64
    expected = {"area", "cons", "b2b", "p2b", "pins", "hpwl_ref", "area_ref",
                "scale", "n_soft", "tau_sharp", "tau_soft"}
    assert set(seen["batch"]) == expected
    batch = seen["batch"]
    assert torch.equal(batch["scale"], torch.tensor([3.0], dtype=torch.float64))
    assert torch.equal(batch["tau_sharp"], 1e-5 * batch["scale"])
    assert torch.equal(batch["tau_soft"], 1e-2 * batch["scale"])
    assert torch.equal(batch["area"], torch.tensor([[4.0, 5.0]], dtype=torch.float64))
    assert torch.equal(batch["hpwl_ref"], torch.tensor([7.0], dtype=torch.float64))
    assert torch.equal(batch["area_ref"], torch.tensor([9.0], dtype=torch.float64))
    assert torch.equal(batch["b2b"], torch.tensor([[[0.0, 1.0, 2.5]]], dtype=torch.float64))
    assert torch.equal(batch["p2b"], torch.tensor([[[0.0, 1.0, 3.5]]], dtype=torch.float64))
    assert torch.equal(batch["pins"], torch.tensor([[[7.0, 11.0]]], dtype=torch.float64))
    assert torch.equal(batch["n_soft"], t._ENERGY.n_soft(batch["cons"], batch["area"]))
    for key in ("area", "cons", "b2b", "p2b", "pins", "hpwl_ref", "area_ref",
                "scale", "n_soft", "tau_sharp", "tau_soft"):
        assert batch[key].device.type == "cpu" and batch[key].dtype == torch.float64


def test_task4_p1b_sanitized_empty_edges_reaches_official_and_energy(monkeypatch):
    case = _p1b_case() | {
        "b2b": [], "p2b": [], "pins": [],
        "hpwl_ref": 0.0, "area_ref": 2.0,
    }
    scorer = _P1BScorer((True, 2.0))
    out, trace = _p1b_run(monkeypatch, scorer, case=case, stream=[("base", _p1b_raw())], energy_results=[4.0])
    assert out.winner_ordinal == 0 and out.candidates[0].official_cost == 2.0
    assert out.candidates[0].diagnostic_energy == 4.0
    args, _kwargs = scorer.calls[0]
    assert args[1] == {"hpwl_baseline": 0.0, "area_baseline": 2.0}
    assert args[3].shape == (0, 3) and args[4].shape == (0, 3) and args[5].shape == (0, 2)
    assert all(x.device.type == "cpu" and x.dtype == torch.float64 for x in args[3:7])
    assert len(trace["admit"]) == len(trace["hard"]) == len(trace["intent"]) == len(trace["energy"]) == 1


def _p1b_named_case():
    return _p1b_case() | {
        "cons": [[0, 1, 0, 1, 0], [0, 0, 0, 1, 0]],
        "tp": [[0.0, 0.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
    }


@pytest.mark.parametrize("cfg,stream", [
    (topology_prior.ProposalConfig(total_cap=1), ["base", "axis:0:1:0:0"]),
    (topology_prior.ProposalConfig(total_cap=2), ["base", "axis:0:1:0:0", "axis:0:1:1:0"]),
    (topology_prior.ProposalConfig(axis_exchange_cap=0), ["base", "axis:0:1:0:0"]),
    (topology_prior.ProposalConfig(axis_exchange_cap=1), ["base", "axis:0:1:0:0", "axis:0:1:1:0"]),
    (topology_prior.ProposalConfig(pin_repair_cap=0), ["base", "pin:0:1:0:1"]),
    (topology_prior.ProposalConfig(pin_repair_cap=1), ["base", "pin:0:1:0:1", "pin:0:1:1:0"]),
    (topology_prior.ProposalConfig(group_contact_cap=0), ["base", "contact:1:0:1:0:1"]),
    (topology_prior.ProposalConfig(group_contact_cap=1), ["base", "contact:1:0:1:0:1", "contact:1:0:1:1:0"]),
])
def test_task4_p1b_named_caps_reject_overproposal_before_sinks(monkeypatch, cfg, stream):
    trace = {}; scorer = _P1BScorer()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _p1b_run(monkeypatch, scorer, case=_p1b_named_case(), cfg=cfg,
                 stream=[(name, _p1b_raw()) for name in stream], trace_out=trace)
    assert scorer.calls == []
    assert trace["admit"] == trace["hard"] == trace["intent"] == trace["energy"] == []


@pytest.mark.parametrize("name", [
    "axis:1:0:0:0", "axis:0:0:0:0", "axis:0:9:0:0", "axis:0:1:2:0", "axis:0:1:0:2",
    "pin:1:0:0:1", "pin:0:0:0:1", "pin:0:9:0:1", "pin:0:1:2:0", "pin:0:1:0:2",
    "contact:0:0:1:0:1", "contact:2:0:1:0:1", "contact:1:1:0:0:1", "contact:1:0:9:0:1",
    "contact:1:0:1:2:1", "contact:1:0:1:0:2", "contact:1:0:1:0:1:extra",
])
def test_task4_p1b_semantic_names_fail_closed_before_sinks(monkeypatch, name):
    trace = {}; scorer = _P1BScorer()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _p1b_run(monkeypatch, scorer, case=_p1b_named_case(),
                 stream=[("base", _p1b_raw()), (name, _p1b_raw())], trace_out=trace)
    assert scorer.calls == []
    assert trace["admit"] == trace["hard"] == trace["intent"] == trace["energy"] == []


class _P1BOverCapStream:
    def __init__(self):
        self.requests = 0
    def __len__(self):
        return 3
    def __iter__(self):
        self.requests += 1; yield ("base", _p1b_raw())
        self.requests += 1; yield ("axis:0:1:0:0", _p1b_raw())
        self.requests += 1; raise AssertionError("over-consumed")


def test_task4_p1b_total_cap_bounded_consumption(monkeypatch):
    stream = _P1BOverCapStream(); trace = {}; scorer = _P1BScorer()
    with pytest.raises((TypeError, ValueError, RuntimeError, AssertionError)):
        _p1b_run(monkeypatch, scorer, cfg=topology_prior.ProposalConfig(total_cap=1),
                 stream=stream, trace_out=trace)
    assert stream.requests == 2
    assert scorer.calls == [] and trace["admit"] == trace["hard"] == trace["intent"] == trace["energy"] == []


@pytest.mark.parametrize("stream,cfg", [
    (["base", "axis:0:1:0:0"], topology_prior.ProposalConfig(total_cap=1)),
    (["base", "axis:0:1:0:0"], topology_prior.ProposalConfig(axis_exchange_cap=0)),
    (["base", "pin:0:1:0:1"], topology_prior.ProposalConfig(pin_repair_cap=0)),
    (["base", "contact:1:0:1:0:1"], topology_prior.ProposalConfig(group_contact_cap=0)),
])
def test_task4_p1b_caps_and_names_fail_closed_before_candidate_sinks(monkeypatch, stream, cfg):
    t = _teacher(); scorer = _P1BScorer(); trace = {}
    templates = [(name, _p1b_raw()) for name in stream]
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _p1b_run(monkeypatch, scorer, stream=templates, cfg=cfg, trace_out=trace)
    assert scorer.calls == []
    assert trace["admit"] == trace["hard"] == trace["intent"] == trace["energy"] == []


def _task4_p1c_case_input(t, index=0, *, partition="heldout", seed=7123):
    case = dict(_task4_expected_case(index))
    receipt = CorpusSourceReceipt(
        relative_path="worker_2/layouts_0.th",
        file_sha256=hashlib.sha256(b"p1c-source").hexdigest(),
        layout_index=index,
        fingerprint=fingerprint_case(case),
    )
    return t._CaseInput(case, receipt, partition, seed)


def _task4_p1c_raw(case):
    # This is a complete sanitized corpus case: block 1 preserves its fixed
    # dimensions and block 2 its preplaced origin.  It is deliberately not the
    # minimal P1-B adapter fixture, which is not fingerprintable corpus data.
    assert case["n"] == 3 and set(case) >= {
        "instance_id", "area", "cons", "tp", "b2b", "p2b", "pins",
        "hpwl_ref", "area_ref",
    }
    return torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 3.0],
         [10.0, 11.0, 3.0, 3.0]], dtype=torch.float64,
    )


def _task4_p1c_mutation(raw):
    changed = raw.clone()
    changed[0, 0] = 5.0
    return changed


def _task4_p1c_sha_topology(rects, cons):
    """Independent canonical relation oracle; it never calls production fp code."""
    relations = []
    for first in range(rects.shape[0]):
        for second in range(first + 1, rects.shape[0]):
            gaps = []
            for axis in (0, 1):
                a, b = rects[first], rects[second]
                gaps.append(max(float(a[axis] - b[axis] - b[axis + 2]),
                                float(b[axis] - a[axis] - a[axis + 2])))
            axis = 0 if gaps[0] >= gaps[1] else 1
            first_center = float(rects[first, axis] + rects[first, axis + 2] / 2)
            second_center = float(rects[second, axis] + rects[second, axis + 2] / 2)
            relations.append(("pair", first, second, axis,
                              int((first_center, first) <= (second_center, second))))
    for first in range(rects.shape[0]):
        for second in range(first + 1, rects.shape[0]):
            gid = cons[first][3]
            if gid <= 0 or gid != cons[second][3]:
                continue
            for axis in (0, 1):
                perpendicular = 1 - axis
                overlap = min(float(rects[first, perpendicular] + rects[first, perpendicular + 2]),
                              float(rects[second, perpendicular] + rects[second, perpendicular + 2])) - max(
                                  float(rects[first, perpendicular]), float(rects[second, perpendicular]))
                for order in (0, 1):
                    face = (rects[first, axis] + rects[first, axis + 2] == rects[second, axis]
                            if order else rects[second, axis] + rects[second, axis + 2] == rects[first, axis])
                    if bool(face) and overlap > 0:
                        relations.append(("contact", gid, first, second, axis, order))
    canonical = json.dumps(relations, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _task4_p1c_record(t, ordinal, name, original, legal, *, cost=None,
                      energy=None, energy_status="not_reached", reason=None,
                      drift=None, hard=None):
    return t._CandidateRecord(
        ordinal, name, original, legal,
        {} if drift is None else drift,
        {} if hard is None else hard,
        cost, energy, energy_status, reason,
    )


def _task4_p1c_label(case_input, legal, ordinal, name, base_cost, teacher_cost):
    label = topology_prior.extract_sparse_label(
        legal, case_input.case, case_input.case["instance_id"],
        case_input.sample_seed, teacher_cost, base_cost,
    )
    return {
        "edges": [dataclasses.asdict(edge) for edge in label.edges],
        "contacts": [dataclasses.asdict(contact) for contact in label.contacts],
        "pin_paths": [list(path) for path in label.pin_paths],
        "proposal_ordinal": ordinal, "proposal_name": name,
        "base_cost": base_cost, "teacher_cost": teacher_cost,
        "record_weight": base_cost / teacher_cost,
    }


def _task4_p1c_expected_row(case_input, raw, record, lifecycle):
    name = record.name
    reason = record.rejection_reason
    row = {
        "ordinal": record.ordinal, "name": name, "intended_intent": name,
        "raw_topology_fingerprint": _task4_p1c_sha_topology(raw, case_input.case["cons"]),
        "intended_topology_fingerprint": _task4_p1c_sha_topology(record.original, case_input.case["cons"]),
        "realized_topology_fingerprint": (
            _task4_p1c_sha_topology(record.legal, case_input.case["cons"])
            if record.legal is not None else None),
        "admission_status": "admitted" if record.legal is not None else "failed",
        "admission_reason": None if record.legal is not None else "admission_failed",
        "drift": None, "hard": None, "hard_status": "not_reached",
        "intent_status": "not_reached", "official_cost": None,
        "feasible": None, "official_status": "not_reached",
        "diagnostic_energy": None, "energy_status": "not_reached",
        "winner": False, "status": "rejected",
    }
    if record.legal is None:
        return row
    row.update(drift=dict(record.drift), hard=dict(record.hard))
    if reason == "hard_audit_failed":
        row["hard_status"] = "failed"
        return row
    row["hard_status"] = "passed"
    if reason == "intent_not_survived":
        row["intent_status"] = "failed"
        return row
    row["intent_status"] = "passed"
    if reason in {"official_infeasible", "official_invalid_cost", "official_evaluator_error"}:
        row.update(
            official_status={
                "official_infeasible": "infeasible",
                "official_invalid_cost": "invalid_cost",
                "official_evaluator_error": "error",
            }[reason],
            feasible=False if reason == "official_infeasible" else None,
        )
        return row
    assert record.official_cost is not None
    row.update(
        official_cost=record.official_cost, feasible=True, official_status="scored",
        diagnostic_energy=record.diagnostic_energy,
        energy_status=record.energy_status,
    )
    if lifecycle.winner_ordinal is None:
        row["status"] = "base_unavailable"
    elif record.ordinal == lifecycle.winner_ordinal:
        row.update(winner=True, status=("winner_base_no_improvement" if name == "base"
                                        else "winner_mutation"))
    else:
        row["status"] = "not_selected"
    return row


def _task4_p1c_expected_rejections(lifecycle):
    mapping = {
        "admission_failed": "admission", "hard_audit_failed": "hard",
        "intent_not_survived": "intent", "official_infeasible": "official",
        "official_invalid_cost": "official", "official_evaluator_error": "official",
        "base_unavailable": "selection",
    }
    return tuple(
        {"ordinal": r.ordinal, "name": r.name, "stage": mapping[r.rejection_reason],
         "reason": r.rejection_reason}
        for r in lifecycle.candidates
        if r.rejection_reason in mapping
    )


def _task4_p1c_outcome(case_input, raw, lifecycle):
    rows = tuple(_task4_p1c_expected_row(case_input, raw, r, lifecycle)
                 for r in lifecycle.candidates)
    if lifecycle.winner_ordinal is None:
        return _Task4CaseOutcome(None, rows, _task4_p1c_expected_rejections(lifecycle),
                                 None, None, False, False, "base_unavailable")
    winner = next(r for r in lifecycle.candidates if r.ordinal == lifecycle.winner_ordinal)
    return _Task4CaseOutcome(
        _task4_p1c_label(case_input, winner.legal, winner.ordinal, winner.name,
                         lifecycle.base_cost, lifecycle.teacher_cost),
        rows, _task4_p1c_expected_rejections(lifecycle), lifecycle.base_cost,
        lifecycle.teacher_cost, True, True,
        "winner_base_no_improvement" if winner.name == "base" else "winner_mutation",
    )


def _task4_p1c_assert_valid_current_contract_outcome(case_input, value):
    """Independent P1-C acceptance oracle for fixtures before they are poisoned.

    This deliberately does not invoke the production validator: pre-GREEN it
    still implements the incompatible P1-A/B label schema.  Its purpose is to
    prove that a later poison is the one semantic defect under test, rather
    than a malformed runtime fixture.
    """
    assert type(value) is _Task4CaseOutcome
    raw = _task4_p1c_raw(case_input.case); mutation = _task4_p1c_mutation(raw)
    rows = tuple(value.proposal_rows)
    assert rows and all(set(row) == _TASK4_P1C_PROPOSAL_KEYS for row in rows)
    assert [(row["ordinal"], row["name"]) for row in rows] == [
        (index, "base" if index == 0 else "axis:0:1:0:0")
        for index in range(len(rows))
    ]
    assert all(type(row["intended_intent"]) is str
               and row["intended_intent"] == row["name"] for row in rows)
    expected_raw_sha = _task4_p1c_sha_topology(raw, case_input.case["cons"])
    expected_mutation_sha = _task4_p1c_sha_topology(mutation, case_input.case["cons"])
    for row in rows:
        expected_layout_sha = expected_raw_sha if row["ordinal"] == 0 else expected_mutation_sha
        assert row["raw_topology_fingerprint"] == expected_raw_sha
        assert row["intended_topology_fingerprint"] == expected_layout_sha
        assert row["realized_topology_fingerprint"] == expected_layout_sha

    if value.case_status == "base_unavailable":
        assert value.legal is False and value.covered is False
        assert value.label_row is value.base_cost is value.teacher_cost is None
        assert not any(row["winner"] for row in rows)
        assert len(rows) == 2
        assert rows[0]["admission_status"] == "admitted"
        assert rows[0]["admission_reason"] is None
        assert rows[0]["hard_status"] == "passed"
        assert rows[0]["intent_status"] == "passed"
        assert rows[0]["status"] == "rejected"
        assert rows[0]["official_status"] == "invalid_cost"
        assert rows[0]["official_cost"] is rows[0]["feasible"] is None
        assert rows[0]["diagnostic_energy"] is None
        assert rows[0]["energy_status"] == "not_reached"
        assert rows[1]["admission_status"] == "admitted"
        assert rows[1]["admission_reason"] is None
        assert rows[1]["hard_status"] == "passed"
        assert rows[1]["intent_status"] == "passed"
        assert rows[1]["status"] == "base_unavailable"
        assert rows[1]["official_status"] == "scored"
        assert rows[1]["feasible"] is True and rows[1]["official_cost"] > 0
        assert rows[1]["diagnostic_energy"] == 3.0
        assert rows[1]["energy_status"] == "recorded"
        assert tuple(value.rejection_rows) == (
            {"ordinal": 0, "name": "base", "stage": "official",
             "reason": "official_invalid_cost"},
            {"ordinal": 1, "name": "axis:0:1:0:0", "stage": "selection",
             "reason": "base_unavailable"},
        )
        return

    assert value.case_status in {"winner_base_no_improvement", "winner_mutation"}
    assert value.legal is True and value.covered is True
    assert type(value.base_cost) in (int, float) and value.base_cost > 0
    assert type(value.teacher_cost) in (int, float) and 0 < value.teacher_cost <= value.base_cost
    winners = [row for row in rows if row["winner"]]
    assert len(winners) == 1
    assert len(rows) == 2
    assert all(row["admission_status"] == "admitted" for row in rows)
    assert all(row["admission_reason"] is None for row in rows)
    assert all(row["hard_status"] == "passed" for row in rows)
    assert all(row["intent_status"] == "passed" for row in rows)
    assert all(row["official_status"] == "scored" and row["feasible"] is True for row in rows)
    assert all(row["energy_status"] == "recorded" for row in rows)
    winner = winners[0]
    assert winner["official_status"] == "scored" and winner["official_cost"] == value.teacher_cost
    base = next(row for row in rows if row["ordinal"] == 0)
    assert base["official_status"] == "scored" and base["official_cost"] == value.base_cost
    assert base["winner"] is False and base["status"] == "not_selected"
    assert winner["status"] == value.case_status
    assert value.rejection_rows == ()
    assert value.label_row == _task4_p1c_label(
        case_input, raw if winner["ordinal"] == 0 else mutation,
        winner["ordinal"], winner["name"], value.base_cost, value.teacher_cost,
    )


def _task4_p1c_outcome_snapshot(value):
    return dataclasses.asdict(value)


def _task4_p1c_semantic_baseline(t, case_input, mode):
    if mode == "covered":
        value = _task4_p1c_mutation_outcome(t, case_input)
    elif mode == "base_unavailable":
        value = _task4_p1c_unavailable_outcome(t, case_input)
    else:
        raise AssertionError(f"unknown semantic baseline mode: {mode}")
    _task4_p1c_assert_valid_current_contract_outcome(case_input, value)
    return value


def _task4_p1c_require_production_baseline(t, case_input, invalid):
    """Gate poison rows on the real validator accepting their pristine basis."""
    mode = ("base_unavailable" if invalid.startswith("unavailable_") else "covered")
    value = _task4_p1c_semantic_baseline(t, case_input, mode)
    legacy_error = "runtime costs" if mode == "base_unavailable" else "label schema"
    try:
        accepted = t._validate_outcome(value)
    except ValueError as exc:
        if str(exc) == legacy_error:
            pytest.skip(f"blocked by baseline migration: legacy {legacy_error}")
        raise
    assert accepted is value
    return value


def _task4_p1c_difference_paths(before, after, prefix=()):
    if type(before) is not type(after):
        return {prefix}
    if isinstance(before, dict):
        if set(before) != set(after):
            return {prefix}
        return set().union(*(
            _task4_p1c_difference_paths(before[key], after[key], prefix + (key,))
            for key in before
        ))
    if isinstance(before, (tuple, list)):
        if len(before) != len(after):
            return {prefix}
        return set().union(*(
            _task4_p1c_difference_paths(left, right, prefix + (index,))
            for index, (left, right) in enumerate(zip(before, after))
        ))
    return set() if before == after else {prefix}


def _task4_p1c_poison_semantic_outcome(t, case_input, invalid, *, valid):
    _task4_p1c_assert_valid_current_contract_outcome(case_input, valid)
    snapshot = copy.deepcopy(_task4_p1c_outcome_snapshot(valid))
    label = snapshot["label_row"]
    proposals = snapshot["proposal_rows"]
    changed_path = None
    if invalid == "case_status_unknown":
        snapshot["case_status"] = "winner_unknown"; changed_path = ("case_status",)
    elif invalid == "case_status_mismatch":
        snapshot["case_status"] = "winner_base_no_improvement"; changed_path = ("case_status",)
    elif invalid == "covered_legal_mismatch":
        snapshot["legal"] = False; changed_path = ("legal",)
    elif invalid == "covered_flag_mismatch":
        snapshot["covered"] = False; changed_path = ("covered",)
    elif invalid == "covered_null_cost":
        snapshot["base_cost"] = None; changed_path = ("base_cost",)
    elif invalid == "covered_null_label":
        snapshot["label_row"] = None; changed_path = ("label_row",)
    elif invalid == "unavailable_flags":
        snapshot["legal"] = True; changed_path = ("legal",)
    elif invalid == "unavailable_label":
        snapshot["label_row"] = _task4_good_label(case_input); changed_path = ("label_row",)
    elif invalid == "unavailable_cost":
        snapshot["base_cost"] = 1.1; changed_path = ("base_cost",)
    elif invalid == "label_sparse_payload":
        assert label["edges"]
        label["edges"][0]["weight"] = 2.0
        changed_path = ("label_row", "edges", 0, "weight")
    elif invalid == "label_base_cost":
        label["base_cost"] = 1.2; changed_path = ("label_row", "base_cost")
    elif invalid == "label_teacher_cost":
        label["teacher_cost"] = 0.9; changed_path = ("label_row", "teacher_cost")
    elif invalid == "label_record_weight":
        label["record_weight"] = 1.0; changed_path = ("label_row", "record_weight")
    else:
        raise AssertionError(invalid)
    poisoned = _Task4CaseOutcome(
        snapshot["label_row"], tuple(snapshot["proposal_rows"]),
        tuple(snapshot["rejection_rows"]), snapshot["base_cost"],
        snapshot["teacher_cost"], snapshot["legal"], snapshot["covered"],
        snapshot["case_status"],
    )
    assert _task4_p1c_difference_paths(
        _task4_p1c_outcome_snapshot(valid), _task4_p1c_outcome_snapshot(poisoned),
    ) == {changed_path}
    return poisoned


def _task4_p1c_assert_poisoned_semantic_invariant(value, invalid):
    if invalid == "case_status_unknown":
        assert value.case_status == "winner_unknown"
    elif invalid == "case_status_mismatch":
        assert value.case_status == "winner_base_no_improvement"
        assert value.proposal_rows[1]["winner"] is True
    elif invalid == "covered_legal_mismatch":
        assert value.covered is True and value.legal is False
    elif invalid == "covered_flag_mismatch":
        assert value.legal is True and value.covered is False
    elif invalid == "covered_null_cost":
        assert value.covered is value.legal is True
        assert value.base_cost is None and value.teacher_cost > 0 and value.label_row is not None
    elif invalid == "covered_null_label":
        assert value.covered is value.legal is True
        assert value.label_row is None and value.base_cost > 0 and value.teacher_cost > 0
    elif invalid == "unavailable_flags":
        assert value.case_status == "base_unavailable" and value.legal is True and value.covered is False
    elif invalid == "unavailable_label":
        assert value.case_status == "base_unavailable" and value.label_row is not None
        assert value.base_cost is value.teacher_cost is None
    elif invalid == "unavailable_cost":
        assert value.case_status == "base_unavailable" and value.label_row is None
        assert value.base_cost == 1.1 and value.teacher_cost is None
    elif invalid == "label_sparse_payload":
        assert value.label_row["edges"][0]["weight"] == 2.0
    elif invalid == "label_base_cost":
        assert value.label_row["base_cost"] != value.base_cost
    elif invalid == "label_teacher_cost":
        assert value.label_row["teacher_cost"] != value.teacher_cost
    elif invalid == "label_record_weight":
        assert value.label_row["record_weight"] != value.base_cost / value.teacher_cost
    else:
        raise AssertionError(invalid)


def _task4_p1c_mutation_outcome(t, case_input, *, base_cost=1.1, teacher_cost=1.0):
    raw = _task4_p1c_raw(case_input.case); mutation = _task4_p1c_mutation(raw)
    base = _task4_p1c_record(
        t, 0, "base", raw, raw, cost=base_cost, energy=-1000.0,
        energy_status="recorded", hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    winner = _task4_p1c_record(
        t, 1, "axis:0:1:0:0", mutation, mutation, cost=teacher_cost, energy=1000.0,
        energy_status="recorded", hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    return _task4_p1c_outcome(
        case_input, raw,
        t._CandidateLifecycle((base, winner), 1, base_cost, teacher_cost),
    )


def _task4_p1c_base_no_improvement_outcome(t, case_input, *, cost=1.1):
    raw = _task4_p1c_raw(case_input.case)
    base = _task4_p1c_record(
        t, 0, "base", raw, raw, cost=cost, energy=None,
        energy_status="unavailable", hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    return _task4_p1c_outcome(
        case_input, raw, t._CandidateLifecycle((base,), 0, cost, cost)
    )


def _task4_p1c_unavailable_outcome(t, case_input, *, base_reason="official_invalid_cost"):
    raw = _task4_p1c_raw(case_input.case); mutation = _task4_p1c_mutation(raw)
    base_legal = None if base_reason == "admission_failed" else raw
    base = _task4_p1c_record(
        t, 0, "base", raw, base_legal, reason=base_reason,
        hard={} if base_legal is None else ({"hard_audit": False}
                                             if base_reason == "hard_audit_failed"
                                             else {"hard_audit": True}),
        drift={} if base_legal is None else {"max_abs": 0.0},
    )
    scored = _task4_p1c_record(
        t, 1, "axis:0:1:0:0", mutation, mutation, cost=1.0, energy=3.0,
        energy_status="recorded", reason="base_unavailable",
        hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    return _task4_p1c_outcome(
        case_input, raw, t._CandidateLifecycle((base, scored), None, None, None)
    )


def _task4_p1c_rejection_outcome(t, case_input, reason):
    """A valid lifecycle fact for one terminal rejection reason."""
    if reason == "base_unavailable":
        return _task4_p1c_unavailable_outcome(t, case_input)
    raw = _task4_p1c_raw(case_input.case); mutation = _task4_p1c_mutation(raw)
    base = _task4_p1c_record(
        t, 0, "base", raw, raw, cost=1.1, energy=None,
        energy_status="unavailable", hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    legal = None if reason == "admission_failed" else mutation
    hard = ({} if legal is None else
            {"hard_audit": False} if reason == "hard_audit_failed" else
            {"hard_audit": True})
    candidate = _task4_p1c_record(
        t, 1, "axis:0:1:0:0", mutation, legal,
        cost=None,
        energy_status="not_reached", reason=reason, hard=hard,
        drift={} if legal is None else {"max_abs": 0.0},
    )
    return _task4_p1c_outcome(
        case_input, raw, t._CandidateLifecycle((base, candidate), 0, 1.1, 1.1)
    )


def _task4_p1c_canonical_population(rows):
    ordered = sorted(rows, key=lambda row: (row["relative_path"], row["layout_index"]))
    encoded = b"[" + b",".join(
        json.dumps({
            "relative_path": row["relative_path"], "layout_index": row["layout_index"],
            "instance_id": row["instance_id"], "n": row["n"],
            "weight": math.exp(row["n"] / 12),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
           allow_nan=False).encode("utf-8") for row in ordered
    ) + b"]"
    denominator = sum(math.exp(row["n"] / 12) for row in ordered)
    scored = [row for row in ordered if "base_cost" in row]
    scored_denominator = sum(math.exp(row["n"] / 12) for row in scored)
    if len(scored) != len(ordered) or not ordered:
        metrics = {"B_H": None, "T_H": None, "Delta_H": None}
    else:
        b_h = sum(math.exp(row["n"] / 12) * row["base_cost"] for row in ordered) / denominator
        t_h = sum(math.exp(row["n"] / 12) * row["teacher_cost"] for row in ordered) / denominator
        metrics = {"B_H": b_h, "T_H": t_h, "Delta_H": b_h - t_h}
    return {
        "eligible_count": len(ordered), "scored_winner_count": len(scored),
        "denominator": denominator, "scored_denominator": scored_denominator,
        **metrics, "population_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _task4_p1c_verified_runtime(t, tmp_path, monkeypatch):
    root = tmp_path / "verified"; root.mkdir()
    checkpoint, policy = _task4_verified_checkpoint(t, root)
    runtime = t._runtime_hooks()
    assert runtime.preflight(policy, checkpoint) == _task4_expected_trust(policy)
    return runtime


def test_task4_p1c_runtime_runs_one_authorizing_sample_lifecycle_compiler_chain(tmp_path, monkeypatch):
    t = _teacher(); runtime = _task4_p1c_verified_runtime(t, tmp_path, monkeypatch)
    case_input = _task4_p1c_case_input(t)
    raw = _task4_p1c_raw(case_input.case); lifecycle = object(); outcome = object(); events = []
    model = object()
    monkeypatch.setattr(t, "_select_teacher_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(t, "_materialize_teacher_model",
                        lambda payload, device: events.append(("materialize", payload, device)) or model)
    monkeypatch.setattr(t, "_sample_direct_once", lambda got, case, seed:
                        events.append(("sample", got, case, seed)) or raw)
    monkeypatch.setattr(t, "_run_candidate_lifecycle", lambda got, case, *, scorer, cfg:
                        events.append(("lifecycle", got, case, scorer, cfg)) or lifecycle)
    # This deliberate wished-for seam makes the first pre-GREEN failure the
    # missing bridge behavior, never fixture construction.
    monkeypatch.setattr(t, "_outcome_from_lifecycle", lambda got_ci, got_raw, got_lifecycle:
                        events.append(("compiler", got_ci, got_raw, got_lifecycle)) or outcome,
                        raising=False)
    assert runtime.authorizing is True
    assert runtime.process_case(case_input) is outcome
    assert [event[0] for event in events] == ["materialize", "sample", "lifecycle", "compiler"]
    assert events[1][1:] == (model, case_input.case, case_input.sample_seed)
    assert events[2][1] is raw and events[2][2] is case_input.case
    assert events[2][3] is t._EVALUATOR and events[2][4] == t.ProposalConfig()
    assert events[3][1] is case_input and events[3][2] is raw and events[3][3] is lifecycle


@pytest.mark.parametrize("seam", ["sample", "lifecycle", "compiler"])
def test_task4_p1c_runtime_bridge_errors_are_transactional(tmp_path, monkeypatch, seam):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir(); _task4_shard(root)
    _checkpoint, policy = _task4_verified_checkpoint(t, root); out = tmp_path / "out"
    sentinel = RuntimeError(f"p1c-{seam}-fatal")
    monkeypatch.setattr(t, "_materialize_teacher_model", lambda *args: object())
    if seam == "sample":
        monkeypatch.setattr(t, "_sample_direct_once", lambda *args: (_ for _ in ()).throw(sentinel))
    else:
        monkeypatch.setattr(t, "_sample_direct_once", lambda *_: _task4_p1c_raw(_task4_expected_case(0)))
        if seam == "lifecycle":
            monkeypatch.setattr(t, "_run_candidate_lifecycle", lambda *args, **kwargs: (_ for _ in ()).throw(sentinel))
        else:
            monkeypatch.setattr(t, "_run_candidate_lifecycle", lambda *args, **kwargs: object())
            monkeypatch.setattr(t, "_outcome_from_lifecycle", lambda *args: (_ for _ in ()).throw(sentinel), raising=False)
    with pytest.raises(RuntimeError, match=rf"p1c-{seam}-fatal"):
        t.teacher_main(_task4_args(root, out), _trust_policy=policy)
    assert not out.exists()


def test_task4_p1c_case_outcome_schema_is_explicit_and_process_none_is_invalid(tmp_path, monkeypatch):
    t = _teacher()
    assert [field.name for field in dataclasses.fields(t._CaseOutcome)] == [
        "label_row", "proposal_rows", "rejection_rows", "base_cost", "teacher_cost",
        "legal", "covered", "case_status",
    ]
    assert _Task4CaseOutcome.__dataclass_fields__.keys() == t._CaseOutcome.__dataclass_fields__.keys()
    runtime = _task4_p1c_verified_runtime(t, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="case input"):
        runtime.process_case(None)


def test_task4_p1c_compiler_uses_official_mutation_and_independent_topology_sha_oracles():
    t = _teacher(); case_input = _task4_p1c_case_input(t); raw = _task4_p1c_raw(case_input.case)
    mutation = _task4_p1c_mutation(raw)
    base = _task4_p1c_record(t, 0, "base", raw, raw, cost=10.0, energy=-999.0,
                              energy_status="recorded", hard={"hard_audit": True}, drift={"max_abs": 0.0})
    candidate = _task4_p1c_record(t, 1, "axis:0:1:0:0", mutation, mutation,
                                   cost=8.0, energy=999.0, energy_status="recorded",
                                   hard={"hard_audit": True}, drift={"max_abs": 0.0})
    lifecycle = t._CandidateLifecycle((base, candidate), 1, 10.0, 8.0)
    compiler = t._outcome_from_lifecycle
    signature = inspect.signature(compiler)
    assert [(parameter.name, parameter.kind, parameter.default)
            for parameter in signature.parameters.values()] == [
                ("case_input", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("raw_rects", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("lifecycle", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
            ]
    outcome = compiler(case_input, raw, lifecycle)
    expected = _task4_p1c_outcome(case_input, raw, lifecycle)
    assert type(outcome) is t._CaseOutcome
    assert dataclasses.asdict(outcome) == dataclasses.asdict(expected)
    assert set(outcome.proposal_rows[1]) == _TASK4_P1C_PROPOSAL_KEYS
    assert [row["raw_topology_fingerprint"] for row in outcome.proposal_rows] == [
        _task4_p1c_sha_topology(raw, case_input.case["cons"]),
        _task4_p1c_sha_topology(raw, case_input.case["cons"]),
    ]
    assert outcome.proposal_rows[1]["intended_topology_fingerprint"] == _task4_p1c_sha_topology(
        mutation, case_input.case["cons"]
    )
    assert outcome.proposal_rows[1]["realized_topology_fingerprint"] == _task4_p1c_sha_topology(
        mutation, case_input.case["cons"]
    )
    assert all(type(row["intended_intent"]) is str and row["intended_intent"] == row["name"]
               for row in outcome.proposal_rows)
    assert set(outcome.label_row) == {
        "edges", "contacts", "pin_paths", "proposal_ordinal", "proposal_name",
        "base_cost", "teacher_cost", "record_weight",
    }
    assert outcome.label_row["edges"] == expected.label_row["edges"]
    assert outcome.label_row["contacts"] == expected.label_row["contacts"]
    assert outcome.label_row["pin_paths"] == expected.label_row["pin_paths"]
    assert all(set(edge) == {"src", "dst", "axis", "margin", "kind", "weight"}
               for edge in outcome.label_row["edges"])
    assert all(set(contact) == {"a", "b", "axis", "a_before_b", "perp_margin", "weight"}
               for contact in outcome.label_row["contacts"])
    assert all(all(type(node) is int for node in path)
               for path in outcome.label_row["pin_paths"])
    assert outcome.label_row["proposal_ordinal"] == 1
    assert outcome.label_row["proposal_name"] == "axis:0:1:0:0"
    assert outcome.label_row["base_cost"] == 10.0 and outcome.label_row["teacher_cost"] == 8.0
    assert outcome.label_row["record_weight"] == 1.25


@pytest.mark.parametrize("mutation_cost", [10.0, 12.0])
def test_task4_p1c_compiler_tie_or_worse_keeps_base_and_omits_scored_loser_rejection(mutation_cost):
    t = _teacher(); case_input = _task4_p1c_case_input(t); raw = _task4_p1c_raw(case_input.case)
    base = _task4_p1c_record(t, 0, "base", raw, raw, cost=10.0, energy=20.0,
                              energy_status="recorded", hard={"hard_audit": True}, drift={"max_abs": 0.0})
    loser = _task4_p1c_record(t, 1, "axis:0:1:0:0", _task4_p1c_mutation(raw), _task4_p1c_mutation(raw),
                               cost=mutation_cost, energy=-20.0, energy_status="recorded",
                               hard={"hard_audit": True}, drift={"max_abs": 0.0}, reason="not_selected")
    lifecycle = t._CandidateLifecycle((base, loser), 0, 10.0, 10.0)
    outcome = t._outcome_from_lifecycle(case_input, raw, lifecycle)
    expected = _task4_p1c_outcome(case_input, raw, lifecycle)
    assert type(outcome) is t._CaseOutcome
    assert dataclasses.asdict(outcome) == dataclasses.asdict(expected)
    assert outcome.case_status == "winner_base_no_improvement"
    assert outcome.label_row["record_weight"] == 1.0
    assert outcome.rejection_rows == ()


@pytest.mark.parametrize("reason", ["admission_failed", "hard_audit_failed", "intent_not_survived",
                                     "official_infeasible", "official_invalid_cost", "official_evaluator_error"])
def test_task4_p1c_compiler_stage_matrix_serializes_exact_nulls_and_rejection_bijection(reason):
    t = _teacher(); case_input = _task4_p1c_case_input(t); raw = _task4_p1c_raw(case_input.case)
    base = _task4_p1c_record(t, 0, "base", raw, raw, cost=10.0, energy=None,
                              energy_status="unavailable", hard={"hard_audit": True}, drift={"max_abs": 0.0})
    legal = None if reason == "admission_failed" else _task4_p1c_mutation(raw)
    hard = {} if legal is None else ({"hard_audit": False} if reason == "hard_audit_failed" else {"hard_audit": True})
    candidate = _task4_p1c_record(
        t, 1, "axis:0:1:0:0", _task4_p1c_mutation(raw), legal,
        cost=8.0 if reason not in {"official_infeasible", "official_invalid_cost", "official_evaluator_error"} else None,
        energy=None, energy_status="not_reached", reason=reason,
        hard=hard, drift={} if legal is None else {"max_abs": 0.0},
    )
    lifecycle = t._CandidateLifecycle((base, candidate), 0, 10.0, 10.0)
    outcome = t._outcome_from_lifecycle(case_input, raw, lifecycle)
    expected = _task4_p1c_outcome(case_input, raw, lifecycle)
    assert type(outcome) is t._CaseOutcome
    assert dataclasses.asdict(outcome) == dataclasses.asdict(expected)
    assert outcome.proposal_rows[0]["diagnostic_energy"] is None
    assert outcome.proposal_rows[0]["energy_status"] == "unavailable"
    assert outcome.proposal_rows[1]["intended_intent"] == outcome.proposal_rows[1]["name"]
    assert tuple(outcome.rejection_rows) == _task4_p1c_expected_rejections(lifecycle)


@pytest.mark.parametrize("base_reason", ["admission_failed", "hard_audit_failed", "intent_not_survived",
                                          "official_infeasible", "official_invalid_cost", "official_evaluator_error"])
def test_task4_p1c_compiler_base_unavailable_never_substitutes_scored_mutation(base_reason):
    t = _teacher(); case_input = _task4_p1c_case_input(t); raw = _task4_p1c_raw(case_input.case)
    base_legal = None if base_reason == "admission_failed" else raw
    base = _task4_p1c_record(
        t, 0, "base", raw, base_legal, reason=base_reason,
        hard={} if base_legal is None else ({"hard_audit": False} if base_reason == "hard_audit_failed" else {"hard_audit": True}),
        drift={} if base_legal is None else {"max_abs": 0.0},
    )
    mutation = _task4_p1c_mutation(raw)
    scored_mutation = _task4_p1c_record(
        t, 1, "axis:0:1:0:0", mutation, mutation, cost=8.0, energy=1.0,
        energy_status="recorded", reason="base_unavailable",
        hard={"hard_audit": True}, drift={"max_abs": 0.0},
    )
    lifecycle = t._CandidateLifecycle((base, scored_mutation), None, None, None)
    outcome = t._outcome_from_lifecycle(case_input, raw, lifecycle)
    expected = _task4_p1c_outcome(case_input, raw, lifecycle)
    assert type(outcome) is t._CaseOutcome
    assert dataclasses.asdict(outcome) == dataclasses.asdict(expected)
    assert outcome.case_status == "base_unavailable"
    assert outcome.label_row is None and outcome.base_cost is None and outcome.teacher_cost is None
    assert not any(row["winner"] for row in outcome.proposal_rows)
    assert outcome.proposal_rows[1]["official_cost"] == 8.0
    assert tuple(outcome.rejection_rows) == _task4_p1c_expected_rejections(lifecycle)


def test_task4_p1c_population_two_phase_hashes_full_eligible_population_and_nulls_incomplete():
    t = _teacher()
    rows = [
        {"relative_path": "worker_2/layouts_0.th", "layout_index": 0,
         "instance_id": "worker_2/layouts_0.th#0", "n": 0,
         "base_cost": 4.0, "teacher_cost": 3.0},
        {"relative_path": "worker_10/layouts_0.th", "layout_index": 1,
         "instance_id": "worker_10/layouts_0.th#1", "n": 12},
    ]
    expected = _task4_p1c_canonical_population(rows)
    population = t._PopulationAccumulator()
    for row in rows:
        population.register({key: row[key] for key in ("relative_path", "layout_index", "instance_id", "n")})
    population.record_winner(rows[0]["instance_id"], rows[0]["base_cost"], rows[0]["teacher_cost"])
    assert population.finish() == expected


def test_task4_p1c_population_complete_metrics_and_protocol_errors_are_exact():
    t = _teacher()
    first = {"relative_path": "worker_2/layouts_0.th", "layout_index": 0,
             "instance_id": "a", "n": 0}
    second = {"relative_path": "worker_10/layouts_0.th", "layout_index": 1,
              "instance_id": "b", "n": 12}
    expected = _task4_p1c_canonical_population([
        {**first, "base_cost": 4.0, "teacher_cost": 3.0},
        {**second, "base_cost": 10.0, "teacher_cost": 7.0},
    ])
    population = t._PopulationAccumulator()
    population.register(first); population.register(second)
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        population.register(first)
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        population.record_winner("unknown", 1.0, 1.0)
    population.record_winner("a", 4.0, 3.0)
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        population.record_winner("a", 4.0, 3.0)
    population.record_winner("b", 10.0, 7.0)
    assert population.finish() == expected


def test_task4_p1c_teacher_main_registers_every_heldout_identity_before_process_and_cleans_population(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir()
    _task4_shard(root)
    _task4_shard(root, metric_delta=2, relative_path="worker_2/layouts_2.th")
    _checkpoint, policy = _task4_verified_checkpoint(t, root); out = tmp_path / "out"; events = []

    class PopulationSpy:
        instances = []
        def __init__(self):
            self.rows = {}; self.closed = False
            self.spool_path = tmp_path / f"population-{len(type(self).instances)}.spool"
            self.spool_path.touch(); type(self).instances.append(self)
        def register(self, row):
            events.append(("register", row["instance_id"])); self.rows[row["instance_id"]] = dict(row)
        def record_winner(self, instance_id, base_cost, teacher_cost):
            events.append(("winner", instance_id)); self.rows[instance_id].update(base_cost=base_cost, teacher_cost=teacher_cost)
        def add(self, _row):
            pytest.fail("legacy population.add used instead of global register/record_winner")
        def finish(self):
            events.append(("finish",))
            self.closed = True; self.spool_path.unlink()
            return _task4_p1c_canonical_population(list(self.rows.values()))
        def abort(self):
            self.closed = True
            self.spool_path.unlink(missing_ok=True)
            events.append(("abort",))

    def preflight(_policy, _checkpoint):
        return _task4_expected_trust(policy)
    def process(case_input):
        events.append(("process", case_input.case["instance_id"]))
        return _task4_p1c_mutation_outcome(t, case_input)
    monkeypatch.setattr(t, "_PopulationAccumulator", PopulationSpy)
    monkeypatch.setattr(t, "_CaseOutcome", _Task4CaseOutcome)
    # Isolate the global registration seam from the separately RED P1-C
    # validator migration.  The runtime still returns a rich P1-C outcome.
    monkeypatch.setattr(t, "_validate_outcome", lambda value: value)
    monkeypatch.setattr(t, "_runtime_hooks", lambda: _Task4Runtime(preflight, process, True))
    assert t.teacher_main(_task4_args(root, out), _trust_policy=policy) == 0
    heldout_ids = ["worker_2/layouts_0.th#1", "worker_2/layouts_2.th#1"]
    register_events = [event for event in events if event[0] == "register"]
    process_events = [event for event in events if event[0] == "process"]
    winner_events = [event for event in events if event[0] == "winner"]
    assert register_events == [("register", instance_id) for instance_id in heldout_ids]
    assert process_events and events[:len(register_events)] == register_events
    first_process = next(index for index, event in enumerate(events) if event[0] == "process")
    assert all(events.index(event) > first_process for event in winner_events)
    assert winner_events == [("winner", instance_id) for instance_id in heldout_ids]
    for instance_id in heldout_ids:
        assert events.index(("register", instance_id)) < events.index(("process", instance_id))
        assert events.index(("process", instance_id)) < events.index(("winner", instance_id))
    assert PopulationSpy.instances and PopulationSpy.instances[0].closed
    assert not PopulationSpy.instances[0].spool_path.exists()


def _task4_p1c_artifact_rows(out):
    return {
        name: _task4_jsonl(out / name)
        for name in _TASK4_JSONL
    } | {
        "index": json.loads((out / "training_index.json").read_text()),
        "manifest": json.loads((out / "g0_manifest.json").read_text()),
    }


@pytest.mark.parametrize("mode", ["mutation", "base_no_improvement"])
def test_task4_p1c_teacher_main_artifacts_preserve_dynamic_winner_label_and_receipt_joins(
        tmp_path, monkeypatch, mode):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    outcome_factory = (_task4_p1c_mutation_outcome if mode == "mutation"
                       else _task4_p1c_base_no_improvement_outcome)
    expected_status = "winner_mutation" if mode == "mutation" else "winner_base_no_improvement"
    expected_name = "axis:0:1:0:0" if mode == "mutation" else "base"
    expected_ordinal = 1 if mode == "mutation" else 0
    expected_teacher_cost = 1.0 if mode == "mutation" else 1.1
    expected_proposal_count = 2 if mode == "mutation" else 1
    expected_positive_gain = 1 if mode == "mutation" else 0
    calls, _ = _task4_install_custom_runtime(
        t, monkeypatch, outcome_factory=lambda module, case_input:
        outcome_factory(module, case_input),
    )
    result = t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert result == (0 if mode == "mutation" else 1)
    rows = _task4_p1c_artifact_rows(out)
    corpus = rows["train_corpus.jsonl"] + rows["heldout_corpus.jsonl"]
    labels = rows["train_labels.jsonl"] + rows["heldout_labels.jsonl"]
    assert len(calls) == len(corpus) == len(labels) == 2
    for case_input in calls:
        iid = case_input.case["instance_id"]
        index_row = next(row for row in rows["index"]["rows"] if row["instance_id"] == iid)
        label = next(row for row in labels if row["instance_id"] == iid)
        envelope = _task4_envelope(
            case_input.case, dataclasses.asdict(case_input.receipt),
            case_input.partition, case_input.sample_seed,
        )
        expected = outcome_factory(t, case_input)
        assert next(row for row in corpus if row["instance_id"] == iid) == case_input.case
        assert fingerprint_case(next(row for row in corpus if row["instance_id"] == iid)) == case_input.receipt.fingerprint
        assert index_row == {
            "receipt": dataclasses.asdict(case_input.receipt), "instance_id": iid,
            "source_row_count": 2, "block_count": case_input.case["n"],
            "partition": case_input.partition, "sample_ordinal": 0,
            "sample_seed": case_input.sample_seed, "status": expected_status,
        }
        proposals = [row for row in rows["proposals.jsonl"] if row["instance_id"] == iid]
        rejections = [row for row in rows["rejections.jsonl"] if row["instance_id"] == iid]
        assert len(proposals) == expected_proposal_count
        assert rejections == []
        assert {key: value for key, value in label.items() if key not in envelope} == expected.label_row
        assert [{key: value for key, value in row.items() if key not in envelope}
                for row in proposals] == list(expected.proposal_rows)
        assert [{key: value for key, value in row.items() if key not in envelope}
                for row in rejections] == list(expected.rejection_rows)
        for row in (label, *proposals):
            assert {key: row[key] for key in envelope} == envelope
        assert all(set(row) == _TASK4_P1C_PROPOSAL_KEYS | set(envelope) for row in proposals)
        assert all(type(row["intended_intent"]) is str and row["intended_intent"] == row["name"]
                   for row in proposals)
        winner = next(row for row in proposals if row["winner"])
        assert label["proposal_ordinal"] == expected_ordinal and label["proposal_name"] == expected_name
        assert label["base_cost"] == 1.1 and label["teacher_cost"] == expected_teacher_cost
        assert label["record_weight"] == 1.1 / expected_teacher_cost
        assert winner["ordinal"] == expected_ordinal and winner["name"] == expected_name
        assert winner["status"] == expected_status
    assert rows["manifest"]["population"] == _task4_p1c_canonical_population([{
        "relative_path": "worker_2/layouts_0.th", "layout_index": 1,
        "instance_id": "worker_2/layouts_0.th#1", "n": 3,
        "base_cost": 1.1, "teacher_cost": expected_teacher_cost,
    }])
    assert rows["manifest"]["coverage"] == {
        "eligible_train": 1, "eligible_heldout": 1, "heldout_winners": 1,
        "legal": True, "covered": True,
        "mutation_proposals": 2 if mode == "mutation" else 0,
        "mutation_admitted": 2 if mode == "mutation" else 0,
        "mutation_intent_survived": 2 if mode == "mutation" else 0,
        "positive_gain_heldout": expected_positive_gain,
        "eligible_heldout_weight": math.exp(3 / 12),
        "positive_gain_heldout_weight": expected_positive_gain * math.exp(3 / 12),
        "rejection_counts": {
            "admission_failed": 0, "hard_audit_failed": 0,
            "intent_not_survived": 0, "official_infeasible": 0,
            "official_invalid_cost": 0, "official_evaluator_error": 0,
            "base_unavailable": 0,
        },
    }
    assert rows["manifest"]["status"] == "complete"
    assert rows["manifest"]["state"] == (
        "TARGET_GAIN_MET" if mode == "mutation" else "STOP_HARD_GAIN_MISSED"
    )
    assert rows["manifest"]["training_authorized"] is (mode == "mutation")


def test_task4_p1c_teacher_main_base_unavailable_retains_corpus_and_rejections_but_no_label(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    calls, _ = _task4_install_custom_runtime(
        t, monkeypatch, outcome_factory=lambda module, case_input:
        _task4_p1c_unavailable_outcome(module, case_input),
    )
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 1
    rows = _task4_p1c_artifact_rows(out)
    corpus = rows["train_corpus.jsonl"] + rows["heldout_corpus.jsonl"]
    assert len(corpus) == 2 and rows["train_labels.jsonl"] == rows["heldout_labels.jsonl"] == []
    assert {row["status"] for row in rows["index"]["rows"]} == {"base_unavailable"}
    assert len(rows["proposals.jsonl"]) == len(rows["rejections.jsonl"]) == 4
    for case_input in calls:
        iid = case_input.case["instance_id"]
        envelope = _task4_envelope(case_input.case, dataclasses.asdict(case_input.receipt),
                                   case_input.partition, case_input.sample_seed)
        assert fingerprint_case(next(row for row in corpus if row["instance_id"] == iid)) == case_input.receipt.fingerprint
        index_row = next(row for row in rows["index"]["rows"] if row["instance_id"] == iid)
        assert index_row == {
            "receipt": dataclasses.asdict(case_input.receipt), "instance_id": iid,
            "source_row_count": 2, "block_count": case_input.case["n"],
            "partition": case_input.partition, "sample_ordinal": 0,
            "sample_seed": case_input.sample_seed, "status": "base_unavailable",
        }
        expected = _task4_p1c_unavailable_outcome(t, case_input)
        proposals = [row for row in rows["proposals.jsonl"] if row["instance_id"] == iid]
        rejections = [row for row in rows["rejections.jsonl"] if row["instance_id"] == iid]
        assert [{key: value for key, value in row.items() if key not in envelope}
                for row in proposals] == list(expected.proposal_rows)
        assert [{key: value for key, value in row.items() if key not in envelope}
                for row in rejections] == list(expected.rejection_rows)
        assert all({key: row[key] for key in envelope} == envelope for row in (*proposals, *rejections))
        assert all(set(row) == _TASK4_P1C_PROPOSAL_KEYS | set(envelope) for row in proposals)
        assert all(set(row) == {"ordinal", "name", "stage", "reason"} | set(envelope)
                   for row in rejections)
    manifest = rows["manifest"]
    assert manifest["status"] == "complete" and manifest["state"] == "KILLED_LEGALITY_OR_COVERAGE"
    assert manifest["training_authorized"] is False
    assert manifest["secondary_reasons"].count("base_unavailable") == 1
    assert manifest["population"] == _task4_p1c_canonical_population([{
        "relative_path": "worker_2/layouts_0.th", "layout_index": 1,
        "instance_id": "worker_2/layouts_0.th#1", "n": 3,
    }])
    assert manifest["coverage"] == {
        "eligible_train": 1, "eligible_heldout": 1, "heldout_winners": 0,
        "legal": False, "covered": False,
        "mutation_proposals": 2, "mutation_admitted": 2,
        "mutation_intent_survived": 2, "positive_gain_heldout": 0,
        "eligible_heldout_weight": math.exp(3 / 12),
        "positive_gain_heldout_weight": 0.0,
        "rejection_counts": {
            "admission_failed": 0, "hard_audit_failed": 0,
            "intent_not_survived": 0, "official_infeasible": 0,
            "official_invalid_cost": 2, "official_evaluator_error": 0,
            "base_unavailable": 2,
        },
    }


@pytest.mark.parametrize("reason", [
    "admission_failed", "hard_audit_failed", "intent_not_survived",
    "official_infeasible", "official_invalid_cost", "official_evaluator_error",
    "base_unavailable",
])
def test_task4_p1c_teacher_main_rejection_reason_coverage_matrix_is_exact(
        tmp_path, monkeypatch, reason):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    factory = lambda module, case_input: _task4_p1c_rejection_outcome(
        module, case_input, reason,
    )
    calls, _ = _task4_install_custom_runtime(t, monkeypatch, outcome_factory=factory)
    assert t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root)) == 1
    rows = _task4_p1c_artifact_rows(out)
    outcomes = [factory(t, case_input) for case_input in calls]
    for case_input, outcome in zip(calls, outcomes):
        envelope = _task4_envelope(
            case_input.case, dataclasses.asdict(case_input.receipt),
            case_input.partition, case_input.sample_seed,
        )
        proposals = [row for row in rows["proposals.jsonl"]
                     if row["instance_id"] == case_input.case["instance_id"]]
        rejections = [row for row in rows["rejections.jsonl"]
                      if row["instance_id"] == case_input.case["instance_id"]]
        assert proposals == [
            {**envelope, **dict(row)} for row in outcome.proposal_rows
        ]
        assert rejections == [
            {**envelope, **dict(row)} for row in outcome.rejection_rows
        ]
        assert all(set(row) == _TASK4_P1C_PROPOSAL_KEYS | set(envelope)
                   for row in proposals)
        assert all(set(row) == {"ordinal", "name", "stage", "reason"} | set(envelope)
                   for row in rejections)
        assert all({key: row[key] for key in envelope} == envelope
                   for row in (*proposals, *rejections))
    assert rows["manifest"]["coverage"] == _task4_p1c_coverage(calls, outcomes)
    counts = rows["manifest"]["coverage"]["rejection_counts"]
    expected_counts = {
        "admission_failed": 0, "hard_audit_failed": 0,
        "intent_not_survived": 0, "official_infeasible": 0,
        "official_invalid_cost": 0, "official_evaluator_error": 0,
        "base_unavailable": 0,
    }
    expected_counts[reason] = 2
    if reason == "base_unavailable":
        # The missing base itself retains its truthful official terminal fact.
        expected_counts["official_invalid_cost"] = 2
    assert counts == expected_counts
    assert all(type(value) is int for value in counts.values())
    coverage = rows["manifest"]["coverage"]
    assert coverage["mutation_proposals"] == 2
    assert coverage["mutation_admitted"] == (0 if reason == "admission_failed" else 2)
    assert coverage["mutation_intent_survived"] == (
        2 if reason in {"official_infeasible", "official_invalid_cost",
                         "official_evaluator_error", "base_unavailable"} else 0
    )
    assert coverage["positive_gain_heldout"] == 0
    assert coverage["positive_gain_heldout_weight"] == 0.0
    assert coverage["eligible_heldout_weight"] == math.exp(3 / 12)
    if reason == "base_unavailable":
        assert rows["manifest"]["state"] == "KILLED_LEGALITY_OR_COVERAGE"
        assert rows["manifest"]["secondary_reasons"].count("base_unavailable") == 1
        assert {row["status"] for row in rows["index"]["rows"]} == {"base_unavailable"}
    else:
        assert rows["manifest"]["state"] == "STOP_HARD_GAIN_MISSED"
        assert rows["manifest"]["secondary_reasons"] == []
        assert {row["status"] for row in rows["index"]["rows"]} == {
            "winner_base_no_improvement"
        }
    assert rows["manifest"]["training_authorized"] is False


@pytest.mark.parametrize("invalid", ["tampered_intent", "nonstring_intent", "two_winners",
                                     "baseless_winner", "stage_null", "duplicate_rejection",
                                     "label_mismatch"])
def test_task4_p1c_invalid_runtime_outcome_aborts_transaction_before_destination(tmp_path, monkeypatch, invalid):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root); out = tmp_path / "out"
    def outcome(module, case_input):
        value = _task4_p1c_mutation_outcome(module, case_input)
        proposals = [dict(row) for row in value.proposal_rows]
        rejections = list(value.rejection_rows); label = dict(value.label_row)
        if invalid == "tampered_intent": proposals[1]["intended_intent"] = "base"
        elif invalid == "nonstring_intent": proposals[1]["intended_intent"] = 1
        elif invalid == "two_winners": proposals[0]["winner"] = True; proposals[0]["status"] = "winner_base_no_improvement"
        elif invalid == "baseless_winner": proposals[0]["official_cost"] = None; proposals[0]["winner"] = False; proposals[0]["status"] = "not_selected"
        elif invalid == "stage_null": proposals[1]["admission_status"] = "failed"
        elif invalid == "duplicate_rejection":
            rejections = [{"ordinal": 1, "name": "axis:0:1:0:0", "stage": "selection", "reason": "base_unavailable"}] * 2
        elif invalid == "label_mismatch": label["proposal_ordinal"] = 0; label["proposal_name"] = "base"
        poisoned = _Task4CaseOutcome(label, tuple(proposals), tuple(rejections), value.base_cost,
                                     value.teacher_cost, value.legal, value.covered, value.case_status)
        if invalid == "stage_null":
            assert _task4_p1c_difference_paths(
                dataclasses.asdict(value), dataclasses.asdict(poisoned),
            ) == {("proposal_rows", 1, "admission_status")}
        return poisoned
    _task4_install_custom_runtime(t, monkeypatch, outcome_factory=outcome)
    expected_error = {
        "tampered_intent": "intended_intent", "nonstring_intent": "intended_intent",
        "two_winners": "winner",
        "baseless_winner": "base", "stage_null": "stage",
        "duplicate_rejection": "rejection", "label_mismatch": "label mismatch",
    }[invalid]
    with pytest.raises(ValueError, match=expected_error):
        t.teacher_main(_task4_args(root, out), _trust_policy=_policy_for(root))
    assert not out.exists()


@pytest.mark.parametrize("invalid,expected_error", [
    ("case_status_unknown", r"case[ _]status"),
    ("case_status_mismatch", r"case[ _]status"),
    ("covered_legal_mismatch", r"runtime[ _]flags"),
    ("covered_flag_mismatch", r"runtime[ _]flags"),
    ("covered_null_cost", r"runtime[ _]costs"),
    ("covered_null_label", "label"),
    ("unavailable_flags", r"runtime[ _]flags"),
    ("unavailable_label", r"base[ _]unavailable"),
    ("unavailable_cost", r"base[ _]unavailable"),
    ("label_sparse_payload", r"label[ _]sparse[ _]payload"),
    ("label_base_cost", r"label[ _]costs"),
    ("label_teacher_cost", r"label[ _]costs"),
    ("label_record_weight", r"label[ _]record[ _]weight"),
])
def test_task4_p1c_validator_semantic_mismatches_follow_accepted_baseline(
        monkeypatch, invalid, expected_error):
    t = _teacher()
    monkeypatch.setattr(t, "_CaseOutcome", _Task4CaseOutcome)
    case_input = _task4_p1c_case_input(t)
    # This gate is deliberately the real production validator.  Until the
    # dedicated baseline test below migrates its schema, each poison is an
    # explicit dependency skip—not a misleading legacy-schema test failure.
    valid = _task4_p1c_require_production_baseline(t, case_input, invalid)
    value = _task4_p1c_poison_semantic_outcome(
        t, case_input, invalid, valid=valid,
    )
    _task4_p1c_assert_poisoned_semantic_invariant(value, invalid)
    with pytest.raises(ValueError, match=expected_error):
        t._validate_outcome(value)


@pytest.mark.parametrize("mode", ["covered", "base_unavailable"])
def test_task4_p1c_production_validator_accepts_exact_valid_baseline_before_poisons(monkeypatch, mode):
    """The sole pre-GREEN RED gate for semantic poison validation."""
    t = _teacher()
    monkeypatch.setattr(t, "_CaseOutcome", _Task4CaseOutcome)
    case_input = _task4_p1c_case_input(t)
    baseline = _task4_p1c_semantic_baseline(t, case_input, mode)
    assert t._validate_outcome(baseline) is baseline


def test_task4_p1c_evidence_compiler_label_and_fingerprint_failures_abort_without_destination(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir(); _task4_shard(root)
    _checkpoint, policy = _task4_verified_checkpoint(t, root)
    monkeypatch.setattr(t, "_materialize_teacher_model", lambda *args: object())
    for seam in ("compiler", "label", "fingerprint"):
        out = tmp_path / seam
        case = _task4_expected_case(0); raw = _task4_p1c_raw(case)
        if seam == "compiler":
            lifecycle = object()  # deliberately malformed compiler input
        elif seam == "label":
            bad_legal = raw.to(dtype=torch.float32)
            record = _task4_p1c_record(
                t, 0, "base", raw, bad_legal, cost=1.0, energy=None,
                energy_status="unavailable", hard={"hard_audit": True}, drift={"max_abs": 0.0},
            )
            lifecycle = t._CandidateLifecycle((record,), 0, 1.0, 1.0)
        else:
            raw = raw.clone(); raw[0, 0] = float("nan")
            record = _task4_p1c_record(
                t, 0, "base", _task4_p1c_raw(case), _task4_p1c_raw(case), cost=1.0,
                energy=None, energy_status="unavailable", hard={"hard_audit": True}, drift={"max_abs": 0.0},
            )
            lifecycle = t._CandidateLifecycle((record,), 0, 1.0, 1.0)
        monkeypatch.setattr(t, "_sample_direct_once", lambda *args, _raw=raw: _raw)
        monkeypatch.setattr(t, "_run_candidate_lifecycle", lambda *args, _life=lifecycle, **kwargs: _life)
        with pytest.raises((ValueError, TypeError)):
            t.teacher_main(_task4_args(root, out), _trust_policy=policy)
        assert not out.exists()


def test_task4_p1c_two_fresh_destinations_are_byte_identical_and_evidence_has_no_geometry_leaks(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    first, second = tmp_path / "first", tmp_path / "second"
    _task4_install_custom_runtime(t, monkeypatch, outcome_factory=lambda module, ci: _task4_p1c_mutation_outcome(module, ci))
    assert t.teacher_main(_task4_args(root, first), _trust_policy=_policy_for(root)) == 0
    _task4_install_custom_runtime(t, monkeypatch, outcome_factory=lambda module, ci: _task4_p1c_mutation_outcome(module, ci))
    assert t.teacher_main(_task4_args(root, second), _trust_policy=_policy_for(root)) == 0
    assert {name: (first / name).read_bytes() for name in _TASK4_FILES} == {
        name: (second / name).read_bytes() for name in _TASK4_FILES
    }
    evidence = {
        "labels": _task4_jsonl(first / "train_labels.jsonl") + _task4_jsonl(first / "heldout_labels.jsonl"),
        "proposals": _task4_jsonl(first / "proposals.jsonl"),
        "rejections": _task4_jsonl(first / "rejections.jsonl"),
        "manifest": json.loads((first / "g0_manifest.json").read_text()),
    }
    assert evidence["manifest"]["coverage"]["legal"] is True

    def assert_path_aware_evidence_hygiene(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = path + (key,)
                if key == "legal":
                    assert child_path == ("manifest", "coverage", "legal")
                else:
                    assert key not in {"original", "positions", "rects", "golden"}
                assert_path_aware_evidence_hygiene(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                assert_path_aware_evidence_hygiene(child, path + (index,))
        elif isinstance(value, str):
            assert not os.path.isabs(value)

    assert_path_aware_evidence_hygiene(evidence)
    serialized = _task4_canonical_json(evidence)
    for forbidden in (b'"original"', b'"positions"', b'"rects"',
                      b'"golden"', b"timestamp", str(root).encode()):
        assert forbidden not in serialized
    _task4_assert_no_forbidden_semantic_values(
        evidence, (701, 703, 401, 403, 709, 719, 809, 811)
    )


def _p1b_reachable_forbidden(source):
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    methods = {
        f"{owner.name}.{node.name}": node
        for owner in ast.walk(tree)
        if isinstance(owner, ast.ClassDef)
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lambdas = {
        f"<lambda@{node.lineno}>": node
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda)
    }
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    callables = {**functions, **methods, **lambdas}
    if "_run_candidate_lifecycle" not in functions:
        return {"missing:_run_candidate_lifecycle"}

    global_aliases = {}

    def resolve(node, aliases):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value, aliases)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            return resolve(node.func, aliases)
        if isinstance(node, ast.Lambda):
            return f"<lambda@{node.lineno}>"
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            base = resolve(node.value, aliases)
            key = str(node.slice.value)
            return aliases.get(f"{base}.{key}", f"{base}.{key}")
        return ""

    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                global_aliases[item.asname or item.name.split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                global_aliases[item.asname or item.name] = f"{node.module}.{item.name}"
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            value = resolve(node.value, global_aliases)
            if value:
                global_aliases[node.targets[0].id] = value
        elif isinstance(node, ast.ClassDef):
            global_aliases[node.name] = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = resolve(node.value, global_aliases)
            if value:
                global_aliases[node.target.id] = value

    forbidden_tokens = (
        "extract_sparse_label", "_g0_state", "_weighted_population",
        "teacher_main", "writer", "jsonl", "manifest", "publish", "fsync",
        "stagedpublication", "write_text", "write_bytes",
    )
    pending = ["_run_candidate_lifecycle"]
    reachable = set()
    findings = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        aliases = dict(global_aliases)

        def executable_nodes(root):
            for child in ast.iter_child_nodes(root):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                yield child
                yield from executable_nodes(child)

        body_nodes = list(executable_nodes(callables[name]))
        assignments = sorted(
            (
                node for node in body_nodes
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ),
            key=lambda node: node.lineno,
        )
        for assignment in assignments:
            target_node = (
                assignment.targets[0]
                if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                else assignment.target if isinstance(assignment, ast.AnnAssign) else None
            )
            if target_node is None:
                continue
            if isinstance(target_node, ast.Name) and isinstance(assignment.value, ast.Dict):
                target = target_node.id
                for key, value_node in zip(assignment.value.keys, assignment.value.values):
                    if isinstance(key, ast.Constant):
                        value = resolve(value_node, aliases)
                        if value:
                            aliases[f"{target}.{key.value}"] = value
                continue
            if (isinstance(target_node, ast.Subscript)
                    and isinstance(target_node.slice, ast.Constant)):
                target = resolve(target_node, aliases)
                value = resolve(assignment.value, aliases)
                if target and value:
                    aliases[target] = value
                continue
            if not isinstance(target_node, ast.Name):
                continue
            value = resolve(assignment.value, aliases)
            if value:
                aliases[target_node.id] = value
        for node in body_nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                path = resolve(node, aliases)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                path = resolve(node, aliases)
            else:
                continue
            lowered = path.lower()
            if any(token in lowered for token in forbidden_tokens):
                findings.add(f"{name}:value:{path}")
        for call in (
            node for node in body_nodes if isinstance(node, ast.Call)
        ):
            path = resolve(call.func, aliases)
            lowered = path.lower()
            if any(token in lowered for token in forbidden_tokens):
                findings.add(f"{name}:{path}")
            if lowered == "open" or lowered.endswith(".open"):
                mode_node = (
                    call.args[1] if len(call.args) > 1 else
                    next((item.value for item in call.keywords if item.arg == "mode"), None)
                )
                if (isinstance(mode_node, ast.Constant)
                        and isinstance(mode_node.value, str)
                        and any(flag in mode_node.value for flag in "wax+")):
                    findings.add(f"{name}:write-open:{path}")
            if (lowered == "write" or lowered.endswith(".write")
                    or lowered.endswith(".write_text")
                    or lowered.endswith(".write_bytes")):
                findings.add(f"{name}:write-call:{path}")
            target = path if path in callables else path.rsplit(".", 1)[-1]
            if target in callables and target not in reachable:
                pending.append(target)
            if path in classes and f"{path}.__init__" in callables:
                pending.append(f"{path}.__init__")
    return findings


def test_task4_p1b_reachable_scope_excludes_label_writer_and_g0():
    source = Path("scripts/probes/icdc_topology_teacher.py").read_text()
    assert _task4_static_forbidden(source, require_exact_loads=True)
    assert _p1b_reachable_forbidden(source) == set()


def test_task4_p1b_reachable_scope_checker_detects_indirect_calls():
    assert _p1b_reachable_forbidden(
        "def helper():\n  extract_sparse_label()\n"
        "def _run_candidate_lifecycle():\n  helper()\n"
    )
    assert _p1b_reachable_forbidden(
        "def helper():\n  sink = extract_sparse_label\n  sink()\n"
        "def _run_candidate_lifecycle():\n  helper()\n"
    )
    assert _p1b_reachable_forbidden(
        "def helper(writer):\n  writer.write({})\n"
        "def _run_candidate_lifecycle():\n  helper(None)\n"
    )
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n"
        "  hooks = {'x': extract_sparse_label}\n"
        "  hooks['x']()\n"
    )
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n  globals()['_g0_state']()\n"
    )
    assert _p1b_reachable_forbidden(
        "class Sink:\n  def run(self):\n    extract_sparse_label()\n"
        "def _run_candidate_lifecycle():\n  sink = Sink()\n  sink.run()\n"
    )
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n"
        "  def never_called():\n    _g0_state()\n"
        "  return 1\n"
    ) == set()
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n"
        "  sink = lambda: extract_sparse_label()\n"
        "  sink()\n"
    )
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n"
        "  sink: object = extract_sparse_label\n"
        "  sink()\n"
    )
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n"
        "  hooks = {}\n"
        "  hooks['x'] = _g0_state\n"
        "  hooks['x']()\n"
    )
    assert _p1b_reachable_forbidden(
        "class Sink:\n"
        "  def __init__(self):\n    extract_sparse_label()\n"
        "def _run_candidate_lifecycle():\n  Sink()\n"
    )
    assert _p1b_reachable_forbidden(
        "import functools\n"
        "def _run_candidate_lifecycle():\n"
        "  sink = functools.partial(extract_sparse_label)\n"
        "  sink()\n"
    )
    assert _p1b_reachable_forbidden(
        "def _run_candidate_lifecycle():\n"
        "  open('proposals.jsonl', 'w')\n"
    )
    assert _p1b_reachable_forbidden(
        "from pathlib import Path\n"
        "def _run_candidate_lifecycle():\n"
        "  Path('proposals.jsonl').write_text('x')\n"
    )
    assert _p1b_reachable_forbidden(
        "def unrelated():\n  _g0_state()\n"
        "def _run_candidate_lifecycle():\n  return 1\n"
    ) == set()
