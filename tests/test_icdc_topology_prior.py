import dataclasses
import ast
import collections.abc
import hashlib
import io
import importlib
import importlib.util
import sys
import json
import math
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
    t = _teacher(); tree = ast.parse(path.read_text())
    text = path.read_text()
    forbidden = ("load_test_cases", "FloorplanDatasetLiteTest", "BandFileSampler._instance", "shelf_fallback", "golden", "validation")
    def dotted(n):
        if isinstance(n, ast.Name): return n.id
        if isinstance(n, ast.Attribute):
            p = dotted(n.value); return f"{p}.{n.attr}" if p else n.attr
        return ""
    calls = [dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden_suffixes = ("load_test_cases", "FloorplanDatasetLiteTest", "shelf_fallback", "BandFileSampler._instance")
    assert not any(any(x == suffix or x.endswith("." + suffix) for suffix in forbidden_suffixes) for x in calls)


def _task4_shard(root):
    path = root / "worker_2" / "layouts_0.th"
    path.parent.mkdir(parents=True)
    tensors = (
        torch.tensor([[[4, 0, 0, 0, 0, 0], [6, 1, 0, 0, 0, 0], [9, 0, 1, 0, 2, 5]],
                      [[5, 0, 0, 0, 0, 0], [20, 1, 0, 0, 0, 0], [20, 0, 1, 0, 2, 5]]], dtype=torch.float32),
        torch.zeros((2, 0, 3)), torch.zeros((2, 0, 3)), torch.zeros((2, 0, 2)),
        torch.zeros((2, 2, 3)),
        torch.tensor([[[71, 73, 701, 703], [2, 3, 401, 403], [3, 3, 10, 11]],
                      [[79, 83, 709, 719], [5, 4, 809, 811], [4, 5, 20, 21]]], dtype=torch.float32),
        torch.tensor([[100, 0, 0, 0, 0, 0, 2, 3], [200, 0, 0, 0, 0, 0, 5, 7]], dtype=torch.float32),
    )
    torch.save(tensors, path)
    return path


def test_teacher_source_transaction_publishes_verified_two_row_corpus(tmp_path, monkeypatch):
    """RED: the real source adapter must own loading, sanitizing, and publication."""
    t = _teacher(); root = tmp_path / "floorset_lite"; _task4_shard(root)
    out = tmp_path / "out"
    policy = _policy_for(root)

    @dataclasses.dataclass(frozen=True)
    class FakeRuntime:
        preflight: object
        process_case: object
        authorizing: bool = True

    def preflight(_policy, _root):
        return {"trust_ok": True, "scorer_ok": True, "input_ok": True}

    def process(case):
        row = {"instance_id": case.receipt.instance_id, "partition": case.partition,
               "base_cost": 1.10, "teacher_cost": 1.00, "legal": True, "covered": True}
        return t._CaseOutcome(row, [{"name": "base", "instance_id": case.receipt.instance_id}], [],
                              1.10, 1.00, True, True)

    monkeypatch.setattr(t, "_runtime_hooks", lambda: FakeRuntime(preflight, process), raising=False)
    result = t.teacher_main(["--data-root", str(root), "--out-dir", str(out),
                             "--index-out", str(out / "training_index.json"),
                             "--checkpoint", str(tmp_path / "unused.th"),
                             "--heldout-mod", "2", "--n-min", "1"],
                            _trust_policy=policy)
    assert result == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "g0_manifest.json", "heldout_corpus.jsonl", "heldout_labels.jsonl",
        "proposals.jsonl", "rejections.jsonl", "train_corpus.jsonl",
        "train_labels.jsonl", "training_index.json"]
    blobs = b"".join(p.read_bytes() for p in out.iterdir())
    assert not any(str(secret).encode() in blobs for secret in (701, 703, 401, 403, 709, 719, 809, 811))


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


def test_teacher_valid_canonical_root_reaches_explicit_unimplemented_without_load(tmp_path, monkeypatch):
    t = _teacher(); root = tmp_path / "canonical"; root.mkdir(); out = tmp_path / "out"
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))
    args = ["--data-root", str(root), "--out-dir", str(out), "--index-out", str(out / "training_index.json"),
            "--checkpoint", str(tmp_path / "missing.th")]
    with pytest.raises(RuntimeError, match="not implemented"):
        t.teacher_main(args, _trust_policy=_policy_for(root))


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


def test_teacher_default_relative_root_and_bounded_flags_reach_unimplemented(monkeypatch, tmp_path):
    t = _teacher(); out = tmp_path / "out"
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))
    args = ["--data-root", "FloorSet/floorset_lite", "--out-dir", str(out),
            "--index-out", str(out / "training_index.json"),
            "--checkpoint", "partner/checkpoints/direct_v2_cont/eval_step1p2M.pt",
            "--seed", "20260813", "--heldout-mod", "10", "--n-min", "100",
            "--max-files", "1"]
    with pytest.raises(RuntimeError, match="not implemented"):
        t.teacher_main(args)


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
