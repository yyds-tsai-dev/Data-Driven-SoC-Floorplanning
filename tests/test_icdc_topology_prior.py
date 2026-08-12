import pytest

from icdc.topology_data import (
    fingerprint_case, load_sanitized_corpus, save_sanitized_corpus,
    split_for_id,
)
from icdc.topology_data import ContactLabel, SparseEdge, TopologyLabel, collate_labels, canonical_jsonl, canonical_jsonl_sha256, write_sha256_manifest
import torch
from dataclasses import fields, FrozenInstanceError
import dataclasses


def _case(**extra):
    row = {"instance_id": "train-7", "n": 3,
           "area": [4.0, 12.0, 30.0], "cons": [[0, 0], [1, 0], [0, 1]],
           "tp": [[7., 8., 2., 2.], [9., 9., 3., 4.], [5., 6., 5., 6.]],
           "b2b": [], "p2b": [], "pins": [], "hpwl_ref": 10.0,
           "area_ref": 100.0}
    row.update(extra)
    return row


def test_sanitize_excludes_golden_and_masks_non_input_geometry(tmp_path):
    case = _case(golden=[[99., 99., 99., 99.]])
    save_sanitized_corpus(tmp_path / "c.jsonl", [case])
    row = load_sanitized_corpus(tmp_path / "c.jsonl")[0]
    assert set(row) == {"instance_id", "n", "area", "cons", "tp", "b2b", "p2b", "pins", "hpwl_ref", "area_ref"}
    assert row["tp"] == [[-1., -1., -1., -1.], [-1., -1., 3., 4.], [5., 6., 5., 6.]]
    assert "golden" not in row and "test_id" not in row


def test_corpus_rejects_duplicates_bad_values_and_provenance(tmp_path):
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(), _case()])
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(area=[float("nan"), 1, 1])])
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(cons=[[0, 0]])])
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(source_split="validation")])


def test_fingerprint_and_split_are_deterministic():
    assert fingerprint_case(_case()) == fingerprint_case(_case())
    assert split_for_id("abc") in {"train", "heldout"}
    assert split_for_id("abc") == split_for_id("abc")

def test_rejects_test_id_and_explicit_none_root(tmp_path):
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(test_id=1)])
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case()], source_root=None)

def test_duplicate_content_under_distinct_ids_and_invalid_split():
    with pytest.raises(ValueError): save_sanitized_corpus("/tmp/dup.jsonl", [_case(), _case(instance_id="other")])
    for args in (("", 10), ("x", True), ("x", 0)):
        with pytest.raises(ValueError): split_for_id(*args)

@pytest.mark.parametrize("bad", [None, "abc", {"x": 1}, [[1, 2]]])
def test_nested_schema_errors_are_value_errors(tmp_path, bad):
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(b2b=bad)])

def test_geometry_indices_and_recursive_canonical_forbidden(tmp_path):
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(b2b=[[True, 1, 1]])])
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "x", [_case(b2b=[[0, 9, 1]])])
    with pytest.raises(ValueError): canonical_jsonl(tmp_path / "x", [{"nested": [{"golden": 1}]}])

def test_manifest_hash_and_collate_contract(tmp_path):
    p = tmp_path / "x.jsonl"; canonical_jsonl(p, [{"b": 1, "a": 2}])
    assert canonical_jsonl_sha256(p) == canonical_jsonl_sha256(p)
    assert write_sha256_manifest(tmp_path / "m", [p])[str(p)] == canonical_jsonl_sha256(p)
    label = TopologyLabel("x", 2, 1, 1., 1., 1., (SparseEdge(0, 1, 0, 1., "sep", 1.),), (), ())
    batch = collate_labels([label], torch.device("cpu"), torch.float32)
    assert batch.edge_batch.shape == (1,) and batch.edge_weight.dtype == torch.float32
    empty = collate_labels([], torch.device("cpu"), torch.float64)
    assert all(getattr(empty, f).shape == (0,) for f in empty.__dataclass_fields__)

def test_schema_field_order_and_frozen_contract():
    assert [f.name for f in fields(SparseEdge)] == ["src", "dst", "axis", "margin", "kind", "weight"]
    assert [f.name for f in fields(ContactLabel)] == ["a", "b", "axis", "a_before_b", "perp_margin", "weight"]
    assert [f.name for f in fields(TopologyLabel)] == ["instance_id", "n", "sample_seed", "teacher_cost", "base_cost", "record_weight", "edges", "contacts", "pin_paths"]
    with pytest.raises(FrozenInstanceError):
        SparseEdge(0, 1, 0, 1., "x", 1.).src = 2

def test_five_column_constraints_are_retained_and_two_column_masks(tmp_path):
    row = _case(cons=[[1, 1, 1, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 1, 1]])
    save_sanitized_corpus(tmp_path / "five", [row])
    got = load_sanitized_corpus(tmp_path / "five")[0]
    assert got["cons"] == row["cons"]
    assert got["tp"] == [[7., 8., 2., 2.], [-1., -1., -1., -1.], [5., 6., 5., 6.]]

@pytest.mark.parametrize("bad", ["abc", {"x": 1}, None, 4])
def test_sequence_inputs_reject_non_sequences(tmp_path, bad):
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "bad", bad)

def test_zero_hpwl_and_self_b2b_allowed(tmp_path):
    row = _case(hpwl_ref=0., b2b=[[0, 0, 0.]])
    save_sanitized_corpus(tmp_path / "ok", [row])
    assert load_sanitized_corpus(tmp_path / "ok")[0]["hpwl_ref"] == 0.

@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf"), "0"])
def test_bad_hpwl_rejected(tmp_path, bad):
    with pytest.raises(ValueError): save_sanitized_corpus(tmp_path / "bad", [_case(hpwl_ref=bad)])

def test_canonical_nonfinite_unserializable_and_no_partial(tmp_path):
    p = tmp_path / "x"; p.write_text("original")
    for row in ({"x": float("nan")}, {"x": object()}):
        with pytest.raises(ValueError): canonical_jsonl(p, [row])
        assert p.read_text() == "original"

def test_manifest_missing_and_no_partial(tmp_path):
    p = tmp_path / "manifest"; p.write_text("original")
    with pytest.raises(ValueError): write_sha256_manifest(p, [tmp_path / "missing"])
    assert p.read_text() == "original"

@pytest.mark.parametrize("bad", [True, 1.0, -1, "1"])
def test_split_invalid_matrix(bad):
    with pytest.raises(ValueError): split_for_id("x", bad)

@pytest.mark.parametrize("bad", [object(), None, {"src": 1}, "x"])
def test_collate_rejects_bad_labels_container(bad):
    with pytest.raises(ValueError): collate_labels(bad, torch.device("cpu"), torch.float32)
