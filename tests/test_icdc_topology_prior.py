import pytest

from icdc.topology_data import (
    fingerprint_case, load_sanitized_corpus, save_sanitized_corpus,
    split_for_id,
)


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
