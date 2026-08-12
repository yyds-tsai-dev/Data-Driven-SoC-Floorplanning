import dataclasses
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import pytest
import torch
from icdc.data import target_positions_from_rects

from icdc.topology_data import (
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
)

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
    return save_sanitized_corpus(path, cases, source_root=CANONICAL_ROOT, **kwargs)


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
        save_sanitized_corpus(tmp_path / "x", [_case()], source_root=None)


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


def test_empty_case_is_valid(tmp_path):
    row = _case(n=0, area=[], cons=[], tp=[], b2b=[], p2b=[], pins=[])
    _save(tmp_path / "empty", [row])
    assert load_sanitized_corpus(tmp_path / "empty")[0]["n"] == 0


@pytest.mark.parametrize(
    "root", [None, "/definitely/wrong", "wrong-relative", object()]
)
def test_source_root_restrictions(tmp_path, root):
    with pytest.raises(ValueError):
        save_sanitized_corpus(tmp_path / "x", [_case()], source_root=root)


@pytest.mark.parametrize("root", [Path("FloorSet/floorset_lite"), CANONICAL_ROOT])
def test_explicit_canonical_source_root(tmp_path, root):
    save_sanitized_corpus(
        tmp_path / "x",
        [_case()],
        source_root=root,
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
