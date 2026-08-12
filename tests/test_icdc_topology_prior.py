import dataclasses
import hashlib
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import pytest
import torch
from icdc.data import target_positions_from_rects

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
    topology_data._CANONICAL_ROOT = root.resolve()
    source_cases = []
    receipts = []
    for index, case in enumerate(cases):
        try:
            n = int(case["n"])
            cons = torch.as_tensor(case["cons"], dtype=torch.float64)
            area = torch.as_tensor(case["area"], dtype=torch.float64)
            tp = torch.as_tensor(case["tp"], dtype=torch.float64)
            fp = []
            for row in tp:
                width = float(row[2]) if float(row[2]) > 0 else 1.0
                height = float(row[3]) if float(row[3]) > 0 else 1.0
                x = float(row[0]) if float(row[0]) >= 0 else 0.0
                y = float(row[1]) if float(row[1]) >= 0 else 0.0
                fp.append([width, height, x, y])
            input_rows = torch.cat((area.reshape(n, 1), cons), dim=1)
            shard = [
                [input_rows],
                [torch.as_tensor(case["b2b"], dtype=torch.float64).reshape(-1, 3)],
                [torch.as_tensor(case["p2b"], dtype=torch.float64).reshape(-1, 3)],
                [torch.as_tensor(case["pins"], dtype=torch.float64).reshape(-1, 2)],
                [torch.zeros((n, 1), dtype=torch.float64)],
                [torch.tensor(fp, dtype=torch.float64)],
                [
                    torch.tensor(
                        [float(case["area_ref"]), 0.0, float(case["hpwl_ref"])],
                        dtype=torch.float64,
                    )
                ],
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
            source_cases.append(dict(case, instance_id=source_instance_id(receipt)))
            receipts.append(receipt)
        except Exception:
            source_cases.append(case)
            receipts.append(None)
    for i, receipt in enumerate(receipts):
        if receipt is not None:
            receipts[i] = dataclasses.replace(
                receipt,
                fingerprint=fingerprint_case(source_cases[i]),
            )
    return save_sanitized_corpus(
        path,
        source_cases,
        source_root=root,
        source_receipts=receipts,
        **kwargs,
    )


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
        metrics.append(torch.tensor([100.0 + index, 0.0, 5.0], dtype=torch.float64))
    shard = [
        raw_inputs,
        [torch.empty((0, 3), dtype=torch.float64) for _ in range(count)],
        [torch.empty((0, 3), dtype=torch.float64) for _ in range(count)],
        [torch.empty((0, 2), dtype=torch.float64) for _ in range(count)],
        [torch.zeros((2, 1), dtype=torch.float64) for _ in range(count)],
        raw_fps,
        metrics,
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


def test_empty_case_is_valid(tmp_path):
    row = _case(n=0, area=[], cons=[], tp=[], b2b=[], p2b=[], pins=[])
    _save(tmp_path / "empty", [row])
    assert load_sanitized_corpus(tmp_path / "empty")[0]["n"] == 0


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
