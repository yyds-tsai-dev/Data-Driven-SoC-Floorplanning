"""Strict, leak-free data contracts for topology-prior training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import torch

CORPUS_KEYS = {
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
_FORBIDDEN_KEYS = {"test_id", "golden", "validation", "loader", "provenance"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_ROOT = (_REPO_ROOT / "FloorSet" / "floorset_lite").resolve()


@dataclass(frozen=True)
class SparseEdge:
    src: int
    dst: int
    axis: int
    margin: float
    kind: str
    weight: float


@dataclass(frozen=True)
class ContactLabel:
    a: int
    b: int
    axis: int
    a_before_b: bool
    perp_margin: float
    weight: float


@dataclass(frozen=True)
class TopologyLabel:
    instance_id: str
    n: int
    sample_seed: int
    teacher_cost: float
    base_cost: float
    record_weight: float
    edges: Tuple[SparseEdge, ...]
    contacts: Tuple[ContactLabel, ...]
    pin_paths: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class SparseTopologyBatch:
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_axis: torch.Tensor
    edge_margin: torch.Tensor
    edge_weight: torch.Tensor
    contact_batch: torch.Tensor
    contact_a: torch.Tensor
    contact_b: torch.Tensor
    contact_axis: torch.Tensor
    contact_order: torch.Tensor
    contact_margin: torch.Tensor
    contact_weight: torch.Tensor


def _path(value: str | Path) -> Path:
    try:
        return Path(value)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("invalid path") from exc


def _python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("requires-grad tensor")
        if value.device.type != "cpu":
            raise ValueError("only CPU tensors are accepted")
        if value.ndim == 0:
            return value.item()
        try:
            return value.tolist()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("invalid tensor") from exc
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    value = _python_value(value)
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a sequence")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _real(value: Any, name: str) -> float:
    value = _python_value(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    result = _real(value, name)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _label_float(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a Python scalar")
    return _float(value, name, positive=positive, nonnegative=nonnegative)


def _integer(value: Any, name: str, *, nonnegative: bool = False) -> int:
    result = _real(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be integral")
    integer = int(result)
    if nonnegative and integer < 0:
        raise ValueError(f"{name} must be nonnegative")
    return integer


def _normalize_constraints(value: Any, n: int) -> list[list[int]]:
    rows = _sequence(value, "cons")
    if len(rows) != n:
        raise ValueError("cons length")
    normalized: list[list[int]] = []
    widths: set[int] = set()
    for row in rows:
        values = _sequence(row, "cons row")
        widths.add(len(values))
        if len(values) not in (2, 5):
            raise ValueError("cons width")
        flags = [_integer(values[i], "constraint flag") for i in range(2)]
        if any(flag not in (0, 1) for flag in flags):
            raise ValueError("constraint flag")
        if len(values) == 2:
            normalized.append(flags)
            continue
        mib_id = _integer(values[2], "mib id", nonnegative=True)
        cluster_id = _integer(values[3], "cluster id", nonnegative=True)
        boundary_code = _integer(values[4], "boundary code", nonnegative=True)
        if boundary_code & ~15:
            raise ValueError("boundary code")
        normalized.append(flags + [mib_id, cluster_id, boundary_code])
    if len(widths) > 1:
        raise ValueError("mixed constraint widths")
    return normalized


def _normalize_geometry(value: Any, n: int) -> list[list[float]]:
    rows = _sequence(value, "tp")
    if len(rows) != n:
        raise ValueError("tp length")
    normalized: list[list[float]] = []
    for row in rows:
        values = _sequence(row, "tp row")
        if len(values) != 4:
            raise ValueError("tp width")
        normalized.append([_real(item, "tp value") for item in values])
    return normalized


def _normalize_area(value: Any, n: int) -> list[float]:
    values = _sequence(value, "area")
    if len(values) != n:
        raise ValueError("area length")
    return [_float(item, "area", positive=True) for item in values]


def _normalize_edges(value: Any, n: int, name: str) -> list[list[float | int]]:
    rows = _sequence(value, name)
    normalized: list[list[float | int]] = []
    for row in rows:
        values = _sequence(row, f"{name} row")
        if len(values) != 3:
            raise ValueError(f"{name} width")
        first = _integer(values[0], f"{name} endpoint", nonnegative=True)
        second = _integer(values[1], f"{name} endpoint", nonnegative=True)
        weight = _float(values[2], f"{name} weight", nonnegative=True)
        normalized.append([first, second, weight])
    return normalized


def _normalize_pins(value: Any) -> list[list[float]]:
    rows = _sequence(value, "pins")
    normalized: list[list[float]] = []
    for row in rows:
        values = _sequence(row, "pins row")
        if len(values) != 2:
            raise ValueError("pins width")
        normalized.append(
            [_real(values[0], "pin coordinate"), _real(values[1], "pin coordinate")]
        )
    return normalized


def _sanitize(case: Mapping[str, Any], *, artifact: bool = False) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise ValueError("case must be mapping")
    keys = set(case)
    if artifact:
        if keys != CORPUS_KEYS:
            raise ValueError("sanitized schema mismatch")
    else:
        if keys.intersection({"test_id", "validation", "loader", "provenance"}):
            raise ValueError("forbidden provenance")
        allowed = CORPUS_KEYS | {"golden", "source_split"}
        if keys - allowed:
            raise ValueError("unknown corpus key")
        if "source_split" in case and case["source_split"] != "train":
            raise ValueError("source_split")
        if not CORPUS_KEYS.issubset(keys):
            raise ValueError("missing corpus key")

    instance_id = case.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id")
    n = case.get("n")
    if type(n) is not int or n < 0:
        raise ValueError("n")

    area = _normalize_area(case["area"], n)
    constraints = _normalize_constraints(case["cons"], n)
    geometry = _normalize_geometry(case["tp"], n)
    b2b = _normalize_edges(case["b2b"], n, "b2b")
    p2b = _normalize_edges(case["p2b"], n, "p2b")
    pins = _normalize_pins(case["pins"])

    for row in b2b:
        if row[0] >= n or row[1] >= n:
            raise ValueError("b2b endpoint range")
    for row in p2b:
        if row[0] >= len(pins) or row[1] >= n:
            raise ValueError("p2b endpoint range")

    hpwl_ref = _float(case["hpwl_ref"], "hpwl_ref", nonnegative=True)
    area_ref = _float(case["area_ref"], "area_ref", positive=True)
    masked_geometry: list[list[float]] = []
    for row, constraint in zip(geometry, constraints):
        fixed = constraint[0] == 1
        preplaced = constraint[1] == 1
        masked_geometry.append(
            [
                row[0] if preplaced else -1.0,
                row[1] if preplaced else -1.0,
                row[2] if fixed or preplaced else -1.0,
                row[3] if fixed or preplaced else -1.0,
            ]
        )

    return {
        "instance_id": instance_id,
        "n": n,
        "area": area,
        "cons": constraints,
        "tp": masked_geometry,
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "hpwl_ref": hpwl_ref,
        "area_ref": area_ref,
    }


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("not canonical JSON") from exc


def fingerprint_case(case: Mapping[str, Any]) -> str:
    clean = _sanitize(case)
    clean.pop("instance_id")
    return hashlib.sha256(_canonical(clean)).hexdigest()


def split_for_id(instance_id: str, heldout_mod: int = 10) -> str:
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id")
    if type(heldout_mod) is not int or heldout_mod <= 0:
        raise ValueError("heldout_mod")
    digest = int(hashlib.sha256(instance_id.encode("utf-8")).hexdigest(), 16)
    return "heldout" if digest % heldout_mod == 0 else "train"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_sanitized_corpus(
    path: str | Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    source_root: str | Path | None,
) -> None:
    try:
        root = _path(source_root).resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("source_root") from exc
    if root != _CANONICAL_ROOT:
        raise ValueError("source_root")
    if isinstance(cases, (str, bytes, Mapping)) or not isinstance(cases, Sequence):
        raise ValueError("cases")
    rows = [_sanitize(case) for case in cases]
    fingerprints = [fingerprint_case(row) for row in rows]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("duplicate fingerprint")
    data = b"".join(_canonical(row) + b"\n" for row in rows)
    _atomic_write(_path(path), data)


def load_sanitized_corpus(path: str | Path) -> list[dict[str, Any]]:
    corpus_path = _path(path)
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read corpus") from exc
    rows = [json.loads(line) for line in lines if line.strip()]
    clean = [_sanitize(row, artifact=True) for row in rows]
    fingerprints = [fingerprint_case(row) for row in clean]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("duplicate fingerprint")
    return clean


def _check_canonical(value: Any) -> None:
    if isinstance(value, Mapping):
        if _FORBIDDEN_KEYS.intersection(value):
            raise ValueError("forbidden canonical field")
        if "source_split" in value and value["source_split"] != "train":
            raise ValueError("source_split")
        for item in value.values():
            _check_canonical(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_canonical(item)


def canonical_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence):
        raise ValueError("rows")
    materialized = list(rows)
    for row in materialized:
        if not isinstance(row, Mapping):
            raise ValueError("row")
        _check_canonical(row)
    data = b"".join(_canonical(row) + b"\n" for row in materialized)
    _atomic_write(_path(path), data)


def canonical_jsonl_sha256(path: str | Path) -> str:
    target = _path(path)
    if not target.is_file():
        raise ValueError("manifest file")
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("cannot hash file") from exc


def write_sha256_manifest(
    path: str | Path,
    files: Iterable[str | Path],
) -> dict[str, str]:
    if isinstance(files, (str, bytes, Mapping)):
        raise ValueError("files")
    try:
        materialized = list(files)
    except TypeError as exc:
        raise ValueError("files") from exc
    resolved: list[Path] = []
    for file in materialized:
        try:
            resolved.append(_path(file).resolve())
        except (TypeError, ValueError, OSError) as exc:
            raise ValueError("manifest path") from exc
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate manifest path")
    if any(not file.is_file() for file in resolved):
        raise ValueError("manifest file")
    manifest = {
        str(file): canonical_jsonl_sha256(file) for file in sorted(resolved, key=str)
    }
    _atomic_write(_path(path), _canonical(manifest) + b"\n")
    return manifest


def sha256_manifest(path: str | Path, files: Iterable[str | Path]) -> dict[str, str]:
    return write_sha256_manifest(path, files)


def _validate_device_dtype(device: torch.device, dtype: torch.dtype) -> torch.device:
    try:
        resolved_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("invalid device") from exc
    if resolved_device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("unsupported device")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    return resolved_device


def _float_tensor(
    values: list[float], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    try:
        tensor = torch.tensor(values, device=device, dtype=dtype)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cannot create tensor") from exc
    if not torch.isfinite(tensor).all().item():
        raise ValueError("nonfinite tensor")
    return tensor


def _long_tensor(values: list[int], device: torch.device) -> torch.Tensor:
    try:
        return torch.tensor(values, device=device, dtype=torch.long)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cannot create tensor") from exc


def collate_labels(
    labels: Sequence[TopologyLabel],
    device: torch.device,
    dtype: torch.dtype,
) -> SparseTopologyBatch:
    resolved_device = _validate_device_dtype(device, dtype)
    if isinstance(labels, (str, bytes, Mapping)) or not isinstance(
        labels, (list, tuple)
    ):
        raise ValueError("labels")

    edges: list[tuple[int, int, int, int, float, float]] = []
    contacts: list[tuple[int, int, int, int, int, float, float]] = []
    for batch_index, label in enumerate(labels):
        if type(label) is not TopologyLabel:
            raise ValueError("label type")
        if not isinstance(label.instance_id, str) or not label.instance_id.strip():
            raise ValueError("instance_id")
        if type(label.n) is not int or label.n < 0:
            raise ValueError("label n")
        if type(label.sample_seed) is not int:
            raise ValueError("sample_seed")
        teacher_cost = _label_float(label.teacher_cost, "teacher_cost", positive=True)
        base_cost = _label_float(label.base_cost, "base_cost", positive=True)
        record_weight = _label_float(
            label.record_weight, "record_weight", positive=True
        )
        del teacher_cost, base_cost
        if type(label.edges) is not tuple:
            raise ValueError("edges container")
        if type(label.contacts) is not tuple:
            raise ValueError("contacts container")
        if type(label.pin_paths) is not tuple:
            raise ValueError("pin paths container")

        for edge in label.edges:
            if type(edge) is not SparseEdge:
                raise ValueError("edge type")
            if type(edge.src) is not int or type(edge.dst) is not int:
                raise ValueError("edge endpoint type")
            if not 0 <= edge.src < label.n or not 0 <= edge.dst < label.n:
                raise ValueError("edge endpoint range")
            if edge.src == edge.dst:
                raise ValueError("edge self endpoint")
            if type(edge.axis) is not int or edge.axis not in (0, 1):
                raise ValueError("edge axis")
            if not isinstance(edge.kind, str) or not edge.kind.strip():
                raise ValueError("edge kind")
            margin = _label_float(edge.margin, "edge margin", nonnegative=True)
            weight = _label_float(edge.weight, "edge weight", positive=True)
            effective_weight = weight * record_weight
            if not math.isfinite(effective_weight):
                raise ValueError("edge effective weight")
            edges.append(
                (batch_index, edge.src, edge.dst, edge.axis, margin, effective_weight)
            )

        for contact in label.contacts:
            if type(contact) is not ContactLabel:
                raise ValueError("contact type")
            if type(contact.a) is not int or type(contact.b) is not int:
                raise ValueError("contact endpoint type")
            if not 0 <= contact.a < label.n or not 0 <= contact.b < label.n:
                raise ValueError("contact endpoint range")
            if contact.a == contact.b:
                raise ValueError("contact self endpoint")
            if type(contact.axis) is not int or contact.axis not in (0, 1):
                raise ValueError("contact axis")
            if type(contact.a_before_b) is not bool:
                raise ValueError("contact order")
            perp_margin = _label_float(
                contact.perp_margin,
                "contact perpendicular margin",
                positive=True,
            )
            weight = _label_float(contact.weight, "contact weight", positive=True)
            effective_weight = weight * record_weight
            if not math.isfinite(effective_weight):
                raise ValueError("contact effective weight")
            contacts.append(
                (
                    batch_index,
                    contact.a,
                    contact.b,
                    contact.axis,
                    int(contact.a_before_b),
                    perp_margin,
                    effective_weight,
                )
            )

        for path in label.pin_paths:
            if type(path) is not tuple or not path:
                raise ValueError("pin path")
            if any(type(index) is not int for index in path):
                raise ValueError("pin path index type")
            if any(not 0 <= index < label.n for index in path):
                raise ValueError("pin path index range")
            if any(left == right for left, right in zip(path, path[1:])):
                raise ValueError("pin path repeated adjacent index")

    edge_batch = _long_tensor([row[0] for row in edges], resolved_device)
    edge_src = _long_tensor([row[1] for row in edges], resolved_device)
    edge_dst = _long_tensor([row[2] for row in edges], resolved_device)
    edge_axis = _long_tensor([row[3] for row in edges], resolved_device)
    edge_margin = _float_tensor([row[4] for row in edges], resolved_device, dtype)
    edge_weight = _float_tensor([row[5] for row in edges], resolved_device, dtype)
    contact_batch = _long_tensor([row[0] for row in contacts], resolved_device)
    contact_a = _long_tensor([row[1] for row in contacts], resolved_device)
    contact_b = _long_tensor([row[2] for row in contacts], resolved_device)
    contact_axis = _long_tensor([row[3] for row in contacts], resolved_device)
    contact_order = _long_tensor([row[4] for row in contacts], resolved_device)
    contact_margin = _float_tensor([row[5] for row in contacts], resolved_device, dtype)
    contact_weight = _float_tensor([row[6] for row in contacts], resolved_device, dtype)
    return SparseTopologyBatch(
        edge_batch=edge_batch,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_axis=edge_axis,
        edge_margin=edge_margin,
        edge_weight=edge_weight,
        contact_batch=contact_batch,
        contact_a=contact_a,
        contact_b=contact_b,
        contact_axis=contact_axis,
        contact_order=contact_order,
        contact_margin=contact_margin,
        contact_weight=contact_weight,
    )
