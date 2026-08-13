#!/usr/bin/env python3
"""Fail-closed Task 4 topology-teacher trust and deterministic helpers."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import numbers
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Optional

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "partner"))

import torch
from icdc.topology_data import CorpusSourceReceipt, fingerprint_case, split_for_id

from icdc.checkpoint_identity import (  # noqa: E402
    IDENTITY_SCHEMA,
    canonical_checkpoint_identity,
)


__all__ = ["teacher_main"]

@dataclass(frozen=True)
class _CaseInput:
    case: Mapping[str, Any]
    receipt: CorpusSourceReceipt
    partition: str
    sample_seed: int

@dataclass(frozen=True)
class _CaseOutcome:
    label_row: Mapping[str, Any]
    proposal_rows: Sequence[Mapping[str, Any]]
    rejection_rows: Sequence[Mapping[str, Any]]
    base_cost: float
    teacher_cost: float
    legal: bool
    covered: bool

@dataclass(frozen=True)
class _TeacherRuntime:
    preflight: Callable[[TeacherTrustPolicy, Path], Mapping[str, Any]]
    process_case: Callable[[_CaseInput], _CaseOutcome]
    authorizing: bool

def _runtime_hooks() -> _TeacherRuntime:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("teacher runtime not implemented")
    return _TeacherRuntime(fail, fail, False)

_CHECKPOINT_SHA256 = "508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659"
_SCORER_SHA256 = "7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1"
_MODEL_IDENTITY = MappingProxyType({
    "identity_schema": IDENTITY_SCHEMA,
    "model_config_sha256": "4c6a1e19f0574af348efa81a758c05524522ad3501d46f18e839774fa933971b",
    "model_keyset_sha256": "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7",
    "ema_keyset_sha256": "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7",
    "ema_state_sha256": "92838740993a697a56f3afdfba4402eb83c8dc095fe43462f8bdaffdb4ef5ecb",
})


@dataclass(frozen=True)
class TeacherTrustPolicy:
    canonical_root: Path
    expected_checkpoint_sha256: str
    allowed_model_identity: Mapping[str, str]
    expected_scorer_sha256: str
    scorer_contract: str
    shapely_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_model_identity, Mapping):
            raise TypeError("allowed_model_identity must be a mapping")
        object.__setattr__(
            self,
            "allowed_model_identity",
            MappingProxyType(dict(self.allowed_model_identity)),
        )


def _production_trust_policy() -> TeacherTrustPolicy:
    return TeacherTrustPolicy(
        canonical_root=(_REPO / "FloorSet" / "floorset_lite").resolve(),
        expected_checkpoint_sha256=_CHECKPOINT_SHA256,
        allowed_model_identity=_MODEL_IDENTITY,
        expected_scorer_sha256=_SCORER_SHA256,
        scorer_contract="iccad2026_evaluate_cost_no_runtime_v1",
        shapely_version="2.0.5",
    )


def _checkpoint_identity_schema() -> str:
    return IDENTITY_SCHEMA


def _checkpoint_identity(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    return canonical_checkpoint_identity(checkpoint)


def _load_verified_checkpoint_bytes(
    path: Path, policy: TeacherTrustPolicy
) -> tuple[Mapping[str, Any], dict[str, str]]:
    try:
        exact_bytes = Path(path).read_bytes()
    except Exception as exc:
        raise ValueError("checkpoint read failed") from exc
    actual_sha256 = hashlib.sha256(exact_bytes).hexdigest()
    if actual_sha256 != policy.expected_checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    try:
        payload = torch.load(
            io.BytesIO(exact_bytes), weights_only=True, map_location="cpu"
        )
    except Exception as exc:
        raise ValueError("checkpoint load failed") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    try:
        actual_identity = canonical_checkpoint_identity(payload)
    except Exception as exc:
        raise ValueError("checkpoint identity failed") from exc
    if actual_identity["model_keyset_sha256"] != actual_identity["ema_keyset_sha256"]:
        raise ValueError("model and EMA keysets differ")
    if not isinstance(policy.allowed_model_identity, Mapping):
        raise ValueError("allowed identity must be a mapping")
    if dict(policy.allowed_model_identity) != actual_identity:
        raise ValueError("checkpoint identity is not trusted")
    return payload, {**actual_identity, "checkpoint_sha256": policy.expected_checkpoint_sha256}


def _sample_seed(seed: int, case_id: str, ordinal: int) -> int:
    material = f"{seed}\0{case_id}\0{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _is_integral(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return math.isfinite(numeric) and numeric.is_integer()
    return False


def _validate_case(case: Any) -> tuple[int, list[list[int]], torch.Tensor]:
    if not isinstance(case, Mapping):
        raise ValueError("case must be a mapping")
    n = case.get("n")
    if type(n) is not int or n <= 0:
        raise ValueError("case n")
    cons = case.get("cons")
    if not isinstance(cons, (list, tuple)) or len(cons) != n:
        raise ValueError("case constraints")
    widths: set[int] = set()
    for row in cons:
        if not isinstance(row, (list, tuple)):
            raise ValueError("case constraint row")
        widths.add(len(row))
    if widths not in ({2}, {5}):
        raise ValueError("case constraint widths")
    normalized_cons: list[list[int]] = []
    for row in cons:
        if len(row) not in (2, 5):
            raise ValueError("case constraint row")
        if any(not _is_integral(value) for value in row):
            raise ValueError("case constraint value")
        normalized = [int(value) for value in row]
        if normalized[0] not in (0, 1) or normalized[1] not in (0, 1):
            raise ValueError("case fixed/preplaced flag")
        if len(normalized) == 2:
            normalized.extend((0, 0, 0))
        elif normalized[2] < 0 or normalized[3] < 0 or not 0 <= normalized[4] <= 15:
            raise ValueError("case constraint metadata")
        normalized_cons.append(normalized)
    area = case.get("area")
    if not isinstance(area, (list, tuple)) or len(area) != n:
        raise ValueError("case area")
    for value in area:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("case area")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("case area")
    try:
        tp = torch.as_tensor(case.get("tp"), dtype=torch.float64, device="cpu")
    except Exception as exc:
        raise ValueError("case tp") from exc
    if tp.shape != (n, 4) or not bool(torch.isfinite(tp).all()):
        raise ValueError("case tp")
    for index, row in enumerate(normalized_cons):
        if row[0] != 0 or row[1] != 0:
            if tp[index, 2] <= 0 or tp[index, 3] <= 0:
                raise ValueError("case fixed dimensions")
        if row[1] != 0 and (tp[index, 0] < 0 or tp[index, 1] < 0):
            raise ValueError("case preplaced origin")
    return n, normalized_cons, tp


def _validate_rects(rects: Any, n: int) -> None:
    if not isinstance(rects, torch.Tensor):
        raise ValueError("rects")
    if rects.device.type != "cpu" or not rects.is_floating_point():
        raise ValueError("rects")
    if rects.ndim != 2 or tuple(rects.shape) != (n, 4):
        raise ValueError("rects")
    if not bool(torch.isfinite(rects).all()) or not bool((rects[:, 2:] > 0).all()):
        raise ValueError("rects")


def _pair_state(rects: torch.Tensor, first: int, second: int) -> tuple[int, int]:
    def gap(axis: int) -> float:
        a = rects[first]
        b = rects[second]
        return max(float(a[axis] - b[axis] - b[axis + 2]),
                   float(b[axis] - a[axis] - a[axis + 2]))

    axis = 0 if gap(0) >= gap(1) else 1
    first_center = float(rects[first, axis] + rects[first, axis + 2] / 2)
    second_center = float(rects[second, axis] + rects[second, axis + 2] / 2)
    return axis, int((first_center, first) <= (second_center, second))


def _exact_contact(
    rects: torch.Tensor, first: int, second: int, axis: int, order: int
) -> bool:
    first_end = rects[first, axis] + rects[first, axis + 2]
    second_end = rects[second, axis] + rects[second, axis + 2]
    exact_face = first_end == rects[second, axis] if order == 1 else second_end == rects[first, axis]
    if not bool(exact_face):
        return False
    perpendicular = 1 - axis
    overlap = min(
        rects[first, perpendicular] + rects[first, perpendicular + 2],
        rects[second, perpendicular] + rects[second, perpendicular + 2],
    ) - max(rects[first, perpendicular], rects[second, perpendicular])
    return bool(overlap > 0)


def _proposal_intent_holds(intent: str, rects: torch.Tensor, case: Mapping[str, Any]) -> bool:
    """Return whether a realized layout satisfies one exact named intent."""

    try:
        n, cons, tp = _validate_case(case)
        _validate_rects(rects, n)
        if not isinstance(intent, str) or not intent:
            return False
        if intent == "base":
            return True
        tokens = intent.split(":")
        kind = tokens[0]
        expected_len = 5 if kind in {"axis", "pin"} else 6 if kind == "contact" else -1
        if len(tokens) != expected_len:
            return False
        try:
            values = [int(token) for token in tokens[1:]]
        except (TypeError, ValueError):
            return False
        if any(token == "" or str(value) != token for token, value in zip(tokens[1:], values)):
            return False
        if kind == "contact":
            gid, first, second, axis, order = values
        else:
            first, second, axis, order = values
        if not (0 <= first < n and 0 <= second < n and first != second):
            return False
        if axis not in (0, 1) or order not in (0, 1):
            return False
        if kind == "axis":
            return _pair_state(rects, first, second) == (axis, order)
        if kind == "pin":
            authorized = cons[first][1] != 0 and bool((tp[first, :2] >= 0).all())
            return authorized and torch.equal(rects[first, :2], tp[first, :2]) and _pair_state(rects, first, second) == (axis, order)
        if cons[first][3] != gid or cons[second][3] != gid or gid <= 0:
            return False
        return _exact_contact(rects, first, second, axis, order)
    except Exception:
        return False


def _select_official_winner(rows: Sequence[Mapping[str, Any]]) -> int:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("winner rows")
    normalized: list[tuple[str, int, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("winner row")
        name = row.get("name")
        ordinal = row.get("ordinal")
        cost = row.get("cost_no_runtime")
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("winner name")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("winner ordinal")
        if type(cost) not in (int, float) or isinstance(cost, bool) or not math.isfinite(float(cost)):
            raise ValueError("winner cost")
        if row.get("feasible") is not True:
            raise ValueError("winner feasibility")
        normalized.append((name, ordinal, float(cost)))
    if sum(name == "base" for name, _ordinal, _cost in normalized) != 1:
        raise ValueError("winner must contain exactly one base")
    if len({name for name, _ordinal, _cost in normalized}) != len(normalized):
        raise ValueError("winner names must be unique")
    if len({ordinal for _name, ordinal, _cost in normalized}) != len(normalized):
        raise ValueError("winner ordinals must be unique")
    return min(
        range(len(normalized)),
        key=lambda index: (
            normalized[index][2],
            normalized[index][1],
            normalized[index][0],
        ),
    )


def _canonical_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError("relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("relative_path")
    return value


def _finite_number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(field)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(field)
    return number


def _weighted_population(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | str]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("population rows")
    required = {"relative_path", "layout_index", "instance_id", "n", "base_cost", "teacher_cost"}
    normalized: list[dict[str, Any]] = []
    source_keys: set[tuple[str, int]] = set()
    instance_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("population row keys")
        relative_path = _canonical_relative_path(row["relative_path"])
        layout_index = row["layout_index"]
        if type(layout_index) is not int or layout_index < 0:
            raise ValueError("layout_index")
        instance_id = row["instance_id"]
        if not isinstance(instance_id, str) or not instance_id or "\0" in instance_id:
            raise ValueError("instance_id")
        if type(row["n"]) is not int or row["n"] < 0:
            raise ValueError("n")
        base_cost = _finite_number(row["base_cost"], "base_cost")
        teacher_cost = _finite_number(row["teacher_cost"], "teacher_cost")
        normalized_layout_index = layout_index
        source_key = (relative_path, normalized_layout_index)
        if source_key in source_keys or instance_id in instance_ids:
            raise ValueError("duplicate population identity")
        source_keys.add(source_key)
        instance_ids.add(instance_id)
        try:
            normalized_n = row["n"]
            weight = math.exp(normalized_n / 12)
        except (OverflowError, ValueError) as exc:
            raise ValueError("weight") from exc
        if not math.isfinite(weight):
            raise ValueError("weight")
        normalized.append({
            "relative_path": relative_path,
            "layout_index": normalized_layout_index,
            "instance_id": instance_id,
            "n": normalized_n,
            "base_cost": base_cost,
            "teacher_cost": teacher_cost,
            "weight": weight,
        })
    normalized.sort(key=lambda row: (row["relative_path"], row["layout_index"]))
    population_rows = [
        {
            "relative_path": row["relative_path"],
            "layout_index": row["layout_index"],
            "instance_id": row["instance_id"],
            "n": row["n"],
            "weight": row["weight"],
        }
        for row in normalized
    ]
    try:
        population_bytes = json.dumps(
            population_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("population identity") from exc
    denominator = 0.0
    base_total = 0.0
    teacher_total = 0.0
    for row in normalized:
        denominator += row["weight"]
        base_product = row["weight"] * row["base_cost"]
        teacher_product = row["weight"] * row["teacher_cost"]
        if not math.isfinite(base_product) or not math.isfinite(teacher_product):
            raise ValueError("weighted cost")
        base_total += base_product
        teacher_total += teacher_product
        if not math.isfinite(denominator) or not math.isfinite(base_total) or not math.isfinite(teacher_total):
            raise ValueError("weighted population arithmetic")
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("denominator")
    base_mean = base_total / denominator
    teacher_mean = teacher_total / denominator
    delta = base_mean - teacher_mean
    if not all(math.isfinite(value) for value in (base_mean, teacher_mean, delta)):
        raise ValueError("weighted population result")
    return {
        "denominator": denominator,
        "B_H": base_mean,
        "T_H": teacher_mean,
        "Delta_H": delta,
        "population_sha256": hashlib.sha256(population_bytes).hexdigest(),
    }


def _manifest_self_sha256(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest")
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    try:
        encoded = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("manifest") from exc
    return hashlib.sha256(encoded).hexdigest()


def _g0_state(values: Mapping[str, Any]) -> str:
    if not isinstance(values, Mapping):
        raise ValueError("G0 state")
    for field in ("trust_ok", "scorer_ok", "input_ok"):
        value = values.get(field)
        if type(value) is not bool:
            raise ValueError(field)
        if not value:
            return "KILLED_INPUT_CHECKPOINT_OR_SCORER"
    for field in ("legal", "coverage"):
        value = values.get(field)
        if type(value) is not bool:
            raise ValueError(field)
        if not value:
            return "KILLED_LEGALITY_OR_COVERAGE"
    teacher_value = values.get("teacher_mean")
    teacher_mean = _finite_number(teacher_value, "teacher_mean")
    delta = _finite_number(values.get("delta"), "delta")
    if teacher_mean > 1.5:
        return "KILLED_TEACHER_GT_1_5"
    if delta < 0.0181504738793652:
        return "STOP_HARD_GAIN_MISSED"
    if delta >= 0.0261247299384228:
        return "TARGET_GAIN_MET"
    return "TARGET_GAIN_MISSED_NO_TRAINING_AUTHORITY"


def _path_has_dot_component(raw: str) -> bool:
    if "\0" in raw:
        return True
    return any(part in {".", ".."} for part in raw.replace(os.sep, "/").split("/"))


def _reject_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor) if absolute.anchor else Path.cwd()
    for component in absolute.parts:
        if component == absolute.anchor:
            continue
        current /= component
        if current.is_symlink():
            raise ValueError("symlink path component")


def _validate_data_root(raw: str, policy: TeacherTrustPolicy) -> Path:
    if not isinstance(raw, str) or _path_has_dot_component(raw):
        raise ValueError("canonical data root")
    root = Path(raw)
    _reject_symlink_components(root)
    if not root.exists() or not root.is_dir():
        raise ValueError("canonical data root")
    resolved = root.resolve()
    if resolved != policy.canonical_root.resolve():
        raise ValueError("canonical data root")
    return resolved


def _validate_outputs(raw_out: str, raw_index: str) -> None:
    if not isinstance(raw_out, str) or not isinstance(raw_index, str):
        raise ValueError("output paths")
    if _path_has_dot_component(raw_out) or _path_has_dot_component(raw_index):
        raise ValueError("output paths")
    out = Path(raw_out)
    index = Path(raw_index)
    _reject_symlink_components(out)
    _reject_symlink_components(index)
    if out.exists() or out.is_symlink():
        raise ValueError("existing output directory")
    expected_index = out / "training_index.json"
    if index != expected_index or index.resolve() != expected_index.resolve():
        raise ValueError("index-out must be out-dir/training_index.json")
    if index.exists() or index.is_symlink():
        raise ValueError("index output must be absent")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed

def _publish_staging(staging: Path, destination: Path) -> None:
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("invalid staging")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, "renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    flags = 1  # RENAME_NOREPLACE
    result = renameat2(-100, os.fsencode(str(staging)), -100,
                       os.fsencode(str(destination)), flags)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ValueError("existing output directory")
    raise OSError(error, os.strerror(error))


def _new_staging(destination: Path) -> Path:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=parent))
    return path


def _iter_approved_shards(root: Path) -> list[tuple[int, int, Path]]:
    found: list[tuple[int, int, Path]] = []
    for worker in root.iterdir():
        if worker.is_symlink() or not worker.is_dir() or not worker.name.startswith("worker_"):
            continue
        try: wid = int(worker.name[7:])
        except ValueError: continue
        for shard in worker.iterdir():
            if shard.is_symlink() or not shard.is_file() or not shard.name.startswith("layouts_") or not shard.name.endswith(".th"):
                continue
            try: lid = int(shard.name[8:-3])
            except ValueError: continue
            found.append((wid, lid, shard))
    return sorted(found, key=lambda x: (x[0], x[1]))


def _read_verified_shard(root: Path, worker: int, layout: int) -> tuple[bytes, Any]:
    wname, sname = f"worker_{worker}", f"layouts_{layout}.th"
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        wfd = os.open(wname, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            fd = os.open(sname, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=wfd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError("shard is not regular")
                chunks = []
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk: break
                    chunks.append(chunk)
            finally: os.close(fd)
        finally: os.close(wfd)
    finally: os.close(root_fd)
    raw = b"".join(chunks)
    return raw, torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")


def _validate_source_shard(source: Any) -> tuple[int, int]:
    if not isinstance(source, (tuple, list)) or len(source) != 7:
        raise ValueError("source schema")
    if any(not isinstance(t, torch.Tensor) or t.dtype == torch.bool or not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in source):
        raise ValueError("source tensors")
    inp, b2b, p2b, pins, tree, fp, metrics = source
    if inp.ndim != 3 or inp.shape[2] != 6 or b2b.ndim != 3 or b2b.shape[2] != 3 or p2b.ndim != 3 or p2b.shape[2] != 3 or pins.ndim != 3 or pins.shape[2] != 2 or tree.ndim != 3 or tree.shape[2] != 3 or fp.ndim != 3 or fp.shape[2] != 4 or metrics.ndim != 2 or metrics.shape[1] != 8:
        raise ValueError("source shapes")
    b, n = inp.shape[:2]
    if b < 1 or n < 1 or any(t.shape[0] != b for t in source[1:]):
        raise ValueError("source batch")
    if tree.shape[1] not in (n - 1,): raise ValueError("source tree")
    for row in inp:
        seen_pad = False
        for item in row:
            pad = float(item[0]) == -1.0
            if pad: seen_pad = True
            elif seen_pad or float(item[0]) <= 0: raise ValueError("area padding")
    for tensor, width in ((b2b, 3), (p2b, 3), (pins, 2)):
        for row in tensor.reshape(-1, width):
            pads = [float(x) == -1.0 for x in row]
            if any(pads) and not all(pads): raise ValueError("partial padding")
    return int(b), int(n)

def _dump_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode() + b"\n")

def _source_case(source: Sequence[torch.Tensor], index: int, instance_id: str) -> dict[str, Any]:
    inp, _b2b, _p2b, _pins, _tree, fp, metrics = source
    row = inp[index].tolist(); fprow = fp[index].tolist(); metric = metrics[index].tolist()
    n = len(row)
    area = [float(x[0]) for x in row]
    cons = [[int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])] for x in row]
    tp = []
    for vals, flags in zip(fprow, cons):
        fixed, pre = flags[:2]
        tp.append([float(vals[2]) if pre else -1.0, float(vals[3]) if pre else -1.0,
                   float(vals[0]) if fixed or pre else -1.0, float(vals[1]) if fixed or pre else -1.0])
    return {"instance_id": instance_id, "n": n, "area": area, "cons": cons, "tp": tp,
            "b2b": [], "p2b": [], "pins": [], "hpwl_ref": float(metric[6] + metric[7]),
            "area_ref": float(metric[0])}


def teacher_main(argv: Optional[Sequence[str]] = None, *, _trust_policy: Optional[TeacherTrustPolicy] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--heldout-mod", type=_positive_int, default=10)
    parser.add_argument("--n-min", type=_nonnegative_int, default=100)
    parser.add_argument("--max-files", type=_positive_int, default=None)
    args = parser.parse_args(argv)
    policy = _trust_policy if _trust_policy is not None else _production_trust_policy()
    _validate_data_root(args.data_root, policy)
    _validate_outputs(args.out_dir, args.index_out)
    root, destination = Path(args.data_root).resolve(), Path(args.out_dir)
    runtime = _runtime_hooks()
    trust = dict(runtime.preflight(policy, Path(args.checkpoint)))
    rows: list[dict[str, Any]] = []; train_c: list[Any] = []; held_c: list[Any] = []
    train_l: list[Any] = []; held_l: list[Any] = []; proposals: list[Any] = []; rejections: list[Any] = []
    population: list[dict[str, Any]] = []; legal = covered = True; processed = 0
    try:
        files = _iter_approved_shards(root)
        if args.max_files is not None: files = files[:args.max_files]
        for _worker, _layout, path in files:
            raw, source = _read_verified_shard(root, _worker, _layout); digest = hashlib.sha256(raw).hexdigest()
            count, blocks = _validate_source_shard(source)
            rel = path.relative_to(root).as_posix()
            for index in range(count):
                iid = f"{rel}#{index}"; case = _source_case(source, index, iid)
                receipt = CorpusSourceReceipt(rel, digest, index, fingerprint_case(case))
                entry = {"receipt": dataclass_to_dict(receipt), "instance_id": iid, "source_row_count": count, "block_count": blocks}
                if case["n"] < args.n_min:
                    rows.append({**entry, "partition": None, "sample_ordinal": None, "sample_seed": None, "status": "excluded_n_min"}); continue
                partition = split_for_id(iid, args.heldout_mod); seed = _sample_seed(args.seed, iid, 0)
                ci = _CaseInput(case, receipt, partition, seed); outcome = runtime.process_case(ci); processed += 1
                env = {"receipt": dataclass_to_dict(receipt), "instance_id": iid, "partition": partition, "sample_seed": seed, "n": case["n"]}
                label = {**env, "proposal_ordinal": 0, "proposal_name": "base", "base_cost": outcome.base_cost, "teacher_cost": outcome.teacher_cost, "record_weight": outcome.base_cost, **dict(outcome.label_row)}
                for p in outcome.proposal_rows: proposals.append({**env, **dict(p)})
                for r in outcome.rejection_rows: rejections.append({**env, **dict(r)})
                (train_c if partition == "train" else held_c).append(case); (train_l if partition == "train" else held_l).append(label)
                if partition == "heldout":
                    population.append({"relative_path": rel, "layout_index": index, "instance_id": iid, "n": case["n"], "base_cost": outcome.base_cost, "teacher_cost": outcome.teacher_cost})
                legal = legal and outcome.legal; covered = covered and outcome.covered
                rows.append({**entry, "partition": partition, "sample_ordinal": 0, "sample_seed": seed, "status": "processed"})
        pop = _weighted_population(population) if population else {"denominator": 0.0, "B_H": 0.0, "T_H": 0.0, "Delta_H": 0.0, "population_sha256": hashlib.sha256(b"[]").hexdigest()}
        coverage = {"eligible_train": len(train_c), "eligible_heldout": len(held_c), "heldout_winners": len(held_l), "legal": legal, "covered": covered and bool(processed)}
        state = _g0_state({**trust, "legal": legal, "coverage": coverage["covered"], "teacher_mean": pop["T_H"], "delta": pop["Delta_H"]})
        authorized = runtime.authorizing and state == "TARGET_GAIN_MET" and args.max_files is None
        if not processed: state = "KILLED_LEGALITY_OR_COVERAGE"; authorized = False
        manifest = {"schema":"icdc_topology_teacher_g0_v1","status":"complete","state":state,"secondary_reasons":[],"training_authorized":authorized,"bounded_max_files":args.max_files is not None,"trust":trust,"population":pop,"coverage":coverage}
        staging = _new_staging(destination)
        outputs = {"train_corpus.jsonl":train_c,"heldout_corpus.jsonl":held_c,"train_labels.jsonl":train_l,"heldout_labels.jsonl":held_l,"proposals.jsonl":proposals,"rejections.jsonl":rejections}
        for name, vals in outputs.items():
            ordered = sorted(vals, key=lambda x: (x.get("instance_id", ""), x.get("proposal_ordinal", 0)))
            (staging / name).write_bytes(b"".join(json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n" for v in ordered))
        _dump_json(staging / "training_index.json", {"schema":"icdc_topology_training_index_v1","rows":rows})
        manifest["support_hashes"] = {n: hashlib.sha256((staging/n).read_bytes()).hexdigest() for n in (*outputs,"training_index.json")}; manifest["self_sha256"] = _manifest_self_sha256(manifest); _dump_json(staging/"g0_manifest.json", manifest)
        _publish_staging(staging, destination)
        return 0 if authorized else 1
    except Exception:
        if 'staging' in locals() and staging.exists():
            import shutil; shutil.rmtree(staging)
        raise

def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return {"relative_path": value.relative_path, "file_sha256": value.file_sha256, "layout_index": value.layout_index, "fingerprint": value.fingerprint}


if __name__ == "__main__":
    teacher_main(sys.argv[1:])
