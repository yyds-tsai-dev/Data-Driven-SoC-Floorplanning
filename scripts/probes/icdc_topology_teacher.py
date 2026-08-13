#!/usr/bin/env python3
"""Fail-closed Task 4 topology-teacher trust and deterministic helpers."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import numbers
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import torch


_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "partner"))

from icdc.checkpoint_identity import (  # noqa: E402
    IDENTITY_SCHEMA,
    canonical_checkpoint_identity,
)


__all__ = ["teacher_main"]

_CHECKPOINT_SHA256 = "508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659"
_SCORER_SHA256 = "7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1"
_MODEL_IDENTITY = {
    "identity_schema": IDENTITY_SCHEMA,
    "model_config_sha256": "4c6a1e19f0574af348efa81a758c05524522ad3501d46f18e839774fa933971b",
    "model_keyset_sha256": "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7",
    "ema_keyset_sha256": "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7",
    "ema_state_sha256": "92838740993a697a56f3afdfba4402eb83c8dc095fe43462f8bdaffdb4ef5ecb",
}


@dataclass(frozen=True)
class TeacherTrustPolicy:
    canonical_root: Path
    expected_checkpoint_sha256: str
    allowed_model_identity: Mapping[str, str]
    expected_scorer_sha256: str
    scorer_contract: str
    shapely_version: str


def _production_trust_policy() -> TeacherTrustPolicy:
    return TeacherTrustPolicy(
        canonical_root=(_REPO / "FloorSet" / "floorset_lite").resolve(),
        expected_checkpoint_sha256=_CHECKPOINT_SHA256,
        allowed_model_identity=dict(_MODEL_IDENTITY),
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
    if isinstance(cons, (str, bytes)) or not isinstance(cons, Sequence) or len(cons) != n:
        raise ValueError("case constraints")
    normalized_cons: list[list[int]] = []
    for row in cons:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 5:
            raise ValueError("case constraint row")
        if any(not _is_integral(value) for value in row):
            raise ValueError("case constraint value")
        normalized_cons.append([int(value) for value in row])
    try:
        tp = torch.as_tensor(case.get("tp"), dtype=torch.float64, device="cpu")
    except Exception as exc:
        raise ValueError("case tp") from exc
    if tp.shape != (n, 4) or not bool(torch.isfinite(tp).all()):
        raise ValueError("case tp")
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
        raise ValueError("output directory must be absent")
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


def teacher_main(argv: Optional[Sequence[str]] = None, *, _trust_policy: Optional[TeacherTrustPolicy] = None) -> None:
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
    raise RuntimeError("teacher pipeline not implemented")


if __name__ == "__main__":
    teacher_main(sys.argv[1:])
