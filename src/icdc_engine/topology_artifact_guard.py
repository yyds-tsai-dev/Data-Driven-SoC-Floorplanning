"""Canonical, dense-fp-free evidence persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


_DENSE_KEYS = frozenset(
    {
        "fp_sol",
        "fp_xywh",
        "dense_fp",
        "rects",
        "positions",
        "origin",
        "width",
        "height",
        "x",
        "y",
        "w",
        "h",
        "gap",
        "overlap",
        "overlap_magnitude",
    }
)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("canonical JSON") from exc


def assert_no_dense_fp_artifact(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key in _DENSE_KEYS:
                raise ValueError("dense fp artifact")
            assert_no_dense_fp_artifact(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_dense_fp_artifact(child)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        return
    else:
        raise ValueError("canonical JSON")


def write_checked_json(path: Path, value: object) -> str:
    assert_no_dense_fp_artifact(value)
    payload = canonical_json_bytes(value) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
