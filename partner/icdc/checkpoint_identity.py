"""Canonical, fail-closed identities for checkpoint state dictionaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

import torch


IDENTITY_SCHEMA = "icdc_canonical_state_v1"

_IDENTITY_FIELDS = (
    "identity_schema",
    "model_config_sha256",
    "model_keyset_sha256",
    "ema_keyset_sha256",
    "ema_state_sha256",
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash one canonical JSON configuration mapping."""

    if not isinstance(config, Mapping):
        raise ValueError("model_config must be a mapping")
    return hashlib.sha256(_canonical_json_bytes(dict(config))).hexdigest()


def _validate_tensor(key: str, tensor: Any) -> None:
    if not isinstance(key, str) or not key or "\0" in key:
        raise ValueError("state keys must be non-empty strings without NUL")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"state value for {key!r} is not a tensor")
    if tensor.is_meta:
        raise ValueError("meta tensors are not supported")
    if tensor.is_quantized:
        raise ValueError("quantized tensors are not supported")
    if tensor.layout is not torch.strided:
        raise ValueError("only strided tensors are supported")
    try:
        finite = torch.isfinite(tensor).all().item()
    except Exception as exc:
        raise ValueError("tensor finiteness could not be checked") from exc
    if not bool(finite):
        raise ValueError("state tensors must be finite")


def _state_entries(state: Mapping[str, Any]) -> Iterator[tuple[bytes, torch.Tensor]]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a non-empty mapping")
    for key in sorted(state):
        tensor = state[key]
        _validate_tensor(key, tensor)
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = _canonical_json_bytes(list(tensor.shape))
        header = key.encode("utf-8") + b"\0" + dtype.encode("ascii") + b"\0" + shape + b"\0"
        yield header, tensor


def canonical_keyset_sha256(state: Mapping[str, Any]) -> str:
    """Hash sorted state headers without copying tensor values to CPU."""

    digest = hashlib.sha256()
    for header, _tensor in _state_entries(state):
        digest.update(header)
    return digest.hexdigest()


def canonical_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash sorted headers and one detached, contiguous tensor at a time."""

    digest = hashlib.sha256()
    for header, tensor in _state_entries(state):
        digest.update(header)
        try:
            contiguous = tensor.detach().cpu().contiguous()
            raw = memoryview(contiguous.view(torch.uint8).numpy())
            digest.update(raw)
        except Exception as exc:
            raise ValueError("tensor bytes could not be encoded") from exc
    return digest.hexdigest()


def canonical_checkpoint_identity(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact five-field identity for a model/EMA checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    model = checkpoint.get("model")
    ema = checkpoint.get("ema")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("checkpoint model must be a non-empty mapping")
    if not isinstance(ema, Mapping) or not ema:
        raise ValueError("checkpoint EMA must be a non-empty mapping")

    identity = {
        "identity_schema": IDENTITY_SCHEMA,
        "model_config_sha256": canonical_config_sha256(checkpoint.get("model_config")),
        "model_keyset_sha256": canonical_keyset_sha256(model),
        "ema_keyset_sha256": canonical_keyset_sha256(ema),
        "ema_state_sha256": canonical_state_sha256(ema),
    }
    if tuple(identity) != _IDENTITY_FIELDS:
        raise ValueError("invalid checkpoint identity schema")
    return identity
