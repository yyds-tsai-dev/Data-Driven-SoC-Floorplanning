"""Strict, leak-free data contracts for topology-prior training."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch

CORPUS_KEYS = {"instance_id", "n", "area", "cons", "tp", "b2b", "p2b", "pins", "hpwl_ref", "area_ref"}
_FORBIDDEN = {"test_id", "golden", "validation", "loader", "provenance"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_ROOT = (_REPO_ROOT / "FloorSet" / "floorset_lite").resolve()
_DEFAULT_ROOT = object()

@dataclass(frozen=True)
class SparseEdge:
    src: int; dst: int; axis: int; margin: float; kind: str; weight: float

@dataclass(frozen=True)
class ContactLabel:
    a: int; b: int; axis: int; a_before_b: bool; perp_margin: float; weight: float

@dataclass(frozen=True)
class TopologyLabel:
    instance_id: str; n: int; sample_seed: int; teacher_cost: float; base_cost: float; record_weight: float
    edges: Tuple[SparseEdge, ...]; contacts: Tuple[ContactLabel, ...]; pin_paths: Tuple[Tuple[int, ...], ...]

@dataclass(frozen=True)
class SparseTopologyBatch:
    edge_batch: torch.Tensor; edge_src: torch.Tensor; edge_dst: torch.Tensor; edge_axis: torch.Tensor
    edge_margin: torch.Tensor; edge_weight: torch.Tensor; contact_batch: torch.Tensor; contact_a: torch.Tensor
    contact_b: torch.Tensor; contact_axis: torch.Tensor; contact_order: torch.Tensor; contact_margin: torch.Tensor
    contact_weight: torch.Tensor

def _num(v: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return False
    return (not positive or float(v) > 0) and (not nonnegative or float(v) >= 0)

def _sanitize(case: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(case, Mapping): raise ValueError("case must be mapping")
    if any(k in case for k in ("test_id", "validation", "loader", "provenance")): raise ValueError("forbidden provenance")
    if "source_split" in case and case["source_split"] != "train": raise ValueError("source_split")
    allowed_input = CORPUS_KEYS | {"golden", "source_split"}
    if set(case) - allowed_input: raise ValueError("unknown corpus key")
    if not CORPUS_KEYS.issubset(case): raise ValueError("missing corpus key")
    if not isinstance(case["instance_id"], str) or not case["instance_id"].strip(): raise ValueError("instance_id")
    n = case["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 0: raise ValueError("n")
    for key in ("area", "cons", "tp"):
        if not isinstance(case[key], (list, tuple)) or len(case[key]) != n: raise ValueError(key)
    if any(not _num(x, positive=True) for x in case["area"]): raise ValueError("area")
    widths = {len(r) for r in case["cons"] if isinstance(r, (list, tuple))}
    if widths not in ({2}, {5}) or any(not isinstance(r, (list, tuple)) or len(r) not in (2, 5) or any(type(x) is not int or x not in (0, 1) for x in r) for r in case["cons"]): raise ValueError("cons")
    for r in case["tp"]:
        if not isinstance(r, (list, tuple)) or len(r) != 4 or any(not _num(x) for x in r): raise ValueError("tp")
    for key, width in (("b2b", 3), ("p2b", 3), ("pins", 2)):
        if not isinstance(case[key], (list, tuple)): raise ValueError(key)
        if any(not isinstance(r, (list, tuple)) or len(r) != width or any(not _num(x) for x in r) for r in case[key]): raise ValueError(key)
    for r in case["b2b"]:
        if type(r[0]) is not int or type(r[1]) is not int or not (0 <= r[0] < n and 0 <= r[1] < n) or not _num(r[2], nonnegative=True): raise ValueError("b2b endpoint")
    for r in case["p2b"]:
        if type(r[0]) is not int or type(r[1]) is not int or not (0 <= r[0] < len(case["pins"]) and 0 <= r[1] < n) or not _num(r[2], nonnegative=True): raise ValueError("p2b endpoint")
    if not _num(case["hpwl_ref"], nonnegative=True) or not _num(case["area_ref"], positive=True): raise ValueError("reference")
    out = {k: case[k] for k in CORPUS_KEYS}
    out["tp"] = [[float(v) if (c[1] or (c[0] and j >= 2)) else -1.0 for j, v in enumerate(r)] for r, c in zip(case["tp"], case["cons"])]
    return out

def _canonical(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("not canonical JSON") from exc

def fingerprint_case(case: Mapping[str, Any]) -> str:
    clean = _sanitize(case); clean.pop("instance_id")
    return hashlib.sha256(_canonical(clean)).hexdigest()

def split_for_id(instance_id: str, heldout_mod: int = 10) -> str:
    if not isinstance(instance_id, str) or not instance_id.strip(): raise ValueError("instance_id")
    if isinstance(heldout_mod, bool) or not isinstance(heldout_mod, int) or heldout_mod <= 0: raise ValueError("heldout_mod")
    return "heldout" if int(hashlib.sha256(instance_id.encode()).hexdigest(), 16) % heldout_mod == 0 else "train"

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def save_sanitized_corpus(path: str | Path, cases: Sequence[Mapping[str, Any]], *, source_root: str | Path | None | object = _DEFAULT_ROOT) -> None:
    try:
        root_ok = source_root is _DEFAULT_ROOT or (source_root is not None and Path(source_root).resolve() == _CANONICAL_ROOT)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("source_root") from exc
    if not root_ok: raise ValueError("source_root")
    if isinstance(cases, (str, bytes, Mapping)) or not isinstance(cases, Sequence): raise ValueError("cases")
    rows = [_sanitize(c) for c in cases]
    if len({fingerprint_case(r) for r in rows}) != len(rows): raise ValueError("duplicate fingerprint")
    _atomic_write(Path(path), b"".join(_canonical(r) + b"\n" for r in rows))

def load_sanitized_corpus(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    clean = [_sanitize(r) for r in rows]
    if len({fingerprint_case(r) for r in clean}) != len(clean): raise ValueError("duplicate fingerprint")
    return clean

def canonical_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def check(x: Any) -> None:
        if isinstance(x, Mapping):
            if _FORBIDDEN.intersection(x) or ("source_split" in x and x["source_split"] != "train"): raise ValueError("forbidden canonical field")
            for v in x.values(): check(v)
        elif isinstance(x, (list, tuple)):
            for v in x: check(v)
    for row in rows:
        if not isinstance(row, Mapping): raise ValueError("row")
        check(row)
    _atomic_write(Path(path), b"".join(_canonical(r) + b"\n" for r in rows))

def canonical_jsonl_sha256(path: str | Path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha256_manifest(path: str | Path, files: Sequence[str | Path]) -> dict[str, str]: return write_sha256_manifest(path, files)
def write_sha256_manifest(path: str | Path, files: Sequence[str | Path]) -> dict[str, str]:
    resolved = [Path(f).resolve() for f in files]
    if len(set(resolved)) != len(resolved): raise ValueError("duplicate manifest path")
    if any(not p.is_file() for p in resolved): raise ValueError("manifest file")
    manifest = {str(p): canonical_jsonl_sha256(p) for p in sorted(resolved, key=str)}
    _atomic_write(Path(path), _canonical(manifest) + b"\n")
    return manifest

def collate_labels(labels: Sequence[TopologyLabel], device: torch.device, dtype: torch.dtype) -> SparseTopologyBatch:
    if not isinstance(labels, (list, tuple)): raise ValueError("labels")
    edges: list[tuple] = []; contacts: list[tuple] = []
    for bi, label in enumerate(labels):
        if type(label) is not TopologyLabel or not isinstance(label.instance_id, str) or not label.instance_id.strip() or type(label.n) is not int or label.n < 0 or type(label.sample_seed) is not int: raise ValueError("label")
        if any(not _num(v, positive=True) for v in (label.teacher_cost, label.base_cost, label.record_weight)): raise ValueError("cost")
        if type(label.edges) is not tuple or type(label.contacts) is not tuple or type(label.pin_paths) is not tuple: raise ValueError("containers")
        for e in label.edges:
            if type(e) is not SparseEdge or type(e.src) is not int or type(e.dst) is not int or not (0 <= e.src < label.n and 0 <= e.dst < label.n) or e.src == e.dst or type(e.axis) is not int or e.axis not in (0, 1) or not isinstance(e.kind, str) or not e.kind.strip() or not _num(e.margin, nonnegative=True) or not _num(e.weight, positive=True): raise ValueError("edge")
            edges.append((bi, e.src, e.dst, e.axis, e.margin, e.weight * label.record_weight))
        for c in label.contacts:
            if type(c) is not ContactLabel or type(c.a) is not int or type(c.b) is not int or not (0 <= c.a < label.n and 0 <= c.b < label.n) or c.a == c.b or type(c.axis) is not int or c.axis not in (0, 1) or type(c.a_before_b) is not bool or not _num(c.perp_margin, positive=True) or not _num(c.weight, positive=True): raise ValueError("contact")
            contacts.append((bi, c.a, c.b, c.axis, int(c.a_before_b), c.perp_margin, c.weight * label.record_weight))
        for p in label.pin_paths:
            if type(p) is not tuple or not p or any(type(i) is not int or not 0 <= i < label.n for i in p) or any(a == b for a, b in zip(p, p[1:])): raise ValueError("pin path")
    def tensor(vals: list, dt: torch.dtype) -> torch.Tensor: return torch.tensor(vals, device=device, dtype=dt)
    el = lambda i, dt=torch.long: tensor([x[i] for x in edges], dt)
    cl = lambda i, dt=torch.long: tensor([x[i] for x in contacts], dt)
    return SparseTopologyBatch(el(0), el(1), el(2), el(3), el(4, dtype), el(5, dtype), cl(0), cl(1), cl(2), cl(3), cl(4), cl(5, dtype), cl(6, dtype))
