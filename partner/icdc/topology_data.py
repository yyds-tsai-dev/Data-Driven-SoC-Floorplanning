"""Leak-free data contracts for the sparse topology prior."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch

CORPUS_KEYS = {"instance_id", "n", "area", "cons", "tp", "b2b", "p2b", "pins", "hpwl_ref", "area_ref"}
_DEFAULT_ROOT = object()


@dataclass(frozen=True)
class SparseEdge:
    src: int; dst: int; axis: int; margin: float; kind: str; weight: float

@dataclass(frozen=True)
class ContactLabel:
    a: int; b: int; axis: int; a_before_b: bool; perp_margin: float; weight: float

@dataclass(frozen=True)
class TopologyLabel:
    instance_id: str; n: int; sample_seed: int; teacher_cost: float; base_cost: float
    record_weight: float; edges: Tuple[SparseEdge, ...]; contacts: Tuple[ContactLabel, ...]
    pin_paths: Tuple[Tuple[int, ...], ...]

@dataclass(frozen=True)
class SparseTopologyBatch:
    edge_batch: torch.Tensor; edge_src: torch.Tensor; edge_dst: torch.Tensor
    edge_axis: torch.Tensor; edge_margin: torch.Tensor; edge_weight: torch.Tensor
    contact_batch: torch.Tensor; contact_a: torch.Tensor; contact_b: torch.Tensor
    contact_axis: torch.Tensor; contact_order: torch.Tensor; contact_margin: torch.Tensor
    contact_weight: torch.Tensor


def _finite(x: Any) -> bool:
    if isinstance(x, (int, float)) and not isinstance(x, bool): return math.isfinite(float(x))
    if isinstance(x, (list, tuple)): return all(_finite(v) for v in x)
    return True

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _sanitize(case: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(case, Mapping): raise ValueError("case")
    if "test_id" in case: raise ValueError("test_id is forbidden")
    if any(k in case for k in ("validation", "source_split", "loader", "provenance")):
        if case.get("source_split") not in (None, "train") or any(k in case for k in ("validation", "loader", "provenance")):
            raise ValueError("validation/test provenance is forbidden")
    if not isinstance(case.get("instance_id"), str) or not case["instance_id"]: raise ValueError("instance_id")
    out = {k: case[k] for k in CORPUS_KEYS if k in case}
    if set(out) != CORPUS_KEYS: raise ValueError("corpus schema mismatch")
    n = case["n"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0: raise ValueError("n")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)) or float(x) <= 0 for x in case["area"]): raise ValueError("area")
    if any(not isinstance(x, list) or len(x) != 2 or any(type(v) is not int or v not in (0, 1) for v in x) for x in case["cons"]): raise ValueError("cons")
    for k in ("area", "cons", "tp"):
        if len(case[k]) != n: raise ValueError("shape mismatch")
    if any(not isinstance(row, list) or len(row) != 4 or any(isinstance(v, bool) or not isinstance(v, (int,float)) for v in row) for row in case["tp"]): raise ValueError("tp")
    for k, width in (("b2b", 3), ("p2b", 3), ("pins", 2)):
        if not isinstance(case[k], list) or any(not isinstance(row, list) or len(row) != width or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in row) for row in case[k]): raise ValueError(k)
    if any(not isinstance(case[k], (int,float)) or isinstance(case[k], bool) or not math.isfinite(float(case[k])) or float(case[k]) <= 0 for k in ("hpwl_ref", "area_ref")): raise ValueError("refs")
    if not _finite(out): raise ValueError("nonfinite value")
    tp = []
    for i, row in enumerate(case["tp"]):
        if len(row) != 4: raise ValueError("shape mismatch")
        c = case["cons"][i]
        fixed = bool(c[0]) if len(c) else False
        pre = bool(c[1]) if len(c) > 1 else False
        tp.append([float(row[j]) if (pre or (fixed and j >= 2)) else -1.0 for j in range(4)])
    out["tp"] = tp
    return out

def fingerprint_case(case: Mapping[str, Any]) -> str:
    clean = _sanitize(case); clean.pop("instance_id", None)
    return hashlib.sha256(_canon(clean).encode()).hexdigest()

def split_for_id(instance_id: str, heldout_mod: int = 10) -> str:
    if not isinstance(instance_id, str) or not instance_id: raise ValueError("instance_id")
    if not isinstance(heldout_mod, int) or isinstance(heldout_mod, bool) or heldout_mod <= 0: raise ValueError("heldout_mod")
    bucket = int(hashlib.sha256(instance_id.encode()).hexdigest()[:16], 16) % heldout_mod
    return "heldout" if bucket == 0 else "train"

def save_sanitized_corpus(path: str | Path, cases: Sequence[Mapping[str, Any]], *, source_root: str | Path | None | object = _DEFAULT_ROOT) -> None:
    if source_root is None: raise ValueError("source_root must be canonical")
    if source_root is not _DEFAULT_ROOT and Path(source_root).resolve() != (Path("FloorSet/floorset_lite").resolve()):
        raise ValueError("source must be FloorSet/floorset_lite")
    rows = [_sanitize(c) for c in cases]
    fps = [fingerprint_case(r) for r in rows]
    if len(set(fps)) != len(fps): raise ValueError("duplicate fingerprint")
    Path(path).write_text("".join(_canon(r) + "\n" for r in rows), encoding="utf-8")

def load_sanitized_corpus(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    clean = [_sanitize(r) for r in rows]
    if len({fingerprint_case(r) for r in clean}) != len(clean): raise ValueError("duplicate fingerprint")
    return clean

def canonical_jsonl_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows in the byte-stable sorted-key JSONL representation."""
    forbidden = {"test_id", "golden", "validation", "loader", "provenance"}
    for r in rows:
        if not isinstance(r, Mapping) or forbidden.intersection(r) or r.get("source_split") not in (None, "train"):
            raise ValueError("forbidden canonical field")
    Path(path).write_text("".join(_canon(r) + "\n" for r in rows), encoding="utf-8")

def sha256_manifest(path: str | Path, files: Sequence[str | Path]) -> dict[str, str]:
    return write_sha256_manifest(path, files)

def write_sha256_manifest(path: str | Path, files: Sequence[str | Path]) -> dict[str, str]:
    manifest = {str(Path(f)): canonical_jsonl_sha256(f) for f in sorted(files, key=str)}
    Path(path).write_text(_canon(manifest) + "\n", encoding="utf-8")
    return manifest

def collate_labels(labels: Sequence[TopologyLabel], device: torch.device, dtype: torch.dtype) -> SparseTopologyBatch:
    e=[]; c=[]
    for bi, label in enumerate(labels):
        if not isinstance(label, TopologyLabel) or not isinstance(label.instance_id, str) or not label.instance_id or type(label.n) is not int or label.n < 0:
            raise ValueError("malformed label")
        vals = (label.teacher_cost, label.base_cost, label.record_weight)
        if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(float(v)) for v in vals) or label.record_weight < 0: raise ValueError("label weights")
        for x in label.edges:
            if type(x.src) is not int or type(x.dst) is not int or not (0 <= x.src < label.n and 0 <= x.dst < label.n) or x.src == x.dst or x.axis not in (0,1) or not math.isfinite(x.margin) or x.margin < 0 or not math.isfinite(x.weight) or x.weight < 0: raise ValueError("edge")
        for x in label.contacts:
            if not (0 <= x.a < label.n and 0 <= x.b < label.n) or x.a == x.b or x.axis not in (0,1) or type(x.a_before_b) is not bool or not math.isfinite(x.perp_margin) or x.perp_margin < 0 or not math.isfinite(x.weight) or x.weight < 0: raise ValueError("contact")
        for path in label.pin_paths:
            if not path or any(type(i) is not int or not 0 <= i < label.n for i in path): raise ValueError("pin path")
        e += [(bi,x.src,x.dst,x.axis,x.margin,x.weight*label.record_weight) for x in label.edges]
        c += [(bi,x.a,x.b,x.axis,int(x.a_before_b),x.perp_margin,x.weight*label.record_weight) for x in label.contacts]
    def t(vals, dt=torch.long): return torch.tensor(vals, device=device, dtype=dt)
    return SparseTopologyBatch(t([x[0] for x in e]), t([x[1] for x in e]), t([x[2] for x in e]), t([x[3] for x in e]), t([x[4] for x in e],dtype), t([x[5] for x in e],dtype), t([x[0] for x in c]), t([x[1] for x in c]), t([x[2] for x in c]), t([x[3] for x in c]), t([x[4] for x in c]), t([x[5] for x in c],dtype), t([x[6] for x in c],dtype))
