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
import sqlite3
import stat
import shutil
import sys
import tempfile
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Optional

_REPO = Path(__file__).resolve().parents[2]
_SCORER_SHA256 = "7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1"
_SCORER_PATH = (_REPO / "scripts" / "iccad2026_evaluate.py").absolute()


def _verify_scorer_source_file(path: Path, expected_sha: str) -> str:
    try:
        if (not isinstance(path, Path)
                or any(component.is_symlink() for component in (path, *path.parents))
                or not path.is_file()):
            raise ValueError
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError
        return actual_sha
    except Exception as exc:
        raise ValueError("scorer_sha256") from exc


for _path in (_REPO / "FloorSet" / "iccad2026contest", _REPO / "FloorSet", _REPO / "scripts", _REPO / "partner"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_verify_scorer_source_file(_SCORER_PATH, _SCORER_SHA256)
import shapely
import iccad2026_evaluate as _EVALUATOR

import torch
from direct_diffusion_model import DirectDenoiser, DirectModelConfig, sample_direct_dpmpp
from diffusion_model import DiffusionSchedule
import icdc.engine as _ENGINE
import icdc.energy as _ENERGY
from icdc.energy import decode_rects
from icdc.topology_prior import ProposalConfig, generate_proposals, pin_feasible_then_exact_tfdl, extract_sparse_label, _proposal_fingerprint
from icdc.topology_data import CorpusSourceReceipt, fingerprint_case, split_for_id
from icdc.topology_data import _sanitize as _sanitize_case

from icdc.checkpoint_identity import (  # noqa: E402
    IDENTITY_SCHEMA,
    canonical_checkpoint_identity,
)


__all__ = ["teacher_main"]

_SAMPLE_DIRECT_DPM = sample_direct_dpmpp
_DECODE_RECTS = decode_rects
_GENERATE_PROPOSALS = generate_proposals

def _admit_proposal(proposal, case, _admit=pin_feasible_then_exact_tfdl):
    return _admit(proposal, case)

_ADMIT_PROPOSAL = _admit_proposal

def _verify_hard_legal(rects, case):
    n, cons, tp = _validate_case(case)
    area = torch.as_tensor(case["area"], dtype=torch.float64, device="cpu")
    result = _ENGINE.verify_hard_legal(rects.numpy(), area.numpy(), torch.as_tensor(cons, dtype=torch.int64).numpy(), tp.numpy())
    if not isinstance(result, Mapping) or not result or any(not isinstance(k, str) or not k for k in result) or any(type(v) is not bool for v in result.values()):
        raise ValueError("hard audit")
    return dict(result)

_VERIFY_HARD_LEGAL = _verify_hard_legal

def _diagnostic_energy(rects, case):
    n, cons, tp = _validate_case(case)
    if (not isinstance(rects, torch.Tensor) or rects.device.type != "cpu"
            or rects.dtype is not torch.float64 or tuple(rects.shape) != (n, 4)
            or not bool(torch.isfinite(rects).all())
            or not bool((rects[:, 2:] > 0).all())):
        raise ValueError("energy rects")
    try:
        area = torch.as_tensor(case["area"], dtype=torch.float64, device="cpu")
        cons_t = torch.as_tensor(cons, dtype=torch.float64, device="cpu")
        hpwl_ref = torch.as_tensor(case["hpwl_ref"], dtype=torch.float64, device="cpu").reshape(1)
        area_ref = torch.as_tensor(case["area_ref"], dtype=torch.float64, device="cpu").reshape(1)
    except Exception as exc:
        raise ValueError("energy references") from exc
    if (not isinstance(case.get("hpwl_ref"), numbers.Real)
            or isinstance(case.get("hpwl_ref"), bool)
            or not isinstance(case.get("area_ref"), numbers.Real)
            or isinstance(case.get("area_ref"), bool)
            or tuple(area.shape) != (n,) or not bool(torch.isfinite(area).all())
            or not bool((area > 0).all()) or tuple(cons_t.shape) != (n, 5)
            or not bool(torch.isfinite(cons_t).all())
            or tuple(hpwl_ref.shape) != (1,) or not bool(torch.isfinite(hpwl_ref).all())
            or bool((hpwl_ref < 0).any()) or tuple(area_ref.shape) != (1,)
            or not bool(torch.isfinite(area_ref).all()) or bool((area_ref <= 0).any())):
        raise ValueError("energy references")

    def relation_tensor(name: str, width: int) -> torch.Tensor:
        try:
            value = torch.as_tensor(case[name], dtype=torch.float64, device="cpu")
        except Exception as exc:
            raise ValueError(name) from exc
        if value.numel() == 0:
            return torch.empty((1, 0, width), dtype=torch.float64, device="cpu")
        if (value.ndim != 2 or tuple(value.shape[-1:]) != (width,)
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name)
        return value.reshape(1, -1, width)

    area = area.unsqueeze(0)
    cons_t = cons_t.unsqueeze(0)
    scale = torch.sqrt(area.clamp_min(0).sum()).clamp_min(1).reshape(1)
    batch = {
        "area": area,
        "cons": cons_t,
        "b2b": relation_tensor("b2b", 3),
        "p2b": relation_tensor("p2b", 3),
        "pins": relation_tensor("pins", 2),
        "hpwl_ref": hpwl_ref,
        "area_ref": area_ref,
        "scale": scale,
        "n_soft": _ENERGY.n_soft(cons_t, area).to(dtype=torch.float64, device="cpu"),
        "tau_sharp": 1e-5 * scale,
        "tau_soft": 1e-2 * scale,
    }
    result = _ENERGY.energy(rects.unsqueeze(0), batch)
    if (not isinstance(result, Mapping) or "E" not in result
            or not isinstance(result["E"], torch.Tensor)
            or result["E"].device.type != "cpu" or result["E"].dtype is not torch.float64
            or tuple(result["E"].shape) != (1,) or not bool(torch.isfinite(result["E"]).all())):
        raise ValueError("energy result")
    return float(result["E"][0].item())

_DIAGNOSTIC_ENERGY = _diagnostic_energy

@dataclass(frozen=True)
class _TeacherModelState:
    model: Any
    schedule: Any
    cfg: Any
    device: torch.device

def _select_teacher_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def _validate_teacher_device(device: torch.device) -> torch.device:
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        raise ValueError("teacher device")
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise ValueError("teacher device")
    elif (not torch.cuda.is_available() or (device.index is not None and (device.index < 0 or device.index >= torch.cuda.device_count()))):
        raise ValueError("teacher device")
    return torch.device("cuda", torch.cuda.current_device()) if device.type == "cuda" and device.index is None else device

def _materialize_teacher_model(payload: Mapping[str, Any], device: torch.device) -> _TeacherModelState:
    try:
        device = _validate_teacher_device(device)
        if not isinstance(payload, Mapping): raise ValueError("payload")
        cfg_data = payload.get("model_config")
        cfg_fields = [f.name for f in fields(DirectModelConfig)]
        if not isinstance(cfg_data, Mapping) or list(cfg_data) != cfg_fields: raise ValueError("config")
        positive_ints = {"node_feat_dim", "z_dim", "d_model", "layers", "heads", "timesteps"}
        nonnegative_ints = {"relation_feat_dim"}
        for k in positive_ints | nonnegative_ints:
            v = cfg_data[k]
            if type(v) is not int or (k in positive_ints and v <= 0) or (k in nonnegative_ints and v < 0): raise ValueError("config")
        if cfg_data["z_dim"] != 4 or cfg_data["d_model"] % cfg_data["heads"]: raise ValueError("config")
        if cfg_data["z_repr"] != "xyaspect" or type(cfg_data["self_conditioning"]) is not bool: raise ValueError("config")
        if type(cfg_data["dropout"]) not in (int, float) or not math.isfinite(float(cfg_data["dropout"])) or not 0 <= float(cfg_data["dropout"]) < 1: raise ValueError("config")
        model_weights, ema = payload.get("model"), payload.get("ema")
        if not isinstance(model_weights, Mapping) or not model_weights or not isinstance(ema, Mapping) or not ema: raise ValueError("weights")
        cfg = DirectModelConfig(
            node_feat_dim=cfg_data["node_feat_dim"],
            relation_feat_dim=cfg_data["relation_feat_dim"],
            z_dim=cfg_data["z_dim"],
            z_repr=cfg_data["z_repr"],
            d_model=cfg_data["d_model"],
            layers=cfg_data["layers"],
            heads=cfg_data["heads"],
            dropout=cfg_data["dropout"],
            timesteps=cfg_data["timesteps"],
            self_conditioning=cfg_data["self_conditioning"],
        )
        cpu_state = torch.get_rng_state()
        try:
            model = DirectDenoiser(cfg)
        finally:
            torch.set_rng_state(cpu_state)
        expected = list(model.state_dict())
        if list(model_weights) != expected or list(ema) != expected: raise ValueError("keys")
        reference = model.state_dict()
        for values in (model_weights, ema):
            for key in expected:
                value = values[key]; want = reference[key]
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.shape != want.shape or value.dtype != want.dtype: raise ValueError("tensor")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()): raise ValueError("finite")
        model.load_state_dict(ema, strict=True)
        model.to(device=device, dtype=torch.float32).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return _TeacherModelState(model, DiffusionSchedule(cfg.timesteps, device=device), cfg, device)
    except ValueError: raise
    except Exception as exc: raise ValueError("materialize") from exc

def _build_teacher_batches(case: Mapping[str, Any], device: torch.device, cfg: DirectModelConfig):
    try:
        device = _validate_teacher_device(device)
        if not isinstance(cfg, DirectModelConfig):
            raise ValueError("teacher config")
        n, normalized_cons, normalized_tp = _validate_case(case)
        area = torch.as_tensor(case["area"], dtype=torch.float32, device=device)
        tp = torch.as_tensor(normalized_tp, dtype=torch.float32, device=device)
        cons = torch.as_tensor(normalized_cons, dtype=torch.int64, device=device)
        if tuple(area.shape) != (n,) or tuple(tp.shape) != (n, 4) or tuple(cons.shape) != (n, 5):
            raise ValueError("teacher batch shape")

        def relation_tensor(name: str, width: int) -> torch.Tensor:
            try:
                value = torch.as_tensor(case[name], dtype=torch.float32, device=device)
            except Exception as exc:
                raise ValueError(name) from exc
            if value.numel() == 0:
                return torch.empty((1, 0, width), dtype=torch.float32, device=device)
            if value.ndim != 2 or value.shape[-1] != width or not bool(torch.isfinite(value).all()):
                raise ValueError(name)
            return value.reshape(1, -1, width)

        b2b = relation_tensor("b2b", 3)
        p2b = relation_tensor("p2b", 3)
        pins = relation_tensor("pins", 2)
        direct = {"area": area.unsqueeze(0), "tp": tp.unsqueeze(0), "b2b": b2b,
                  "p2b": p2b, "pins": pins, "cons": cons.unsqueeze(0)}
        direct["scale"] = torch.sqrt(direct["area"][direct["area"] > 0].sum()).clamp_min(1.0).reshape(1)
        raw_shapes = {
            "area": (1, n), "tp": (1, n, 4), "b2b": tuple(b2b.shape),
            "p2b": tuple(p2b.shape), "pins": tuple(pins.shape), "scale": (1,),
        }
        for key, shape in raw_shapes.items():
            value = direct[key]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.device != device or value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("teacher direct tensor")
        if not bool((direct["area"] > 0).all()):
            raise ValueError("teacher area")
        if (not isinstance(direct["cons"], torch.Tensor)
                or tuple(direct["cons"].shape) != (1, n, 5)
                or direct["cons"].device != device
                or direct["cons"].dtype != torch.int64):
            raise ValueError("teacher constraints")
        condition = _ENGINE.build_cond(direct, cfg)
        expected_condition_keys = {"node_feat", "adj", "mask", "scale", "rel_feat"}
        if not isinstance(condition, Mapping) or set(condition) != expected_condition_keys:
            raise ValueError("teacher condition keys")
        condition_shapes = {
            "node_feat": (1, n, cfg.node_feat_dim),
            "adj": (1, n, n),
            "mask": (1, n),
            "scale": (1,),
            "rel_feat": (1, n, n, cfg.relation_feat_dim),
        }
        for key, shape in condition_shapes.items():
            value = condition[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != device:
                raise ValueError("teacher condition tensor")
            if key == "mask":
                if value.dtype != torch.bool:
                    raise ValueError("teacher condition mask")
            elif value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError("teacher condition float")
        if not torch.equal(condition["scale"], direct["scale"]):
            raise ValueError("teacher condition scale")
        direct.update(condition)
        diagnostic = {
            "area": torch.as_tensor(case["area"], dtype=torch.float64).unsqueeze(0),
            "tp": torch.as_tensor(normalized_tp, dtype=torch.float64).unsqueeze(0),
            "b2b": torch.as_tensor(case["b2b"], dtype=torch.float64).reshape(1, -1, 3),
            "p2b": torch.as_tensor(case["p2b"], dtype=torch.float64).reshape(1, -1, 3),
            "pins": torch.as_tensor(case["pins"], dtype=torch.float64).reshape(1, -1, 2),
            "cons": cons.detach().to("cpu", dtype=torch.int64).clone().unsqueeze(0),
            "scale": direct["scale"].detach().to("cpu", dtype=torch.float64).clone(),
            "hpwl_ref": torch.tensor([case["hpwl_ref"]], dtype=torch.float64),
            "area_ref": torch.tensor([case["area_ref"]], dtype=torch.float64),
        }
        return direct, diagnostic, condition
    except Exception as exc: raise ValueError("teacher batches") from exc

def _sample_direct_once(state: _TeacherModelState, case: Mapping[str, Any], seed: int) -> torch.Tensor:
    direct, diagnostic, cond = _build_teacher_batches(case, state.device, state.cfg)
    known = _ENGINE.known_channels(direct)
    if type(known) is not tuple or len(known) != 2:
        raise ValueError("teacher known channels")
    z_known, known_mask = known
    n = direct["area"].shape[1]
    if (not isinstance(z_known, torch.Tensor) or tuple(z_known.shape) != (1, n, 4)
            or z_known.device != state.device or z_known.dtype != torch.float32
            or not bool(torch.isfinite(z_known).all())):
        raise ValueError("teacher known z")
    if (not isinstance(known_mask, torch.Tensor) or tuple(known_mask.shape) != (1, n, 4)
            or known_mask.device != state.device or known_mask.dtype != torch.bool):
        raise ValueError("teacher known mask")
    generator = torch.Generator(device=state.device); generator.manual_seed(seed)
    raw = _SAMPLE_DIRECT_DPM(state.model, cond, state.schedule, steps=2, generator=generator, z_known=z_known, known_mask=known_mask)
    if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (1, n, 4) or raw.device != state.device or raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()): raise ValueError("sample")
    decoded = _DECODE_RECTS(raw.detach().to("cpu", dtype=torch.float64), diagnostic["area"], diagnostic["cons"], diagnostic["tp"], diagnostic["scale"])
    if not isinstance(decoded, torch.Tensor) or tuple(decoded.shape) != (1, n, 4) or decoded.device.type != "cpu" or decoded.dtype != torch.float64 or not bool(torch.isfinite(decoded).all()) or not bool((decoded[...,2:] > 0).all()): raise ValueError("decoded")
    return decoded[0]

@dataclass(frozen=True)
class _CaseInput:
    case: Mapping[str, Any]
    receipt: CorpusSourceReceipt
    partition: str
    sample_seed: int

@dataclass(frozen=True)
class _VerifiedShardSummary:
    worker: int
    layout: int
    relative_path: str
    file_sha256: str
    source_row_count: int

def _verified_shard_summary(raw: bytes, source: Any, worker: int, layout: int) -> _VerifiedShardSummary:
    if type(worker) is not int or worker < 0 or type(layout) is not int or layout < 0:
        raise ValueError("shard identity")
    if not isinstance(raw, bytes):
        raise ValueError("shard bytes")
    count, _ = _validate_source_shard(source)
    return _VerifiedShardSummary(worker, layout, f"worker_{worker}/layouts_{layout}.th",
                                 hashlib.sha256(raw).hexdigest(), count)

def _validate_spooled_case_row(record: Any, shard_summary: Any) -> tuple[Mapping[str, Any], CorpusSourceReceipt]:
    if not isinstance(record, Mapping) or not isinstance(shard_summary, Mapping) and not dataclass_is_instance(shard_summary):
        raise ValueError("spool record")
    if isinstance(shard_summary, Mapping):
        expected = {k: shard_summary[k] for k in ("worker", "layout", "relative_path", "file_sha256", "source_row_count")}
    else:
        expected = {"worker": shard_summary.worker, "layout": shard_summary.layout,
                    "relative_path": shard_summary.relative_path,
                    "file_sha256": shard_summary.file_sha256,
                    "source_row_count": shard_summary.source_row_count}
    keys = {"worker", "layout", "relative_path", "file_sha256", "source_row_count", "source_row_index", "instance_id", "case_json", "case", "fingerprint", "receipt"}
    if set(record) != keys:
        raise ValueError("spool record keys")
    if any(record[k] != expected[k] for k in ("worker", "layout", "relative_path", "file_sha256", "source_row_count")):
        raise ValueError("spool binding")
    digest = record["file_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("spool digest")
    index = record["source_row_index"]
    if type(index) is not int or not 0 <= index < expected["source_row_count"]:
        raise ValueError("spool index")
    iid = f'{expected["relative_path"]}#{index}'
    if record["instance_id"] != iid:
        raise ValueError("spool instance")
    text = record["case_json"]
    if not isinstance(text, str) or any(ord(c) > 127 for c in text) or text != text.strip() or "\n" in text or "\r" in text:
        raise ValueError("spool json")
    try:
        decoded = json.loads(text)
    except Exception as exc:
        raise ValueError("spool json") from exc
    if not isinstance(decoded, Mapping) or decoded != record["case"]:
        raise ValueError("spool case")
    case = _sanitize_case(decoded)
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    if text != canonical or case.get("instance_id") != iid:
        raise ValueError("spool canonical case")
    fp = fingerprint_case(case)
    if record["fingerprint"] != fp:
        raise ValueError("spool fingerprint")
    receipt = record["receipt"]
    if type(receipt) is not CorpusSourceReceipt or (receipt.relative_path, receipt.file_sha256, receipt.layout_index, receipt.fingerprint) != (expected["relative_path"], digest, index, fp):
        raise ValueError("spool receipt")
    return case, receipt

def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)

@dataclass(frozen=True)
class _CaseOutcome:
    label_row: Mapping[str, Any]
    proposal_rows: Sequence[Mapping[str, Any]]
    rejection_rows: Sequence[Mapping[str, Any]]
    base_cost: float
    teacher_cost: float
    legal: bool
    covered: bool
    case_status: str

@dataclass(frozen=True)
class _TeacherRuntime:
    preflight: Callable[[TeacherTrustPolicy, Path], Mapping[str, Any]]
    process_case: Callable[[_CaseInput], _CaseOutcome]
    authorizing: bool

def _runtime_hooks() -> _TeacherRuntime:
    trusted: Optional[Mapping[str, Any]] = None
    cached: Optional[_TeacherModelState] = None

    def preflight(policy: TeacherTrustPolicy, checkpoint: Path) -> Mapping[str, Any]:
        nonlocal trusted, cached
        trusted = None
        cached = None
        verified_scorer = _verify_frozen_scorer_contract(policy)
        payload, identity = _load_verified_checkpoint_bytes(checkpoint, policy)
        model_identity = {
            key: value for key, value in identity.items() if key != "checkpoint_sha256"
        }
        value = {"trust_ok": True, "input_ok": True, "scorer_ok": True,
                 "checkpoint_sha256": identity["checkpoint_sha256"],
                 "model_identity": model_identity,
                 "scorer_sha256": verified_scorer["scorer_sha256"],
                 "scorer_contract": verified_scorer["scorer_contract"],
                 "shapely_version": verified_scorer["shapely_version"]}
        trusted = {"payload": payload, "scorer": _EVALUATOR, "trust": value}
        return value

    def process_case(_case: _CaseInput) -> _CaseOutcome:
        nonlocal cached
        if trusted is None:
            raise RuntimeError("teacher process before trusted preflight")
        if _case is None:
            raise ValueError("case input")
        if not isinstance(_case, _CaseInput):
            raise ValueError("case input")
        if cached is None:
            cached = _materialize_teacher_model(trusted["payload"], _select_teacher_device())
        raw = _sample_direct_once(cached, _case.case, _case.sample_seed)
        lifecycle = _run_candidate_lifecycle(raw, _case.case, scorer=_EVALUATOR, cfg=ProposalConfig())
        return _outcome_from_lifecycle(_case, raw, lifecycle)

    return _TeacherRuntime(preflight, process_case, True)


def _verify_frozen_scorer_contract(policy: TeacherTrustPolicy) -> Mapping[str, str]:
    try:
        actual = _verify_scorer_source_file(_SCORER_PATH, policy.expected_scorer_sha256)
        origin = Path(_EVALUATOR.__file__).absolute()
        if origin != _SCORER_PATH or origin.is_symlink():
            raise ValueError
    except Exception as exc:
        raise ValueError("scorer_sha256") from exc
    if actual != _SCORER_SHA256:
        raise ValueError("scorer_sha256")
    if policy.scorer_contract != "iccad2026_evaluate_cost_no_runtime_v1":
        raise ValueError("scorer_contract")
    try:
        shapely_available = _EVALUATOR.SHAPELY_AVAILABLE
    except Exception as exc:
        raise ValueError("shapely_available") from exc
    if shapely_available is not True:
        raise ValueError("shapely_available")
    try:
        shapely_version = shapely.__version__
    except Exception as exc:
        raise ValueError("shapely_version") from exc
    if shapely_version != policy.shapely_version:
        raise ValueError("shapely_version")
    expected = ["solution", "baseline_metrics", "target_constraints", "b2b_connectivity", "p2b_connectivity", "pins_pos", "target_areas", "target_positions", "median_runtime"]
    try:
        params = list(inspect.signature(_EVALUATOR.evaluate_solution).parameters.values())
        if (len(params) != 9 or [p.name for p in params] != expected
                or any(p.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
                or any(p.default is not inspect.Parameter.empty for p in params[:7])
                or params[7].default is not None
                or type(params[8].default) is not float or params[8].default != 1.0):
            raise ValueError
    except Exception as exc:
        raise ValueError("evaluate_solution_signature") from exc
    try:
        metric_fields = fields(_EVALUATOR.SolutionMetrics)
    except Exception as exc:
        raise ValueError("cost_no_runtime") from exc
    if not any(f.name == "cost_no_runtime" for f in metric_fields):
        raise ValueError("cost_no_runtime")
    counts = [7, 19, 43]
    costs = [1.25, 2.5, 4.75]
    weights = [math.exp(n / 12) for n in counts]
    expected_score = sum(c * w for c, w in zip(costs, weights)) / sum(weights)
    try:
        actual_score = _EVALUATOR.compute_total_score(costs, counts)
        if not math.isfinite(actual_score) or not math.isclose(actual_score, expected_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError
    except Exception as exc:
        raise ValueError("compute_total_score_weighting") from exc
    return {"scorer_sha256": actual, "scorer_contract": policy.scorer_contract, "shapely_version": policy.shapely_version}

_CHECKPOINT_SHA256 = "508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659"
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
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.device.type != "cpu" or value.dtype == torch.bool:
            return False
        try:
            value = value.item()
        except RuntimeError:
            return False
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

@dataclass(frozen=True)
class _CandidateRecord:
    ordinal: int; name: str; original: torch.Tensor; legal: Optional[torch.Tensor]
    drift: Mapping[str, float]; hard: Mapping[str, bool]; official_cost: Optional[float]
    diagnostic_energy: Optional[float]; energy_status: str; rejection_reason: Optional[str]

@dataclass(frozen=True)
class _CandidateLifecycle:
    candidates: tuple[_CandidateRecord, ...]; winner_ordinal: Optional[int]
    base_cost: Optional[float]; teacher_cost: Optional[float]

def _run_candidate_lifecycle(raw_rects, case, *, scorer, cfg=ProposalConfig()):
    if not hasattr(scorer, "evaluate_solution") or not callable(scorer.evaluate_solution):
        raise TypeError("scorer")
    if type(cfg) is not ProposalConfig:
        raise TypeError("cfg")
    n, normalized_cons, normalized_tp = _validate_case(case)
    if not isinstance(raw_rects, torch.Tensor) or raw_rects.device.type != "cpu" or raw_rects.dtype is not torch.float64 or raw_rects.shape != (n, 4) or not bool(torch.isfinite(raw_rects).all()) or not bool((raw_rects[:, 2:] > 0).all()):
        raise ValueError("raw rects")
    try:
        area = torch.as_tensor(case["area"], dtype=torch.float64, device="cpu")
        cons = torch.as_tensor(normalized_cons, dtype=torch.int64, device="cpu")
    except Exception as exc:
        raise ValueError("adapter") from exc
    if (not isinstance(case.get("hpwl_ref"), numbers.Real)
            or isinstance(case.get("hpwl_ref"), bool)
            or not isinstance(case.get("area_ref"), numbers.Real)
            or isinstance(case.get("area_ref"), bool)
            or area.shape != (n,) or not bool(torch.isfinite(area).all())
            or not bool((area > 0).all())
            or not math.isfinite(float(case["hpwl_ref"]))
            or float(case["hpwl_ref"]) < 0
            or not math.isfinite(float(case["area_ref"]))
            or float(case["area_ref"]) <= 0):
        raise ValueError("adapter")

    def relation_tensor(name: str, width: int) -> torch.Tensor:
        try:
            value = torch.as_tensor(case[name], dtype=torch.float64, device="cpu")
        except Exception as exc:
            raise ValueError(name) from exc
        if value.numel() == 0:
            return torch.empty((0, width), dtype=torch.float64, device="cpu")
        if (value.ndim != 2 or tuple(value.shape[-1:]) != (width,)
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name)
        return value

    tensors = {
        "b2b": relation_tensor("b2b", 3),
        "p2b": relation_tensor("p2b", 3),
        "pins": relation_tensor("pins", 2),
    }

    generated = []
    phase = {"base": 0, "axis": 1, "pin": 2, "contact": 3}
    counts = {"axis": 0, "pin": 0, "contact": 0}
    seen_names: set[str] = set()
    last_phase = -1
    try:
        proposals = iter(_GENERATE_PROPOSALS(raw_rects, case, cfg))
        for ordinal in range(cfg.total_cap + 1):
            try:
                item = next(proposals)
            except StopIteration:
                break
            if ordinal >= cfg.total_cap:
                raise ValueError("proposal total cap")
            if (not isinstance(item, (tuple, list)) or len(item) != 2
                    or not isinstance(item[0], str)):
                raise ValueError("proposal")
            name, candidate = item
            if not name or "\0" in name or name in seen_names:
                raise ValueError("proposal name")
            tokens = name.split(":")
            kind = tokens[0]
            if kind not in phase:
                raise ValueError("proposal name")
            if kind == "base":
                if name != "base" or ordinal != 0:
                    raise ValueError("base proposal")
                values = []
            else:
                expected_len = 6 if kind == "contact" else 5
                if len(tokens) != expected_len:
                    raise ValueError("proposal name")
                try:
                    values = [int(token) for token in tokens[1:]]
                except Exception as exc:
                    raise ValueError("proposal name") from exc
                if any(token == "" or str(value) != token for token, value in zip(tokens[1:], values)):
                    raise ValueError("proposal name")
                if kind == "axis":
                    first, second, axis, order = values
                    if (not first < second or not 0 <= first < n or not 0 <= second < n
                            or axis not in (0, 1) or order not in (0, 1)):
                        raise ValueError("axis proposal")
                elif kind == "pin":
                    target, peer, axis, order = values
                    authorized = (0 <= target < n and 0 <= peer < n and target != peer
                                  and normalized_cons[target][1] != 0
                                  and bool((normalized_tp[target, :2] >= 0).all()))
                    if not authorized or axis not in (0, 1) or order not in (0, 1):
                        raise ValueError("pin proposal")
                else:
                    gid, first, second, axis, order = values
                    if (gid <= 0 or not first < second or not 0 <= first < n
                            or not 0 <= second < n or axis not in (0, 1)
                            or order not in (0, 1)
                            or normalized_cons[first][3] != gid
                            or normalized_cons[second][3] != gid):
                        raise ValueError("contact proposal")
                counts[kind] += 1
                cap = {"axis": cfg.axis_exchange_cap, "pin": cfg.pin_repair_cap,
                       "contact": cfg.group_contact_cap}[kind]
                if counts[kind] > cap:
                    raise ValueError("proposal kind cap")
            if phase[kind] < last_phase:
                raise ValueError("proposal phase")
            last_phase = phase[kind]
            if (not isinstance(candidate, torch.Tensor) or candidate.device.type != "cpu"
                    or candidate.dtype is not torch.float64 or tuple(candidate.shape) != (n, 4)
                    or not bool(torch.isfinite(candidate).all())
                    or not bool((candidate[:, 2:] > 0).all())):
                raise ValueError("proposal rects")
            seen_names.add(name)
            generated.append((name, candidate.clone()))
    except StopIteration:
        pass
    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise RuntimeError("proposal generation") from exc
    if not generated or generated[0][0] != "base":
        raise ValueError("base proposal")
    names = set(); records = []
    for ordinal, item in enumerate(generated):
        if not isinstance(item, (tuple, list)) or len(item) != 2: raise ValueError("proposal")
        name, candidate = item
        if not isinstance(name, str) or not name or name in names or "\0" in name: raise ValueError("proposal name")
        tokens = name.split(":"); kind = tokens[0]
        if kind not in phase or (kind == "base" and (name != "base" or ordinal != 0)) or (kind != "base" and len(tokens) != (6 if kind == "contact" else 5)): raise ValueError("proposal name")
        if kind != "base":
            try: vals = [int(x) for x in tokens[1:]]
            except Exception as exc: raise ValueError("proposal name") from exc
            if any(x == "" or str(v) != x for x, v in zip(tokens[1:], vals)): raise ValueError("proposal name")
        if records and phase[kind] < phase[records[-1].name.split(":")[0]]: raise ValueError("proposal phase")
        if not isinstance(candidate, torch.Tensor) or candidate.device.type != "cpu" or candidate.dtype is not torch.float64 or candidate.shape != (n, 4) or not bool(torch.isfinite(candidate).all()) or not bool((candidate[:, 2:] > 0).all()): raise ValueError("proposal rects")
        names.add(name); original = candidate.clone(); legal = None; drift = {}; hard = {}; cost = energy_value = None; energy_status = "not_reached"; reason = None
        stage = "admission"
        try:
            admitted = _ADMIT_PROPOSAL(candidate.clone(), case)
            if not isinstance(admitted, (tuple, list)) or len(admitted) != 2: raise ValueError
            legal, drift_tensor = admitted
            if not isinstance(legal, torch.Tensor) or legal.device.type != "cpu" or legal.dtype is not torch.float64 or legal.shape != (n, 4) or not bool(torch.isfinite(legal).all()) or not bool((legal[:, 2:] > 0).all()) or not isinstance(drift_tensor, torch.Tensor) or drift_tensor.device.type != "cpu" or drift_tensor.dtype is not torch.float64 or drift_tensor.shape != (n, 2) or not bool(torch.isfinite(drift_tensor).all()) or not bool(torch.count_nonzero(drift_tensor) == 0): raise ValueError
            legal = legal.clone(); drift = {"max_abs": 0.0}; stage = "hard"; hard = _VERIFY_HARD_LEGAL(legal.clone(), case)
            if not hard or any(type(v) is not bool or not v for v in hard.values()): raise ValueError
            stage = "intent"
            if _proposal_intent_holds(name, legal.clone(), case) is not True: raise ValueError
            stage = "official"
            result = scorer.evaluate_solution({"positions": [[float(v) for v in row] for row in legal.tolist()], "runtime": 1.0}, {"hpwl_baseline": float(case["hpwl_ref"]), "area_baseline": float(case["area_ref"])}, cons, tensors["b2b"], tensors["p2b"], tensors["pins"], area, normalized_tp.tolist(), median_runtime=1.0)
            feasible = result.is_feasible; cost_value = result.cost_no_runtime
            if type(feasible) is not bool: raise ValueError("feasible")
            if not feasible: reason = "official_infeasible"
            elif not isinstance(cost_value, numbers.Real) or isinstance(cost_value, bool) or not math.isfinite(float(cost_value)) or float(cost_value) <= 0: reason = "official_invalid_cost"
            else: cost = float(cost_value)
        except Exception:
            if reason is None: reason = {"admission": "admission_failed", "hard": "hard_audit_failed", "intent": "intent_not_survived", "official": "official_evaluator_error"}[stage]
        if legal is not None and hard and cost is not None:
            try:
                value = _DIAGNOSTIC_ENERGY(legal.clone(), case)
                if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value)): energy_value = float(value); energy_status = "recorded"
                else: energy_status = "unavailable"
            except Exception: energy_status = "unavailable"
            reason = None
        records.append(_CandidateRecord(ordinal, name, original, legal, drift, hard, cost, energy_value, energy_status, reason))
    base = next((r for r in records if r.name == "base" and r.official_cost is not None and r.rejection_reason is None), None)
    if base is None:
        records = [replace(r, rejection_reason=(r.rejection_reason or "base_unavailable")) for r in records]
        return _CandidateLifecycle(tuple(records), None, None, None)
    winner = min((r for r in records if r.official_cost is not None and r.rejection_reason is None), key=lambda r: (r.official_cost, r.ordinal, r.name))
    records = [replace(r, rejection_reason=(None if r is winner else ("not_selected" if r.rejection_reason is None else r.rejection_reason))) for r in records]
    return _CandidateLifecycle(tuple(records), winner.ordinal, base.official_cost, winner.official_cost)


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

@dataclass(frozen=True)
class _StagingLease:
    path: Path
    st_dev: int
    st_ino: int


def _new_staging_lease(path: Path) -> _StagingLease:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("invalid staging")
    path = Path(path)
    return _StagingLease(path, info.st_dev, info.st_ino)


def _path_matches_lease(path: Path, lease: _StagingLease) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_dev == lease.st_dev and info.st_ino == lease.st_ino


def _cleanup_owned_staging(lease: _StagingLease) -> bool:
    _after_ownership_validation("staging_root", lease.path)
    root_fd = None
    try:
        fd = os.open(lease.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW); root_fd = fd
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != (lease.st_dev, lease.st_ino):
            os.close(fd); return False
        _remove_owned_staging_contents_fd(fd)
        os.close(fd); root_fd = None
        if not _path_matches_lease(lease.path, lease):
            return False
        _after_ownership_validation("staging_root", lease.path)
        if not _path_matches_lease(lease.path, lease): return False
        lease.path.rmdir()
        return True
    except OSError:
        return False
    finally:
        if root_fd is not None:
            try: os.close(root_fd)
            except OSError: pass

def _after_ownership_validation(operation: str, target: Path) -> None:
    return None

def _remove_owned_staging_contents_fd(owned_fd: int) -> None:
    for name in os.listdir(owned_fd):
        st = os.stat(name, dir_fd=owned_fd, follow_symlinks=False)
        if stat.S_ISDIR(st.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=owned_fd)
            try:
                now = os.fstat(child)
                if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino):
                    continue
                _remove_owned_staging_contents_fd(child)
            finally:
                os.close(child)
            _after_ownership_validation("staging_child", Path(os.readlink(f"/proc/self/fd/{owned_fd}")) / name)
            try: os.rmdir(name, dir_fd=owned_fd)
            except FileNotFoundError: pass
        else:
            target = Path(os.readlink(f"/proc/self/fd/{owned_fd}")) / name
            _after_ownership_validation("staging_leaf", target)
            try:
                now = os.stat(name, dir_fd=owned_fd, follow_symlinks=False)
                if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino): continue
                os.unlink(name, dir_fd=owned_fd)
            except FileNotFoundError: pass


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, "renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    flags = 1  # RENAME_NOREPLACE
    result = renameat2(-100, os.fsencode(str(source)), -100,
                       os.fsencode(str(destination)), flags)
    if result == 0:
        return
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))


def _publish_staging(lease: _StagingLease, destination: Path) -> None:
    if not _path_matches_lease(lease.path, lease):
        raise ValueError("invalid staging")
    try:
        _renameat2_noreplace(lease.path, destination)
        return
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise ValueError("existing output directory") from exc
        if exc.errno != errno.EINTR:
            raise
        source_present = _path_matches_lease(lease.path, lease)
        try:
            destination_info = os.lstat(destination)
        except OSError:
            destination_info = None
        destination_present = destination_info is not None
        if (not source_present and destination_present
                and stat.S_ISDIR(destination_info.st_mode)
                and destination_info.st_dev == lease.st_dev
                and destination_info.st_ino == lease.st_ino):
            return
        if source_present and not destination_present:
            raise
        raise _PublishAmbiguousError("ambiguous publish outcome") from exc


class _PublishAmbiguousError(RuntimeError):
    pass


def _new_staging(destination: Path) -> Path:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=parent))
    return path


def _iter_approved_shards(root: Path) -> list[tuple[int, int, Path]]:
    def canonical_decimal(body: str) -> bool:
        if not body or any(char < "0" or char > "9" for char in body):
            return False
        try:
            return str(int(body)) == body
        except ValueError:
            return False

    found: list[tuple[int, int, Path]] = []
    for worker in root.iterdir():
        wbody = worker.name[7:] if worker.name.startswith("worker_") else ""
        if canonical_decimal(wbody) and worker.is_symlink(): raise ValueError("numeric worker symlink")
        if not worker.is_dir() or not canonical_decimal(wbody):
            continue
        wid = int(wbody)
        for shard in worker.iterdir():
            sbody = shard.name[8:-3] if shard.name.startswith("layouts_") and shard.name.endswith(".th") else ""
            if canonical_decimal(sbody) and shard.is_symlink(): raise ValueError("numeric shard symlink")
            if not shard.is_file() or not canonical_decimal(sbody):
                continue
            lid = int(sbody)
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
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.requires_grad or t.layout != torch.strided or t.dtype == torch.bool or not t.is_floating_point() for t in source):
        raise ValueError("source tensors")
    inp, b2b, p2b, pins, tree, fp, metrics = source
    if inp.ndim != 3 or inp.shape[2] != 6 or b2b.ndim != 3 or b2b.shape[2] != 3 or p2b.ndim != 3 or p2b.shape[2] != 3 or pins.ndim != 3 or pins.shape[2] != 2 or tree.ndim != 3 or tree.shape[2] != 3 or fp.ndim != 3 or fp.shape[2] != 4 or metrics.ndim != 2 or metrics.shape[1] != 8:
        raise ValueError("source shapes")
    b, n = inp.shape[:2]
    if b < 1 or n < 1 or any(t.shape[0] != b for t in source[1:]):
        raise ValueError("source batch")
    if tree.shape[1] != n - 1: raise ValueError("source tree")
    for tensor in source:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("source tensors")
    for row in inp:
        seen_pad = False
        for item in row:
            pad = float(item[0]) == -1.0
            if pad: seen_pad = True
            elif seen_pad or float(item[0]) <= 0: raise ValueError("area padding")
            elif any(not _is_integral(x) for x in item[1:]): raise ValueError("constraint")
    for tensor, width, kind in ((b2b, 3, "b2b"), (p2b, 3, "p2b"), (pins, 2, "pin")):
        for batch in tensor:
            padded = False
            for row in batch:
                is_pad = all(float(v) == -1.0 for v in row)
                if is_pad: padded = True
                elif padded: raise ValueError("noncontiguous padding")
        for row in tensor.reshape(-1, width):
            pads = [float(x) == -1.0 for x in row]
            if any(pads) and not all(pads): raise ValueError("partial padding")
            if not any(pads):
                if width == 3:
                    if not all(_is_integral(x) for x in row[:2]): raise ValueError("edge endpoint")
                    if kind == "b2b" and (float(row[0]) < 0 or float(row[0]) >= n or float(row[1]) < 0 or float(row[1]) >= n): raise ValueError("b2b endpoint")
                    if kind == "p2b" and (float(row[1]) < 0 or float(row[1]) >= n or float(row[0]) < 0): raise ValueError("p2b endpoint")
                    if float(row[2]) < 0: raise ValueError(f"{kind} weight")
                elif any(not math.isfinite(float(x)) for x in row):
                    raise ValueError("pin value")
    return int(b), int(n)

def _trim_rows(tensor: torch.Tensor, width: int, index: int) -> list[list[float]]:
    out = []
    padded = False
    for row in tensor[index].tolist():
        is_pad = all(float(v) == -1.0 for v in row)
        if is_pad:
            padded = True
        elif padded:
            raise ValueError("noncontiguous padding")
        else:
            out.append([float(v) for v in row])
    return out

def _dump_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode() + b"\n")


def _write_json_fsync(path: Path, value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode() + b"\n"
    with open(path, "wb") as fd:
        fd.write(encoded)
        fd.flush()
        os.fsync(fd.fileno())


_TASK4_JSONL = ("train_corpus.jsonl", "heldout_corpus.jsonl", "train_labels.jsonl",
                "heldout_labels.jsonl", "proposals.jsonl", "rejections.jsonl")
_POP_OWNERS: dict[int, set[tuple[int, int]]] = {}
_POP_SIDECARS: dict[int, dict[str, tuple[int, int]]] = {}


class _JsonlWriter:
    def __init__(self, staging: Path) -> None:
        self._files: dict[str, Any] = {}
        try:
            for name in _TASK4_JSONL:
                self._files[name] = open(staging / name, "wb")
        except BaseException:
            for handle in self._files.values():
                try:
                    handle.close()
                except BaseException:
                    pass
            raise

    def write(self, name: str, value: Any) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
        handle = self._files[name]
        handle.write(encoded)
        handle.flush()

    def close(self) -> None:
        primary: Optional[BaseException] = None
        for name in _TASK4_JSONL:
            handle = self._files.get(name)
            if handle is None:
                continue
            try:
                if not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException as exc:
                if primary is None:
                    primary = exc
            finally:
                try:
                    if not handle.closed:
                        handle.close()
                except BaseException as exc:
                    if primary is None:
                        primary = exc
        if primary is not None:
            raise primary

    def abort(self) -> None:
        for handle in self._files.values():
            try:
                if not handle.closed:
                    handle.close()
            except BaseException:
                pass


class _PopulationAccumulator:
    def __init__(self) -> None:
        self.denominator = self.base_total = self.teacher_total = 0.0
        self.count = 0
        self._finished: Optional[dict[str, float | str]] = None
        temp = tempfile.NamedTemporaryFile(prefix="floorset-population-", suffix=".sqlite", delete=False)
        self._db_path = Path(temp.name)
        temp.close()
        _POP_OWNERS[id(self)] = {(os.stat(self._db_path).st_dev, os.stat(self._db_path).st_ino)}
        _POP_SIDECARS[id(self)] = {}
        self._db: Optional[sqlite3.Connection] = None
        try:
            self._db = sqlite3.connect(str(self._db_path))
            self._db.execute("PRAGMA cache_size=-64")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute("PRAGMA mmap_size=0")
            self._db.execute(
                "CREATE TABLE population ("
                "relative_path TEXT NOT NULL, layout_index INTEGER NOT NULL, "
                "instance_id TEXT NOT NULL, n INTEGER NOT NULL, base_cost REAL, "
                "teacher_cost REAL, weight REAL, "
                "UNIQUE(relative_path, layout_index), UNIQUE(instance_id))"
            )
            self._db.commit()
            self._capture_sidecars()
        except BaseException:
            self.abort()
            raise

    def add(self, row: Mapping[str, Any]) -> None:
        self.register({k: row[k] for k in ("relative_path", "layout_index", "instance_id", "n")})
        self.record_winner(row["instance_id"], row["base_cost"], row["teacher_cost"])

    def _capture_sidecars(self) -> None:
        for suffix in ("-journal", "-wal", "-shm"):
            path = Path(f"{self._db_path}{suffix}")
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            _POP_SIDECARS[id(self)][suffix] = (st.st_dev, st.st_ino)

    def register(self, row: Mapping[str, Any]) -> None:
        if self._finished is not None: raise RuntimeError("population already finished")
        required = {"relative_path","layout_index","instance_id","n"}
        if not isinstance(row, Mapping) or set(row) != required: raise ValueError("population row keys")
        iid = row["instance_id"]
        if not isinstance(iid,str) or not iid or "\0" in iid: raise ValueError("instance_id")
        rel = _canonical_relative_path(row["relative_path"])
        if type(row["layout_index"]) is not int or row["layout_index"] < 0 or type(row["n"]) is not int or row["n"] < 0: raise ValueError("population identity")
        assert self._db is not None
        try: weight = math.exp(row["n"] / 12)
        except (OverflowError, ValueError): weight = None
        try:
            self._db.execute("INSERT INTO population(relative_path,layout_index,instance_id,n,base_cost,teacher_cost,weight) VALUES (?,?,?,?,NULL,NULL,?)", (rel,row["layout_index"],iid,row["n"],weight))
        except sqlite3.IntegrityError as exc: raise ValueError("duplicate population identity") from exc
        self.count += 1

    def record_winner(self, instance_id: str, base_cost: float, teacher_cost: float) -> None:
        base_cost = _finite_number(base_cost,"base_cost"); teacher_cost = _finite_number(teacher_cost,"teacher_cost")
        if base_cost <= 0 or teacher_cost <= 0: raise ValueError("cost")
        assert self._db is not None
        row = self._db.execute("SELECT n FROM population WHERE instance_id=? AND base_cost IS NULL", (instance_id,)).fetchone()
        if row is None: raise ValueError("unknown or duplicate winner")
        try: weight = math.exp(int(row[0]) / 12)
        except (OverflowError, ValueError) as exc: raise ValueError("weight") from exc
        if not math.isfinite(weight) or not math.isfinite(weight*base_cost) or not math.isfinite(weight*teacher_cost): raise ValueError("weighted cost")
        cur = self._db.execute("UPDATE population SET weight=?,base_cost=?,teacher_cost=? WHERE instance_id=? AND base_cost IS NULL", (weight,base_cost,teacher_cost,instance_id)); self._db.commit()
        if cur.rowcount != 1: raise ValueError("unknown or duplicate winner")
        self._capture_sidecars()

    def _legacy_add(self, row: Mapping[str, Any]) -> None:
        if self._finished is not None:
            raise RuntimeError("population already finished")
        required = {"relative_path", "layout_index", "instance_id", "n", "base_cost", "teacher_cost"}
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("population row keys")
        relative_path = _canonical_relative_path(row["relative_path"])
        layout_index = row["layout_index"]
        if type(layout_index) is not int or layout_index < 0:
            raise ValueError("layout_index")
        instance_id = row["instance_id"]
        if not isinstance(instance_id, str) or not instance_id or "\0" in instance_id:
            raise ValueError("instance_id")
        n = row["n"]
        if type(n) is not int or n < 0:
            raise ValueError("n")
        base_cost = _finite_number(row["base_cost"], "base_cost")
        teacher_cost = _finite_number(row["teacher_cost"], "teacher_cost")
        try:
            weight = math.exp(n / 12)
        except (OverflowError, ValueError) as exc:
            raise ValueError("weight") from exc
        if not math.isfinite(weight):
            raise ValueError("weight")
        base_product = weight * base_cost
        teacher_product = weight * teacher_cost
        if not math.isfinite(base_product) or not math.isfinite(teacher_product):
            raise ValueError("weighted cost")
        assert self._db is not None
        try:
            self._db.execute(
                "INSERT INTO population VALUES (?, ?, ?, ?, ?, ?, ?)",
                (relative_path, str(layout_index), instance_id, str(n), base_cost,
                 teacher_cost, weight),
            )
            self._db.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate population identity") from exc
        self.count += 1

    def finish(self) -> dict[str, float | str]:
        if self._finished is not None:
            return dict(self._finished)
        if self._db is None:
            raise RuntimeError("population spool closed")
        try:
            hasher = hashlib.sha256(); hasher.update(b"[")
            total = scored = 0; denominator = sden = base_total = teacher_total = 0.0
            first = True
            for r in self._db.execute("SELECT relative_path,layout_index,instance_id,n,base_cost,teacher_cost,weight FROM population ORDER BY relative_path,layout_index"):
                weight = r[6] if r[6] is not None else math.exp(int(r[3]) / 12)
                if not math.isfinite(weight): raise ValueError("weight")
                payload = json.dumps({"relative_path":r[0],"layout_index":r[1],"instance_id":r[2],"n":r[3],"weight":weight}, sort_keys=True, separators=(",",":"), ensure_ascii=True, allow_nan=False).encode()
                if not first: hasher.update(b",")
                hasher.update(payload); first = False
                total += 1; denominator += weight
                if not all(math.isfinite(v) for v in (denominator, sden, base_total, teacher_total)):
                    raise ValueError("nonfinite aggregate")
                if r[4] is not None and r[5] is not None:
                    scored += 1; sden += weight; base_total += weight * r[4]; teacher_total += weight * r[5]
                if not all(math.isfinite(v) for v in (denominator, sden, base_total, teacher_total)):
                    raise ValueError("nonfinite aggregate")
            hasher.update(b"]")
            if total > 0 and (not math.isfinite(denominator) or denominator <= 0):
                raise ValueError("invalid denominator")
            complete = total > 0 and scored == total
            if complete and not all(math.isfinite(v) for v in (base_total/denominator, teacher_total/denominator, base_total/denominator-teacher_total/denominator)):
                raise ValueError("nonfinite aggregate")
            result = {"eligible_count":total,"scored_winner_count":scored,"denominator":denominator,"scored_denominator":sden,"B_H":base_total/denominator if complete else None,"T_H":teacher_total/denominator if complete else None,"Delta_H":(base_total/denominator-teacher_total/denominator) if complete else None,"population_sha256":hasher.hexdigest()}
            self._finished = result
            self.denominator = denominator; self.base_total = base_total; self.teacher_total = teacher_total
            return dict(result)
        finally:
            self._close_spool()

    def _close_spool(self) -> None:
        db, self._db = self._db, None
        primary: Optional[BaseException] = None
        foreign_sidecar = False
        for suffix in ("-journal", "-wal", "-shm"):
            try:
                os.stat(f"{self._db_path}{suffix}")
                if suffix not in _POP_SIDECARS.get(id(self), {}):
                    foreign_sidecar = True
            except OSError:
                pass
        if db is not None:
            preserved = []
            for suffix in ("-journal", "-wal", "-shm"):
                path = Path(f"{self._db_path}{suffix}")
                if path.exists() and suffix not in _POP_SIDECARS.get(id(self), {}):
                    backup = Path(f"{path}.preserve-{os.getpid()}-{id(self)}")
                    try: os.link(path, backup); preserved.append((path, backup))
                    except OSError: pass
            try:
                db.close()
            except BaseException as exc:
                primary = exc
            for path, backup in preserved:
                try:
                    if not path.exists(): os.rename(backup, path)
                    else: backup.unlink(missing_ok=True)
                except OSError: pass
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                target = Path(f"{self._db_path}{suffix}")
                if suffix == "":
                    info = os.stat(target)
                    if (info.st_dev, info.st_ino) not in _POP_OWNERS.get(id(self), set()): continue
                    _after_ownership_validation("population_main", target)
                    info2 = os.stat(target)
                    if (info2.st_dev, info2.st_ino) not in _POP_OWNERS.get(id(self), set()): continue
                else:
                    continue
                target.unlink()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if primary is None:
                    primary = exc
        if primary is not None:
            raise primary
        _POP_OWNERS.pop(id(self), None); _POP_SIDECARS.pop(id(self), None)

    def abort(self) -> None:
        try:
            self._close_spool()
        except BaseException:
            pass

def _source_case(source: Sequence[torch.Tensor], index: int, instance_id: str) -> dict[str, Any]:
    inp, b2b, p2b, pins, _tree, fp, metrics = source
    row = inp[index].tolist(); fprow = fp[index].tolist(); metric = metrics[index].tolist()
    n = next((i for i, x in enumerate(row) if float(x[0]) == -1.0), len(row))
    if n <= 0 or any(float(x[0]) != -1.0 for x in row[n:]): raise ValueError("area padding")
    row, fprow = row[:n], fprow[:n]
    area = [float(x[0]) for x in row]
    cons = [[int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])] for x in row]
    tp = []
    for vals, flags in zip(fprow, cons):
        fixed, pre = flags[:2]
        tp.append([float(vals[2]) if pre else -1.0, float(vals[3]) if pre else -1.0,
                   float(vals[0]) if fixed or pre else -1.0, float(vals[1]) if fixed or pre else -1.0])
    return {"instance_id": instance_id, "n": n, "area": area, "cons": cons, "tp": tp,
            "b2b": _trim_rows(b2b, 3, index), "p2b": _trim_rows(p2b, 3, index),
            "pins": _trim_rows(pins, 2, index), "hpwl_ref": float(metric[6] + metric[7]),
            "area_ref": float(metric[0])}

def _source_case_from_shard(source: Sequence[torch.Tensor], index: int, instance_id: str) -> dict[str, Any]:
    return _sanitize_case(_source_case(source, index, instance_id))

_TRUST_FIELDS = ("trust_ok", "input_ok", "scorer_ok", "checkpoint_sha256", "model_identity", "scorer_sha256", "scorer_contract", "shapely_version")
_PROTECTED = {"receipt", "instance_id", "partition", "sample_seed", "n", "base_cost", "teacher_cost", "record_weight"}

def _topology_sha(rects: Optional[torch.Tensor], case: Mapping[str, Any]) -> Optional[str]:
    if rects is None:
        return None
    rel = _proposal_fingerprint(rects, case["cons"])
    payload = json.dumps(rel, sort_keys=False, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()

def _outcome_from_lifecycle(case_input: _CaseInput, raw_rects: torch.Tensor, lifecycle: _CandidateLifecycle) -> _CaseOutcome:
    if not isinstance(lifecycle, _CandidateLifecycle) or not isinstance(raw_rects, torch.Tensor):
        raise ValueError("lifecycle")
    if raw_rects.ndim != 2 or raw_rects.shape[-1] != 4 or raw_rects.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(raw_rects).all()):
        raise ValueError("raw compiler input")
    rows = []
    rejects = []
    base = next((r for r in lifecycle.candidates if r.name == "base"), None)
    winner = next((r for r in lifecycle.candidates if r.ordinal == lifecycle.winner_ordinal), None)
    base_ok = base is not None and base.official_cost is not None and base.rejection_reason is None
    status = "base_unavailable"
    if base_ok and winner is not None:
        status = "winner_mutation" if winner.name != "base" else "winner_base_no_improvement"
    for r in lifecycle.candidates:
        reason = r.rejection_reason
        admitted = r.legal is not None
        admission = "admitted" if admitted else "failed"
        if not admitted: reason = "admission_failed"
        hard_pass = admitted and bool(r.hard) and all(r.hard.values())
        hard_status = "passed" if hard_pass else ("failed" if admitted else "not_reached")
        intent_failed = reason == "intent_not_survived"
        intent_status = "failed" if intent_failed else ("passed" if hard_pass else "not_reached")
        official_status = {"official_infeasible":"infeasible","official_invalid_cost":"invalid_cost","official_evaluator_error":"error"}.get(reason, "scored" if r.official_cost is not None else "not_reached")
        terminal = status if lifecycle.winner_ordinal == r.ordinal else ("base_unavailable" if lifecycle.winner_ordinal is None and r.official_cost is not None else ("not_selected" if official_status == "scored" else "rejected"))
        if reason in {"admission_failed", "hard_audit_failed", "intent_not_survived", "official_infeasible", "official_invalid_cost", "official_evaluator_error"}:
            official_status = {"official_infeasible":"infeasible","official_invalid_cost":"invalid_cost","official_evaluator_error":"error"}.get(reason, "not_reached")
        if reason in {"admission_failed", "hard_audit_failed", "intent_not_survived"}:
            terminal = "rejected"
        reached_official = official_status == "scored"
        row = {"ordinal":r.ordinal,"name":r.name,"intended_intent":r.name,"raw_topology_fingerprint":_topology_sha(raw_rects,case_input.case),"intended_topology_fingerprint":_topology_sha(r.original,case_input.case),"realized_topology_fingerprint":_topology_sha(r.legal,case_input.case),"admission_status":admission,"admission_reason":(None if admitted else "admission_failed"),"drift":(dict(r.drift) if admitted else None),"hard":(dict(r.hard) if admitted else None),"hard_status":hard_status,"intent_status":intent_status,"official_cost":r.official_cost if reached_official else None,"feasible":(True if reached_official else (False if reason == "official_infeasible" else None)),"official_status":official_status,"diagnostic_energy":r.diagnostic_energy if reached_official else None,"energy_status":r.energy_status if reached_official else "not_reached","winner":lifecycle.winner_ordinal == r.ordinal,"status":terminal}
        rows.append(row)
        if r.rejection_reason and r.rejection_reason != "not_selected":
            reason = r.rejection_reason
            stage = {"admission_failed":"admission","hard_audit_failed":"hard","intent_not_survived":"intent","official_infeasible":"official","official_invalid_cost":"official","official_evaluator_error":"official","base_unavailable":"selection"}.get(reason,"selection")
            rejects.append({"ordinal":r.ordinal,"name":r.name,"stage":stage,"reason":reason})
    label = None; bc = tc = None
    if base_ok and winner is not None:
        bc, tc = float(lifecycle.base_cost), float(lifecycle.teacher_cost)
        sparse = extract_sparse_label(winner.legal, case_input.case, case_input.case["instance_id"], case_input.sample_seed, tc, bc)
        label = {"edges":[{"src":x.src,"dst":x.dst,"axis":x.axis,"margin":x.margin,"kind":x.kind,"weight":x.weight} for x in sparse.edges],"contacts":[{"a":x.a,"b":x.b,"axis":x.axis,"a_before_b":x.a_before_b,"perp_margin":x.perp_margin,"weight":x.weight} for x in sparse.contacts],"pin_paths":[list(x) for x in sparse.pin_paths],"proposal_ordinal":winner.ordinal,"proposal_name":winner.name,"base_cost":bc,"teacher_cost":tc,"record_weight":bc/tc}
    return _CaseOutcome(label, tuple(rows), tuple(rejects), bc, tc, bool(base_ok), bool(base_ok), status)

def _finite_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("noncanonical runtime evidence") from exc

def _validate_preflight(value: Any, policy: TeacherTrustPolicy) -> dict[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != _TRUST_FIELDS:
        raise ValueError("preflight schema")
    expected = {"checkpoint_sha256": policy.expected_checkpoint_sha256, "model_identity": dict(policy.allowed_model_identity), "scorer_sha256": policy.expected_scorer_sha256, "scorer_contract": policy.scorer_contract, "shapely_version": policy.shapely_version}
    for key in ("trust_ok", "input_ok", "scorer_ok"):
        if type(value[key]) is not bool or not value[key]: raise ValueError("preflight policy")
    for key, expected_value in expected.items():
        if value[key] != expected_value: raise ValueError("preflight policy")
    _finite_json(value)
    return dict(value)

def _validate_outcome(value: Any) -> _CaseOutcome:
    if not isinstance(value, _CaseOutcome): raise ValueError("runtime outcome type")
    if type(value.legal) is not bool or type(value.covered) is not bool: raise ValueError("runtime flags")
    if value.case_status not in {"winner_base_no_improvement","winner_mutation","base_unavailable"}: raise ValueError("case status")
    if value.case_status == "base_unavailable":
        if type(value.legal) is not bool or type(value.covered) is not bool or value.legal or value.covered: raise ValueError("runtime flags")
        if value.label_row is not None or value.base_cost is not None or value.teacher_cost is not None: raise ValueError("base unavailable")
    else:
        if not value.legal or not value.covered: raise ValueError("runtime flags")
        if not all(isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x)) and float(x)>0 for x in (value.base_cost,value.teacher_cost)): raise ValueError("runtime costs")
        if value.teacher_cost > value.base_cost: raise ValueError("runtime costs")
        if not isinstance(value.label_row, Mapping): raise ValueError("label schema")
    rows = list(value.proposal_rows)
    if any(not isinstance(row, Mapping) for row in rows): raise ValueError("proposal schema")
    names = [r.get("name") for r in rows]; ords = [r.get("ordinal") for r in rows]
    if any(not isinstance(n, str) or not n.strip() for n in names) or any(type(o) is not int or o < 0 for o in ords): raise ValueError("proposal identity")
    expected = {"ordinal","name","intended_intent","raw_topology_fingerprint","intended_topology_fingerprint","realized_topology_fingerprint","admission_status","admission_reason","drift","hard","hard_status","intent_status","official_cost","feasible","official_status","diagnostic_energy","energy_status","winner","status"}
    if any(set(r) != expected for r in rows): raise ValueError("proposal schema")
    if value.case_status != "base_unavailable":
        base_row = next((r for r in rows if r.get("name") == "base"), None)
        if base_row is None or base_row.get("official_cost") != value.base_cost: raise ValueError("base")
    def _num(x: Any, *, positive: bool = False) -> bool:
        return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x)) and (not positive or float(x) > 0)
    def _label_payload(label: Mapping[str, Any]) -> None:
        keys = {"edges", "contacts", "pin_paths", "proposal_ordinal", "proposal_name", "base_cost", "teacher_cost", "record_weight"}
        if set(label) != keys or any(k in label for k in (_PROTECTED - keys)): raise ValueError("label schema")
        if type(label["proposal_ordinal"]) is not int or label["proposal_ordinal"] < 0 or not isinstance(label["proposal_name"], str) or not label["proposal_name"].strip(): raise ValueError("label schema")
        if not all(_num(label[k], positive=True) for k in ("base_cost", "teacher_cost", "record_weight")): raise ValueError("label schema")
        if not isinstance(label["edges"], (list, tuple)) or not isinstance(label["contacts"], (list, tuple)) or not isinstance(label["pin_paths"], (list, tuple)): raise ValueError("label schema")
        edge_keys = {"src", "dst", "axis", "margin", "kind", "weight"}; contact_keys = {"a", "b", "axis", "a_before_b", "perp_margin", "weight"}
        for e in label["edges"]:
            if not isinstance(e, Mapping) or set(e) != edge_keys or any(k in e for k in _PROTECTED): raise ValueError("label schema")
            if any(type(e[k]) is not int or e[k] < 0 for k in ("src", "dst", "axis")) or not isinstance(e["kind"], str) or not e["kind"] or not _num(e["margin"]) or not _num(e["weight"], positive=True): raise ValueError("label schema")
        for c in label["contacts"]:
            if not isinstance(c, Mapping) or set(c) != contact_keys or any(k in c for k in _PROTECTED): raise ValueError("label schema")
            if any(type(c[k]) is not int or c[k] < 0 for k in ("a", "b", "axis")) or type(c["a_before_b"]) is not bool or not _num(c["perp_margin"]) or not _num(c["weight"], positive=True): raise ValueError("label schema")
        for path in label["pin_paths"]:
            if not isinstance(path, (list, tuple)) or not path or any(type(x) is not int or x < 0 for x in path): raise ValueError("label schema")
    for row in rows:
        if row["diagnostic_energy"] is not None and not _num(row["diagnostic_energy"]): raise ValueError("diagnostic_energy")
        if row["drift"] is not None:
            if not isinstance(row["drift"], Mapping) or not row["drift"] or any(not isinstance(k, str) or not _num(v) or float(v) < 0 for k, v in row["drift"].items()) or "max_abs" not in row["drift"] or row["drift"]["max_abs"] != max(row["drift"].values()): raise ValueError("drift")
        if row["hard"] is not None:
            if not isinstance(row["hard"], Mapping) or not row["hard"] or any(not isinstance(k, str) or type(v) is not bool for k, v in row["hard"].items()): raise ValueError("hard")
    if len(names) != len(set(names)) or len(ords) != len(set(ords)) or ords != list(range(len(rows))) or any(k in r for r in rows for k in _PROTECTED): raise ValueError("proposal provenance")
    if any(r["intended_intent"] != r["name"] or type(r["intended_intent"]) is not str for r in rows): raise ValueError("intended_intent")
    allowed_status = {"admitted", "failed"}; allowed_hard = {"passed", "failed", "not_reached"}
    for r in rows:
        if r["admission_status"] not in allowed_status or r["hard_status"] not in allowed_hard or r["intent_status"] not in {"passed", "failed", "not_reached"}: raise ValueError("stage")
        if r["admission_status"] == "failed" and any(r[k] is not None for k in ("drift", "hard", "official_cost", "feasible", "diagnostic_energy")): raise ValueError("stage null")
        if r["admission_status"] == "failed" and r["official_status"] != "not_reached": raise ValueError("stage")
        if r["admission_status"] == "admitted":
            if r["drift"] is None or r["hard"] is None: raise ValueError("stage")
            expected_hard = "passed" if all(r["hard"].values()) else "failed"
            if r["hard_status"] != expected_hard: raise ValueError("hard status")
            if r["hard_status"] == "failed" and r["intent_status"] != "not_reached": raise ValueError("stage")
            if r["hard_status"] == "passed" and r["intent_status"] not in {"passed", "failed"}: raise ValueError("stage")
        if r["official_status"] == "scored":
            if r["official_cost"] is None or not _num(r["official_cost"], positive=True) or r["feasible"] is not True: raise ValueError("official")
        elif r["official_status"] == "infeasible":
            if r["feasible"] is not False or r["official_cost"] is not None: raise ValueError("official")
        elif r["official_status"] in {"invalid_cost", "error", "not_reached"} and r["feasible"] is not None: raise ValueError("official")
    if value.case_status != "base_unavailable":
        base_row = next((r for r in rows if r["name"] == "base"), None)
        if base_row is None or base_row["official_cost"] != value.base_cost: raise ValueError("base")
    winners = [r for r in rows if r.get("winner") is True]
    if value.case_status != "base_unavailable" and len(winners) != 1: raise ValueError("runtime winner count")
    if value.case_status == "base_unavailable" and winners: raise ValueError("runtime winner count")
    if value.case_status != "base_unavailable":
        winner = winners[0]; base = next((r for r in rows if r["name"] == "base"), None)
        if base is None or winner["official_cost"] != value.teacher_cost or base["official_cost"] != value.base_cost: raise ValueError("winner/base costs")
        if value.label_row["proposal_ordinal"] != winner["ordinal"] or value.label_row["proposal_name"] != winner["name"]: raise ValueError("label mismatch")
        if value.label_row["base_cost"] != value.base_cost or value.label_row["teacher_cost"] != value.teacher_cost: raise ValueError("label costs")
        if value.label_row["record_weight"] != value.base_cost / value.teacher_cost: raise ValueError("label record weight")
        _label_payload(value.label_row)
        if any(edge.get("weight") != 1.0 for edge in value.label_row.get("edges", ())) or any(c.get("weight") != 1.0 for c in value.label_row.get("contacts", ())): raise ValueError("label sparse payload")
        if value.case_status != winner["status"]: raise ValueError("case status")
    if len({(r.get("ordinal"), r.get("name")) for r in value.rejection_rows}) != len(value.rejection_rows): raise ValueError("rejection duplicate")
    if any(k in r for r in value.rejection_rows for k in _PROTECTED): raise ValueError("rejection provenance")
    for r in value.rejection_rows:
        if not isinstance(r, Mapping) or set(r) != {"ordinal", "name", "stage", "reason"}: raise ValueError("rejection schema")
        if type(r["ordinal"]) is not int or r["ordinal"] < 0 or not isinstance(r["name"], str) or not r["name"].strip() or not isinstance(r["stage"], str) or not isinstance(r["reason"], str): raise ValueError("rejection schema")
    if value.case_status != "base_unavailable":
        for r in rows:
            if r["winner"] is True and r["status"] != value.case_status: raise ValueError("winner status")
            if r["winner"] is not True and r["status"] in {"winner_base_no_improvement", "winner_mutation"}: raise ValueError("winner status")
    if value.label_row is not None:
        _label_payload(value.label_row)
    for row in [value.label_row, *rows, *value.rejection_rows]: _finite_json(row)
    return value


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
    trust = _validate_preflight(runtime.preflight(policy, Path(args.checkpoint)), policy)
    staging: Optional[Path] = None
    lease: Optional[_StagingLease] = None
    writer: Optional[_JsonlWriter] = None
    population: Optional[_PopulationAccumulator] = None
    spool: Optional[sqlite3.Connection] = None
    spool_identity: Optional[tuple[int, int]] = None
    spool_mismatch = False
    index_fd: Optional[Any] = None
    try:
        staging = _new_staging(destination)
        try:
            lease = _new_staging_lease(staging)
        except BaseException:
            raise
        writer = _JsonlWriter(lease.path)
        spool_path = lease.path / "case_spool.sqlite"
        spool = sqlite3.connect(str(spool_path))
        spool_identity = (os.stat(spool_path).st_dev, os.stat(spool_path).st_ino)
        spool.execute("PRAGMA cache_size=-64")
        spool.execute("PRAGMA temp_store=FILE")
        spool.execute("PRAGMA mmap_size=0")
        spool.execute("CREATE TABLE shards(worker INTEGER,layout INTEGER,summary_json TEXT,PRIMARY KEY(worker,layout))")
        spool.execute("CREATE TABLE cases(worker INTEGER,layout INTEGER,source_row_index INTEGER,relative_path TEXT,file_sha256 TEXT,source_row_count INTEGER,instance_id TEXT,case_json TEXT,fingerprint TEXT,PRIMARY KEY(worker,layout,source_row_index))")
        spool.commit()
        population = _PopulationAccumulator()
        train_count = held_count = heldout_winners = 0
        legal = covered = True; processed = 0
        files = _iter_approved_shards(root)
        if args.max_files is not None: files = files[:args.max_files]
        # Registration is a global pre-pass: every eligible heldout identity is
        # known before either train or heldout runtime call.
        for _worker, _layout, path in files:
            raw, source = _read_verified_shard(root, _worker, _layout)
            source_witness = tuple(x.detach().clone() for x in source)
            direct = _VerifiedShardSummary(_worker, _layout,
                f"worker_{_worker}/layouts_{_layout}.th",
                hashlib.sha256(raw).hexdigest(), _validate_source_shard(source)[0])
            summary = _verified_shard_summary(raw, source, _worker, _layout)
            if len(source) != len(source_witness) or any(not torch.equal(a, b) for a, b in zip(source, source_witness)):
                raise ValueError("mutated verified source")
            if type(summary) is not type(direct) or summary != direct:
                raise ValueError("forged shard provenance")
            try:
                spool.execute("INSERT INTO shards VALUES (?,?,?)", (_worker, _layout, json.dumps(
                    [summary.worker, summary.layout, summary.relative_path,
                     summary.file_sha256, summary.source_row_count],
                    ensure_ascii=True, separators=(",", ":"), allow_nan=False)))
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate shard identity") from exc
            source_count, _ = _validate_source_shard(source)
            digest, count, rel = direct.file_sha256, direct.source_row_count, direct.relative_path
            for index in range(count):
                iid = f"{rel}#{index}"
                case = _source_case_from_shard(source, index, iid)
                try:
                    spool.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?)", (_worker, _layout, index, rel, digest, count, iid, json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False), fingerprint_case(case)))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate case spool identity") from exc
                if case["n"] >= args.n_min and split_for_id(iid, args.heldout_mod) == "heldout":
                    population.register({"relative_path": rel, "layout_index": index,
                                         "instance_id": iid, "n": case["n"]})
            del source, raw
        spool.commit()
        mutation_proposals = mutation_admitted = mutation_intent_survived = 0
        positive_gain_heldout = 0
        positive_gain_heldout_weight = 0.0
        eligible_heldout_weight = 0.0
        rejection_counts = {k: 0 for k in ("admission_failed", "hard_audit_failed",
            "intent_not_survived", "official_infeasible", "official_invalid_cost",
            "official_evaluator_error", "base_unavailable")}
        index_fd = open(lease.path / "training_index.json", "wb")
        index_fd.write(b'{"rows":['); first_index = True
        for _worker, _layout, index, rel, digest, count, iid, case_text, fp, summary_text in spool.execute("SELECT c.worker,c.layout,c.source_row_index,c.relative_path,c.file_sha256,c.source_row_count,c.instance_id,c.case_json,c.fingerprint,s.summary_json FROM cases c JOIN shards s ON s.worker=c.worker AND s.layout=c.layout ORDER BY c.worker,c.layout,c.source_row_index"):
                case = json.loads(case_text)
                summary = _VerifiedShardSummary(*json.loads(summary_text))
                record = {"worker": _worker, "layout": _layout, "relative_path": rel,
                          "file_sha256": digest, "source_row_count": count,
                          "source_row_index": index, "instance_id": iid,
                          "case_json": case_text, "case": case, "fingerprint": fp,
                          "receipt": CorpusSourceReceipt(rel, digest, index, fp)}
                case, receipt = _validate_spooled_case_row(record, summary)
                entry = {"receipt": dataclass_to_dict(receipt), "instance_id": iid, "source_row_count": count, "block_count": case["n"]}
                if case["n"] < args.n_min:
                    row_index = {**entry, "partition": None, "sample_ordinal": None, "sample_seed": None, "status": "excluded_n_min"}
                    if not first_index: index_fd.write(b",")
                    index_fd.write(json.dumps(row_index,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()); index_fd.flush(); first_index=False; continue
                partition = split_for_id(iid, args.heldout_mod); seed = _sample_seed(args.seed, iid, 0)
                ci = _CaseInput(case, receipt, partition, seed); outcome = _validate_outcome(runtime.process_case(ci)); processed += 1
                env = {"receipt": dataclass_to_dict(receipt), "instance_id": iid, "partition": partition, "sample_seed": seed, "n": case["n"]}
                label = None if outcome.label_row is None else {**dict(outcome.label_row), **env}
                writer.write("train_corpus.jsonl" if partition == "train" else "heldout_corpus.jsonl", case)
                if label is not None: writer.write("train_labels.jsonl" if partition == "train" else "heldout_labels.jsonl", label)
                for p in sorted(outcome.proposal_rows, key=lambda x: (x["ordinal"], x["name"])): writer.write("proposals.jsonl", {**env, **dict(p)})
                for r in sorted(outcome.rejection_rows, key=lambda x: (x.get("ordinal", 0), x.get("name", ""))): writer.write("rejections.jsonl", {**env, **dict(r)})
                if partition == "train": train_count += 1
                else:
                    held_count += 1
                    eligible_heldout_weight += math.exp(case["n"] / 12)
                    if outcome.base_cost is not None:
                        heldout_winners += 1
                        population.record_winner(iid, outcome.base_cost, outcome.teacher_cost)
                        if outcome.teacher_cost < outcome.base_cost:
                            positive_gain_heldout += 1
                            positive_gain_heldout_weight += math.exp(case["n"] / 12)
                for rejection in outcome.rejection_rows:
                    reason = rejection.get("reason")
                    if reason in rejection_counts:
                        rejection_counts[reason] += 1
                if outcome.case_status == "base_unavailable" and not any(
                        r.get("reason") == "base_unavailable" for r in outcome.rejection_rows):
                    rejection_counts["base_unavailable"] += 1
                mutation_rows = [p for p in outcome.proposal_rows if p.get("name") != "base"]
                mutation_proposals += len(mutation_rows)
                mutation_admitted += sum(p.get("admission_status") == "admitted" for p in mutation_rows)
                mutation_intent_survived += sum(p.get("intent_status") == "passed" for p in mutation_rows)
                legal = legal and outcome.legal; covered = covered and outcome.covered
                row_index = {**entry, "partition": partition, "sample_ordinal": 0, "sample_seed": seed, "status": outcome.case_status}
                if not first_index: index_fd.write(b",")
                index_fd.write(json.dumps(row_index,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()); index_fd.flush(); first_index=False
        writer.close()
        index_fd.write(b'],"schema":"icdc_topology_training_index_v1"}\n'); index_fd.flush(); os.fsync(index_fd.fileno()); index_fd.close()
        spool.close(); spool = None
        try:
            info = os.stat(spool_path)
            if (info.st_dev, info.st_ino) == spool_identity:
                _after_ownership_validation("case_spool", spool_path)
                info2 = os.stat(spool_path)
                if (info2.st_dev, info2.st_ino) == spool_identity: spool_path.unlink()
                else: raise ValueError("case spool ownership mismatch")
            else: raise ValueError("case spool ownership mismatch")
        except FileNotFoundError: pass
        pop = population.finish()
        coverage = {"eligible_train": train_count, "eligible_heldout": held_count, "heldout_winners": heldout_winners, "legal": legal, "covered": covered and bool(train_count) and bool(held_count) and heldout_winners == held_count,
                    "mutation_proposals": mutation_proposals, "mutation_admitted": mutation_admitted,
                    "mutation_intent_survived": mutation_intent_survived, "positive_gain_heldout": positive_gain_heldout,
                    "eligible_heldout_weight": eligible_heldout_weight, "positive_gain_heldout_weight": positive_gain_heldout_weight,
                    "rejection_counts": rejection_counts}
        state = _g0_state({**trust, "legal": legal, "coverage": coverage["covered"], "teacher_mean": pop["T_H"], "delta": pop["Delta_H"]})
        authorized = runtime.authorizing and state == "TARGET_GAIN_MET" and args.max_files is None
        if not processed: state = "KILLED_LEGALITY_OR_COVERAGE"; authorized = False
        secondary_reasons = ["base_unavailable"] if rejection_counts["base_unavailable"] else []
        manifest = {"schema":"icdc_topology_teacher_g0_v1","status":"complete","state":state,"secondary_reasons":secondary_reasons,"training_authorized":authorized,"bounded_max_files":args.max_files is not None,"trust":trust,"population":pop,"coverage":coverage}
        writer.close()
        def _fsync_dir() -> None:
            fd = os.open(lease.path, os.O_RDONLY | os.O_DIRECTORY)
            primary: Optional[BaseException] = None
            try:
                os.fsync(fd)
            except BaseException as exc:
                primary = exc
            try:
                os.close(fd)
            except BaseException:
                if primary is None:
                    raise
            if primary is not None:
                raise primary
        _fsync_dir()
        manifest["support_hashes"] = {n: hashlib.sha256((lease.path/n).read_bytes()).hexdigest() for n in (*_TASK4_JSONL,"training_index.json")}; manifest["self_sha256"] = _manifest_self_sha256(manifest); _write_json_fsync(lease.path/"g0_manifest.json", manifest)
        _fsync_dir()
        _publish_staging(lease, destination)
        return 0 if authorized else 1
    except BaseException:
        for handle in (index_fd,):
            try:
                if handle is not None and not handle.closed:
                    handle.close()
            except BaseException:
                pass
        try:
            if spool is not None:
                spool.close()
        except BaseException:
            pass
        if spool_identity is not None:
            try:
                info = os.stat(lease.path / "case_spool.sqlite") if lease is not None else None
                spool_mismatch = info is not None and (info.st_dev, info.st_ino) != spool_identity
            except OSError:
                pass
        if population is not None:
            population.abort()
        if writer is not None:
            writer.abort()
        if lease is not None and not spool_mismatch:
            try:
                _cleanup_owned_staging(lease)
            except BaseException:
                pass
        raise

def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return {"relative_path": value.relative_path, "file_sha256": value.file_sha256, "layout_index": value.layout_index, "fingerprint": value.fingerprint}


if __name__ == "__main__":
    teacher_main(sys.argv[1:])
