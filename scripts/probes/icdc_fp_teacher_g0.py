#!/usr/bin/env python3
"""Streaming formal G0-v2 runner for the transient training-fp oracle."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch


_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO / "partner", _REPO / "scripts", _REPO / "FloorSet"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from icdc.g0_v2 import (  # noqa: E402
    G0CaseResult,
    G0Summary,
    PopulationAccumulator,
    canonical_case_record,
    evaluate_case,
    transient_fp_xywh,
)
from icdc.topology_data import (  # noqa: E402
    CorpusSourceReceipt,
    TopologyLabel,
    _sanitize,
    fingerprint_case,
    split_for_id,
    validate_raw_source,
)


_SCORER_SHA256 = "7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1"
_QA_SHA256 = "60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db"
_DIRECT_SHA256 = "2b9ce827aed93443e442a002d178e8e6282cb4c6148818c9122a6ff0411c8a02"
_FLOW_SHA256 = "110c1d84d74ee88d94cf8d3be9ac464602747db8c301d95b3ca69a2cb8bd2f09"
_WRAPPER_SHA256 = "15419b21acc629934181c552e0ba2798582e220edef26fcff63eef7710d3c577"
_SOURCE_SHA256 = "6b31e01c87ff1d8e157a116551dac31115b9de82ce7fe68e3ab1670daf872a3c"
_CANONICAL_DATA_ROOT = (_REPO / "FloorSet" / "floorset_lite").resolve()
_FORBIDDEN_SOLVER_KEYS = {
    "PARTNER_GPU_ARM",
    "PARTNER_FLOW_ZORDER",
    "PARTNER_FLOW_NOPT",
    "PARTNER_NOISE_OPT",
    "PARTNER_PHYSICS_GUIDE",
    "PARTNER_ORACLE_PRED_FILE",
    "PARTNER_RETRIEVAL",
}


@dataclass(frozen=True)
class TrainingRow:
    receipt: CorpusSourceReceipt
    case: Mapping[str, Any]
    fp_xywh: torch.Tensor


@dataclass(frozen=True)
class BaseSolveResult:
    rects: torch.Tensor
    portfolio_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeDependencies:
    iter_shards: Callable[[Path], Iterable[tuple[int, int, Path]]]
    read_shard: Callable[[Path, int, int], tuple[bytes, Any]]
    validate_source: Callable[[Any], tuple[int, int]]
    make_row: Callable[[Any, str, str, int], TrainingRow]
    build_optimizer: Callable[[], Any]
    solve_base: Callable[[Any, TrainingRow], BaseSolveResult]
    scorer: Any
    evaluate: Callable[..., G0CaseResult]
    bindings: Callable[[], Mapping[str, Any]]


def production_environment() -> dict[str, str]:
    return {
        "PARTNER_NREF": "6",
        "PARTNER_FLOW_SLOTS": "3",
        "PARTNER_OVERSAMPLE": "1",
        "PARTNER_DIRECT_SOLVER": "dpmpp",
        "PARTNER_DDIM_STEPS": "2",
        "PARTNER_FLOW_SOLVER": "euler",
        "PARTNER_FLOW_STEPS": "8",
    }


def _apply_production_environment() -> None:
    for key in _FORBIDDEN_SOLVER_KEYS:
        os.environ.pop(key, None)
    for key, value in production_environment().items():
        os.environ[key] = value


def validate_portfolio_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "pool_ready": True,
        "requested_k": 6,
        "raw_candidate_count": 6,
        "direct_count": 3,
        "flow_count": 3,
        "oversample": False,
        "flow_exception": None,
        "fallback": False,
    }
    if not isinstance(receipt, Mapping) or set(receipt) != set(expected):
        raise ValueError("portfolio contract")
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError("portfolio contract")
    return receipt


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _label_payload(label: TopologyLabel) -> dict[str, Any]:
    if type(label) is not TopologyLabel:
        raise ValueError("sparse label")
    payload = dataclasses.asdict(label)
    if set(payload) != {
        "instance_id",
        "n",
        "sample_seed",
        "teacher_cost",
        "base_cost",
        "record_weight",
        "edges",
        "contacts",
        "pin_paths",
    }:
        raise ValueError("sparse label")
    return payload


def canonical_sparse_label(label: TopologyLabel) -> bytes:
    return _json_bytes(_label_payload(label))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str, field: str) -> str:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(field) from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(field)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(field)
    return actual


def _static_bindings() -> Mapping[str, Any]:
    paths = {
        "submission_wrapper_sha256": (
            _REPO / "submission/cadc1013/op_wrapper.py",
            _WRAPPER_SHA256,
        ),
        "submission_source_sha256": (
            _REPO / "submission/cadc1013/op_src.py",
            _SOURCE_SHA256,
        ),
        "direct_checkpoint_sha256": (
            _REPO / "submission/cadc1013/checkpoints/direct_v2_final.pt",
            _DIRECT_SHA256,
        ),
        "flow_checkpoint_sha256": (
            _REPO / "submission/cadc1013/checkpoints/flow_matching_v1_final.pt",
            _FLOW_SHA256,
        ),
        "scorer_sha256": (_REPO / "scripts/iccad2026_evaluate.py", _SCORER_SHA256),
        "qa_pdf_sha256": (_REPO / "docs/official/C_QA_20260804.pdf", _QA_SHA256),
    }
    result = {
        field: _verify_file(path, expected, field)
        for field, (path, expected) in paths.items()
    }
    result["production_environment"] = production_environment()
    try:
        result["source_commit"] = (
            __import__("subprocess")
            .check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
            )
            .strip()
        )
    except Exception as exc:
        raise ValueError("source commit") from exc
    return result


def _canonical_decimal(value: str) -> bool:
    return bool(value) and value.isascii() and value.isdigit() and str(int(value)) == value


def _iter_approved_shards(root: Path) -> list[tuple[int, int, Path]]:
    found: list[tuple[int, int, Path]] = []
    for worker in root.iterdir():
        body = worker.name[7:] if worker.name.startswith("worker_") else ""
        if not _canonical_decimal(body):
            continue
        if worker.is_symlink() or not worker.is_dir():
            raise ValueError("numeric worker")
        worker_id = int(body)
        for shard in worker.iterdir():
            name = shard.name
            layout_body = (
                name[8:-3]
                if name.startswith("layouts_") and name.endswith(".th")
                else ""
            )
            if not _canonical_decimal(layout_body):
                continue
            if shard.is_symlink() or not shard.is_file():
                raise ValueError("numeric shard")
            found.append((worker_id, int(layout_body), shard))
    return sorted(found, key=lambda item: (item[0], item[1]))


def _read_verified_shard(root: Path, worker: int, layout: int) -> tuple[bytes, Any]:
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        worker_fd = os.open(
            f"worker_{worker}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            shard_fd = os.open(
                f"layouts_{layout}.th",
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=worker_fd,
            )
            try:
                if not stat.S_ISREG(os.fstat(shard_fd).st_mode):
                    raise ValueError("shard type")
                chunks: list[bytes] = []
                while chunk := os.read(shard_fd, 1024 * 1024):
                    chunks.append(chunk)
            finally:
                os.close(shard_fd)
        finally:
            os.close(worker_fd)
    finally:
        os.close(root_fd)
    raw = b"".join(chunks)
    return raw, torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")


def _trim_relation(tensor: torch.Tensor, index: int, width: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in tensor[index]:
        values = [float(value) for value in raw]
        if all(value == -1.0 for value in values):
            break
        if len(values) != width:
            raise ValueError("source relation")
        rows.append(values)
    return rows


def _validated_training_row(
    source: Sequence[torch.Tensor], relative_path: str, digest: str, index: int
) -> TrainingRow:
    batch = int(source[0].shape[0])
    if index < 0 or index >= batch:
        raise ValueError("source row")
    input_row = source[0][index]
    n = int(input_row[:, 0].ne(-1).sum().item())
    instance_id = f"{relative_path}#{index}"
    fp_xywh = transient_fp_xywh(source, index, n)
    area = [float(value) for value in input_row[:n, 0]]
    cons = [[int(float(value)) for value in row[1:]] for row in input_row[:n]]
    tp: list[list[float]] = []
    for rect, constraint in zip(fp_xywh.tolist(), cons):
        fixed, preplaced = bool(constraint[0]), bool(constraint[1])
        tp.append(
            [
                rect[0] if preplaced else -1.0,
                rect[1] if preplaced else -1.0,
                rect[2] if fixed or preplaced else -1.0,
                rect[3] if fixed or preplaced else -1.0,
            ]
        )
    metrics = source[6][index]
    case = _sanitize(
        {
            "instance_id": instance_id,
            "n": n,
            "area": area,
            "cons": cons,
            "tp": tp,
            "b2b": _trim_relation(source[1], index, 3),
            "p2b": _trim_relation(source[2], index, 3),
            "pins": _trim_relation(source[3], index, 2),
            "hpwl_ref": float(metrics[6] + metrics[7]),
            "area_ref": float(metrics[0]),
        }
    )
    fingerprint = fingerprint_case(case)
    return TrainingRow(
        CorpusSourceReceipt(relative_path, digest, index, fingerprint),
        case,
        fp_xywh,
    )


def _load_scorer() -> Any:
    path = _REPO / "scripts/iccad2026_evaluate.py"
    _verify_file(path, _SCORER_SHA256, "scorer_sha256")
    spec = importlib.util.spec_from_file_location("icdc_g0_v2_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError("scorer import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "SHAPELY_AVAILABLE", False) is not True:
        raise ValueError("scorer shapely")
    return module


def _build_optimizer() -> Any:
    _apply_production_environment()
    submission = _REPO / "submission/cadc1013"
    if str(submission) not in sys.path:
        sys.path.insert(0, str(submission))
    path = submission / "op_wrapper.py"
    spec = importlib.util.spec_from_file_location("icdc_g0_v2_wrapper", path)
    if spec is None or spec.loader is None:
        raise ValueError("wrapper import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    optimizer = module.ContestOptimizer()
    if getattr(optimizer, "direct_model", None) is None:
        raise ValueError("direct checkpoint")
    if getattr(optimizer, "flow_model", None) is None:
        raise ValueError("flow checkpoint")
    return optimizer


def _case_tensor(case: Mapping[str, Any], name: str, width: Optional[int] = None) -> torch.Tensor:
    value = case[name]
    if width is not None and not value:
        return torch.empty((0, width), dtype=torch.float32)
    out = torch.as_tensor(value, dtype=torch.float32, device="cpu").contiguous()
    if width is not None and (out.ndim != 2 or out.shape[1] != width):
        raise ValueError(name)
    return out


def _solve_production_base(optimizer: Any, row: TrainingRow) -> BaseSolveResult:
    case = row.case
    n = int(case["n"])
    events: dict[str, Any] = {
        "requested_k": None,
        "raw_candidate_count": None,
        "flow_requested": None,
        "flow_count": None,
        "oversample": None,
        "flow_exception": None,
    }
    original_raw = optimizer._sample_direct_raw_preds
    original_flow = optimizer._sample_flow_preds

    def traced_flow(*args, **kwargs):
        requested = int(args[7] if len(args) > 7 else kwargs["K"])
        events["flow_requested"] = requested
        try:
            result = original_flow(*args, **kwargs)
        except Exception as exc:
            events["flow_exception"] = type(exc).__name__
            raise
        events["flow_count"] = len(result)
        return result

    def traced_raw(*args, **kwargs):
        requested = int(args[7] if len(args) > 7 else kwargs["K"])
        oversample = bool(args[8] if len(args) > 8 else kwargs.get("oversample", True))
        events["requested_k"] = requested
        events["oversample"] = oversample
        result = original_raw(*args, **kwargs)
        events["raw_candidate_count"] = len(result)
        return result

    optimizer._sample_flow_preds = traced_flow
    optimizer._sample_direct_raw_preds = traced_raw
    try:
        import column_sa_legalizer as legalizer

        pool_ready = bool(
            getattr(legalizer, "_POOL", None) is not None
            and getattr(legalizer, "_POOL_READY", False)
        )
        result = optimizer.solve(
            n,
            _case_tensor(case, "area"),
            _case_tensor(case, "b2b", 3),
            _case_tensor(case, "p2b", 3),
            _case_tensor(case, "pins", 2),
            _case_tensor(case, "cons", 5),
            target_positions=_case_tensor(case, "tp", 4),
        )
    finally:
        optimizer._sample_direct_raw_preds = original_raw
        optimizer._sample_flow_preds = original_flow
    rects = torch.as_tensor(result, dtype=torch.float64, device="cpu").contiguous()
    flow_count = events["flow_count"]
    raw_count = events["raw_candidate_count"]
    receipt = {
        "pool_ready": pool_ready,
        "requested_k": events["requested_k"],
        "raw_candidate_count": raw_count,
        "direct_count": None if raw_count is None or flow_count is None else raw_count - flow_count,
        "flow_count": flow_count,
        "oversample": events["oversample"],
        "flow_exception": events["flow_exception"],
        "fallback": events["requested_k"] is None,
    }
    validate_portfolio_receipt(receipt)
    return BaseSolveResult(rects, receipt)


def _production_dependencies() -> RuntimeDependencies:
    _apply_production_environment()
    return RuntimeDependencies(
        iter_shards=_iter_approved_shards,
        read_shard=_read_verified_shard,
        validate_source=validate_raw_source,
        make_row=_validated_training_row,
        build_optimizer=_build_optimizer,
        solve_base=_solve_production_base,
        scorer=_load_scorer(),
        evaluate=evaluate_case,
        bindings=_static_bindings,
    )


def _sample_seed(instance_id: str) -> int:
    return int.from_bytes(hashlib.sha256(instance_id.encode("utf-8")).digest()[:8], "big")


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_training_row(
    row: TrainingRow, relative_path: str, digest: str, index: int
) -> None:
    if type(row) is not TrainingRow:
        raise ValueError("training row")
    receipt = row.receipt
    if (
        receipt.relative_path != relative_path
        or receipt.file_sha256 != digest
        or receipt.layout_index != index
        or row.case.get("instance_id") != f"{relative_path}#{index}"
        or receipt.fingerprint != fingerprint_case(row.case)
    ):
        raise ValueError("training row binding")
    if (
        not isinstance(row.fp_xywh, torch.Tensor)
        or row.fp_xywh.device.type != "cpu"
        or row.fp_xywh.dtype is not torch.float64
        or tuple(row.fp_xywh.shape) != (int(row.case["n"]), 4)
    ):
        raise ValueError("training fp")


def run_g0(
    data_root: Path,
    out_dir: Path,
    *,
    n_min: int = 100,
    heldout_mod: int = 10,
    max_files: Optional[int] = None,
    deps: Optional[RuntimeDependencies] = None,
) -> G0Summary:
    if type(n_min) is not int or n_min < 0:
        raise ValueError("n_min")
    if type(heldout_mod) is not int or heldout_mod <= 0:
        raise ValueError("heldout_mod")
    if max_files is not None and (type(max_files) is not int or max_files <= 0):
        raise ValueError("max_files")
    root = Path(data_root)
    destination = Path(out_dir)
    if destination.exists() or destination.is_symlink():
        raise ValueError("existing output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime = deps if deps is not None else _production_dependencies()
    bindings = dict(runtime.bindings())
    optimizer = runtime.build_optimizer()
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    cases_path = stage / "cases.jsonl"
    labels_path = stage / "labels.jsonl"
    authorizing = max_files is None
    population = PopulationAccumulator(authorizing=authorizing)
    positive_gain = teacher_admitted = 0
    try:
        shards = sorted(runtime.iter_shards(root), key=lambda item: (item[0], item[1]))
        if max_files is not None:
            shards = shards[:max_files]
        with cases_path.open("wb") as cases_fd, labels_path.open("wb") as labels_fd:
            for worker, layout, _path in shards:
                relative_path = f"worker_{worker}/layouts_{layout}.th"
                raw, source = runtime.read_shard(root, worker, layout)
                digest = hashlib.sha256(raw).hexdigest()
                count, _ = runtime.validate_source(source)
                for index in range(count):
                    input_row = source[0][index]
                    n = int(input_row[:, 0].ne(-1).sum().item())
                    instance_id = f"{relative_path}#{index}"
                    if n < n_min or split_for_id(instance_id, heldout_mod) != "heldout":
                        continue
                    row = runtime.make_row(source, relative_path, digest, index)
                    _validate_training_row(row, relative_path, digest, index)
                    population.register(instance_id, n)
                    base = runtime.solve_base(optimizer, row)
                    validate_portfolio_receipt(base.portfolio_receipt)
                    result = runtime.evaluate(
                        base.rects,
                        row.fp_xywh,
                        row.case,
                        runtime.scorer,
                        sample_seed=_sample_seed(instance_id),
                    )
                    if type(result) is not G0CaseResult or result.sparse_label is None:
                        raise ValueError("case result")
                    population.add(result)
                    teacher_admitted += int(result.teacher_candidate_cost is not None)
                    positive_gain += int(result.teacher_cost < result.base_cost)
                    case_payload = {
                        "receipt": dataclasses.asdict(row.receipt),
                        "portfolio": dict(base.portfolio_receipt),
                        "result": json.loads(canonical_case_record(result)),
                    }
                    label_payload = {
                        "receipt": dataclasses.asdict(row.receipt),
                        "label": _label_payload(result.sparse_label),
                    }
                    cases_fd.write(_json_bytes(case_payload))
                    labels_fd.write(_json_bytes(label_payload))
                    del row, base, result
                del source, raw
            for handle in (cases_fd, labels_fd):
                handle.flush()
                os.fsync(handle.fileno())
        summary = population.finish(complete=True)
        population_payload = {
            **dataclasses.asdict(summary),
            "positive_gain_count": positive_gain,
            "teacher_admitted_count": teacher_admitted,
        }
        _write_fsync(stage / "population.json", _json_bytes(population_payload))
        support_hashes = {
            name: _sha256(stage / name)
            for name in ("cases.jsonl", "labels.jsonl", "population.json")
        }
        manifest = {
            "schema": "icdc_g0_v2.transient_fp.v1",
            "status": "complete",
            "authorizing": authorizing,
            "terminal_state": summary.terminal_state,
            "n_min": n_min,
            "heldout_mod": heldout_mod,
            "max_files": max_files,
            "bindings": bindings,
            "support_sha256": support_hashes,
        }
        _write_fsync(stage / "manifest.json", _json_bytes(manifest))
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists() or destination.is_symlink():
            raise ValueError("existing output")
        os.rename(stage, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return summary
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def _validate_production_root(path: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or supplied.resolve() != _CANONICAL_DATA_ROOT:
        raise ValueError("data root")
    current = supplied.absolute()
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("data root")
        current = current.parent
    return _CANONICAL_DATA_ROOT


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-min", type=int, default=100)
    parser.add_argument("--heldout-mod", type=int, default=10)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args(argv)
    root = _validate_production_root(args.data_root)
    summary = run_g0(
        root,
        Path(args.out_dir),
        n_min=args.n_min,
        heldout_mod=args.heldout_mod,
        max_files=args.max_files,
    )
    print(json.dumps(dataclasses.asdict(summary), sort_keys=True, allow_nan=False))
    if not summary.authorizing:
        return 3
    return 0 if summary.terminal_state == "TARGET_GAIN_MET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
