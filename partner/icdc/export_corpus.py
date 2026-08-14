"""Stream receipt-bound sanitized model inputs for sealed sparse labels."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from .topology_data import (
    CorpusSourceReceipt,
    fingerprint_case,
    source_instance_id,
    validate_raw_source,
    verified_training_fp_row,
)


_REPO = Path(__file__).resolve().parents[2]
_CANONICAL_ROOT = (_REPO / "FloorSet/floorset_lite").resolve()
_SHARD = re.compile(r"^worker_([0-9]+)/layouts_([0-9]+)\.th$")


def _canonical(value: Any) -> bytes:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(value: Any) -> CorpusSourceReceipt:
    if not isinstance(value, Mapping) or set(value) != {
        "relative_path", "file_sha256", "layout_index", "fingerprint"
    }:
        raise ValueError("receipt")
    receipt = CorpusSourceReceipt(**value)
    source_instance_id(receipt)
    for digest in (receipt.file_sha256, receipt.fingerprint):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("receipt")
    if _SHARD.fullmatch(receipt.relative_path) is None:
        raise ValueError("receipt path")
    return receipt


def _read_source(root: Path, relative_path: str) -> tuple[bytes, Any]:
    match = _SHARD.fullmatch(relative_path)
    if match is None:
        raise ValueError("source path")
    worker, layout = int(match.group(1)), int(match.group(2))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        worker_fd = os.open(
            f"worker_{worker}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            source_fd = os.open(
                f"layouts_{layout}.th",
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=worker_fd,
            )
            try:
                if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                    raise ValueError("source type")
                chunks = []
                while chunk := os.read(source_fd, 1024 * 1024):
                    chunks.append(chunk)
            finally:
                os.close(source_fd)
        finally:
            os.close(worker_fd)
    finally:
        os.close(root_fd)
    raw = b"".join(chunks)
    return raw, torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def export_corpus(
    labels_path: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    *,
    canonical_root: Optional[Path] = None,
) -> dict[str, Any]:
    expected_root = (canonical_root or _CANONICAL_ROOT).resolve()
    supplied_root = Path(data_root)
    if supplied_root.is_symlink() or supplied_root.resolve() != expected_root:
        raise ValueError("data root")
    labels = Path(labels_path)
    destination = Path(out_dir)
    if destination.exists() or destination.is_symlink():
        raise ValueError("existing output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    corpus_path = stage / "corpus.jsonl"
    count = 0
    previous_id: Optional[str] = None
    current_path: Optional[str] = None
    current_digest: Optional[str] = None
    current_source: Any = None
    try:
        with labels.open("r", encoding="ascii") as label_stream, corpus_path.open(
            "wb"
        ) as corpus_stream:
            for line in label_stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("label json") from exc
                if not isinstance(record, Mapping) or set(record) != {"receipt", "label"}:
                    raise ValueError("label schema")
                receipt = _receipt(record["receipt"])
                label = record["label"]
                if not isinstance(label, Mapping):
                    raise ValueError("label schema")
                instance_id = source_instance_id(receipt)
                if label.get("instance_id") != instance_id:
                    raise ValueError("label binding")
                if previous_id is not None and instance_id == previous_id:
                    raise ValueError("duplicate label")
                if receipt.relative_path != current_path:
                    raw, source = _read_source(expected_root, receipt.relative_path)
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest != receipt.file_sha256:
                        raise ValueError("source hash")
                    validate_raw_source(source)
                    current_path = receipt.relative_path
                    current_digest = digest
                    current_source = source
                if current_digest != receipt.file_sha256:
                    raise ValueError("source hash")
                verified = verified_training_fp_row(current_source, receipt)
                if (
                    verified.instance_id != instance_id
                    or verified.input_fingerprint != receipt.fingerprint
                    or fingerprint_case(verified.case) != receipt.fingerprint
                ):
                    raise ValueError("source binding")
                corpus_stream.write(
                    _canonical(
                        {
                            "receipt": dataclasses.asdict(receipt),
                            "case": dict(verified.case),
                        }
                    )
                )
                del verified
                previous_id = instance_id
                count += 1
            corpus_stream.flush()
            os.fsync(corpus_stream.fileno())
        if count == 0:
            raise ValueError("empty labels")
        manifest = {
            "schema": "icdc_bound_corpus_export_v1",
            "status": "complete",
            "record_count": count,
            "labels_sha256": _sha256(labels),
            "corpus_sha256": _sha256(corpus_path),
            "source_root": "FloorSet/floorset_lite",
        }
        with (stage / "manifest.json").open("wb") as stream:
            stream.write(_canonical(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(stage, destination)
        return manifest
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    result = export_corpus(args.labels, args.data_root, args.out_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
