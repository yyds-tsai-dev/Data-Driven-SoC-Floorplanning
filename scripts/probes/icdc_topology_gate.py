#!/usr/bin/env python3
"""Seal evaluator/receipt joins and adjudicate the one authorized G1 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


_REPO = Path(__file__).resolve().parents[2]
_PARTNER = _REPO / "partner"
if str(_PARTNER) not in sys.path:
    sys.path.insert(0, str(_PARTNER))

from icdc.g1_evidence import G1ArmEvidence, compare_g1_arms  # noqa: E402
from icdc.g1_runtime import seal_evaluator_arm  # noqa: E402
from icdc.topology_artifact_guard import write_checked_json  # noqa: E402


def _record(path: str) -> dict:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("input artifact")
    value = json.loads(source.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("input artifact")
    return value


def _sha256(path: str) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("freeze artifact")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _seal(args: argparse.Namespace) -> int:
    smoke = _record(args.causal_smoke)
    if smoke.get("causal_smoke_ok") is not True:
        raise ValueError("causal smoke")
    arm = seal_evaluator_arm(
        args.arm_id,
        _record(args.evaluator),
        _record(args.receipt),
        _REPO,
        freeze_sha256=_sha256(args.freeze),
        causal_smoke_ok=True,
    )
    write_checked_json(Path(args.out), arm.to_record())
    return 0


def _compare(args: argparse.Namespace) -> int:
    control = G1ArmEvidence.from_record(_record(args.control))
    candidate = G1ArmEvidence.from_record(_record(args.candidate))
    result = compare_g1_arms(control, candidate)
    write_checked_json(Path(args.out), result.to_record())
    return 0 if result.state == "HIGH_TAIL_CAUSAL_PROOF" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-arm")
    seal.add_argument("--arm-id", required=True)
    seal.add_argument("--evaluator", required=True)
    seal.add_argument("--receipt", required=True)
    seal.add_argument("--freeze", required=True)
    seal.add_argument("--causal-smoke", required=True)
    seal.add_argument("--out", required=True)
    seal.set_defaults(run=_seal)
    compare = commands.add_parser("compare")
    compare.add_argument("--control", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(run=_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
