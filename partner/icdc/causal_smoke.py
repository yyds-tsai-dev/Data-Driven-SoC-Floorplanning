"""Non-validation causal reachability smoke for two exact 3D/3F Direct arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from .g1_runtime import build_solver_environment, validate_matched_environments
from .train_topology_prior import _atomic_bytes, _canonical_bytes, _sha256, load_paired_records


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def compare_smoke_arms(
    control: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not control or len(control) != len(candidate):
        raise ValueError("smoke coverage")
    direct_changed = 0
    post_rank_changed = 0
    for left, right in zip(control, candidate):
        required = {
            "instance_id", "direct_hashes", "flow_hashes", "post_rank_hashes"
        }
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or set(left) != required
            or set(right) != required
            or left["instance_id"] != right["instance_id"]
        ):
            raise ValueError("smoke identity")
        for record in (left, right):
            if (
                not isinstance(record["direct_hashes"], list)
                or len(record["direct_hashes"]) != 3
                or not isinstance(record["flow_hashes"], list)
                or len(record["flow_hashes"]) != 3
                or not isinstance(record["post_rank_hashes"], list)
                or len(record["post_rank_hashes"]) != 6
            ):
                raise ValueError("smoke 3D3F")
        if left["flow_hashes"] != right["flow_hashes"]:
            raise ValueError("Flow hashes")
        direct_changed += int(left["direct_hashes"] != right["direct_hashes"])
        post_rank_changed += int(
            left["post_rank_hashes"] != right["post_rank_hashes"]
        )
    return {
        "case_count": len(control),
        "direct_changed_count": direct_changed,
        "flow_identical_count": len(control),
        "post_rank_changed_count": post_rank_changed,
        "causal_smoke_ok": direct_changed > 0 and post_rank_changed > 0,
    }


def _case_tensors(case: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    def matrix(name: str, width: int) -> torch.Tensor:
        rows = case[name]
        return (
            torch.tensor(rows, dtype=torch.float32)
            if rows
            else torch.empty((0, width), dtype=torch.float32)
        )

    return (
        torch.tensor(case["area"], dtype=torch.float32),
        torch.tensor(case["cons"], dtype=torch.float32),
        torch.tensor(case["tp"], dtype=torch.float32),
        matrix("b2b", 3),
        matrix("p2b", 3),
        matrix("pins", 2),
    )


def _sample_arm(
    direct: Path,
    flow: Path,
    records: Sequence[Any],
    inherited: Mapping[str, str],
    device: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    execution_env = build_solver_environment(inherited, direct, flow)
    old_env = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(execution_env)
        partner = Path(__file__).resolve().parents[1]
        contest = Path(__file__).resolve().parents[2] / "FloorSet/iccad2026contest"
        for path in (partner, contest):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from contest_optimizer import MyOptimizer

        optimizer = MyOptimizer.__new__(MyOptimizer)
        optimizer.verbose = False
        optimizer.device = torch.device(device)
        optimizer.direct_model = None
        optimizer.flow_model = None
        optimizer._load_direct_model()
        optimizer._load_flow_model()
        if optimizer.direct_model is None or optimizer.flow_model is None:
            raise ValueError("smoke model load")
        rows: list[dict[str, object]] = []
        for record in records:
            n = int(record.case["n"])
            area, cons, tp, b2b, p2b, pins = _case_tensors(record.case)
            mixed = optimizer._sample_direct_raw_preds(
                n, area, cons, tp, b2b, p2b, pins,
                6, oversample=False, gen_seed=0,
            )
            if len(mixed) != 6:
                raise ValueError("smoke 3D3F")
            hashes = [_array_sha256(value) for value in mixed]
            order = optimizer._rank_portfolio(mixed, n, area, cons, b2b)
            if sorted(order) != list(range(6)):
                raise ValueError("smoke rank")
            rows.append({
                "instance_id": record.label.instance_id,
                "direct_hashes": hashes[:3],
                "flow_hashes": hashes[3:],
                "post_rank_hashes": [hashes[index] for index in order],
            })
        return rows, execution_env
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def run(args: argparse.Namespace) -> dict[str, object]:
    records = load_paired_records(
        args.corpus,
        args.labels,
        selection_mod=1,
        max_records=args.max_records,
    )
    control_rows, control_env = _sample_arm(
        Path(args.control), Path(args.flow), records, os.environ, args.device
    )
    candidate_rows, candidate_env = _sample_arm(
        Path(args.candidate), Path(args.flow), records, os.environ, args.device
    )
    environment = validate_matched_environments(control_env, candidate_env)
    comparison = compare_smoke_arms(control_rows, candidate_rows)
    result = {
        "schema": "icdc_topology_causal_smoke_v1",
        "control_sha256": _sha256(Path(args.control)),
        "candidate_sha256": _sha256(Path(args.candidate)),
        "flow_sha256": _sha256(Path(args.flow)),
        "corpus_sha256": _sha256(Path(args.corpus)),
        "labels_sha256": _sha256(Path(args.labels)),
        "environment": environment,
        **comparison,
        "control": control_rows,
        "candidate": candidate_rows,
    }
    _atomic_bytes(Path(args.out), _canonical_bytes(result, newline=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_records <= 0:
        raise ValueError("smoke config")
    result = run(args)
    print(json.dumps({
        key: value
        for key, value in result.items()
        if key not in {"control", "candidate"}
    }, sort_keys=True))
    return 0 if result["causal_smoke_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
