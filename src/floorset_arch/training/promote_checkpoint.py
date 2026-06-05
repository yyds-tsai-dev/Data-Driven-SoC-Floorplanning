from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

from floorset_arch.training.selection import (
    CheckpointMetricRecord,
    append_metric_record,
    better_checkpoint_metric,
    metric_sort_key,
    read_metric_records,
)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _soft_violations(row: dict) -> int:
    if row.get("total_soft_violations") is not None:
        return int(row["total_soft_violations"])
    return (
        int(row.get("boundary_violations", 0))
        + int(row.get("grouping_violations", 0))
        + int(row.get("mib_violations", 0))
    )


def _weighted_no_runtime_score(rows: list[dict]) -> float | None:
    if not rows:
        return None
    block_counts = [int(row.get("block_count", 0)) for row in rows]
    costs = [float(row["cost_no_runtime"]) for row in rows]
    max_n = max(block_counts)
    weights = [math.exp((n - max_n) / 12.0) for n in block_counts]
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(cost * weight for cost, weight in zip(costs, weights)) / total_weight


def _parse_tail_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def checkpoint_metric_from_eval_json(
    checkpoint: str | Path,
    eval_json: str | Path,
    epoch: int,
    metric_source: str = "full_eval",
    val_loss: float | None = None,
    tail_ids: set[int] | None = None,
) -> CheckpointMetricRecord:
    data = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    results = list(data.get("test_results") or data.get("results") or [])
    scored_rows = [
        row
        for row in results
        if row.get("cost_no_runtime") is not None and row.get("block_count") is not None
    ]
    if tail_ids is not None:
        scored_rows = [row for row in scored_rows if int(row.get("test_id", -1)) in tail_ids]

    total_score_no_runtime = _optional_float(data.get("total_score_no_runtime"))
    tail_weighted_no_runtime = (
        _weighted_no_runtime_score(scored_rows) if tail_ids is not None else None
    )
    soft_violations = sum(_soft_violations(row) for row in scored_rows)

    return CheckpointMetricRecord(
        checkpoint=str(checkpoint),
        epoch=int(epoch),
        metric_source=metric_source,
        feasible=int(summary.get("num_feasible", 0)),
        val_loss=val_loss,
        total_score_no_runtime=total_score_no_runtime,
        tail_weighted_no_runtime=tail_weighted_no_runtime,
        soft_violations=soft_violations,
        avg_runtime=_optional_float(summary.get("avg_runtime")),
    )


def select_best_checkpoint(records: Iterable[CheckpointMetricRecord]) -> CheckpointMetricRecord:
    best: CheckpointMetricRecord | None = None
    for record in records:
        if better_checkpoint_metric(record, best):
            best = record
    if best is None:
        raise ValueError("No checkpoint metric records found")
    return best


def promote_best_checkpoint(
    manifest: str | Path,
    promote_to: str | Path | None = None,
) -> CheckpointMetricRecord:
    records = read_metric_records(manifest)
    best = select_best_checkpoint(records)
    if promote_to is not None:
        source = Path(best.checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"Selected checkpoint does not exist: {source}")
        destination = Path(promote_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return best


def _print_ranking(records: list[CheckpointMetricRecord]) -> None:
    ranked = sorted(records, key=metric_sort_key)
    print("| Rank | Checkpoint | Source | Feasible | No-runtime | Tail no-runtime | Soft | Runtime | Val loss |")
    print("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, record in enumerate(ranked, 1):
        print(
            f"| {rank} | `{record.checkpoint}` | {record.metric_source} | "
            f"{record.feasible} | {_fmt(record.total_score_no_runtime)} | "
            f"{_fmt(record.tail_weighted_no_runtime)} | "
            f"{'' if record.soft_violations is None else record.soft_violations} | "
            f"{_fmt(record.avg_runtime)} | {_fmt(record.val_loss)} |"
        )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{float(value):.4f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the best checkpoint using evaluator metric records."
    )
    parser.add_argument(
        "--manifest",
        default="checkpoints/checkpoint_metrics.jsonl",
        help="JSONL metric manifest to read/write.",
    )
    parser.add_argument(
        "--append-eval-json",
        help="Append one CheckpointMetricRecord from an evaluator JSON output.",
    )
    parser.add_argument("--checkpoint", help="Checkpoint path for --append-eval-json.")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch for appended metric.")
    parser.add_argument("--metric-source", default="full_eval")
    parser.add_argument("--val-loss", type=float, default=None)
    parser.add_argument(
        "--tail-ids",
        help="Comma-separated test ids for tail_weighted_no_runtime calculation.",
    )
    parser.add_argument(
        "--promote-to",
        help="Copy selected checkpoint to this stable promoted checkpoint path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = Path(args.manifest)
    if args.append_eval_json:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required with --append-eval-json")
        record = checkpoint_metric_from_eval_json(
            checkpoint=args.checkpoint,
            eval_json=args.append_eval_json,
            epoch=args.epoch,
            metric_source=args.metric_source,
            val_loss=args.val_loss,
            tail_ids=_parse_tail_ids(args.tail_ids),
        )
        append_metric_record(manifest, record)
        print(f"Appended metric record for {record.checkpoint} to {manifest}")

    records = read_metric_records(manifest)
    best = promote_best_checkpoint(manifest, promote_to=args.promote_to)
    _print_ranking(records)
    print()
    print(f"Selected checkpoint: {best.checkpoint}")
    if args.promote_to:
        print(f"Promoted checkpoint: {args.promote_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
