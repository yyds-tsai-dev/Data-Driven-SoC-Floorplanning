from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointMetricRecord:
    checkpoint: str
    epoch: int
    metric_source: str
    feasible: int
    val_loss: float | None = None
    total_score_no_runtime: float | None = None
    tail_weighted_no_runtime: float | None = None
    soft_violations: int | None = None
    avg_runtime: float | None = None


def _score_value(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def _violation_value(value: int | None) -> int:
    return 10**9 if value is None else int(value)


def metric_sort_key(record: CheckpointMetricRecord) -> tuple:
    return (
        -int(record.feasible),
        _score_value(record.total_score_no_runtime),
        _score_value(record.tail_weighted_no_runtime),
        _violation_value(record.soft_violations),
        _score_value(record.avg_runtime),
        _score_value(record.val_loss),
        int(record.epoch),
    )


def better_checkpoint_metric(
    candidate: CheckpointMetricRecord, current: CheckpointMetricRecord | None
) -> bool:
    if current is None:
        return True
    return metric_sort_key(candidate) < metric_sort_key(current)


def append_metric_record(path: str | Path, record: CheckpointMetricRecord) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_metric_records(path: str | Path) -> list[CheckpointMetricRecord]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(CheckpointMetricRecord(**json.loads(line)))
    return records
