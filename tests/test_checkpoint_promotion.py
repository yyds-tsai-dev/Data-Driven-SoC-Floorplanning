import json
import os
import subprocess
from pathlib import Path

from floorset_arch.training.selection import CheckpointMetricRecord, append_metric_record


def test_promote_best_checkpoint_uses_evaluator_comparator_and_copies(tmp_path):
    from floorset_arch.training.promote_checkpoint import promote_best_checkpoint

    bad = tmp_path / "bad_val_loss.pt"
    good = tmp_path / "good_eval.pt"
    bad.write_bytes(b"bad")
    good.write_bytes(b"good")
    manifest = tmp_path / "checkpoint_metrics.jsonl"
    destination = tmp_path / "gnn_hgt_best_eval.pt"

    append_metric_record(
        manifest,
        CheckpointMetricRecord(
            checkpoint=str(bad),
            epoch=3,
            metric_source="full_eval",
            feasible=100,
            val_loss=0.001,
            total_score_no_runtime=2.70,
            tail_weighted_no_runtime=2.70,
            soft_violations=10,
            avg_runtime=1.0,
        ),
    )
    append_metric_record(
        manifest,
        CheckpointMetricRecord(
            checkpoint=str(good),
            epoch=2,
            metric_source="full_eval",
            feasible=100,
            val_loss=0.010,
            total_score_no_runtime=2.05,
            tail_weighted_no_runtime=2.05,
            soft_violations=4,
            avg_runtime=1.2,
        ),
    )

    selected = promote_best_checkpoint(manifest, promote_to=destination)

    assert selected.checkpoint == str(good)
    assert destination.read_bytes() == b"good"


def test_checkpoint_metric_from_eval_json_extracts_full_eval_metrics(tmp_path):
    from floorset_arch.training.promote_checkpoint import checkpoint_metric_from_eval_json

    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "total_score": 2.4,
                "total_score_no_runtime": 2.1,
                "summary": {"num_feasible": 99, "avg_runtime": 1.25},
                "test_results": [
                    {
                        "test_id": 95,
                        "block_count": 116,
                        "cost_no_runtime": 2.0,
                        "total_soft_violations": 3,
                    },
                    {
                        "test_id": 99,
                        "block_count": 120,
                        "cost_no_runtime": 3.0,
                        "total_soft_violations": 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    record = checkpoint_metric_from_eval_json(
        checkpoint="checkpoints/candidate.pt",
        eval_json=eval_json,
        epoch=4,
        metric_source="full_eval",
        val_loss=0.123,
    )

    assert record.checkpoint == "checkpoints/candidate.pt"
    assert record.epoch == 4
    assert record.metric_source == "full_eval"
    assert record.feasible == 99
    assert record.total_score_no_runtime == 2.1
    assert record.tail_weighted_no_runtime is None
    assert record.soft_violations == 8
    assert record.avg_runtime == 1.25
    assert record.val_loss == 0.123


def test_checkpoint_metric_from_eval_json_can_compute_tail_weight(tmp_path):
    from floorset_arch.training.promote_checkpoint import checkpoint_metric_from_eval_json

    eval_json = tmp_path / "tail_eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "summary": {"num_feasible": 2, "avg_runtime": 0.5},
                "test_results": [
                    {"test_id": 10, "block_count": 21, "cost_no_runtime": 1.0},
                    {"test_id": 99, "block_count": 120, "cost_no_runtime": 3.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    record = checkpoint_metric_from_eval_json(
        checkpoint="tail.pt",
        eval_json=eval_json,
        epoch=1,
        metric_source="tail_eval",
        tail_ids={99},
    )

    assert record.total_score_no_runtime is None
    assert record.tail_weighted_no_runtime == 3.0


def test_promote_checkpoint_script_wraps_module():
    script = Path("scripts/promote_checkpoint.sh")

    text = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert 'uv run -m floorset_arch.training.promote_checkpoint "$@"' in text
    assert os.access(script, os.X_OK)


def test_promote_checkpoint_cli_appends_metric_and_promotes(tmp_path):
    good = tmp_path / "good.pt"
    good.write_bytes(b"good")
    manifest = tmp_path / "checkpoint_metrics.jsonl"
    destination = tmp_path / "best_eval.pt"
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "total_score_no_runtime": 2.0,
                "summary": {"num_feasible": 100, "avg_runtime": 1.0},
                "test_results": [],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "uv",
            "run",
            "-m",
            "floorset_arch.training.promote_checkpoint",
            "--manifest",
            str(manifest),
            "--append-eval-json",
            str(eval_json),
            "--checkpoint",
            str(good),
            "--epoch",
            "2",
            "--promote-to",
            str(destination),
        ],
        check=True,
    )

    assert destination.read_bytes() == b"good"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["checkpoint"] == str(good)
