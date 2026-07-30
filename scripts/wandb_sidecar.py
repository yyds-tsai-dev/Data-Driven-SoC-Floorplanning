"""Wandb sidecar: tail a train_log.jsonl and mirror every entry to wandb.

Runs as an independent process so the training loop stays untouched; killing
the sidecar never affects training. Replays existing history on start (idempotent
via monotonically increasing step), then follows new lines.

Usage:
    python scripts/wandb_sidecar.py <train_log.jsonl> <project> <run_name>
"""
import json
import sys
import time
from pathlib import Path

import wandb


def main() -> None:
    log_path = Path(sys.argv[1])
    project = sys.argv[2] if len(sys.argv) > 2 else "floorset-flow"
    run_name = sys.argv[3] if len(sys.argv) > 3 else log_path.parent.name
    run = wandb.init(project=project, name=run_name,
                     id=run_name.replace("/", "-"), resume="allow")
    print(f"wandb run: {run.url}", flush=True)
    pos = 0
    last_step = -1
    idle = 0
    while True:
        if log_path.exists():
            with open(log_path) as f:
                f.seek(pos)
                while True:
                    line = f.readline()
                    if not line or not line.endswith("\n"):
                        break
                    pos = f.tell()
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = int(rec.get("step", -1))
                    if step <= last_step:
                        continue
                    last_step = step
                    wandb.log(rec, step=step)
                    idle = 0
        idle += 1
        if idle > 240:   # ~1h with no new lines: note it but keep waiting
            wandb.log({"sidecar_idle_minutes": idle * 15 / 60}, step=max(last_step, 0))
            idle = 0
        time.sleep(15)


if __name__ == "__main__":
    main()
