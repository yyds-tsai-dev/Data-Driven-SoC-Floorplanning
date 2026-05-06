from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="src/architecture_v1_optimizer.py")
    parser.add_argument("--test-id", type=int)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    contest_dir = root / "FloorSet" / "iccad2026contest"
    optimizer = (root / args.optimizer).resolve()
    cmd = [sys.executable, "iccad2026_evaluate.py"]
    if args.validate:
        cmd.extend(["--validate", str(optimizer)])
    else:
        cmd.extend(["--evaluate", str(optimizer)])
    if args.test_id is not None:
        cmd.extend(["--test-id", str(args.test_id)])
    if args.verbose:
        cmd.append("--verbose")
    return subprocess.call(cmd, cwd=contest_dir)


if __name__ == "__main__":
    raise SystemExit(main())

