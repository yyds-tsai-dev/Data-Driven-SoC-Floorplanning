#!/usr/bin/env python3
"""Compare two already-sealed, matched topology G1 arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


_REPO = Path(__file__).resolve().parents[2]
_PARTNER = _REPO / "src" / "solver"
if str(_PARTNER) not in sys.path:
    sys.path.insert(0, str(_PARTNER))

from icdc_engine.g1_evidence import G1ArmEvidence, compare_g1_arms  # noqa: E402
from icdc_engine.topology_artifact_guard import write_checked_json  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    control = G1ArmEvidence.from_record(
        json.loads(Path(args.control).read_text(encoding="utf-8"))
    )
    candidate = G1ArmEvidence.from_record(
        json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    )
    result = compare_g1_arms(control, candidate)
    write_checked_json(Path(args.out), result.to_record())
    return 0 if result.state == "HIGH_TAIL_CAUSAL_PROOF" else 1


if __name__ == "__main__":
    raise SystemExit(main())
