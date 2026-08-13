from __future__ import annotations
import argparse, json
from pathlib import Path
from icdc.g1_evidence import G1ArmEvidence, compare_g1_arms
from icdc.topology_artifact_guard import assert_no_dense_fp_artifact, write_checked_json

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--control",required=True); p.add_argument("--candidate",required=True); p.add_argument("--out",required=True); a=p.parse_args(argv)
    control=G1ArmEvidence.from_record(json.loads(Path(a.control).read_text(encoding="utf-8"))); candidate=G1ArmEvidence.from_record(json.loads(Path(a.candidate).read_text(encoding="utf-8")))
    result=compare_g1_arms(control,candidate); record=result.to_record(); assert_no_dense_fp_artifact(record); write_checked_json(Path(a.out),record)
    return 0 if result.state=="HIGH_TAIL_CAUSAL_PROOF" else 1
if __name__ == "__main__": raise SystemExit(main())
