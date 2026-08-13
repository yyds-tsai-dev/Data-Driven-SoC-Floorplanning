from pathlib import Path
from icdc.g1_evidence import G1CaseRow,G1PortfolioReceipt,seal_g1_arm,compare_g1_arms

def _rows(offset=0): return tuple(G1CaseRow(str(i),100+i%3,.1+i/1000,1+offset) for i in range(100))
def _receipts(): return tuple(G1PortfolioReceipt(str(i),3,3,"dpmpp",2,"euler",8,True) for i in range(100))
def test_g1_arm_is_canonical_binds_alpha_and_uses_nearest_rank_p90():
    arm=seal_g1_arm("C0",_rows(),_receipts(),Path.cwd(),feasible_count=100,error_count=0,freeze_sha256="a"*64,calculator_path=Path("partner/icdc/g1_evidence.py"))
    assert arm.runtime_p90==sorted(r.runtime for r in _rows())[89]
    assert arm.alpha_sha256.startswith("804c3432")
    assert compare_g1_arms(arm,arm).binding_pass
