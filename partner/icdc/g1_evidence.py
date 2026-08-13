"""Sealed, matched 100-case G1 evidence and adjudication."""
from __future__ import annotations
import csv, hashlib, json, math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

ALPHA_RELATIVE_PATH = "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"
ALPHA_SHA256 = "804c3432febb88a8f8ee0a8c0ede4b4598a5cf3c109d68de6a6ca86f05d211bd"

def canonical_g1_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()

@dataclass(frozen=True)
class G1CaseRow:
    case_id: str; n: int; runtime: float; cost_no_runtime: float
    alpha_median: float = 0.0
    def to_record(self): return {"case_id": self.case_id, "n": self.n, "runtime": self.runtime, "cost_no_runtime": self.cost_no_runtime, "alpha_median": self.alpha_median}
    @classmethod
    def from_record(cls, value): return cls(str(value["case_id"]), int(value["n"]), float(value["runtime"]), float(value["cost_no_runtime"]), float(value["alpha_median"]))

@dataclass(frozen=True)
class G1PortfolioReceipt:
    case_id: str; direct_count: int; flow_count: int; direct_sampler: str; direct_steps: int; flow_sampler: str; flow_steps: int; normal_pool: bool
    def to_record(self): return self.__dict__

@dataclass(frozen=True)
class G1ArmEvidence:
    arm_id: str; rows: tuple[G1CaseRow, ...]; receipts: tuple[G1PortfolioReceipt, ...]; weighted_combined: float; runtime_sum: float; runtime_mean: float; runtime_p90: float; runtime_max: float; alpha_relative_path: str; alpha_sha256: str; feasible_count: int; error_count: int; freeze_sha256: str; portfolio_ok: bool; feasibility_ok: bool; freeze_ok: bool; causal_smoke_ok: bool; evidence_sha256: str
    def to_record(self):
        d = {"arm_id":self.arm_id,"rows":[r.to_record() for r in self.rows],"receipts":[r.to_record() for r in self.receipts],"weighted_combined":self.weighted_combined,"runtime_sum":self.runtime_sum,"runtime_mean":self.runtime_mean,"runtime_p90":self.runtime_p90,"runtime_max":self.runtime_max,"alpha_relative_path":self.alpha_relative_path,"alpha_sha256":self.alpha_sha256,"feasible_count":self.feasible_count,"error_count":self.error_count,"freeze_sha256":self.freeze_sha256,"portfolio_ok":self.portfolio_ok,"feasibility_ok":self.feasibility_ok,"freeze_ok":self.freeze_ok,"causal_smoke_ok":self.causal_smoke_ok,"evidence_sha256":self.evidence_sha256}
        return d
    @classmethod
    def from_record(cls,d):
        return cls(d["arm_id"],tuple(G1CaseRow.from_record(x) for x in d["rows"]),tuple(G1PortfolioReceipt(**x) for x in d["receipts"]),*(d[k] for k in ("weighted_combined","runtime_sum","runtime_mean","runtime_p90","runtime_max","alpha_relative_path","alpha_sha256","feasible_count","error_count","freeze_sha256","portfolio_ok","feasibility_ok","freeze_ok","causal_smoke_ok","evidence_sha256")))

def load_alpha_medians(repo_root: Path) -> dict[str,float]:
    path=repo_root/ALPHA_RELATIVE_PATH
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=ALPHA_SHA256: raise ValueError("Alpha SHA256")
    with path.open(newline="", encoding="utf-8") as f: rows=list(csv.DictReader(f))
    out={r["test_id"]:float(r["median_runtime_s"]) for r in rows}
    if set(out)!={str(i) for i in range(100)} or any(not math.isfinite(v) or v<=0 for v in out.values()): raise ValueError("Alpha IDs")
    return out

def seal_g1_arm(arm_id, rows, receipts, repo_root, *, feasible_count, error_count, freeze_sha256, calculator_path, causal_smoke_ok=True):
    rows=tuple(rows); receipts=tuple(receipts)
    if len(rows)!=100 or tuple(r.case_id for r in rows)!=tuple(str(i) for i in range(100)): raise ValueError("ordered rows")
    alpha=load_alpha_medians(repo_root); rows=tuple(replace(r,alpha_median=alpha[r.case_id]) for r in rows)
    if len(receipts)!=100 or any((r.case_id!=str(i) or r.direct_count!=3 or r.flow_count!=3 or r.direct_sampler!="dpmpp" or r.direct_steps!=2 or r.flow_sampler!="euler" or r.flow_steps!=8 or not r.normal_pool) for i,r in enumerate(receipts)): raise ValueError("portfolio")
    runtimes=[r.runtime for r in rows]; weights=[math.exp(r.n/12) for r in rows]
    q=[max(0.7,max(0.01,r.runtime/max(r.alpha_median,0.01))**0.3) for r in rows]
    combined=sum(w*r.cost_no_runtime*qq for w,r,qq in zip(weights,rows,q))/sum(weights)
    base=G1ArmEvidence(arm_id,rows,receipts,combined,sum(runtimes),sum(runtimes)/100,sorted(runtimes)[89],max(runtimes),ALPHA_RELATIVE_PATH,ALPHA_SHA256,feasible_count,error_count,freeze_sha256,True,feasible_count==100 and error_count==0, bool(freeze_sha256),causal_smoke_ok,"")
    digest=hashlib.sha256(canonical_g1_bytes({k:v for k,v in base.to_record().items() if k!="evidence_sha256"})).hexdigest()
    return replace(base,evidence_sha256=digest)

@dataclass(frozen=True)
class G1Comparison:
    control_combined: float; candidate_combined: float; binding_pass: bool; internal_runtime_target_met: bool; matched_ids: bool; matched_alpha_medians: bool; portfolio_ok: bool; feasibility_ok: bool; freeze_ok: bool; causal_smoke_ok: bool; state: str; comparison_sha256: str
    def to_record(self): return self.__dict__

def compare_g1_arms(control, candidate):
    ids=tuple(r.case_id for r in control.rows)==tuple(r.case_id for r in candidate.rows)
    if not ids: raise ValueError("matched IDs")
    alpha=tuple(r.alpha_median for r in control.rows)==tuple(r.alpha_median for r in candidate.rows)
    if not alpha: raise ValueError("matched Alpha medians")
    binding=candidate.weighted_combined<=control.weighted_combined+1e-12
    portfolio=control.portfolio_ok and candidate.portfolio_ok; feas=control.feasibility_ok and candidate.feasibility_ok; freeze=control.freeze_ok and candidate.freeze_ok; causal=control.causal_smoke_ok and candidate.causal_smoke_ok
    state="HIGH_TAIL_CAUSAL_PROOF" if binding and portfolio and feas and freeze and causal else "KILLED_G1_COMBINED"
    result=G1Comparison(control.weighted_combined,candidate.weighted_combined,binding,candidate.runtime_mean<=.300,True,True,portfolio,feas,freeze,causal,state,"")
    return replace(result,comparison_sha256=hashlib.sha256(canonical_g1_bytes({k:v for k,v in result.to_record().items() if k!="comparison_sha256"})).hexdigest())
