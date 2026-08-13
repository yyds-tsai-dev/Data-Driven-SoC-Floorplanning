"""SHA-bound QA and checked-in scorer adapter."""
from __future__ import annotations
import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import torch
import shapely

QA_RELATIVE_PATH = "docs/official/C_QA_20260804.pdf"
QA_SHA256 = "60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db"
SCORER_RELATIVE_PATH = "scripts/iccad2026_evaluate.py"
SCORER_CONTRACT = "iccad2026_evaluate_cost_no_runtime_v1"

@dataclass(frozen=True)
class QAContractEvidence:
    qa_relative_path: str
    qa_sha256: str
    scorer_relative_path: str
    scorer_sha256: str
    scorer_contract: str
    shapely_version: str

@dataclass(frozen=True)
class LocalScoreAudit:
    feasible: bool
    cost_no_runtime: float
    boundary_violations: int
    grouping_violations: int
    mib_violations: int
    area_violations: int
    dimension_violations: int

def preflight_qa_contract(repo_root: Path) -> QAContractEvidence:
    qa = repo_root / QA_RELATIVE_PATH
    scorer = repo_root / SCORER_RELATIVE_PATH
    if not qa.is_file() or hashlib.sha256(qa.read_bytes()).hexdigest() != QA_SHA256:
        raise ValueError("QA PDF SHA256")
    if not scorer.is_file():
        raise ValueError("scorer source")
    return QAContractEvidence(QA_RELATIVE_PATH, QA_SHA256, SCORER_RELATIVE_PATH,
                              hashlib.sha256(scorer.read_bytes()).hexdigest(),
                              SCORER_CONTRACT, shapely.__version__)

def qa_manifest_fields(evidence: QAContractEvidence) -> dict[str, str]:
    return {"qa_relative_path": evidence.qa_relative_path, "qa_sha256": evidence.qa_sha256,
            "scorer_relative_path": evidence.scorer_relative_path, "scorer_sha256": evidence.scorer_sha256,
            "scorer_contract": evidence.scorer_contract, "shapely_version": evidence.shapely_version}

def score_provided_local_no_runtime(case: Mapping[str, object], rects_xywh: torch.Tensor,
                                    evidence: QAContractEvidence) -> LocalScoreAudit:
    if rects_xywh.device.type != "cpu" or rects_xywh.dtype != torch.float64 or rects_xywh.ndim != 2 or rects_xywh.shape[1] != 4:
        raise ValueError("rectangles must be CPU float64 [N,4]")
    root = Path(__file__).resolve().parents[2]
    scorer = root / evidence.scorer_relative_path
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != evidence.scorer_sha256:
        raise ValueError("scorer SHA256")
    spec = importlib.util.spec_from_file_location("icdc_checked_scorer", scorer)
    if spec is None or spec.loader is None:
        raise ValueError("scorer import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def get(*names, default=None):
        for name in names:
            if name in case:
                return case[name]
        return default
    positions = [tuple(float(v) for v in row) for row in rects_xywh.tolist()]
    result = module.evaluate_solution({"positions": positions, "runtime": 1.0},
        get("baseline_metrics", default={}), torch.as_tensor(get("cons", "target_constraints")),
        torch.as_tensor(get("b2b", "b2b_connectivity", default=[])),
        torch.as_tensor(get("p2b", "p2b_connectivity", default=[])),
        torch.as_tensor(get("pins", "pins_pos", default=[])),
        torch.as_tensor(get("area", "target_areas")), get("tp", "target_positions", default=None), median_runtime=1.0)
    cost = float(result.cost_no_runtime)
    if not torch.isfinite(torch.tensor(cost)):
        raise ValueError("non-finite score")
    return LocalScoreAudit(bool(result.is_feasible), cost, int(result.boundary_violations),
        int(result.grouping_violations), int(result.mib_violations), int(result.area_violations), int(result.dimension_violations))
