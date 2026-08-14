"""SHA-bound official QA and checked-in scorer adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import shapely
import torch


QA_RELATIVE_PATH = "docs/official/C_QA_20260804.pdf"
QA_SHA256 = "60286cf3eb05ff41732d83fc681506b001e283141223d69bbbb9c27c9f25c5db"
SCORER_RELATIVE_PATH = "scripts/iccad2026_evaluate.py"
SCORER_SHA256 = "7fa64bbbad201f3f6be2a6e426bc141bff7a5b14522bf309c77e055a09bbc6a1"
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


def _sha256(path: Path, error: str) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise ValueError(error) from exc


def preflight_qa_contract(repo_root: Path) -> QAContractEvidence:
    root = Path(repo_root)
    if _sha256(root / QA_RELATIVE_PATH, "QA PDF SHA256") != QA_SHA256:
        raise ValueError("QA PDF SHA256")
    if _sha256(root / SCORER_RELATIVE_PATH, "scorer source") != SCORER_SHA256:
        raise ValueError("scorer SHA256")
    return QAContractEvidence(
        qa_relative_path=QA_RELATIVE_PATH,
        qa_sha256=QA_SHA256,
        scorer_relative_path=SCORER_RELATIVE_PATH,
        scorer_sha256=SCORER_SHA256,
        scorer_contract=SCORER_CONTRACT,
        shapely_version=shapely.__version__,
    )


def qa_manifest_fields(evidence: QAContractEvidence) -> dict[str, str]:
    if not isinstance(evidence, QAContractEvidence):
        raise ValueError("QA evidence")
    return {
        "qa_relative_path": evidence.qa_relative_path,
        "qa_sha256": evidence.qa_sha256,
        "scorer_relative_path": evidence.scorer_relative_path,
        "scorer_sha256": evidence.scorer_sha256,
        "scorer_contract": evidence.scorer_contract,
        "shapely_version": evidence.shapely_version,
    }


@lru_cache(maxsize=1)
def _checked_scorer(evidence: QAContractEvidence) -> Any:
    expected = QAContractEvidence(
        QA_RELATIVE_PATH,
        QA_SHA256,
        SCORER_RELATIVE_PATH,
        SCORER_SHA256,
        SCORER_CONTRACT,
        shapely.__version__,
    )
    if evidence != expected:
        raise ValueError("QA evidence")
    root = Path(__file__).resolve().parents[2]
    scorer_path = root / SCORER_RELATIVE_PATH
    if _sha256(scorer_path, "scorer source") != SCORER_SHA256:
        raise ValueError("scorer SHA256")
    module_name = "_icdc_qa_checked_scorer"
    spec = importlib.util.spec_from_file_location(module_name, scorer_path)
    if spec is None or spec.loader is None:
        raise ValueError("scorer import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    floorset_path = str(root / "FloorSet")
    inserted = floorset_path not in sys.path
    if inserted:
        sys.path.insert(0, floorset_path)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError("scorer import") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(floorset_path)
            except ValueError:
                pass
    return module


def _relation(value: object, width: int, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(name) from exc
    if tensor.numel() == 0:
        return torch.empty((0, width), dtype=torch.float64)
    if tensor.ndim != 2 or tensor.shape[1] != width or not bool(torch.isfinite(tensor).all()):
        raise ValueError(name)
    return tensor


def score_provided_local_no_runtime(
    case: Mapping[str, object],
    rects_xywh: torch.Tensor,
    evidence: QAContractEvidence,
) -> LocalScoreAudit:
    if (
        not isinstance(case, Mapping)
        or not isinstance(rects_xywh, torch.Tensor)
        or rects_xywh.device.type != "cpu"
        or rects_xywh.dtype is not torch.float64
        or rects_xywh.ndim != 2
        or rects_xywh.shape[1] != 4
        or not bool(torch.isfinite(rects_xywh).all())
        or not bool((rects_xywh[:, 2:] > 0).all())
    ):
        raise ValueError("rectangles must be CPU float64 [N,4]")
    n = rects_xywh.shape[0]
    try:
        area = torch.as_tensor(case["area"], dtype=torch.float64, device="cpu")
        cons = torch.as_tensor(case["cons"], dtype=torch.float64, device="cpu")
        tp = torch.as_tensor(case["tp"], dtype=torch.float64, device="cpu")
        hpwl_ref = float(case["hpwl_ref"])
        area_ref = float(case["area_ref"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("case") from exc
    if (
        tuple(area.shape) != (n,)
        or tuple(cons.shape) not in {(n, 2), (n, 5)}
        or tuple(tp.shape) != (n, 4)
        or not bool(torch.isfinite(area).all())
        or not bool(torch.isfinite(cons).all())
        or not bool(torch.isfinite(tp).all())
        or not math.isfinite(hpwl_ref)
        or hpwl_ref < 0
        or not math.isfinite(area_ref)
        or area_ref <= 0
    ):
        raise ValueError("case")
    scorer = _checked_scorer(evidence)
    positions = [tuple(float(item) for item in row) for row in rects_xywh.tolist()]
    result = scorer.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        {"hpwl_baseline": hpwl_ref, "area_baseline": area_ref},
        cons,
        _relation(case.get("b2b", ()), 3, "b2b"),
        _relation(case.get("p2b", ()), 3, "p2b"),
        _relation(case.get("pins", ()), 2, "pins"),
        area,
        tp.tolist(),
        median_runtime=1.0,
    )
    cost = float(result.cost_no_runtime)
    if not math.isfinite(cost):
        raise ValueError("non-finite score")
    return LocalScoreAudit(
        feasible=bool(result.is_feasible),
        cost_no_runtime=cost,
        boundary_violations=int(result.boundary_violations),
        grouping_violations=int(result.grouping_violations),
        mib_violations=int(result.mib_violations),
        area_violations=int(result.area_violations),
        dimension_violations=int(result.dimension_violations),
    )
