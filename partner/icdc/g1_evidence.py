"""Canonical matched-arm G1 evidence and adjudication."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


ALPHA_RELATIVE_PATH = "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"
ALPHA_SHA256 = "804c3432febb88a8f8ee0a8c0ede4b4598a5cf3c109d68de6a6ca86f05d211bd"
_SHA_RE = re.compile(r"[0-9a-f]{64}")


def canonical_g1_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("canonical G1 JSON") from exc


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(name)
    return result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(name)
    return value


@dataclass(frozen=True)
class G1CaseRow:
    case_id: str
    n: int
    runtime: float
    cost_no_runtime: float
    alpha_median: float | None = None

    def to_record(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "G1CaseRow":
        if not isinstance(value, Mapping) or set(value) != {
            "case_id", "n", "runtime", "cost_no_runtime", "alpha_median"
        }:
            raise ValueError("G1 row")
        alpha = value["alpha_median"]
        return cls(
            case_id=value["case_id"],
            n=value["n"],
            runtime=value["runtime"],
            cost_no_runtime=value["cost_no_runtime"],
            alpha_median=alpha,
        )


@dataclass(frozen=True)
class G1PortfolioReceipt:
    case_id: str
    direct_count: int
    flow_count: int
    direct_sampler: str
    direct_steps: int
    flow_sampler: str
    flow_steps: int
    normal_pool: bool
    direct_gate_open: bool = True

    def to_record(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "G1PortfolioReceipt":
        names = {field for field in cls.__dataclass_fields__}
        if not isinstance(value, Mapping) or set(value) != names:
            raise ValueError("G1 portfolio receipt")
        return cls(**value)


@dataclass(frozen=True)
class G1ArmEvidence:
    schema: str
    arm_id: str
    rows: tuple[G1CaseRow, ...]
    receipts: tuple[G1PortfolioReceipt, ...]
    weighted_combined: float
    runtime_sum: float
    runtime_mean: float
    runtime_p90: float
    runtime_max: float
    alpha_relative_path: str
    alpha_sha256: str
    calculator_sha256: str
    feasible_count: int
    error_count: int
    freeze_sha256: str
    portfolio_ok: bool
    feasibility_ok: bool
    freeze_ok: bool
    causal_smoke_ok: bool
    evidence_sha256: str

    def to_record(self) -> dict[str, object]:
        value = asdict(self)
        value["rows"] = [row.to_record() for row in self.rows]
        value["receipts"] = [receipt.to_record() for receipt in self.receipts]
        return value

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "G1ArmEvidence":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("G1 arm schema")
        parsed = cls(
            **{
                **dict(value),
                "rows": tuple(G1CaseRow.from_record(row) for row in value["rows"]),
                "receipts": tuple(
                    G1PortfolioReceipt.from_record(row) for row in value["receipts"]
                ),
            }
        )
        _validate_arm(parsed, verify_digest=True)
        return parsed


@dataclass(frozen=True)
class G1Comparison:
    schema: str
    control_combined: float
    candidate_combined: float
    binding_pass: bool
    internal_runtime_target_met: bool
    matched_ids: bool
    matched_alpha_medians: bool
    matched_statuses: bool
    portfolio_ok: bool
    feasibility_ok: bool
    freeze_ok: bool
    causal_smoke_ok: bool
    state: str
    comparison_sha256: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def load_alpha_medians(repo_root: Path) -> dict[str, float]:
    path = Path(repo_root) / ALPHA_RELATIVE_PATH
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError("Alpha CSV") from exc
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != ALPHA_SHA256:
        raise ValueError("Alpha SHA256")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ValueError("Alpha CSV") from exc
    if len(source) != 100:
        raise ValueError("Alpha IDs")
    result: dict[str, float] = {}
    for row in source:
        case_id = row.get("test_id")
        if case_id in result:
            raise ValueError("Alpha IDs")
        try:
            median = float(row.get("median_runtime_s", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("Alpha median") from exc
        result[str(case_id)] = _finite(median, "Alpha median", positive=True)
    if tuple(result) != tuple(str(index) for index in range(100)):
        raise ValueError("Alpha IDs")
    return result


def _validate_input_row(row: G1CaseRow, case_id: str) -> None:
    if type(row) is not G1CaseRow or row.case_id != case_id:
        raise ValueError("ordered rows")
    if type(row.n) is not int or row.n <= 0:
        raise ValueError("block count")
    _finite(row.runtime, "runtime", positive=True)
    _finite(row.cost_no_runtime, "cost_no_runtime", positive=True)


def _validate_receipt(receipt: G1PortfolioReceipt, case_id: str) -> None:
    if type(receipt) is not G1PortfolioReceipt or receipt.case_id != case_id:
        raise ValueError("portfolio")
    if type(receipt.direct_gate_open) is not bool:
        raise ValueError("portfolio")
    expected = G1PortfolioReceipt(
        case_id,
        3 if receipt.direct_gate_open else 0,
        3 if receipt.direct_gate_open else 0,
        "dpmpp",
        2,
        "euler",
        8,
        receipt.direct_gate_open,
        receipt.direct_gate_open,
    )
    if receipt != expected:
        raise ValueError("portfolio")


def _runtime_summary(rows: Sequence[G1CaseRow]) -> tuple[float, float, float, float]:
    values = sorted(_finite(row.runtime, "runtime", positive=True) for row in rows)
    total = math.fsum(values)
    return total, total / 100.0, values[89], values[-1]


def _weighted_combined(rows: Sequence[G1CaseRow]) -> float:
    weights: list[float] = []
    terms: list[float] = []
    for row in rows:
        if row.alpha_median is None:
            raise ValueError("Alpha median")
        median = _finite(row.alpha_median, "Alpha median", positive=True)
        weight = math.exp(row.n / 12.0)
        factor = max(0.7, max(0.01, row.runtime / median) ** 0.3)
        term = weight * row.cost_no_runtime * factor
        if not math.isfinite(weight) or not math.isfinite(term):
            raise ValueError("weighted combined")
        weights.append(weight)
        terms.append(term)
    result = math.fsum(terms) / math.fsum(weights)
    if not math.isfinite(result):
        raise ValueError("weighted combined")
    return result


def _arm_digest(arm: G1ArmEvidence) -> str:
    record = arm.to_record()
    record.pop("evidence_sha256")
    return hashlib.sha256(canonical_g1_bytes(record)).hexdigest()


def _validate_arm(arm: G1ArmEvidence, *, verify_digest: bool) -> None:
    if arm.schema != "icdc_topology_g1_arm_v1" or not isinstance(arm.arm_id, str) or not arm.arm_id:
        raise ValueError("G1 arm schema")
    if len(arm.rows) != 100 or len(arm.receipts) != 100:
        raise ValueError("ordered rows")
    for index, (row, receipt) in enumerate(zip(arm.rows, arm.receipts)):
        case_id = str(index)
        _validate_input_row(row, case_id)
        _validate_receipt(receipt, case_id)
        _finite(row.alpha_median, "Alpha median", positive=True)
    expected_runtime = _runtime_summary(arm.rows)
    actual_runtime = (arm.runtime_sum, arm.runtime_mean, arm.runtime_p90, arm.runtime_max)
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15) for a, b in zip(actual_runtime, expected_runtime)):
        raise ValueError("runtime summary")
    expected_combined = _weighted_combined(arm.rows)
    if not math.isclose(arm.weighted_combined, expected_combined, rel_tol=1e-15, abs_tol=1e-15):
        raise ValueError("weighted combined")
    if arm.alpha_relative_path != ALPHA_RELATIVE_PATH or arm.alpha_sha256 != ALPHA_SHA256:
        raise ValueError("Alpha binding")
    _sha(arm.calculator_sha256, "calculator SHA256")
    _sha(arm.freeze_sha256, "freeze SHA256")
    if type(arm.feasible_count) is not int or type(arm.error_count) is not int:
        raise ValueError("feasibility")
    if any(type(value) is not bool for value in (
        arm.portfolio_ok, arm.feasibility_ok, arm.freeze_ok, arm.causal_smoke_ok
    )):
        raise ValueError("G1 gates")
    if not arm.portfolio_ok or arm.feasibility_ok != (arm.feasible_count == 100 and arm.error_count == 0):
        raise ValueError("G1 gates")
    if not arm.freeze_ok:
        raise ValueError("G1 gates")
    if verify_digest and (_sha(arm.evidence_sha256, "evidence SHA256") != _arm_digest(arm)):
        raise ValueError("evidence SHA256")


def seal_g1_arm(
    arm_id: str,
    rows: Sequence[G1CaseRow],
    receipts: Sequence[G1PortfolioReceipt],
    repo_root: Path,
    *,
    feasible_count: int,
    error_count: int,
    freeze_sha256: str,
    calculator_path: Path,
    causal_smoke_ok: bool = True,
) -> G1ArmEvidence:
    rows = tuple(rows)
    receipts = tuple(receipts)
    if len(rows) != 100 or len(receipts) != 100:
        raise ValueError("ordered rows")
    alpha = load_alpha_medians(repo_root)
    sealed_rows: list[G1CaseRow] = []
    for index, row in enumerate(rows):
        case_id = str(index)
        _validate_input_row(row, case_id)
        if row.alpha_median is not None:
            raise ValueError("caller-supplied Alpha median")
        sealed_rows.append(replace(row, alpha_median=alpha[case_id]))
        _validate_receipt(receipts[index], case_id)
    calculator = Path(repo_root) / Path(calculator_path)
    try:
        calculator_sha256 = hashlib.sha256(calculator.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("calculator source") from exc
    runtime_sum, runtime_mean, runtime_p90, runtime_max = _runtime_summary(sealed_rows)
    arm = G1ArmEvidence(
        schema="icdc_topology_g1_arm_v1",
        arm_id=arm_id,
        rows=tuple(sealed_rows),
        receipts=receipts,
        weighted_combined=_weighted_combined(sealed_rows),
        runtime_sum=runtime_sum,
        runtime_mean=runtime_mean,
        runtime_p90=runtime_p90,
        runtime_max=runtime_max,
        alpha_relative_path=ALPHA_RELATIVE_PATH,
        alpha_sha256=ALPHA_SHA256,
        calculator_sha256=calculator_sha256,
        feasible_count=feasible_count,
        error_count=error_count,
        freeze_sha256=_sha(freeze_sha256, "freeze SHA256"),
        portfolio_ok=True,
        feasibility_ok=feasible_count == 100 and error_count == 0,
        freeze_ok=True,
        causal_smoke_ok=causal_smoke_ok,
        evidence_sha256="",
    )
    _validate_arm(arm, verify_digest=False)
    return replace(arm, evidence_sha256=_arm_digest(arm))


def _comparison_digest(comparison: G1Comparison) -> str:
    record = comparison.to_record()
    record.pop("comparison_sha256")
    return hashlib.sha256(canonical_g1_bytes(record)).hexdigest()


def compare_g1_arms(control: G1ArmEvidence, candidate: G1ArmEvidence) -> G1Comparison:
    _validate_arm(control, verify_digest=True)
    _validate_arm(candidate, verify_digest=True)
    control_ids = tuple((row.case_id, row.n) for row in control.rows)
    candidate_ids = tuple((row.case_id, row.n) for row in candidate.rows)
    if control_ids != candidate_ids:
        raise ValueError("matched IDs")
    control_alpha = tuple(row.alpha_median for row in control.rows)
    candidate_alpha = tuple(row.alpha_median for row in candidate.rows)
    if control_alpha != candidate_alpha:
        raise ValueError("matched Alpha medians")
    control_statuses = tuple(
        (receipt.case_id, receipt.direct_gate_open, receipt.normal_pool)
        for receipt in control.receipts
    )
    candidate_statuses = tuple(
        (receipt.case_id, receipt.direct_gate_open, receipt.normal_pool)
        for receipt in candidate.receipts
    )
    if control_statuses != candidate_statuses:
        raise ValueError("status vector")
    freeze_ok = (
        control.freeze_ok
        and candidate.freeze_ok
        and control.freeze_sha256 == candidate.freeze_sha256
        and control.calculator_sha256 == candidate.calculator_sha256
    )
    portfolio_ok = control.portfolio_ok and candidate.portfolio_ok
    feasibility_ok = control.feasibility_ok and candidate.feasibility_ok
    causal_smoke_ok = control.causal_smoke_ok and candidate.causal_smoke_ok
    binding_pass = candidate.weighted_combined <= control.weighted_combined + 1e-12
    mandatory = portfolio_ok and feasibility_ok and freeze_ok and causal_smoke_ok
    state = "HIGH_TAIL_CAUSAL_PROOF" if binding_pass and mandatory else "KILLED_G1_COMBINED"
    comparison = G1Comparison(
        schema="icdc_topology_g1_comparison_v1",
        control_combined=control.weighted_combined,
        candidate_combined=candidate.weighted_combined,
        binding_pass=binding_pass,
        internal_runtime_target_met=candidate.runtime_mean <= 0.300,
        matched_ids=True,
        matched_alpha_medians=True,
        matched_statuses=True,
        portfolio_ok=portfolio_ok,
        feasibility_ok=feasibility_ok,
        freeze_ok=freeze_ok,
        causal_smoke_ok=causal_smoke_ok,
        state=state,
        comparison_sha256="",
    )
    return replace(comparison, comparison_sha256=_comparison_digest(comparison))
