from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from icdc.g1_evidence import (
    ALPHA_RELATIVE_PATH,
    ALPHA_SHA256,
    G1ArmEvidence,
    G1CaseRow,
    G1PortfolioReceipt,
    canonical_g1_bytes,
    compare_g1_arms,
    seal_g1_arm,
)


REPO = Path(__file__).resolve().parents[1]


def _rows(
    *, cost: float = 1.0, runtime: float = 0.25, n_offset: int = 0
) -> tuple[G1CaseRow, ...]:
    return tuple(
        G1CaseRow(str(index), 100 + n_offset + index % 3, runtime + index / 10000.0, cost)
        for index in range(100)
    )


def _receipts() -> tuple[G1PortfolioReceipt, ...]:
    return tuple(
        G1PortfolioReceipt(str(index), 3, 3, "dpmpp", 2, "euler", 8, True)
        for index in range(100)
    )


def _arm(
    arm_id: str,
    *,
    cost: float = 1.0,
    runtime: float = 0.25,
    n_offset: int = 0,
    freeze: str = "a" * 64,
    feasible_count: int = 100,
    error_count: int = 0,
    causal_smoke_ok: bool = True,
) -> G1ArmEvidence:
    return seal_g1_arm(
        arm_id,
        _rows(cost=cost, runtime=runtime, n_offset=n_offset),
        _receipts(),
        REPO,
        feasible_count=feasible_count,
        error_count=error_count,
        freeze_sha256=freeze,
        calculator_path=Path("partner/icdc/g1_evidence.py"),
        causal_smoke_ok=causal_smoke_ok,
    )


def test_g1_arm_is_canonical_binds_alpha_and_uses_nearest_rank_p90():
    rows = _rows()
    arm = _arm("C0")

    assert arm.runtime_sum == pytest.approx(sum(row.runtime for row in rows))
    assert arm.runtime_mean == pytest.approx(sum(row.runtime for row in rows) / 100)
    assert arm.runtime_p90 == pytest.approx(sorted(row.runtime for row in rows)[89])
    assert arm.runtime_max == pytest.approx(max(row.runtime for row in rows))
    assert arm.alpha_relative_path == ALPHA_RELATIVE_PATH
    assert arm.alpha_sha256 == ALPHA_SHA256
    record = arm.to_record()
    without_digest = {key: value for key, value in record.items() if key != "evidence_sha256"}
    assert arm.evidence_sha256 == hashlib.sha256(canonical_g1_bytes(without_digest)).hexdigest()
    assert G1ArmEvidence.from_record(json.loads(canonical_g1_bytes(record))) == arm


@pytest.mark.parametrize(
    "rows,receipts,error",
    (
        (lambda: tuple(reversed(_rows())), _receipts, "ordered rows"),
        (lambda: _rows()[:-1], _receipts, "ordered rows"),
        (_rows, lambda: _receipts()[:-1], "ordered rows"),
        (
            _rows,
            lambda: (dataclasses.replace(_receipts()[0], direct_count=2), *_receipts()[1:]),
            "portfolio",
        ),
    ),
)
def test_g1_arm_rejects_incomplete_or_non_3d3f_evidence(rows, receipts, error):
    with pytest.raises(ValueError, match=error):
        seal_g1_arm(
            "C0",
            rows(),
            receipts(),
            REPO,
            feasible_count=100,
            error_count=0,
            freeze_sha256="a" * 64,
            calculator_path=Path("partner/icdc/g1_evidence.py"),
        )


def test_g1_comparison_uses_matched_combined_score_and_point300_is_diagnostic():
    control = _arm("C0", cost=1.0, runtime=0.25)
    candidate = _arm("C1", cost=0.5, runtime=0.31)
    result = compare_g1_arms(control, candidate)

    assert result.binding_pass
    assert not result.internal_runtime_target_met
    assert result.state == "HIGH_TAIL_CAUSAL_PROOF"
    assert result.candidate_combined <= result.control_combined + 1e-12

    too_expensive = _arm("C1", cost=2.0, runtime=0.01)
    assert compare_g1_arms(control, too_expensive).state == "KILLED_G1_COMBINED"
    assert not compare_g1_arms(control, too_expensive).binding_pass

    unmatched = _arm("C1", cost=0.5, runtime=0.31, n_offset=1)
    with pytest.raises(ValueError, match="matched IDs"):
        compare_g1_arms(control, unmatched)
    assert compare_g1_arms(control, _arm("C1", freeze="b" * 64)).state == "KILLED_G1_COMBINED"
    assert compare_g1_arms(
        control, _arm("C1", feasible_count=99)
    ).state == "KILLED_G1_COMBINED"
    assert compare_g1_arms(
        control, _arm("C1", causal_smoke_ok=False)
    ).state == "KILLED_G1_COMBINED"


def test_g1_arm_loader_rejects_tampered_runtime_and_digest():
    record = _arm("C0").to_record()
    record["runtime_p90"] += 0.1
    with pytest.raises(ValueError, match="runtime summary"):
        G1ArmEvidence.from_record(record)

    record = _arm("C0").to_record()
    record["evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence SHA256"):
        G1ArmEvidence.from_record(record)
