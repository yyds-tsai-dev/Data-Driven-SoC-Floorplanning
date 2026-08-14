"""Fail-closed runtime contracts for the frozen G1 3-Direct/3-Flow pair."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class PortfolioContractError(ValueError):
    """Raised when an arm is not the approved exact 3D/3F portfolio."""


_GENERIC_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES")
_ARM_TRANSPORT_KEYS = (
    "DIRECT_CKPT",
    "FLOORSET_TOPOLOGY_RECEIPT",
    "FLOORSET_OPAQUE_ARM_ID",
)
_SOLVER_ENV = {
    "DIRECT_OFF": "",
    "VKILL_OFF": "1",
    "PARTNER_POOL": "24",
    "PARTNER_NREF": "6",
    "PARTNER_NREF_MIN_N": "95",
    "PARTNER_DIRECT_MIN": "0.3",
    "PARTNER_DIRECT_SEAT_FIX": "0",
    # Normative G0-v2/G1 amendment: the raw pool itself is six candidates.
    "PARTNER_OVERSAMPLE": "1",
    "PARTNER_KS_CAP": "6",
    "PARTNER_DIRECT_SOLVER": "dpmpp",
    "PARTNER_DDIM_STEPS": "2",
    "PARTNER_FLOW_SLOTS": "3",
    "PARTNER_FLOW_SOLVER": "euler",
    "PARTNER_FLOW_STEPS": "8",
    "PARTNER_FLOW_ANTITHETIC": "1",
    "PARTNER_PRESCREEN_V": "1",
    "PARTNER_TAG_ANCHOR_EXTRA": "3",
    "PARTNER_BUDGET_SCALE": "8.498e-5",
    "PARTNER_BUDGET_TAU": "12",
    "PARTNER_BUDGET_MIN": "0.05",
    "PARTNER_BUDGET_MAX": "1.22",
    "PARTNER_POOL_GATE": "0",
    "PARTNER_REFINE_STALL_STOP": "1",
    "PARTNER_SA_KERNEL": "numba",
    "PARTNER_REFINE_KERNEL": "numba",
    "PARTNER_REFINE_FASTBUILD": "1",
    "PARTNER_FAST_SETUP": "1",
    "PARTNER_EDGE_SEAT_V2": "1",
    "PARTNER_FRAME_WPIN": "1",
    "PARTNER_COORD_POLISH": "1",
    "PARTNER_FRAME_SCALE_LADDER": "1",
    "PARTNER_FRAME_SCALE_SET": "1.02",
    "PARTNER_SEAT_FINAL": "1",
    "PARTNER_TAG_COMPRESS": "1",
    "PARTNER_GROUP_BRIDGE": "1",
    "PARTNER_FLOW_ZORDER": "",
    "PARTNER_FLOW_NOPT": "",
    "PARTNER_NOISE_OPT": "",
    "PARTNER_PHYSICS_GUIDE": "",
    "PARTNER_GPU_ARM": "0",
    "PARTNER_RETRIEVAL_INDEX": "",
    "PARTNER_RETRIEVAL_SLOTS": "0",
    "PARTNER_GROUP_DAG_BRIDGE": "",
    "PARTNER_GROUP_DAG_BRIDGE_DEBUG": "",
    "PARTNER_ORACLE_PRED_FILE": "",
    "PARTNER_PSEL_DUMP": "",
    "PARTNER_SEAT_DEBUG": "",
    "PARTNER_DIRECT_WARM": "",
    "PARTNER_TAG_COMPRESS_DEBUG": "",
    "PARTNER_GROUP_BRIDGE_DEBUG": "",
    "PYTHONHASHSEED": "0",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: str, name: str) -> str:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise PortfolioContractError(name) from exc
    if source.is_symlink() or not source.is_file():
        raise PortfolioContractError(name)
    return hashlib.sha256(payload).hexdigest()


def build_solver_environment(
    inherited: Mapping[str, str], direct_ckpt: Path, flow_ckpt: Path
) -> dict[str, str]:
    """Build an allowlisted environment; no ambient solver flag survives."""

    direct = Path(direct_ckpt).resolve(strict=True)
    flow = Path(flow_ckpt).resolve(strict=True)
    result = {
        key: str(inherited[key])
        for key in _GENERIC_ENV_KEYS
        if key in inherited and str(inherited[key])
    }
    result.update(_SOLVER_ENV)
    result["DIRECT_CKPT"] = str(direct)
    result["FLOW_CKPT"] = str(flow)
    return result


def assert_solver_environment(execution_env: Mapping[str, str]) -> None:
    """Reject any solver namespace value outside the frozen G1 mapping."""

    env = {str(key): str(value) for key, value in execution_env.items()}
    for key, expected in _SOLVER_ENV.items():
        if env.get(key) != expected:
            raise PortfolioContractError(f"solver environment: {key}")
    for key in ("DIRECT_CKPT", "FLOW_CKPT"):
        if not env.get(key):
            raise PortfolioContractError(f"solver environment: {key}")
        _file_sha256(env[key], f"solver environment: {key}")
    allowed = set(_SOLVER_ENV) | {
        "DIRECT_CKPT",
        "FLOW_CKPT",
        "FLOORSET_TOPOLOGY_RECEIPT",
        "FLOORSET_OPAQUE_ARM_ID",
        "FLOORSET_TOPOLOGY_EXPECTED_CASES",
    }
    prefixes = ("PARTNER_", "DIRECT_", "FLOW_", "VKILL")
    unexpected = sorted(
        key for key in env if key.startswith(prefixes) and key not in allowed
    )
    if unexpected:
        raise PortfolioContractError(
            "unexpected solver environment: " + ",".join(unexpected)
        )


def _normalized_environment(execution_env: Mapping[str, str]) -> dict[str, str]:
    result = {str(key): str(value) for key, value in execution_env.items()}
    result["DIRECT_CKPT"] = "<OPAQUE_DIRECT_ARM>"
    result.pop("FLOORSET_TOPOLOGY_RECEIPT", None)
    result.pop("FLOORSET_OPAQUE_ARM_ID", None)
    return result


def environment_identities(execution_env: Mapping[str, str]) -> dict[str, str]:
    env = {str(key): str(value) for key, value in execution_env.items()}
    if "DIRECT_CKPT" not in env or "FLOW_CKPT" not in env:
        raise PortfolioContractError("checkpoint environment")
    return {
        "full_execution_env_sha256": hashlib.sha256(_canonical_bytes(env)).hexdigest(),
        "matched_solver_env_sha256": hashlib.sha256(
            _canonical_bytes(_normalized_environment(env))
        ).hexdigest(),
        "direct_file_sha256": _file_sha256(env["DIRECT_CKPT"], "Direct checkpoint"),
        "flow_file_sha256": _file_sha256(env["FLOW_CKPT"], "Flow checkpoint"),
    }


def validate_matched_environments(
    control: Mapping[str, str], candidate: Mapping[str, str]
) -> dict[str, str]:
    control_normalized = _normalized_environment(control)
    candidate_normalized = _normalized_environment(candidate)
    if control_normalized != candidate_normalized:
        raise PortfolioContractError("environment delta")
    control_identity = environment_identities(control)
    candidate_identity = environment_identities(candidate)
    if control_identity["flow_file_sha256"] != candidate_identity["flow_file_sha256"]:
        raise PortfolioContractError("environment delta")
    return {
        "matched_solver_env_sha256": control_identity["matched_solver_env_sha256"],
        "control_direct_file_sha256": control_identity["direct_file_sha256"],
        "candidate_direct_file_sha256": candidate_identity["direct_file_sha256"],
        "flow_file_sha256": control_identity["flow_file_sha256"],
    }


_PORTFOLIO_FIELDS = {
    "case_id",
    "direct_gate_open",
    "pool_ready",
    "requested_K",
    "oversample",
    "direct_count",
    "flow_count",
    "post_candidate_count",
    "direct_sampler",
    "direct_steps",
    "flow_sampler",
    "flow_steps",
    "flow_exception",
    "fallback",
}


def validate_portfolio_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(receipt, Mapping) or set(receipt) != _PORTFOLIO_FIELDS:
        raise PortfolioContractError("receipt schema")
    value = dict(receipt)
    if not isinstance(value["case_id"], str) or not value["case_id"]:
        raise PortfolioContractError("case_id")
    if type(value["direct_gate_open"]) is not bool:
        raise PortfolioContractError("direct_gate_open")
    if type(value["pool_ready"]) is not bool or type(value["oversample"]) is not bool:
        raise PortfolioContractError("pool_ready")
    if type(value["fallback"]) is not bool or value["fallback"]:
        raise PortfolioContractError("fallback")
    if value["flow_exception"] is not None:
        raise PortfolioContractError("flow_exception")
    if (
        value["direct_sampler"],
        value["direct_steps"],
        value["flow_sampler"],
        value["flow_steps"],
    ) != ("dpmpp", 2, "euler", 8):
        raise PortfolioContractError("sampler contract")
    counts = (
        value["requested_K"],
        value["direct_count"],
        value["flow_count"],
        value["post_candidate_count"],
    )
    if any(type(item) is not int or item < 0 for item in counts):
        raise PortfolioContractError("candidate counts")
    if value["direct_gate_open"]:
        if not value["pool_ready"]:
            raise PortfolioContractError("pool_ready")
        if value["requested_K"] != 6 or value["oversample"]:
            raise PortfolioContractError("requested_K")
        if value["direct_count"] != 3:
            raise PortfolioContractError("direct_count")
        if value["flow_count"] != 3:
            raise PortfolioContractError("flow_count")
        if value["post_candidate_count"] != 6:
            raise PortfolioContractError("post_candidate_count")
    elif counts != (0, 0, 0, 0):
        raise PortfolioContractError("direct gate closed")
    return value


def seal_evaluator_arm(
    arm_id: str,
    evaluator: Mapping[str, Any],
    wrapper_receipt: Mapping[str, Any],
    repo_root: Path,
    *,
    freeze_sha256: str,
    causal_smoke_ok: bool,
):
    """Join the two independently written arm artifacts and seal G1 evidence."""

    from .g1_evidence import (
        G1CaseRow,
        G1PortfolioReceipt,
        seal_g1_arm,
    )

    if not isinstance(arm_id, str) or not arm_id:
        raise PortfolioContractError("arm id")
    required_receipt = {
        "schema", "arm_id", "complete", "expected_cases", "environment",
        "cases", "invalid_events",
    }
    if (
        not isinstance(wrapper_receipt, Mapping)
        or set(wrapper_receipt) != required_receipt
        or wrapper_receipt["schema"] != "icdc_topology_3d3f_arm_receipt_v1"
        or wrapper_receipt["arm_id"] != arm_id
        or wrapper_receipt["complete"] is not True
        or wrapper_receipt["expected_cases"] != 100
        or wrapper_receipt["invalid_events"] != []
        or not isinstance(wrapper_receipt["environment"], Mapping)
    ):
        raise PortfolioContractError("arm receipt")
    cases = wrapper_receipt["cases"]
    results = evaluator.get("test_results") if isinstance(evaluator, Mapping) else None
    if not isinstance(cases, list) or not isinstance(results, list):
        raise PortfolioContractError("100 ordered cases")
    if len(cases) != 100 or len(results) != 100:
        raise PortfolioContractError("100 ordered cases")

    rows = []
    receipts = []
    feasible_count = 0
    error_count = 0
    for index, (case, result) in enumerate(zip(cases, results)):
        if not isinstance(case, Mapping) or not isinstance(result, Mapping):
            raise PortfolioContractError("ordered cases")
        if case.get("ordinal") != index or result.get("test_id") != index:
            raise PortfolioContractError("ordered cases")
        n = result.get("block_count")
        if type(n) is not int or n <= 0 or case.get("block_count") != n:
            raise PortfolioContractError("block count")
        portfolio = validate_portfolio_receipt(case.get("portfolio"))
        gate = portfolio["direct_gate_open"]
        expected_status = "normal_pool_valid" if gate else "direct_gate_closed"
        if case.get("status") != expected_status:
            raise PortfolioContractError("gate status")
        runtime = result.get("runtime_seconds")
        cost = result.get("cost_no_runtime")
        if (
            isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not math.isfinite(float(runtime))
            or float(runtime) <= 0
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) <= 0
        ):
            raise PortfolioContractError("evaluator row")
        error = result.get("error")
        feasible = result.get("is_feasible")
        if error is not None and not isinstance(error, str):
            raise PortfolioContractError("evaluator error")
        if type(feasible) is not bool:
            raise PortfolioContractError("evaluator feasibility")
        error_count += int(error is not None)
        feasible_count += int(feasible)
        rows.append(G1CaseRow(str(index), n, float(runtime), float(cost)))
        receipts.append(G1PortfolioReceipt(
            str(index),
            int(portfolio["direct_count"]),
            int(portfolio["flow_count"]),
            str(portfolio["direct_sampler"]),
            int(portfolio["direct_steps"]),
            str(portfolio["flow_sampler"]),
            int(portfolio["flow_steps"]),
            bool(gate),
            direct_gate_open=bool(gate),
        ))
    return seal_g1_arm(
        arm_id,
        rows,
        receipts,
        Path(repo_root),
        feasible_count=feasible_count,
        error_count=error_count,
        freeze_sha256=freeze_sha256,
        calculator_path=Path("partner/icdc/g1_evidence.py"),
        causal_smoke_ok=causal_smoke_ok,
    )
