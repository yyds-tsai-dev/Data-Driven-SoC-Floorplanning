from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


from icdc.g1_runtime import (
    PortfolioContractError,
    assert_solver_environment,
    build_solver_environment,
    environment_identities,
    validate_matched_environments,
    validate_portfolio_receipt,
)


def test_g1_environment_is_exact_raw_six_candidate_3d3f(tmp_path: Path) -> None:
    direct = tmp_path / "direct.pt"
    flow = tmp_path / "flow.pt"
    direct.write_bytes(b"direct")
    flow.write_bytes(b"flow")
    inherited = {
        "PATH": "/bin",
        "HOME": "/tmp/home",
        "PARTNER_EVIL": "1",
        "PARTNER_OVERSAMPLE": "4",
        "DIRECT_OFF": "1",
        "FLOW_EXTRA": "1",
        "VKILL_FAKE": "1",
        "PYTHONPATH": "/poison",
    }

    env = build_solver_environment(inherited, direct, flow)

    assert env["PARTNER_NREF"] == "6"
    assert env["PARTNER_OVERSAMPLE"] == "1"
    assert env["PARTNER_KS_CAP"] == "6"
    assert env["PARTNER_FLOW_SLOTS"] == "3"
    assert env["PARTNER_DIRECT_SOLVER"] == "dpmpp"
    assert env["PARTNER_DDIM_STEPS"] == "2"
    assert env["PARTNER_FLOW_SOLVER"] == "euler"
    assert env["PARTNER_FLOW_STEPS"] == "8"
    assert env["DIRECT_OFF"] == ""
    assert env["DIRECT_CKPT"] == str(direct.resolve())
    assert env["FLOW_CKPT"] == str(flow.resolve())
    assert not ({"PARTNER_EVIL", "FLOW_EXTRA", "VKILL_FAKE", "PYTHONPATH"} & env.keys())
    assert_solver_environment(env)
    with pytest.raises(PortfolioContractError, match="unexpected solver environment"):
        assert_solver_environment({**env, "PARTNER_EVIL": "1"})


def test_g1_portfolio_receipt_is_exact_and_flow_failure_fails_closed() -> None:
    valid = {
        "case_id": "7",
        "direct_gate_open": True,
        "pool_ready": True,
        "requested_K": 6,
        "oversample": False,
        "direct_count": 3,
        "flow_count": 3,
        "post_candidate_count": 6,
        "direct_sampler": "dpmpp",
        "direct_steps": 2,
        "flow_sampler": "euler",
        "flow_steps": 8,
        "flow_exception": None,
        "fallback": False,
    }
    assert validate_portfolio_receipt(valid) == valid
    with pytest.raises(PortfolioContractError, match="flow_exception"):
        validate_portfolio_receipt({**valid, "flow_exception": "boom"})
    with pytest.raises(PortfolioContractError, match="direct_count"):
        validate_portfolio_receipt({**valid, "direct_count": 4, "flow_count": 2})

    closed = {
        **valid,
        "direct_gate_open": False,
        "pool_ready": True,
        "requested_K": 0,
        "oversample": False,
        "direct_count": 0,
        "flow_count": 0,
        "post_candidate_count": 0,
    }
    assert validate_portfolio_receipt(closed) == closed


def test_g1_matched_environment_masks_only_direct_arm_transport(tmp_path: Path) -> None:
    control_path = tmp_path / "control.pt"
    candidate_path = tmp_path / "candidate.pt"
    flow_path = tmp_path / "flow.pt"
    control_path.write_bytes(b"control")
    candidate_path.write_bytes(b"candidate")
    flow_path.write_bytes(b"flow")
    control = build_solver_environment({}, control_path, flow_path)
    candidate = build_solver_environment({}, candidate_path, flow_path)
    control.update({
        "FLOORSET_TOPOLOGY_RECEIPT": "/tmp/A.json",
        "FLOORSET_OPAQUE_ARM_ID": "A",
    })
    candidate.update({
        "FLOORSET_TOPOLOGY_RECEIPT": "/tmp/B.json",
        "FLOORSET_OPAQUE_ARM_ID": "B",
    })

    c_identity = environment_identities(control)
    x_identity = environment_identities(candidate)
    assert c_identity["full_execution_env_sha256"] != x_identity["full_execution_env_sha256"]
    assert c_identity["matched_solver_env_sha256"] == x_identity["matched_solver_env_sha256"]
    assert validate_matched_environments(control, candidate)["matched_solver_env_sha256"]
    assert c_identity["direct_file_sha256"] == hashlib.sha256(b"control").hexdigest()
    assert x_identity["direct_file_sha256"] == hashlib.sha256(b"candidate").hexdigest()

    candidate["PARTNER_FLOW_STEPS"] = "7"
    with pytest.raises(PortfolioContractError, match="environment delta"):
        validate_matched_environments(control, candidate)
